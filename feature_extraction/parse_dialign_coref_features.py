"""
Parse dialign and coreference solver outputs and add expected feature columns.

For each segment (row) in the corpus CSV the script:
  1. Finds the matching dialign output files and parses them to extract:
       - other_repetition_phrases: phrases in the target segment repeated from the other speaker
       - other_speaker_self_rep: # patterns the other speaker repeats from themselves (from self-rep lexicon TSV)
       - other_speaker_other_rep: # patterns the other speaker repeats from the target speaker
  2. Finds the matching coref output (.jsonlines) and parses it to extract:
       - coref_related_phrases: phrases in the target segment that belong to a multi-mention coref cluster
       - coref_global_pos: number of coref cluster mentions in the target segment
       - coref_local_pos: same as above

Dialign file naming convention
  NOXI: noxi_{session_id}_{global_row_idx}_tsv-{suffix}
  CABB-S: cabbs_{conv_id}_{ID}_tsv-{suffix}

Coref file naming convention
  NOXI: noxi_{session_id}_{global_row_idx}.jsonlines
  CABB-S: cabbs_{conv_id}_{ID}.jsonlines
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

# corpus-specific
N_PRIOR  = 4
N_SUBSEQ = 3

EMPTY_TOKENS = {"", "0", "#", "(#)", "nan", "n/a"}

OUTPUT_COLS = [
    "other_repetition_phrases",
    "other_speaker_self_rep",
    "other_speaker_other_rep",
    "coref_related_phrases",
    "coref_global_pos",
    "coref_local_pos",
]

_CONSTRAINED_RE = re.compile(r"\[([^\]]+)\]")
_TURN_LINE_RE = re.compile(r"^\S+\|\d+\|")

def is_empty(text) -> bool:
    if pd.isna(text):
        return True
    return str(text).strip().lower() in EMPTY_TOKENS

def session_id_from_noxi(session_name: str) -> str:
    m = re.search(r"pair(\d+)", str(session_name))
    return m.group(1) if m else str(session_name)

def safe_conv_id(conv_id: str) -> str:
    return str(conv_id).replace(".", "_")

def _build_stem_index(df: pd.DataFrame, corpus: str) -> pd.Series:
    """
    Return a Series (same index as df) with the dialign/coref filename for each row.
    """
    if corpus == "noxi":
        sorted_df = df.sort_values(["session", "onset_msec"])
        global_idx = pd.Series(range(len(sorted_df)), index=sorted_df.index)
        sid_map = df["session"].map(session_id_from_noxi)
        stems = "noxi_" + sid_map + "_" + global_idx.astype(str)
    else:
        stems = "cabbs_" + df["conv_id"].map(safe_conv_id) + "_" + df["ID"].astype(str)
    return stems

def _get_local_pos(df: pd.DataFrame, corpus: str) -> pd.Series:
    session_col = "conv_id" if corpus == "cabbs" else "session"
    df = df.copy()
    df["_sort_key"] = df["onset_msec"]
    df = df.sort_values([session_col, "_sort_key"])
    local_pos = df.groupby(session_col).cumcount()
    local_pos.name = "_local_pos"
    return local_pos

def _target_dialogue_pos(session_series: pd.DataFrame, local_pos: int) -> int:
    prior_start = max(0, local_pos - N_PRIOR)
    prior_rows = session_series.iloc[prior_start:local_pos]
    n_non_empty = prior_rows["speech_original"].apply(lambda x: not is_empty(x)).sum()
    return int(n_non_empty)

def _find_dialign_file(dialign_dir: Path, stem: str, suffix: str) -> Optional[Path]:
    candidates = [
        dialign_dir / f"{stem}_tsv-{suffix}",
        dialign_dir / f"{stem}.tsv-{suffix}",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _parse_constrained_patterns(dialogue_path: Path, target_pos: int) -> List[str]:
    try:
        lines = [
            l.rstrip()
            for l in dialogue_path.read_text(encoding="utf-8").splitlines()
            if _TURN_LINE_RE.match(l.strip())
        ]
    except (OSError, UnicodeDecodeError):
        return []

    if target_pos >= len(lines):
        return []

    target_line = lines[target_pos]
    return _CONSTRAINED_RE.findall(target_line)

def _count_self_rep_patterns(lexicon_path: Path) -> int:
    """Count patterns in a self-rep lexicon TSV (other speaker's self-repetitions)."""
    try:
        df = pd.read_csv(lexicon_path, sep="\t")
        return max(0, len(df))
    except (OSError, pd.errors.EmptyDataError):
        return 0

def _count_other_rep_by_speaker(lexicon_path: Path, first_speaker: str) -> int:
    """
    Count patterns in the inter-speaker lexicon - the other speaker repeats the current speaker.
    """
    try:
        df = pd.read_csv(lexicon_path, sep="\t")
        if df.empty or "First Speaker" not in df.columns:
            return 0
        return int((df["First Speaker"] == first_speaker).sum())
    except (OSError, pd.errors.EmptyDataError):
        return 0

def _parse_dialign_features(dialign_dir: Path, stem: str, target_pos: int, target_speaker: str, other_speaker: str) -> dict:
    """
    Parse the dialign output files for a given segment and extract the relevant features.
    """
    dialogue_path = _find_dialign_file(dialign_dir, stem, "dialogue.txt")
    lexicon_path = _find_dialign_file(dialign_dir, stem, "lexicon.tsv")
    self_rep_path = _find_dialign_file(dialign_dir, stem, f"lexicon-self-rep-{other_speaker}.tsv")

    if dialogue_path:
        phrases = _parse_constrained_patterns(dialogue_path, target_pos)
    else:
        phrases = []

    self_rep_count = _count_self_rep_patterns(self_rep_path) if self_rep_path else 0
    other_rep_count = (_count_other_rep_by_speaker(lexicon_path, target_speaker) if lexicon_path else 0)

    return {
        "other_repetition_phrases": phrases,
        "other_speaker_self_rep": self_rep_count,
        "other_speaker_other_rep": other_rep_count,
    }

def _find_coref_file(coref_dir: Path, stem: str) -> Optional[Path]:
    p = coref_dir / f"{stem}.jsonlines"
    return p if p.exists() else None

def _spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return max(s1, s2) <= min(e1, e2)

def _parse_coref_features(coref_path: Path, target_utt_idx: int) -> dict:
    """
    Parse coref output step 3 .jsonlines file and extract coref chains.
    """
    empty = {"coref_related_phrases": [], "coref_global_pos": 0, "coref_local_pos": 0}

    if coref_path is None:
        return empty

    try:
        with open(coref_path, encoding="utf-8") as f:
            data = json.loads(f.readline())
    except (OSError, json.JSONDecodeError):
        return empty

    tokens = data.get("tokens", [])
    utterance_spans = data.get("utterance_span", [])
    predicted = data.get("predicted_clusters", [])
    subtoken_map = data.get("subtoken_map", [])

    if target_utt_idx >= len(utterance_spans):
        return empty

    utt_sub_start, utt_sub_end = utterance_spans[target_utt_idx][:2]

    coref_phrases: List[str] = []
    for cluster in predicted:
        if len(cluster) < 2:
            continue
        for span_start, span_end in cluster:
            if not _spans_overlap(span_start, span_end, utt_sub_start, utt_sub_end):
                continue
            clamped_start = max(span_start, utt_sub_start)
            clamped_end = min(span_end, utt_sub_end)
            if clamped_start > len(subtoken_map) - 1:
                continue
            word_start = subtoken_map[clamped_start]
            word_end = subtoken_map[min(clamped_end, len(subtoken_map) - 1)]
            phrase = " ".join(tokens[word_start: word_end + 1])
            if phrase.strip():
                coref_phrases.append(phrase)

    n = len(coref_phrases)
    # print(n)
    return {
        "coref_related_phrases": coref_phrases,
        "coref_global_pos": n,
        "coref_local_pos": n,
    }

def _infer_other_speaker(speaker: str) -> str:
    if speaker == "A":
        return "B"
    if speaker == "B":
        return "A"
    return "B"

def _extract_row(row, stem: str, local_pos: int, session_df: pd.DataFrame, dialign_dir: Path,
                 coref_dir: Optional[Path]) -> dict:
    speaker = str(row.get("speaker", "A"))
    other_speaker = _infer_other_speaker(speaker)

    target_dialogue_pos = _target_dialogue_pos(session_df, local_pos)

    dialign_feats = _parse_dialign_features(
        dialign_dir=dialign_dir,
        stem=stem,
        target_pos=target_dialogue_pos,
        target_speaker=speaker,
        other_speaker=other_speaker,
    )

    if coref_dir is not None:
        coref_path = _find_coref_file(coref_dir, stem)
        coref_feats = _parse_coref_features(coref_path, target_dialogue_pos)
    else:
        coref_feats = {
            "coref_related_phrases": [],
            "coref_global_pos": 0,
            "coref_local_pos": 0,
        }
    return {**dialign_feats, **coref_feats}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse dialign + coref outputs for self/other-repetition and coref features."
    )
    parser.add_argument("corpus", choices=["cabbs", "noxi"])
    parser.add_argument("--input",   type=Path, required=True,
                        help="Full corpus CSV (all turns, sorted by session+onset_msec)")
    parser.add_argument("--dialign", type=Path, required=True,
                        help="Directory containing dialign output files")
    parser.add_argument("--coref",   type=Path, default=None,
                        help="Directory containing coref .jsonlines files")
    parser.add_argument("--output",  type=Path, required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    session_col = "conv_id" if args.corpus == "cabbs" else "session"

    df = pd.read_csv(args.input)
    df["onset_msec"] = pd.to_numeric(df["onset_msec"], errors="coerce")
    df = df.dropna(subset=["onset_msec"])
    df = df.sort_values([session_col, "onset_msec"]).reset_index(drop=True)

    stem_series = _build_stem_index(df, args.corpus)
    local_pos_series = _get_local_pos(df, args.corpus)

    session_groups = {
        name: grp.sort_values("onset_msec").reset_index(drop=True)
        for name, grp in df.groupby(session_col)
    }

    records = []
    n_missing_dialign = 0
    n_missing_coref   = 0

    for idx in tqdm(df.index, total=len(df), desc="Extracting features"):
        row = df.loc[idx]
        stem = stem_series.loc[idx]
        local_pos = int(local_pos_series.loc[idx])
        sess_df = session_groups[row[session_col]]

        dial_path = _find_dialign_file(args.dialign, stem, "dialogue.txt")
        if dial_path is None:
            n_missing_dialign += 1

        feats = _extract_row(
            row=row,
            stem=stem,
            local_pos=local_pos,
            session_df=sess_df,
            dialign_dir=args.dialign,
            coref_dir=args.coref,
        )

        if args.coref and _find_coref_file(args.coref, stem) is None:
            n_missing_coref += 1

        rec = row.to_dict()
        feats["other_repetition_phrases"] = repr(feats["other_repetition_phrases"])
        feats["coref_related_phrases"] = repr(feats["coref_related_phrases"])
        rec.update(feats)
        records.append(rec)

    out_df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    if args.coref:
        print(f"Missing coref files!")


if __name__ == "__main__":
    main()
