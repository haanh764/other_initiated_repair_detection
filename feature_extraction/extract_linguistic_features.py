import argparse
import ast
import sys
import traceback
from pathlib import Path
from typing import List, Optional
import pandas as pd
from linguistic_extractor import LinguisticExtractor
from linguistic_extractor import _BIGRAM_PATTERNS, _RATIO_TAGS

N_PRIOR = 4
COREF_COL = "coref_related_phrases"
REP_COL   = "other_repetition_phrases"
SELF_REP_COL  = "other_speaker_self_rep"
OTHER_REP_COL = "other_speaker_other_rep"

def _parse_phrases(cell) -> List[str]:
    if pd.isna(cell) or str(cell).strip() in ("", "0", "nan"):
        return []
    s = str(cell).strip()
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
        except (ValueError, SyntaxError):
            pass
    return [s]

def _has_phrases(cell) -> bool:
    return len(_parse_phrases(cell)) > 0

def _safe_count(cell) -> int:
    if pd.isna(cell):
        return 0
    try:
        return int(float(cell))
    except (ValueError, TypeError):
        return 0

def _nan_features(extractor: LinguisticExtractor) -> dict:
    feats: dict = {"cleaned_speech_original": ""}
    for a, b in _BIGRAM_PATTERNS:
        feats[f"ling_contains_{a}_{b}"] = 0
    for tag in _RATIO_TAGS:
        feats[f"ling_ratio_{tag}"] = float("nan")
    for kw in extractor.keywords:
        feats[f"ling_contains_{kw}"] = 0
    feats["ling_is_containing_laughings"] = False
    feats["ling_is_containing_breathings"] = False
    feats["ling_is_containing_mouth noise"] = False
    feats["ling_ends_with_question_mark"] = False
    feats["ling_past_coref_used_ratio"] = float("nan")
    feats["ling_past_other_repetition_used_ratio"] = float("nan")
    feats["ling_other_speaker_self_rep_ratio"] = float("nan")
    feats["ling_other_speaker_other_rep_ratio"] = float("nan")
    return feats

def extract_session(session_df: pd.DataFrame, extractor: LinguisticExtractor, session_name: str, has_coref: bool,
                    has_rep: bool) -> pd.DataFrame:
    rows = session_df.reset_index(drop=True)
    records = []

    for i, row in rows.iterrows():
        coref_phrases: Optional[List[str]] = None
        if has_coref and COREF_COL in rows.columns:
            coref_phrases = _parse_phrases(row.get(COREF_COL))

        past_coref_flags: List[bool] = []
        past_rep_flags: List[bool] = []

        prior_rows = rows.iloc[max(0, i - N_PRIOR): i]
        for _, pr in prior_rows.iterrows():
            if has_coref and COREF_COL in rows.columns:
                past_coref_flags.append(_has_phrases(pr.get(COREF_COL)))
            if has_rep and REP_COL in rows.columns:
                past_rep_flags.append(_has_phrases(pr.get(REP_COL)))

        speaker = str(row.get("speaker", ""))
        other_self_rep = 0
        other_other_rep = 0
        other_total = 0

        if has_rep:
            for _, pr in prior_rows.iterrows():
                if str(pr.get("speaker", "")) != speaker:
                    other_total += 1
                    other_self_rep += _safe_count(pr.get(SELF_REP_COL, 0))
                    other_other_rep += _safe_count(pr.get(OTHER_REP_COL, 0))

        try:
            feats = extractor.extract(
                text=str(row.get("speech_original", "")) if pd.notna(row.get("speech_original")) else "",
                coref_phrases=coref_phrases,
                past_coref_flags=past_coref_flags,
                past_rep_flags=past_rep_flags,
                other_self_rep_count=other_self_rep,
                other_other_rep_count=other_other_rep,
                other_total=other_total,
            )
        except Exception as exc:
            print(f"Feature extraction failed for row {i} in {session_name}: {exc}")
            feats = _nan_features(extractor)

        rec = row.to_dict()
        rec.update(feats)
        records.append(rec)
    return pd.DataFrame(records)

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract linguistic features for OIR detection")
    parser.add_argument("corpus", choices=["cabbs", "noxi"], help="Corpus name to process")
    parser.add_argument("--input", type=Path, required=True, help="Full corpus CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--n-prior", type=int, default=N_PRIOR,
                        help=f"Prior turns for context-related features (default: {N_PRIOR})")
    args = parser.parse_args()

    language = "nl" if args.corpus == "cabbs" else "fr"
    session_col = "conv_id" if args.corpus == "cabbs" else "session"

    df = pd.read_csv(args.input)
    required = {session_col, "speaker", "speech_original", "onset_msec"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Input CSV is missing required columns: {missing}")

    df["onset_msec"] = pd.to_numeric(df["onset_msec"], errors="coerce")
    df = df.dropna(subset=["onset_msec"])
    df = df.sort_values([session_col, "onset_msec"]).reset_index(drop=True)

    has_coref = COREF_COL in df.columns
    has_rep = REP_COL in df.columns

    if not has_coref:
        print(f"Column '{COREF_COL}' not found!")
    if not has_rep:
        print(f"Columns for repetition not found!")

    print(f"Loading spaCy model for language '{language}'...")
    try:
        extractor = LinguisticExtractor(language)
    except OSError as exc:
        sys.exit(str(exc))

    all_results = []
    n_sessions = df[session_col].nunique()
    for idx, (session_name, sess_df) in enumerate(df.groupby(session_col, sort=False), 1):
        sess_sorted = sess_df.sort_values("onset_msec")
        n_turns = len(sess_sorted)
        print(n_turns)
        try:
            result = extract_session(
                session_df=sess_sorted,
                extractor=extractor,
                session_name=str(session_name),
                has_coref=has_coref,
                has_rep=has_rep,
            )
            all_results.append(result)
        except Exception as exc:
            print(f"Session {session_name} failed: {exc} !")
            traceback.print_exc()
            all_results.append(sess_sorted)

    out_df = pd.concat(all_results, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    # ling_cols = [c for c in out_df.columns if c.startswith("ling_") or c == "cleaned_speech_original"]
    # nan_pct = out_df[ling_cols].isna().mean().mean() * 100

if __name__ == "__main__":
    main()
