"""
Prepare dialign input TSV files for repetition feature extraction.
For every segment (each row in the corpus CSV), creates one TSV file containing:
4 prior segment + target segment + 3 subsequent segment (these numbers are corpus specific)

Output format (standard dialign input):
    SPEAKER<TAB>utterance
    ...one line per non-empty segment (utterance)...

Output filename:
    CABB-S: cabbs_{conv_id}_{ID}.tsv        (conv_id = pairnr_trial, ID = row ID)
    NOXI: noxi_{session_id}_{row_idx}.tsv (session_id = numeric pair id)
"""

import argparse
import re
import sys
from pathlib import Path
import pandas as pd

N_PRIOR = 4       # prior segments in model window
N_SUBSEQ = 3      # subsequent segments in model window
EMPTY_TOKENS = {"", "0", "#", "(#)", "nan", "n/a"}

def is_empty(text) -> bool:
    if pd.isna(text):
        return True
    return str(text).strip().lower() in EMPTY_TOKENS

def clean(text) -> str:
    return str(text).strip()

def session_id_from_noxi(session_name: str) -> str:
    """'batch1_pair3_processed_segments' -> '3'"""
    m = re.search(r"pair(\d+)", str(session_name))
    return m.group(1) if m else str(session_name)

def write_tsv(turns: pd.DataFrame, speaker_col: str, speech_col: str, filepath: Path) -> int:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    speakers_seen: set = set()
    for _, row in turns.iterrows():
        utt = clean(row[speech_col])
        if not is_empty(utt):
            lines.append(f"{row[speaker_col]}\t{utt}")
            speakers_seen.add(row[speaker_col])

    if len(speakers_seen) < 2:
        return -1

    if lines:
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)

def build_windows(group_df: pd.DataFrame, speaker_col: str, speech_col: str, output_dir: Path, prefix: str,
                  id_col: str) -> tuple[int, int]:
    rows = group_df.reset_index(drop=True)
    n = len(rows)
    written = 0
    skipped = 0

    for i in range(n):
        prior_start = max(0, i - N_PRIOR)
        subseq_end = min(n, i + N_SUBSEQ + 1)

        window = rows.iloc[prior_start:subseq_end]
        row_id = str(rows.iloc[i][id_col]).replace("/", "_").replace(" ", "_")
        filename = f"{prefix}_{row_id}.tsv"

        n_lines = write_tsv(window, speaker_col, speech_col, output_dir / filename)
        if n_lines == -1:
            skipped += 1
        elif n_lines > 0:
            written += 1
    return written, skipped

def prepare_cabbs(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    required = {"conv_id", "speaker", "speech_original", "ID"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CABB-S CSV missing columns: {missing}")

    total_written = 0
    total_skipped = 0
    for conv_id, group in df.groupby("conv_id", sort=False):
        prefix = f"cabbs_{str(conv_id).replace('.', '_')}"
        written, skipped = build_windows(
            group_df=group,
            speaker_col="speaker",
            speech_col="speech_original",
            output_dir=output_dir,
            prefix=prefix,
            id_col="ID",
        )
        total_written += written
        total_skipped += skipped

    if total_skipped:
        print(f"Skipped: {total_skipped}")

def prepare_noxi(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv, sep=",")
    required = {"session", "speaker", "speech_original", "onset_msec"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"NOXI CSV missing columns: {missing}")

    df["onset_msec"] = pd.to_numeric(df["onset_msec"], errors="coerce")
    df = df.dropna(subset=["onset_msec"])
    df = df.sort_values(["session", "onset_msec"]).reset_index(drop=True)
    df["_row_idx"] = df.index.astype(str)

    total_written = 0
    total_skipped = 0
    for session_name, group in df.groupby("session", sort=False):
        sid = session_id_from_noxi(session_name)
        prefix = f"noxi_{sid}"
        written, skipped = build_windows(
            group_df=group,
            speaker_col="speaker",
            speech_col="speech_original",
            output_dir=output_dir,
            prefix=prefix,
            id_col="_row_idx",
        )
        total_written += written
        total_skipped += skipped

    if total_skipped:
        print(f"Skipped: {total_skipped}")

def main():
    parser = argparse.ArgumentParser(description="Prepare dialign input TSV files.")
    parser.add_argument("corpus", choices=["cabbs", "noxi"], help="Corpus name")
    parser.add_argument("--input", type=Path, required=True, help="Path to corpus CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for TSV files")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.corpus == "cabbs":
        prepare_cabbs(args.input, args.output)
    else:
        prepare_noxi(args.input, args.output)

if __name__ == "__main__":
    main()
