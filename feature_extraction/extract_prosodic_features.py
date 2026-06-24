import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from prosodic_extractor import ProsodicExtractor, compute_latency_features, compute_transition_features

def _nan_rows(df: pd.DataFrame) -> list:
    records = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["utt_duration"] = (row.offset_msec - row.onset_msec) / 1000.0
        records.append(rec)
    return records

def cabbs_audio_path(audio_root: Path, pairnr: int, speaker: str) -> Path:
    """Filename convention: pair{N}_synced_ppA.wav  or  pair{N}_synced_ppB.wav"""
    return audio_root / f"pair{pairnr}_synced_pp{speaker}.wav"

def noxi_audio_path(audio_root: Path, session_id: int, role: str) -> Path:
    """Filename convention: 003_expert.wav  or  003_novice.wav"""
    return audio_root / f"{session_id:03d}_{role}.wav"

def extract_session_cabbs(session_df: pd.DataFrame, pairnr: int, audio_root: Path) -> pd.DataFrame:
    records = []
    for speaker in session_df["speaker"].unique():
        spk_df = session_df[session_df["speaker"] == speaker].copy()
        audio_path = cabbs_audio_path(audio_root, pairnr, speaker)

        if not audio_path.exists():
            print(f"Audio not found: {audio_path} — skipping speaker {speaker}")
            records.extend(_nan_rows(spk_df))
            continue

        extractor = ProsodicExtractor(str(audio_path))
        non_oir = spk_df[spk_df["repair_part"].isna()]
        baseline_segs = [
            (row.onset_msec, row.offset_msec)
            for _, row in non_oir.iterrows()
            if (row.offset_msec - row.onset_msec) >= 100
        ]
        if baseline_segs:
            extractor.compute_session_baseline(baseline_segs)

        for _, row in spk_df.iterrows():
            try:
                feats = extractor.extract(
                    onset_ms=row.onset_msec,
                    offset_ms=row.offset_msec,
                    transcript=str(row.speech_original) if pd.notna(row.speech_original) else "",
                )
            except Exception as e:
                print(f"Extraction failed for {row.ID}: {e}")
                feats = extractor._nan_features((row.offset_msec - row.onset_msec) / 1000.0)

            rec = row.to_dict()
            rec.update(feats)
            records.append(rec)
    return pd.DataFrame(records)

def extract_session_noxi(session_df: pd.DataFrame, session_id: int, audio_root: Path) -> pd.DataFrame:
    records = []
    spk_role_map = (session_df[["speaker", "role"]].drop_duplicates().set_index("speaker")["role"].to_dict())

    for speaker, role in spk_role_map.items():
        spk_df = session_df[session_df["speaker"] == speaker].copy()
        audio_path = noxi_audio_path(audio_root, session_id, role)

        if not audio_path.exists():
            print(f"Audio not found: {audio_path} — skipping speaker {speaker} ({role})")
            records.extend(_nan_rows(spk_df))
            continue

        extractor = ProsodicExtractor(str(audio_path))
        non_oir = spk_df[spk_df["repair_part"].isna()]
        baseline_segs = [
            (row.onset_msec, row.offset_msec)
            for _, row in non_oir.iterrows()
            if (row.offset_msec - row.onset_msec) >= 100
        ]
        if baseline_segs:
            extractor.compute_session_baseline(baseline_segs)

        for _, row in spk_df.iterrows():
            try:
                feats = extractor.extract(
                    onset_ms=row.onset_msec,
                    offset_ms=row.offset_msec,
                    transcript=str(row.speech_original) if pd.notna(row.speech_original) else "",
                )
            except Exception as e:
                print(f"Extraction failed row {row.onset_msec}: {e}")
                feats = extractor._nan_features((row.offset_msec - row.onset_msec) / 1000.0)

            rec = row.to_dict()
            rec.update(feats)
            records.append(rec)
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Extract prosodic features for OIR detection.")
    parser.add_argument("corpus", choices=["cabbs", "noxi"])
    parser.add_argument("--input",  type=Path, required=True, help="Full corpus CSV")
    parser.add_argument("--audio",  type=Path, required=True, help="Directory containing speaker WAV files")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=',')
    df["onset_msec"] = pd.to_numeric(df["onset_msec"],  errors="coerce")
    df["offset_msec"] = pd.to_numeric(df["offset_msec"], errors="coerce")
    df = df.dropna(subset=["onset_msec", "offset_msec"])

    if args.corpus == "noxi" and "session" not in df.columns:
        df["session"] = df["session"]
    print(f"Loaded {len(df)} segments")

    all_results = []
    if args.corpus == "cabbs":
        for pairnr, sess_df in df.groupby("pairnr"):
            result = extract_session_cabbs(
                session_df=sess_df.sort_values("onset_msec"),
                pairnr=int(pairnr),
                audio_root=args.audio,
            )
            all_results.append(result)
    else:  # noxi
        for session_name, sess_df in df.groupby("session"):
            session_id = int(sess_df["session_id"].iloc[0])
            result = extract_session_noxi(
                session_df=sess_df.sort_values("onset_msec"),
                session_name=session_name,
                session_id=session_id,
                audio_root=args.audio,
            )
            all_results.append(result)

    out_df = pd.concat(all_results, ignore_index=True)

    print("Computing transition features...")
    if args.corpus == "cabbs" and "session" not in out_df.columns:
        out_df["session"] = out_df["pairnr"].astype(str)
    out_df = compute_transition_features(out_df)

    print("Computing latency features...")
    out_df = compute_latency_features(out_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    # pros_cols = [c for c in out_df.columns if c.startswith("pros_")]
    # nan_pct = out_df[pros_cols].isna().mean().mean() * 100

if __name__ == "__main__":
    main()
