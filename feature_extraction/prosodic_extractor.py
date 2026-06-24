"""
Prosodic feature extractor for OIR detection.

Extracts the full prosodic feature set from per-speaker WAV files given segment onset/offset times.
Uses Praat via parselmouth.

Features brief description:
Segment-level
  Pitch features        : min, max, mean, std, range, num_peaks
  Pitch slope           : slope (semitones/sec linear fit over voiced frames)
  Intensity features    : min, max, mean, std, range
  Voice quality         : jitter, shimmer, hnr
  Pause features        : num_pauses, pause_duration, short/medium/long counts, initial/middle/final positional counts,
                          relative_position_longest_pause
  Speech timing         : duration, speech_rate, articulation_rate

Cross-segment / baseline
  Baseline comparison   : pitch z_score, rel_change, range_pos, intensity z_score, rel_change, range_pos
  Transition features   : end_slope, start_slope, transition (computed at segment boundaries)
  Latency               : ts_to_ri_ms, ri_to_rs_ms (gap between TS→RI and RI→RS, added per OIR sequence)
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks
import re
import pandas as pd

# these threshold are either based on literatures or on empirical observations from the data analysis
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 500.0
PITCH_OUTLIER_SIGMA = 2.5

INTENSITY_MIN_PITCH_HZ = 100.0

SILENCE_THRESHOLD_DB = -25.0
MIN_SILENT_PAUSE_MS = 50.0

SHORT_PAUSE_MAX_MS = 200.0
MEDIUM_PAUSE_MAX_MS = 500.0

VOICE_REPORT_MIN_PITCH = 75.0
VOICE_REPORT_MAX_PITCH = 500.0

SEMITONE_REF_HZ = 1.0


def _hz_to_semitones(hz_values: np.ndarray) -> np.ndarray:
    """Convert Hz to semitones relative to 1 Hz (Praat convention)."""
    valid = hz_values > 0
    out = np.full_like(hz_values, np.nan)
    out[valid] = 12.0 * np.log2(hz_values[valid] / SEMITONE_REF_HZ)
    return out

def _remove_pitch_outliers(pitch_hz: np.ndarray) -> np.ndarray:
    """Return voiced pitch values with outliers removed based on interquartile range."""
    voiced = pitch_hz[pitch_hz > 0]
    if voiced.size < 4:
        return voiced
    q1, q3 = np.percentile(voiced, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return voiced[(voiced >= lo) & (voiced <= hi)]

def _linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope of values over times"""
    if len(times) < 2:
        return 0.0
    coeffs = np.polyfit(times, values, 1)
    return float(coeffs[0])

def _count_syllables(text: str) -> int:
    """
    Approx syllable count based on vowel-group counting
    """
    if not text or not text.strip():
        return 0
    vowels = re.findall(r'[aeiouàáâãäåæèéêëìíîïòóôõöùúûüœAEIOUÀÁÂÃÄÅÆÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜŒ]+', text)
    return max(1, len(vowels))

def _seq_number(seq_id_val) -> str:
    """
    Extract a numeric key from a repair sequence ID.
    Eg:
      - CABB-S floats: 46.0  → '46'
      - NOXI strings: 'T0_3' → '3', 'T-1_3' → '3'
    """
    try:
        return str(int(float(seq_id_val)))
    except (ValueError, TypeError):
        m = re.search(r"(\d+)$", str(seq_id_val))
        return m.group(1) if m else str(seq_id_val)

@dataclass
class SpeakerBaseline:
    """Per-speaker pitch and intensity baseline computed from non-OIR segments (all regular dialogue segments)."""
    pitch_mean: float = 0.0
    pitch_std: float = 1.0
    pitch_min: float = 0.0
    pitch_max: float = 1.0
    intensity_mean: float = 0.0
    intensity_std: float = 1.0
    intensity_min: float = 0.0
    intensity_max: float = 1.0


class ProsodicExtractor:
    """
    Extracts prosodic features for each speaker (single file)

    audio_path : Path to the audio WAV file for each speaker.
    """

    def __init__(self, audio_path: str) -> None:
        self.audio_path = audio_path
        self._sound: Optional[parselmouth.Sound] = None
        self.baseline: SpeakerBaseline = SpeakerBaseline()

    @property
    def sound(self) -> parselmouth.Sound:
        if self._sound is None:
            self._sound = parselmouth.Sound(self.audio_path)
        return self._sound

    def _extract_segment(self, onset_ms: float, offset_ms: float) -> parselmouth.Sound:
        """Extract a Sound object for the given time range (ms)."""
        start = onset_ms / 1000.0
        end = offset_ms / 1000.0
        duration = self.sound.get_total_duration()
        start = max(0.0, min(start, duration))
        end = max(start + 0.01, min(end, duration))
        return self.sound.extract_part(from_time=start, to_time=end,
                                       window_shape=parselmouth.WindowShape.RECTANGULAR,
                                       relative_width=1.0, preserve_times=False)

    def compute_session_baseline(self,
                                 non_oir_segments: List[Tuple[float, float]],) -> None:
        """
        Compute per-speaker pitch and intensity baseline from non-OIR segments.
            non_oir_segments : list of (onset_ms, offset_ms)
        """
        all_pitch: List[float] = []
        all_intensity: List[float] = []

        for onset_ms, offset_ms in non_oir_segments:
            if (offset_ms - onset_ms) < 100:
                continue
            try:
                seg = self._extract_segment(onset_ms, offset_ms)
                pitch_vals = self._get_voiced_pitch(seg)
                int_vals = self._get_intensity_values(seg)
                if len(pitch_vals) > 0:
                    all_pitch.extend(pitch_vals.tolist())
                if len(int_vals) > 0:
                    all_intensity.extend(int_vals.tolist())
            except Exception:
                continue

        if all_pitch:
            arr = np.array(all_pitch)
            self.baseline.pitch_mean = float(np.mean(arr))
            self.baseline.pitch_std = float(np.std(arr)) or 1.0
            self.baseline.pitch_min = float(np.min(arr))
            self.baseline.pitch_max = float(np.max(arr))
        if all_intensity:
            arr = np.array(all_intensity)
            self.baseline.intensity_mean = float(np.mean(arr))
            self.baseline.intensity_std = float(np.std(arr)) or 1.0
            self.baseline.intensity_min = float(np.min(arr))
            self.baseline.intensity_max = float(np.max(arr))

    def _get_voiced_pitch(self, seg: parselmouth.Sound) -> np.ndarray:
        """Return array of voiced pitch values (Hz), outliers removed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pitch_obj = seg.to_pitch(pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
        raw = pitch_obj.selected_array["frequency"]
        return _remove_pitch_outliers(raw)

    def _get_pitch_with_times(self, seg: parselmouth.Sound) -> Tuple[np.ndarray, np.ndarray]:
        """Return (times, pitch_hz) for voiced frames, outliers removed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pitch_obj = seg.to_pitch(pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
        times = pitch_obj.xs()
        raw = pitch_obj.selected_array["frequency"]
        voiced_mask = raw > 0
        t = times[voiced_mask]
        v = raw[voiced_mask]
        if v.size > 3:
            q1, q3 = np.percentile(v, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            keep = (v >= lo) & (v <= hi)
            return t[keep], v[keep]
        return t, v

    def _pitch_slope(self, seg: parselmouth.Sound) -> float:
        """
        Pitch slope in semitones/sec
        """
        times, pitch_hz = self._get_pitch_with_times(seg)
        if times.size < 2:
            return 0.0
        semitones = _hz_to_semitones(pitch_hz)
        valid = ~np.isnan(semitones)
        if valid.sum() < 2:
            return 0.0
        return _linear_slope(times[valid], semitones[valid])

    def _num_pitch_peaks(self, seg: parselmouth.Sound) -> float:
        """Count peaks in smoothed pitch contour"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pitch_obj = seg.to_pitch(pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
        voiced = pitch_obj.selected_array["frequency"]
        voiced = voiced[voiced > 0]
        if voiced.size < 3:
            return 0.0
        smoothed = np.convolve(voiced, np.ones(3) / 3, mode="valid")
        peaks, _ = find_peaks(smoothed)
        return float(len(peaks))

    def _get_intensity_values(self, seg: parselmouth.Sound) -> np.ndarray:
        int_obj = seg.to_intensity(minimum_pitch=INTENSITY_MIN_PITCH_HZ)
        vals = int_obj.values.flatten()
        return vals[vals > 0]

    def _voice_quality(self, seg: parselmouth.Sound) -> Tuple[float, float, float]:
        """
        Return (jitter_local, shimmer_local, hnr_mean).
        """
        try:
            point_process = call(seg, "To PointProcess (periodic, cc)", VOICE_REPORT_MIN_PITCH, VOICE_REPORT_MAX_PITCH)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([seg, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            harmonicity = seg.to_harmonicity_cc(
                time_step=0.01,
                minimum_pitch=VOICE_REPORT_MIN_PITCH,
                silence_threshold=0.1,
                periods_per_window=1.0,
            )
            hnr_vals = harmonicity.values.flatten()
            hnr_voiced = hnr_vals[hnr_vals > -200]
            hnr = float(np.mean(hnr_voiced)) if hnr_voiced.size > 0 else float("nan")
            return float(jitter), float(shimmer), hnr
        except Exception:
            return float("nan"), float("nan"), float("nan")

    def _detect_pauses(self, seg: parselmouth.Sound, duration_ms: float) -> Dict[str, float]:
        """
        Detect silent pauses within the segment using intensity thresholding.

        Returns a dict including:
            num_pauses, pause_duration, short_pause_count, medium_pause_count, long_pause_count, initial_segment_pauses,
            middle_segment_pauses, final_segment_pauses, relative_position_longest_pause
        """
        result: Dict[str, float] = {
            "num_pauses": 0, "pause_duration": 0.0, "short_pause_count": 0, "medium_pause_count": 0,
            "long_pause_count": 0, "initial_segment_pauses": 0, "middle_segment_pauses": 0, "final_segment_pauses": 0,
            "relative_position_longest_pause": float("nan"),
        }
        if duration_ms < 100:
            return result

        try:
            int_obj = seg.to_intensity(minimum_pitch=INTENSITY_MIN_PITCH_HZ)
            times = int_obj.xs()
            vals = int_obj.values.flatten()
        except Exception:
            return result

        if vals.size == 0 or np.all(vals <= 0):
            return result

        nonzero = vals[vals > 0]
        if nonzero.size == 0:
            return result
        threshold = np.percentile(nonzero, 95) + SILENCE_THRESHOLD_DB

        is_silent = vals < threshold
        dt = float(times[1] - times[0]) if len(times) > 1 else 0.01
        dt_ms = dt * 1000.0

        pauses: List[Tuple[float, float, float]] = []  # (onset_ms, offset_ms, dur_ms)
        in_pause = False
        pause_start = 0.0
        for i, silent in enumerate(is_silent):
            if silent and not in_pause:
                in_pause = True
                pause_start = times[i]
            elif not silent and in_pause:
                in_pause = False
                dur_ms = (times[i] - pause_start) * 1000.0
                if dur_ms >= MIN_SILENT_PAUSE_MS:
                    pauses.append((pause_start * 1000.0,
                                   times[i] * 1000.0,
                                   dur_ms))
        if in_pause:
            dur_ms = (times[-1] - pause_start) * 1000.0
            if dur_ms >= MIN_SILENT_PAUSE_MS:
                pauses.append((pause_start * 1000.0,
                                times[-1] * 1000.0,
                                dur_ms))

        if not pauses:
            return result

        # Relative postion within segment (early, mid, late)
        third = duration_ms / 3.0
        total_pause_ms = 0.0
        short = medium = long_ = 0
        initial = middle = final = 0
        longest_dur = 0.0
        longest_onset_ms = float("nan")

        for onset_p, offset_p, dur in pauses:
            total_pause_ms += dur
            mid_p = (onset_p + offset_p) / 2.0  # relative to segment start

            if dur <= SHORT_PAUSE_MAX_MS:
                short += 1
            elif dur <= MEDIUM_PAUSE_MAX_MS:
                medium += 1
            else:
                long_ += 1

            if mid_p < third:
                initial += 1
            elif mid_p < 2 * third:
                middle += 1
            else:
                final += 1

            if dur > longest_dur:
                longest_dur = dur
                longest_onset_ms = onset_p

        result["num_pauses"] = float(len(pauses))
        result["pause_duration"] = total_pause_ms
        result["short_pause_count"] = float(short)
        result["medium_pause_count"] = float(medium)
        result["long_pause_count"] = float(long_)
        result["initial_segment_pauses"] = float(initial)
        result["middle_segment_pauses"] = float(middle)
        result["final_segment_pauses"] = float(final)
        if duration_ms > 0 and not np.isnan(longest_onset_ms):
            result["relative_position_longest_pause"] = longest_onset_ms / duration_ms
        return result

    def _baseline_comparison(self, mean_val: float, bl_mean: float, bl_std: float, bl_min: float,
                             bl_max: float) -> Tuple[float, float, float]:
        """Return (z_score, rel_change, range_pos)."""
        z = (mean_val - bl_mean) / bl_std if bl_std > 0 else 0.0
        rel = (mean_val - bl_mean) / bl_mean if bl_mean != 0 else 0.0
        span = bl_max - bl_min
        rp = (mean_val - bl_min) / span if span > 0 else 0.5
        return float(z), float(rel), float(rp)

    def extract(self, onset_ms: float, offset_ms: float, transcript: str = "") -> Dict[str, float]:
        """
        Extract all segment-level prosodic features for given [onset_ms, offset_ms].
            onset_ms, offset_ms: segment boundaries in milliseconds
            transcript: optional transcript text for syllable-based speech rate (approx.)

        Returns
            Final features extracted (NaN where unavailable)
        """
        features: Dict[str, float] = {}
        duration_ms = offset_ms - onset_ms
        duration_sec = duration_ms / 1000.0
        features["utt_duration"] = duration_sec

        if duration_ms < 50:
            return self._nan_features(duration_sec)

        try:
            seg = self._extract_segment(onset_ms, offset_ms)
        except Exception:
            return self._nan_features(duration_sec)

        try:
            voiced = self._get_voiced_pitch(seg)
            if voiced.size > 0:
                features["pros_utt_pitch_min"] = float(np.min(voiced))
                features["pros_utt_pitch_max"] = float(np.max(voiced))
                features["pros_utt_pitch_mean"] = float(np.mean(voiced))
                features["pros_utt_pitch_std"] = float(np.std(voiced))
                features["pros_utt_pitch_range"] = float(np.max(voiced) - np.min(voiced))
            else:
                for k in ("pitch_min", "pitch_max", "pitch_mean", "pitch_std", "pitch_range"):
                    features[f"pros_utt_{k}"] = float("nan")

            features["pros_utt_pitch_slope"] = self._pitch_slope(seg)

            features["pros_utt_num_peaks"] = self._num_pitch_peaks(seg)

            pmean = features.get("pros_utt_pitch_mean", float("nan"))
            if not np.isnan(pmean):
                z, rel, rp = self._baseline_comparison(
                    pmean,
                    self.baseline.pitch_mean, self.baseline.pitch_std,
                    self.baseline.pitch_min, self.baseline.pitch_max,
                )
                features["pros_utt_pitch_z_score"] = z
                features["pros_utt_pitch_rel_change"] = rel
                features["pros_utt_pitch_range_pos"] = rp
            else:
                features["pros_utt_pitch_z_score"] = float("nan")
                features["pros_utt_pitch_rel_change"] = float("nan")
                features["pros_utt_pitch_range_pos"] = float("nan")

            # Normalised against the same speaker/session baseline as the mean z-score.
            pmax = features.get("pros_utt_pitch_max", float("nan"))
            if not np.isnan(pmax) and self.baseline.pitch_std > 0:
                features["pros_utt_pitch_peak_z"] = ((pmax - self.baseline.pitch_mean) / self.baseline.pitch_std)
            else:
                features["pros_utt_pitch_peak_z"] = float("nan")

        except Exception:
            for k in ("pitch_min", "pitch_max", "pitch_mean", "pitch_std", "pitch_range", "pitch_slope", "num_peaks",
                      "pitch_z_score", "pitch_rel_change", "pitch_range_pos", "pitch_peak_z"):
                features[f"pros_utt_{k}"] = float("nan")

        try:
            int_vals = self._get_intensity_values(seg)
            if int_vals.size > 0:
                features["pros_utt_intensity_min"] = float(np.min(int_vals))
                features["pros_utt_intensity_max"] = float(np.max(int_vals))
                features["pros_utt_intensity_mean"] = float(np.mean(int_vals))
                features["pros_utt_intensity_std"] = float(np.std(int_vals))
                features["pros_utt_intensity_range"] = float(np.max(int_vals) - np.min(int_vals))
            else:
                for k in ("intensity_min", "intensity_max", "intensity_mean", "intensity_std", "intensity_range"):
                    features[f"pros_utt_{k}"] = float("nan")

            imean = features.get("pros_utt_intensity_mean", float("nan"))
            if not np.isnan(imean):
                z, rel, rp = self._baseline_comparison(
                    imean,
                    self.baseline.intensity_mean, self.baseline.intensity_std,
                    self.baseline.intensity_min, self.baseline.intensity_max,
                )
                features["pros_utt_intensity_z_score"] = z
                features["pros_utt_intensity_rel_change"] = rel
                features["pros_utt_intensity_range_pos"] = rp
            else:
                features["pros_utt_intensity_z_score"] = float("nan")
                features["pros_utt_intensity_rel_change"] = float("nan")
                features["pros_utt_intensity_range_pos"] = float("nan")

            # Peak-based intensity z-score
            imax = features.get("pros_utt_intensity_max", float("nan"))
            if not np.isnan(imax) and self.baseline.intensity_std > 0:
                features["pros_utt_intensity_peak_z"] = (
                    (imax - self.baseline.intensity_mean) / self.baseline.intensity_std
                )
            else:
                features["pros_utt_intensity_peak_z"] = float("nan")

        except Exception:
            for k in ("intensity_min", "intensity_max", "intensity_mean", "intensity_std", "intensity_range",
                      "intensity_z_score", "intensity_rel_change", "intensity_range_pos", "intensity_peak_z"):
                features[f"pros_utt_{k}"] = float("nan")

        jitter, shimmer, hnr = self._voice_quality(seg)
        features["pros_utt_jitter"] = jitter
        features["pros_utt_shimmer"] = shimmer
        features["pros_utt_hnr"] = hnr

        pause_feats = self._detect_pauses(seg, duration_ms)
        features["pros_utt_num_pauses"] = pause_feats["num_pauses"]
        features["pros_utt_pause_duration"] = pause_feats["pause_duration"]
        features["pros_utt_short_pause_count"] = pause_feats["short_pause_count"]
        features["pros_utt_medium_pause_count"] = pause_feats["medium_pause_count"]
        features["pros_utt_long_pause_count"] = pause_feats["long_pause_count"]
        features["pros_utt_initial_segment_pauses"] = pause_feats["initial_segment_pauses"]
        features["pros_utt_middle_segment_pauses"] = pause_feats["middle_segment_pauses"]
        features["pros_utt_final_segment_pauses"] = pause_feats["final_segment_pauses"]
        features["pros_utt_relative_position_longest_pause"] = (
            pause_feats["relative_position_longest_pause"]
        )

        voiced_ms = duration_ms - pause_feats["pause_duration"]
        voiced_sec = max(voiced_ms / 1000.0, 1e-6)
        n_syllables = _count_syllables(transcript)
        features["pros_utt_speech_rate"] = (n_syllables / duration_sec if duration_sec > 0 else 0.0)
        features["pros_utt_articulation_rate"] = (n_syllables / voiced_sec if voiced_sec > 0 else 0.0)

        return features

    def _nan_features(self, duration_sec: float) -> Dict[str, float]:
        keys = [
            "pros_utt_pitch_min", "pros_utt_pitch_max", "pros_utt_pitch_mean", "pros_utt_pitch_std",
            "pros_utt_pitch_range", "pros_utt_pitch_slope", "pros_utt_pitch_z_score", "pros_utt_pitch_rel_change",
            "pros_utt_pitch_range_pos", "pros_utt_pitch_peak_z", "pros_utt_intensity_min", "pros_utt_intensity_max",
            "pros_utt_intensity_mean", "pros_utt_intensity_std", "pros_utt_intensity_range", "pros_utt_intensity_z_score",
            "pros_utt_intensity_rel_change", "pros_utt_intensity_range_pos", "pros_utt_intensity_peak_z",
            "pros_utt_num_peaks", "pros_utt_speech_rate", "pros_utt_articulation_rate", "pros_utt_jitter",
            "pros_utt_shimmer", "pros_utt_hnr", "pros_utt_num_pauses", "pros_utt_pause_duration",
            "pros_utt_short_pause_count", "pros_utt_medium_pause_count", "pros_utt_long_pause_count",
            "pros_utt_initial_segment_pauses", "pros_utt_middle_segment_pauses", "pros_utt_final_segment_pauses",
            "pros_utt_relative_position_longest_pause",
        ]
        feats = {k: float("nan") for k in keys}
        feats["utt_duration"] = duration_sec
        return feats


def compute_peak_z_columns(df: "pd.DataFrame", session_col: str = "session", speaker_col: str = "speaker",
                           label_col: str = "label") -> "pd.DataFrame":
    """
    Add ``pros_utt_pitch_peak_z`` and ``pros_utt_intensity_peak_z`` to an existing feature extracted DataFrame without
    re-running the whole pipeline.
        df          : features_all DataFrame
        session_col : column grouping rows into one conversation
        speaker_col : column identifying the speaker
        label_col   : 0 = non-OIR baseline, 1 = OIR

    Returns
        Copy of df with two new float columns appended.
    """

    df = df.copy()
    df["pros_utt_pitch_peak_z"] = float("nan")
    df["pros_utt_intensity_peak_z"] = float("nan")

    baseline_mask = df[label_col] == 0

    for (session, speaker), grp in df.groupby([session_col, speaker_col]):
        bl = grp[baseline_mask.reindex(grp.index, fill_value=False)]

        pitch_bl = pd.to_numeric(bl["pros_utt_pitch_max"], errors="coerce").dropna()
        if len(pitch_bl) >= 3:
            bl_pitch_mean = float(pitch_bl.mean())
            bl_pitch_std  = float(pitch_bl.std()) or 1.0
            pitch_max_vals = pd.to_numeric(grp["pros_utt_pitch_max"], errors="coerce")
            df.loc[grp.index, "pros_utt_pitch_peak_z"] = ((pitch_max_vals - bl_pitch_mean) / bl_pitch_std)

        int_bl = pd.to_numeric(bl["pros_utt_intensity_max"], errors="coerce").dropna()
        if len(int_bl) >= 3:
            bl_int_mean = float(int_bl.mean())
            bl_int_std  = float(int_bl.std()) or 1.0
            int_max_vals = pd.to_numeric(grp["pros_utt_intensity_max"], errors="coerce")
            df.loc[grp.index, "pros_utt_intensity_peak_z"] = ((int_max_vals - bl_int_mean) / bl_int_std)
    return df

def compute_transition_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Add pitch transition features between consecutive segments of the same speaker within the same session.

    Returns
        Copy of df with three new float columns added.
        pros_cross_end_slope    : pitch slope at end of previous segment
        pros_cross_start_slope  : pitch slope at start of current segment
        pros_cross_transition   : (prev end pitch - cur start pitch) in st/sec
    """
    df = df.copy()
    for col in ("pros_cross_end_slope", "pros_cross_start_slope", "pros_cross_transition"):
        df[col] = float("nan")

    for (session, speaker), grp in df.groupby(["session", "speaker"]):
        idx = grp.index.tolist()
        for i, cur_idx in enumerate(idx):
            if i == 0:
                continue
            prev_idx = idx[i - 1]
            prev_slope = df.at[prev_idx, "pros_utt_pitch_slope"]
            cur_slope = df.at[cur_idx, "pros_utt_pitch_slope"]
            df.at[cur_idx, "pros_cross_end_slope"] = prev_slope
            df.at[cur_idx, "pros_cross_start_slope"] = cur_slope
            if not (np.isnan(prev_slope) or np.isnan(cur_slope)):
                df.at[cur_idx, "pros_cross_transition"] = cur_slope - prev_slope
    return df

def compute_latency_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Add latency features. Requires columns: session, OIR_seq_id (or repair_seq_id), repair_part, onset_msec, offset_msec.

    Returns:
        Copy of df with three new columns added.
        pros_latency_ts_to_ri_ms : silence gap (ms) from TS end → RI onset (or prev speaker segment -> current segment)
        pros_latency_ri_to_rs_ms : silence gap (ms) from RI end → RS onset (or current segment -> next segment from
                                   other speaker)
    """
    df = df.copy()
    df["pros_latency_ts_to_ri_ms"] = float("nan")
    df["pros_latency_ri_to_rs_ms"] = float("nan")

    if "repair_part" not in df.columns:
        return df

    seq_col = "OIR_seq_id" if "OIR_seq_id" in df.columns else "repair_seq_id"
    if seq_col not in df.columns:
        return df

    TS_LABELS = {"Tmin1", "TS", "TroubleSource", "0"}
    RI_LABELS = {"T0", "RI", "RepairInitiation"}
    RS_LABELS = {"Tplus1", "RS", "RepairSolution"}

    df["_seq_num"] = df[seq_col].apply(lambda x: _seq_number(x) if pd.notna(x) else None)
    group_cols = ["session", "_seq_num"] if "session" in df.columns else ["_seq_num"]

    for keys, grp in df.groupby(group_cols, dropna=True):
        ts_rows = grp[grp["repair_part"].isin(TS_LABELS)]
        ri_rows = grp[grp["repair_part"].isin(RI_LABELS)]
        rs_rows = grp[grp["repair_part"].isin(RS_LABELS)]

        if not ts_rows.empty and not ri_rows.empty:
            ts_end = ts_rows["offset_msec"].max()
            ri_start = ri_rows["onset_msec"].min()
            gap_ts_ri = max(0.0, float(ri_start - ts_end))
            for idx in ri_rows.index:
                df.at[idx, "pros_latency_ts_to_ri_ms"] = gap_ts_ri

        if not ri_rows.empty and not rs_rows.empty:
            ri_end = ri_rows["offset_msec"].max()
            rs_start = rs_rows["onset_msec"].min()
            gap_ri_rs = max(0.0, float(rs_start - ri_end))
            for idx in rs_rows.index:
                df.at[idx, "pros_latency_ri_to_rs_ms"] = gap_ri_rs

    df.drop(columns=["_seq_num"], inplace=True, errors="ignore")
    return df
