"""
Linguistic feature extractor for OIR detection.

Extracts the following features (columns) from dialogue transcripts:
  - cleaned_speech_original: original cleaned transcript from data (already annotated in our corpora)
  - ling_contains_*: binary POS tag bigram flags
  - ling_ratio_*: token-normalised POS tag ratios
  - ling_contains_<keyword>: language-specific keyword flags (some keywords that we found in our pre-analysis
                                    that are most indicative of OIR)
  - ling_is_containing_*: paralinguistic markers (laugh / breath / mouth) - pre-annotated in our corpus
  - ling_ends_with_question_mark: surface question marker - pre-annotated in our corpus
  - ling_past_coref_used_ratio: fraction of prior context window (prior segments) with coref phrase
  - ling_past_other_repetition_used_ratio: fraction of context window (prior segments) with repetition (from the other
                                                speaker)
  - ling_other_speaker_self_rep_ratio: other-speaker self-repetition rate in window
  - ling_other_speaker_other_rep_ratio: other-speaker other-repetition rate in window

Languages:
  'nl' — Dutch  (CABB-S corpus, spaCy nl_core_news_sm)
  'fr' — French (NOXI corpus, spaCy fr_core_news_sm)

The coref- and repetition-based features are pre-computed based on the coreference tool and repetition analysis tools
"""

import re
from typing import Any, Dict, List, Optional
import math
import spacy


SPACY_MODELS: Dict[str, str] = {
    "nl": "nl_core_news_sm",
    "fr": "fr_core_news_sm",
}

# Language-specific OIR-diagnostic keywords (lemma / surface forms).
# Based on our analysis in previous work, they are CORPUS-SPECIFIC, NOT generalize for all corpora
LANGUAGE_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "nl": {
        "nog":     ["nog"],
        "kunnen":  ["kunnen", "kan", "kunt", "kon", "konden"],
        "wachten": ["wachten", "wacht", "wachtte", "wachtten"],
        "zitten":  ["zitten", "zit", "zat", "zaten"],
        "aan":     ["aan"],
        "wat":     ["wat"],
        "zijn":    ["zijn", "is", "was", "waren"],
    },
    "fr": {
        "quoi":     ["quoi"],
        "comment":  ["comment"],
        "pardon":   ["pardon"],
        "hein":     ["hein"],
        "attendre": ["attendre", "attends", "attend", "attendez", "attendait"],
        "pouvoir":  ["pouvoir", "peut", "peux", "pouvez", "pouvait", "pouvaient"],
        "encore":   ["encore"],
        "quel":     ["quel", "quelle", "quels", "quelles"],
    },
}

# POS bigrams that are indicative of OIR repair initiations - based on our analysis in previous work, COREF: coreference

_BIGRAM_PATTERNS: List[tuple] = [
    ("PRON_Int", "PRON_Prs"),
    ("AUX",      "VERB"),
    ("COREF",    "ADP"),
    ("VERB",     "PRON_Prs"),
    ("PRON_Dem", "AUX"),
    ("ADP",      "DET"),
    ("COREF",    "VERB"),
    ("ADJ",      "NOUN"),
    ("DET",      "NOUN"),
    ("VERB",     "COREF"),
    ("ADP",      "NOUN"),
    ("INTJ",     "DET"),
    ("PRON_Prs", "VERB"),
    ("VERB",     "ADV"),
    ("ADV",      "NOUN"),
    ("ADV",      "VERB"),
    ("ADP",      "COREF"),
    ("PRON_Int", "VERB"),
    ("PRON_Prs", "ADV"),
]

# POS tags for ratio features
_RATIO_TAGS: List[str] = [
    "CCONJ", "PRON_Prs", "AUX", "SCONJ", "ADJ", "PRON_Int",
    "NOUN", "ADP", "DET", "PRON_Dem", "INTJ", "ADV", "SYM", "VERB",
]

 # Paralinguistic marker patterns
_LAUGHING_RE = re.compile(
    r"\[laughter\]|\[laugh\]|\*laughter\*|\*laugh\*|\*g\*"
    r"|\[giggles?\]|\bhaha+\b|\bhehe\b|\bahhah\b|\bghghg\b",
    re.IGNORECASE,
)
_BREATHING_RE = re.compile(
    r"\[breath(?:ing)?\]|\*breath(?:ing)?\*|\(\.+\)|\[intake\]|\bh+m+\b",
    re.IGNORECASE,
)
_MOUTH_RE = re.compile(
    r"\[mouth noise\]|\[click\]|\[smack\]|\[lip\b[^\]]*\]|\*mouth\*",
    re.IGNORECASE,
)

_EMPTY_TOKENS = {"", "0", "#", "(#)", "nan", "n/a"} # for cleaning purpose


def _is_empty(text: Any) -> bool:
    if text is None:
        return True
    try:
        if math.isnan(float(text)):
            return True
    except (TypeError, ValueError):
        pass
    return str(text).strip().lower() in _EMPTY_TOKENS


class LinguisticExtractor:
    """Extract linguistic features from a given dialogue segment."""

    def __init__(self, language: str) -> None:
        """
            language: 'nl' for Dutch (CABB-S) or 'fr' for French (NOXI)
        """
        if language not in SPACY_MODELS:
            raise ValueError(
                f"Unsupported language '{language}'. Choose from {list(SPACY_MODELS)}"
            )
        self.language = language
        model_name = SPACY_MODELS[language]
        try:
            self.nlp = spacy.load(model_name, disable=["ner", "parser"])
        except OSError:
            raise OSError(
                f"spaCy model '{model_name}' not found.\n"
                f"Install with: python -m spacy download {model_name}"
            )
        self.keywords = LANGUAGE_KEYWORDS.get(language, {})

    def clean_speech(self, text: str) -> str:
        if _is_empty(text):
            return ""
        s = str(text).strip()
        s = re.sub(r"\*[^*]+\*", " ", s)
        s = re.sub(r"\[[^\]]*\]", " ", s)
        s = re.sub(r"\([^)]*\)", " ", s)
        s = re.sub(r"(\w+)-\s", r"\1 ", s)
        s = " ".join(s.split())
        return s

    def _pos_sequence(
        self,
        cleaned_text: str,
        coref_phrases: Optional[List[str]]) -> List[str]:
        """
        Return a POS tag sequence.

        Pronouns are split into PRON_Int/PRON_Prs/PRON_Dem using spaCy morphology. Tokens that appear in coref_phrases
        are replaced by the pseudo-tag 'COREF'.
        """
        if not cleaned_text:
            return []

        coref_tokens: set = set()
        if coref_phrases:
            for phrase in coref_phrases:
                if isinstance(phrase, str) and phrase.strip():
                    for tok in phrase.lower().split():
                        coref_tokens.add(re.sub(r"[^\w]", "", tok))

        doc = self.nlp(cleaned_text)
        seq: List[str] = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            tok_lower = re.sub(r"[^\w]", "", token.lower_)
            if coref_tokens and tok_lower in coref_tokens:
                seq.append("COREF")
                continue
            if token.pos_ == "PRON":
                pron_types = token.morph.get("PronType")
                pt = pron_types[0] if pron_types else ""
                if pt in ("Int", "Rel"):
                    seq.append("PRON_Int")
                elif pt == "Dem":
                    seq.append("PRON_Dem")
                else:
                    seq.append("PRON_Prs")
            else:
                seq.append(token.pos_)
        return seq

    def _bigram_features(self, pos_seq: List[str]) -> Dict[str, int]:
        bigrams = set(zip(pos_seq[:-1], pos_seq[1:]))
        return {
            f"ling_contains_{a}_{b}": int((a, b) in bigrams)
            for a, b in _BIGRAM_PATTERNS
        }

    def _ratio_features(self, pos_seq: List[str]) -> Dict[str, float]:
        n = len(pos_seq) or 1
        return {
            f"ling_ratio_{tag}": pos_seq.count(tag) / n
            for tag in _RATIO_TAGS
        }

    def _keyword_features(self, text_lower: str) -> Dict[str, int]:
        tokens = set(re.findall(r"\b\w+\b", text_lower))
        return {
            f"ling_contains_{kw}": int(bool(tokens & set(forms)))
            for kw, forms in self.keywords.items()
        }

    def _paralinguistic_features(self, raw_text: str) -> Dict[str, bool]:
        return {
            "ling_is_containing_laughings":  bool(_LAUGHING_RE.search(raw_text)),
            "ling_is_containing_breathings": bool(_BREATHING_RE.search(raw_text)),
            "ling_is_containing_mouth noise": bool(_MOUTH_RE.search(raw_text)),
        }

    def _surface_features(self, raw_text: str) -> Dict[str, bool]:
        return {
            "ling_ends_with_question_mark": raw_text.rstrip().endswith("?"),
        }

    @staticmethod
    def _context_ratio_features(
            past_coref_flags: List[bool],
            past_rep_flags: List[bool],
            other_self_rep_count: int,
            other_other_rep_count: int,
            other_total: int,
    ) -> Dict[str, float]:
        n_past = len(past_coref_flags)
        return {
            "ling_past_coref_used_ratio": (
                sum(past_coref_flags) / n_past if n_past > 0 else 0.0
            ),
            "ling_past_other_repetition_used_ratio": (
                sum(past_rep_flags) / n_past if n_past > 0 else 0.0
            ),
            "ling_other_speaker_self_rep_ratio": (
                other_self_rep_count / other_total if other_total > 0 else 0.0
            ),
            "ling_other_speaker_other_rep_ratio": (
                other_other_rep_count / other_total if other_total > 0 else 0.0
            ),
        }

    def extract(
        self,
        text: str,
        coref_phrases: Optional[List[str]] = None,
        past_coref_flags: Optional[List[bool]] = None,
        past_rep_flags: Optional[List[bool]] = None,
        other_self_rep_count: int = 0,
        other_other_rep_count: int = 0,
        other_total: int = 0,
    ) -> Dict[str, Any]:
        """
        Extract all linguistic features for one segment.
            text:                    Raw transcript text.
            coref_phrases:           List of coreferent phrases (from coref solver).
                                     Used to tag tokens as 'COREF' in POS sequence.
            past_coref_flags:        Per-prior-segment flag: was coref phrase present?
            past_rep_flags:          Per-prior-segment flag: was repetition phrase present?
            other_self_rep_count:    # other-speaker segments in window with self-repetition.
            other_other_rep_count:   # other-speaker segments in window with other-repetition.
            other_total:             Total # other-speaker segments in window.

        Returns:
            Final features dictionary.
        """
        raw = text if isinstance(text, str) else ""
        cleaned = self.clean_speech(raw)
        pos_seq = self._pos_sequence(cleaned, coref_phrases) if cleaned else []

        feats: Dict[str, Any] = {"cleaned_speech_original": cleaned}
        feats.update(self._bigram_features(pos_seq))
        feats.update(self._ratio_features(pos_seq))
        feats.update(self._keyword_features(cleaned.lower()))
        feats.update(self._paralinguistic_features(raw))
        feats.update(self._surface_features(raw))
        feats.update(
            self._context_ratio_features(
                past_coref_flags=past_coref_flags or [],
                past_rep_flags=past_rep_flags or [],
                other_self_rep_count=other_self_rep_count,
                other_other_rep_count=other_other_rep_count,
                other_total=other_total,
            )
        )
        return feats
