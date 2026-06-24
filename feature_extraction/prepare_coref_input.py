"""
Prepare coreference solver input (jsonlines).
"""
import argparse
import json
import re
import sys
from pathlib import Path
import nltk
import pandas as pd
import spacy
from transformers import AutoTokenizer


N_PRIOR  = 4
N_SUBSEQ = 3

LANG_CONFIGS = {
    "cabbs": {
        "spacy_model":     "nl_core_news_sm",
        "bert_model":      "GroNLP/bert-base-dutch-cased",
        "csv_sep":         ",",
        "session_col":     "conv_id",
        "speaker_col":     "speaker",
        "speech_col":      "speech_original",
        "id_col":          "ID",          # unique segment ID
        "sort_col":        None,          # rows already in order
        "file_prefix":     "cabbs",
    },
    "noxi": {
        "spacy_model":     "fr_core_news_sm",
        "bert_model":      "camembert-base",
        "csv_sep":         ",",
        "session_col":     "session",
        "speaker_col":     "speaker",
        "speech_col":      "speech_original",
        "id_col":          None,
        "sort_col":        "onset_msec",
        "file_prefix":     "noxi",
    },
}

PRONOUNS_NL = {
    'ik', 'jij', 'u', 'je', 'hij', 'zij', 'het', 'wij', 'jullie', 'we', 'ze',
    'mijn', 'jouw', 'uw', 'zijn', 'haar', 'ons', 'onze', 'hun',
    'zich', 'mij', 'me', 'jou', 'hem', 'hen',
    'deze', 'die', 'dat', 'wat', 'wie', 'wiens', 'wier',
    'alles', 'iedereen', 'al', 'alle', 'allen', 'allemaal',
    'elk', 'ieder', 'iets', 'niets', 'iemand', 'niemand',
    'beide', 'allebei', 'men', 'sommige', 'meerdere', 'veel', 'weinig', 'genoeg',
}

PRONOUNS_FR = {
    'je', "j'", 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
    'me', "m'", 'te', "t'", 'se', "s'", 'le', 'la', 'les', 'lui', 'leur', 'y', 'en',
    'moi', 'toi', 'soi', 'eux',
    'ce', "c'", 'ça', 'cela', 'ceci', 'celui', 'celle', 'ceux', 'celles',
    'qui', 'que', "qu'", 'quoi', 'dont', 'où',
    'lequel', 'laquelle', 'lesquels', 'lesquelles',
    'auquel', 'auxquels', 'auxquelles', 'duquel', 'desquels', 'desquelles',
    'tout', 'tous', 'toutes', 'toute',
    'quelque', 'quelqu', 'quelques',
    'chacun', 'chacune', 'aucun', 'aucune',
    'rien', 'personne', 'nul', 'nulle',
    'autre', 'autres', 'même', 'mêmes',
}

PRONOUNS_BY_LANG = {"cabbs": PRONOUNS_NL, "noxi": PRONOUNS_FR}

def session_id_from_noxi(session_name: str) -> str:
    m = re.search(r"pair(\d+)", str(session_name))
    return m.group(1) if m else str(session_name)

def is_empty(text) -> bool:
    if pd.isna(text):
        return True
    return str(text).strip().lower() in {"", "0", "#", "(#)", "nan", "n/a"}

def make_doc_key(corpus: str, session: str, row_id: str) -> str:
    """Consistent with dialign filename: cabbs_{conv_id}_{ID} or noxi_{sid}_{idx}"""
    if corpus == "cabbs":
        safe_session = str(session).replace(".", "_")
        return f"cabbs_{safe_session}_{row_id}"
    else:
        sid = session_id_from_noxi(session)
        return f"noxi_{sid}_{row_id}"

def _split_into_segments(all_subtokens, all_speakers, all_utt_ids,
                         subtoken_map, max_seg_len, constraints1, constraints2,
                         tokenizer):
    curr_idx = 0
    prev_token_idx = 0
    segments, speakers, utt_ids = [], [], []

    while curr_idx < len(all_subtokens):
        end_idx = min(curr_idx + max_seg_len - 1 - 2, len(all_subtokens) - 1)
        while end_idx >= curr_idx and not constraints1[end_idx]:
            end_idx -= 1
        if end_idx < curr_idx:
            end_idx = min(curr_idx + max_seg_len - 1 - 2, len(all_subtokens) - 1)
            while end_idx >= curr_idx and not constraints2[end_idx]:
                end_idx -= 1

        segment = [tokenizer.cls_token] + all_subtokens[curr_idx:end_idx + 1] + [tokenizer.sep_token]
        spk_info = ["[SPL]"] + all_speakers[curr_idx:end_idx + 1] + ["[SPL]"]
        utt_info = ["[UTT]"] + all_utt_ids[curr_idx:end_idx + 1] + ["[UTT]"]

        segments.append(segment)
        speakers.append(spk_info)
        utt_ids.append(utt_info)

        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[end_idx] if end_idx < len(subtoken_map) else prev_token_idx
    return segments, speakers, utt_ids

def build_jsonlines_doc(window_df: pd.DataFrame, doc_key: str, speaker_col: str, speech_col: str, id_col: str, nlp,
                        tokenizer, all_pronouns: set) -> dict:
    conv_tokens = []
    tokens_by_sent_and_utt = []
    speakers_by_utt = []
    utterance_ids = []
    token_num = 0

    for _, row in window_df.iterrows():
        utt_text = str(row[speech_col]) if not pd.isna(row[speech_col]) else ""
        speaker = str(row[speaker_col])
        utt_id = str(row[id_col])

        utt_sent_tokennums = []
        try:
            doc = nlp(utt_text)
            sentences = list(doc.sents)
        except Exception:
            sentences = [utt_text]

        for sent in sentences:
            sent_text = sent.text if hasattr(sent, "text") else str(sent)
            try:
                tokens = nltk.word_tokenize(sent_text)
            except Exception:
                tokens = sent_text.split()
            sent_tokennums = []
            for tok in tokens:
                conv_tokens.append(tok)
                sent_tokennums.append(token_num)
                token_num += 1
            if sent_tokennums:
                utt_sent_tokennums.append(sent_tokennums)

        if not utt_sent_tokennums:
            utt_sent_tokennums = [[]]

        tokens_by_sent_and_utt.append(utt_sent_tokennums)
        speakers_by_utt.append(speaker)
        utterance_ids.append(utt_id)

    all_subtokens = []
    subtoken_map = []
    sentence_end = []
    token_end = []
    all_speakers = []
    all_utt_ids = []
    sentence_map = []
    utterance_spans = []

    word_idx = -1
    sent_idx = -1
    subtk_idx = -1

    for utt_sents, utt_speaker, utt_id in zip(tokens_by_sent_and_utt, speakers_by_utt, utterance_ids):
        utt_start = 0 if subtk_idx < 0 else subtk_idx + 1
        for sent in utt_sents:
            sent_idx += 1
            for idx_in_sent, word_num in enumerate(sent):
                word_idx += 1
                word = conv_tokens[word_num]
                try:
                    subtokens = tokenizer.tokenize(word)
                except Exception:
                    subtokens = [word]
                if not subtokens:
                    subtokens = [word]

                is_last_in_sent = (idx_in_sent == len(sent) - 1)
                token_end += [False] * (len(subtokens) - 1) + [True]

                for idx, subtoken in enumerate(subtokens):
                    subtk_idx += 1
                    all_subtokens.append(subtoken)
                    sentence_end.append(True if (is_last_in_sent and idx == len(subtokens) - 1) else False)
                    subtoken_map.append(word_idx)
                    all_speakers.append(utt_speaker)
                    all_utt_ids.append(utt_id)
                    sentence_map.append(sent_idx)

        utt_end = subtk_idx
        utterance_spans.append((utt_start, utt_end, "utterance"))

    segments, speakers_seg, utt_ids_seg = _split_into_segments(
        all_subtokens, all_speakers, all_utt_ids, subtoken_map,
        max_seg_len=512,
        constraints1=sentence_end,
        constraints2=token_end,
        tokenizer=tokenizer,
    )

    pronouns = []
    for x in range(len(subtoken_map)):
        for y in range(x, min(x + 10, len(subtoken_map))):
            st = subtoken_map[x]
            ed = subtoken_map[y]
            tok = " ".join(conv_tokens[st:ed + 1]).lower()
            if tok in all_pronouns:
                pronouns.append([x, y])

    doc = {
        "doc_key": doc_key,
        "tokens": conv_tokens,
        "tokens_by_sentence_and_utterance": tokens_by_sent_and_utt,
        "speakers_by_utterance": speakers_by_utt,
        "utterance_ids": utterance_ids,
        "ner": [],
        "constituents": [],
        "clusters": [],
        "preceding_5_words": [],
        "following_5_words": [],
        "candidate_entity_spans": [],
        "pronouns": pronouns,
        "subtoken_map": subtoken_map,
        "utterance_span": utterance_spans,
        "clusters_dd": [],
        "gold_anaphors": [],
        "sentences": segments,
        "speakers": speakers_seg,
        "utt_ids": utt_ids_seg,
        "sentence_map": sentence_map,
    }
    return doc

def load_cabbs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=",")
    required = {"conv_id", "speaker", "speech_original", "ID"}
    if missing := required - set(df.columns):
        sys.exit(f"CABB-S CSV missing columns: {missing}")
    return df

def load_noxi(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=",")
    required = {"session", "speaker", "speech_original", "onset_msec"}
    if missing := required - set(df.columns):
        sys.exit(f"NOXI CSV missing columns: {missing}")
    df["onset_msec"] = pd.to_numeric(df["onset_msec"], errors="coerce")
    df = df.dropna(subset=["onset_msec"]).sort_values(["session", "onset_msec"]).reset_index(drop=True)
    df["_row_id"] = df.index.astype(str)
    return df

def get_window(group_rows: pd.DataFrame, target_pos: int) -> pd.DataFrame:
    start = max(0, target_pos - N_PRIOR)
    end = min(len(group_rows), target_pos + N_SUBSEQ + 1)
    return group_rows.iloc[start:end]

def process(corpus: str, csv_path: Path, output_dir: Path) -> None:
    cfg = LANG_CONFIGS[corpus]
    pronouns = PRONOUNS_BY_LANG[corpus]
    session_col = cfg["session_col"]
    speaker_col = cfg["speaker_col"]
    speech_col = cfg["speech_col"]
    id_col_name = cfg["id_col"] if cfg["id_col"] else "_row_id"

    print(f"Loading spaCy model: {cfg['spacy_model']}")
    try:
        nlp = spacy.load(cfg["spacy_model"])
    except OSError:
        sys.exit(
            f"spaCy model '{cfg['spacy_model']}' not found.\n"
            f"Install with: python -m spacy download {cfg['spacy_model']}"
        )

    print(f"Loading BERT tokenizer: {cfg['bert_model']}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["bert_model"])
    except Exception as e:
        sys.exit(f"Could not load tokenizer '{cfg['bert_model']}': {e}")

    print(f"Loading corpus CSV: {csv_path}")
    if corpus == "cabbs":
        df = load_cabbs(csv_path)
    else:
        df = load_noxi(csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_skipped = 0

    for session_name, group in df.groupby(session_col, sort=False):
        group = group.reset_index(drop=True)

        for target_pos in range(len(group)):
            target_row = group.iloc[target_pos]

            if is_empty(target_row[speech_col]):
                total_skipped += 1
                continue

            row_id = str(target_row[id_col_name])
            doc_key = make_doc_key(corpus, session_name, row_id)
            window = get_window(group, target_pos)

            window_clean = window[~window[speech_col].apply(is_empty)].copy()
            if len(window_clean) == 0:
                total_skipped += 1
                continue

            doc = build_jsonlines_doc(
                window_df=window_clean,
                doc_key=doc_key,
                speaker_col=speaker_col,
                speech_col=speech_col,
                id_col=id_col_name,
                nlp=nlp,
                tokenizer=tokenizer,
                all_pronouns=pronouns,
            )

            out_path = output_dir / f"{doc_key}.jsonlines"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False))
                f.write("\n")
            total_written += 1
    print(f"Written: {total_written} jsonlines files to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare CoDiCRAC-format coref input")
    parser.add_argument("corpus", choices=["cabbs", "noxi"], help="Corpus name to process")
    parser.add_argument("--input",  type=Path, required=True, help="Full corpus CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for .jsonlines files")
    args = parser.parse_args()
    process(args.corpus, args.input, args.output)


if __name__ == "__main__":
    main()
