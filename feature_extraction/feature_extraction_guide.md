# Feature Extraction Guide (Linguistic + Prosodic)

## 1) Prerequisites

```bash
pip install spacy praat-parselmouth numpy scipy pandas transformers
```

Download spaCy language models:
```bash
python -m spacy download nl_core_news_sm
python -m spacy download fr_core_news_sm
```

## 2) Input Expectations

### Linguistic extractor expects
- `text` (raw transcript segment)
- Coref and repetition context: because we compute the features using coreference and self/other-repetition, it is needed
to run the coreference solver and the dialign tools first.
- For coreference solver, we use the source code from this [repo](https://github.com/samlee946/utd-codi-crac2022).
- For the dialign tools, we use the source code from this [repo](https://github.com/GuillaumeDD/dialign).
- To prepare the input for the coreference solver, we can use the following script:

```bash
python prepare_coref_input.py noxi \\
        --input /path/to/noxi_corpus.csv \\
        --output /path/to/coref_input_noxi
```
- To prepare the input for the dialign tools, we can use the following script:

```bash
python prepare_dialign_input.py noxi \\
        --input  /path/to/noxi_corpus.csv \\
        --output /path/to/output_dir
```        

- After running the coreference solver and dialign tools, we can use the following script to generate the expected columns
for the linguistic extractor:

```bash
python parse_dialign_coref_features.py noxi \\
        --input   /path/to/noxi_corpus.csv \\
        --dialign /path/to/noxi_dialign_output \\
        --coref   /path/to/noxi_coref_output \\
        --output  /path/to/noxi_repetition_coref_features.csv
```

## 3) Linguistic Extractor Usage

To run the linguistic extractor, use the following command:

```bash
python extract_linguistic_features.py noxi \\
       --input  /path/to/noxi_repetition_coref_features.csv \\
       --output /path/to/noxi_linguistic_features.csv
```

## 4) Prosodic Extractor Usage

To run the prosodic extractor, use the following command:

```bash
python extract_prosodic_features.py noxi \\
        --input  /path/to/noxi_corpus.csv \\
        --audio  /path/to/noxi_media_root/audio \\
        --output /path/to/noxi_prosodic_features.csv
```