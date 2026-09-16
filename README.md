# Food Security Interviews: Fine-tuned GLiNER Baseline

Anonymization baseline for Spanish  interview transcripts from a food security study.
Instead of combining several models and an LLM, this project fine-tunes a single GLiNER model
on manually annotated interviews and uses its predictions directly to anonymize the text.
Detected entities: people (`PERSONA`), places (`LUGAR`) and organizations (`ORGANIZACION`).

This baseline is meant to be compared with the hybrid pipeline (spaCy + GLiNER + LLM).

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Data layout](#data-layout)
5. [Running the notebooks step by step](#running-the-notebooks-step-by-step)
6. [Results so far](#results-so-far)
7. [Next steps](#next-steps)

## Pipeline overview

```
 annotated interviews (C1-C6)            raw interviews (59)
            │                                    │
            ▼                                    ▼
 01 build dataset + fine-tune ──► model ──► 02 GLiNER NER ──► 03 accept all + anonymize
   (difflib alignment, 250-word chunks)        (threshold 0.55)     (no LLM, EntityReplacer)
```

| Stage | Notebook | Input | Output |
|---|---|---|---|
| 1 | `01_Fine_Tunning.ipynb` | `data/raw/fine_tunning/*.txt` + `data/ground_truth/entrevistas_anotadas/*.txt` | `data/processed/gliner_train_dataset.json`, `gliner_eval_dataset.json`, `models/gliner_entrevistas_finetuned/` |
| 2 | `02_gliner_ner.ipynb` | `data/raw/entrevistas_originales/*.txt` + fine-tuned model | `data/processed/entidades_candidatas_gliner/*_candidates.json` |
| 3 | `03_Baseline_gliner.ipynb` | GLiNER candidates + original text | `data/processed/entidades_gliner_validated/*_validated.json`, `data/processed/entrevistas_anonimizadas_gliner/*_anonimizado.txt` |

## Repository structure

```
.
├── notebooks/
│   ├── 01_Fine_Tunning.ipynb
│   ├── 02_gliner_ner.ipynb
│   └── 03_Baseline_gliner.ipynb
├── src/
│   ├── ner/gliner_ner.py            # GlinerNER, DocumentResult
│   └── anonymization/replacer.py    # EntityReplacer
├── data/                            # not versioned
├── models/                          # not versioned
├── requirements.txt
├── .gitignore
└── README.md
```

All notebooks add the project root to `sys.path` (`PROJECT_ROOT = Path("../")`), so run them
from inside `notebooks/`.

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

# 2. Virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt
```

The base model `urchade/gliner_multi_pii-v1` (about 500 MB) is downloaded from Hugging Face on
first use. Training works on CPU but is much faster with a CUDA GPU.

## Data layout

Create these folders locally and place the files there (they are ignored by git):

```
data/
├── raw/
│   ├── fine_tunning/                # interviews used for training (C1-C6)
│   └── entrevistas_originales/      # the 59 interviews to anonymize
├── ground_truth/
│   └── entrevistas_anotadas/        # annotated versions, same file names, entities marked as **tag**
└── processed/                       # created by the notebooks
```

## Running the notebooks step by step

### 01 · Build the training set and fine-tune (`01_Fine_Tunning.ipynb`)

**Label mapping.** `clasificar_etiqueta()` maps the annotation tags to the three main labels and
discards everything else (events, miscellaneous):

| Tag contains | Label |
|---|---|
| `person` | `PERSONA` |
| `university`, `company`, `org` | `ORGANIZACION` |
| `place`, `city`, `country` | `LUGAR` |

**Alignment and chunking.** `procesar_documento_completo()` aligns the original and the annotated
text word by word with `difflib.SequenceMatcher` (punctuation removed and lower-cased only for
matching). Every replaced block whose annotated side contains `**...**` becomes a span
`[start, end, label]`. The text is then split into 250-word chunks, and only chunks with at least
one entity are kept, in GLiNER format:

```json
{"tokenized_text": ["Bueno", "pues", "..."], "ner": [[12, 13, "PERSONA"]]}
```

**Split.** Interviews C1 to C6 go to the training set. Any other annotated file goes to the
evaluation set.

**Training** with the GLiNER API:

| Hyperparameter | Value |
|---|---|
| Base model | `urchade/gliner_multi_pii-v1` |
| `learning_rate` | 5e-6 |
| `others_lr` | 1e-5 |
| `per_device_train_batch_size` | 8 (use 4 or 2 if memory runs out) |
| `num_train_epochs` | 5 |
| `save_steps` / `save_total_limit` | 100 / 2 |
| Evaluation during training | none (all data used for training) |

The model is saved to `models/gliner_entrevistas_finetuned/` and checkpoints to `models/checkpoints/`.

### 02 · Run the fine-tuned model (`02_gliner_ner.ipynb`)

```python
gliner = GlinerNER(
    model     = "models/gliner_entrevistas_finetuned",
    threshold = 0.55,
    labels    = ["persona", "lugar", "organizacion"],
)
results = gliner.process_batch(texts, doc_ids=doc_ids)
```

The notebook also calibrates the threshold, shows the entities with `displacy`, reviews
frequencies and likely errors, maps secondary labels (`pais`, `avenida` to `LUGAR`;
`universidad`, `secretaria` to `ORGANIZACION`) and exports one `*_candidates.json` per interview.

### 03 · Baseline anonymization (`03_Baseline_gliner.ipynb`)

There is no LLM and no filtering step: every GLiNER candidate is accepted as confirmed.
Candidates are de-duplicated by `(lower-cased text, label)` and saved in the same
`*_validated.json` format used by the hybrid pipeline, so both projects share the same
`EntityReplacer`:

```python
replacer = EntityReplacer(symbol="brackets")
replacer.anonymize_directory(validated_dir=ENTS_VAL, originals_dir=ORIGINAL_DIR, output_dir=OUTPUT_DIR)
```

The output is 59 `*_anonimizado.txt` files with placeholders such as `[PERSONA_4]` or `[LUGAR_14]`.

## Results so far

| Metric (59 interviews) | Value |
|---|---|
| Replacements in the final texts | 15,425 |
| Unique entities replaced | 2,580 |

The hybrid pipeline replaces 9,120 occurrences (2,549 unique entities) on the same corpus, so
this baseline masks much more text. Whether that means better recall or more false positives has
to be measured against the ground truth.

## Next steps


- Add a validation split during fine-tuning to monitor overfitting.
