# RecXML

Modular Python implementation of the RecXML data-preparation and news-label generation pipeline.

## Project structure

```text
recxml/
├── notebooks/
│   └── RecXML_pipeline_clean.ipynb
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── recxml/
│       ├── config.py
│       ├── interactions.py
│       ├── io.py
│       ├── labels.py
│       ├── mappings.py
│       ├── news.py
│       ├── pipeline.py
│       └── users.py
├── data/
│   └── MINDsmall_train/        # place dataset here; ignored by Git
├── artifacts/                  # generated files; ignored by Git
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Place the MINDsmall training files (`behaviors.tsv` and `news.tsv`) in:

```text
data/MINDsmall_train/
```

## Run

From the repository root:

```bash
set PYTHONPATH=src
python scripts/run_pipeline.py
```

On macOS/Linux:

```bash
PYTHONPATH=src python scripts/run_pipeline.py
```

Generated intermediate files and arrays are written to `artifacts/` and are excluded from the public repository by `.gitignore`.

## Notebook

`notebooks/RecXML_pipeline_clean.ipynb` contains the notebook workflow with cell outputs removed. The reusable implementation lives in `src/recxml/`.
