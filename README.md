# AI-Driven Predictive Maintenance System for Milling Machines

COS40007 Artificial Intelligence Engineering – Applied Project (Group HCM-6), Swinburne University of Technology, 2026 HE Semester (Sep).

The system predicts machine failure from sensor readings so that maintenance can be planned before a failure happens. It is a reproducible, monitored MLOps pipeline: data validation, feature engineering, model training and threshold selection, drift and performance monitoring, and automated retraining with model promotion on GitHub Actions.

## Team

| Member | Role | Student ID |
|---|---|---|
| Phan Quang Huy | ML Engineer & Data Scientist | 106219394 |
| Đặng Võ Hoàng Long | MLOps Engineer & Data Engineer | 104808745 |

## Pipeline

```
new batch in data/new/ ─┐
push / schedule / manual ┴─> unit tests ─> dvc pull ─> ingest + validate ─> preprocess ─> train
  ─> select_threshold ─> evaluate ─> monitor champion (drift + performance) ─> promotion ─> model-vX.Y tag
```

| Stage (`dvc.yaml`) | Script | What it does |
|---|---|---|
| `ingest` | `src/ingest_validate.py` | Validates `data/ai4i2020.csv` against `config/ingestion_schema.yaml` |
| `ingest_batches` | `src/ingest_batches.py` | Validates every new batch in `data/new/` |
| `preprocess` | `src/preprocess.py`, `src/features.py` | Removes leakage columns, adds physics features, stratified 80/20 split (new rows get their own split) |
| `train` | `src/train.py` | Random Forest (`class_weight="balanced"`), parameters in `params.yaml` |
| `select_threshold` | `src/select_threshold.py` | 5-fold CV on training data: highest F1 with recall ≥ 0.80 |
| `evaluate` | `src/evaluate.py` | Test metrics and plots |

Monitoring and promotion run after the pipeline in CI:

- `src/detect_drift.py` – KS-test + PSI for the sensors, chi-square for `Type`
- `src/monitor.py` – champion performance on the newest batch, alerts, log and drift plot
- `src/promote_model.py` / `src/apply_promotion.py` – the candidate replaces the champion only if recall ≥ 0.80 and F1 is higher

`src/compare_models.py` is a separate experiment (Random Forest vs XGBoost with GridSearchCV) used to choose the model.

## Dataset and DVC

AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020) – 10,000 records of a milling process ([UCI](https://doi.org/10.24432/C5HS5C)).

Data and models are versioned with DVC and stored on DagsHub: https://dagshub.com/long-blip/COS40007-HCM6

- `data/ai4i2020.csv` – original dataset (`data/ai4i2020.csv.dvc`)
- `data/new/` – new labelled batches (`data/new.dvc`)
- `models/champion/` – current production model (`models/champion.dvc`), versioned with `model-vX.Y` Git tags

## Quick start

```bash
pip install -r requirements.txt

# DagsHub credentials (stored locally only, never committed)
dvc remote modify --local dagshub access_key_id <DAGSHUB_TOKEN>
dvc remote modify --local dagshub secret_access_key <DAGSHUB_TOKEN>

dvc pull                                  # data, new batches and champion
dvc repro                                 # run the whole pipeline
python -m unittest discover -s tests      # 71 unit tests
```

Monitor a batch with the current champion:

```bash
python src/detect_drift.py data/new/batch_001.csv
python src/monitor.py data/new/batch_001.csv --model models/champion/model.pkl --threshold models/champion/threshold.json
```

## Adding new data (triggers retraining)

1. `python src/simulate_batches.py` creates simulated batches in `data/simulated/`
2. Copy a batch into `data/new/`, then `dvc add data/new` and `dvc push`
3. Commit `data/new.dvc`, open a pull request and merge it into `main`

See `data/README.md` for details.

## GitHub Actions

Workflow: `.github/workflows/train.yml`

| Trigger | When |
|---|---|
| `push` | Code or data pointers pushed to `main` |
| `pull_request` | Checks only – never publishes a model |
| `schedule` | Daily at 07:17 Vietnam time |
| `workflow_dispatch` | Manual “Run workflow” button |

Each run: unit tests → pull data and champion from DagsHub (`DAGSHUB_TOKEN` secret) → `dvc repro` → monitor the champion on the newest batch → promotion check → on `main`, publish an approved champion as a new `model-vX.Y` tag → upload metrics, reports and plots as artefacts.

## Repository structure

| Path | Contents |
|---|---|
| `src/` | Pipeline, monitoring and promotion scripts |
| `tests/` | Unit tests (`unittest`) |
| `config/` | Ingestion schema |
| `data/` | DVC pointers and data README |
| `models/` | DVC pointer for the champion model |
| `notebooks/` | Task 1 EDA notebook |
| `docs/` | Reports and the system blueprint |
| `params.yaml`, `dvc.yaml`, `dvc.lock` | Parameters and DVC pipeline |

## Project management

- Trello board: https://trello.com/b/VTa0ps5c/cos40007-hcmc6
- Notion documentation: https://app.notion.com/p/COS40007-HCM-6-40152f6e95ab8061ae41c3954da15b2a

| Sprint | Dates | Goal | Assessment |
|---|---|---|---|
| 1 | 19 Oct – 22 Oct | Planning & design, EDA | Task 1 |
| 2 | 23 Oct – 5 Nov | Data pipeline & baseline model, first GitHub Actions workflow | Task 2 |
| 3 | 6 Nov – 26 Nov | DVC, monitoring, automated retraining and promotion | Task 3 |
| 4 | 27 Nov – 4 Dec | Final report, presentation & live demo | Final Report, Demo |

## Branching and commits

- `main` is always runnable; each task uses a `feature/<task>` branch and a pull request reviewed by the other member.
- Semantic commit messages: `feat:`, `fix:`, `refactor:`, `test:`, `ci:`, `docs:`, `chore:`.