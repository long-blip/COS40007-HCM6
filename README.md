# AI-Driven Predictive Maintenance System for Milling Machines

COS40007 Artificial Intelligence Engineering – Applied Project (Group HCM-6), Swinburne University of Technology, 2026 HE Semester (Sep).

The system predicts machine failure risk from sensor readings so that maintenance can be scheduled before a failure happens. It covers the full AI engineering lifecycle: data ingestion, preprocessing, feature engineering, model training, a Streamlit dashboard, monitoring, drift detection and automated retraining with GitHub Actions.

## Team

| Member | Role | Student ID |
|---|---|---|
| Phan Quang Huy | ML Engineer & Data Scientist | 106219394 |
| Đặng Võ Hoàng Long | MLOps Engineer & Data Engineer | 104808745 |

## Dataset

AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020) – 10,000 synthetic records of a milling process.
Source: [UCI Machine Learning Repository](https://doi.org/10.24432/C5HS5C) · [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).  
A copy of the dataset is committed at data/ai4i2020.csv so the GitHub Actions pipeline can run.

## Repository structure

| Folder | Contents |
|---|---|
| `data/` | Datasets (tracked with DVC from Sprint 3) |
| `notebooks/` | EDA notebook and figures |
| `src/` | Pipeline code: ingestion, preprocessing, features, training (Sprint 2+) |
| `models/` | Trained models (tracked with DVC) |
| `tests/` | Unit tests |
| `docs/` | Reports and the system blueprint |
| `.github/workflows/` | GitHub Actions CI/CD workflows (Sprint 2+) |

## Project links

- Trello board: https://trello.com/b/VTa0ps5c/cos40007-hcmc6
- Notion documentation: https://app.notion.com/p/COS40007-HCM-6-40152f6e95ab8061ae41c3954da15b2a

## Sprint plan

| Sprint | Dates | Goal | Assessment |
|---|---|---|---|
| 1 | 19 Oct – 22 Oct | Planning & design, EDA | Task 1 |
| 2 | 23 Oct – 5 Nov | Data pipeline & baseline model, first GitHub Actions workflow | Task 2 |
| 3 | 6 Nov – 26 Nov | Automation, DVC, monitoring & dashboard | Task 3 |
| 4 | 27 Nov – 4 Dec | Final report, presentation & live demo | Final Report, Demo |

## Branching and commits

- `main` – stable, always runnable · `dev` – integration · `feature/<task>` – one branch per task
- All changes go through pull requests reviewed by the other member.
- Semantic commit messages: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.

## Run the EDA notebook

```bash
pip install -r requirements.txt
# put ai4i2020.csv next to notebooks/Task1_EDA_AI4I.ipynb, then open it in Jupyter / VS Code
```
