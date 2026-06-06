# FinPersona

FinPersona is a banking analytics project with three model workflows:
credit default risk, customer churn prediction, and product recommendation. A
Flask gateway provides browser UI for the three models.

This repository includes synthetic sample data, data-cleaning scripts, trained
model artifacts, output reports, and a pytest suite for the core data and model
logic.
Synthetic sample data is used here because the real data is confidential.

## Project layout

```text
FinPersona/
|-- main.py                         # Starts all backend services and the UI gateway
|-- UI/                             # Unified Flask gateway and browser interface
|   |-- app.py                      # Proxies UI API calls to the module servers
|   `-- templates/index.html
|-- data/                           # Synthetic data
|   |-- generate_fake_data.py       # Generates prime CSVs and transaction XLSX files that have the same features as the real data
|   |-- prime/                      # Raw monthly prime/card snapshots
|   |-- transaction/                # Raw monthly transaction files
|-- data_cleaning/                  # Prime and transaction cleaning pipelines
|-- train/
|   |-- credit/                     # Credit default risk module
|   |-- churn/                      # Churn prediction module
|   `-- recommendation/             # Product recommendation module
|-- models/                         # Saved model and app-state artifacts
|-- outputs/                        # Generated score CSVs and evaluation reports
|-- tests/                          # Tests
```

Note: the image asset directory is currently named `assests/`, and the code
uses that spelling.

## Architecture

The application runs as four local Flask processes:

| Service | File | Port | Purpose |
| --- | --- | ---: | --- |
| Recommendation | `train/recommendation/app.py` | 5000 | Preprocesses data, trains XGBoost and content-based recommenders, predicts products |
| Credit Risk | `train/credit/app.py` | 5005 | Trains and scores credit default risk |
| Churn | `train/churn/app.py` | 5006 | Trains churn classifiers and looks up customer churn risk |
| Gateway UI | `UI/app.py` | 5050 | Serves the unified UI and proxies requests to the three module services |

`main.py` starts all four services and opens the gateway at
`http://127.0.0.1:5050`.

## Data flow

1. `data/generate_fake_data.py` creates monthly raw prime and transaction files.
2. `data_cleaning/prime_id_creation.py` builds a stable `CUSTOMER_ID` mapping
   from prime files and writes active and historical cleaned prime CSVs.
3. `data_cleaning/transaction_id_mapping.py` maps transaction rows to
   `CUSTOMER_ID` using cleaned active prime records, then writes matched rows
   and missing-ID rows.
4. The training modules read cleaned data from `data/prime_cleaned/` and
   `data/transaction_cleaned/`.
5. Model artifacts are written to `models/`, reports and batch scores are
   written to `outputs/`, and recommendation features are written to
   `data/engineered_recommendation_features/`.

## Module summary

### Credit risk

The credit module creates a default target from card account statuses, engineers
prime, transaction, and temporal trend features, checks for high-risk leakage
features, trains an XGBoost binary classifier, and writes customer-level default
scores.

Important outputs:

- `models/credit_default_model.joblib`
- `models/credit_app_state.json`
- `outputs/credit_risk_scores.csv`
- `outputs/credit_evaluation_report.txt`

The default credit configuration uses XGBoost with `device = "cuda"`. On a
machine without CUDA support, change `train/credit/config.py` to use CPU before
running a full training job.

### Churn

The churn module labels customers by comparing a reference month
with a target month, while also treating configured write-off statuses
as churn. It engineers customer activity and prime/account features, compares
multiple classifiers, optionally tunes selected models, and saves the best
model.

Important outputs:

- `models/churn_model.joblib`
- `models/churn_app_state.json`
- `outputs/churn_scores.csv`
- `outputs/churn_evaluation_report.txt`

### Recommendation

The recommendation module builds a customer-product matrix, RFM features,
merchant category spend features, foreign transaction features, and demographic
features. It trains one-vs-rest XGBoost product models and a content-based
filtering model, then supports customer lookup and batch scoring.

Important outputs:

- `data/engineered_recommendation_features/final_customer_profile.csv`
- `models/recommendation_xgb_models.pkl`
- `models/recommendation_xgb_meta.json`
- `models/recommendation_cbf_sim_matrix.pkl`
- `models/recommendation_cbf_meta.json`
- `outputs/recommendation_batch_predictions.csv`

## Setup with Poetry

Install Poetry if it is not already available, then install the project
dependencies from the repository root:

```powershell
poetry install
```

Run commands inside the Poetry environment with `poetry run`, for example:

```powershell
poetry run python main.py
```


## Common workflows

Generate the local synthetic data:

```powershell
poetry run python data/generate_fake_data.py
```

Start the full local application:

```powershell
poetry run python main.py
```

Then open:

```text
http://127.0.0.1:5050
```

Running only the credit pipeline using CLI:

```powershell
poetry run python train/credit/pipeline.py train
poetry run python train/credit/pipeline.py train --sample
poetry run python train/credit/pipeline.py tune --sample
poetry run python train/credit/pipeline.py score
```

Run only the churn pipeline using CLI:

```powershell
poetry run python train/churn/pipeline.py train
poetry run python train/churn/pipeline.py tune
```

Run only the recommendation pipeline:

```powershell
poetry run python train/recommendation/pipeline.py train
poetry run python train/recommendation/pipeline.py score --prime-dir data/prime_cleaned --txn-dir data/transaction_cleaned
```

## API overview

The UI calls the gateway routes below, and the gateway forwards them to the
corresponding module server.

| Module | Gateway prefix | Common routes |
| --- | --- | --- |
| Credit risk | `/credit/...` | `/api/status`, `/api/train`, `/api/score`, `/api/lookup`, `/api/customers`, `/api/report` |
| Churn | `/churn/...` | `/api/status`, `/api/train`, `/api/lookup`, `/api/customers`, `/api/report` |
| Recommendation | `/recommend/...` | `/api/status`, `/api/run_PREPROCESSING`, `/api/train`, `/api/train_cbf`, `/api/predict`, `/api/predict_cbf`, `/api/predict_batch`, `/api/download_batch` |

The gateway also exposes `/api/browse`, which opens a native directory picker
on Windows for selecting local data folders.

## Testing and quality checks

Run the full test suite:

```powershell
poetry run pytest tests
```

Format Python code with Black:

```powershell
poetry run black .
```

Check formatting without changing files:

```powershell
poetry run black --check .
```

Sort imports with isort:

```powershell
poetry run isort .
```

Check import order without changing files:

```powershell
poetry run isort --check-only .
```

Check Python code with Flake8:

```powershell
poetry run flake8 .
```

The tests use small deterministic data frames, temporary output files, and
monkeypatched model stages. They cover data cleaning, and important module-level features in the three modules
