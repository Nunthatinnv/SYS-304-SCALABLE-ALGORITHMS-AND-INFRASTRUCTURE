# Backend — House Price Prediction API

FastAPI service that wraps the Milestone 1 XGBoost baseline and exposes it over
HTTP.

## What it does

The Milestone 1 notebook saved a complete scikit-learn `Pipeline`
(`ColumnTransformer` + `XGBRegressor`) to `models/xgb_baseline_pipeline.pkl`. It
was fitted on a 1460-row DataFrame with **79 raw columns** and the target
`log1p(SalePrice)`.

Asking a user to fill in 79 fields is not realistic, so the service takes **10
high-signal fields**, widens that partial input to the full 79-column row using
training-set defaults, runs the pipeline, and inverts the log transform.

```
JSON (10 fields)
  -> HouseFeatures        validate types, ranges, category membership
  -> build_row()          widen to 79 columns using defaults.json
  -> Pipeline.predict()   preprocessing + XGBoost, returns log1p(price)
  -> expm1()              back to dollars
  -> JSON response
```

## Module map

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, routes, CORS, startup model load |
| `app/schemas.py` | Pydantic request/response models and validation rules |
| `app/model.py` | Model loading, row widening, inference |
| `app/reference.py` | Typed accessor over `defaults.json` |
| `app/config.py` | Environment-driven configuration |
| `app/defaults.json` | Frozen column order, defaults, categories, ranges |

## `defaults.json`

Generated once from `data/train.csv`:

- `columns` — the 79 columns in training order (`Id` and `SalePrice` removed).
- `numeric_columns` (36) / `categorical_columns` (43) — the exact split
  `select_dtypes` produced during training, so serving-time dtypes match.
- `defaults` — median for numeric columns, mode for categorical columns.
- `categories` — observed values per categorical column, used to validate
  `Neighborhood` and to populate the frontend dropdown.
- `numeric_ranges` — observed min/max, sent to the frontend as input hints.

Freezing this into JSON keeps `data/train.csv` out of the container image.

## Endpoints

### `GET /health`

Liveness plus whether the pipeline is in memory. Used by the Docker
`HEALTHCHECK`, by `docker-compose` `depends_on`, and by `deploy.sh`.

```json
{"status": "ok", "service": "house-price-api", "version": "0.2.0", "model_loaded": true}
```

### `GET /schema`

Describes the input form (name, label, type, default, bounds, options) so the
frontend never hard-codes field names. The backend stays the single source of
truth.

### `POST /predict`

Request:

```json
{
  "OverallQual": 7,
  "GrLivArea": 1710,
  "TotalBsmtSF": 856,
  "1stFlrSF": 856,
  "YearBuilt": 2003,
  "GarageCars": 2,
  "FullBath": 2,
  "TotRmsAbvGrd": 8,
  "LotArea": 8450,
  "Neighborhood": "CollgCr"
}
```

Response:

```json
{"sale_price": 178938.64, "log_price": 12.0948, "currency": "USD"}
```

Status codes: `200` on success, `422` on validation failure (bad range, unknown
`Neighborhood`, missing or extra field), `503` if the model artefact is missing.

Interactive docs are served at `/docs` (Swagger) and `/redoc`.

## Design notes

- **The model is loaded once**, in the FastAPI `lifespan` handler, not per
  request. Unpickling costs far more than inference.
- **`1stFlrSF` is not a valid Python identifier**, so the Pydantic field is
  named `FirstFlrSF` with `alias="1stFlrSF"`. `to_raw_columns()` dumps
  `by_alias=True` so the pipeline sees the original Kaggle name.
- **`extra="forbid"`** on the request model. A typo must fail loudly rather than
  silently fall back to a default.
- **Dependency versions are pinned exactly** to those that produced the pickle
  (scikit-learn 1.9.0, xgboost 3.4.1, numpy 2.5.2). Loading a pickle with a
  different scikit-learn version can warn, break, or change behaviour.
- **`libgomp1`** is installed in the image because the XGBoost wheel needs
  OpenMP at runtime.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `<repo>/models/xgb_baseline_pipeline.pkl` | Pipeline artefact |
| `DEFAULTS_PATH` | `backend/app/defaults.json` | Reference metadata |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## Running locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn backend.app.main:app --reload --port 8000
```

Run from the repository root: `backend.app.main` is an absolute package path.

## Tests

```bash
pytest backend/tests
```

- `test_reference.py` — column counts, disjoint numeric/categorical split, every
  column has a default, defaults are valid categories.
- `test_schemas.py` — alias handling, range rejection, unknown neighborhood,
  missing and extra fields.
- `test_model.py` — row widening, unknown-column rejection, plausible and
  deterministic predictions, monotonicity in `OverallQual`.
- `test_api.py` — integration tests through the real app: `/health`, `/schema`,
  `/predict` 200 and 422 paths, and the OpenAPI document.

## Regenerating `defaults.json`

Only needed if the training data changes:

```bash
python backend/scripts/generate_defaults.py
```
