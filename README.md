# CarInsight Pro — Full-stack AI Vehicle Valuation

A full-stack used-car valuation platform with a **Next.js frontend on Vercel** and a **Flask + XGBoost + SHAP API on Render**.

## Architecture

```text
Browser
  |
  v
Next.js dashboard (Vercel)
  |  NEXT_PUBLIC_API_BASE_URL
  v
Flask REST API (Render)
  |-- /health
  |-- /metadata
  |-- /predict
  |-- /batch-predict
  `-- /explain (SHAP)
       |
       v
XGBoost model bundle + preprocessing pipeline
```

The model bundle in `backend/artifacts/model_bundle.joblib` is the trained model generated from the supplied CarDekho dataset. The recorded held-out metrics are in `backend/artifacts/model_comparison.csv`. Training source is in `backend/train_model.py`.

## Repository structure

```text
CarInsight-Pro/
├── backend/
│   ├── api/index.py
│   ├── artifacts/model_bundle.joblib
│   ├── artifacts/model_comparison.csv
│   ├── data/cardekho_dataset.csv
│   ├── data/sample_input.csv
│   ├── train_model.py
│   ├── requirements.txt
│   └── render.yaml
├── frontend/
│   ├── app/page.tsx
│   ├── app/globals.css
│   ├── app/layout.tsx
│   ├── package.json
│   └── .env.example
└── README.md
```

## 1. Run the backend locally

Use Python 3.13.2 (to match the packaged environment):

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python -m api.index
```

API: `http://127.0.0.1:5000`

Health: `http://127.0.0.1:5000/health`

If retraining is needed, run `python train_model.py` from `backend/`. This recreates the model bundle and model comparison report.

## 2. Run the frontend locally

In another terminal:

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:5000
```

Then run:

```bash
npm run dev
```

Open `http://localhost:3000`.

## 3. Deploy Flask API to Render

This repository includes a root `render.yaml` blueprint. In Render, you can use **New > Blueprint** to apply it, or create a Web Service manually:

1. Create a new **Web Service** and connect this GitHub repository (or use Blueprint if you prefer the YAML).
2. Set **Root Directory** to `backend`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn api.index:app --workers 1 --threads 4 --timeout 120`
5. Add environment variable `CORS_ORIGINS` with the exact Vercel frontend origin, e.g. `https://carinsight-pro.vercel.app`.
6. Deploy and verify `https://YOUR-RENDER-SERVICE.onrender.com/health`.

The backend needs both `artifacts/model_bundle.joblib` and `data/cardekho_dataset.csv`, which are included in `backend/`.

## 4. Deploy Next.js frontend to Vercel

1. Import this GitHub repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Framework preset: **Next.js** (auto-detected).
4. Add Environment Variable:
   - Name: `NEXT_PUBLIC_API_BASE_URL`
   - Value: `https://YOUR-RENDER-SERVICE.onrender.com` (no trailing slash)
5. Deploy.
6. Copy the production Vercel domain and add it to Render's `CORS_ORIGINS` value, then redeploy the backend if necessary.

Vercel automatically detects Next.js and its build settings when the frontend directory is selected.

## API endpoints

- `GET /health` — service and model status
- `GET /metadata` — dropdown values and model metrics
- `POST /predict` — single prediction; body `{"record": {...}}`
- `POST /batch-predict` — bulk predictions; body `{"records": [{...}, {...}]}`
- `POST /explain` — SHAP feature impacts for one vehicle; body `{"record": {...}}`

Required vehicle fields:

`brand`, `model`, `seller_type`, `fuel_type`, `transmission_type`, `vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, `seats`.

## Model notes

The packaged model bundle selects the model with the best R² score on a fixed held-out test split. The SHAP endpoint uses TreeExplainer against the selected tree-based model and returns the top feature contributions for a single estimate. Positive contributions increase the estimate relative to the model baseline; negative contributions decrease it.

The current bundle's model comparison is stored in `backend/artifacts/model_comparison.csv`. Re-run training after changing the dataset, preprocessing, features or hyperparameters.

## Important deployment notes

- `NEXT_PUBLIC_API_BASE_URL` is public frontend configuration, not a secret.
- Set `CORS_ORIGINS` to your exact deployed frontend domain (include `https://`).
- Keep model artifacts versioned or in a managed artifact store. Never delete `model_bundle.joblib` from the backend deployment.
- Render free services may spin down when idle, so the first request can be slower.
- The valuation is an estimate, not a certified appraisal.
