# Immo Eliza — Property Price Prediction

This project is the final stage of the Immo Eliza project: after collecting real estate data through web scraping, analyzing it, and building a machine learning model to predict property prices, this last step focuses on deployment — exposing the trained model through a FastAPI backend and a Streamlit frontend, so both developers and non-technical users can get instant price estimates.

## 🔗 Live demo

- **Web app (Streamlit):** [immo-eliza-deployment-hivyq3a4ecgfmgi3uyfpe5.streamlit.app](https://immo-eliza-deployment-hivyq3a4ecgfmgi3uyfpe5.streamlit.app/)
- **API (FastAPI, on Render):** `https://immo-eliza-deployment-xvsh.onrender.com`
  - Interactive docs: `/docs`
  - Health check: `/`

## 📐 Architecture

```
User
  │
  ▼
Streamlit app  ──(address)──▶  Geoapify Geocoding API
  │                              (lat/lon + postal code)
  │
  ▼
FastAPI /predict  ──▶  preprocessing pipeline  ──▶  XGBoost model  ──▶  price estimate
  (Render, Docker)
```

The Streamlit app and the API are deployed **separately**:
- The **API** is containerized with Docker and deployed on **Render**.
- The **Streamlit app** is deployed on **Streamlit Community Cloud** and calls the API over HTTP.

## ✨ Features

- **Address-only input** — no manual region/province selection. The app geocodes the address via the Geoapify API and automatically derives the Belgian region and province from the postal code.
- **Simple property form** — type, subtype, living area, bedrooms, bathrooms, facades, building year, garden, condition, and EPC score.
- **Instant price estimate**, returned by a trained XGBoost regression model.
- Clean, custom-designed interface (no default Streamlit look).

## 🗂 Repository structure

```
immo-eliza-deployment/
├── api/
│   ├── app.py                  # FastAPI app (routes: /, /predict)
│   ├── predict.py              # preprocess() + predict() logic
│   ├── schemas.py               # Pydantic input/output models
│   ├── XGBoost_production.pkl   # trained model
│   ├── artifacts/                # preprocessing artifacts
│   │   ├── cleaning_stats.pkl
│   │   ├── ordinal_medians.pkl
│   │   ├── onehot_categories.pkl
│   │   └── feature_columns.pkl
│   ├── preprocessing/             # preprocessing pipeline modules
│   │   ├── cleaning.py
│   │   ├── ordinal_encoding.py
│   │   ├── feature_engineering.py
│   │   └── onehot_encoding.py
│   ├── Dockerfile
│   └── requirements.txt
├── streamlit/
│   └── web_app.py               # Streamlit frontend
├── .env                          # local secrets (not committed)
├── .gitignore
└── README.md
```



## 🧠 Model

The production model is an **XGBoost regressor**, trained on log-transformed property prices.

**Performance on the test set:**

| Metric | Value |
|---|---|
| R² | 0.8009 |
| MAE | €72,398 |
| RMSE | €116,939 |
| MAPE | 21.15% |

## 🔌 API reference

### `GET /`
Health check. Returns `"alive"` if the server is running.

### `POST /predict`
**Request body**:
```json
{
  "latitude": 50.8503,
  "longitude": 4.3517,
  "property_type": "House",
  "property_subtype": "residence",
  "region": "Brussels",
  "province": "Brussels Capital Region",
  "living_area_m2": 120,
  "bedrooms": 3,
  "bathrooms": 1,
  "facades": 2,
  "building_year": 1975,
  "garden_area_m2": 50,
  "has_garden": true,
  "state_of_the_building": "Normal",
  "epc_score": "C"
}
```

**Response**:
```json
{
  "prediction": 349500.0,
  "status_code": 200
}
```

On error, `prediction` is `null` and `status_code` is `400`.

## 🛠 Tech stack

- **Modeling:** Python, pandas, NumPy, scikit-learn, XGBoost, joblib
- **API:** FastAPI, Pydantic, Docker
- **Frontend:** Streamlit
- **Geocoding:** Geoapify Geocoding API
- **Deployment:** Render (API), Streamlit Community Cloud (frontend)

## 🚀 Running locally

### 1. Clone and set up environment variables
```bash
git clone <repo-url>
cd immo-eliza-deployment
```

Create a `.env` file at the project root:
```
URL_ML=http://localhost:8000
API_KEY_GEO=your_geoapify_api_key
```

### 2. Run the API
```bash
cd api
pip install -r requirements.txt
fastapi run app.py --port 8000
```

### 3. Run the Streamlit app
In a separate terminal, from the project root:
```bash
pip install -r streamlit/requirements.txt
streamlit run streamlit/web_app.py
```

The app will be available at `http://localhost:8501`.

## ☁️ Deployment notes

- The API's `Dockerfile` must copy the **entire** `api/` folder (model, artifacts, preprocessing modules) into the image — not just `app.py` — or the container will fail to start.
- Render auto-deploys on push if enabled; otherwise trigger a manual deploy from the dashboard.
- Streamlit Community Cloud auto-deploys on push to the connected branch.

## 👤 Author

**Victor Courtois**

- GitHub: https://github.com/VictorCourtois135
- LinkedIn: www.linkedin.com/in/victor-courtois-303690274

This project was completed in 5 days as part of the BeCode AI/Data Science bootcamp.