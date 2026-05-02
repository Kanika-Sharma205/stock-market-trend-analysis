# How to Run the Project

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| pip | latest |

---

## Quickstart — One Command

**Linux / macOS:**
```bash
chmod +x run-linux.sh
./run-linux.sh
```

**Windows:**
```
run-windows.bat
```

The script will:
1. Find your virtual environment (`backend/venv` → `./venv` → `~/ml_env`) or **create one automatically**
2. Install all Python dependencies from `backend/requirements.txt`
3. Install frontend npm packages (only on first run)
4. Start backend on **http://localhost:8000**
5. Start frontend on **http://localhost:5173**

Press `Ctrl+C` once to cleanly shut down both servers (Linux). On Windows, two separate console windows are opened — close them individually.

---

## Running Separately

**Backend only:**
```bash
cd backend
./run-linux.sh        # Linux / macOS
run-windows.bat       # Windows
```

**Frontend only:**
```bash
cd frontend
./run-linux.sh        # Linux / macOS
run-windows.bat       # Windows
```

---

## Manual Setup

If you prefer full control over each step:

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# 2. Install backend dependencies
pip install -r backend/requirements.txt

# 3. Start the backend
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. In a new terminal — install and start frontend
cd frontend
npm install
npm run dev
```

**API documentation** is available at http://localhost:8000/docs once the backend is running.

---

## Running Training Notebooks

The research notebooks are in `training/`. Run them in order:

```bash
# Install Jupyter if needed
pip install jupyter

# Chapter 0 — Extract features from raw OHLCV
jupyter notebook training/chapter0_feature_extraction/Feature_Extraction.ipynb

# Chapter 1A — ML data preparation and EDA
jupyter notebook training/chapter1_ml_models/Data_Preparation_ML.ipynb

# Chapter 1B — Train Random Forest, XGBoost, LightGBM
jupyter notebook training/chapter1_ml_models/Training_ML_Models.ipynb

# Chapter 2A — DL feature engineering and windowing
jupyter notebook training/chapter2_dl_models/Data_Preparation_DL.ipynb

# Chapter 2B — Build and train CNN + LSTM + Attention
jupyter notebook training/chapter2_dl_models/Model_Architecture_Training.ipynb

# Chapter 2C — Evaluation, threshold tuning, saliency analysis
jupyter notebook training/chapter2_dl_models/Evaluation_Analysis.ipynb
```

Notebooks must be run in chapter order — each chapter depends on output files from the previous one.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: tensorflow` | Ensure the venv is activated before running |
| `Port 8000 already in use` | `lsof -ti:8000 \| xargs kill` (Linux) or `netstat -ano \| findstr :8000` then kill in Task Manager (Windows) |
| `npm: command not found` | Install Node.js 18+ from https://nodejs.org |
| Frontend can't reach backend | Confirm backend is running and CORS is not blocked; check http://localhost:8000/docs |
| Keras model load error (`TemporalAttention`) | Ensure `dl_service.py` `_register_custom_layers()` is called before `load_model()` — it already is in the current code |
