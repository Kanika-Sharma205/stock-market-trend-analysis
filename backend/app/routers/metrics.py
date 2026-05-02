import os, re, csv
from fastapi import APIRouter
from app.services import dl_service

router = APIRouter(prefix="/metrics", tags=["metrics"])

ML_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "ml")
)


def _parse_report(path: str) -> dict:
    """Parse sklearn classification_report text into structured data."""
    with open(path) as f:
        text = f.read()

    lines = [l for l in text.strip().splitlines() if l.strip()]
    # First line is "Model: <name>"
    model_name = lines[0].replace("Model:", "").strip()

    rows = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 5 and parts[0] not in ("macro", "weighted"):
            label = parts[0]
            rows[label] = {
                "precision": float(parts[1]),
                "recall":    float(parts[2]),
                "f1":        float(parts[3]),
                "support":   int(parts[4]),
            }
        elif parts[0] == "accuracy":
            rows["accuracy"] = float(parts[1])

    return {"model": model_name, "classes": rows}


def _load_importance(path: str) -> list[dict]:
    result = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.append({"feature": row["feature"], "importance": float(row["importance"])})
    return sorted(result, key=lambda x: x["importance"], reverse=True)[:15]


@router.get("/ml")
def get_ml_metrics():
    models = []
    specs = [
        ("Random Forest", "rf_report.txt",  "rf_importance.csv",  0.4921),
        ("XGBoost",       "xgb_report.txt", "xgb_importance.csv", 0.4833),
        ("LightGBM",      "lgb_report.txt", "lgb_importance.csv", 0.4793),
    ]
    for name, report_file, imp_file, auc in specs:
        report = _parse_report(os.path.join(ML_DIR, report_file))
        importance = _load_importance(os.path.join(ML_DIR, imp_file))
        classes = report["classes"]
        models.append({
            "name": name,
            "auc_roc": auc,
            "accuracy": classes.get("accuracy", 0.0),
            "up_precision":   classes.get("UP",   {}).get("precision", 0),
            "up_recall":      classes.get("UP",   {}).get("recall",    0),
            "up_f1":          classes.get("UP",   {}).get("f1",        0),
            "down_precision": classes.get("DOWN", {}).get("precision", 0),
            "down_recall":    classes.get("DOWN", {}).get("recall",    0),
            "down_f1":        classes.get("DOWN", {}).get("f1",        0),
            "feature_importance": importance,
        })
    return {"models": models}


@router.get("/dl")
def get_dl_metrics():
    results = dl_service.get_final_results()
    history = dl_service.get_training_history()
    importance = dl_service.get_feature_importance()

    meta = results.get("preprocessing_meta", {})

    return {
        "test_auc_roc":          round(results["test_auc_roc"], 4),
        "test_accuracy_default": round(results["test_accuracy_default"], 4),
        "test_accuracy_optimal": round(results["test_accuracy_optimal"], 4),
        "test_f1_optimal":       round(results["test_f1_optimal"], 4),
        "final_threshold":       round(results["final_threshold"], 4),
        "val_auc_roc":           round(results["val_auc_roc"], 4),
        "best_epoch":            results["best_epoch"],
        "top3_features":         results["top3_features"],
        "architecture": {
            "window_size":   meta.get("window_size", 30),
            "n_features":    meta.get("n_features", 16),
            "feature_cols":  meta.get("feature_cols", []),
            "train_samples": meta.get("X_train_shape", [0])[0] if meta.get("X_train_shape") else 0,
            "val_samples":   meta.get("X_val_shape",   [0])[0] if meta.get("X_val_shape")   else 0,
            "test_samples":  meta.get("X_test_shape",  [0])[0] if meta.get("X_test_shape")  else 0,
            "train_range":   list(meta.get("train_date_range", [])),
            "val_range":     list(meta.get("val_date_range",   [])),
            "test_range":    list(meta.get("test_date_range",  [])),
        },
        "training_history": history,
        "feature_importance": importance,
    }
