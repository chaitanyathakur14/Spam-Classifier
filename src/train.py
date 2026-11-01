"""Train several models, track them with MLflow and save the best one."""
import json
from datetime import datetime
from pathlib import Path
 
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
 
from src.etl import FEATURE_DIR, run_etl
 
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
EXPERIMENT = "SMS_Spam_MLOps_Capstone"
 
CANDIDATES = {
    "NaiveBayes_a1.0": lambda: MultinomialNB(alpha=1.0),
    "NaiveBayes_a0.3": lambda: MultinomialNB(alpha=0.3),
    "LinearSVM_C1.0": lambda: CalibratedClassifierCV(LinearSVC(C=1.0), cv=3),
}
 
 
def build_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("clf", clf),
    ])
 
 
def load_features():
    train_p, test_p = FEATURE_DIR / "train_features.csv", FEATURE_DIR / "test_features.csv"
    if not (train_p.exists() and test_p.exists()):
        return run_etl()
    return pd.read_csv(train_p), pd.read_csv(test_p)
 
 
def evaluate(model, test):
    pred = model.predict(test["text_clean"])
    y = test["target"]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }, confusion_matrix(y, pred).tolist()
 
 
def train_all(log_to_mlflow: bool = True):
    train, test = load_features()
    train = train.dropna(subset=["text_clean"])
    test = test.dropna(subset=["text_clean"])
    mlflow = None
    if log_to_mlflow:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(EXPERIMENT)
 
    results = []
    for name, make_clf in CANDIDATES.items():
        model = build_pipeline(make_clf()).fit(train["text_clean"], train["target"])
        metrics, cm = evaluate(model, test)
        run_id = None
        if mlflow:
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_param("model", name)
                mlflow.log_param("max_features", 3000)
                mlflow.log_param("ngram_range", "(1, 2)")
                mlflow.log_param("train_rows", len(train))
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
                run_id = run.info.run_id
        print(f"{name:18s} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
        results.append({"name": name, "model": model, "metrics": metrics,
                        "confusion_matrix": cm, "run_id": run_id})
 
    best = max(results, key=lambda r: r["metrics"]["f1"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "best_model": best["name"],
        "metrics": best["metrics"],
        "confusion_matrix": best["confusion_matrix"],
        "train_rows": len(train),
        "test_rows": len(test),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mlflow_run_id": best["run_id"],
        "all_runs": {r["name"]: r["metrics"] for r in results},
    }, indent=2))
    print(f"\nBest model: {best['name']}  ->  saved to {MODEL_PATH}")
    return best
 
 
if __name__ == "__main__":
    train_all()
