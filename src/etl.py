"""ETL pipeline: Extract raw SMS data, Transform it, Load features into the feature store."""
from pathlib import Path
 
import pandas as pd
from sklearn.model_selection import train_test_split
 
from src.preprocess import clean_text
 
RAW_PATH = Path("datasets/spam.csv")
FEATURE_DIR = Path("feature_store")
 
 
def extract(path: Path = RAW_PATH) -> pd.DataFrame:
    """E - read the raw CSV (latin-1 encoded, columns v1 = label, v2 = text)."""
    df = pd.read_csv(path, encoding="latin-1", usecols=["v1", "v2"])
    df.columns = ["label", "text"]
    return df
 
 
def validate(df: pd.DataFrame) -> None:
    assert set(df["label"].unique()) == {"ham", "spam"}, "Unexpected labels"
    assert df["text"].notna().all(), "Null messages found"
 
 
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """T - clean text, remove duplicates and engineer simple numeric features."""
    df = df.drop_duplicates().reset_index(drop=True)
    out = pd.DataFrame()
    out["text_clean"] = df["text"].apply(clean_text)
    out["char_len"] = df["text"].str.len()
    out["word_count"] = df["text"].str.split().str.len()
    out["digit_count"] = df["text"].str.count(r"\d")
    out["exclaim_count"] = df["text"].str.count("!")
    out["target"] = (df["label"] == "spam").astype(int)
    return out[out["text_clean"].str.len() > 0].reset_index(drop=True)
 
 
def load(features: pd.DataFrame, out_dir: Path = FEATURE_DIR):
    """L - split and write reusable features (our simple feature store)."""
    out_dir.mkdir(exist_ok=True)
    train, test = train_test_split(
        features, test_size=0.2, random_state=42, stratify=features["target"]
    )
    train.to_csv(out_dir / "train_features.csv", index=False)
    test.to_csv(out_dir / "test_features.csv", index=False)
    return train, test
 
 
def run_etl():
    raw = extract()
    print("Raw data shape:", raw.shape)
    validate(raw)
    feats = transform(raw)
    print("After cleaning:", feats.shape)
    train, test = load(feats)
    print("Train rows:", len(train), "| Test rows:", len(test))
    print("Features stored in:", FEATURE_DIR.resolve())
    return train, test
 
 
if __name__ == "__main__":
    run_etl()
