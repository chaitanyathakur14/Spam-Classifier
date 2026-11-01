import pandas as pd
import pytest
 
from src.etl import extract, transform, validate
from src.predict import predict_message
from src.preprocess import clean_text
from src.train import CANDIDATES, build_pipeline, evaluate
 
 
def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("WIN a FREE Prize!!!") == "win free prize"
 
 
def test_clean_text_masks_urls():
    assert "urltoken" in clean_text("Visit http://spam.example.com now")
 
 
def test_etl_produces_valid_features():
    raw = extract()
    validate(raw)
    feats = transform(raw)
    assert {"text_clean", "char_len", "target"} <= set(feats.columns)
    assert feats["target"].isin([0, 1]).all()
    assert len(feats) > 4000
 
 
@pytest.fixture(scope="module")
def trained_model():
    feats = transform(extract())
    split = int(len(feats) * 0.8)
    train, test = feats.iloc[:split], feats.iloc[split:]
    model = build_pipeline(CANDIDATES["NaiveBayes_a0.3"]()).fit(
        train["text_clean"], train["target"])
    return model, test
 
 
def test_model_accuracy_threshold(trained_model):
    model, test = trained_model
    metrics, _ = evaluate(model, test)
    assert metrics["accuracy"] >= 0.95
 
 
def test_spam_and_ham_examples(trained_model):
    model, _ = trained_model
    spam = predict_message(model, "WINNER!! Claim your free prize now, call 09061701461")
    ham = predict_message(model, "Ok, see you at lunch tomorrow")
    assert spam["label"] == "Spam"
    assert ham["label"] == "Not Spam"
