"""
BenefitBridge — Test Suite
"""

import pytest
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.insert(0, '.')

PROGRAMS = ["SNAP", "Medicaid", "Section8_Housing", "TANF", "LIHEAP"]


@pytest.fixture(scope="module")
def artifacts():
    with open("models/benefit_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return model, pipeline


def make_text(employment="unemployed", housing="renting", children=True,
              income=500, citizenship="citizen", utilities=True):
    parts = []
    if employment == "unemployed":
        parts.append("I am currently unemployed and have been looking for work for 6 months.")
    else:
        parts.append(f"I work part-time earning about ${income} per month.")
    if housing == "renting":
        parts.append("I am renting an apartment for $850 per month.")
    elif housing == "unhoused":
        parts.append("I have been unhoused for 3 months and am staying in a shelter.")
    if children:
        parts.append("I have 2 children ages 4 and 8.")
    if utilities:
        parts.append("My electricity bill has been very high this winter, around $200 per month.")
    parts.append("I have no health insurance.")
    if citizenship == "citizen":
        parts.append("I am a US citizen.")
    return " ".join(parts)


# ─── Feature Pipeline Tests ───────────────────────────────────────────────────

class TestFeaturePipeline:
    def test_transform_returns_array(self, artifacts):
        _, pipeline = artifacts
        texts = pd.Series([make_text(), make_text(employment="employed")])
        X = pipeline.transform(texts)
        assert X.shape[0] == 2
        assert X.shape[1] > 20

    def test_transform_consistent_shape(self, artifacts):
        _, pipeline = artifacts
        texts1 = pd.Series([make_text()])
        texts2 = pd.Series([make_text(), make_text(housing="unhoused")])
        X1 = pipeline.transform(texts1)
        X2 = pipeline.transform(texts2)
        assert X1.shape[1] == X2.shape[1]

    def test_no_nan_in_features(self, artifacts):
        _, pipeline = artifacts
        texts = pd.Series([make_text(), make_text(employment="employed", housing="unhoused")])
        X = pipeline.transform(texts)
        assert not np.isnan(X).any()


# ─── Model Tests ──────────────────────────────────────────────────────────────

class TestModel:
    def test_predict_proba_shape(self, artifacts):
        model, pipeline = artifacts
        X = pipeline.transform(pd.Series([make_text()]))
        proba = model.predict_proba(X)
        assert set(proba.keys()) == set(PROGRAMS)
        for p in PROGRAMS:
            assert len(proba[p]) == 1
            assert 0.0 <= proba[p][0] <= 1.0

    def test_predict_binary_output(self, artifacts):
        model, pipeline = artifacts
        texts = pd.Series([make_text(), make_text(employment="employed", income=4000)])
        X = pipeline.transform(texts)
        preds = model.predict(X)
        assert set(preds.columns) == set(PROGRAMS)
        for p in PROGRAMS:
            assert set(preds[p].unique()).issubset({0, 1})

    def test_high_income_predicts_ineligible(self, artifacts):
        model, pipeline = artifacts
        text = "I work full-time earning about $5000 per month. I own my home. I live alone. I have employer health insurance. I am a US citizen."
        X = pipeline.transform(pd.Series([text]))
        proba = model.predict_proba(X)
        for p in ["SNAP", "Medicaid", "LIHEAP"]:
            assert proba[p][0] < 0.5, f"{p} should be low probability for high-income applicant"

    def test_unemployed_unhoused_predicts_eligible_snap(self, artifacts):
        model, pipeline = artifacts
        text = make_text(employment="unemployed", housing="unhoused", income=0, children=False)
        X = pipeline.transform(pd.Series([text]))
        proba = model.predict_proba(X)
        assert proba["SNAP"][0] > 0.3, "Unhoused unemployed person should have some SNAP probability"

    def test_undocumented_predicts_ineligible(self, artifacts):
        model, pipeline = artifacts
        text = make_text() + " I do not have legal immigration status."
        X = pipeline.transform(pd.Series([text]))
        proba = model.predict_proba(X)
        assert proba["TANF"][0] < 0.5, "Undocumented applicant should score low for TANF"

    def test_thresholds_in_valid_range(self, artifacts):
        model, _ = artifacts
        for p in PROGRAMS:
            assert 0.1 <= model.thresholds[p] <= 0.9

    def test_batch_prediction_consistent(self, artifacts):
        model, pipeline = artifacts
        text = make_text()
        X_single = pipeline.transform(pd.Series([text]))
        X_batch = pipeline.transform(pd.Series([text, text]))
        p_single = model.predict_proba(X_single)
        p_batch = model.predict_proba(X_batch)
        for prog in PROGRAMS:
            assert abs(p_single[prog][0] - p_batch[prog][0]) < 1e-6
            assert abs(p_single[prog][0] - p_batch[prog][1]) < 1e-6


# ─── Feature Extractor Unit Tests ─────────────────────────────────────────────

class TestStructuredFeatures:
    def test_income_signal_extracted(self):
        from src.features import extract_income_signal
        assert extract_income_signal("I earn $800 per month") > 0
        assert extract_income_signal("no income mentioned") == 0.5

    def test_structured_features_shape(self):
        from src.features import extract_structured_features
        texts = pd.Series([make_text(), make_text(employment="employed")])
        feats = extract_structured_features(texts)
        assert feats.shape[0] == 2
        assert feats.shape[1] == 20

    def test_unemployed_flag(self):
        from src.features import extract_structured_features
        employed = pd.Series(["I work full time and earn $3000 per month."])
        unemployed = pd.Series(["I am currently unemployed and have been looking for work."])
        fe = extract_structured_features(employed)
        fu = extract_structured_features(unemployed)
        assert fu[0][0] == 1  # unemployed flag
        assert fe[0][0] == 0

    def test_utility_flag(self):
        from src.features import extract_structured_features
        with_util = pd.Series(["My electricity bill is very high this month."])
        without_util = pd.Series(["I need help with food and rent."])
        fw = extract_structured_features(with_util)
        fwo = extract_structured_features(without_util)
        assert fw[0][13] == 1  # has_utilities flag
        assert fwo[0][13] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
