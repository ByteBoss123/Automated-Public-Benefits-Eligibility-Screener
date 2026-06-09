# BenefitBridge

**Automated public benefits eligibility screener for underserved communities.**

BenefitBridge classifies free-text intake descriptions from applicants and predicts eligibility likelihood across five federal assistance programs: SNAP, Medicaid, Section 8 Housing, TANF, and LIHEAP. It replaces slow, manual intake screening with a production-ready ML pipeline that surfaces the right programs for each applicant in milliseconds.

---

## Problem

Millions of Americans eligible for public benefits never apply — not because they don't qualify, but because navigating the system is opaque and exhausting. Community organizations and social service agencies that handle intake are often overwhelmed, manually reviewing descriptions and cross-checking eligibility rules across multiple programs. This creates delays, missed referrals, and burned-out caseworkers.

BenefitBridge automates the initial screening step, giving caseworkers an instant eligibility signal so they can focus on high-value human work: follow-up, document collection, and advocacy.

---

## What It Does

- Accepts free-text intake descriptions (typed or transcribed)
- Extracts structured signals: income level, housing status, family composition, citizenship, utility burden, health coverage
- Classifies eligibility probability across 5 programs simultaneously
- Returns calibrated confidence scores with HIGH / MEDIUM / LOW indicators
- Flags uncertain cases for human review
- Serves predictions via a REST API with sub-100ms latency

---

## Model Performance (Test Set, n=400)

| Program | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| SNAP | 0.834 | 0.675 | 0.669 | 0.681 |
| Medicaid | 0.843 | 0.710 | 0.654 | 0.777 |
| Section 8 Housing | 0.779 | 0.195 | 0.364 | 0.133 |
| TANF | 0.899 | 0.471 | 0.606 | 0.385 |
| LIHEAP | 0.875 | 0.732 | 0.674 | 0.800 |
| **Macro** | **0.846** | **0.556** | | |

> Section 8 and TANF score lower due to class imbalance (6% and 10% prevalence). Thresholds are tuned per-program to maximize F1. In production, false negatives (missing eligible applicants) are treated as more costly than false positives — thresholds are set conservatively for high-prevalence programs.

---

## Architecture

```
Intake Text
     │
     ▼
┌────────────────────────────────┐
│       Feature Pipeline         │
│  ┌──────────────┐  ┌────────┐  │
│  │  TF-IDF      │  │ Rule-  │  │
│  │  (3000 feat) │  │ based  │  │
│  │  bigrams     │  │ extrac-│  │
│  │              │  │ tors   │  │
│  └──────────────┘  └────────┘  │
│         └──────┬───────┘        │
│                ▼                │
│         Feature Matrix          │
│         (3020 dimensions)       │
└────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│         Per-Program Classifiers            │
│                                            │
│  SNAP      │ GradientBoosting + Isotonic   │
│  Medicaid  │ calibration (CalibratedCV)    │
│  Section 8 │                               │
│  TANF      │ Thresholds tuned per-program  │
│  LIHEAP    │ to maximize F1                │
└────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────┐
│   FastAPI REST API     │
│   /screen endpoint     │
│   Confidence scores    │
│   + review flagging    │
└────────────────────────┘
```

**Feature extraction combines:**
- **TF-IDF with bigrams** (sublinear TF, min_df=3) on raw intake text
- **20 hand-crafted structured features** extracted via regex: unemployment signals, income estimates, housing type, family composition, citizenship markers, utility burden, health coverage, hardship language

**Model:** One `CalibratedClassifierCV(GradientBoostingClassifier, method='isotonic')` per program. Isotonic calibration ensures probability outputs reflect true likelihood rather than raw scores. Experiment tracking via MLflow.

---

## Project Structure

```
benefitbridge/
├── data/
│   ├── generate_dataset.py     # Synthetic intake data generator (2000 records)
│   └── intake_dataset.csv      # Generated dataset
├── src/
│   ├── features.py             # TF-IDF + structured feature extraction pipeline
│   ├── model.py                # Per-program ensemble training with MLflow tracking
│   └── evaluate.py             # Evaluation, error analysis, demo predictions
├── api/
│   └── app.py                  # FastAPI inference service
├── models/
│   ├── benefit_classifier.pkl  # Trained model artifact
│   ├── feature_pipeline.pkl    # Fitted feature pipeline
│   └── metrics.json            # Per-program evaluation metrics
├── tests/
│   └── test_model.py           # 14 unit tests (pipeline, model, features)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/ByteBoss123/BenefitBridge
cd benefitbridge
pip install -r requirements.txt

# Generate data and train
python data/generate_dataset.py
python src/model.py

# Run evaluation + demo predictions
python src/evaluate.py

# Run tests
pytest tests/ -v

# Start API
uvicorn api.app:app --reload
# → http://localhost:8000/docs
```

---

## API Usage

```bash
curl -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am currently unemployed and have been searching for work for 4 months. I am renting an apartment for $850 per month with my two children ages 5 and 9. I have no health insurance and my heating bills have been very high this winter. I am a US citizen.",
    "applicant_id": "APP-001"
  }'
```

**Response:**
```json
{
  "applicant_id": "APP-001",
  "eligible_programs": ["SNAP", "Medicaid", "LIHEAP", "TANF"],
  "total_programs_screened": 5,
  "results": [
    {
      "program": "SNAP",
      "full_name": "Supplemental Nutrition Assistance Program",
      "eligible": true,
      "confidence": 0.71,
      "confidence_level": "HIGH",
      "resource_url": "https://www.fns.usda.gov/snap"
    }
    ...
  ],
  "flagged_for_review": false,
  "processing_time_ms": 12.4,
  "disclaimer": "This tool provides a preliminary screening only..."
}
```

---

## Design Decisions

**Why per-program classifiers instead of a single multi-label model?**
Each program has different eligibility rules, class imbalance, and optimal decision thresholds. Per-program models allow independent threshold tuning, which is critical when false negatives (missing eligible applicants) carry a higher real-world cost than false positives.

**Why calibrated probabilities?**
Raw GBM scores are not well-calibrated — a score of 0.7 doesn't mean 70% likely eligible. Isotonic calibration ensures the confidence scores the API returns are meaningful and can be used for triage and review prioritization.

**Why combined TF-IDF + structured features?**
Pure TF-IDF misses signals like income level (extracted from dollar amounts) and specific patterns like "shutoff notice" that don't appear frequently enough to get high TF-IDF weight. Structured features capture these directly.

---

## Limitations & Future Work

- **Synthetic data:** The current dataset is generated, not from real intake records. Production deployment would require partnering with a social service organization to collect and annotate real intake descriptions.
- **Section 8 / TANF performance:** Low prevalence classes are hard to classify accurately. Collecting more positive examples or using oversampling (SMOTE) would improve recall.
- **Language:** Currently English-only. A production system serving underserved communities should support Spanish and other languages.
- **Fine-tuned BERT:** Replacing TF-IDF with a fine-tuned DistilBERT or similar transformer would significantly improve understanding of complex intake narratives.
- **Active learning:** Caseworker corrections could feed back into model retraining to improve accuracy over time.

---

## Stack

Python · Scikit-learn · GradientBoosting · FastAPI · MLflow · Pandas · NumPy · Pydantic · Pytest

---

*Built to demonstrate applied ML for social impact — the kind of problem where the model's output changes someone's access to food, housing, and healthcare.*
