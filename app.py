"""
BenefitBridge — FastAPI Inference Service
Production-ready REST API for benefits eligibility screening.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import pandas as pd
import numpy as np
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benefitbridge")

app = FastAPI(
    title="BenefitBridge API",
    description="Automated public benefits eligibility screener. Classifies intake text across SNAP, Medicaid, Section 8 Housing, TANF, and LIHEAP programs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROGRAMS = ["SNAP", "Medicaid", "Section8_Housing", "TANF", "LIHEAP"]

PROGRAM_INFO = {
    "SNAP": {
        "full_name": "Supplemental Nutrition Assistance Program",
        "description": "Monthly food benefits for low-income individuals and families",
        "url": "https://www.fns.usda.gov/snap",
    },
    "Medicaid": {
        "full_name": "Medicaid",
        "description": "Free or low-cost health coverage for eligible adults, children, and families",
        "url": "https://www.medicaid.gov",
    },
    "Section8_Housing": {
        "full_name": "Section 8 Housing Choice Voucher",
        "description": "Rental assistance for very low-income families",
        "url": "https://www.hud.gov/topics/housing_choice_voucher_program_section_8",
    },
    "TANF": {
        "full_name": "Temporary Assistance for Needy Families",
        "description": "Cash assistance and support services for families with children",
        "url": "https://www.acf.hhs.gov/ofa/programs/tanf",
    },
    "LIHEAP": {
        "full_name": "Low Income Home Energy Assistance Program",
        "description": "Help paying heating and cooling bills for low-income households",
        "url": "https://www.acf.hhs.gov/ocs/low-income-home-energy-assistance-program-liheap",
    },
}

# ─── Globals ──────────────────────────────────────────────────────────────────
_model = None
_pipeline = None


def load_models():
    global _model, _pipeline
    logger.info("Loading model artifacts...")
    with open("models/benefit_classifier.pkl", "rb") as f:
        _model = pickle.load(f)
    with open("models/feature_pipeline.pkl", "rb") as f:
        _pipeline = pickle.load(f)
    logger.info("Models loaded successfully.")


@app.on_event("startup")
def startup():
    load_models()


# ─── Schemas ──────────────────────────────────────────────────────────────────

class ScreeningRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        max_length=5000,
        description="Free-text intake description from the applicant",
        example="I am currently unemployed and have been searching for work for 4 months. I am renting an apartment for $850 per month with my two children ages 5 and 9. I have no health insurance and my heating bills have been very high this winter."
    )
    applicant_id: Optional[str] = Field(None, description="Optional applicant identifier for tracking")


class ProgramResult(BaseModel):
    program: str
    full_name: str
    eligible: bool
    confidence: float
    confidence_level: str  # HIGH / MEDIUM / LOW
    resource_url: str


class ScreeningResponse(BaseModel):
    applicant_id: Optional[str]
    eligible_programs: list[str]
    total_programs_screened: int
    results: list[ProgramResult]
    flagged_for_review: bool
    processing_time_ms: float
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    programs: list[str]
    version: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        programs=PROGRAMS,
        version="1.0.0",
    )


@app.post("/screen", response_model=ScreeningResponse)
def screen(request: ScreeningRequest):
    if _model is None or _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")

    start = time.time()

    texts = pd.Series([request.text])
    X = _pipeline.transform(texts)
    proba_dict = _model.predict_proba(X)
    preds_df = _model.predict(X)

    results = []
    eligible_programs = []
    low_confidence_count = 0

    for program in PROGRAMS:
        prob = float(proba_dict[program][0])
        pred = bool(preds_df[program].iloc[0])
        info = PROGRAM_INFO[program]

        if prob >= 0.7:
            conf_level = "HIGH"
        elif prob >= 0.4:
            conf_level = "MEDIUM"
            low_confidence_count += 1
        else:
            conf_level = "LOW"

        if pred:
            eligible_programs.append(program)

        results.append(ProgramResult(
            program=program,
            full_name=info["full_name"],
            eligible=pred,
            confidence=round(prob, 3),
            confidence_level=conf_level,
            resource_url=info["url"],
        ))

    # Sort: eligible first, then by confidence descending
    results.sort(key=lambda r: (-int(r.eligible), -r.confidence))

    processing_ms = round((time.time() - start) * 1000, 2)

    return ScreeningResponse(
        applicant_id=request.applicant_id,
        eligible_programs=eligible_programs,
        total_programs_screened=len(PROGRAMS),
        results=results,
        flagged_for_review=low_confidence_count >= 2,
        processing_time_ms=processing_ms,
        disclaimer=(
            "This tool provides a preliminary screening only and does not constitute "
            "a formal eligibility determination. Applicants should contact their local "
            "benefits office to verify eligibility and apply. Results are based on "
            "machine learning and may not capture all individual circumstances."
        ),
    )


@app.get("/programs")
def list_programs():
    return {"programs": PROGRAM_INFO}


@app.get("/")
def root():
    return {
        "project": "BenefitBridge",
        "description": "Automated public benefits eligibility screener for underserved communities",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
