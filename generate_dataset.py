"""
BenefitBridge — Synthetic Dataset Generator
Generates realistic intake form responses for public benefits eligibility classification.
Programs: SNAP, Medicaid, Section8_Housing, TANF, LIHEAP
"""

import pandas as pd
import numpy as np
import json
import random
import os

random.seed(42)
np.random.seed(42)

PROGRAMS = ["SNAP", "Medicaid", "Section8_Housing", "TANF", "LIHEAP"]

# Eligibility rules (simplified federal guidelines)
ELIGIBILITY_RULES = {
    "SNAP": lambda r: r["household_income_pct_fpl"] <= 130 and r["citizenship_status"] in ["citizen", "qualified_alien"],
    "Medicaid": lambda r: r["household_income_pct_fpl"] <= 138 and r["citizenship_status"] in ["citizen", "qualified_alien"],
    "Section8_Housing": lambda r: r["household_income_pct_fpl"] <= 50 and r["housing_status"] in ["renting", "unhoused"],
    "TANF": lambda r: r["household_income_pct_fpl"] <= 100 and r["has_dependent_children"] and r["citizenship_status"] == "citizen",
    "LIHEAP": lambda r: r["household_income_pct_fpl"] <= 150 and r["has_utility_bills"],
}

EMPLOYMENT_TEMPLATES = [
    "I am currently unemployed and have been looking for work for {months} months.",
    "I work part-time at a {job} earning about ${income} per month.",
    "I recently lost my job at {company} due to layoffs.",
    "I am self-employed doing {gig} work, my income varies month to month.",
    "I receive disability benefits of ${income} per month and cannot work.",
    "I am a full-time caregiver for my {relation} and have no outside income.",
    "I work {hours} hours a week at {job}, making ${income} monthly.",
    "I am a student and work part-time, earning around ${income} a month.",
]

HOUSING_TEMPLATES = [
    "I am currently renting an apartment for ${rent} per month.",
    "I am staying with family temporarily as I cannot afford my own place.",
    "I have been unhoused for {months} months and am staying in a shelter.",
    "I own my home but am struggling to keep up with the mortgage.",
    "I am renting a room in a shared house for ${rent} per month.",
    "I recently became homeless after my landlord evicted me.",
]

FAMILY_TEMPLATES = [
    "I live alone.",
    "I have {n} children ages {ages}.",
    "I am a single parent with {n} kids.",
    "I live with my spouse and {n} children.",
    "I am caring for my elderly {relation} who lives with me.",
    "My household includes myself, my partner, and our {n} children.",
]

HEALTH_TEMPLATES = [
    "I have no health insurance.",
    "I recently lost my employer-sponsored health insurance.",
    "I have a chronic condition ({condition}) that requires regular medication.",
    "My child has {condition} and needs ongoing medical care.",
    "I am pregnant and currently uninsured.",
    "I have been managing {condition} without proper medical care due to cost.",
]

UTILITY_TEMPLATES = [
    "My electricity bill has been very high this winter, around ${amt} per month.",
    "I am behind on my heating bills by ${amt}.",
    "I received a shutoff notice for my gas service.",
    "I struggle every month to pay my utility bills on time.",
    "My utility bills take up a large portion of my income.",
]

def random_record():
    income_pct = random.choice([
        random.randint(20, 80),    # very low
        random.randint(81, 130),   # low
        random.randint(131, 200),  # moderate
        random.randint(201, 400),  # above threshold
    ])
    housing = random.choice(["renting", "unhoused", "owned", "staying_with_family"])
    citizenship = random.choice(["citizen", "citizen", "citizen", "qualified_alien", "undocumented"])
    has_children = random.random() > 0.45
    has_utilities = random.random() > 0.3

    record = {
        "household_income_pct_fpl": income_pct,
        "housing_status": housing,
        "citizenship_status": citizenship,
        "has_dependent_children": has_children,
        "has_utility_bills": has_utilities,
    }

    labels = {p: int(ELIGIBILITY_RULES[p](record)) for p in PROGRAMS}

    # Build narrative intake text
    monthly_income = int(income_pct * 1200 / 100)
    employment = random.choice(EMPLOYMENT_TEMPLATES).format(
        months=random.randint(1, 18),
        job=random.choice(["grocery store", "warehouse", "restaurant", "retail shop", "cleaning service"]),
        income=monthly_income,
        company=random.choice(["Amazon", "Target", "a local restaurant", "a small business"]),
        gig=random.choice(["rideshare", "food delivery", "freelance cleaning", "handyman"]),
        hours=random.choice([10, 15, 20, 25]),
        relation=random.choice(["mother", "father", "spouse"]),
    )

    housing_text = random.choice(HOUSING_TEMPLATES).format(
        rent=random.randint(600, 1800),
        months=random.randint(1, 12),
    )

    family_text = random.choice(FAMILY_TEMPLATES).format(
        n=random.randint(1, 4),
        ages=", ".join([str(random.randint(1, 16)) for _ in range(random.randint(1, 3))]),
        relation=random.choice(["mother", "father", "grandmother"]),
    ) if has_children else random.choice(["I live alone.", "I live with my partner.", "I share an apartment with a roommate."])

    health_text = random.choice(HEALTH_TEMPLATES).format(
        condition=random.choice(["diabetes", "asthma", "hypertension", "depression", "arthritis"]),
    )

    utility_text = random.choice(UTILITY_TEMPLATES).format(
        amt=random.randint(150, 600),
    ) if has_utilities else ""

    parts = [employment, housing_text, family_text, health_text]
    if utility_text:
        parts.append(utility_text)

    if citizenship == "qualified_alien":
        parts.append("I am a legal permanent resident and have been in the US for over 5 years.")
    elif citizenship == "undocumented":
        parts.append("I do not have legal immigration status.")

    text = " ".join(parts)

    row = {"text": text, **labels, **{k: v for k, v in record.items()}}
    return row


def generate(n=2000):
    records = [random_record() for _ in range(n)]
    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/intake_dataset.csv", index=False)
    print(f"Generated {n} records.")
    print(df[PROGRAMS].mean().round(3).to_string())
    return df


if __name__ == "__main__":
    generate(2000)
