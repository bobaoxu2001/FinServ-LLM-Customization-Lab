"""Weak-label escalation classifier.

No hand-labeled escalation ground truth exists in CFPB data, so we build weak
labels with labeling functions over structured fields + keyword heuristics
(Snorkel-style majority vote), then distill them into a TF-IDF + LogisticRegression
model so escalation generalizes to raw narratives at inference time.

Weak-label sources:
  LF1  company_response indicates relief ("Closed with monetary relief") -> escalate
  LF2  timely_response == "No" -> escalate
  LF3  issue category is high-risk (fraud, identity theft, foreclosure, ...) -> escalate
  LF4  narrative contains urgency/legal/hardship keywords -> escalate
  LF5  routine informational issues -> do not escalate
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.models.baseline import build_pipeline

HIGH_RISK_ISSUES = (
    "fraud",
    "identity theft",
    "foreclosure",
    "repossession",
    "threat",
    "harassment",
    "unauthorized",
    "scam",
)

URGENT_KEYWORDS = re.compile(
    r"\b(lawyer|attorney|lawsuit|sue|cfpb|regulator|fraud|stolen|identity theft|"
    r"unauthorized|police|eviction|foreclos\w+|homeless|hardship|disabled|"
    r"veteran|elderly|threat\w*|harass\w*|urgent|immediately|desperate)\b",
    re.IGNORECASE,
)

ROUTINE_ISSUES = ("getting a credit card", "closing an account", "struggling to pay")


@dataclass
class EscalationResult:
    escalate: bool
    score: float
    triggered_rules: list[str]


def weak_label_row(row: pd.Series) -> tuple[int, list[str]]:
    """Vote labeling functions on one complaint row. Returns (label, fired LFs)."""
    votes: list[int] = []
    fired: list[str] = []

    resp = str(row.get("company_response", "")).lower()
    if "monetary relief" in resp or "non-monetary relief" in resp:
        votes.append(1)
        fired.append("LF1:company_granted_relief")

    if str(row.get("timely_response", "Yes")).lower().startswith("n"):
        votes.append(1)
        fired.append("LF2:untimely_response")

    issue = str(row.get("issue", "")).lower() + " " + str(row.get("sub_issue", "")).lower()
    if any(k in issue for k in HIGH_RISK_ISSUES):
        votes.append(1)
        fired.append("LF3:high_risk_issue")

    n_hits = len(URGENT_KEYWORDS.findall(str(row.get("narrative", ""))))
    if n_hits >= 2:
        votes.append(1)
        fired.append("LF4:urgent_keywords")

    if any(k in issue for k in ROUTINE_ISSUES) and n_hits == 0:
        votes.append(0)
        fired.append("LF5:routine_issue")

    if not votes:
        return 0, fired
    return int(sum(votes) >= max(1, len(votes) / 2) and sum(votes) >= 1), fired


def build_weak_labels(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: weak_label_row(r)[0], axis=1)


class EscalationClassifier:
    """Distilled model: narrative text -> P(escalation required)."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline or build_pipeline()

    def fit(self, df: pd.DataFrame) -> "EscalationClassifier":
        labels = build_weak_labels(df)
        self.pipeline.fit(df["narrative"].tolist(), labels.tolist())
        return self

    def predict(self, narrative: str, threshold: float = 0.5) -> EscalationResult:
        proba = float(self.pipeline.predict_proba([narrative])[0][list(self.pipeline.classes_).index(1)])
        # Surface which rules fire on the raw text for auditability.
        _, fired = weak_label_row(pd.Series({"narrative": narrative}))
        return EscalationResult(escalate=proba >= threshold, score=proba, triggered_rules=fired)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "EscalationClassifier":
        return cls(pipeline=joblib.load(path))
