"""End-to-end triage pipeline: classify -> retrieve -> generate.

This is the single object the API and evaluation suite share, so what is
measured offline is exactly what is served.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.models.baseline import BaselineClassifier
from src.models.escalation import EscalationClassifier
from src.models.transformer_classifier import load_if_available
from src.rag.generator import build_generator
from src.rag.retriever import build_retriever

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "artifacts" / "models"
POLICY_DIR = ROOT / "data" / "policies"


@dataclass
class TriageResult:
    product: dict
    issue: dict
    escalation: dict
    agent_assist: dict
    model_tier: str
    latency_ms: float
    explanation: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class TriagePipeline:
    def __init__(
        self,
        models_dir: str | Path = MODELS_DIR,
        policy_dir: str | Path = POLICY_DIR,
        retriever_backend: str = "auto",
    ):
        models_dir = Path(models_dir)
        self.product_baseline = BaselineClassifier.load(models_dir / "baseline_product.joblib")
        self.issue_baseline = BaselineClassifier.load(models_dir / "baseline_issue.joblib")
        self.escalation = EscalationClassifier.load(models_dir / "escalation.joblib")

        # Prefer fine-tuned transformers when artifacts + deps are present.
        self.product_transformer = load_if_available(models_dir / "transformer" / "product")
        self.issue_transformer = load_if_available(models_dir / "transformer" / "issue")

        self.retriever = build_retriever(policy_dir, backend=retriever_backend)
        self.generator = build_generator()

    @property
    def model_tier(self) -> str:
        return "transformer" if self.product_transformer else "baseline"

    def triage(self, narrative: str, k: int = 4) -> TriageResult:
        t0 = time.perf_counter()

        product_model = self.product_transformer or self.product_baseline
        issue_model = self.issue_transformer or self.issue_baseline
        product = product_model.predict(narrative)
        issue = issue_model.predict(narrative)
        esc = self.escalation.predict(narrative)

        query = f"{product.label} {issue.label} {narrative}"
        retrieved = self.retriever.retrieve(query, k=k)

        assist = self.generator.generate(
            narrative=narrative,
            product=product.label,
            issue=issue.label,
            escalation_score=esc.score,
            classifier_confidence=min(product.confidence, issue.confidence),
            retrieved=retrieved,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return TriageResult(
            product={"label": product.label, "confidence": round(product.confidence, 4),
                     "top_k": [[l, round(p, 4)] for l, p in product.top_k]},
            issue={"label": issue.label, "confidence": round(issue.confidence, 4),
                   "top_k": [[l, round(p, 4)] for l, p in issue.top_k]},
            escalation={"required": esc.escalate, "score": round(esc.score, 4),
                        "triggered_rules": esc.triggered_rules},
            agent_assist=asdict(assist),
            model_tier=self.model_tier,
            latency_ms=round(latency_ms, 1),
            explanation=[[t, round(w, 4)] for t, w in self.product_baseline.explain(narrative)],
        )
