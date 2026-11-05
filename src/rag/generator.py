"""Grounded agent-assist response generation with evidence citations.

Two generation modes behind one interface:

- TemplateGenerator (default): deterministic and offline. Retrieved policy text
  informs suggestions, which still require review; no unsupported-answer rate
  has been measured.
- ClaudeGenerator (optional): calls the Claude API with retrieved snippets and
  strict grounding instructions. Enable with `pip install anthropic` and
  ANTHROPIC_API_KEY set.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from src.rag.retriever import RetrievedChunk

# Below this retrieval score the snippet is likely off-topic; cite nothing
# rather than cite noise.
MIN_EVIDENCE_SCORE = 0.05


@dataclass
class AgentAssistResponse:
    summary: str
    recommended_action: str
    suggested_response: str
    evidence: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    human_review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    generator: str = "template"


def _summarize(narrative: str, max_words: int = 45) -> str:
    words = narrative.split()
    head = " ".join(words[:max_words])
    return head + ("…" if len(words) > max_words else "")


def _first_action_bullet(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^(\d+\.|[-*])\s+", line):
            return re.sub(r"^(\d+\.|[-*])\s+", "", line)
    return None


class TemplateGenerator:
    name = "template"

    def generate(
        self,
        narrative: str,
        product: str,
        issue: str,
        escalation_score: float,
        classifier_confidence: float,
        retrieved: list[RetrievedChunk],
    ) -> AgentAssistResponse:
        evidence = [
            {
                "chunk_id": r.chunk.chunk_id,
                "doc_title": r.chunk.doc_title,
                "section": r.chunk.section,
                "text": r.chunk.text,
                "score": round(r.score, 4),
            }
            for r in retrieved
            if r.score >= MIN_EVIDENCE_SCORE
        ]

        review_reasons = []
        if escalation_score >= 0.5:
            review_reasons.append(f"escalation score {escalation_score:.2f} ≥ 0.50")
        if classifier_confidence < 0.5:
            review_reasons.append(
                f"classifier confidence {classifier_confidence:.2f} < 0.50"
            )
        if not evidence:
            review_reasons.append("no policy evidence retrieved above threshold")

        top = evidence[0] if evidence else None
        action = (
            _first_action_bullet(top["text"]) if top else None
        ) or "Route to a servicing specialist for manual policy lookup."

        cites = ", ".join(f"[{i+1}] {e['doc_title']} — {e['section']}" for i, e in enumerate(evidence))
        suggested = (
            f"Thank you for contacting us about your {product.lower()} concern. "
            f"I understand the issue relates to: {issue.lower()}. "
            + (
                f"Per our servicing policy ({cites}), the next step is: {action} "
                if evidence
                else "I am escalating this to a specialist who can review the applicable policy. "
            )
            + "A servicing specialist should confirm the applicable next step and timeline."
        )

        # Confidence blends classifier certainty with evidence strength.
        evidence_strength = min(1.0, (evidence[0]["score"] * 2) if evidence else 0.0)
        confidence = round(0.6 * classifier_confidence + 0.4 * evidence_strength, 3)

        return AgentAssistResponse(
            summary=_summarize(narrative),
            recommended_action=action,
            suggested_response=suggested,
            evidence=evidence,
            confidence=confidence,
            human_review_required=bool(review_reasons),
            review_reasons=review_reasons,
            generator=self.name,
        )


class ClaudeGenerator:  # pragma: no cover - requires API key
    """LLM-backed generation with the same grounding contract."""

    name = "claude"

    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self._fallback = TemplateGenerator()

    def generate(self, narrative, product, issue, escalation_score,
                 classifier_confidence, retrieved) -> AgentAssistResponse:
        base = self._fallback.generate(
            narrative, product, issue, escalation_score, classifier_confidence, retrieved
        )
        snippets = "\n\n".join(
            f"[{i+1}] {e['doc_title']} — {e['section']}\n{e['text']}"
            for i, e in enumerate(base.evidence)
        )
        prompt = (
            "You are an agent-assist system for a consumer finance servicing team.\n"
            "Using ONLY the policy snippets below, draft a short empathetic response "
            "to the consumer and a one-line recommended action for the agent. "
            "Cite snippets as [n]. If the snippets do not cover the situation, say "
            "the case needs specialist review — do not invent policy.\n\n"
            f"Complaint: {narrative}\n\nPolicy snippets:\n{snippets}"
        )
        msg = self.client.messages.create(
            model=self.model, max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        base.suggested_response = msg.content[0].text
        base.generator = self.name
        return base


def build_generator() -> TemplateGenerator:
    """Claude if configured, deterministic templates otherwise."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeGenerator()
        except ImportError:
            pass
    return TemplateGenerator()
