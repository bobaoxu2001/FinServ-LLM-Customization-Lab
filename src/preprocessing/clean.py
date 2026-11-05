"""Text cleaning and defensive PII scrubbing for complaint narratives.

CFPB narratives are pre-redacted by the bureau ("XXXX"), but we run a second
pass so the pipeline is safe against arbitrary CSV inputs.
"""

from __future__ import annotations

import re

import pandas as pd

# Defensive PII patterns (belt-and-suspenders on top of CFPB redaction).
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD_NUMBER]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?1[ -.]?)?\(?\d{3}\)?[ -.]?\d{3}[ -.]?\d{4}\b"), "[PHONE]"),
]

_REDACTION = re.compile(r"X{2,}")
_WHITESPACE = re.compile(r"\s+")


def scrub_pii(text: str) -> str:
    for pattern, token in _PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


def clean_narrative(text: str) -> str:
    """Normalize a single complaint narrative."""
    text = str(text)
    text = scrub_pii(text)
    text = _REDACTION.sub("[REDACTED]", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def prepare_dataset(df: pd.DataFrame, min_words: int = 5) -> pd.DataFrame:
    """Clean narratives and normalize label columns for modeling."""
    df = df.copy()
    df["narrative"] = df["narrative"].map(clean_narrative)
    df = df[df["narrative"].str.split().str.len() >= min_words]
    for col in ("product", "issue", "sub_issue"):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").str.strip()
    if "timely_response" in df.columns:
        df["timely_response"] = (
            df["timely_response"].fillna("Yes").str.strip().str.capitalize()
        )
    return df.reset_index(drop=True)


def train_test_split_df(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic narrative-group split; duplicate text never crosses folds."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must lie strictly between zero and one")
    groups = df["narrative"].drop_duplicates().sample(frac=1.0, random_state=seed)
    if len(groups) < 2:
        raise ValueError("at least two distinct narratives are required")
    n_test = min(len(groups) - 1, max(1, int(len(groups) * test_size)))
    is_test = df["narrative"].isin(groups.iloc[:n_test])
    return df.loc[~is_test].reset_index(drop=True), df.loc[is_test].reset_index(drop=True)
