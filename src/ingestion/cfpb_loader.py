"""Load CFPB Consumer Complaint Database records from CSV or the public API.

Public data only: https://www.consumerfinance.gov/data-research/consumer-complaints/
Narratives published by the CFPB are already scrubbed of direct PII by the
bureau (names, account numbers are redacted as XXXX); we additionally run a
defensive PII scrub in preprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CFPB_API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

# Canonical column names used throughout the project.
COLUMNS = {
    "Consumer complaint narrative": "narrative",
    "Product": "product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
    "Submitted via": "submitted_via",
    "State": "state",
    "Complaint ID": "complaint_id",
}

REQUIRED = ["narrative", "product", "issue"]


def load_from_csv(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load a CFPB complaint export CSV (as downloaded from the CFPB site)."""
    df = pd.read_csv(path, nrows=max_rows, dtype=str)
    # Accept either raw CFPB headers or already-canonical headers.
    rename = {k: v for k, v in COLUMNS.items() if k in df.columns}
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return _standardize(df)


def load_from_api(
    size: int = 1000,
    products: Iterable[str] | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch complaints with narratives from the public CFPB search API."""
    if size < 1:
        raise ValueError("size must be positive")
    params: dict = {
        "size": min(size, 10_000),
        "has_narrative": "true",
        "format": "json",
        "no_aggs": "true",
    }
    if products:
        params["product"] = list(products)
    resp = requests.get(CFPB_API_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    rows = []
    for h in hits:
        s = h.get("_source", {})
        rows.append(
            {
                "complaint_id": s.get("complaint_id"),
                "narrative": s.get("complaint_what_happened"),
                "product": s.get("product"),
                "issue": s.get("issue"),
                "sub_issue": s.get("sub_issue"),
                "company_response": s.get("company_response"),
                "timely_response": s.get("timely"),
                "submitted_via": s.get("submitted_via"),
                "state": s.get("state"),
            }
        )
    logger.info("Fetched %d complaints from CFPB API", len(rows))
    return _standardize(pd.DataFrame(rows, columns=list(COLUMNS.values())))


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Keep canonical columns, drop rows without a usable narrative/labels."""
    keep = [c for c in COLUMNS.values() if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=[c for c in REQUIRED if c in df.columns])
    df = df[df["narrative"].str.strip().str.len() > 20]
    df = df.reset_index(drop=True)
    return df
