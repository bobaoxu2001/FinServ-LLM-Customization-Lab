from unittest.mock import Mock

import pandas as pd
import pytest

from src.ingestion.cfpb_loader import load_from_api
from src.models.baseline import BaselineClassifier
from src.preprocessing.clean import clean_narrative, train_test_split_df
from src.rag.generator import TemplateGenerator


def test_empty_api_result_has_a_usable_schema(monkeypatch):
    response = Mock()
    response.json.return_value = {"hits": {"hits": []}}
    monkeypatch.setattr("src.ingestion.cfpb_loader.requests.get", lambda *a, **kw: response)
    result = load_from_api()
    assert result.empty
    assert {"narrative", "product", "issue"} <= set(result.columns)
    with pytest.raises(ValueError):
        load_from_api(size=0)


def test_split_keeps_duplicate_narratives_together_and_rejects_empty_training():
    frame = pd.DataFrame({"narrative": ["duplicate"] * 4 + ["other", "third"]})
    train, test = train_test_split_df(frame, test_size=0.5)
    assert set(train.narrative).isdisjoint(test.narrative)
    assert len(train) + len(test) == len(frame)
    pd.testing.assert_frame_equal(train, train_test_split_df(frame, test_size=0.5)[0])
    with pytest.raises(ValueError):
        train_test_split_df(frame.iloc[:1])
    with pytest.raises(ValueError):
        train_test_split_df(frame, test_size=1)


def test_binary_explanation_supports_the_predicted_class():
    model = BaselineClassifier().fit(
        ["mortgage home loan"] * 4 + ["card credit payment"] * 4,
        ["mortgage"] * 4 + ["card"] * 4,
    )
    for text, label in [("mortgage home loan", "mortgage"), ("card credit payment", "card")]:
        assert model.predict(text).label == label
        assert sum(weight for _, weight in model.explain(text)) > 0
    assert model.predict_batch([]) == []
    with pytest.raises(ValueError):
        model.predict("home loan", k=0)


def test_no_evidence_requires_review_without_a_policy_promise():
    result = TemplateGenerator().generate("A complaint", "Card", "Fees", 0.1, 0.9, [])
    assert result.human_review_required
    assert not result.evidence
    assert "policy timeline" not in result.suggested_response
    cleaned = clean_narrative("Email alice@example.com or call 212-555-1234. XXXX")
    assert "alice@example.com" not in cleaned
    assert "212-555-1234" not in cleaned
