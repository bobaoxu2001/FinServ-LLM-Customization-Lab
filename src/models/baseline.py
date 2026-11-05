"""TF-IDF + Logistic Regression baselines for product and issue classification.

The baseline is the reference point every LLM customization must beat, and the
cheap fallback the API serves when transformer artifacts are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class Prediction:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, C=4.0, class_weight="balanced", n_jobs=-1
                ),
            ),
        ]
    )


class BaselineClassifier:
    """Top-k model probabilities and serialization; probabilities are not calibrated."""

    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline or build_pipeline()

    def fit(self, texts: list[str], labels: list[str]) -> "BaselineClassifier":
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, text: str, k: int = 3) -> Prediction:
        return self.predict_batch([text], k=k)[0]

    def predict_batch(self, texts: list[str], k: int = 3) -> list[Prediction]:
        if k < 1:
            raise ValueError("k must be positive")
        if not texts:
            return []
        proba = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_
        out = []
        for row in proba:
            order = np.argsort(row)[::-1][:k]
            top = [(str(classes[i]), float(row[i])) for i in order]
            out.append(Prediction(label=top[0][0], confidence=top[0][1], top_k=top))
        return out

    def explain(self, text: str, top_n: int = 8) -> list[tuple[str, float]]:
        """Per-prediction token attributions from linear coefficients."""
        tfidf: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        vec = tfidf.transform([text])
        pred_idx = int(np.argmax(clf.predict_proba(vec)))
        if clf.coef_.shape[0] > 1:
            coefs = clf.coef_[pred_idx]
        else:
            # sklearn's single binary coefficient row supports classes_[1].
            # Reverse its sign when explaining a prediction of classes_[0].
            coefs = clf.coef_[0] * (1 if pred_idx == 1 else -1)
        feature_names = tfidf.get_feature_names_out()
        contributions = vec.multiply(coefs).tocoo()
        scored = sorted(
            ((feature_names[j], float(v)) for j, v in zip(contributions.col, contributions.data)),
            key=lambda x: -abs(x[1]),
        )
        return scored[:top_n]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineClassifier":
        return cls(pipeline=joblib.load(path))
