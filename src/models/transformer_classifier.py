"""Fine-tuned transformer classifiers (Hugging Face) for product / issue.

Training requires the optional deps: torch, transformers, datasets, accelerate.
Inference falls back gracefully if artifacts or deps are missing — the API then
serves the TF-IDF baseline and flags the model tier in its response.

Train:
    python scripts/train_transformer.py --task product --model distilbert-base-uncased
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "distilbert-base-uncased"


def _require_transformers():
    try:
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as e:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "Transformer path requires optional deps: "
            "pip install torch transformers datasets accelerate"
        ) from e


def train(
    texts: list[str],
    labels: list[str],
    output_dir: str | Path,
    base_model: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 256,
    lr: float = 5e-5,
) -> dict:
    """Fine-tune a sequence classifier and save it with its label map."""
    _require_transformers()
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    label_names = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    ds = Dataset.from_dict({"text": texts, "label": [label2id[l] for l in labels]})
    ds = ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
        batched=True,
    )
    split = ds.train_test_split(test_size=0.1, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label_names),
        id2label={i: l for l, i in label2id.items()},
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    Path(output_dir, "label_map.json").write_text(json.dumps(label2id, indent=2))
    return {"train_loss": result.training_loss, "labels": label_names}


class TransformerClassifier:
    """Inference wrapper matching BaselineClassifier's predict interface."""

    def __init__(self, model_dir: str | Path):
        _require_transformers()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict_batch(self, texts: list[str], k: int = 3):
        from src.models.baseline import Prediction

        enc = self.tokenizer(
            texts, truncation=True, max_length=256, padding=True, return_tensors="pt"
        )
        with self.torch.no_grad():
            logits = self.model(**enc).logits
        proba = self.torch.softmax(logits, dim=-1).numpy()
        out = []
        for row in proba:
            order = np.argsort(row)[::-1][:k]
            top = [(self.id2label[int(i)], float(row[i])) for i in order]
            out.append(Prediction(label=top[0][0], confidence=top[0][1], top_k=top))
        return out

    def predict(self, text: str, k: int = 3):
        return self.predict_batch([text], k=k)[0]


def load_if_available(model_dir: str | Path) -> "TransformerClassifier | None":
    """Return a TransformerClassifier if artifacts + deps exist, else None."""
    if not Path(model_dir, "config.json").exists():
        return None
    try:
        return TransformerClassifier(model_dir)
    except ImportError:
        logger.warning("Transformer artifacts found at %s but torch/transformers "
                       "not installed; falling back to baseline.", model_dir)
        return None
