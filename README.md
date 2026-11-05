# FinServ LLM Customization Lab

A local complaint-triage library with CFPB CSV/API ingestion, defensive text
scrubbing, TF-IDF classification, escalation rules, and policy retrieval.
The default response generator uses deterministic templates. Transformer,
embedding and Claude integrations are optional code paths, not evidence of a
completed fine-tuning run or production deployment.

## Run the offline checks

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

The tests use small fixtures and mocked HTTP responses. No credentials, paid
inference, private complaints or model artifacts are required. On Apple Silicon,
create the environment and install its packages with the same Python architecture.

## Scope and limits

- Narrative groups stay together across train/test splits to prevent duplicate
  text leaking across the evaluation boundary.
- Classifier probabilities and the generator's confidence blend are heuristic;
  calibration and real-world quality have not been established.
- PII scrubbing is a defensive pattern-based pass, not a guarantee of anonymity.
- No serving API, frontend, trained model or measured business outcome is
  supplied. The `src/api`, `frontend` and notebook folders are placeholders.
- `TriagePipeline` requires locally trained model artifacts and a policy corpus.
  Never load untrusted joblib artifacts.
- Raw CFPB downloads and credentials stay out of Git.
