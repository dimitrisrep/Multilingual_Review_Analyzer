# Multilingual Review Analyzer

Binary sentiment classification for Greek and English reviews. The default model combines word and character TF-IDF features with logistic regression. The project also supports partial fine-tuning of multilingual DistilBERT. Both models use the same CLI and local FastAPI demo.

## Quick start

Use Python 3.12 and run these commands from the project directory.

Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

```bash
python -m pip install -e .
review-analyzer download
review-analyzer train
review-analyzer analyze "Great quality and fast delivery."
python -m uvicorn review_analyzer.api:app --host 127.0.0.1 --port 8002
```

The download command verifies both source archives and saves them under `data/raw/`. Training writes the baseline model and evaluation report to `artifacts/`. Open the local demo to analyze a review or use the interactive API docs for single and batch requests. [Demo](http://127.0.0.1:8002) · [API docs](http://127.0.0.1:8002/docs)

## DistilBERT

Install the optional dependencies with `python -m pip install -e ".[deep]"`. For Linux CPU, the pinned dependencies are in `requirements-transformer.lock`.

```bash
review-analyzer train --model transformer
review-analyzer analyze "Η παράδοση άργησε και το προϊόν ήρθε χαλασμένο." --model transformer
```

The first run downloads the pinned pretrained model from Hugging Face (about 540 MB). Training saves the updated parameters for the final encoder block and classifier head. The demo shows DistilBERT once the trained parameters and configuration are available. Use `review-analyzer train --model both` to train both models in one run.

## Test results

Both models used the same 1,200 training reviews and the same held-out set of 1,130 reviews. The scores below are macro F1.

| Model | Overall | Greek (982) | English (148) | Mean across languages |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + logistic regression | 0.906 | 0.916 | 0.838 | 0.877 |
| Multilingual DistilBERT | 0.793 | 0.794 | 0.782 | 0.788 |

The Greek subset contains shop-service reviews from Skroutz, while the English subset contains Amazon product reviews. [Data card](docs/DATA_CARD.md) · [Model card](docs/MODEL_CARD.md) · [Experiment reports](reports/)

## API

- `POST /analyze` accepts a review and optional `model` and `language` fields.
- `POST /analyze-batch` accepts up to 50 reviews and returns predictions and label counts.
- `GET /health` reports model availability. `GET /ready` checks whether the baseline is loaded.

Each result includes a sentiment label, an uncalibrated model score and any keyword mentions for delivery, service, quality or price. Inputs are limited to 4,000 characters per review.

## Development

The following commands reproduce the Linux CPU test environment.

```bash
python -m pip install --require-hashes -r requirements.lock
python -m pip install --require-hashes -r requirements-transformer.lock
python -m pip install --no-deps -e .
ruff check src tests
pytest
```

## Docker

Train the baseline first, then mount the generated `artifacts/` directory when starting the container.

```bash
docker build -t review-analyzer .
docker run --rm -p 127.0.0.1:8002:8002 \
  --mount "type=bind,source=$PWD/artifacts,target=/app/artifacts,readonly" \
  review-analyzer
```
