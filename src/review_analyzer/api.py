import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from review_analyzer.service import ReviewAnalyzer


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    model: Literal["baseline", "transformer"] = "baseline"
    language: Literal["el", "en"] | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[str] = Field(min_length=1, max_length=50)
    model: Literal["baseline", "transformer"] = "baseline"


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service or ReviewAnalyzer(
            Path(os.environ.get("REVIEW_ARTIFACTS", "artifacts")),
            os.environ.get("REVIEW_MODEL_CACHE"),
        )
        yield

    app = FastAPI(
        title="Multilingual Review Analyzer", version="0.1.0", lifespan=lifespan
    )

    @app.get("/health")
    def health():
        s = app.state.service
        return {
            "status": "ok",
            "baseline_ready": s.baseline is not None,
            "transformer_loaded": s.deep is not None,
            "transformer_artifact_present": s.transformer_artifact_present,
        }

    @app.get("/ready")
    def ready():
        if app.state.service.baseline is None:
            raise HTTPException(503, "No trained baseline loaded.")
        return {"status": "ready"}

    def call(operation):
        try:
            return operation()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, OSError, ImportError) as exc:
            raise HTTPException(
                503,
                "Requested model unavailable; check artifacts and optional dependencies.",
            ) from exc

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest):
        return call(
            lambda: app.state.service.analyze(
                request.text, request.model, request.language
            )
        )

    @app.post("/analyze-batch")
    def batch(request: BatchRequest):
        return call(
            lambda: app.state.service.analyze_batch(request.texts, request.model)
        )

    @app.get("/", response_class=HTMLResponse)
    def demo():
        transformer_option = (
            '<option value="transformer">Fine-tuned DistilBERT</option>'
            if app.state.service.transformer_artifact_present
            else ""
        )
        page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <title>Multilingual Review Analyzer</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      max-width: 800px;
      margin: 48px auto;
      padding: 0 20px;
      color: #202124;
    }
    textarea, select, button { font: inherit; padding: 10px; }
    textarea { width: 95%; min-height: 120px; }
    label { display: block; margin: 16px 0 8px; }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f4f4f4;
      padding: 20px;
    }
  </style>
</head>
<body>
  <h1>Multilingual Review Analyzer</h1>
  <p>Greek and English sentiment classification with keyword mentions.</p>
  <form id="form">
    <label for="text">Review</label>
    <textarea id="text" name="text" maxlength="4000" required></textarea>
    <label for="model">Model</label>
    <select id="model" name="model">
      <option value="baseline">TF-IDF baseline</option>
      <!-- transformer-option -->
    </select>
    <button type="submit">Analyze</button>
  </form>
  <pre id="result" aria-live="polite">Ready.</pre>
  <script>
    const form = document.getElementById('form');
    const output = document.getElementById('result');

    form.addEventListener('submit', async event => {
      event.preventDefault();
      const model = form.elements.model.value;
      const text = form.elements.text.value;
      output.textContent = model === 'transformer'
        ? 'Analyzing. The pretrained model may download on first use.'
        : 'Analyzing.';

      try {
        const response = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, model }),
        });
        output.textContent = JSON.stringify(await response.json(), null, 2);
      } catch (error) {
        output.textContent = String(error);
      }
    });
  </script>
</body>
</html>"""
        return page.replace("<!-- transformer-option -->", transformer_option)

    return app


app = create_app()
