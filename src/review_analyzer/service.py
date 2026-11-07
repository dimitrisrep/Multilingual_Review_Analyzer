"""Shared inference path for the CLI, API and batch analysis."""

import re
import threading
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import sklearn

ASPECTS = {
    "delivery": r"deliver\w*|shipping|arriv\w*|παραδο\w*|παραλαβ\w*|μεταφορ\w*",
    "service": r"service|support|staff|εξυπηρετ\w*|προσωπικ\w*|υποστηριξ\w*",
    "quality": r"quality|broken|defect\w*|ποι[οό]τητ\w*|ελαττωμα\w*|χαλασ\w*",
    "price": r"price|cost|expensive|cheap|τιμ[ηε]\w*|κοστο\w*|ακριβ\w*|φθην\w*",
}


def aspect_mentions(text: str):
    normalized, offsets = [], []
    for position, char in enumerate(text):
        for c in unicodedata.normalize("NFD", char.casefold()):
            if not unicodedata.combining(c):
                normalized.append(c)
                offsets.append(position)
    normalized = "".join(normalized)
    results = []
    for aspect, pattern in ASPECTS.items():
        for match in re.finditer(r"\b(?:" + pattern + r")\b", normalized):
            start, end = offsets[match.start()], offsets[match.end() - 1] + 1
            results.append(
                {
                    "aspect": aspect,
                    "quote": text[start:end],
                    "start": start,
                    "end": end,
                    "method": "keyword_mention_not_aspect_sentiment",
                }
            )
    return results


class ReviewAnalyzer:
    def __init__(self, directory: Path, cache_dir=None):
        self.directory = Path(directory)
        self.cache_dir = cache_dir
        self.baseline = None
        self.deep = None
        self.lock = threading.Lock()
        path = self.directory / "baseline.joblib"
        if path.exists():
            bundle = joblib.load(path)  # Load trusted local training output only.
            if (
                bundle.get("schema_version") != 1
                or bundle["sklearn_version"] != sklearn.__version__
            ):
                raise ValueError(
                    "Baseline artifact schema or sklearn version mismatch."
                )
            self.baseline = bundle["pipeline"]

    @property
    def transformer_artifact_present(self) -> bool:
        return (self.directory / "transformer_config.json").is_file() and (
            self.directory / "transformer_delta.safetensors"
        ).is_file()

    def analyze(self, text: str, model="baseline", language=None):
        if not text.strip() or len(text) > 4000:
            raise ValueError("Text must contain 1-4000 characters.")
        if model not in {"baseline", "transformer"}:
            raise ValueError("Unknown model.")
        if language not in {None, "el", "en"}:
            raise ValueError("Supported languages are el and en.")
        if model == "baseline":
            if self.baseline is None:
                raise RuntimeError("No trained baseline artifact.")
            scores = self.baseline.predict_proba([text])[0]
        else:
            with self.lock:
                from review_analyzer.transformer import (
                    load_finetuned,
                    predict_probabilities,
                )

                if self.deep is None:
                    if not self.transformer_artifact_present:
                        raise RuntimeError("No fine-tuned Transformer artifact.")
                    self.deep = load_finetuned(self.directory, self.cache_dir)
                tokenizer, backbone, config = self.deep
                scores = predict_probabilities(
                    backbone, tokenizer, [text], max_length=config["max_length"]
                )[0]
        winner = int(np.argmax(scores))
        return {
            "sentiment": ["negative", "positive"][winner],
            "model": model,
            "score": float(scores[winner]),
            "score_type": "uncalibrated_model_score",
            "language": language
            or ("el" if re.search(r"[\u0370-\u03ff]", text) else "en"),
            "language_source": "declared" if language else "script_heuristic",
            "review_required": bool(scores[winner] < 0.65),
            "review_threshold_type": "fixed_demo_heuristic_not_validated_routing",
            "aspect_mentions": aspect_mentions(text),
        }

    def analyze_batch(self, texts, model="baseline"):
        if not 1 <= len(texts) <= 50:
            raise ValueError("A batch must contain 1-50 texts.")
        rows = [self.analyze(text, model) for text in texts]
        return {
            "results": rows,
            "counts": {
                label: sum(r["sentiment"] == label for r in rows)
                for label in ["negative", "positive"]
            },
        }
