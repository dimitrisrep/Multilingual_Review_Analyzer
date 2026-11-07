"""Pinned source ingestion, label-conflict quarantine and deterministic splits."""

import hashlib
import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCES = {
    "greek_reviews.zip": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/nikosfragkis/skroutz-shop-reviews-sentiment-analysis?datasetVersionNumber=1",
        "sha256": "2bba715f4d86c623f7c1c2e7e65ce2d2dc5991b90033b1c0a7b4bf9d6b6832f5",
        "license": "Other (specified in description); no redistribution grant located; raw text not bundled",
    },
    "uci_sentiment.zip": {
        "url": "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip",
        "sha256": "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343",
        "license": "CC BY 4.0; Kotzias (2015), UCI DOI 10.24432/C57604",
    },
}


def download(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, info in SOURCES.items():
        path = directory / filename
        if not path.exists():
            with urlopen(
                Request(info["url"], headers={"User-Agent": "review-analyzer/0.1"}),
                timeout=60,
            ) as response:
                body = response.read(5_000_000)
            if hashlib.sha256(body).hexdigest() != info["sha256"]:
                raise ValueError(f"Source checksum mismatch: {filename}")
            path.write_bytes(body)
        verify_source(path, info["sha256"])


def verify_source(path: Path, expected: str):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Source checksum mismatch: {path.name}")


def text_key(text: str):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def load_sources(directory: Path):
    for name, info in SOURCES.items():
        verify_source(directory / name, info["sha256"])
    with zipfile.ZipFile(directory / "greek_reviews.zip") as archive:
        greek = pd.read_excel(io.BytesIO(archive.read("Skroutz_dataset.xlsx")))
    rows = [
        {
            "id": f"skroutz-v1-{r.id}",
            "text": str(r.review),
            "label": {"Positive": 1, "Negative": 0}[r.label],
            "language": "el",
            "source": "skroutz_shop_reviews_v1",
        }
        for r in greek.itertuples()
    ]
    with zipfile.ZipFile(directory / "uci_sentiment.zip") as archive:
        text = archive.read(
            "sentiment labelled sentences/amazon_cells_labelled.txt"
        ).decode("utf-8")
    for i, line in enumerate(text.splitlines()):
        message, label = line.rsplit("\t", 1)
        rows.append(
            {
                "id": f"uci-amazon-{i}",
                "text": message,
                "label": int(label),
                "language": "en",
                "source": "uci_amazon_cells",
            }
        )
    return pd.DataFrame(rows)


def partition(frame: pd.DataFrame, seed: int = 42, max_fit=1200):
    frame = frame.copy()
    if not set(frame.label.unique()) <= {0, 1} or not set(frame.language.unique()) <= {
        "en",
        "el",
    }:
        raise ValueError("Expected binary labels and en/el languages.")
    if frame.id.duplicated().any() or frame.text.isna().any():
        raise ValueError("Review IDs must be unique and text must be present.")
    frame["key"] = frame.text.map(text_key)
    original = len(frame)
    empty = frame.key.eq("")
    conflicts = set(frame.groupby("key").label.nunique().loc[lambda x: x > 1].index)
    frame = frame[~empty & ~frame.key.isin(conflicts)].drop_duplicates("key").copy()
    frame["stratum"] = frame.language + ":" + frame.label.astype(str)
    development, test = train_test_split(
        frame, test_size=0.15, stratify=frame.stratum, random_state=seed
    )
    fit, validation = train_test_split(
        development,
        test_size=0.15 / 0.85,
        stratify=development.stratum,
        random_state=seed,
    )
    if max_fit < 4:
        raise ValueError("max_fit must allow both labels in both languages.")
    per_class = min(max_fit // 4, int(fit.stratum.value_counts().min()))
    selected = [
        part.sample(n=per_class, random_state=seed)
        for _, part in fit.groupby("stratum", sort=True)
    ]
    fit = pd.concat(selected).sort_values("id").reset_index(drop=True)
    splits = {
        "fit": fit,
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    for name, part in splits.items():
        for other_name, other in splits.items():
            if name != other_name:
                assert not set(part.key) & set(other.key)
    manifest = {
        "seed": seed,
        "source_rows": original,
        "retained_unique_rows": len(frame),
        "removed_empty_conflicting_or_duplicate_rows": original - len(frame),
        "conflicting_text_keys_quarantined": len(conflicts),
        "fit_cap_requested": max_fit,
        "fit_sampling": "equal samples from each language/label stratum after the split",
        "splits": {
            key: {
                "rows": len(value),
                "ids": value.id.tolist(),
                "class_counts": value.stratum.value_counts().to_dict(),
            }
            for key, value in splits.items()
        },
    }
    return splits, manifest


def save_manifest(manifest: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
