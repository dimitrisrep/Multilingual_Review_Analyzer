import pandas as pd
import pytest

from review_analyzer.data import partition, text_key, verify_source
from review_analyzer.metrics import evaluate_predictions


def frame():
    return pd.DataFrame(
        [
            {
                "id": f"{language}-{label}-{i}",
                "text": f"review {language} {label} {i}",
                "language": language,
                "label": label,
            }
            for language in ["el", "en"]
            for label in [0, 1]
            for i in range(30)
        ]
    )


def test_duplicate_and_conflicting_texts_never_leak():
    original = frame()
    duplicate = {
        **original.iloc[0].to_dict(),
        "id": "duplicate",
        "text": "  REVIEW el 0 0  ",
    }
    conflict = {**original.iloc[1].to_dict(), "id": "conflict", "label": 1}
    splits, manifest = partition(
        pd.concat([original, pd.DataFrame([duplicate, conflict])]), max_fit=40
    )
    sets = [set(part.text.map(text_key)) for part in splits.values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1 :])
    assert all(text_key(conflict["text"]) not in keys for keys in sets)
    assert manifest["conflicting_text_keys_quarantined"] == 1
    assert len(splits["fit"]) == 40
    assert set(splits["fit"].groupby(["language", "label"]).size()) == {10}
    assert partition(original, max_fit=40)[1] == partition(original, max_fit=40)[1]


def test_integrity_failure_does_not_silently_accept_new_dataset(tmp_path):
    source = tmp_path / "data.zip"
    source.write_bytes(b"changed upstream data")
    with pytest.raises(ValueError, match="checksum"):
        verify_source(source, "0" * 64)


def test_per_language_metric_does_not_hide_minority_language():
    data = pd.DataFrame({"language": ["el"] * 8 + ["en"] * 2, "label": [0, 1] * 5})
    metric = evaluate_predictions(data, [0, 1] * 4 + [1, 0])
    assert metric["accuracy"] == 0.8
    assert metric["by_language"]["en"]["macro_f1"] == 0
    assert metric["mean_language_macro_f1"] == 0.5
