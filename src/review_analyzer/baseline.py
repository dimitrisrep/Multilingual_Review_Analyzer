import json
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from review_analyzer.metrics import evaluate_predictions


def make_baseline():
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                                strip_accents="unicode",
                                max_features=30000,
                            ),
                        ),
                        (
                            "character",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                sublinear_tf=True,
                                strip_accents="unicode",
                                max_features=30000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=2, max_iter=2000, class_weight="balanced"),
            ),
        ]
    )


def train_baseline(splits: dict, output: Path):
    fit, validation, test = (splits[key] for key in ["fit", "validation", "test"])
    pipeline = make_baseline().fit(fit.text.tolist(), fit.label.tolist())
    report = {
        "model": "word_character_tfidf_logistic",
        "sklearn_version": sklearn.__version__,
        "fit_rows": len(fit),
        "validation": evaluate_predictions(
            validation, pipeline.predict(validation.text.tolist())
        ),
        "test": evaluate_predictions(test, pipeline.predict(test.text.tolist())),
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "sklearn_version": sklearn.__version__,
            "schema_version": 1,
        },
        output / "baseline.joblib",
    )
    (output / "baseline_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
