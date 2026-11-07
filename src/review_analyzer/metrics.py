import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_predictions(frame, predicted):
    predicted = np.asarray(predicted)
    result = {
        "rows": len(frame),
        "accuracy": accuracy_score(frame.label, predicted),
        "macro_f1": f1_score(frame.label, predicted, average="macro", zero_division=0),
        "labels": ["negative", "positive"],
        "confusion_matrix": confusion_matrix(
            frame.label, predicted, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            frame.label, predicted, output_dict=True, zero_division=0
        ),
        "by_language": {},
    }
    for language in ["el", "en"]:
        mask = frame.language.eq(language).to_numpy()
        if mask.any():
            result["by_language"][language] = {
                "rows": int(mask.sum()),
                "accuracy": accuracy_score(frame.loc[mask, "label"], predicted[mask]),
                "macro_f1": f1_score(
                    frame.loc[mask, "label"],
                    predicted[mask],
                    average="macro",
                    zero_division=0,
                ),
            }
    result["mean_language_macro_f1"] = float(
        np.mean([r["macro_f1"] for r in result["by_language"].values()])
    )
    return result
