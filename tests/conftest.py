import joblib
import pytest
import sklearn

from review_analyzer.baseline import make_baseline
from review_analyzer.service import ReviewAnalyzer


@pytest.fixture
def analyzer(tmp_path):
    texts = [
        "excellent product fast delivery",
        "wonderful service good quality",
        "great purchase good price",
        "bad product broken delivery",
        "terrible support poor quality",
        "awful purchase expensive broken",
        "άριστη εξυπηρέτηση πολύ καλό",
        "κακή ποιότητα απαράδεκτο προϊόν",
    ]
    pipeline = make_baseline().fit(texts, [1, 1, 1, 0, 0, 0, 1, 0])
    joblib.dump(
        {
            "schema_version": 1,
            "sklearn_version": sklearn.__version__,
            "pipeline": pipeline,
        },
        tmp_path / "baseline.joblib",
    )
    return ReviewAnalyzer(tmp_path)
