import pytest
from fastapi.testclient import TestClient

from review_analyzer.api import create_app
from review_analyzer.service import ReviewAnalyzer, aspect_mentions


def test_actual_baseline_round_trip_batch_and_schema(analyzer):
    with TestClient(create_app(analyzer)) as client:
        assert client.get("/ready").status_code == 200
        response = client.post(
            "/analyze", json={"text": "excellent product fast delivery"}
        )
        assert response.status_code == 200
        result = response.json()
        assert result["sentiment"] == "positive"
        assert result["score_type"] == "uncalibrated_model_score"
        assert result["aspect_mentions"][0]["quote"] == "delivery"
        batch = client.post(
            "/analyze-batch", json={"texts": ["good product", "bad broken product"]}
        )
        assert sum(batch.json()["counts"].values()) == 2
        for payload in [
            {"text": " "},
            {"text": "x" * 4001},
            {"text": "test", "model": "unknown"},
            {"text": "test", "extra": "ignored?"},
        ]:
            assert client.post("/analyze", json=payload).status_code == 422
        assert (
            client.post("/analyze-batch", json={"texts": ["x"] * 51}).status_code == 422
        )


def test_readiness_and_missing_transformer_do_not_fake_results(tmp_path):
    with TestClient(create_app(ReviewAnalyzer(tmp_path))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/analyze", json={"text": "great"}).status_code == 503
        assert 'value="transformer"' not in client.get("/").text
        (tmp_path / "transformer_config.json").touch()
        assert not client.get("/health").json()["transformer_artifact_present"]
        response = client.post(
            "/analyze", json={"text": "great", "model": "transformer"}
        )
        assert response.status_code == 503
        (tmp_path / "transformer_delta.safetensors").touch()
        assert client.get("/health").json()["transformer_artifact_present"]
        assert 'value="transformer"' in client.get("/").text


@pytest.mark.parametrize(
    "text", ["Κακή εξυπηρέτηση, ακριβή τιμή.", "Delivery and support were good."]
)
def test_mentions_preserve_original_offsets_and_do_not_claim_aspect_sentiment(text):
    mentions = aspect_mentions(text)
    assert len(mentions) >= 2
    for item in mentions:
        assert text[item["start"] : item["end"]] == item["quote"]
        assert item["method"] == "keyword_mention_not_aspect_sentiment"
