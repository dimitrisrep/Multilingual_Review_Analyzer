import hashlib
import json

import pytest

from review_analyzer.transformer import (
    MODEL_ID,
    MODEL_REVISION,
    configure_trainable,
    load_finetuned,
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def tiny_model():
    config = transformers.DistilBertConfig(
        vocab_size=40,
        dim=24,
        hidden_dim=48,
        n_layers=2,
        n_heads=2,
        num_labels=2,
        dropout=0,
        attention_dropout=0,
    )
    return transformers.DistilBertForSequenceClassification(config)


def test_one_gradient_step_updates_encoder_but_preserves_frozen_layers():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = tiny_model()
    names = configure_trainable(model)
    assert any("transformer.layer.1.attention.q_lin.weight" in name for name in names)
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    model(
        input_ids=torch.randint(1, 40, (4, 8)), labels=torch.tensor([0, 1, 0, 1])
    ).loss.backward()
    optimizer.step()
    assert not torch.equal(
        before["distilbert.transformer.layer.1.attention.q_lin.weight"],
        model.state_dict()["distilbert.transformer.layer.1.attention.q_lin.weight"],
    )
    for name, parameter in model.named_parameters():
        if name not in names:
            assert torch.equal(before[name], parameter), name


def test_delta_round_trip_and_tamper_rejection(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    import review_analyzer.transformer as module

    model = tiny_model()
    names = configure_trainable(model)
    path = tmp_path / "transformer_delta.safetensors"
    save_file(
        {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name in names
        },
        str(path),
    )
    config = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "max_length": 96,
        "trainable_names": names,
        "delta_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (tmp_path / "transformer_config.json").write_text(json.dumps(config))
    monkeypatch.setattr(
        module, "load_backbone", lambda cache_dir: (object(), tiny_model())
    )
    _, loaded, _ = load_finetuned(tmp_path)
    assert torch.equal(model.classifier.weight, loaded.classifier.weight)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_finetuned(tmp_path)
