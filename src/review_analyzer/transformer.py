"""Train the final encoder block and classification head, preserving the pretrained backbone."""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from review_analyzer.metrics import evaluate_predictions

MODEL_ID = "distilbert/distilbert-base-multilingual-cased"
MODEL_REVISION = "45c032ab32cc946ad88a166f7cb282f58c753c2e"


def configure_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in [
        model.distilbert.transformer.layer[-1],
        model.pre_classifier,
        model.classifier,
    ]:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]


def load_backbone(cache_dir=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        trust_remote_code=False,
        use_fast=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
        trust_remote_code=False,
        attn_implementation="eager",
    )
    return tokenizer, model


def predict_probabilities(model, tokenizer, texts, batch_size=16, max_length=96):
    import torch

    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batches.append(model(**tokens).logits.softmax(-1).cpu().numpy())
    return np.concatenate(batches)


def train_transformer(
    splits,
    output: Path,
    cache_dir=None,
    epochs=2,
    batch_size=16,
    max_length=96,
    learning_rate=2e-4,
    threads=4,
    seed=42,
):
    import torch
    from safetensors.torch import save_file

    if epochs < 1 or batch_size < 1 or not 8 <= max_length <= 512:
        raise ValueError("Invalid training configuration.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    tokenizer, model = load_backbone(cache_dir)
    trainable_names = configure_trainable(model)
    fit, validation, test = (splits[key] for key in ["fit", "validation", "test"])
    tokenized = tokenizer(
        fit.text.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(fit.label.to_numpy(), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(
        tokenized["input_ids"], tokenized["attention_mask"], labels
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )
    watched_name = next(n for n in trainable_names if ".attention.q_lin.weight" in n)
    initial_encoder_weight = model.state_dict()[watched_name].detach().clone()
    output.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        # Frozen embeddings/early layers stay in eval mode; train only the final block and head.
        model.eval()
        model.distilbert.transformer.layer[-1].train()
        model.dropout.train()
        total_loss = 0.0
        for step, (input_ids, mask, answers) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, attention_mask=mask, labels=answers).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            total_loss += float(loss.detach())
            if step % 20 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "steps": len(loader),
                            "loss": total_loss / step,
                        }
                    ),
                    flush=True,
                )
        probabilities = predict_probabilities(
            model, tokenizer, validation.text.tolist(), batch_size, max_length
        )
        metrics = evaluate_predictions(validation, probabilities.argmax(axis=1))
        score = metrics["mean_language_macro_f1"]
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": total_loss / len(loader),
                "validation": metrics,
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            selected_state = {
                name: parameter.detach().cpu().clone().contiguous()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            save_file(selected_state, str(output / "transformer_delta.safetensors"))
        print(
            json.dumps(
                {
                    "epoch_finished": epoch + 1,
                    "validation_mean_language_macro_f1": score,
                }
            ),
            flush=True,
        )
    from safetensors.torch import load_file

    delta = load_file(str(output / "transformer_delta.safetensors"))
    model.load_state_dict(delta, strict=False)
    encoder_update_norm = float(
        torch.norm(model.state_dict()[watched_name] - initial_encoder_weight)
    )
    assert encoder_update_norm > 0, (
        "The encoder did not update. This is not fine-tuning."
    )
    # No test text/labels is used for epoch or hyperparameter selection.
    probabilities = predict_probabilities(
        model, tokenizer, test.text.tolist(), batch_size, max_length
    )
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_license": "Apache-2.0",
        "training": "partial fine-tuning: final encoder block plus classifier head; all earlier parameters frozen",
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "fit_rows": len(fit),
        "epochs": epochs,
        "selected_epoch": best_epoch,
        "selection_metric": "mean per-language validation macro F1",
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "threads": threads,
        "torch_version": torch.__version__,
        "device": "cpu",
        "elapsed_seconds": time.perf_counter() - started,
        "encoder_update_l2_norm": encoder_update_norm,
        "history": history,
        "test": evaluate_predictions(test, probabilities.argmax(axis=1)),
        "delta_sha256": hashlib.sha256(
            (output / "transformer_delta.safetensors").read_bytes()
        ).hexdigest(),
    }
    configuration = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "max_length": max_length,
        "trainable_names": trainable_names,
        "delta_sha256": report["delta_sha256"],
    }
    (output / "transformer_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    (output / "transformer_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def load_finetuned(directory: Path, cache_dir=None):
    import torch
    from safetensors.torch import load_file

    threads = int(os.environ.get("REVIEW_TORCH_THREADS", "4"))
    if not 1 <= threads <= 16:
        raise ValueError("REVIEW_TORCH_THREADS must be in 1-16.")
    torch.set_num_threads(threads)

    config = json.loads(
        (directory / "transformer_config.json").read_text(encoding="utf-8")
    )
    if (
        config["schema_version"] != 1
        or config["model_id"] != MODEL_ID
        or config["revision"] != MODEL_REVISION
    ):
        raise ValueError("Unexpected Transformer artifact contract.")
    path = directory / "transformer_delta.safetensors"
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["delta_sha256"]:
        raise ValueError("Transformer delta checksum mismatch.")
    tokenizer, model = load_backbone(cache_dir)
    names = configure_trainable(model)
    delta = load_file(str(path))
    if set(delta) != set(names) or set(names) != set(config["trainable_names"]):
        raise ValueError("Incomplete or unexpected trainable parameter set.")
    model.load_state_dict(delta, strict=False)
    model.eval()
    return tokenizer, model, config
