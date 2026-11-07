# Model card

## Models

The baseline combines word unigrams and bigrams with character 3–5 grams using TF-IDF. The classifier is balanced logistic regression with `C=2`.

The Transformer updates the final encoder block and binary classification head, with the embeddings and earlier encoder blocks frozen. It uses the Apache-2.0 multilingual DistilBERT base at revision `45c032ab32cc946ad88a166f7cb282f58c753c2e`. [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)

## Training and evaluation

The recorded Transformer run used CPU, seed 42, batch size 16, a maximum length of 96 tokens and AdamW with a learning rate of 0.0002. The checkpoint from epoch one of two had the highest mean per-language validation macro F1.

The baseline achieved higher validation and test macro F1 averaged across the two languages. It is the default for inference. [Baseline metrics](../reports/baseline_metrics.json) · [Transformer metrics](../reports/transformer_metrics.json)

## Inference

The classifier predicts positive or negative sentiment. The score is uncalibrated, and `review_required` flags scores below 0.65. Keyword matches return character offsets for the original text. Language is supplied by the caller or inferred from Greek script. DistilBERT processes the first 96 tokens.

Load `baseline.joblib` only from a trusted training run. The Transformer loader checks the checksum and parameter names of the saved delta against the configuration.
