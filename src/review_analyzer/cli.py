import argparse
import json
from pathlib import Path

from review_analyzer.baseline import train_baseline
from review_analyzer.data import download, load_sources, partition, save_manifest


def main():
    parser = argparse.ArgumentParser(prog="review-analyzer")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--data", type=Path, default=Path("data/raw"))
    fit = sub.add_parser("train")
    fit.add_argument("--data", type=Path, default=Path("data/raw"))
    fit.add_argument("--output", type=Path, default=Path("artifacts"))
    fit.add_argument(
        "--model",
        choices=["baseline", "transformer", "both"],
        default="baseline",
        help="model to train (default: baseline)",
    )
    fit.add_argument("--max-fit", type=int, default=1200)
    fit.add_argument("--epochs", type=int, default=2)
    fit.add_argument("--threads", type=int, default=4)
    fit.add_argument("--cache-dir")
    predict = sub.add_parser("analyze")
    predict.add_argument("text")
    predict.add_argument(
        "--model", choices=["baseline", "transformer"], default="baseline"
    )
    predict.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    predict.add_argument("--cache-dir")
    args = parser.parse_args()
    if args.command == "download":
        download(args.data)
        print("Source archives verified.")
    elif args.command == "train":
        splits, manifest = partition(load_sources(args.data), max_fit=args.max_fit)
        save_manifest(manifest, args.output / "split_manifest.json")
        report = {}
        if args.model in {"baseline", "both"}:
            report["baseline"] = train_baseline(splits, args.output)["test"]
        if args.model in {"transformer", "both"}:
            from review_analyzer.transformer import train_transformer

            report["transformer"] = train_transformer(
                splits,
                args.output,
                args.cache_dir,
                epochs=args.epochs,
                threads=args.threads,
            )["test"]
        print(json.dumps(report, indent=2))
    else:
        from review_analyzer.service import ReviewAnalyzer

        print(
            json.dumps(
                ReviewAnalyzer(args.artifacts, args.cache_dir).analyze(
                    args.text, args.model
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
