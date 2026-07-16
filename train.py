# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train a YOLO26 object detector on the local Nailong dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO26 on the local Nailong detection dataset.")
    parser.add_argument("--model", default=str(ROOT / "yolo26n.pt"), help="Initial model weights or model YAML.")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset.yaml", help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Images per training batch.")
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu.")
    parser.add_argument("--workers", type=int, default=2, help="Number of data-loading workers.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "my_dataset", help="Training output directory.")
    parser.add_argument("--name", default="yolo26n", help="Experiment name.")
    parser.add_argument("--patience", type=int, default=0, help="Early-stopping patience; 0 disables early stopping.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the experiment directory if it already exists.")
    return parser.parse_args()


def main() -> None:
    """Train the model and print the best checkpoint path."""
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data.expanduser().resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        exist_ok=args.exist_ok,
    )
    best_model = Path(model.trainer.best)
    print(f"训练完成，最佳模型：{best_model}")


if __name__ == "__main__":
    main()
