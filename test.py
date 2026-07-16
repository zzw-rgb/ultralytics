# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    """使用训练好的模型进行实时摄像头检测。"""
    model_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/my_dataset/nailong_yolo26n_full/weights/best.pt"
    camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    results = YOLO(model_path).predict(source=camera_id, conf=0.25, show=True, stream=True)
    for result in results:
        print(f"检测到 {len(result.boxes)} 个目标")


if __name__ == "__main__":
    main()
