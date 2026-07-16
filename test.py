# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys

from ultralytics import YOLO


def main():
    """加载训练好的模型并保存可视化检测结果。"""
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/train/weights/best.pt"
    source = sys.argv[2] if len(sys.argv) > 2 else "test.jpg"

    results = YOLO(model_path).predict(source=source, conf=0.25, save=True)
    for result in results:
        print(f"{result.path}: 检测到 {len(result.boxes)} 个目标")


if __name__ == "__main__":
    main()
