# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Capture dataset images from a local camera."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preview a camera and save images for a YOLO dataset.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0 for the first USB camera.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/my_dataset/images/raw",
        help="Directory in which captured JPG images are saved.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested camera frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera frame height.")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between images in automatic mode.")
    parser.add_argument("--quality", type=int, default=95, choices=range(1, 101), metavar="1-100", help="JPG quality.")
    return parser.parse_args()


def save_image(frame, output_dir: Path, sequence: int, quality: int) -> Path:
    """Save one camera frame and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    image_path = output_dir / f"capture_{timestamp}_{sequence:04d}.jpg"
    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"无法保存图片：{image_path}")
    return image_path


def main() -> None:
    """Open the camera preview and handle manual or timed image capture."""
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval 必须大于 0")

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}，请检查 /dev/video* 和摄像头占用情况")

    saved = 0
    auto_capture = False
    last_capture_time = 0.0
    window_name = "YOLO Dataset Capture"
    print(f"图片保存目录：{output_dir}")
    print("空格/S：拍照，A：开启或关闭自动拍照，Q/ESC：退出")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("摄像头已打开，但无法读取画面")

            now = time.monotonic()
            if auto_capture and now - last_capture_time >= args.interval:
                saved += 1
                image_path = save_image(frame, output_dir, saved, args.quality)
                last_capture_time = now
                print(f"已保存：{image_path.name}")

            status = f"Saved: {saved} | Auto: {'ON' if auto_capture else 'OFF'}"
            controls = "SPACE/S: save | A: auto | Q/ESC: quit"
            cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, controls, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S"), ord(" ")):
                saved += 1
                image_path = save_image(frame, output_dir, saved, args.quality)
                print(f"已保存：{image_path.name}")
            elif key in (ord("a"), ord("A")):
                auto_capture = not auto_capture
                last_capture_time = 0.0
                print(f"自动拍照：{'开启' if auto_capture else '关闭'}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"拍照结束，共保存 {saved} 张图片到：{output_dir}")


if __name__ == "__main__":
    main()
