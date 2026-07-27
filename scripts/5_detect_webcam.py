import argparse
from pathlib import Path
from ultralytics import YOLO


def detect_webcam(weights: Path, source: str):
    model = YOLO(str(weights))
    model(source=source, stream=True, show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO detection on a webcam or RTSP stream")
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"), help="YOLO weights file")
    parser.add_argument("--source", type=str, default="0", help="Webcam device index or RTSP stream URL")
    args = parser.parse_args()
    detect_webcam(args.weights, args.source)
