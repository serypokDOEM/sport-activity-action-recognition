import argparse
from pathlib import Path
from ultralytics import YOLO


def detect_video(weights: Path, source_video: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    model(source=str(source_video), save=True, project=str(output_dir), name="detect_video", exist_ok=True)
    print(f"Saved detection results to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO detection on a video file")
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"), help="YOLO weights file")
    parser.add_argument("--source", type=Path, default=Path("data/raw_videos"), help="Path to input video file")
    parser.add_argument("--output", type=Path, default=Path("runs"), help="Directory to save output video")
    args = parser.parse_args()
    detect_video(args.weights, args.source, args.output)
