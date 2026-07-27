import argparse
from pathlib import Path
import cv2


def extract_frames(input_dir: Path, output_dir: Path, fps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = sorted(input_dir.glob("*.mp4")) + sorted(input_dir.glob("*.avi"))
    if not video_paths:
        raise FileNotFoundError(f"No video files found in {input_dir}")

    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping unreadable video: {video_path}")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(round(video_fps / fps)))
        frame_index = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_step == 0:
                out_path = output_dir / f"{video_path.stem}_{saved_count:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved_count += 1
            frame_index += 1

        cap.release()
        print(f"Extracted {saved_count} frames from {video_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from raw videos")
    parser.add_argument("--input", type=Path, default=Path("data/raw_videos"), help="Folder with raw .mp4/.avi videos")
    parser.add_argument("--output", type=Path, default=Path("data/extracted_frames"), help="Folder to save extracted frames")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract")
    args = parser.parse_args()
    extract_frames(args.input, args.output, args.fps)
