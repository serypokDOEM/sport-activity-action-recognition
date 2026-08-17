"""
5_detect_video.py

Runs sliding-window skeletal pose action recognition on a video file.

Processing pipeline per frame:
    1. Read frame from input video
    2. Detect 33 body landmarks via MediaPipe Pose
    3. Extract 239-dim scale-invariant features
    4. Append features to a rolling buffer of 30 frames
    5. When buffer is full, run LSTM model inference → action prediction
    6. Apply Exponential Moving Average (EMA) smoothing to prevent flickering
    7. Draw skeleton overlay and prediction banner on the frame
    8. Write annotated frame to output video (H.264 if ffmpeg available, else mp4v)

EMA Smoothing:
    Raw frame-by-frame predictions can flicker between classes due to noisy poses.
    EMA smoothing blends the current prediction probabilities with previous ones:
        smoothed = α × current + (1 - α) × previous
    With α=0.3, the prediction responds to genuine action changes within ~5-10 frames
    while filtering out single-frame noise.

Supported input formats: .mp4, .avi, .mov, .mkv, .webm, .flv

Usage:
    python scripts/5_detect_video.py --input path/to/video.mp4 --output runs/output.mp4
"""

import argparse
import collections
import importlib
import joblib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import torch

# MediaPipe for real-time pose detection
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# imageio-ffmpeg provides H.264 encoding for browser-compatible output videos
try:
    import imageio_ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False

# Add project root to Python path so we can import from scripts/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import feature extraction function from the pose sequence extraction module
extract_pose_seq = importlib.import_module("scripts.1b_extract_pose_sequences")
extract_features_from_landmarks = extract_pose_seq.extract_features_from_landmarks
from scripts.model import SportActionLSTM


def process_video(video_path: Path, output_path: Path, weights_path: Path,
                  encoder_path: Path, seq_len: int = 30, ema_alpha: float = 0.3):
    """
    Process a video file with sport action recognition and save annotated output.

    Args:
        video_path:   path to input video file
        output_path:  path for output video with pose overlay and prediction banner
        weights_path: path to trained PyTorch model checkpoint (.pth)
        encoder_path: path to label encoder pickle file (.pkl)
        seq_len:      number of frames in the sliding window (default 30)
        ema_alpha:    EMA smoothing factor (0-1). Higher = more responsive,
                      lower = smoother predictions (default 0.3)
    """
    if not MP_AVAILABLE:
        print("Error: MediaPipe is required. Please install with `pip install mediapipe`.")
        return

    if not weights_path.exists() or not encoder_path.exists():
        print(f"Error: Model or encoder file missing ({weights_path}, {encoder_path}). "
              f"Run 3_train.py first.")
        return

    # Load label encoder (maps integer predictions back to action names)
    encoder = joblib.load(encoder_path)
    class_names = list(encoder.classes_)

    # Load trained LSTM model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    model = SportActionLSTM(
        input_dim=checkpoint['input_dim'],
        hidden_dim=128,
        num_classes=checkpoint['num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Disable dropout for deterministic inference

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Unable to open input video '{video_path}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Initialize output video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_FFMPEG:
        # H.264 encoding for browser-compatible playback
        ffmpeg_writer = imageio_ffmpeg.write_frames(
            str(output_path), (width, height), fps=fps,
            codec='libx264', pix_fmt_in='rgb24'
        )
        ffmpeg_writer.send(None)  # Initialize the generator
    else:
        # Fallback: mp4v codec (may not play in browsers)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Initialize MediaPipe Pose detector
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Rolling buffer of the last seq_len feature vectors (sliding window)
    buffer = collections.deque(maxlen=seq_len)
    current_action = "Initializing..."
    confidence = 0.0
    smoothed_probs = None  # EMA-smoothed probability distribution
    prev_norm_lms = None   # Previous frame's normalized landmarks (for velocity)

    print(f"Processing '{video_path.name}' -> Saving to '{output_path}'...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Pose Detection ──
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw skeleton overlay on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)
            # Extract 239-dim features from detected landmarks
            feat, norm_lms = extract_features_from_landmarks(
                results.pose_landmarks.landmark, prev_norm_lms)
            prev_norm_lms = norm_lms
        else:
            # No pose detected → use zero vector (model handles gracefully)
            feat = np.zeros(checkpoint['input_dim'], dtype=np.float32)

        # Add current frame features to sliding window buffer
        buffer.append(feat)

        # ── Prediction (once buffer has seq_len frames) ──
        if len(buffer) == seq_len:
            seq_tensor = torch.tensor(
                np.array(buffer), dtype=torch.float32
            ).unsqueeze(0).to(device)  # Shape: (1, 30, 239)

            with torch.no_grad():
                logits = model(seq_tensor)
                raw_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                # ── EMA Temporal Smoothing ──
                # Blend current probabilities with accumulated history
                if smoothed_probs is None:
                    smoothed_probs = raw_probs
                else:
                    smoothed_probs = (ema_alpha * raw_probs +
                                      (1.0 - ema_alpha) * smoothed_probs)

                pred_idx = np.argmax(smoothed_probs)
                confidence = float(smoothed_probs[pred_idx])
                current_action = class_names[pred_idx]

        # ── Draw Prediction Overlay Banner ──
        cv2.rectangle(frame, (10, 10), (450, 85), (0, 0, 0), -1)       # Black background
        cv2.rectangle(frame, (10, 10), (450, 85), (0, 255, 128), 2)    # Green border
        cv2.putText(frame, f"Action: {current_action}", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Write annotated frame to output video
        if HAS_FFMPEG:
            rgb_out_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ffmpeg_writer.send(rgb_out_frame)
        else:
            out.write(frame)

        frame_idx += 1

    # Cleanup resources
    cap.release()
    if HAS_FFMPEG:
        ffmpeg_writer.close()
    else:
        out.release()
    pose.close()
    print(f"[+] Processed {frame_idx} frames. Output saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Sport Action Recognition on a video file")
    parser.add_argument("--input", type=Path,
                        default=Path("data/raw_videos/running/running_demo.mp4"),
                        help="Input video file path")
    parser.add_argument("--output", type=Path,
                        default=Path("runs/output_detection.mp4"),
                        help="Output video file path")
    parser.add_argument("--weights", type=Path,
                        default=Path("models/best_lstm_model.pth"),
                        help="Trained PyTorch model weights")
    parser.add_argument("--encoder", type=Path,
                        default=Path("models/label_encoder.pkl"),
                        help="Label encoder pkl file")
    args = parser.parse_args()
    process_video(args.input, args.output, args.weights, args.encoder)
