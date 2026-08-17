"""
6_detect_webcam.py

Real-time live sport activity recognition using webcam or camera feed URL.

Connects to a webcam device (default: device 0) or an RTSP/HTTP stream URL,
detects body pose in real-time using MediaPipe, and predicts the current
sport action using the trained Attention-LSTM model.

Features:
    - Horizontal frame flip for natural mirror-view webcam display
    - MediaPipe skeletal landmark overlay on the live video
    - EMA temporal smoothing for stable, flicker-free action predictions
    - Prediction banner showing current action + confidence percentage
    - Press 'q' or ESC to exit the live feed

Processing pipeline per frame:
    1. Capture frame from webcam
    2. Flip horizontally (mirror view)
    3. Detect 33 body landmarks via MediaPipe Pose
    4. Extract 239-dim scale-invariant features (with velocity from previous frame)
    5. Append to 30-frame sliding window buffer
    6. When buffer is full, run LSTM inference → action prediction
    7. Apply EMA smoothing → update displayed action label
    8. Draw skeleton + prediction banner overlay
    9. Display in OpenCV window

Usage:
    python scripts/6_detect_webcam.py
    python scripts/6_detect_webcam.py --source 1           # Use camera device 1
    python scripts/6_detect_webcam.py --source rtsp://...  # Use network camera
"""

import argparse
import collections
import importlib
import joblib
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# MediaPipe for real-time pose detection
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# Add project root to Python path for cross-module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import feature extraction function and model architecture
extract_pose_seq = importlib.import_module("scripts.1b_extract_pose_sequences")
extract_features_from_landmarks = extract_pose_seq.extract_features_from_landmarks
from scripts.model import SportActionLSTM


def run_webcam(device_source: str, weights_path: Path, encoder_path: Path,
               seq_len: int = 30, ema_alpha: float = 0.3):
    """
    Run real-time sport action recognition on a live webcam feed.

    Args:
        device_source: webcam device index ("0", "1") or stream URL string
        weights_path:  path to trained PyTorch model checkpoint (.pth)
        encoder_path:  path to label encoder pickle file (.pkl)
        seq_len:       number of frames in the sliding window (default 30)
        ema_alpha:     EMA smoothing factor (0-1). Controls prediction stability.
                       0.3 = smooth (less flicker), 0.7 = responsive (faster changes)
    """
    if not MP_AVAILABLE:
        print("Error: MediaPipe required. Install with `pip install mediapipe`.")
        return

    if not weights_path.exists() or not encoder_path.exists():
        print(f"Error: Missing model ({weights_path}) or encoder ({encoder_path}). "
              f"Run 3_train.py first.")
        return

    # Convert string to integer for device index (e.g., "0" → 0)
    source = int(device_source) if device_source.isdigit() else device_source

    # Load label encoder and trained model
    encoder = joblib.load(encoder_path)
    class_names = list(encoder.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    model = SportActionLSTM(
        input_dim=checkpoint['input_dim'],
        hidden_dim=128,
        num_classes=checkpoint['num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Disable dropout for inference

    # Open webcam/camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open webcam/camera source '{device_source}'")
        return

    # Initialize MediaPipe Pose detector
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Sliding window buffer for 30 consecutive feature vectors
    buffer = collections.deque(maxlen=seq_len)
    current_action = "Buffering..."  # Displayed until buffer fills to 30 frames
    confidence = 0.0
    smoothed_probs = None   # EMA-smoothed probability distribution
    prev_norm_lms = None    # Previous frame's normalized landmarks for velocity

    print(f"Starting Live Webcam Recognition on source {device_source} "
          f"(Press 'q' to quit)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror view (left/right matches user)
        frame = cv2.flip(frame, 1)

        # ── Pose Detection ──
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw skeleton overlay on frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)
            # Extract 239-dim features with velocity tracking from previous frame
            feat, norm_lms = extract_features_from_landmarks(
                results.pose_landmarks.landmark, prev_norm_lms)
            prev_norm_lms = norm_lms
        else:
            # No person detected → use zero vector
            feat = np.zeros(checkpoint['input_dim'], dtype=np.float32)

        buffer.append(feat)

        # ── Action Prediction (once buffer has 30 frames) ──
        if len(buffer) == seq_len:
            seq_tensor = torch.tensor(
                np.array(buffer), dtype=torch.float32
            ).unsqueeze(0).to(device)  # Shape: (1, 30, 239)

            with torch.no_grad():
                logits = model(seq_tensor)
                raw_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                # ── EMA Temporal Smoothing ──
                # Prevents rapid flickering between predicted actions
                if smoothed_probs is None:
                    smoothed_probs = raw_probs
                else:
                    smoothed_probs = (ema_alpha * raw_probs +
                                      (1.0 - ema_alpha) * smoothed_probs)

                pred_idx = np.argmax(smoothed_probs)
                confidence = float(smoothed_probs[pred_idx])
                current_action = class_names[pred_idx]

        # ── Draw Prediction Overlay Banner ──
        cv2.rectangle(frame, (10, 10), (460, 90), (0, 0, 0), -1)       # Black bg
        cv2.rectangle(frame, (10, 10), (460, 90), (0, 255, 128), 2)    # Green border
        cv2.putText(frame, f"Action: {current_action}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (25, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Sport Activity Action Recognition - Live Feed", frame)

        # Exit on 'q' key or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run real-time webcam sport activity action recognition")
    parser.add_argument("--source", type=str, default="0",
                        help="Webcam device index or stream URL")
    parser.add_argument("--weights", type=Path,
                        default=Path("models/best_lstm_model.pth"),
                        help="Path to PyTorch model weights")
    parser.add_argument("--encoder", type=Path,
                        default=Path("models/label_encoder.pkl"),
                        help="Path to label encoder pkl file")
    args = parser.parse_args()
    run_webcam(args.source, args.weights, args.encoder)
