"""
1b_extract_pose_sequences.py

Extracts scale-invariant skeletal landmark sequence features from real video files
located in data/raw_videos/{action_name}/ directories.

This script processes actual video recordings (unlike 0_generate_sample_dataset.py which
creates synthetic data). It uses MediaPipe Pose to detect 33 body landmarks in each
video frame, then computes the same 239-dimensional feature vector used by the model:
    - 132 hip-centered, torso-normalized landmark coordinates
    - 8 key joint angles
    - 99 inter-frame velocity features

Supported video formats: .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv, .m4v

Output:
    - data/processed/X_sequences.npy  (merged with any existing synthetic data)
    - data/processed/y_sequences.npy
    - data/processed/class_names.json

Usage:
    python scripts/1b_extract_pose_sequences.py
    python scripts/1b_extract_pose_sequences.py --input data/raw_videos --output data/processed --seq_len 30
"""

import os
import argparse
import json
import math
import warnings
from pathlib import Path
import cv2
import numpy as np

# Suppress noisy MediaPipe & TensorFlow C++ log messages
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

# MediaPipe is optional — script gracefully reports if missing
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# All video container formats we can read via OpenCV
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def calculate_angle(a, b, c):
    """
    Calculate the 2D angle (in degrees) at joint 'b' formed by points a-b-c.

    Used to compute joint angles (elbows, knees, hips, shoulders) that serve
    as discriminative pose features for distinguishing sport actions.

    Args:
        a: numpy array [x, y, ...] — first endpoint
        b: numpy array [x, y, ...] — vertex joint (angle is measured here)
        c: numpy array [x, y, ...] — second endpoint

    Returns:
        float: angle in degrees [0°, 180°]. Returns 0.0 if either vector has zero length.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def extract_features_from_landmarks(landmarks, prev_landmarks=None):
    """
    Convert 33 MediaPipe pose landmarks into a scale-invariant 1D feature array.

    Produces exactly 239 features per frame:
        1. Hip-centered & torso-scale normalized coordinates: 33 × 4 = 132 features
           (x, y, z, visibility for each of the 33 body landmarks)
        2. 8 key joint angles (normalized to [0, 1]): 8 features
           (left/right elbows, knees, hips, shoulders)
        3. Inter-frame velocity vectors: 33 × 3 = 99 features
           (change in x, y, z from previous frame for motion dynamics)

    Why 239 features?
        - The coordinates capture the body's spatial pose
        - The angles capture joint configurations (bent vs straight)
        - The velocities capture motion dynamics (speed of movement)
        Together, these allow the LSTM to distinguish actions with similar poses
        but different motion patterns (e.g., standing still vs running).

    Args:
        landmarks:      list of 33 MediaPipe landmark objects (from pose.process())
        prev_landmarks: numpy array (33, 4) from previous frame (for velocity calc)

    Returns:
        tuple: (feature_vector [239,], normalized_landmarks [33, 4])
               The normalized_landmarks are returned for velocity computation in the next frame.
    """
    # Convert MediaPipe landmark objects to numpy array
    lm_coords = []
    for lm in landmarks:
        lm_coords.append([lm.x, lm.y, lm.z, lm.visibility])
    lm_coords = np.array(lm_coords, dtype=np.float32)

    # ── Step 1: Position Normalization ──
    # Center all landmarks on the mid-hip point (average of left hip 23 and right hip 24)
    hip_center = (lm_coords[23, :3] + lm_coords[24, :3]) / 2.0
    lm_coords[:, :3] -= hip_center  # Remove absolute position dependency

    # ── Step 2: Scale Normalization ──
    # Normalize by torso height (shoulder 11 to hip 23 distance)
    # This makes the features invariant to how far the person is from the camera
    torso_size = np.linalg.norm(lm_coords[11, :3] - lm_coords[23, :3])
    if torso_size < 1e-4:
        torso_size = 1.0  # Prevent division by zero for degenerate poses
    lm_coords[:, :3] /= torso_size

    # Flatten to 1D: 33 landmarks × 4 values = 132 features
    flat_coords = lm_coords.flatten()

    # ── Step 3: 8 Key Joint Angles ──
    ang_l_elbow    = calculate_angle(lm_coords[11], lm_coords[13], lm_coords[15])
    ang_r_elbow    = calculate_angle(lm_coords[12], lm_coords[14], lm_coords[16])
    ang_l_knee     = calculate_angle(lm_coords[23], lm_coords[25], lm_coords[27])
    ang_r_knee     = calculate_angle(lm_coords[24], lm_coords[26], lm_coords[28])
    ang_l_hip      = calculate_angle(lm_coords[11], lm_coords[23], lm_coords[25])
    ang_r_hip      = calculate_angle(lm_coords[12], lm_coords[24], lm_coords[26])
    ang_l_shoulder = calculate_angle(lm_coords[13], lm_coords[11], lm_coords[23])
    ang_r_shoulder = calculate_angle(lm_coords[14], lm_coords[12], lm_coords[24])

    # Normalize angles from [0°, 180°] → [0.0, 1.0]
    angles = np.array([
        ang_l_elbow, ang_r_elbow, ang_l_knee, ang_r_knee,
        ang_l_hip, ang_r_hip, ang_l_shoulder, ang_r_shoulder
    ], dtype=np.float32) / 180.0

    # ── Step 4: Velocity Features ──
    # Compute per-landmark movement between consecutive frames
    if prev_landmarks is not None:
        velocity = (lm_coords[:, :3] - prev_landmarks[:, :3]).flatten()  # 99 features
    else:
        velocity = np.zeros(33 * 3, dtype=np.float32)  # No velocity for first frame

    # Return the 239-dim feature vector and normalized landmarks for next frame's velocity
    return np.concatenate([flat_coords, angles, velocity]), lm_coords


def process_video_file(video_path: Path, pose, seq_len: int = 30):
    """
    Extract all fixed-length sequence windows from a single video file.

    Uses a sliding window approach to extract multiple training samples from
    one video. If the video is shorter than seq_len frames, it is padded by
    repeating the last frame.

    Args:
        video_path: path to the video file
        pose:       MediaPipe Pose detector instance
        seq_len:    number of frames per sequence window (default 30)

    Returns:
        list of numpy arrays, each shape (seq_len, 239) — one per sequence window
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return []

    frame_features = []
    prev_norm_lms = None
    last_valid_feat = None

    # ── Frame-by-frame pose extraction ──
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB input (OpenCV reads BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Pose detected → compute 239-dim features
            feat, norm_lms = extract_features_from_landmarks(
                results.pose_landmarks.landmark, prev_norm_lms)
            prev_norm_lms = norm_lms
            last_valid_feat = feat
        elif last_valid_feat is not None:
            # No pose detected → reuse last valid features (temporal smoothing)
            feat = last_valid_feat.copy()
        else:
            # No pose ever detected yet → zero vector placeholder
            feat = np.zeros(239, dtype=np.float32)

        frame_features.append(feat)

    cap.release()

    # ── Create fixed-length sequence windows via sliding window ──
    sequences = []
    num_frames = len(frame_features)

    if num_frames < seq_len:
        # Short video: pad by repeating the last frame to reach seq_len
        if num_frames > 0:
            padded = frame_features + [frame_features[-1]] * (seq_len - num_frames)
            sequences.append(np.array(padded, dtype=np.float32))
    else:
        # Sliding window with step = max(2, seq_len // 6) ≈ 5 frames
        # This creates overlapping windows for more training data from one video
        step = max(2, seq_len // 6)
        for start in range(0, num_frames - seq_len + 1, step):
            seq = frame_features[start:start + seq_len]
            sequences.append(np.array(seq, dtype=np.float32))

    return sequences


def find_all_video_files(folder_path: Path):
    """
    Recursively find all video files with supported extensions in a directory.

    Args:
        folder_path: root directory to search

    Returns:
        sorted list of Path objects for each video file found
    """
    video_paths = []
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            video_paths.append(file_path)
    return sorted(video_paths)


def main(input_dir: Path, output_dir: Path, seq_len: int = 30):
    """
    Main entry point: extract pose sequences from all videos in input_dir.

    Expects directory structure:
        input_dir/
            running/
                video1.mp4
                video2.avi
            pushups/
                ...
            jumping_jacks/
                ...

    The action label for each video is inferred from its parent directory name.

    If existing X_sequences.npy and y_sequences.npy files exist in output_dir
    (e.g., from 0_generate_sample_dataset.py), the new real-video sequences are
    merged with the existing synthetic data.

    Args:
        input_dir:  path to raw_videos directory (default: data/raw_videos)
        output_dir: path to save output numpy files (default: data/processed)
        seq_len:    frames per sequence window (default: 30)
    """
    if not MP_AVAILABLE:
        print("MediaPipe not installed. Run `pip install mediapipe`.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Each subdirectory name = action class label
    action_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not action_dirs:
        print(f"No action subdirectories found in {input_dir}. "
              f"Expected data/raw_videos/{{action_name}}/")
        return

    # Initialize MediaPipe Pose detector
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,        # Video mode (uses temporal tracking)
        model_complexity=1,             # Balanced accuracy/speed
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    all_sequences = []
    all_labels = []

    for action_dir in action_dirs:
        action = action_dir.name  # Directory name = action label
        video_paths = find_all_video_files(action_dir)

        print(f"Processing action '{action}' ({len(video_paths)} video files)...")
        action_seq_count = 0

        for vid_path in video_paths:
            seqs = process_video_file(vid_path, pose, seq_len)
            for s in seqs:
                all_sequences.append(s)
                all_labels.append(action)
                action_seq_count += 1

        print(f"  [+] Extracted {action_seq_count} sequences for '{action}'")

    pose.close()

    if all_sequences:
        X_new = np.array(all_sequences, dtype=np.float32)
        y_new = np.array(all_labels)

        # ── Merge with existing data (e.g., synthetic samples from step 0) ──
        x_existing_path = output_dir / "X_sequences.npy"
        y_existing_path = output_dir / "y_sequences.npy"

        if x_existing_path.exists() and y_existing_path.exists():
            try:
                X_old = np.load(x_existing_path)
                y_old = np.load(y_existing_path)
                # Only merge if feature dimensions match (both should be 239)
                if X_old.shape[1:] == X_new.shape[1:]:
                    X = np.concatenate([X_old, X_new], axis=0)
                    y = np.concatenate([y_old, y_new], axis=0)
                else:
                    X, y = X_new, y_new
            except Exception:
                X, y = X_new, y_new
        else:
            X, y = X_new, y_new

        # Save merged dataset
        np.save(output_dir / "X_sequences.npy", X)
        np.save(output_dir / "y_sequences.npy", y)

        # Update class names list
        with open(output_dir / "class_names.json", "w") as f:
            json.dump(sorted(list(set(y))), f, indent=2)

        print(f"\nPose extraction complete. Saved to '{output_dir}':")
        print(f"  - X shape: {X.shape} (Sequences, TimeSteps, Features)")
        print(f"  - y shape: {y.shape}")
        print(f"  - Features dimension: {X.shape[2]} (Scale-Invariant + Angles + Velocities)")
    else:
        print("No valid sequences extracted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract scale-invariant pose landmark sequences from videos")
    parser.add_argument("--input", type=Path, default=Path("data/raw_videos"),
                        help="Folder containing action subdirectories with raw videos")
    parser.add_argument("--output", type=Path, default=Path("data/processed"),
                        help="Folder to save extracted numpy sequence datasets")
    parser.add_argument("--seq_len", type=int, default=30,
                        help="Frames per sequence window")
    args = parser.parse_args()
    main(args.input, args.output, args.seq_len)
