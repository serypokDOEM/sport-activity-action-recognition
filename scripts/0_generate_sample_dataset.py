"""
0_generate_sample_dataset.py

Generates synthetic video files (.mp4 and .avi) and scale-invariant keypoint
landmark sequences for 3 sport activity actions:
    1. running
    2. pushups
    3. jumping_jacks

Purpose:
    When no real video dataset is available, this script creates mathematically-
    simulated skeleton pose data that mimics each sport action's characteristic
    body motion patterns. This allows the downstream training pipeline to run
    end-to-end without requiring real video collection.

Output:
    - data/raw_videos/{action}/{action}_demo.mp4  (synthetic stick-figure demo video)
    - data/raw_videos/{action}/{action}_demo.avi  (same demo in AVI container)
    - data/processed/X_sequences.npy              (feature tensor: [N, 30, 239])
    - data/processed/y_sequences.npy              (label array: [N,] string labels)
    - data/processed/class_names.json             (list of 3 action class names)
"""

import json
import math
import os
from pathlib import Path
import cv2
import numpy as np

# Suppress MediaPipe & TensorFlow C++ GLOG messages to keep console output clean
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ──────────────────────────────────────────────────────────────────────
# The 3 sport actions this project recognizes
# ──────────────────────────────────────────────────────────────────────
ACTIONS = [
    "running",
    "pushups",
    "jumping_jacks",
]

# Number of synthetic training samples generated per action class
NUM_SAMPLES_PER_ACTION = 100

# Each sample is a fixed-length sequence of 30 frames (approx. 1 second at 30 FPS)
SEQUENCE_LENGTH = 30


def calculate_angle(a, b, c):
    """
    Calculate the 2D angle (in degrees) at joint 'b' formed by points a-b-c.

    This is used to compute key body joint angles (elbows, knees, hips, shoulders)
    which serve as discriminative features for action classification.

    Args:
        a: numpy array [x, y, ...] — first endpoint (e.g., shoulder)
        b: numpy array [x, y, ...] — vertex joint (e.g., elbow)
        c: numpy array [x, y, ...] — second endpoint (e.g., wrist)

    Returns:
        float: angle in degrees [0°, 180°]. Returns 0.0 if either vector has zero length.

    Example:
        A straight arm gives ~180°, a fully bent elbow gives ~30-60°.
    """
    # Vector from b→a and b→c
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Avoid division by zero when joints overlap
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    # Cosine of the angle via dot product formula: cos(θ) = (ba · bc) / (|ba| × |bc|)
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)

    # np.clip prevents floating-point errors outside [-1, 1] from crashing np.arccos
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def generate_pose_frame(action: str, frame_idx: int, seq_len: int = 30):
    """
    Generate one synthetic pose skeleton frame for a given action.

    Simulates the characteristic body motion pattern of each sport action by
    mathematically varying 33 MediaPipe-format body landmark coordinates over time.

    Each landmark has 4 values: [x, y, z, visibility] where x,y,z ∈ [0,1]
    are normalized coordinates and visibility ∈ [0,1] is detection confidence.

    Key MediaPipe landmark indices used:
        0  = nose/head
        11 = left shoulder,   12 = right shoulder
        13 = left elbow,      14 = right elbow
        15 = left wrist,      16 = right wrist
        23 = left hip,        24 = right hip
        25 = left knee,       26 = right knee
        27 = left ankle,      28 = right ankle

    Args:
        action:    one of "running", "pushups", "jumping_jacks"
        frame_idx: current frame number in the sequence (0-based)
        seq_len:   total frames per cycle (default 30)

    Returns:
        numpy array of shape (33, 4) — one frame of synthetic pose landmarks
    """
    # 't' goes from 0.0 to 1.0 over one cycle, 'phase' converts it to radians
    t = (frame_idx % seq_len) / float(seq_len)
    phase = 2 * math.pi * t

    # Initialize all 33 landmarks to zero with high visibility
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[:, 3] = 0.95  # Default visibility confidence for all joints

    # Default upright body proportions (normalized y-coordinates, 0=top, 1=bottom)
    head_y = 0.2
    shoulder_y = 0.35
    hip_y = 0.55
    knee_y = 0.75
    ankle_y = 0.9
    cx = 0.5  # Horizontal center of frame

    # Add small Gaussian noise to simulate real-world pose estimation jitter
    noise = np.random.normal(0, 0.005, size=landmarks.shape)

    if action == "running":
        # ── RUNNING: Alternating leg swing + opposite arm swing ──
        # Legs swing forward/backward sinusoidally (180° out of phase)
        # Arms swing in opposition to legs for balance (cos vs sin)
        leg_swing = math.sin(phase) * 0.15   # Leg forward/backward displacement
        arm_swing = math.cos(phase) * 0.15   # Arm swing (opposite phase to legs)

        landmarks[0]  = [cx + leg_swing * 0.2, head_y, 0, 0.99]                  # Head bobs slightly
        landmarks[11] = [cx - 0.08 + arm_swing * 0.2, shoulder_y, 0, 0.95]       # Left shoulder
        landmarks[12] = [cx + 0.08 - arm_swing * 0.2, shoulder_y, 0, 0.95]       # Right shoulder
        landmarks[13] = [cx - 0.10 - arm_swing, shoulder_y + 0.1, 0, 0.95]       # Left elbow swings back
        landmarks[14] = [cx + 0.10 + arm_swing, shoulder_y + 0.1, 0, 0.95]       # Right elbow swings forward
        landmarks[15] = [cx - 0.12 - arm_swing * 1.2, shoulder_y + 0.2, 0, 0.95] # Left wrist (amplified swing)
        landmarks[16] = [cx + 0.12 + arm_swing * 1.2, shoulder_y + 0.2, 0, 0.95] # Right wrist
        landmarks[23] = [cx - 0.06, hip_y, 0, 0.95]                              # Left hip (stable)
        landmarks[24] = [cx + 0.06, hip_y, 0, 0.95]                              # Right hip (stable)
        landmarks[25] = [cx - 0.06 + leg_swing, knee_y, 0, 0.95]                 # Left knee drives forward
        landmarks[26] = [cx + 0.06 - leg_swing, knee_y, 0, 0.95]                 # Right knee (opposite phase)
        landmarks[27] = [cx - 0.06 + leg_swing * 1.3, ankle_y, 0, 0.95]          # Left ankle (amplified stride)
        landmarks[28] = [cx + 0.06 - leg_swing * 1.3, ankle_y, 0, 0.95]          # Right ankle

    elif action == "pushups":
        # ── PUSHUPS: Horizontal body, arms push torso up/down ──
        # The body is oriented horizontally (head left, feet right).
        # The vertical position oscillates to simulate the push-up motion.
        push_y = 0.65 + math.sin(phase) * 0.15  # Body height oscillates (down → up → down)

        landmarks[0]  = [0.2, push_y - 0.1, 0, 0.99]    # Head (leftmost, slightly above torso)
        landmarks[11] = [0.35, push_y, 0, 0.95]          # Left shoulder
        landmarks[12] = [0.35, push_y, 0, 0.95]          # Right shoulder (overlapping — side view)
        landmarks[13] = [0.35, push_y + 0.15, 0, 0.95]   # Left elbow bends downward
        landmarks[14] = [0.35, push_y + 0.15, 0, 0.95]   # Right elbow
        landmarks[15] = [0.35, push_y + 0.25, 0, 0.95]   # Left wrist on ground
        landmarks[16] = [0.35, push_y + 0.25, 0, 0.95]   # Right wrist on ground
        landmarks[23] = [0.55, push_y, 0, 0.95]           # Left hip (body midpoint)
        landmarks[24] = [0.55, push_y, 0, 0.95]           # Right hip
        landmarks[25] = [0.70, push_y, 0, 0.95]           # Left knee
        landmarks[26] = [0.70, push_y, 0, 0.95]           # Right knee
        landmarks[27] = [0.85, push_y, 0, 0.95]           # Left ankle (rightmost)
        landmarks[28] = [0.85, push_y, 0, 0.95]           # Right ankle

    elif action == "jumping_jacks":
        # ── JUMPING JACKS: Arms and legs spread symmetrically outward/inward ──
        # Both arms rise laterally while both legs spread sideways, then return.
        jack_spread = math.sin(phase) * 0.25  # Spread amount (0 = together, 0.25 = max spread)

        landmarks[0]  = [cx, head_y, 0, 0.99]                                            # Head stays centered
        landmarks[11] = [cx - 0.08, shoulder_y, 0, 0.95]                                 # Left shoulder
        landmarks[12] = [cx + 0.08, shoulder_y, 0, 0.95]                                 # Right shoulder
        landmarks[13] = [cx - 0.10 - jack_spread, shoulder_y - jack_spread, 0, 0.95]      # Left elbow rises outward
        landmarks[14] = [cx + 0.10 + jack_spread, shoulder_y - jack_spread, 0, 0.95]      # Right elbow rises outward
        landmarks[15] = [cx - 0.12 - jack_spread * 1.2, shoulder_y - jack_spread * 1.4, 0, 0.95]  # Left wrist (above head at max)
        landmarks[16] = [cx + 0.12 + jack_spread * 1.2, shoulder_y - jack_spread * 1.4, 0, 0.95]  # Right wrist
        landmarks[23] = [cx - 0.06, hip_y, 0, 0.95]                                      # Left hip
        landmarks[24] = [cx + 0.06, hip_y, 0, 0.95]                                      # Right hip
        landmarks[25] = [cx - 0.06 - jack_spread * 0.8, knee_y, 0, 0.95]                  # Left knee spreads outward
        landmarks[26] = [cx + 0.06 + jack_spread * 0.8, knee_y, 0, 0.95]                  # Right knee spreads outward
        landmarks[27] = [cx - 0.06 - jack_spread, ankle_y, 0, 0.95]                       # Left ankle (max spread)
        landmarks[28] = [cx + 0.06 + jack_spread, ankle_y, 0, 0.95]                       # Right ankle

    # Add sensor noise and clip all coordinates to valid [0, 1] range
    landmarks = landmarks + noise
    landmarks[:, :3] = np.clip(landmarks[:, :3], 0.0, 1.0)
    return landmarks


def compute_sequence_features(landmarks_seq):
    """
    Convert a sequence of raw pose landmarks into a scale-invariant feature tensor.

    For each frame, computes 239 features:
        1. Hip-centered & torso-normalized coordinates: 33 landmarks × 4 values = 132 features
        2. 8 key joint angles (normalized to [0,1]): 8 features
        3. Inter-frame velocity vectors (motion dynamics): 33 landmarks × 3 axes = 99 features
        Total: 132 + 8 + 99 = 239 features per frame

    The normalization makes features invariant to:
        - Camera distance (scale normalization via torso height)
        - Person position in frame (translation normalization via hip centering)

    Args:
        landmarks_seq: list of numpy arrays, each shape (33, 4) — raw landmark frames

    Returns:
        numpy array of shape (seq_len, 239) — the feature tensor for one sequence sample
    """
    seq_features = []
    prev_coords = None

    for frame_lm in landmarks_seq:
        lm_coords = frame_lm.copy()

        # ── Step 1: Position Normalization (hip-centering) ──
        # Compute mid-hip point as body center reference
        hip_center = (lm_coords[23, :3] + lm_coords[24, :3]) / 2.0
        # Shift all landmarks so the hip center is at origin (0,0,0)
        # This removes the person's absolute position in the frame
        lm_coords[:, :3] -= hip_center

        # ── Step 2: Scale Normalization (torso-height scaling) ──
        # Distance from left shoulder (11) to left hip (23) = torso height
        torso_size = np.linalg.norm(lm_coords[11, :3] - lm_coords[23, :3])
        if torso_size < 1e-4:
            torso_size = 1.0  # Prevent division by zero
        # Divide all coordinates by torso size → zoom/distance invariant
        lm_coords[:, :3] /= torso_size

        # Flatten all 33 landmarks × 4 values into a 1D vector (132 features)
        flat_coords = lm_coords.flatten()

        # ── Step 3: Joint Angle Features ──
        # These 8 angles capture the body's pose configuration at this frame
        ang_l_elbow    = calculate_angle(lm_coords[11], lm_coords[13], lm_coords[15])  # Left elbow bend
        ang_r_elbow    = calculate_angle(lm_coords[12], lm_coords[14], lm_coords[16])  # Right elbow bend
        ang_l_knee     = calculate_angle(lm_coords[23], lm_coords[25], lm_coords[27])  # Left knee bend
        ang_r_knee     = calculate_angle(lm_coords[24], lm_coords[26], lm_coords[28])  # Right knee bend
        ang_l_hip      = calculate_angle(lm_coords[11], lm_coords[23], lm_coords[25])  # Left hip angle
        ang_r_hip      = calculate_angle(lm_coords[12], lm_coords[24], lm_coords[26])  # Right hip angle
        ang_l_shoulder = calculate_angle(lm_coords[13], lm_coords[11], lm_coords[23])  # Left shoulder angle
        ang_r_shoulder = calculate_angle(lm_coords[14], lm_coords[12], lm_coords[24])  # Right shoulder angle

        # Normalize angles from [0°, 180°] → [0.0, 1.0]
        angles = np.array([
            ang_l_elbow, ang_r_elbow, ang_l_knee, ang_r_knee,
            ang_l_hip, ang_r_hip, ang_l_shoulder, ang_r_shoulder
        ], dtype=np.float32) / 180.0

        # ── Step 4: Velocity Features (inter-frame motion) ──
        # Difference in xyz coordinates between current and previous frame
        # Captures the speed and direction of body part movements
        if prev_coords is not None:
            velocity = (lm_coords[:, :3] - prev_coords[:, :3]).flatten()  # 99 features
        else:
            velocity = np.zeros(33 * 3, dtype=np.float32)  # First frame has no velocity
        prev_coords = lm_coords

        # ── Concatenate all features for this frame: 132 + 8 + 99 = 239 ──
        frame_feat = np.concatenate([flat_coords, angles, velocity])
        seq_features.append(frame_feat)

    return np.array(seq_features, dtype=np.float32)


def generate_sample_videos_and_dataset():
    """
    Main entry point: generates the complete synthetic dataset.

    For each of the 3 sport actions:
        1. Creates a demo video (.mp4 + .avi) with stick-figure animation
        2. Generates NUM_SAMPLES_PER_ACTION synthetic pose sequences
        3. Computes 239-dim scale-invariant features for each sequence

    All data is saved to data/processed/ as numpy arrays for use by
    the downstream training pipeline (2_split_dataset.py → 3_train.py).
    """
    raw_video_dir = Path("data/raw_videos")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic sample videos and sequence dataset for {len(ACTIONS)} sport actions...")

    all_sequences = []  # Will hold all feature tensors
    all_labels = []     # Will hold corresponding action labels

    for action in ACTIONS:
        # Create directory structure: data/raw_videos/{action}/
        action_video_dir = raw_video_dir / action
        action_video_dir.mkdir(parents=True, exist_ok=True)

        # ── Generate demo videos (stick-figure animation) ──
        mp4_path = action_video_dir / f"{action}_demo.mp4"
        avi_path = action_video_dir / f"{action}_demo.avi"

        # Initialize video writers for both formats
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        fourcc_avi = cv2.VideoWriter_fourcc(*'MJPG')  # AVI codec (Motion JPEG)

        out_mp4 = cv2.VideoWriter(str(mp4_path), fourcc_mp4, 30.0, (640, 480))
        out_avi = cv2.VideoWriter(str(avi_path), fourcc_avi, 30.0, (640, 480))

        # Write 60 frames (2 seconds at 30 FPS) of stick-figure animation
        for f_idx in range(60):
            # Dark background frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8) + 30
            lms = generate_pose_frame(action, f_idx, 30)

            # Skeleton connections (bone lines between landmark pairs)
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),  # Torso + left leg
                (24, 26), (26, 28)                                  # Right leg
            ]

            # Convert normalized [0,1] coordinates to pixel coordinates
            pts = [(int(lm[0] * 640), int(lm[1] * 480)) for lm in lms]

            # Draw skeleton bones (green lines)
            for p1, p2 in connections:
                cv2.line(frame, pts[p1], pts[p2], (0, 255, 128), 2)

            # Draw joint dots (orange circles)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 165, 255), -1)

            # Label the action in the top-left corner
            cv2.putText(frame, f"Action: {action}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out_mp4.write(frame)
            out_avi.write(frame)

        out_mp4.release()
        out_avi.release()

        # ── Generate synthetic training data sequences ──
        for sample_i in range(NUM_SAMPLES_PER_ACTION):
            # Generate raw pose landmarks for SEQUENCE_LENGTH frames
            seq_lms = [generate_pose_frame(action, f, SEQUENCE_LENGTH)
                       for f in range(SEQUENCE_LENGTH)]
            # Compute 239-dim feature vector per frame
            features = compute_sequence_features(seq_lms)
            all_sequences.append(features)
            all_labels.append(action)

        print(f"  [+] Created {NUM_SAMPLES_PER_ACTION} sequences & demo videos for '{action}'")

    # ── Save dataset to disk ──
    X = np.array(all_sequences, dtype=np.float32)  # Shape: (300, 30, 239)
    y = np.array(all_labels)                        # Shape: (300,)

    np.save(processed_dir / "X_sequences.npy", X)
    np.save(processed_dir / "y_sequences.npy", y)

    # Save action class names for label encoding later
    with open(processed_dir / "class_names.json", "w") as f:
        json.dump(ACTIONS, f, indent=2)

    print(f"\nDataset successfully saved to '{processed_dir}':")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - Classes ({len(ACTIONS)}): {ACTIONS}")


if __name__ == "__main__":
    generate_sample_videos_and_dataset()
