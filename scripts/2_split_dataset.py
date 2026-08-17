"""
2_split_dataset.py

Splits extracted sequence features (X_sequences.npy, y_sequences.npy) into
Train (70%), Validation (15%), and Test (15%) subsets.

Also encodes string action labels (e.g., "running") into integer indices
and saves the label encoder mapping to models/label_encoder.pkl for use
during inference.

Why stratified splitting?
    Stratified splitting ensures that each subset (train/val/test) has the
    same proportion of each action class as the full dataset. This prevents
    the model from being evaluated on a skewed distribution.

Two-phase split strategy:
    Phase 1: Full dataset → Train (70%) + Temp (30%)
    Phase 2: Temp (30%) → Validation (15%) + Test (15%)
    This is done because sklearn's train_test_split only supports binary splits.

Output:
    - data/processed/X_train.npy, y_train.npy  (70% of data)
    - data/processed/X_val.npy, y_val.npy      (15% of data)
    - data/processed/X_test.npy, y_test.npy    (15% of data)
    - models/label_encoder.pkl                  (maps string labels ↔ integer indices)
"""

import argparse
import joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def split_pose_dataset(processed_dir: Path, models_dir: Path,
                       train_ratio: float = 0.70, val_ratio: float = 0.15,
                       seed: int = 42):
    """
    Split the full dataset into train/validation/test subsets with stratification.

    Args:
        processed_dir: directory containing X_sequences.npy and y_sequences.npy
        models_dir:    directory to save the label encoder pickle file
        train_ratio:   fraction of data for training (default 0.70 = 70%)
        val_ratio:     fraction of data for validation (default 0.15 = 15%)
                       (test ratio is implicitly 1.0 - train_ratio - val_ratio = 0.15)
        seed:          random seed for reproducibility (default 42)
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    x_path = processed_dir / "X_sequences.npy"
    y_path = processed_dir / "y_sequences.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing sequence files in {processed_dir}. "
            f"Run 0_generate_sample_dataset.py or 1b_extract_pose_sequences.py first.")

    X = np.load(x_path)        # Shape: (N, 30, 239) — feature sequences
    y_str = np.load(y_path)    # Shape: (N,) — string labels like "running"

    # ── Label Encoding ──
    # Convert string labels to integer indices: "jumping_jacks"→0, "pushups"→1, "running"→2
    # The encoder is saved and reused during inference to decode predictions back to strings
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_str)

    # Save label encoder for use by training (3_train.py) and inference scripts
    joblib.dump(encoder, models_dir / "label_encoder.pkl")
    print(f"Encoded {len(encoder.classes_)} classes: {list(encoder.classes_)}")

    # ── Stratification Safety Check ──
    # Stratified split requires at least 2-3 samples per class in each subset.
    # If any class has fewer than 3 samples, disable stratification to avoid errors.
    _, counts = np.unique(y_encoded, return_counts=True)
    min_count = np.min(counts)
    stratify_full = y_encoded if min_count >= 3 else None

    # ── Phase 1: Split into Train (70%) + Temp (30%) ──
    test_val_ratio = 1.0 - train_ratio  # 0.30
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded,
        test_size=test_val_ratio,   # 30% goes to temp
        random_state=seed,
        stratify=stratify_full       # Maintain class proportions
    )

    # ── Phase 2: Split Temp (30%) into Validation (15%) + Test (15%) ──
    # val_subset_ratio = 0.15 / 0.30 = 0.50 → half of temp goes to val
    val_subset_ratio = val_ratio / test_val_ratio

    _, temp_counts = np.unique(y_temp, return_counts=True)
    stratify_temp = y_temp if np.min(temp_counts) >= 2 else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_subset_ratio),  # 50% of temp → test
        random_state=seed,
        stratify=stratify_temp
    )

    # ── Save all splits as numpy files ──
    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_val.npy", X_val)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_test.npy", y_test)

    print("\nDataset split complete:")
    print(f"  - Train samples: {X_train.shape[0]} ({X_train.shape})")
    print(f"  - Val samples:   {X_val.shape[0]} ({X_val.shape})")
    print(f"  - Test samples:  {X_test.shape[0]} ({X_test.shape})")
    print(f"  - Label encoder saved to '{models_dir / 'label_encoder.pkl'}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split extracted pose sequences into train/val/test datasets")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"),
                        help="Processed dataset folder containing X_sequences.npy and y_sequences.npy")
    parser.add_argument("--models", type=Path, default=Path("models"),
                        help="Models folder to save label encoder")
    parser.add_argument("--train_ratio", type=float, default=0.70,
                        help="Ratio for training set (default: 70%%)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Ratio for validation set (default: 15%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    args = parser.parse_args()
    split_pose_dataset(args.processed, args.models, args.train_ratio,
                       args.val_ratio, args.seed)
