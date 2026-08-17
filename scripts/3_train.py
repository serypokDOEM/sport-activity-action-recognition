"""
3_train.py

Trains two models on preprocessed pose sequences for sport action recognition:

1. Baseline Model (Random Forest):
   - Pools each 30-frame sequence into a single feature vector using mean + std
   - Quick to train, serves as accuracy benchmark for the neural model

2. Main Model (PyTorch Bidirectional LSTM with Temporal Attention):
   - Processes raw 30-frame sequences directly (no pooling)
   - Learns temporal motion patterns via bidirectional LSTM
   - Applies learned attention weights to focus on key action frames
   - Uses data augmentation, cosine LR scheduling, and best-model checkpointing

Output:
    - models/baseline_rf.pkl          (trained Random Forest classifier)
    - models/best_lstm_model.pth      (best PyTorch model checkpoint)
    - runs/training_metrics.json      (epoch-by-epoch loss/accuracy history)
    - runs/training_curves.png        (loss + accuracy plots over epochs)

Usage:
    python scripts/3_train.py
    python scripts/3_train.py --epochs 50 --batch 32 --lr 0.0005
"""

import argparse
import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from model import PoseDataset, SportActionLSTM


def apply_data_augmentation(X_batch):
    """
    Apply real-time data augmentation to a batch of pose sequences during training.

    Two augmentation strategies are used:
        1. Gaussian noise injection (always applied):
           Adds small random perturbations to all features, simulating the natural
           jitter of pose estimation in real-world conditions. Standard deviation
           of 0.005 is small enough to preserve action identity.

        2. Random magnitude scaling (30% probability):
           Multiplies all features by a random factor between 0.95-1.05, simulating
           slight variations in body size or camera distance.

    These augmentations improve generalization by preventing the model from
    memorizing exact numeric values of the synthetic training data.

    Args:
        X_batch: tensor of shape (batch_size, seq_len, features) — input sequences

    Returns:
        augmented tensor of the same shape
    """
    # Always add subtle Gaussian noise to simulate pose estimation uncertainty
    noise = torch.randn_like(X_batch) * 0.005
    augmented = X_batch + noise

    # 30% chance: scale all feature magnitudes slightly (simulates size variation)
    if np.random.rand() < 0.3:
        scale = np.random.uniform(0.95, 1.05)
        augmented = augmented * scale

    return augmented


def train_baseline_rf(X_train, y_train, X_val, y_val, models_dir: Path):
    """
    Train a Random Forest classifier as a baseline benchmark.

    Since Random Forest cannot process sequential data directly, we pool each
    30-frame sequence into a single fixed-length vector by computing the mean
    and standard deviation across the time axis. This gives:
        - Mean features: average pose configuration across 30 frames
        - Std features:  how much each feature varies across frames (motion indicator)

    The pooled vector is 239×2 = 478 features per sample.

    Args:
        X_train: numpy array (N_train, 30, 239) — training sequences
        y_train: numpy array (N_train,) — training labels (integer encoded)
        X_val:   numpy array (N_val, 30, 239) — validation sequences
        y_val:   numpy array (N_val,) — validation labels
        models_dir: directory to save the trained model
    """
    print("\n--- Training Baseline Model (Random Forest) ---")

    # Pool: (N, 30, 239) → mean(N, 239) + std(N, 239) → concat(N, 478)
    X_train_pooled = np.hstack([X_train.mean(axis=1), X_train.std(axis=1)])
    X_val_pooled = np.hstack([X_val.mean(axis=1), X_val.std(axis=1)])

    # Train Random Forest with 150 trees, max depth 12 to prevent overfitting
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    rf.fit(X_train_pooled, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train_pooled))
    val_acc = accuracy_score(y_val, rf.predict(X_val_pooled))

    print(f"Random Forest - Train Accuracy: {train_acc*100:.2f}% | Val Accuracy: {val_acc*100:.2f}%")
    joblib.dump(rf, models_dir / "baseline_rf.pkl")
    print(f"  [+] Baseline model saved to '{models_dir / 'baseline_rf.pkl'}'")


def train_lstm_model(X_train, y_train, X_val, y_val, models_dir: Path,
                     runs_dir: Path, epochs: int = 35, batch_size: int = 16,
                     lr: float = 0.001):
    """
    Train the main Bidirectional LSTM model with Temporal Attention.

    Training strategy:
        - Optimizer: AdamW (Adam with decoupled weight decay for better regularization)
        - LR Schedule: Cosine Annealing (smoothly reduces LR from initial to near-zero)
        - Augmentation: Gaussian noise + random scaling applied to each batch
        - Checkpointing: saves the model with the highest validation accuracy

    Training loop per epoch:
        1. Forward pass: sequence → LSTM → attention → FC → logits
        2. Loss: CrossEntropyLoss (multi-class classification loss)
        3. Backward pass: compute gradients via backpropagation
        4. Optimizer step: update model weights
        5. Validation: evaluate on held-out data (no augmentation, no gradients)

    Args:
        X_train, y_train: training data and labels
        X_val, y_val:     validation data and labels
        models_dir:       directory to save model checkpoints
        runs_dir:         directory to save training curves and metrics
        epochs:           number of training epochs (default 35)
        batch_size:       samples per gradient update (default 16)
        lr:               initial learning rate (default 0.001)
    """
    print("\n--- Training Main Model (PyTorch Attention-LSTM) ---")
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of output classes from the saved label encoder
    encoder_path = models_dir / "label_encoder.pkl"
    if encoder_path.exists():
        encoder = joblib.load(encoder_path)
        num_classes = len(encoder.classes_)
    else:
        num_classes = len(np.unique(np.concatenate([y_train, y_val])))

    input_dim = X_train.shape[2]  # 239 features per frame

    # ── Create PyTorch DataLoaders ──
    train_ds = PoseDataset(X_train, y_train)
    val_ds = PoseDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # ── Initialize Model, Loss, Optimizer, Scheduler ──
    model = SportActionLSTM(
        input_dim=input_dim, hidden_dim=128, num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()  # Standard multi-class classification loss

    # AdamW: Adam optimizer with decoupled weight decay (prevents overfitting)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine Annealing: smoothly decays LR from `lr` to near 0 over `epochs`
    # This helps the model converge to a better minimum in later epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Track metrics for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_model_path = models_dir / "best_lstm_model.pth"

    # ── Training Loop ──
    for epoch in range(1, epochs + 1):
        model.train()  # Enable dropout and batch norm training behavior
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            # Apply data augmentation (noise + random scaling)
            batch_X = apply_data_augmentation(batch_X)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()          # Clear previous gradients
            outputs = model(batch_X)       # Forward pass: (batch, 30, 239) → (batch, 3)
            loss = criterion(outputs, batch_y)  # Compute CrossEntropy loss
            loss.backward()                # Backpropagation: compute gradients
            optimizer.step()               # Update model weights

            # Accumulate batch metrics
            running_loss += loss.item() * batch_X.size(0)
            preds = torch.argmax(outputs, dim=1)  # Predicted class index
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        scheduler.step()  # Update learning rate according to cosine schedule
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # ── Validation Evaluation (no augmentation, no gradient computation) ──
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():  # Disable gradient tracking for efficiency
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_loss = val_loss / total_val
        val_acc = correct_val / total_val

        # Record metrics for plotting
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress every 5 epochs, on last epoch, or when val accuracy improves
        if epoch % 5 == 0 or epoch == epochs or val_acc > best_val_acc:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # ── Best Model Checkpointing ──
        # Save the model weights whenever validation accuracy improves
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'num_classes': num_classes,
                'val_acc': val_acc
            }
            torch.save(checkpoint, best_model_path)

    print(f"\n[+] Best PyTorch Attention-LSTM saved to '{best_model_path}' "
          f"with Val Accuracy: {best_val_acc*100:.2f}%")

    # Save training history as JSON for later analysis
    with open(runs_dir / "training_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Plot Training Curves ──
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'],
             label='Train Loss', color='#4F46E5', linewidth=2)
    plt.plot(range(1, epochs + 1), history['val_loss'],
             label='Val Loss', color='#EF4444', linewidth=2, linestyle='--')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a*100 for a in history['train_acc']],
             label='Train Accuracy', color='#10B981', linewidth=2)
    plt.plot(range(1, epochs + 1), [a*100 for a in history['val_acc']],
             label='Val Accuracy', color='#F59E0B', linewidth=2, linestyle='--')
    plt.title('Training & Validation Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    curves_path = runs_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"[+] Training curves saved to '{curves_path}'")


def main(processed_dir: Path, models_dir: Path, runs_dir: Path,
         epochs: int, batch: int, lr: float):
    """
    Load the split dataset and train both models.

    Args:
        processed_dir: directory containing train/val numpy files
        models_dir:    directory to save trained models
        runs_dir:      directory to save training plots and metrics
        epochs:        number of training epochs
        batch:         batch size for LSTM training
        lr:            learning rate for LSTM training
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed train/val splits from 2_split_dataset.py
    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")

    print(f"Loaded dataset: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Loaded dataset: X_val={X_val.shape}, y_val={y_val.shape}")

    # Train both models sequentially
    train_baseline_rf(X_train, y_train, X_val, y_val, models_dir)
    train_lstm_model(X_train, y_train, X_val, y_val, models_dir, runs_dir,
                     epochs=epochs, batch_size=batch, lr=lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Baseline and PyTorch Attention-LSTM models")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"),
                        help="Path to processed dataset")
    parser.add_argument("--models", type=Path, default=Path("models"),
                        help="Path to models folder")
    parser.add_argument("--runs", type=Path, default=Path("runs"),
                        help="Path to runs folder for plots")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    args = parser.parse_args()
    main(args.processed, args.models, args.runs, args.epochs, args.batch, args.lr)
