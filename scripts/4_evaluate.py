"""
4_evaluate.py

Evaluates trained models on the held-out Test dataset and generates performance reports.

Models evaluated:
    1. Baseline (Random Forest): quick benchmark comparison
    2. Main Model (PyTorch Attention-LSTM): primary action recognition model

Metrics computed:
    - Top-1 Accuracy: percentage of correctly classified samples
    - Macro Precision: average precision across all 3 classes (treats each equally)
    - Macro Recall: average recall across all 3 classes
    - Macro F1-Score: harmonic mean of precision and recall
    - Per-class Classification Report: precision/recall/F1 for each individual action
    - Inference Latency: average milliseconds per sample prediction
    - FPS Throughput: sequences classified per second

Output:
    - Console: detailed classification report and metrics summary
    - runs/confusion_matrix.png: heatmap visualization of prediction vs. ground truth

Usage:
    python scripts/4_evaluate.py
"""

import argparse
import joblib
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)

try:
    from model import SportActionLSTM
except ImportError:
    from scripts.model import SportActionLSTM


def evaluate_models(processed_dir: Path, models_dir: Path, runs_dir: Path):
    """
    Run full evaluation of both trained models on the test set.

    Loads the test data (X_test, y_test), the label encoder, and both trained
    models. Computes classification metrics and generates a confusion matrix plot.

    The confusion matrix shows how often each true action is predicted as each
    other action — diagonal entries are correct predictions, off-diagonal entries
    are misclassifications.

    Args:
        processed_dir: directory containing X_test.npy and y_test.npy
        models_dir:    directory containing trained model files
        runs_dir:      directory to save evaluation plots
    """
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Load test data and label encoder
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    encoder = joblib.load(models_dir / "label_encoder.pkl")
    class_names = list(encoder.classes_)  # e.g., ["jumping_jacks", "pushups", "running"]

    print("==================================================")
    print("       SPORT ACTIVITY ACTION RECOGNITION          ")
    print("              MODEL EVALUATION REPORT             ")
    print("==================================================")
    print(f"Test dataset size: {X_test.shape[0]} sequence samples")
    print(f"Features dimension: {X_test.shape[2]}")
    print(f"Action classes: {class_names}\n")

    # ── 1. Baseline Random Forest Evaluation ──
    rf_path = models_dir / "baseline_rf.pkl"
    if rf_path.exists():
        rf = joblib.load(rf_path)

        # Pool sequences the same way as during training: mean + std → 478-dim vector
        X_test_pooled = np.hstack([X_test.mean(axis=1), X_test.std(axis=1)])

        # Measure inference time
        start_time = time.time()
        rf_preds = rf.predict(X_test_pooled)
        rf_time = (time.time() - start_time) * 1000 / len(X_test)  # ms per sample

        rf_acc = accuracy_score(y_test, rf_preds)
        print("--- Baseline Model (Random Forest) ---")
        print(f"Top-1 Accuracy: {rf_acc*100:.2f}%")
        rf_fps = 1000.0 / rf_time if rf_time > 1e-6 else 9999.0
        print(f"Avg Inference Latency: {rf_time:.3f} ms per sequence ({rf_fps:.1f} FPS)")
        print(classification_report(y_test, rf_preds, target_names=class_names))
        print("-" * 50)

    # ── 2. Main Model PyTorch Attention-LSTM Evaluation ──
    lstm_path = models_dir / "best_lstm_model.pth"
    if not lstm_path.exists():
        print(f"Model checkpoint not found at '{lstm_path}'. Run 3_train.py first.")
        return

    # Load model from saved checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(lstm_path, map_location=device)
    model = SportActionLSTM(
        input_dim=checkpoint['input_dim'],
        hidden_dim=128,
        num_classes=checkpoint['num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Disable dropout for deterministic evaluation

    # Convert test data to PyTorch tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Run inference and measure latency
    start_time = time.time()
    with torch.no_grad():  # No gradients needed for evaluation
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Predicted class indices
    total_time = (time.time() - start_time) * 1000  # Total time in ms
    avg_latency = total_time / len(X_test)           # Per-sample latency
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # Compute classification metrics
    lstm_acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average='macro')  # Macro = unweighted mean across classes

    print("\n--- Main Model (PyTorch Attention-LSTM) ---")
    print(f"Top-1 Accuracy:  {lstm_acc*100:.2f}%")
    print(f"Macro Precision: {precision*100:.2f}%")
    print(f"Macro Recall:    {recall*100:.2f}%")
    print(f"Macro F1-Score:  {f1*100:.2f}%")
    print(f"Latency:         {avg_latency:.3f} ms / sample sequence")
    print(f"Throughput:      {fps:.1f} FPS (Sequence Predictions/sec)")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, preds, target_names=class_names))

    # ── Confusion Matrix Visualization ──
    # Rows = true action, Columns = predicted action
    # Perfect model → all values on diagonal
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Sport Activity Action Recognition - Confusion Matrix")
    plt.ylabel("True Action")
    plt.xlabel("Predicted Action")
    plt.tight_layout()

    cm_plot_path = runs_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix visual saved to '{cm_plot_path}'\n")

    # ── Save Confusion Matrix & Breakdown JSON ──
    classes_list = [str(c) for c in class_names]
    cm_list = cm.tolist()
    breakdown = []
    for i, true_cls in enumerate(classes_list):
        row_total = int(sum(cm_list[i]))
        correct_count = int(cm_list[i][i])
        details = []
        for j, pred_cls in enumerate(classes_list):
            cnt = int(cm_list[i][j])
            is_correct = (i == j)
            pred_name_clean = pred_cls.replace("_", " ")
            if is_correct:
                text = f"{cnt}: samples were correctly predicted as {pred_name_clean}."
            else:
                text = f"{cnt}: samples were incorrectly predicted as {pred_name_clean}."
            details.append({
                "predicted_class": pred_cls,
                "count": cnt,
                "is_correct": is_correct,
                "text": text
            })
        breakdown.append({
            "class_name": true_cls,
            "index": i,
            "total_samples": row_total,
            "correct_samples": correct_count,
            "details": details
        })

    cm_data = {
        "classes": classes_list,
        "matrix": cm_list,
        "total_test_samples": int(len(y_test)),
        "accuracy": float(lstm_acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "breakdown": breakdown
    }

    cm_json_path = runs_dir / "confusion_matrix.json"
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(cm_data, f, indent=2)
    print(f"[+] Confusion matrix data saved to '{cm_json_path}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on the test set")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"),
                        help="Processed dataset directory")
    parser.add_argument("--models", type=Path, default=Path("models"),
                        help="Models directory")
    parser.add_argument("--runs", type=Path, default=Path("runs"),
                        help="Output directory for plots")
    args = parser.parse_args()
    evaluate_models(args.processed, args.models, args.runs)
