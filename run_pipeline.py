"""
run_pipeline.py

Master runner script that executes the complete end-to-end pipeline for
the Sport Activity Action Recognition project.

Pipeline Steps:
    Step 0: Generate synthetic training data (3 sport actions × 100 samples each)
    Step 1: Extract pose sequences from real videos (if any exist in data/raw_videos/)
    Step 2: Split dataset into Train (70%) / Validation (15%) / Test (15%)
    Step 3: Train both models (Random Forest baseline + Attention-LSTM)
    Step 4: Evaluate on test set + generate confusion matrix and training curves
    Step 5: Run video detection test on a demo video
    Step 6: Launch interactive Gradio web interface (optional, skippable)

Usage:
    python run_pipeline.py
    python run_pipeline.py --epochs 50
    python run_pipeline.py --skip-webui
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_command_step(step_name: str, command_args: list):
    """
    Execute a Python script as a subprocess with logging.

    If the subprocess fails (non-zero exit code), the entire pipeline is halted
    because downstream steps depend on the output of previous steps.

    Args:
        step_name:    human-readable description of the step
        command_args: list of command-line arguments (script path + flags)
    """
    print("\n" + "=" * 65)
    print(f"  -> RUNNING: {step_name}")
    print("=" * 65)

    full_cmd = [sys.executable] + command_args
    print(f"Executing: {' '.join(full_cmd)}\n")

    result = subprocess.run(full_cmd)
    if result.returncode != 0:
        print(f"\n[!] ERROR: Step '{step_name}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    else:
        print(f"\n[+] SUCCESS: Completed '{step_name}'.")


def main():
    """
    Execute the full sport action recognition pipeline end-to-end.

    Each step produces output files that the next step consumes:
        Step 0 → data/processed/X_sequences.npy, y_sequences.npy
        Step 1 → merges real video data into the same .npy files
        Step 2 → data/processed/X_train.npy, X_val.npy, X_test.npy + models/label_encoder.pkl
        Step 3 → models/best_lstm_model.pth, models/baseline_rf.pkl
        Step 4 → runs/confusion_matrix.png, runs/training_curves.png
        Step 5 → runs/pipeline_test_output.mp4
    """
    parser = argparse.ArgumentParser(
        description="Master Pipeline Runner for Sport Action Recognition")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Training epochs for PyTorch Attention-LSTM")
    parser.add_argument("--skip-webui", action="store_true",
                        help="Skip launching the Gradio web interface at the end")
    args = parser.parse_args()

    print("==================================================")
    print("       SPORT ACTIVITY ACTION RECOGNITION          ")
    print("              FULL PROCESS FLOW                   ")
    print("==================================================")

    # Step 0: Generate synthetic training data for 3 actions
    run_command_step(
        "Step 0: Generate Synthetic Demo Datasets & Videos",
        ["scripts/0_generate_sample_dataset.py"]
    )

    # Step 1: Extract pose sequences from real videos (if any exist)
    run_command_step(
        "Step 1: Extract MediaPipe Pose Sequences from Real Videos",
        ["scripts/1b_extract_pose_sequences.py"]
    )

    # Step 2: Split dataset into train/val/test (70/15/15)
    run_command_step(
        "Step 2: Split Dataset into Train / Val / Test",
        ["scripts/2_split_dataset.py"]
    )

    # Step 3: Train both models (Random Forest + Attention-LSTM)
    run_command_step(
        "Step 3: Train Attention-LSTM & Baseline Models",
        ["scripts/3_train.py", "--epochs", str(args.epochs)]
    )

    # Step 4: Evaluate on test set and generate metrics plots
    run_command_step(
        "Step 4: Evaluate Test Set & Generate Metrics Plots",
        ["scripts/4_evaluate.py"]
    )

    # Step 5: Run video detection on a demo video (if it exists)
    demo_video = Path("data/raw_videos/running/running_demo.mp4")
    if demo_video.exists():
        run_command_step(
            "Step 5: Run Detection Test on Running Demo Video",
            [
                "scripts/5_detect_video.py",
                "--input", str(demo_video),
                "--output", "runs/pipeline_test_output.mp4"
            ]
        )

    print("\n" + "*" * 65)
    print(" SUCCESS: FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("*" * 65)

    # Step 6: Launch Gradio web interface (optional)
    if not args.skip_webui:
        print("\nStarting Interactive Gradio Web Interface (app.py)...")
        subprocess.run([sys.executable, "app.py"])


if __name__ == "__main__":
    main()
