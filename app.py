"""
app.py

Interactive Gradio Web Application for Sport Activity Action Recognition.

Provides a browser-based interface where users can:
    1. Upload a sports video file (.mp4, .avi, .mov, .mkv, .webm, .flv)
    2. Run the trained Attention-LSTM model for action recognition
    3. View the annotated output video with pose skeleton overlay
    4. See action confidence distribution across the 3 classes
    5. View model performance metrics (training curves, confusion matrix)

Sport Actions Recognized:
    1. Running
    2. Pushups
    3. Jumping Jacks

Technical Architecture:
    - MediaPipe Pose extracts 33 body landmarks per frame
    - 239-dim scale-invariant features (hip-centered coords + angles + velocities)
    - Bidirectional LSTM with Temporal Attention processes 30-frame sliding window
    - EMA temporal smoothing prevents prediction flickering

Usage:
    python app.py
    Then open http://127.0.0.1:7860 in your browser
"""

import os
import asyncio
import collections
from datetime import datetime
import importlib
import json
import joblib
import subprocess
import sys
import warnings
from pathlib import Path

# Suppress noisy MediaPipe, TensorFlow & ABSL C++ log messages
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

# Fix Windows asyncio ProactorEventLoop WinError 10054 ConnectionResetError on socket closure
if sys.platform.startswith("win"):
    import logging
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        from functools import wraps

        _orig_call_conn_lost = _ProactorBasePipeTransport._call_connection_lost

        @wraps(_orig_call_conn_lost)
        def _silenced_call_conn_lost(self, *args, **kwargs):
            try:
                return _orig_call_conn_lost(self, *args, **kwargs)
            except (ConnectionResetError, OSError):
                pass

        _ProactorBasePipeTransport._call_connection_lost = _silenced_call_conn_lost
    except Exception:
        pass

import cv2
import gradio as gr
import numpy as np
import torch

# MediaPipe for real-time pose detection
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# imageio-ffmpeg for H.264 browser-compatible video encoding
try:
    import imageio_ffmpeg
    HAS_IMAGEIO_FFMPEG = True
except ImportError:
    HAS_IMAGEIO_FFMPEG = False

# Add project root to Python path for cross-module imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import feature extraction function and model architecture
extract_pose_seq = importlib.import_module("scripts.1b_extract_pose_sequences")
extract_features_from_landmarks = extract_pose_seq.extract_features_from_landmarks
from scripts.model import SportActionLSTM

# File paths for trained model and evaluation outputs
WEIGHTS_PATH = Path("models/best_lstm_model.pth")
ENCODER_PATH = Path("models/label_encoder.pkl")
CONFUSION_MATRIX_PATH = Path("runs/confusion_matrix.png")
CONFUSION_MATRIX_JSON_PATH = Path("runs/confusion_matrix.json")
TRAINING_CURVES_PATH = Path("runs/training_curves.png")


def load_model_and_encoder():
    """
    Load the trained LSTM model and label encoder from disk.

    Returns:
        tuple: (model, encoder, class_names, device) or (None, None, None, None) if missing
    """
    if not WEIGHTS_PATH.exists() or not ENCODER_PATH.exists():
        return None, None, None, None

    encoder = joblib.load(ENCODER_PATH)
    class_names = list(encoder.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model = SportActionLSTM(
        input_dim=checkpoint['input_dim'],
        hidden_dim=128,
        num_classes=checkpoint['num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, encoder, class_names, device


def generate_confusion_matrix_explanation() -> str:
    """
    Generate dynamic HTML explaining what each row and column
    in the confusion matrix means using actual test evaluation numbers.
    """
    data = None
    if CONFUSION_MATRIX_JSON_PATH.exists():
        try:
            with open(CONFUSION_MATRIX_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

    # If JSON not found but test data and model exist, compute dynamically
    if data is None:
        try:
            processed_dir = Path("data/processed")
            models_dir = Path("models")
            if (processed_dir / "X_test.npy").exists() and (models_dir / "best_lstm_model.pth").exists():
                from sklearn.metrics import confusion_matrix
                X_test = np.load(processed_dir / "X_test.npy")
                y_test = np.load(processed_dir / "y_test.npy")
                encoder = joblib.load(models_dir / "label_encoder.pkl")
                classes_list = [str(c) for c in encoder.classes_]

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(models_dir / "best_lstm_model.pth", map_location=device)
                model = SportActionLSTM(
                    input_dim=checkpoint['input_dim'],
                    hidden_dim=128,
                    num_classes=checkpoint['num_classes']
                ).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                with torch.no_grad():
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                    outputs = model(X_test_tensor)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()

                cm = confusion_matrix(y_test, preds)
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
                data = {
                    "classes": classes_list,
                    "matrix": cm_list,
                    "total_test_samples": int(len(y_test)),
                    "accuracy": float(np.mean(preds == y_test)),
                    "breakdown": breakdown
                }
                runs_dir = Path("runs")
                runs_dir.mkdir(parents=True, exist_ok=True)
                with open(CONFUSION_MATRIX_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except Exception:
            data = None

    if not data or "breakdown" not in data:
        return """
<div class='hero-card'>
<b>📖 Understanding the Confusion Matrix</b><br><br>
The confusion matrix is a table that shows how well the model classified each
action on the <b>test set</b> (data the model has never seen during training).<br><br>
<b>How to Read It</b><br>
• <b>Rows (Y-axis)</b> = the <b>True Action</b> (what the person was actually doing)<br>
• <b>Columns (X-axis)</b> = the <b>Predicted Action</b> (what the model predicted)<br>
• <b>Diagonal cells</b> = <b>correct predictions</b> ✅<br>
• <b>Off-diagonal cells</b> = <b>misclassifications</b> ❌<br><br>
<i>Run the evaluation pipeline (<code>python scripts/4_evaluate.py</code>) to generate live per-class breakdown numbers.</i>
</div>
"""

    breakdown = data.get("breakdown", [])
    acc = data.get("accuracy", 0.0) * 100
    total = data.get("total_test_samples", 0)

    items_html = []
    for item in breakdown:
        cls_idx = item.get("index", 0) + 1
        cls_name = item.get("class_name", "")
        tot_cls = item.get("total_samples", 0)

        detail_lines = []
        for d in item.get("details", []):
            cnt = d.get("count", 0)
            is_cor = d.get("is_correct", False)
            pred_cls_clean = d.get("predicted_class", "").replace("_", " ")
            if is_cor:
                detail_lines.append(f"&nbsp;&nbsp;• <b>{cnt}</b>: samples were correctly predicted as {pred_cls_clean}. ✅")
            else:
                mark = " ⚠️" if cnt > 0 else ""
                detail_lines.append(f"&nbsp;&nbsp;• <b>{cnt}</b>: samples were incorrectly predicted as {pred_cls_clean}.{mark}")

        details_str = "<br>\n".join(detail_lines)
        items_html.append(f"""
<div style="background: rgba(15, 23, 42, 0.65); padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 10px;">
<b style="color: #38BDF8; font-size: 1.05rem;">{cls_idx}. {cls_name}</b> <span style="color: #94A3B8; font-size: 0.85rem;">({tot_cls} total test samples)</span><br>
{details_str}
</div>
""")

    breakdown_section = "\n".join(items_html)

    return f"""
<div class='hero-card'>
<b>📖 Understanding the Confusion Matrix</b><br><br>

The confusion matrix is a table that shows how well the model classified each
action on the <b>test set</b> (data the model has never seen during training).<br><br>

<b>How to Read It</b><br>
• <b>Rows (Y-axis)</b> = the <b>True Action</b> (what the person was actually doing)<br>
• <b>Columns (X-axis)</b> = the <b>Predicted Action</b> (what the model predicted)<br>
• <b>Diagonal cells</b> (top-left to bottom-right) = <b>correct predictions</b> ✅<br>
• <b>Off-diagonal cells</b> = <b>misclassifications</b> ❌<br><br>

<b>Action Labels on Each Axis</b><br>
• <b>jumping_jacks</b>: symmetric arm raise + leg spread motion pattern<br>
• <b>pushups</b>: horizontal body with vertical push-up/down arm motion<br>
• <b>running</b>: alternating leg stride + opposite arm swing pattern<br><br>

<b>What the Numbers Mean (Actual Test Evaluation Breakdown)</b><br>
Each row represents the actual ground-truth activity, and each number breaks down the model's predictions across columns:<br><br>
{breakdown_section}
<br>
<b>Summary & Key Takeaway:</b><br>
Overall Test Accuracy: <b>{acc:.2f}%</b> ({sum(x.get('correct_samples', 0) for x in breakdown)} / {total} correct test sequences).<br>
• High diagonal counts indicate high precision and recall across all sports classes.<br>
• Off-diagonal counts show specific edge-case confusions between motions.
</div>
"""


def analyze_video(file_input, video_input=None, ema_alpha=0.3):
    """
    Process an uploaded video file and return annotated output with action predictions.

    This is the main Gradio callback function. It:
        1. Opens the uploaded video
        2. Runs MediaPipe pose detection on each frame
        3. Extracts 239-dim features and feeds them to the LSTM model
        4. Applies EMA smoothing for stable predictions
        5. Draws skeleton overlay and prediction banner on each frame
        6. Writes the annotated video to runs/web_output.mp4
        7. Returns the video, probability distribution, and summary HTML

    Args:
        file_input:  Gradio File component input (uploaded video file)
        video_input: alternative Gradio Video component input (unused, kept for API)
        ema_alpha:   EMA smoothing factor (default 0.3)

    Returns:
        tuple: (output_video_path, probability_dict, summary_html)
    """
    # Retrieve input file path from either File Uploader or Video Uploader
    input_obj = file_input if file_input is not None else video_input

    if input_obj is None:
        return None, {}, "<div class='error-badge'>⚠️ Please select or drop a sports video file (.avi, .mp4, .mov, .mkv).</div>"

    # Extract clean file path string from Gradio input object
    if hasattr(input_obj, "name"):
        video_path = str(input_obj.name)
    elif isinstance(input_obj, dict) and "name" in input_obj:
        video_path = str(input_obj["name"])
    else:
        video_path = str(input_obj)

    if not MP_AVAILABLE:
        return None, {}, "<div class='error-badge'>⚠️ MediaPipe is required. Please run `pip install -r requirements.txt`.</div>"

    model, encoder, class_names, device = load_model_and_encoder()
    if model is None:
        return None, {}, "<div class='error-badge'>⚠️ Trained model missing. Please run `python run_pipeline.py`.</div>"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {}, f"<div class='error-badge'>⚠️ Unable to open video file: {Path(video_path).name}</div>"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Scale to optimal resolution (max 720 width, height capped at 480-540)
    # Divisible by 16 for H.264 macroblock encoding and clean video player display
    if width >= height:
        target_w = 720
        target_h = int(round((height / width) * target_w))
    else:
        target_h = 540
        target_w = int(round((width / height) * target_h))

    out_width = max(16, (target_w // 16) * 16)
    out_height = max(16, (target_h // 16) * 16)

    output_video_path = "runs/web_output.mp4"
    Path("runs").mkdir(parents=True, exist_ok=True)

    # Initialize output video writer
    if HAS_IMAGEIO_FFMPEG:
        ffmpeg_writer = imageio_ffmpeg.write_frames(
            output_video_path, (out_width, out_height), fps=fps,
            codec='libx264', pix_fmt_in='rgb24'
        )
        ffmpeg_writer.send(None)
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Processing state
    buffer = collections.deque(maxlen=30)
    last_probs_dict = {str(c): 0.0 for c in class_names}
    top_action = "Analyzing..."
    top_confidence = 0.0
    smoothed_probs = None
    prev_norm_lms = None
    input_dim = model.lstm.input_size  # 239 features per frame
    frame_count = 0

    # ── Frame-by-frame processing loop ──
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)
            feat, norm_lms = extract_features_from_landmarks(
                results.pose_landmarks.landmark, prev_norm_lms)
            prev_norm_lms = norm_lms
        else:
            feat = np.zeros(input_dim, dtype=np.float32)

        buffer.append(feat)

        if len(buffer) == 30:
            seq_tensor = torch.tensor(
                np.array(buffer), dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(seq_tensor)
                raw_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                # EMA Prediction Smoothing
                if smoothed_probs is None:
                    smoothed_probs = raw_probs
                else:
                    smoothed_probs = (ema_alpha * raw_probs +
                                      (1.0 - ema_alpha) * smoothed_probs)

                pred_idx = np.argmax(smoothed_probs)
                top_action = str(class_names[pred_idx])
                top_confidence = float(smoothed_probs[pred_idx])
                last_probs_dict = {
                    str(class_names[i]): float(smoothed_probs[i])
                    for i in range(len(class_names))
                }

        # Draw prediction overlay banner
        cv2.rectangle(frame, (12, 12), (460, 90), (15, 23, 42), -1)
        cv2.rectangle(frame, (12, 12), (460, 90), (56, 189, 248), 2)
        cv2.putText(frame, f"Action: {top_action}", (25, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {top_confidence*100:.1f}%", (25, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)

        # Resize for macroblock compatibility if needed
        if (frame.shape[1], frame.shape[0]) != (out_width, out_height):
            frame_to_write = cv2.resize(frame, (out_width, out_height))
        else:
            frame_to_write = frame

        if HAS_IMAGEIO_FFMPEG:
            rgb_out_frame = cv2.cvtColor(frame_to_write, cv2.COLOR_BGR2RGB)
            ffmpeg_writer.send(rgb_out_frame)
        else:
            out.write(frame_to_write)

    # Cleanup
    cap.release()
    if HAS_IMAGEIO_FFMPEG:
        ffmpeg_writer.close()
    else:
        out.release()
    pose.close()

    # Build summary HTML card
    summary_html = f"""
    <div class='summary-card'>
        <div class='summary-header'>Primary Detected Action</div>
        <div class='summary-action'>{top_action.upper().replace('_', ' ')}</div>
        <div class='summary-grid'>
            <div class='stat-box'>
                <span class='stat-label'>Confidence Score</span>
                <span class='stat-value'>{top_confidence*100:.1f}%</span>
            </div>
            <div class='stat-box'>
                <span class='stat-label'>Processed Frames</span>
                <span class='stat-value'>{frame_count}</span>
            </div>
            <div class='stat-box'>
                <span class='stat-label'>Input File</span>
                <span class='stat-value'>{Path(video_path).name}</span>
            </div>
            <div class='stat-box'>
                <span class='stat-label'>Output Format</span>
                <span class='stat-value'>H.264 MP4</span>
            </div>
        </div>
    </div>
    """
    return output_video_path, last_probs_dict, summary_html


def run_pipeline_streaming():
    """
    Execute run_pipeline.py as a subprocess and stream both structured table entries
    and detailed console log lines in real-time.

    Yields:
        tuple: (table_rows, raw_log_text, button_update)
    """
    start_time = datetime.now().strftime("%H:%M:%S")
    table_rows = [
        [start_time, "Pipeline Launcher", "🚀 Started", "Initiating run_pipeline.py --skip-webui..."]
    ]
    raw_logs = [
        "=" * 65,
        "  🚀 STARTING FULL PIPELINE: run_pipeline.py --skip-webui",
        "=" * 65,
        ""
    ]
    # Disable button and yield initial state
    yield table_rows, "\n".join(raw_logs), gr.Button(interactive=False)

    current_stage = "Pipeline Setup"

    try:
        process = subprocess.Popen(
            [sys.executable, "run_pipeline.py", "--skip-webui"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        for line in iter(process.stdout.readline, ""):
            stripped = line.rstrip("\n\r")
            if not stripped:
                continue

            raw_logs.append(stripped)
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Parse pipeline stages and milestone outputs into table
            if "-> RUNNING: Step 0" in stripped or "Step 0:" in stripped:
                current_stage = "Step 0: Generate Data"
                status = "⏳ Running"
                table_rows.append([timestamp, current_stage, status, "Generating 300 synthetic pose sequences & demo videos..."])
            elif "-> RUNNING: Step 1" in stripped or "Step 1:" in stripped:
                current_stage = "Step 1: Extract Poses"
                status = "⏳ Running"
                table_rows.append([timestamp, current_stage, status, "Extracting MediaPipe landmark sequences from real videos..."])
            elif "-> RUNNING: Step 2" in stripped or "Step 2:" in stripped:
                current_stage = "Step 2: Split Dataset"
                status = "⏳ Running"
                table_rows.append([timestamp, current_stage, status, "Splitting dataset into Train (70%) / Val (15%) / Test (15%)..."])
            elif "-> RUNNING: Step 3" in stripped or "Step 3:" in stripped:
                current_stage = "Step 3: Train Models"
                status = "⏳ Running"
                table_rows.append([timestamp, current_stage, status, "Training Random Forest baseline and PyTorch Attention-LSTM..."])
            elif "-> RUNNING: Step 4" in stripped or "Step 4:" in stripped:
                current_stage = "Step 4: Model Evaluation"
                status = "⏳ Running"
                table_rows.append([timestamp, current_stage, status, "Evaluating test set performance and generating confusion matrix..."])
            elif "-> RUNNING: Step 5" in stripped or "Step 5:" in stripped:
                current_stage = "Step 5: Video Detection"
                status = "⏳ Running"
                table_rows.append([timestamp, current_stage, status, "Running end-to-end action detection on demo video..."])
            elif "SUCCESS: Completed" in stripped or "[+] SUCCESS" in stripped:
                status = "✅ Completed"
                table_rows.append([timestamp, current_stage, status, stripped.replace("[+] SUCCESS: ", "")])
            elif "Epoch [" in stripped:
                status = "📈 Training Progress"
                table_rows.append([timestamp, current_stage, status, stripped])
            elif "Top-1 Accuracy:" in stripped or "Macro F1-Score:" in stripped or "Dataset split complete:" in stripped or "Dataset successfully saved" in stripped:
                status = "📊 Metric"
                table_rows.append([timestamp, current_stage, status, stripped])
            elif "[!]" in stripped or "ERROR" in stripped:
                status = "❌ Error"
                table_rows.append([timestamp, current_stage, status, stripped])
            elif "[+]" in stripped:
                status = "✨ Info"
                table_rows.append([timestamp, current_stage, status, stripped])

            yield table_rows, "\n".join(raw_logs), gr.Button(interactive=False)

        process.stdout.close()
        exit_code = process.wait()

        timestamp = datetime.now().strftime("%H:%M:%S")
        if exit_code == 0:
            table_rows.append([timestamp, "Pipeline Summary", "🏆 Success", "All steps completed successfully with exit code 0!"])
            raw_logs.append("\n" + "=" * 65)
            raw_logs.append("  ✅ ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
            raw_logs.append("=" * 65)
        else:
            table_rows.append([timestamp, "Pipeline Summary", "❌ Failed", f"Pipeline terminated with error code {exit_code}."])
            raw_logs.append("\n" + "=" * 65)
            raw_logs.append(f"  ❌ PIPELINE FAILED (exit code {exit_code})")
            raw_logs.append("=" * 65)

        yield table_rows, "\n".join(raw_logs), gr.Button(interactive=True)

    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        table_rows.append([timestamp, "Pipeline Error", "❌ Exception", str(e)])
        raw_logs.append(f"\n❌ Error launching pipeline: {str(e)}")
        yield table_rows, "\n".join(raw_logs), gr.Button(interactive=True)


# ── Premium Custom CSS ──
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

body {
    font-family: 'Inter', sans-serif;
    background-color: #0F172A !important;
    color: #F8FAFC !important;
}

.gradio-container {
    max-width: 1350px !important;
    margin: 0 auto !important;
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin-top: 15px;
    margin-bottom: 5px;
    letter-spacing: -0.5px;
}

.sub-header {
    text-align: center;
    color: #94A3B8 !important;
    font-size: 1.1rem !important;
    margin-bottom: 25px;
    font-weight: 500;
}

.hero-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-left: 5px solid #38BDF8;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.hero-card b {
    color: #38BDF8;
}

.step-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
    border: 1px solid rgba(129, 140, 248, 0.2);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 15px;
}

.step-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #38BDF8;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.wh-badge {
    display: inline-block;
    background: rgba(56, 189, 248, 0.15);
    color: #38BDF8;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    border: 1px solid rgba(56, 189, 248, 0.3);
    margin-right: 6px;
}

.wh-badge-why {
    background: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border-color: rgba(245, 158, 11, 0.3);
}

.wh-badge-how {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border-color: rgba(16, 185, 129, 0.3);
}

.summary-card {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 1px solid rgba(129, 140, 248, 0.3);
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    margin-top: 15px;
}

.summary-header {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94A3B8;
    font-weight: 600;
}

.summary-action {
    font-size: 2rem;
    font-weight: 800;
    color: #38BDF8;
    margin: 8px 0 16px 0;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}

.stat-box {
    background: rgba(15, 23, 42, 0.6);
    padding: 12px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stat-label {
    display: block;
    font-size: 0.8rem;
    color: #64748B;
}

.stat-value {
    display: block;
    font-size: 1.1rem;
    font-weight: 700;
    color: #F8FAFC;
}

.error-badge {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid #EF4444;
    color: #FCA5A5;
    padding: 15px;
    border-radius: 10px;
    font-weight: 600;
}

button.primary-btn {
    background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4) !important;
}

button.pipeline-btn {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4) !important;
}

.output-video {
    min-height: 400px !important;
    max-height: 460px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    overflow: hidden !important;
    background: #0B1120 !important;
}

.output-video video {
    max-height: 430px !important;
    width: 100% !important;
    object-fit: contain !important;
    margin: 0 auto !important;
    border-radius: 8px !important;
}
"""

# ── Build Gradio Web Interface ──
with gr.Blocks(title="Sport Activity Action Recognition AI",
               theme=gr.themes.Soft(primary_hue="indigo")) as demo:

    gr.Markdown("<h1 class='main-header'>🏀 Sport Activity Action Recognition AI</h1>")
    gr.Markdown("<p class='sub-header'>Real-Time Skeletal Pose Sequence Recognition "
                "powered by MediaPipe & PyTorch Attention-LSTM<br>"
                "Actions: Running · Pushups · Jumping Jacks</p>")

    with gr.Tabs():
        # ────────────────────────────────────────────────────────────
        # Tab 1: Project Overview (Comprehensive WH-Question Breakdown)
        # ────────────────────────────────────────────────────────────
        with gr.TabItem("📖 Project Overview"):
            gr.Markdown("""
            <div class='hero-card'>
            <h2 style='color: #38BDF8; margin-top: 0;'>🌟 Project Vision & System Architecture</h2>
            <b>What is this project?</b><br>
            An end-to-end Computer Vision & Deep Learning system capable of detecting and classifying human sport activities from raw video feeds into 3 distinct actions: <b>Running</b>, <b>Pushups</b>, and <b>Jumping Jacks</b>.<br><br>

            <b>Why use Skeleton-Based Pose AI instead of 3D CNNs?</b><br>
            • ⚡ <b>High Throughput (>2,500 FPS on CPU)</b>: Processing 3D coordinates requires a fraction of the compute of pixel-level 3D-CNNs (like SlowFast or I3D).<br>
            • 🛡️ <b>Privacy & Clutter Invariance</b>: Strips away background, lighting shifts, skin tones, and clothing colors, learning purely from biomechanical movement.<br>
            • 📐 <b>Scale & Zoom Invariant</b>: All landmark coordinates are centered at the mid-hip point and scaled by the person's torso height, allowing recognition regardless of camera distance.<br><br>

            <b>How does Feature Engineering work? (239 Features per Frame)</b><br>
            • <b>132 Spatial Coordinates</b>: 33 body keypoints × [x, y, z, visibility], centered on hips and normalized by torso length.<br>
            • <b>8 Biomechanical Angles</b>: Left/Right Elbows, Knees, Hips, and Shoulders calculated via vector dot products normalized to [0, 1].<br>
            • <b>99 Temporal Velocities</b>: Inter-frame 3D movement difference (dx, dy, dz) for all 33 keypoints across consecutive frames to capture movement speed and acceleration.
            </div>
            """)

            gr.Markdown("### 🔍 Step-by-Step Pipeline Deep Dive (WH-Questions)")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div class='step-card'>
                    <div class='step-header'>⚙️ Step 0: Generate Synthetic Training Data</div>
                    <p><span class='wh-badge'>WHAT</span> Generates 300 balanced synthetic sequence samples (100 running, 100 pushups, 100 jumping jacks) of 30 frames each, along with animated stick-figure demo videos in both <code>.mp4</code> and <code>.avi</code> containers.</p>
                    <p><span class='wh-badge wh-badge-why'>WHY</span> Enables immediate end-to-end training and testing of the entire machine learning pipeline without requiring manual video collection, costly motion capture equipment, or time-consuming annotation.</p>
                    <p><span class='wh-badge wh-badge-how'>HOW</span> Uses harmonic sinusoidal equations that model human kinematics (alternating leg/arm swing for running, vertical torso oscillation for pushups, and lateral arm/leg abduction for jumping jacks) mixed with Gaussian sensor noise.</p>
                    </div>
                    """)

                    gr.Markdown("""
                    <div class='step-card'>
                    <div class='step-header'>🎥 Step 1: Extract Pose Sequences from Real Videos</div>
                    <p><span class='wh-badge'>WHAT</span> Scans <code>data/raw_videos/{action}/</code> for real video files (<code>.mp4, .avi, .mov, .mkv, .webm, .flv</code>) and extracts 33 MediaPipe 3D skeletal landmarks per frame.</p>
                    <p><span class='wh-badge wh-badge-why'>WHY</span> Converts raw video pixels into structured, normalized 239-dimensional feature tensors that standard sequential neural networks can process.</p>
                    <p><span class='wh-badge wh-badge-how'>HOW</span> Uses MediaPipe Pose for keypoint tracking → computes hip-centered and torso-scaled coordinates, joint angles, and velocities → slices into 30-frame sliding windows (step=5 frames) → merges with existing synthetic data.</p>
                    </div>
                    """)

                    gr.Markdown("""
                    <div class='step-card'>
                    <div class='step-header'>✂️ Step 2: Dataset Train / Val / Test Split</div>
                    <p><span class='wh-badge'>WHAT</span> Partitions the sequence features into 70% Training (209 samples), 15% Validation (45 samples), and 15% Testing (46 samples) subsets.</p>
                    <p><span class='wh-badge wh-badge-why'>WHY</span> Prevents data leakage and overfitting. Ensures that the model is trained, tuned, and evaluated on completely independent data distributions.</p>
                    <p><span class='wh-badge wh-badge-how'>HOW</span> Two-phase stratified split with <code>scikit-learn</code> preserving exact class ratios → saves label encoder to <code>models/label_encoder.pkl</code>.</p>
                    </div>
                    """)

                with gr.Column():
                    gr.Markdown("""
                    <div class='step-card'>
                    <div class='step-header'>🧠 Step 3: Train Random Forest & Attention-LSTM</div>
                    <p><span class='wh-badge'>WHAT</span> Trains two models: (1) Random Forest baseline benchmark and (2) <b>PyTorch 2-Layer Bidirectional Attention-LSTM</b>.</p>
                    <p style='margin-left: 10px; font-size: 0.95rem; line-height: 1.5;'>
                    • <b>What does LSTM stand for?</b><br>
                    <b>LSTM</b> stands for <b><u>L</u>ong <u>S</u>hort-<u>T</u>erm <u>M</u>emory</b>. It is an advanced Recurrent Neural Network (RNN) architecture equipped with memory cells and internal gates (input, forget, output gates) that allow it to remember patterns over time across all 30 video frames without suffering from the vanishing gradient problem.<br>
                    • <b>What does Attention-LSTM stand for & mean?</b><br>
                    <b>Attention-LSTM</b> is the hybrid architecture combining <b>Long Short-Term Memory</b> with a <b>Temporal Attention Mechanism</b>. While the <i>Bidirectional LSTM</i> reads the motion sequence in both directions (forward from $t_0 \\rightarrow t_{29}$ and backward from $t_{29} \\rightarrow t_0$), the <i>Attention</i> layer assigns learned importance weights to each frame. It dynamically "pays attention" to the most critical moments of the action (such as the lowest dip in a pushup or the apex spread in a jumping jack) rather than treating all 30 frames equally.
                    </p>
                    <p><span class='wh-badge wh-badge-why'>WHY</span> The Random Forest establishes an interpretable ML benchmark. The Bidirectional Attention-LSTM captures sequential forward/backward patterns and focuses computational attention on the most discriminative movement phases.</p>
                    <p><span class='wh-badge wh-badge-how'>HOW</span> Online data augmentation (Gaussian noise + magnitude scaling) → AdamW optimizer with Cosine Annealing learning rate schedule over 35 epochs → checkpointing best model to <code>models/best_lstm_model.pth</code>.</p>
                    </div>
                    """)

                    gr.Markdown("""
                    <div class='step-card'>
                    <div class='step-header'>📊 Step 4: Model Evaluation & Metric Plots</div>
                    <p><span class='wh-badge'>WHAT</span> Evaluates both models on the held-out Test set, calculating Top-1 Accuracy, Macro Precision, Recall, F1-Score, and inference latency.</p>
                    <p><span class='wh-badge wh-badge-why'>WHY</span> Quantifies generalization performance, detects any inter-class confusion, and measures real-time inference speed (FPS).</p>
                    <p><span class='wh-badge wh-badge-how'>HOW</span> Computes classification report and generates high-resolution visualizations: <code>runs/confusion_matrix.png</code> and <code>runs/training_curves.png</code>.</p>
                    </div>
                    """)

                    gr.Markdown("""
                    <div class='step-card'>
                    <div class='step-header'>⚡ Step 5: Run Video Action Detection Test</div>
                    <p><span class='wh-badge'>WHAT</span> Runs sliding-window inference on a demo video, overlays 33-point skeleton lines, and outputs an annotated H.264 MP4 video.</p>
                    <p><span class='wh-badge wh-badge-why'>WHY</span> Validates real-time inference readiness, smooth visual rendering, and browser video encoding before production deployment.</p>
                    <p><span class='wh-badge wh-badge-how'>HOW</span> 30-frame rolling deque → PyTorch inference → Exponential Moving Average (EMA, α=0.3) prediction smoothing to eliminate flickering → OpenCV overlay → H.264 export.</p>
                    </div>
                    """)

        # ────────────────────────────────────────────────────────────
        # Tab 2: Video Action Recognizer
        # ────────────────────────────────────────────────────────────
        with gr.TabItem("🎥 Video Action Recognizer"):
            gr.Markdown("""
            <div class='hero-card'>
            <b>📌 Quickstart Instructions:</b><br>
            1. Upload any sports video file (supports <b>.avi, .mp4, .mov, .mkv, .webm, .flv</b>).<br>
            2. Click <b>Run Action Recognition</b>.<br>
            3. View the 3D pose skeletal tracking video, action confidence breakdown, and summary statistics.
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="📥 Upload Sports Video (.avi, .mp4, .mov, .mkv, .webm, .flv)",
                        file_count="single",
                        file_types=["video", ".avi", ".mp4", ".mov", ".mkv",
                                    ".webm", ".flv", ".wmv"]
                    )
                    btn_analyze = gr.Button("⚡ Run Action Recognition",
                                            variant="primary", size="lg",
                                            elem_classes=["primary-btn"])
                    summary_box = gr.HTML(
                        "<div class='summary-card'>"
                        "<div class='summary-header'>Status</div>"
                        "<div class='summary-action'>Ready</div>"
                        "<div>Upload a sports video file and click "
                        "<b>Run Action Recognition</b>.</div></div>")

                with gr.Column(scale=1):
                    video_output = gr.Video(
                        label="🎬 Annotated Pose Output Video (H.264 MP4)",
                        height=420,
                        elem_classes=["output-video"]
                    )
                    label_probabilities = gr.Label(
                        label="📊 Action Confidence Distribution",
                        num_top_classes=3)

            btn_analyze.click(
                fn=analyze_video,
                inputs=[file_input],
                outputs=[video_output, label_probabilities, summary_box]
            )

        # ────────────────────────────────────────────────────────────
        # Tab 3: Model Performance Insights
        # ────────────────────────────────────────────────────────────
        with gr.TabItem("📊 Model Performance Insights"):
            gr.Markdown("### 📈 Model Evaluation Metrics & Training Curves")

            # ── Training Curves Section ──
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📉 Training & Validation Loss & Accuracy")
                    if TRAINING_CURVES_PATH.exists():
                        gr.Image(value=str(TRAINING_CURVES_PATH),
                                 label="Training & Validation Loss & Accuracy")
                    else:
                        gr.Markdown("Run `python run_pipeline.py` or "
                                    "`python scripts/3_train.py` to generate training curves.")

                    # ── Educational Summary: Training Curves ──
                    gr.Markdown("""
                    <div class='hero-card'>
                    <b>📖 Understanding the Training Curves</b><br><br>

                    <b>What is an Epoch? (X-axis: 0 – 35)</b><br>
                    An <b>epoch</b> is one complete pass through the entire training dataset.
                    At epoch 0 the model starts with random weights (knows nothing). By epoch 35,
                    the model has seen every training sample 35 times and has adjusted its internal
                    weights to learn the patterns that distinguish running, pushups, and jumping jacks.<br><br>

                    <b>Left Plot — Loss (CrossEntropy)</b><br>
                    <b>Loss</b> measures how wrong the model's predictions are. A high loss means
                    the model is making many incorrect predictions; a low loss means predictions
                    closely match the true labels.<br>
                    • <b style="color:#4F46E5;">Blue solid line (Train Loss)</b>: error on the training data — should decrease steadily as the model learns.<br>
                    • <b style="color:#EF4444;">Red dashed line (Val Loss)</b>: error on unseen validation data — if this increases while train loss decreases, the model is <i>overfitting</i> (memorizing training data instead of learning general patterns).<br>
                    • <b>Goal</b>: both lines should decrease and stay close together.<br><br>

                    <b>Right Plot — Accuracy (%)</b><br>
                    <b>Accuracy</b> is the percentage of correctly classified action sequences.<br>
                    • <b style="color:#10B981;">Green solid line (Train Accuracy)</b>: percentage correct on training data.<br>
                    • <b style="color:#F59E0B;">Yellow dashed line (Val Accuracy)</b>: percentage correct on unseen validation data — this is the true measure of model quality.<br>
                    • <b>Goal</b>: both lines should increase toward 100% and stay close together.
                    </div>
                    """)

                # ── Confusion Matrix Section ──
                with gr.Column():
                    gr.Markdown("#### 🎯 Test Set Confusion Matrix")
                    if CONFUSION_MATRIX_PATH.exists():
                        gr.Image(value=str(CONFUSION_MATRIX_PATH),
                                 label="Confusion Matrix Plot")
                    else:
                        gr.Markdown("Run `python run_pipeline.py` or "
                                    "`python scripts/4_evaluate.py` to generate confusion matrix.")

                    # ── Educational Summary: Confusion Matrix ──
                    gr.HTML(value=generate_confusion_matrix_explanation)

            gr.Markdown("""
            ---
            #### 🧠 Technical Architecture & Feature Engineering Specs
            - **Supported Actions**: Running, Pushups, Jumping Jacks (3 classes)
            - **Video Extension Support**: `.avi`, `.mp4`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`
            - **Pose Landmark Extraction**: MediaPipe 3D Keypoint Tracking (33 Body Keypoints)
            - **Scale-Invariant Normalization**: Hip-centered coordinate shifting + Torso scale height normalization (distance & zoom invariant)
            - **Motion Dynamics Features**: 8 Key Joint Angles (Elbows, Knees, Hips, Shoulders) + 3D Velocity vectors across consecutive frames → **239 features/frame**
            - **Neural Architecture**: 2-Layer Bidirectional LSTM with **Temporal Attention Pooling**
            - **Temporal Prediction Smoothing**: Exponential Moving Average (α=0.3) for flicker-free action prediction
            """)

        # ────────────────────────────────────────────────────────────
        # Tab 4: Pipeline Execution & Live Logs
        # ────────────────────────────────────────────────────────────
        with gr.TabItem("📋 Pipeline Execution & Live Logs"):
            gr.Markdown("""
            <div class='hero-card'>
            <b>🔧 Full End-to-End Pipeline Execution</b><br><br>
            Click <b>🚀 Run Pipeline</b> to trigger the complete process flow (Steps 0 → 5).<br>
            The button will display a <b>loading spinner and disable automatically</b> during execution.
            All stages, milestones, and output metrics are recorded into the live table and terminal stream below in real time.
            </div>
            """)
            btn_pipeline = gr.Button("🚀 Run Full Pipeline",
                                     variant="primary", size="lg",
                                     elem_classes=["pipeline-btn"])

            gr.Markdown("### 📊 Live Pipeline Stage Execution Table")
            pipeline_table = gr.Dataframe(
                headers=["Time", "Pipeline Stage", "Status", "Details / Output"],
                datatype=["str", "str", "str", "str"],
                value=[
                    ["--:--:--", "Ready", "⏸️ Idle", "Click 'Run Full Pipeline' above to start execution."]
                ],
                wrap=True,
                interactive=False
            )

            gr.Markdown("### 💻 Live Terminal Console Output (stdout/stderr)")
            pipeline_log = gr.Textbox(
                label="Terminal Output Stream",
                lines=14,
                max_lines=30,
                interactive=False,
                value="Terminal log stream will appear here during execution...",
            )

            btn_pipeline.click(
                fn=run_pipeline_streaming,
                inputs=[],
                outputs=[pipeline_table, pipeline_log, btn_pipeline],
            )

    gr.Markdown("---")
    gr.Markdown("<p style='text-align: center; color: #64748B;'>"
                "Sport Activity Action Recognition AI Pipeline</p>")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860,
                css=custom_css, share=False)

