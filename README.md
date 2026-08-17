# Sport Activity Action Recognition

AI-powered sport action recognition system that detects **3 sport activities** from video using skeletal pose estimation and deep learning.

## Supported Actions
1. **Running** — Alternating leg stride + opposite arm swing
2. **Pushups** — Horizontal body with vertical arm push motion
3. **Jumping Jacks** — Symmetric arm raise + leg spread pattern

## Technical Architecture

| Component | Technology |
|---|---|
| Pose Detection | MediaPipe (33 body keypoints) |
| Feature Engineering | 239-dim scale-invariant features per frame |
| Model Architecture | 2-Layer Bidirectional LSTM + Temporal Attention |
| Baseline Model | Random Forest (mean+std pooling) |
| Web Interface | Gradio |

### Feature Engineering (239 features per frame)
- **132 features**: Hip-centered, torso-scaled normalized landmark coordinates (33 × 4)
- **8 features**: Key joint angles — elbows, knees, hips, shoulders (normalized to [0,1])
- **99 features**: Inter-frame velocity vectors (33 × 3 axes)

## Project Structure

```
sport-activity-action-recognition/
├── app.py                              # Gradio web interface
├── run_pipeline.py                     # End-to-end pipeline runner
├── requirements.txt                    # Python dependencies
├── scripts/
│   ├── 0_generate_sample_dataset.py    # Generate synthetic training data
│   ├── 1b_extract_pose_sequences.py    # Extract pose features from real videos
│   ├── 2_split_dataset.py              # Train/Val/Test split (70/15/15)
│   ├── 3_train.py                      # Train Random Forest + LSTM models
│   ├── 4_evaluate.py                   # Evaluate models + generate metrics
│   ├── 5_detect_video.py               # Run detection on video files
│   ├── 6_detect_webcam.py              # Real-time webcam detection
│   └── model.py                        # PyTorch model architecture
├── data/
│   ├── raw_videos/{action}/            # Input videos organized by action
│   └── processed/                      # Extracted features (.npy files)
├── models/                             # Trained model weights
└── runs/                               # Training curves, confusion matrix, output videos
```

## Quick Start

### Git clone project and checkout to master for the latest code.
```bash
git clone https://github.com/serypokDOEM/sport-activity-action-recognition.git

```
### Windows

```bash
venv\Scripts\activate
```

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full pipeline (generate data → train → evaluate)
```bash
python run_pipeline.py
```

### 3. Launch web interface only
```bash
python app.py
```

### Run individual steps
```bash
python scripts/0_generate_sample_dataset.py    # Generate synthetic data
python scripts/2_split_dataset.py              # Split dataset
python scripts/3_train.py --epochs 50          # Train with custom epochs
python scripts/4_evaluate.py                   # Evaluate models
python scripts/5_detect_video.py --input path/to/video.mp4  # Detect from video
python scripts/6_detect_webcam.py              # Real-time webcam detection
```
