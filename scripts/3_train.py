import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data_yaml: Path, epochs: int = 50, batch: int = 16, model_name: str = "yolov8n.pt"):
    model = YOLO(model_name)
    model.train(data=str(data_yaml), epochs=epochs, batch=batch, project="runs/train", name="yolov8", exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model on the prepared dataset")
    parser.add_argument("--data", type=Path, default=Path("data/dataset/data.yaml"), help="YOLO dataset config file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model architecture or weights file")
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch, args.model)
