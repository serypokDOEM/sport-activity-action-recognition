import argparse
import random
import shutil
from pathlib import Path


def copy_files(file_paths, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in file_paths:
        shutil.copy2(path, dest_dir / path.name)


def split_dataset(source_dir: Path, dataset_dir: Path, train_ratio: float = 0.8, seed: int = 42):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    train_images = images_dir / "train"
    val_images = images_dir / "val"
    train_labels = labels_dir / "train"
    val_labels = labels_dir / "val"

    image_paths = sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {source_dir}")

    random.seed(seed)
    random.shuffle(image_paths)
    split_index = int(len(image_paths) * train_ratio)
    train_files = image_paths[:split_index]
    val_files = image_paths[split_index:]

    copy_files(train_files, train_images)
    copy_files(val_files, val_images)

    for image_path in image_paths:
        label_path = source_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            dest = train_labels if image_path in train_files else val_labels
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, dest / label_path.name)

    print(f"Split {len(image_paths)} images into {len(train_files)} train and {len(val_files)} val samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split extracted frames into a YOLO dataset")
    parser.add_argument("--source", type=Path, default=Path("data/extracted_frames"), help="Folder with extracted frame images")
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"), help="Destination dataset folder")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    split_dataset(args.source, args.dataset, args.train_ratio, args.seed)
