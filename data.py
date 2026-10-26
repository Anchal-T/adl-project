"""Kvasir-SEG loading, preprocessing, seeded splits and training augmentation."""

import argparse
import json
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
IMAGE_SIZE = 256
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_ROTATION_DEG = 10.0
SCALE_RANGE = (0.9, 1.1)


def download_kvasir(data_dir: str | Path) -> Path:
    """Download the official Kvasir-SEG archive and extract it. Returns the Kvasir-SEG folder."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "kvasir-seg.zip"
    if not archive.exists():
        urlretrieve(KVASIR_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    return data_dir / "Kvasir-SEG"


def index_by_stem(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES}


def list_sample_ids(data_dir: str | Path) -> list[str]:
    images = index_by_stem(Path(data_dir) / "images")
    masks = index_by_stem(Path(data_dir) / "masks")
    unpaired = sorted(set(images) ^ set(masks))
    if unpaired:
        raise ValueError(f"Images and masks do not pair up, for example {unpaired[:5]}")
    return sorted(images)


def load_or_create_split(
    data_dir: str | Path,
    split_file: str | Path | None = None,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, list[str]]:
    """Return the train/val/test sample ids, creating and saving the seeded split on first use."""
    split_path = Path(split_file) if split_file else Path(data_dir) / "split.json"
    if split_path.exists():
        return json.loads(split_path.read_text())

    ids = list_sample_ids(data_dir)
    random.Random(seed).shuffle(ids)
    n_train = round(len(ids) * train_fraction)
    n_val = round(len(ids) * val_fraction)
    split = {
        "seed": seed,
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2))
    return split


def binarize_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask > 0.5).float()


def augment_pair(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same random flip, rotation and scaling to an image and its mask."""
    if random.random() < 0.5:
        image, mask = TF.hflip(image), TF.hflip(mask)
    if random.random() < 0.5:
        image, mask = TF.vflip(image), TF.vflip(mask)
    angle = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
    scale = random.uniform(*SCALE_RANGE)
    affine = {"angle": angle, "translate": [0, 0], "scale": scale, "shear": [0.0, 0.0]}
    image = TF.affine(image, **affine, interpolation=InterpolationMode.BILINEAR)
    mask = TF.affine(mask, **affine, interpolation=InterpolationMode.NEAREST)
    return image, binarize_mask(mask)


class KvasirSegDataset(Dataset):
    """Returns (image in [0, 1] shaped 3xHxW, binary mask shaped 1xHxW, sample id)."""

    def __init__(
        self,
        data_dir: str | Path,
        sample_ids: list[str],
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
    ):
        data_dir = Path(data_dir)
        self.images = index_by_stem(data_dir / "images")
        self.masks = index_by_stem(data_dir / "masks")
        self.sample_ids = list(sample_ids)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        size = (self.image_size, self.image_size)
        image = Image.open(self.images[sample_id]).convert("RGB")
        mask = Image.open(self.masks[sample_id]).convert("L")
        image = TF.to_tensor(image.resize(size, Image.Resampling.BILINEAR))
        mask = binarize_mask(TF.to_tensor(mask.resize(size, Image.Resampling.NEAREST)))
        if self.augment:
            image, mask = augment_pair(image, mask)
        return image, mask, sample_id


def build_loader(
    data_dir: str | Path,
    sample_ids: list[str],
    batch_size: int,
    image_size: int = IMAGE_SIZE,
    augment: bool = False,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    dataset = KvasirSegDataset(data_dir, sample_ids, image_size=image_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Kvasir-SEG dataset.")
    parser.add_argument("--data_dir", default="data")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(download_kvasir(args.data_dir))
