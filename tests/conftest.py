from pathlib import Path

import numpy as np
import pytest
from PIL import Image

@pytest.fixture
def synthetic_kvasir(tmp_path: Path) -> Path:
    """Write a tiny Kvasir-style folder (images/ and masks/) with random blobs as polyps."""
    root = tmp_path / "Kvasir-SEG"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    rng = np.random.default_rng(0)
    height, width = 96, 112
    yy, xx = np.mgrid[0:height, 0:width]
    for case in range(20):
        cy, cx = rng.integers(20, height - 20), rng.integers(20, width - 20)
        radius = rng.integers(8, 16)
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        image = rng.integers(0, 80, size=(height, width, 3), dtype=np.uint8)
        image[blob] = (200, 60, 60)
        mask = blob.astype(np.uint8) * 255
        name = f"case_{case:03d}.jpg"
        Image.fromarray(image).save(root / "images" / name)
        Image.fromarray(mask).save(root / "masks" / name)
    return root
