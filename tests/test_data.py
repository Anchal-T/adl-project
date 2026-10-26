import torch

from data import IMAGE_SIZE, KvasirSegDataset, list_sample_ids, load_or_create_split


def test_split_is_80_10_10_disjoint_and_saved(synthetic_kvasir):
    split = load_or_create_split(synthetic_kvasir, seed=7)

    assert len(split["train"]) == 16
    assert len(split["val"]) == 2
    assert len(split["test"]) == 2
    everything = split["train"] + split["val"] + split["test"]
    assert sorted(everything) == list_sample_ids(synthetic_kvasir)
    assert (synthetic_kvasir / "split.json").exists()


def test_saved_split_is_reused_and_seed_reproducible(synthetic_kvasir, tmp_path):
    first = load_or_create_split(synthetic_kvasir, seed=7)
    second = load_or_create_split(synthetic_kvasir, seed=7)
    assert first == second

    elsewhere = load_or_create_split(synthetic_kvasir, split_file=tmp_path / "other.json", seed=7)
    assert elsewhere == first


def test_default_preprocessing_resizes_to_256_and_binarizes_mask(synthetic_kvasir):
    ids = list_sample_ids(synthetic_kvasir)
    dataset = KvasirSegDataset(synthetic_kvasir, ids[:1])
    image, mask, sample_id = dataset[0]

    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert mask.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert sample_id == ids[0]


def test_augmentation_keeps_shape_and_binary_masks(synthetic_kvasir):
    torch.manual_seed(0)
    ids = list_sample_ids(synthetic_kvasir)
    dataset = KvasirSegDataset(synthetic_kvasir, ids, image_size=64, augment=True)
    for index in range(len(dataset)):
        image, mask, _ = dataset[index]
        assert image.shape == (3, 64, 64)
        assert mask.shape == (1, 64, 64)
        assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_dataset_covers_all_synthetic_cases(synthetic_kvasir):
    assert len(list_sample_ids(synthetic_kvasir)) == 20
