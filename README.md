# Explainable Polyp Segmentation Using U-Net and Attention U-Net

Advanced Deep Learning project, IIIT Sri City.

Authors:
- Piyush Kumar (S20230010186)
- Anchal Tandekar (S20230010015)
- Monu Agrahari (S20230010148)

We compare a baseline U-Net (Ronneberger et al., 2015) with an Attention U-Net (Oktay et al., 2018) on the Kvasir-SEG colonoscopy dataset, and explain the predictions with Seg-Grad-CAM (Vinogradova et al., 2020) and attention-gate maps.

## Setup

```bash
pip install -r requirements.txt
```

On Google Colab with a GPU runtime, PyTorch is already installed; only run the `pip install` line for the remaining packages.

## Data

```bash
python data.py --data_dir data
```

This downloads the official Kvasir-SEG archive and extracts it to `data/Kvasir-SEG/` with `images/` and `masks/` folders. You can also download it manually from https://datasets.simula.no/kvasir-seg/ and unzip it to the same place.

Preprocessing: images are resized to 256x256 (bilinear) and scaled to [0, 1]; masks are resized with nearest-neighbour and binarized. The 1,000 samples are shuffled with a fixed seed and split 80/10/10 into train, validation and test. The split is written to `data/Kvasir-SEG/split.json` on first use and reused by every script, so both models see identical splits. Flips, small rotations (±10 degrees) and scaling (0.9 to 1.1) are applied to the training split only.

## Train

```bash
python train.py --model unet --epochs 100 --lr 1e-4 --batch_size 8 --lambda_dice 1.0 --patience 15 --out_dir outputs
python train.py --model attention_unet --epochs 100 --lr 1e-4 --batch_size 8 --lambda_dice 1.0 --patience 15 --out_dir outputs
```

Both models use the same settings: loss `L = BCE + lambda * Dice`, Adam, cosine learning-rate decay over `--epochs`, and early stopping on validation Dice. The best checkpoint (highest validation Dice) is saved to `outputs/<model>_best.pt`, and the per-epoch history to `outputs/<model>_history.csv`.

## Evaluate

Run this once, after training is finished, to report test-set metrics at threshold 0.5:

```bash
python evaluate.py --checkpoint outputs/unet_best.pt --out_dir outputs
python evaluate.py --checkpoint outputs/attention_unet_best.pt --out_dir outputs
```

This writes `<model>_test_summary.json` (mean and standard deviation of Dice, IoU, precision and recall) and `<model>_test_per_image.csv`.

## Explain

```bash
python explain.py --checkpoint outputs/unet_best.pt --out_dir outputs/explain
python explain.py --checkpoint outputs/attention_unet_best.pt --out_dir outputs/explain
```

For every test image this computes the Seg-Grad-CAM heatmap and the energy-inside-GT ratio `E_in` (sum of the heatmap over ground-truth pixels divided by the sum over all pixels). The first `--num_figures` images are saved to `figures/` as panels with the image, Seg-Grad-CAM overlay, predicted mask and ground truth. For the Attention U-Net, a second row shows the attention-gate maps.

## Tests

```bash
pytest
```

The tests run on CPU using a small synthetic Kvasir-style dataset generated on the fly. They check model output shapes, the loss and metric values on known tensors, the split and preprocessing, and a short train, evaluate and explain run for both models.

## Structure

```
data.py                 Kvasir-SEG loader, preprocessing, augmentation, seeded split, download helper
models/unet.py          U-Net
models/attention_unet.py  Attention U-Net with additive attention gates
models/__init__.py      model registry
losses.py               BCE + lambda * Dice loss
metrics.py              Dice, IoU, precision, recall at threshold 0.5
utils.py                seeding, device selection, checkpoint save and load
train.py                training with early stopping
evaluate.py             test-set evaluation
explain.py              Seg-Grad-CAM, attention maps, E_in and figures
tests/                  pytest smoke tests
```
