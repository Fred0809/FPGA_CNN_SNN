"""
make_test_features_Assignment_2.py

Creates the test files for the Vitis HLS C simulation of the SNN.

For each test image it runs EXACTLY the same preprocessing as the training
script (Resize -> Sobel -> ON/OFF spikes -> 12x16 grid),
scales the 384 values by 128 and rounds them to integers (same as the
UART script), and writes them to a .txt file, one value per line.

Output layout (read by snn_mlp_tb_Assignment_2.cpp):
    test_features/
        square/        noise0.0_000001.txt ... (384 lines each)
        circle/
        ...
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                       # ...\snn\hls
TRAINING_DIR = HERE.parent / "training"                      # ...\snn\training
TEST_ROOT = Path(r"C:\Users\zagka\Desktop\Assingment_2\dataset\test")
OUT_ROOT = HERE / "test_features"

# How many images to export per class and per noise level
# (60 x 1 noise level = 60 files per class, 360 in total)
N_PER_NOISE = 60

FEATURE_SCALE = 128      # same scale as the weights and the UART script

# -----------------------------------------------------------------------------
# Reuse the preprocessing functions and settings of the TRAINING script,
# so the test files are guaranteed to match what the network was trained on
# -----------------------------------------------------------------------------
sys.path.insert(0, str(TRAINING_DIR))
import snn_mlp_train_Assignment_2 as train  # noqa: E402


def extract_features_int(image_path):
    """Image -> 384 integers, exactly as the network expects them."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    spikes = train.image_to_spikes(img)                         
    features = train.spikes_to_snn_mlp_features(spikes).reshape(-1)   # 384 floats in [0, 1]

    features_int = np.round(features * FEATURE_SCALE).astype(np.int16)
    assert features_int.shape[0] == train.T * train.N_FEATURES
    return features_int


def main():
    noise_dirs = sorted(TEST_ROOT.glob("noise_*"))              # only noise 0_0
    if not noise_dirs:
        raise RuntimeError(f"No noise_* folders found in {TEST_ROOT}")

    total = 0
    for cname in train.CLASS_FOLDERS:                            # same class order as training
        out_dir = OUT_ROOT / cname
        out_dir.mkdir(parents=True, exist_ok=True)

        for noise_dir in noise_dirs:
            images = sorted((noise_dir / cname).glob("*.png"))[:N_PER_NOISE]
            noise_tag = noise_dir.name.replace("_", "")         # "noise_0.4" -> "noise0.4"

            for i, img_path in enumerate(images, start=1):
                values = extract_features_int(img_path)

                # e.g. test_features/circle/noise0.0_000003.txt
                txt_path = out_dir / f"{noise_tag}_{i:06d}.txt"
                with open(txt_path, "w") as f:
                    f.write(f"# source: {img_path.name}\n")      # '#' lines are skipped by the testbench
                    for v in values:
                        f.write(f"{int(v)}\n")
                total += 1

        print(f"{cname}: {len(list(out_dir.glob('*.txt')))} files")

    print(f"Done: {total} files written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
