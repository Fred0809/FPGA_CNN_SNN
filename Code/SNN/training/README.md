# SNN training script


snn_mlp_train_Assignment_2.py   (final, used for the hardware)
--------------------------------------------------------------
The lectures SNN lab training script (snn_mlp.py) adapted to our shapes. It loads the images, does the fixed preprocessing, trains the SNN, saves plots and reports, and exports the int16 weights as fpga_weights.h for HLS.

Run (inside the Python environment with snnTorch / PyTorch / OpenCV):
    
    python snn_mlp_train_Assignment_2.py

Output goes to OUT_DIR (snn_shapes_model). Rename an old OUT_DIR first, otherwise its files are overwritten.

Preprocessing:

Resize 160x120 -> 80x60, scale to [0,1], Sobel Gx + Gy, split into ON (+) and OFF (-) edges, normalise, keep the top 10 % (90th percentile) as spikes, average every 5x5 cell -> 12 x 16 x 2 = 384 features, T = 1.

The data is split 70/30 (random_state 42): 210,000 train / 90,000 validation.

Changes, and why
- line 20
      ROOT = ...\dataset\train\all_noise
our dataset instead of the cable dataset

- line 22

      CLASS_FOLDERS = square, circle, figure_eight, rectangle, triangle, ellipse

  6 shape classes instead of 4 cable classes

- line 43

      EPOCHS = 60   (was 10)

  the larger network keeps improving until about epoch 40-60

- line 50

      SPIKE_PENALTY_WEIGHT = 1e-6   (was 1e-5)

  with 1e-5 the larger network never fired a single spike and stayed at 16.7 % (chance) for all 60 epochs; 1e-6 lets it learn

- line 55

      OUT_DIR = ...\snn_shapes_model

- line 67

      ARCHITECTURE_NAME = "... -> 384 -> 128 -> 6"   (text in the reports)

- line 311, 349

      num_classes = 6   (was 4)

- lines 347-348

      hidden1 = 384, hidden2 = 128   (was 64, 32)

    the 64-32 network could not even fit its own training data (underfitting). Wider layers gave the largest gain; adding a fourth layer made the network stop spiking. 384-128 is the largest size that fits the Zynq-7020 BRAM with margin (76 %).

- line 504-505

      #define HIDDEN1 384 / #define HIDDEN2 128 in the exported header

    these are hard-coded strings in the export function and must match the network, otherwise HLS reads only part of the weights

Unchanged:

Rreprocessing, BATCH_SIZE 32 (16/64/128 tested, 32 was best), LR 1e-3, SEED 42, the 70/30 split.

Result (60 epochs): train 83.8 %, validation 80.65 %, loss 0.402.

## snn_mlp_shapes_centring_version.py   (earlier experiment, NOT used)

Same script plus a center_shape() step that crops and centres the shape
before Sobel. It reached 98.6 %, but the group decided not to use it
because it changes the data before the network. Kept for reference.
