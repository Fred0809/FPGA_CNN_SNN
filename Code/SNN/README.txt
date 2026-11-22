SNN part of Assignment 2 - Accelerated AI for Vision (PYNQ-Z2)
================================================================

Spiking neural network (SNN) that classifies six shapes, trained in Python
(snnTorch), implemented as an HLS accelerator, and run on the PYNQ-Z2.
Based on the professor's SNN lab code (CableSNNMLP); every change is listed
in the README.txt of each folder.

Folders
-------
    training/    Python training script -> trained model + fpga_weights.h
    hls/         Vitis HLS accelerator (C++), testbench, weights, test features
    vitis_app/   bare-metal C program for the Zynq ARM CPU
    pc/          Python scripts that send data over UART and collect results
    ../snn_shapes_model/   the trained model, plots and exported weights
    ../Pics/FInal_pics/    screenshots and figures of the final version

Data flow
---------
    PC (Python): image -> preprocessing -> 384 features
      --UART 4 Mbaud, 770 bytes-->
    CPU (C): writes the features into a shared 8 KB BRAM, starts the IP
    SNN IP (PL): reads the BRAM over AXI, FC1 -> FC2 -> FC3 with LIF neurons
    CPU: reads prediction + 6 scores, measures time
      --UART, 31 bytes-->
    PC: prints prediction, time, accuracy, confusion matrix

Network
-------
    384 -> 384 -> 128 -> 6, LIF neurons (beta = 0.9), T = 1 time step,
    197,894 parameters, weights stored as int16 (value x 128).
    Input encoding: Sobel ON/OFF edge spikes, pooled into a 12 x 16 grid
    per channel = 384 densities.

Order to rebuild everything
---------------------------
    1. generate_shapes_dataset.py          -> dataset/
    2. training/snn_mlp_train_Assignment_2.py -> snn_shapes_model/ (incl. fpga_weights.h)
    3. copy fpga_weights.h into hls/, run hls/make_test_features_Assignment_2.py
    4. Vitis HLS: C simulation, C synthesis, Package (top function snn_mlp)
    5. Vivado: block design, bitstream, export .xsa (include bitstream)
    6. Vitis: platform from the .xsa, Hello World app, replace helloworld.c
       with vitis_app/helloworld_Assignment_2.c, build, run
    7. pc/send_txt_uart_snn_Assignment_2.py  (check: identical to C simulation)
       pc/send_images_uart_snn_Assignment_2.py (accuracy + timing on images)

Final results (version 4: LINE_WIDTHS [9,10], noise 0.4)
-------------------------------------------------------
    Python validation (90,000 images)      80.65 %
    HLS C simulation (360 files)           81.67 %  (294 / 360)
    Board, full test set (24,000 images)   81.23 %  (19,495 / 24,000)
    Board, hand-drawn set (60 images)      33.3 %
    Inference time on the board            1.269 ms (HLS estimate 1.278 ms)
    Round trip PC -> board -> PC           11.85 ms (UART + overhead 10.58 ms)
    Resources (whole design)               LUT 8,437 (15.9 %), FF 8,842 (8.3 %),
                                           BRAM 107 tiles (76.4 %), DSP 8 (3.6 %)
    Timing                                 100 MHz, WNS +0.861 ns
    Power (Vivado estimate)                1.573 W, mostly the ARM processor

Note: all scripts contain absolute Windows paths (C:\Users\zagka\...).
Change ROOT / OUT_DIR / the testbench path to your own folders before running.

grid_versions.txt in this folder documents an earlier experiment with
different grid sizes (not used in the final version).
