hls/  -  Vitis HLS accelerator
=============================

Files
-----
    snn_mlp_Assignment_2.cpp           the accelerator (top function: snn_mlp)
    snn_mlp_tb_Assignment_2.cpp        C-simulation testbench
    fpga_weights.h                     int16 weights exported by the training script
    make_test_features_Assignment_2.py helper: writes test_features/ for the testbench
    test_features/<shape>/*.txt        360 files, 384 integers each (60 per shape)

Vitis HLS settings (hls_config.cfg)
-----------------------------------
    part = xc7z020clg400-1, clock = 10 ns, clock_uncertainty = 27 %
    syn.top = snn_mlp
    syn.file = fpga_weights.h, snn_mlp_Assignment_2.cpp
    tb.file  = snn_mlp_tb_Assignment_2.cpp   (only the .cpp, NOT the test_features
               folder - adding the folder makes C simulation hang)
    package.output.format = ip_catalog
Run: C Simulation -> C Synthesis -> Package. The IP is written to
<component>/snn_mlp/hls/impl/ip and is added to Vivado as an IP repository.

How the accelerator works
-------------------------
    s_axi_CTRL (AXI-Lite): start/done, feature address, prediction, 6 logits
    m_axi_gmem (AXI master): reads the 384 int16 features from the shared BRAM
    read_features_from_bram -> fc1_lif (384->384) -> fc2_lif (384->128)
      -> fc3_lif (128->6) -> argmax
    LIF update: mem = ((mem * 115) >> 7) + input   (beta = 115/128 = 0.9),
    spike if mem > 128, subtractive reset.
    FC1 multiplies (its inputs are densities) -> the 8 DSPs.
    FC2 and FC3 only add weights where the input spike is 1 -> no multipliers.
    Weights are static const arrays -> stored as ROM in BRAM.

Changes to the lab code, and why
----------------------------------------
1. FC1: shift ONCE after the sum instead of after every product   (line ~96)
       before: acc += (x * w) >> 7;               (for every input)
       after:  sum += x * w;  ...  acc = fc1_b[h] + (sum >> 7);
   Both x and w are scaled by 128, so the product must be divided by 128
   once. Shifting every product throws away each fractional part: small
   products become 0 and negative ones round down, so neurons fired less
   and the hardware was clearly less accurate than the Python model
   (e.g. 79 % vs 99 % in an early version). After the fix the integer
   model matches Python. Overflow: 384 x 128 x 510 is far below 2^31.

2. FC3_OUT_UNROLL = 6   (line 27, was 4)
   six output classes.

3. Testbench: 6 class names, N = 60 files per class (line 155), our
   test_features path (line 158).

The network size comes only from fpga_weights.h; the .cpp did not change
between our network versions.

Results (final version)
-----------------------
    C simulation:   294 / 360 = 81.67 %
    Timing:         estimate 7.30 ns (target 10 ns)
    Latency:        127,841 cycles = 1.278 ms at 100 MHz (FC1 ~60 %, FC2 ~39 %)
    Resources:      BRAM_18K 217 (77 %), DSP 8 (3 %), FF 3,915 (3 %), LUT 9,569 (17 %)

make_test_features_Assignment_2.py
----------------------------------
Helper written for this project. It
imports the training script's own preprocessing and writes the first
N_PER_NOISE = 60 test images of each class (noise 0.4) as text files of
384 integers. Delete test_features/ before running it again: it only
overwrites files with the same name.
