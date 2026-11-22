pc/  -  PC scripts (Python, UART)
================================

Both scripts talk to the board running vitis_app/helloworld_Assignment_2.c.
Requirements: pyserial, numpy, opencv-python. Close every serial monitor
first (only one program can open the COM port).
Before running, set:
    ROOT         (line 16)  folder that contains the six shape folders
    SERIAL_PORT  (line 18)  the board's COM port (Device Manager -> Ports)

send_txt_uart_snn_Assignment_2.py  -  hardware check
-----------------------------------------------------
Sends the 360 ready-made feature files from hls/test_features (the same
inputs as the HLS C simulation). The board must give exactly the C-sim
result: 294 / 360 with the identical confusion matrix. This proves that
the FPGA computes the same as the C model, input by input.

send_images_uart_snn_Assignment_2.py  -  accuracy and timing on images
----------------------------------------------------------------------
Reads PNG images, does the SNN preprocessing on the PC (same as training),
sends the 384 features, and prints per image: prediction, inference time
and the six logits; at the end: accuracy, confusion matrix, and the timing.
    ROOT = ...\dataset\test\noise_0.4     full test set (MAX_IMAGES_PER_FOLDER = 4000)
    ROOT = ...\Hand-drawn final test      60 hand-drawn images

Changes to the lab script, and why
------------------------------------------
    CLASS_FOLDERS          the 6 shapes
    ser.read(31), != 31    the reply is 31 bytes (6 logits)
    reply[30] == 0x5A      end marker moved
    "<iiiiii"              unpack 6 int32 logits
    np.zeros((6, 6)), pred < 6
                           confusion matrix for 6 classes (with 4, triangle
                           and ellipse would not be counted)
    timing (new)           time.perf_counter() before ser.write and after
                           ser.read; prints the average round trip and
                           "UART + overhead" = round trip - time_us
                           (the assignment asks for UART time separately)

Results on the board (24,000 test images)
-----------------------------------------
    Accuracy 81.23 %  (19,495 / 24,000)
    Average round trip 11.85 ms, UART + overhead 10.58 ms, inference 1.27 ms
    Min / max round trip 3.58 / 31.09 ms
Calculated wire time: 770 + 31 bytes x 10 bits / 4 Mbaud = 2.0 ms. The rest
of the overhead is most likely the USB-serial driver on Windows (default
latency timer) and Python, not the UART line itself.
