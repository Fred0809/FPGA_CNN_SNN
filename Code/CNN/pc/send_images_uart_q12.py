#!/usr/bin/env python3
# Copy of send_images_uart.py updated for data_t = ap_fixed<24,12>
#   - 12 fractional bits (Q12) instead of 8 (Q8.8)
#   - each pixel sent as int32 (the IP reads 32-bit words from DDR)
#   - reply carries 6 logits as int32 (Q12), converted back to float here

from pathlib import Path
import struct
import time

import cv2
import numpy as np
import serial


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("C:/Users/soren/Documents/Embedded_Assignment_2/Handdrawn_test/")

SERIAL_PORT = "COM5"
BAUD_RATE = 4000000

MAX_IMAGES_PER_FOLDER = 10

IMG_H = 60
IMG_W = 80
N_PIXELS = IMG_H * IMG_W

N_CLASSES = 6

# Fixed-point format. Must match data_t = ap_fixed<24,12> in the HLS code
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS          # 4096

BYTES_PER_PIXEL = 4             # int32
RX_BYTES = N_PIXELS * BYTES_PER_PIXEL      # 19200 payload bytes
LOGIT_BYTES = N_CLASSES * 4     # 6 x int32 = 24

FRAME_START = b"\xA6\x6A"

CLASS_FOLDERS = [
    "square",
    "circle",
    "figure_eight",
    "rectangle",
    "triangle",
    "ellipse",
]


# ============================================================
# IMAGE PREPROCESSING
# Match training:
#   x = u8 / 255
#   z = (x - 0.5) / 0.25 = 4*x - 2         (range -2 .. +2)
# Send Q12 (12 fractional bits):
#   q = round(z * 4096)
#     = round((16384*u8)/255 - 8192)
# ============================================================

def image_to_q12_60x80(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    if img.shape != (IMG_H, IMG_W):
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    img_u8 = img.astype(np.uint8).reshape(-1)

    px = img_u8.astype(np.int64)
    q = (16384 * px + 127) // 255 - 8192
    q = np.clip(q, -(1 << 23), (1 << 23) - 1).astype(np.int32)   # fits ap_fixed<24,*>

    assert q.shape[0] == N_PIXELS

    return q


# ============================================================
# UART COMMUNICATION
# FPGA expects:
#   0xA6 0x6A <len u16 LE> <4800 int32 little-endian>
#
# FPGA replies:
#   0xA5 <pred> <time_us u32 LE> <6 x int32 LE logits (Q12)> 0x5A
# ============================================================

def send_image_and_get_prediction(ser, q_img):
    payload = q_img.astype("<i4", copy=False).tobytes(order="C")

    if len(payload) != RX_BYTES:
        raise RuntimeError(f"Bad payload size: {len(payload)} != {RX_BYTES}")

    packet = FRAME_START + struct.pack("<H", len(payload)) + payload

    ser.reset_input_buffer()

    # Round trip timer: starts when the packet is sent ...
    t_start = time.perf_counter()
    ser.write(packet)
    ser.flush()

    # Wait for binary reply start 0xA5.
    # This skips any FPGA ASCII/debug prints such as:
    #   logits=[...]
    deadline = time.time() + 5.0
    found_start = False

    while time.time() < deadline:
        b = ser.read(1)

        if not b:
            continue

        if b[0] == 0xA5:
            found_start = True
            break

    if not found_start:
        raise RuntimeError("Timeout waiting for FPGA reply start 0xA5")

    pred_byte = ser.read(1)
    time_bytes = ser.read(4)
    logit_bytes = ser.read(LOGIT_BYTES)      # 6 classes x 4 bytes
    end = ser.read(1)

    if (
        len(pred_byte) != 1
        or len(time_bytes) != 4
        or len(logit_bytes) != LOGIT_BYTES
        or len(end) != 1
    ):
        raise RuntimeError("Incomplete FPGA reply")

    # ... and stops when the full reply has been received
    round_trip_ms = (time.perf_counter() - t_start) * 1000.0

    if end[0] != 0x5A:
        raise RuntimeError(f"Bad FPGA reply end: {end}")

    pred = pred_byte[0]
    time_us = struct.unpack("<I", time_bytes)[0]
    time_ms = time_us / 1000.0
    raw = struct.unpack("<6i", logit_bytes)
    logits = tuple(round(v / SCALE, 3) for v in raw)    # back to real values

    return pred, time_us, time_ms, logits, round_trip_ms


# ============================================================
# LABELS
# ============================================================

def label_from_folder(folder_name: str) -> int:
    return CLASS_FOLDERS.index(folder_name)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=====================================")
    print(" FPGA UART CNN TEST")
    print(" Q12 int32 60x80 image sender")
    print("=====================================")
    print()

    print(f"Dataset root : {ROOT}")
    print(f"Serial port  : {SERIAL_PORT}")
    print(f"Baud rate    : {BAUD_RATE}")
    print(f"Images/folder: {MAX_IMAGES_PER_FOLDER}")
    print(f"Image size   : {IMG_H} x {IMG_W}")
    print(f"Fixed point  : {FRAC_BITS} fractional bits (scale {SCALE})")
    print(f"TX bytes/img : {RX_BYTES}")
    print()

    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=5,
        write_timeout=5,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )

    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    total = 0
    correct = 0
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)

    round_trips_ms = []       # PC -> FPGA -> PC, per image
    uart_overheads_ms = []    # round trip minus the time the FPGA reports

    try:
        for class_name in CLASS_FOLDERS:
            class_dir = ROOT / class_name

            if not class_dir.exists():
                print(f"Skipping missing folder: {class_dir}")
                continue

            true_label = label_from_folder(class_name)

            image_files = sorted(
                list(class_dir.glob("*.jpg")) +
                list(class_dir.glob("*.jpeg")) +
                list(class_dir.glob("*.png")) +
                list(class_dir.glob("*.bmp"))
            )[:MAX_IMAGES_PER_FOLDER]

            print()
            print(f"Processing class: {class_name}")
            print(f"Images selected: {len(image_files)}")
            print()

            for image_path in image_files:
                q_img = image_to_q12_60x80(image_path)

                pred, time_us, time_ms, logits, round_trip_ms = send_image_and_get_prediction(ser, q_img)

                round_trips_ms.append(round_trip_ms)
                uart_overheads_ms.append(round_trip_ms - time_ms)

                total += 1
                correct += int(pred == true_label)

                if 0 <= pred < N_CLASSES:
                    confusion[true_label, pred] += 1

                print(
                    f"{image_path.name}: "
                    f"true={true_label} "
                    f"pred={pred} "
                    f"logits={logits} "
                    f"time={time_us} us "
                    f"({time_ms:.6f} ms) "
                    f"rt={round_trip_ms:.3f} ms "
                    f"{'OK' if pred == true_label else 'WRONG'}"
                )

    finally:
        ser.close()

    print()
    print("=====================================")
    print(" FINAL RESULTS")
    print("=====================================")
    print()

    if round_trips_ms:
        print(f"Average round trip     : {np.mean(round_trips_ms):.3f} ms")
        print(f"Average UART + overhead: {np.mean(uart_overheads_ms):.3f} ms")
        print(f"Min / max round trip   : {np.min(round_trips_ms):.3f} / {np.max(round_trips_ms):.3f} ms")
        print()

    print(f"Total   : {total}")
    print(f"Correct : {correct}")

    if total > 0:
        print(f"Accuracy: {correct / total * 100:.2f}%")

    print()
    print("Confusion matrix")
    print("(rows=true, cols=pred)")
    print()
    print(confusion)
    print()


if __name__ == "__main__":
    main()
