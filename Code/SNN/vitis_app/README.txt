vitis_app/  -  CPU program (bare-metal C, Zynq ARM)
==================================================

helloworld_Assignment_2.c
-------------------------
How to use: in Vitis create a Platform component from the .xsa (standalone,
ps7_cortexa9_0), build it, create an application from the "Hello World"
template, replace the contents of src/helloworld.c with this file
(platform.c stays unchanged), build, run on the board.
It works with every version of our SNN hardware without changes, because
the addresses and the protocol stay the same.

What it does, for every image
-----------------------------
    1. wait for the start bytes 0xAA 0x55 on the UART
    2. receive 384 int16 features (768 bytes, little-endian)
    3. write them into the shared BRAM at 0x4000_0000, flush the data cache
    4. write the BRAM address into the SNN registers, read the timer,
       set the GPIO timing pin, start the IP (ap_start)
    5. wait for ap_done, read the timer again
    6. read the prediction and the 6 logits, send the 31-byte reply

Addresses
---------
    AXI BRAM controller (features)   0x4000_0000  (8 KB)
    AXI GPIO (timing pin)            0x4001_0000
    SNN control registers            0x4002_0000

UART protocol (4,000,000 baud, 8N1, little-endian)
--------------------------------------------------
    PC -> board: 0xAA 0x55 + 384 x int16                      = 770 bytes
    board -> PC: 0xA5, pred (1 B), time_us (uint32),
                 6 logits (int32), 0x5A                       = 31 bytes

Changes to the lab code, and why
--------------------------------------------
1. Driver names: *_BASEADDR macros instead of *_DEVICE_ID, xiltimer.h
   instead of xtime_l.h. Vitis 2026.1 (SDT flow) no longer has the old names.
2. Xil_DCacheFlushRange(...) after writing the BRAM, so the data really
   leaves the CPU cache before the IP reads it.
3. Timer: the Zynq global timer is enabled after init_platform()
       Xil_Out32(XPAR_GLOBAL_TIMER_BASEADDR + 0x08, 0x1);
   and read directly before ap_start and after ap_done
       t = Xil_In32(XPAR_GLOBAL_TIMER_BASEADDR + 0x00);
       time_us = dt_ticks * 1e6 / (XPAR_CPU_CORE_CLOCK_FREQ_HZ / 2)
   XTime_GetTime returned 0 in Vitis 2026.1, so every time was 0 us.
   The global timer runs at half the CPU clock (325 MHz).
4. Six logits and a 31-byte reply (the lab had 4 logits, 23 bytes).

Hardware note: the AXI BRAM Controller must have ONE BRAM interface. With
two, the read port is unconnected and every image gives the same scores.

Measured: 1,269 us per inference, the same for every image.
