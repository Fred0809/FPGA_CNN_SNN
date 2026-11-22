import glob
import json
import numpy as np

W = "C:/Users/soren/Documents/Embedded_Assignment_2/code/weights_cstyle_folded/"
TEST_DIR = "C:/Users/soren/Documents/Embedded_Assignment_2/code/dwpw_cnn_testing_data_aug_txt/"

def ld(n):
    return np.array(json.loads(open(W + n).read().replace("{", "[").replace("}", "]")))

c1w, c1b = ld("layer_1_conv1_WEIGHTS.txt"), ld("layer_1_conv1_BIAS.txt")      # (9,16)
dww, dwb = ld("layer_2_dw3x3_WEIGHTS.txt"), ld("layer_2_dw3x3_BIAS.txt")      # (16,9)
pww, pwb = ld("layer_3_pw1x1_WEIGHTS.txt"), ld("layer_3_pw1x1_BIAS.txt")      # (16,8)
clw, clb = ld("layer_4_cls_1x1_WEIGHTS.txt"), ld("layer_4_cls_1x1_BIAS.txt")  # (8,6)

pad = lambda x: np.pad(x, ((1, 1), (1, 1), (0, 0)))

def forward(path, quiet=False):
    u8 = np.fromfile(path, dtype=np.uint8).reshape(60, 80)
    x = ((u8 / 255.0 - 0.5) / 0.25)[..., None]
    xp = pad(x)
    conv = np.zeros((60, 80, 16))
    for kh in range(3):
        for kw in range(3):
            conv += xp[kh:kh+60, kw:kw+80, :] * c1w[kh*3+kw]
    conv = np.maximum(conv + c1b, 0)
    p1 = conv.reshape(30, 2, 40, 2, 16).max(axis=(1, 3))
    pp = pad(p1)
    dw = np.zeros((30, 40, 16))
    for kh in range(3):
        for kw in range(3):
            dw += pp[kh:kh+30, kw:kw+40, :] * dww[:, kh*3+kw]
    dw = np.maximum(dw + dwb, 0)
    pw = np.maximum(dw @ pww + pwb, 0)
    gap = pw.reshape(5, 6, 5, 8, 8).mean(axis=(1, 3)).mean(axis=(0, 1))
    logits = gap @ clw + clb
    if not quiet:
        print("   max conv/dw/pw:", conv.max(), dw.max(), pw.max())
    return logits

CLASSES = ["square", "circle", "figure_eight", "rectangle", "triangle", "ellipse"]
N_PER_CLASS = 100          # raise to 1000 once it works

cm = np.zeros((6, 6), dtype=int)

for gt, name in enumerate(CLASSES):
    files = sorted(glob.glob(TEST_DIR + name + "/*.bin"))[:N_PER_CLASS]
    print(f"{name}: {len(files)} files")
    for i, path in enumerate(files):
        logits = forward(path, quiet=(i > 0))
        cm[gt, int(np.argmax(logits))] += 1
        if i == 0:
            print("   logits:", np.round(logits, 2))

print("\nConfusion matrix (rows=GT, cols=pred):")
print(cm)
print(f"Accuracy: {np.trace(cm) / cm.sum() * 100:.2f}%")

p = TEST_DIR + "square/" + sorted(__import__("os").listdir(TEST_DIR + "square"))[0]
a = np.fromfile(p, dtype=np.uint8)
print("uint8   :", a.size, a.min(), a.max())
f = np.fromfile(p, dtype=np.float32)
print("float32 :", f.size, f.min(), f.max())