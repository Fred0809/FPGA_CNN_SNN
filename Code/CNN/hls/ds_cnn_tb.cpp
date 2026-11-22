// ds_cnn_tb.cpp (for new Depthwise Separable CNN: Conv+DW+PW+AvgPool+CLS+GAP+Argmax)
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <ctime>
#include <cstring>
#include <iomanip>

#include <dirent.h>

#include "ap_fixed.h"
#include "ap_int.h"

typedef ap_fixed<24,12> data_t;



// ---------------- Config ----------------
static const int IMG_H = 60;
static const int IMG_W = 80;
static const int N_IN  = IMG_H * IMG_W;

static const int N_CLASSES = 6; // How many  different classes are used


// DUT (must match your top function name/signature)
extern void ds_cnn(
    const data_t *input_img,
    int *pred_out,
    data_t logits_out[N_CLASSES]
);

// Your actual test data root:
static const std::string BASE_DIR = "C:/Users/soren/Documents/Embedded_Assignment_2/code/dwpw_cnn_testing_data_aug_txt";



// Folder names (GT labels match training order)
static const std::string CLASS_DIRS[N_CLASSES] = {
    "square", // 0
    "circle",   // 1
    "figure_eight",  // 2
    "rectangle",    // 3
    "triangle", // 4
    "ellipse" // 5
};

static const char* CLASS_NAMES[N_CLASSES] = {
    "Square",
    "Circle",
    "Figure Eight",
    "Rectangle",
    "Triangle",
    "Ellipse"
};

// How many random files per class (0 = test ALL files)
static const int SAMPLES_PER_CLASS = 1000;

// Print per-file results
static const bool VERBOSE_PER_FILE = true;

// Training normalization: X = (X - 0.5) / 0.25, where X is in [0,1]
static inline data_t normalize_like_training(int px_0_255) {
    if (px_0_255 < 0) px_0_255 = 0;
    if (px_0_255 > 255) px_0_255 = 255;
    float x01 = (float)px_0_255 / 255.0f;   // [0,1]
    float z   = (x01 - 0.5f) / 0.25f;       // match Keras training
    return (data_t)z;
}
// --------------------------------------


// Section has been changed to look for bin fiels instead of txt files. 
static std::vector<std::string> list_bin_files(const std::string &folder) {
    std::vector<std::string> files;
    DIR *dir = opendir(folder.c_str());
    if (!dir) {
        std::cerr << "⚠️ Could not open directory: " << folder << "\n";
        return files;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() >= 4 && name.substr(name.size() - 4) == ".bin")
            files.push_back(folder + "/" + name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

// Load a 60x80 bin image of integers into a flat normalized buffer
// Code has been changed to read bin files instead of txt files
/*static bool load_bin_image_flat(const std::string &path, data_t flat_out[N_IN]) {
    std::ifstream infile(path, std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "❌ Cannot open " << path << "\n";
        return false;
    }*/

// Source image size on disk (160x120 uint8) -> network input 60x80
static const int SRC_H = 120;
static const int SRC_W = 160;
static const int N_SRC = SRC_H * SRC_W;   // 19200

// Loads a .bin image, downscales 160x120 -> 60x80 if needed, then normalizes.
static bool load_bin_image_flat(const std::string &path, data_t flat_out[N_IN]) {
    std::ifstream infile(path, std::ios::binary | std::ios::ate);
    if (!infile.is_open()) {
        std::cerr << "❌ Cannot open " << path << "\n";
        return false;
    }

 const std::streamsize size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    static unsigned char src[N_SRC];
    static unsigned char dst[N_IN];   // not "small": Windows headers #define that name

    if (size == N_SRC) {
        infile.read(reinterpret_cast<char*>(src), N_SRC);
        if (infile.gcount() != N_SRC) {
            std::cerr << "❌ Short read in " << path << "\n";
            return false;
        }
        // 2x2 block average with rounding (160/80 = 120/60 = 2 exactly)
        for (int r = 0; r < IMG_H; r++) {
            for (int c = 0; c < IMG_W; c++) {
                int s = src[(2*r)   * SRC_W + 2*c] + src[(2*r)   * SRC_W + 2*c + 1]
                      + src[(2*r+1) * SRC_W + 2*c] + src[(2*r+1) * SRC_W + 2*c + 1];
                dst[r * IMG_W + c] = (unsigned char)((s + 2) >> 2);
            }
        }
    } else if (size == N_IN) {
        infile.read(reinterpret_cast<char*>(dst), N_IN);
        if (infile.gcount() != N_IN) {
            std::cerr << "❌ Short read in " << path << "\n";
            return false;
        }
    } else {
        std::cerr << "❌ Unexpected file size " << size << " in " << path
                  << " (expected " << N_SRC << " or " << N_IN << ")\n";
        return false;
    }

    for (int i = 0; i < N_IN; i++)
        flat_out[i] = normalize_like_training((int)dst[i]);
    return true;
}
    static unsigned char buf[N_IN];


// Print confusion matrix counts + row-normalized
static void print_confusion_matrix(const int cm[N_CLASSES][N_CLASSES]) {
    std::cout << "\nConfusion Matrix (rows=GT, cols=Pred) counts:\n\n";

    std::cout << std::setw(22) << "GT \\ Pred";
    for (int p = 0; p < N_CLASSES; p++) {
        std::cout << std::setw(22) << CLASS_NAMES[p];
    }
    std::cout << "\n";

    for (int g = 0; g < N_CLASSES; g++) {
        std::cout << std::setw(22) << CLASS_NAMES[g];
        for (int p = 0; p < N_CLASSES; p++) {
            std::cout << std::setw(22) << cm[g][p];
        }
        std::cout << "\n";
    }

    std::cout << "\nConfusion Matrix (rows normalized):\n\n";

    std::cout << std::setw(22) << "GT \\ Pred";
    for (int p = 0; p < N_CLASSES; p++) {
        std::cout << std::setw(22) << CLASS_NAMES[p];
    }
    std::cout << "\n";

    for (int g = 0; g < N_CLASSES; g++) {
        int row_sum = 0;
        for (int p = 0; p < N_CLASSES; p++) row_sum += cm[g][p];

        std::cout << std::setw(22) << CLASS_NAMES[g];
        for (int p = 0; p < N_CLASSES; p++) {
            double v = (row_sum > 0) ? (double)cm[g][p] / (double)row_sum : 0.0;
            std::cout << std::setw(22) << std::fixed << std::setprecision(3) << v;
        }
        std::cout << "\n";
    }
}

int main() {
    std::mt19937 rng((unsigned)std::time(nullptr));

    static data_t input_flat[N_IN] __attribute__((aligned(64)));

    int cm[N_CLASSES][N_CLASSES] = {};
    int total = 0;
    int correct = 0;

    std::cout << "BASE_DIR: " << BASE_DIR << "\n";
    std::cout << "IMG: " << IMG_H << "x" << IMG_W << " (flat " << N_IN << ")\n";
    std::cout << "Normalization: (x01 - 0.5)/0.25 (matches training)\n";
    std::cout << "Samples/class: " << (SAMPLES_PER_CLASS == 0 ? "ALL" : std::to_string(SAMPLES_PER_CLASS)) << "\n\n";

    for (int gt = 0; gt < N_CLASSES; gt++) {
        const std::string folder = BASE_DIR + "/" + CLASS_DIRS[gt];
        auto files = list_bin_files(folder);

        if (files.empty()) {
            std::cerr << "⚠️ No .bit files in " << folder << "\n"; // Changed from txt to bin
            continue;
        }

        std::shuffle(files.begin(), files.end(), rng);

        int pick = (SAMPLES_PER_CLASS == 0)
                     ? (int)files.size()
                     : std::min((int)files.size(), SAMPLES_PER_CLASS);

        std::cout << "===== GT CLASS " << gt << " (" << CLASS_NAMES[gt]
                  << ") testing " << pick << " files from " << folder << " =====\n";

        for (int i = 0; i < pick; i++) {
            const std::string &path = files[i];

            if (!load_bin_image_flat(path, input_flat)) continue;

            // Run DUT
            int pred_ret = 0;
            data_t logits[N_CLASSES];

            ds_cnn(&input_flat[0], &pred_ret, logits);

            int pred = pred_ret;


            if (pred < 0) pred = 0;
            if (pred >= N_CLASSES) pred = N_CLASSES - 1;

            cm[gt][pred] += 1;
            total += 1;
            if (pred == gt) correct += 1;

            if (VERBOSE_PER_FILE) {
                const char* base = std::strrchr(path.c_str(), '/');
                std::string fname = base ? (base + 1) : path;

                std::cout << "File: " << fname
                          << " | PRED: " << pred << " (" << CLASS_NAMES[pred] << ")"
                          << " | GT: "   << gt   << " (" << CLASS_NAMES[gt] << ")"
                          << " | logits: ["
                          << (float)logits[0] << ", "
                          << (float)logits[1] << ", "
                          << (float)logits[2] << ", "
                          << (float)logits[3] << ", "
                          << (float)logits[4] << ", "
                          << (float)logits[5] << "]"
                          << "\n";
            }
        }
        std::cout << "\n";
    }

    double acc = (total > 0) ? (double)correct / (double)total : 0.0;
    std::cout << "✅ Finished testing Depthwise Separable CNN.\n";
    std::cout << "Total: " << total << "  Correct: " << correct
              << "  Accuracy: " << std::fixed << std::setprecision(4) << acc << "\n";

    print_confusion_matrix(cm);
    return 0;
}

