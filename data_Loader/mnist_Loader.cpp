#include <iostream>
#include <fstream>
#include <vector>
#include "../include/data_Loader/mnist_Loader.h"

// ---------------- Read big-endian int ----------------
int read_int(std::ifstream &f) {
    uint32_t temp;
    f.read(reinterpret_cast<char*>(&temp), 4);
    return __builtin_bswap32(temp);
}

// ---------------- Load Images ----------------
std::vector<std::vector<float>> load_mnist_images(const std::string &path) {

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << path << std::endl;
        exit(1);
    }

    int magic = read_int(file);
    int num_images = read_int(file);
    int rows = read_int(file);
    int cols = read_int(file);

    std::cout << "Images: " << num_images
              << " | Size: " << rows << "x" << cols << std::endl;

    std::vector<std::vector<float>> images(
        num_images, std::vector<float>(rows * cols)
    );

    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, 1);
            images[i][j] = pixel / 255.0f;  // normalize
        }
    }

    return images;
}

// ---------------- Load Labels ----------------
std::vector<int> load_mnist_labels(const std::string &path) {

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << path << std::endl;
        exit(1);
    }

    int magic = read_int(file);
    int num_labels = read_int(file);

    std::vector<int> labels(num_labels);

    for (int i = 0; i < num_labels; i++) {
        unsigned char label = 0;
        file.read((char*)&label, 1);
        labels[i] = (int)label;
    }

    return labels;
}

// ---------------- Debug Print ----------------
void print_image(const std::vector<float>& img, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << (img[i*cols + j] > 0.05 ? "#" : ".");
        }
        std::cout << "\n";
    }
}


