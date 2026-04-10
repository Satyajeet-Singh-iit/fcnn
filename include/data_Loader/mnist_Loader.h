#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H


// ---------------- Read big-endian int ----------------
int read_int(std::ifstream &f);

// ---------------- Load Images ----------------
std::vector<std::vector<float>> load_mnist_images(const std::string &path);

// ---------------- Load Labels ----------------
std::vector<int> load_mnist_labels(const std::string &path);

// ---------------- Debug Print ----------------
void print_image(const std::vector<float>& img, int rows, int cols);

#endif
