#include <random>
#include "include/main/initialize_weights_and_biases.h"

void init_weights_xavier(
    std::vector<float>& weights,
    const std::vector<int>& dims,
    const std::vector<int>& offsets){

    std::mt19937 gen(42);

    for (int l = 0; l < dims.size() - 1; l++) {

        int fan_in = dims[l];
        int fan_out = dims[l+1];

        float limit = 1.0f / sqrtf(fan_in);

        std::uniform_real_distribution<float> dist(-limit, limit);

        int start = offsets[l];
        int size = fan_in * fan_out;

        for (int i = 0; i < size; i++) {
            weights[start + i] = dist(gen);
        }
    }
}
void init_bias(std::vector<float>& bias) {
    for (auto &b : bias)
        b = 0.0f;
}


