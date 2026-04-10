#include <vector>
#include "../../../include/gpu/kernel/global/h2d_data_transfer.cuh"

void h2d_dataTransfer(
    float* d_weights,
    float* d_bias,
    float* d_input,
    int* d_label,           // NEW
    float* d_loss,	    // NEW
    std::vector<float>& h_weights,
    std::vector<float>& h_bias,
    const std::vector<float>& h_input,
    const std::vector<int>& h_label,
    int* d_dims,
    const std::vector<int>& dims, 
    const int total_weights,
    const int total_bias,
    int* d_offsets,
    int* d_bias_offsets,
    std::vector<int>& offsets,
    std::vector<int>& bias_offsets,
    int layers
	){
    cudaMemcpy(d_weights, h_weights.data(), total_weights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), total_bias * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input.data(), dims[0] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_label, h_label.data(), dims[dims.size()-1] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims, dims.data(), dims.size()*sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_offsets, offsets.data(), layers*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_offsets, bias_offsets.data(), layers*sizeof(int), cudaMemcpyHostToDevice);
}


