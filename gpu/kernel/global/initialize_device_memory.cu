#include <vector>
#include "../../../include/gpu/kernel/global/initialize_device_memory.cuh"

void init_device_memory(
		int total_dataSet,
		int total_weights,
		int total_bias,
		const std::vector<int>& dims,
		int layers,
		float *d_input, 
		float *d_weights, 
		float *d_bias, 
		float *d_loss,
		int *d_label, 
		int *d_dims, 
		int *d_offsets,
		int *d_bias_offsets
		){

    cudaMalloc(&d_input, total_dataSet * dims[0] * sizeof(float));
    cudaMalloc(&d_weights, total_weights * sizeof(float));
    cudaMalloc(&d_bias, total_bias * sizeof(float));
    cudaMalloc(&d_label, dims[dims.size()-1] * sizeof(int));
    cudaMalloc(&d_dims, dims.size() * sizeof(int));
    cudaMalloc(&d_offsets, layers * sizeof(int));
    cudaMalloc(&d_bias_offsets, layers * sizeof(int));
}

void free_device_memory(
	float* d_input,
	float* d_weights,
	float* d_bias,
	float* d_loss,
	int*   d_label,
	int*   d_dims,
	int*   d_offsets,
	int*   d_bias_offsets
	){
	
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_loss);
    cudaFree(d_label);
    cudaFree(d_dims);
    cudaFree(d_offsets);
    cudaFree(d_bias_offsets);
}
	
