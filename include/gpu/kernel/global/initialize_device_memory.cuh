#ifndef INITIALIZE_DEVICE_MEMORY_CUH
#define INITIALIZE_DEVICE_MEMORY_CUH


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
		int   *d_label, 
		int   *d_dims, 
		int   *d_offsets,
		int   *d_bias_offsets
		);
		
void free_device_memory(
	float* d_input,
	float* d_weights,
	float* d_bias,
	float* d_loss,
	int*   d_label,
	int*   d_dims,
	int*   d_offsets,
	int*   d_bias_offsets
	);
	

#endif

