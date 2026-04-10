#include "../include/gpu/kernel_Launch.cuh"
void launch_fused_fcnn(
    const float* __restrict__ input,
    const float* __restrict__ label,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    const int* __restrict__ dims,
    const int* __restrict__ offsets,
    const int* __restrict__ bias_offsets,
    int layers,
    float* loss
    ){
        
    // -------- Launch --------
    fused_fcnn<<<1, 256>>>(
        d_input,
        d_weights,
        d_bias,
        d_dims,
        d_offsets,
        d_bias_offsets,
        layers,
        d_output);

    cudaDeviceSynchronize();
        
}
