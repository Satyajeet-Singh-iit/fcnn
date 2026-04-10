#ifndef H2D_DATA_TRANSFER_CUH
#define H2D_DATA_TRANSFER_CUH
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
);
#endif
