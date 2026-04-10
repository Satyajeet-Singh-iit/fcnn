#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

#include "include/data_Loader/mnist_Loader.h"
#include "include/print_variables/print_variable.h"
#include "include/main/initialize_weights_and_biases.h"
#include "include/gpu/kernel/global/initialize_device_memory.cuh"
#include "include/gpu/kernel/global/h2d_data_transfer.cuh"


// ---------------- Main ----------------
int main() {

    std::string image_file = "data_Loader/mnist_data/train-images.idx3-ubyte";
    std::string label_file = "data_Loader/mnist_data/train-labels.idx1-ubyte";

    std::vector<std::vector<float>> images = load_mnist_images(image_file);
    std::vector<int> labels = load_mnist_labels(label_file);

    //std::cout << "\nFirst label: " << labels[0] << "\n\n";

    // Print first image (ASCII)
    //print_image(images[0], 28, 28);
    
    // -------- Network --------
    std::vector<int> dims = {
        784, 512, 256, 128, 64, 32, 16, 10
    };
    if(images[0].size()!=dims[0]){
    	dims[0]=images[0].size();
    	std::cout<<"Error:: input doesnot match 28x28, therefore initializing the model to the input."<<std::endl;
    }

    int layers = dims.size() - 1;

    // -------- Compute weight + bias offsets --------
    std::vector<int> offsets(layers);
    std::vector<int> bias_offsets(layers);

    int total_weights = 0;
    int total_bias = 0;

    for (int l = 0; l < layers; l++) {
        offsets[l] = total_weights;
        bias_offsets[l] = total_bias;

        total_weights += dims[l] * dims[l+1];
        total_bias += dims[l+1];
    }

    // -------- Host Memory --------
    std::vector<float> h_input(dims[0] * images.size());
    std::vector<int> h_label( images.size() );
    std::vector<float> h_weights(total_weights);
    std::vector<float> h_bias(total_bias);
    
    // weights & bias
    init_weights_xavier(h_weights, dims, offsets);
    init_bias(h_bias);
    
    // -------- Device Memory --------
    float *d_input, *d_weights, *d_bias, *d_loss;
    int *d_label, *d_dims, *d_offsets, *d_bias_offsets;
    
    //-----------flattening images inputs --------
    for(int n=0; n<images.size(); n++){
    	for(int j=0; j<images[0].size(); j++)
    		h_input[n*images[0].size() + j]=images[n][j];
        h_label[n]= labels[n];
    }
    
    //printVariable1D(h_weights);
    //------------initialize device memory-------------
    init_device_memory( images.size(), total_weights, total_bias, dims, layers,
    		d_input, d_weights, d_bias, d_loss, d_label, d_dims, d_offsets, d_bias_offsets);
    //------------H2D- memory transfer----------------
    h2d_dataTransfer( d_weights, d_bias, d_input, d_label, d_loss, h_weights, h_bias, h_input, h_label, 
        d_dims, dims, total_weights, total_bias, d_offsets, d_bias_offsets, offsets, bias_offsets, layers);
    
    //------------free cuda device memory---------------
    free_device_memory(d_input, d_weights, d_bias, d_loss, d_label, d_dims, d_offsets, d_bias_offsets);

    return 0;
}
