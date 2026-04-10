#!/bin/bash
echo "Current Directory:"
pwd

export PATH=/usr/local/cuda-12.0/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH

#nvcc main.cpp data_Loader/mnist_Loader.cpp initialize_weights_and_biases.cpp globalCodes/initialize_device_memory.cu print_variables/print_variable.cpp -o app && ./app
rm -rf *.o
rm -rf app


g++ -c main.cpp -o main.o
g++ -c data_Loader/mnist_Loader.cpp -o mnist_Loader.o
g++ -c initialize_weights_and_biases.cpp -o initialize_weights_and_biases.o
g++ -c print_variables/print_variable.cpp -o print_variable.o
nvcc -c gpu/kernel/global/initialize_device_memory.cu -o initialize_device_memory.o
nvcc -c gpu/kernel/global/h2d_data_transfer.cu -o h2d_data_transfer.o

nvcc main.o \
     mnist_Loader.o \
     initialize_weights_and_biases.o \
     print_variable.o \
     initialize_device_memory.o \
     h2d_data_transfer.o \
     -o app

./app
rm -rf *.o
#compute-sanitizer --tool memcheck  --leak-check full  ./app
#compute-sanitizer --target-processes all ./app


