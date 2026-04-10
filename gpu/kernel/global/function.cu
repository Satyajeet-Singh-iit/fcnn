// ---------------- Fused FCNN Kernel ----------------
__global__ void fused_fcnn(
    const float* __restrict__ input,
    const float* __restrict__ label,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    const int* __restrict__ dims,
    const int* __restrict__ offsets,
    const int* __restrict__ bias_offsets,
    int layers,
    float* __restrict__ Z_all,              // store all Z
    float* __restrict__ A_all,              // store all A
    float* __restrict__ loss               // per-sample loss
    ) {

    __shared__ float activations[MAX_DIM];
    __shared__ float next_activations[MAX_DIM];

    int tid = threadIdx.x;

    // -------- Load input --------
    if (tid < dims[0])
        activations[tid] = input[tid];

    __syncthreads();

    // -------- Forward pass --------
    for (int l = 0; l < layers; l++) {

        int in_dim  = dims[l];
        int out_dim = dims[l+1];

        const float* W = weights + offsets[l];
        const float* B = bias + bias_offsets[l];

        for (int neuron = tid; neuron < out_dim; neuron += blockDim.x) {

            float sum = B[neuron];  // bias added

            #pragma unroll 4
            for (int i = 0; i < in_dim; i++) {
                sum += activations[i] * W[i * out_dim + neuron];
            }

            // ReLU except last layer
            if (l != layers - 1)
                sum = (sum > 0.0f) ? sum : 0.0f;

            next_activations[neuron] = sum;
        }

        __syncthreads();

        // Copy back
        for (int i = tid; i < out_dim; i += blockDim.x)
            activations[i] = next_activations[i];

        __syncthreads();
    }

    // -------- Softmax (10 outputs) --------
    float val = (tid < OUT_DIM) ? activations[tid] : -1e20f;

    float max_val = warp_reduce_max(val);
    float exp_val = (tid < OUT_DIM) ? __expf(val - max_val) : 0.0f;
    float sum_exp = warp_reduce_sum(exp_val);

    if (tid == label[){
        output[tid] = exp_val / sum;
    }
}


#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 16

// ---------------- Kernel ----------------
__global__ void wmma_gemm_buffered(
    half* A, half* B, float* C,
    int M, int N, int K)
{
    // Shared memory double buffer
    __shared__ half As[2][BLOCK_M][BLOCK_K];
    __shared__ half Bs[2][BLOCK_K][BLOCK_N];

    int warpId = (threadIdx.x + threadIdx.y * blockDim.x) / 32;
    int laneId = threadIdx.x % 32;

    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;

    // WMMA accumulator (register)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int buffer = 0;

    for (int k = 0; k < K; k += BLOCK_K) {

        // ---------------- Load tile into shared ----------------
        int row = threadIdx.y;
        int col = threadIdx.x;

        if (row < BLOCK_M && col < BLOCK_K)
            As[buffer][row][col] =
                A[(blockRow + row) * K + (k + col)];

        if (row < BLOCK_K && col < BLOCK_N)
            Bs[buffer][row][col] =
                B[(k + row) * N + (blockCol + col)];

        __syncthreads();

        // ---------------- Compute using WMMA ----------------
        for (int i = 0; i < BLOCK_M; i += WMMA_M) {
            for (int j = 0; j < BLOCK_N; j += WMMA_N) {

                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

                // Load from shared → registers
                wmma::load_matrix_sync(a_frag, &As[buffer][i][0], BLOCK_K);
                wmma::load_matrix_sync(b_frag, &Bs[buffer][0][j], BLOCK_N);

                // Tensor Core MMA
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }

        __syncthreads();

        buffer ^= 1;  // swap buffers
    }

    // ---------------- Store result ----------------
    int cRow = blockRow + (warpId / (BLOCK_N / WMMA_N)) * WMMA_M;
    int cCol = blockCol + (warpId % (BLOCK_N / WMMA_N)) * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(
            &C[cRow * N + cCol],
            c_frag,
            N,
            wmma::mem_row_major);
    }
}

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 16

// ---------------- Kernel ----------------
__global__ void wmma_gemm(
    const half* __restrict__ A,   // (M x K)
    const half* __restrict__ B,   // (K x N)
    float* __restrict__ C,        // (M x N)
    int M, int N, int K)
{
    // Shared memory (single buffer for clarity)
    __shared__ half As[BLOCK_M][BLOCK_K];
    __shared__ half Bs[BLOCK_K][BLOCK_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global block offsets
    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;

    // Warp mapping
    int warpId = (ty * blockDim.x + tx) / 32;

    // Each warp computes one 16x16 tile
    int warpRow = (warpId / (BLOCK_N / WMMA_N)) * WMMA_M;
    int warpCol = (warpId % (BLOCK_N / WMMA_N)) * WMMA_N;

    // Accumulator fragment (REGISTER)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K tiles
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {

        // -------- Load A tile into shared (with M guard) --------
        for (int i = ty; i < BLOCK_M; i += blockDim.y) {
            for (int j = tx; j < BLOCK_K; j += blockDim.x) {

                int global_row = blockRow + i;
                int global_col = k0 + j;

                if (global_row < M && global_col < K)
                    As[i][j] = A[global_row * K + global_col];
                else
                    As[i][j] = __float2half(0.0f);
            }
        }

        // -------- Load B tile into shared --------
        for (int i = ty; i < BLOCK_K; i += blockDim.y) {
            for (int j = tx; j < BLOCK_N; j += blockDim.x) {

                int global_row = k0 + i;
                int global_col = blockCol + j;

                if (global_row < K && global_col < N)
                    Bs[i][j] = B[global_row * N + global_col];
                else
                    Bs[i][j] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // -------- WMMA compute (register level) --------
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {

            wmma::fragment<wmma::matrix_a,
                           WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> a_frag;

            wmma::fragment<wmma::matrix_b,
                           WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> b_frag;

            // Load fragments from shared → registers
            wmma::load_matrix_sync(a_frag,
                &As[warpRow][k],
                BLOCK_K);

            wmma::load_matrix_sync(b_frag,
                &Bs[k][warpCol],
                BLOCK_N);

            // Tensor Core MMA
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    // -------- Store result (with M, N guard) --------
    int cRow = blockRow + warpRow;
    int cCol = blockCol + warpCol;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(
            &C[cRow * N + cCol],
            c_frag,
            N,
            wmma::mem_row_major);
    }
}



