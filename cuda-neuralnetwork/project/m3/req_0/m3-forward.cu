#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256

static cudaStream_t stream0;
static cudaStream_t stream1;

static const float* static_host_input = nullptr;
static const float* static_host_mask = nullptr;
static const float* static_host_output = nullptr;

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function parameter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int num_cols = Batch * Height_out * Width_out;
    int num_rows = Channel * K * K;

    if (col >= num_cols || row >= num_rows)
        return;

    int b = col / (Height_out * Width_out);
    int y = (col % (Height_out * Width_out)) / Width_out;
    int x = (col % (Height_out * Width_out)) % Width_out;

    int c = row / (K * K);
    int p = (row % (K * K)) / K;
    int q = (row % (K * K)) % K;

    int input_y = y + p;
    int input_x = x + q;

    size_t input_idx = ((size_t)b * Channel + c) * Height * Width + input_y * Width + input_x;
    size_t unroll_idx = (size_t)row * num_cols + col;

    output[unroll_idx] = input[input_idx];

    #undef in_4d
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    // which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    //     exit(-1);
    // }

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    size_t in_size = (size_t)Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = (size_t)Map_out * Channel * K * K * sizeof(float);
    size_t out_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    static_host_input = host_input;
    static_host_mask = host_mask;
    static_host_output = host_output;

    cudaHostRegister((void*)host_input, in_size, cudaHostRegisterDefault);
    cudaHostRegister((void*)host_mask, mask_size, cudaHostRegisterDefault);
    cudaHostRegister((void*)host_output, out_size, cudaHostRegisterDefault);

    cudaMalloc(device_input_ptr, in_size);
    cudaMalloc(device_mask_ptr, mask_size);
    cudaMalloc(device_output_ptr, out_size);

    size_t half_in_size = in_size / 2;
    cudaMemcpyAsync(*device_input_ptr, host_input, half_in_size, cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync((char*)(*device_input_ptr) + half_in_size, (char*)host_input + half_in_size, half_in_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice, stream0);
    cudaMemsetAsync(*device_output_ptr, 0, out_size, stream0);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    //const int Width_unrolled = Batch * Height_out * Width_out;
    const size_t Width_unrolled = (size_t)Batch * Height_out * Width_out;

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication

    // cudaMalloc((void**)&unrolled_matrix, (size_t) Height_unrolled * Width_unrolled * sizeof(float));
    // cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    cudaMalloc(&unrolled_matrix, (size_t)Height_unrolled * Width_unrolled * sizeof(float));
    cudaMalloc(&matmul_output, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.

    

    // Instead of launching the unrolling kernel once over the entire batch,
    // process the Batch in chunks to avoid exceeding grid size limits.

    int batch_half = Batch / 2;
    size_t half_width_unrolled = (size_t)batch_half * Height_out * Width_out;

    dim3 blockDim(16, 16);
    dim3 gridDim0((half_width_unrolled + blockDim.x - 1) / blockDim.x, (Height_unrolled + blockDim.y - 1) / blockDim.y);
    dim3 gridDim1 = gridDim0;


    // Matrix multiplication and permutation. Do not modify.
    // Multiply the mask with the unrolled matrix

    // Unroll inputs on two streams
    matrix_unrolling_kernel<<<gridDim0, blockDim, 0, stream0>>>(device_input, unrolled_matrix, batch_half, Channel, Height, Width, K);
    matrix_unrolling_kernel<<<gridDim1, blockDim, 0, stream1>>>(device_input + (batch_half * Channel * Height * Width),
                                                                unrolled_matrix + (Height_unrolled * half_width_unrolled),
                                                                batch_half, Channel, Height, Width, K);

    // Matrix multiplication and permutation
    dim3 matmul_grid_dim0((half_width_unrolled - 1) / MATMUL_TILE_WIDTH + 1, (Map_out - 1) / MATMUL_TILE_WIDTH + 1, 1);
    dim3 matmul_block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);

    matrixMultiplyShared<<<matmul_grid_dim0, matmul_block_dim, 0, stream0>>>(device_mask, unrolled_matrix,
                                                                             matmul_output, Map_out,
                                                                             Height_unrolled, Height_unrolled, half_width_unrolled,
                                                                             Map_out, half_width_unrolled);

    matrixMultiplyShared<<<matmul_grid_dim0, matmul_block_dim, 0, stream1>>>(device_mask, unrolled_matrix + (Height_unrolled * half_width_unrolled),
                                                                             matmul_output + (Map_out * batch_half * Height_out * Width_out),
                                                                             Map_out,
                                                                             Height_unrolled, Height_unrolled, half_width_unrolled,
                                                                             Map_out, half_width_unrolled);

    int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1, batch_half, 1);

    matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE, 0, stream0>>>(matmul_output, device_output,
                                                                                       Map_out, batch_half, out_image_size);

    matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE, 0, stream1>>>(matmul_output + (Map_out * batch_half * out_image_size),
                                                                                       device_output + (Map_out * batch_half * out_image_size),
                                                                                       Map_out, batch_half, out_image_size);

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    // TODO: Copy the output back to host

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    size_t out_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);
    size_t half_out_size = out_size / 2;

    cudaMemcpyAsync(host_output, device_output, half_out_size, cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(host_output + (half_out_size / sizeof(float)), device_output + (half_out_size / sizeof(float)), half_out_size, cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaHostUnregister((void*)static_host_input);
    cudaHostUnregister((void*)static_host_mask);
    cudaHostUnregister((void*)static_host_output);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    // TODO: Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
}

__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
