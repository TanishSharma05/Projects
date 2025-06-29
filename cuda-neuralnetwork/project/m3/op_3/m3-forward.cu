#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_SIZE 16
#define COARSEN   4

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{

    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.
    
    Function parameter definitions:
    mask - convolution kernel
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */


    int outH  = Height - K + 1;
    int outW  = Width  - K + 1;
    int Bcols = outH * outW;
    int Acols = Channel * K * K;

    int imgIdx  = blockIdx.z;
    int rowIdx  = blockIdx.y * TILE_SIZE + threadIdx.y;
    int baseCol = blockIdx.x * (TILE_SIZE * COARSEN) + threadIdx.x;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float total[COARSEN] = {0.0f};

    int numChunks = (Acols + TILE_SIZE - 1) / TILE_SIZE;

    for (int chunk = 0; chunk < numChunks; ++chunk) {
        int maskCol = chunk * TILE_SIZE + threadIdx.x;
        if(rowIdx < Map_out && maskCol < Acols){
            tileA[threadIdx.y][threadIdx.x] = mask[rowIdx * Acols + maskCol];
        } 
        else{
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        };

        __syncthreads();

        int inRow = chunk * TILE_SIZE + threadIdx.y;
        int ch    = inRow / (K * K);
        int rem   = inRow % (K * K);
        int p     = rem / K;
        int q     = rem % K;

        for (int c = 0; c < COARSEN; ++c) {
            int colIdx = baseCol + c * TILE_SIZE;
            float v = 0.0f;
            if (inRow < Acols && colIdx < Bcols) {
                int outY = colIdx / outW;
                int outX = colIdx % outW;
                int inY  = outY + p;
                int inX  = outX + q;
                if (inY < Height && inX < Width) {
                    size_t idx = ((size_t)imgIdx * Channel + ch) * Height * Width
                               + inY * Width + inX;
                    v = input[idx];
                }
            }
            tileB[threadIdx.y][threadIdx.x] = v;
            __syncthreads();

            
            for (int k = 0; k < TILE_SIZE; ++k) {
                total[c] += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            }
            __syncthreads();
        }
    }

    if (rowIdx < Map_out) {
        for (int c = 0; c < COARSEN; ++c) {
            int colIdx = baseCol + c * TILE_SIZE;
            if (colIdx < Bcols) {
                int outIdx = ((imgIdx * Map_out + rowIdx) * Bcols) + colIdx;
                output[outIdx] = total[c];
            }
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }


    int oh = Height - K + 1;
    int ow = Width  - K + 1;
    size_t outS  = (size_t)Batch * Map_out * oh * ow * sizeof(float);
    size_t inS   = (size_t)Batch * Channel * Height * Width * sizeof(float);
    size_t maskS = (size_t)Map_out * Channel * K * K * sizeof(float);

    cudaMalloc(device_output_ptr, outS);
    cudaMalloc(device_input_ptr,  inS);
    cudaMalloc(device_mask_ptr,   maskS);

    cudaMemcpy(*device_input_ptr, host_input, inS,    cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  maskS,  cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    // TODO: Set the kernel dimensions and call the fused kernel

    int oh = Height - K + 1;
    int ow = Width  - K + 1;
    int Bcols = oh * ow;

    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim((Bcols + TILE_SIZE * COARSEN - 1) / (TILE_SIZE * COARSEN),
                 (Map_out + TILE_SIZE - 1) / TILE_SIZE,
                  Batch);

    matmul_conv_fused<<<gridDim, blockDim>>>(
        device_mask, device_input, device_output,
        Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    // TODO: Copy the output back to host

    int oh = Height - K + 1;
    int ow = Width  - K + 1;
    size_t outS = (size_t)Batch * Map_out * oh * ow * sizeof(float);

    cudaMemcpy(host_output, device_output, outS, cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}