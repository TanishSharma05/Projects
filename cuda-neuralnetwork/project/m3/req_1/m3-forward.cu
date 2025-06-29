#include <cmath>
#include <iostream>
#include <mma.h>
#include <cuda_runtime.h>
#include "gpu-new-forward.h"

using namespace nvcuda;

#define TILE_SIZE 16

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                       int Batch, int Map_out, int Channel, int Height, int Width, int K){

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
    
    int outH = Height-K+1;
    int outW = Width-K+1;
    int Bcols = outH*outW;
    int Acols = Channel*K*K;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int warpM = (threadIdx.x/32)%2;
    int warpN = (threadIdx.x/32)/2;
    int laneId = threadIdx.x%32;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 8;
    int row = by*WMMA_M;
    int col = bx*WMMA_N;
    __shared__ float s_a[WMMA_M*WMMA_K];
    __shared__ float s_b[WMMA_K*WMMA_N];
    __shared__ float s_c[WMMA_M*WMMA_N];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    int tid = threadIdx.x;

    
    for(int tileK=0; tileK<(Acols+WMMA_K-1)/WMMA_K; ++tileK){
        
        for(int idx=tid; idx<WMMA_M*WMMA_K; idx+=blockDim.x){
            int i = idx/WMMA_K;
            int j = idx%WMMA_K;
            int mask_i = row+i;
            int mask_j = tileK*WMMA_K+j;
            if(mask_i<Map_out && mask_j<Acols){
                s_a[i*WMMA_K+j] = mask[mask_i*Acols+mask_j];
            }
            else{
                s_a[i*WMMA_K+j] = 0.0f;
            }
        }
        for(int idx=tid; idx<WMMA_K*WMMA_N; idx+=blockDim.x){
            int i = idx/WMMA_N;
            int j = idx%WMMA_N;

            int kOffset = tileK*WMMA_K+i;
            if(kOffset<Acols){
                int c = kOffset/(K*K);
                int kIdx = kOffset%(K*K);
                int kh = kIdx/K;
                int kw = kIdx%K;

                int outIdx = col+j;
                int oh = outIdx/outW;
                int ow = outIdx%outW;

                int h = oh+kh;
                int w = ow+kw;

                if(col+j<Bcols && c<Channel && h<Height && w<Width){
                    int input_idx = ((bz*Channel+c)*Height+h)*Width+w;
                    s_b[i*WMMA_N+j] = input[input_idx];
                }else{
                    s_b[i*WMMA_N+j] = 0.0f;
                }
            }else{
                s_b[i*WMMA_N+j] = 0.0f;
            }
        }

        __syncthreads();

        // Perform matrix multiplication using tensor cores
        if(tid<32){
            wmma::load_matrix_sync(a_frag, s_a, WMMA_K);
            wmma::load_matrix_sync(b_frag, s_b, WMMA_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    
    if(tid<32){
        wmma::store_matrix_sync(s_c, c_frag, WMMA_N, wmma::mem_row_major);
    }

    __syncthreads();

   
    for(int idx=tid; idx<WMMA_M*WMMA_N; idx+=blockDim.x){
        int i = idx/WMMA_N;
        int j = idx%WMMA_N;

        int globalRow = row+i;
        int globalCol = col+j;

        if(globalRow<Map_out && globalCol<Bcols){
            output[(bz*Map_out+globalRow)*Bcols+globalCol] = s_c[i*WMMA_N+j];
        }
    }
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K)
{
    int oh = Height - K + 1;
    int ow = Width - K + 1;
    size_t outS = Batch * Map_out * oh * ow * sizeof(float);
    size_t inS = Batch * Channel * Height * Width * sizeof(float);
    size_t maskS = Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void**) device_output_ptr, outS);
    cudaMalloc((void**) device_input_ptr, inS);
    cudaMalloc((void**) device_mask_ptr, maskS);

    cudaMemcpy(*device_input_ptr, host_input, inS, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, maskS, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K)
{
    // Calculate output dimensions
    int oh = Height - K + 1;
    int ow = Width - K + 1;
    
    // Configure kernel execution parameters for tensor cores
    // Use 128 threads per block (4 warps) for better occupancy
    dim3 blockDim(128, 1, 1);
    
    // Calculate grid dimensions to cover the entire output
    dim3 gridDim((ow * oh + TILE_SIZE - 1) / TILE_SIZE, 
                 (Map_out + TILE_SIZE - 1) / TILE_SIZE, 
                 Batch);

    matmul_conv_fused<<<gridDim, blockDim>>>(device_mask, device_input, device_output,
                                                  Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K)
{
    int oh = Height - K + 1;
    int ow = Width - K + 1;
    size_t outS = Batch * Map_out * oh * ow * sizeof(float);

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