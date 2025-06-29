#include <cmath>
#include <iostream>
#include <cuda.h>
#include "gpu-new-forward.h"

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output, int Batch, int Map_out, int Channel, int Height, int Width, int K, int tile_size, int coarsen)
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

    int imgIdx = blockIdx.z;
    int colIdx = blockIdx.x * tile_size + threadIdx.x;
    int baseY  = blockIdx.y * tile_size + threadIdx.y * coarsen;

    extern __shared__ float shmem[];
    float *tileA = shmem;
    float *tileB = shmem + tile_size * tile_size;

    // coearsening factors
    float tv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int numChunks = (Acols + tile_size - 1) / tile_size;

    for (int chunk = 0; chunk < numChunks; ++chunk) {
        // load tiles for each sub-row
        for (int i = 0; i < coarsen; ++i) {
            int rowIdx = baseY + i;
            int mcol   = chunk * tile_size + threadIdx.x;
            float a_value = (rowIdx < Map_out && mcol < Acols) ? mask[rowIdx * Acols + mcol] : 0.0f;
            tileA[(threadIdx.y * coarsen + i) * tile_size + threadIdx.x] = a_value;
            int inRow = chunk * tile_size + (threadIdx.y * coarsen + i);
            float bVal = 0.0f;
            if (inRow < Acols && colIdx < Bcols) {
                int ch   = inRow / (K * K);
                int rem  = inRow % (K * K);
                int p    = rem / K;
                int q    = rem % K;
                int outY = colIdx / outW;
                int outX = colIdx % outW;
                int iY   = outY + p;
                int iX   = outX + q;
                if(iY < Height && iX < Width){
                    int idx = ((imgIdx * Channel + ch) * Height + iY) * Width + iX;
                    bVal = input[idx];
                }
            }
            tileB[(threadIdx.y * coarsen + i) * tile_size + threadIdx.x] = bVal;
        }
        __syncthreads();



        // for(int k = 0; k < TILE_SIZE; ++k) {
        //     total += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        // }
        // compute partial dot for each sub-row
        //#pragma unroll 
        for (int k = 0; k < tile_size; ++k) {
            for (int t = 0; t < coarsen; ++t) {
                int rowBase = (threadIdx.y * coarsen + t) * tile_size;
                tv[t] += tileA[rowBase + k] *
                                   tileB[k * tile_size + threadIdx.x];
            }
        }
        __syncthreads();
    }

    // write back the results
    for (int i = 0; i < coarsen; ++i) {
        int rowIdx = baseY + i;
        if (rowIdx < Map_out && colIdx < Bcols) {
            int outIdx = ((imgIdx * Map_out + rowIdx) * Bcols) + colIdx;
            output[outIdx] = tv[i];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel,const int Height, const int Width, const int K)
{
    int oh = Height - K + 1;
    int ow = Width  - K + 1;
    size_t outS  = (size_t)Batch * Map_out * oh * ow * sizeof(float);
    size_t inS   = (size_t)Batch * Channel * Height * Width * sizeof(float);
    size_t maskS = (size_t)Map_out * Channel * K * K * sizeof(float);

    cudaMalloc(device_output_ptr, outS);
    cudaMalloc(device_input_ptr,  inS);
    cudaMalloc(device_mask_ptr,   maskS);

    cudaMemcpy(*device_input_ptr,  host_input, inS,   cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,   host_mask,  maskS, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int oh    = Height - K + 1;
    int ow    = Width  - K + 1;
    int Bcols = oh * ow;

    int tile_sizes[]   = {8, 16, 32};
    int coarsens[]     = {1, 2, 4};
    float best_ms = 1e9;
    int   best_ts = tile_sizes[0];
    int   best_cf = coarsens[0];

    for (int i = 0; i < 3; ++i) {
        int ts = tile_sizes[i];
        for (int j = 0; j < 3; ++j) {
            int cf = coarsens[j];
            dim3 block(ts, ts/cf, 1);
            dim3 grid((Bcols + ts - 1) / ts,
                      (Map_out + ts - 1) / ts,
                      Batch);
            size_t smem = 2 * ts * ts * sizeof(float);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            matmul_conv_fused<<<grid, block, smem>>>(
                device_mask, device_input, device_output,
                Batch, Map_out, Channel, Height, Width, K,
                ts, cf);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            printf("[op_3] ts=%2d cf=%d → %7.3f ms\n", ts, cf, ms);

            if(ms < best_ms){
                best_ms = ms;
                best_ts = ts;
                best_cf = cf;
            }
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    {
        dim3 block(best_ts, best_ts/best_cf, 1);
        dim3 grid((Bcols + best_ts - 1) / best_ts,
                  (Map_out + best_ts - 1) / best_ts,
                  Batch);
        size_t smem = 2 * best_ts * best_ts * sizeof(float);

        matmul_conv_fused<<<grid, block, smem>>>(
            device_mask, device_input, device_output,
            Batch, Map_out, Channel, Height, Width, K,
            best_ts, best_cf);
    }
}

// Host epilog: copy back and free
__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int oh = Height - K + 1;
    int ow = Width  - K + 1;
    size_t outS = (size_t)Batch * Map_out * oh * ow * sizeof(float);
    cudaMemcpy(host_output, device_output, outS, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

// Query device props
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
