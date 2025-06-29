# CUDA Convolutional Layer Optimization Project

## Overview

This project focuses on implementing and optimizing the forward pass of convolutional layers using CUDA on NVIDIA GPUs. It operates on a modified LeNet-5 neural network using the Fashion MNIST dataset (10,000 grayscale 86×86 images). The work includes CPU and GPU implementations, profiling, kernel fusion, and advanced optimization strategies.

The main goals are:
- Implement fast convolution kernels on GPU using CUDA
- Profile kernel execution using `nsys` and `ncu`
- Fuse multiple kernels to reduce overhead
- Apply advanced CUDA optimizations (streams, tensor cores, tiling, etc.)

---

## 📌 Architecture

- **Model**: Modified LeNet-5 (inference on layers C1 and C3)
- **Framework**: [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp)
- **Dataset**: Fashion MNIST (86×86, single channel, 10 classes)
- **Execution**: Designed for A40-class NVIDIA GPUs on Delta cluster

---

## 🚀 Milestone 1 – Baseline Implementations

### Objective
Implement CPU and GPU versions of convolution:

- CPU-only reference
- Basic CUDA kernel
- Input unrolling + matrix multiplication on GPU

### Files
```
project/src/layer/custom/cpu-new-forward.cc  
project/src/layer/custom/new-forward.cu  
project/src/layer/custom/unroll-new-forward.cu
```

### Run
```bash
./run.sh build
sbatch m1_cpu.slurm
sbatch m1_gpu.slurm
sbatch m1_unroll.slurm
```

### Sample Output
```text
Test batch size: 100  
Op Time: 1451.97 ms  
Test Accuracy: 0.86
```

---

## 🛠 Milestone 2 – Profiling & Kernel Fusion

### Objective
- Profile CPU and GPU implementations
- Fuse unrolling + matmul + permutation into a single kernel

### Profiling Tools
- `gprof` (CPU)
- `nsys` (GPU system-wide)
- `ncu` (GPU kernel-specific)
- `compute-sanitizer` (error checking)

### File
```
project/src/layer/custom/kernel-fusion-forward.cu
```

### Commands
```bash
# CPU
cmake -DCMAKE_CXX_FLAGS=-pg ./project/ && make
srun ./m1_cpu 1000 && gprof -Q ./m1_cpu gmon.out > profile.txt

# GPU profiling
srun nsys profile --stats=true ./m1_gpu > profile.out
srun ncu --set full -f -o analysis_file ./m1_gpu 100
```

---

## ⚡ Milestone 3 – GPU Optimizations

### Goal
Achieve total op time ≤ **50ms** for batch size 10,000 using CUDA optimizations.

### Techniques Used

- Asynchronous streams to overlap data transfers and computation
- Tensor Core acceleration using TF32 instructions
- Memory layout optimizations (constant memory, shared memory tiling)
- Loop unrolling and thread coarsening
- `__restrict__` qualifiers for pointer aliasing
- FP16 usage with precision controls
- Tuning of block/grid dimensions
- cuBLAS integration for matrix multiplication in unrolled input layouts

### Final File
```
project/src/layer/custom/m3-forward.cu
```

### Final Run
```bash
./run.sh build
sbatch m3.slurm
```

---

## 🧪 Testing & Accuracy Targets

| Batch Size | Expected Accuracy |
|------------|-------------------|
| 100        | ~0.86             |
| 1,000      | ~0.886            |
| 10,000     | ~0.8714           |

---

## 🔍 Performance & Profiling Notes

- "Op Time" = kernel-only time
- "Layer Time" = includes memory copies + kernel
- Use `compute-sanitizer` for CUDA memory issues
- CUDA arch targeting: `set(CMAKE_CUDA_ARCHITECTURES 86)`

---

## 📂 File Tree Summary

```
project/
├── src/layer/custom/
│   ├── cpu-new-forward.cc
│   ├── new-forward.cu
│   ├── unroll-new-forward.cu
│   ├── kernel-fusion-forward.cu
│   └── m3-forward.cu
├── m3/
│   ├── req_0/ (streams)
│   ├── req_1/ (tensor cores)
│   ├── op_#/  (other opt folders)
```

---

## 📄 License

NCSA/UIUC © 2020–2025  
Developed by Carl Pearson, Vikram Mailthody, and the Illinois Vision Lab contributors.

## 👥 Contributors

Carl Pearson, Vikram Mailthody, Andrew Schuh, Abdul Dakkak, Zaid Qureshi, Rui Lan, Zhicun Wan, Ben Schreiber, James Cyriac, Jonathan Nativ, Shuangliang Chen, Huili Tao, Howie Liu, Thomas Bae, Yifei Song, Shengjie Ma
