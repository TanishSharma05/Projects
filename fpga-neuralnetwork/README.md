# FPGA Convolutional Neural Network Accelerator

A hardware-accelerated Convolutional Neural Network (CNN) inference engine implemented entirely in SystemVerilog.  
The system performs end-to-end image classification by sequencing convolution, activation, pooling, flattening, fully connected layers, and argmax-based classification.  
The design is modular and synthesizable for FPGA deployment, with result output via LEDs for real-time verification.

---

## Overview

This project implements a deep learning inference pipeline as a custom hardware accelerator.  
Images are loaded into on-chip memory, processed through convolution and pooling layers, flattened, classified with a fully connected network, and reduced to a final predicted class via an argmax unit.

The project was developed in two phases:
- **Phase 1**: Core pipeline (convolution, activation, pooling, flatten, fully connected, argmax).  
- **Phase 2**: Control integration with FSM sequencing, patch extraction for convolution, activation map buffering, and LED-driven output.

---

## Key Features

- **Fully modular CNN pipeline**: convolution, activation (ReLU), pooling, flattening, fully connected, classification.  
- **Patch extractor + kernel bank** for efficient 3×3 sliding-window convolution.  
- **BRAM-backed image storage** for 28×28 grayscale input images.  
- **Activation map buffer** for 26×26 convolution outputs.  
- **Pooling window controller** for 2×2 stride-2 downsampling → 13×13 pooled maps.  
- **Flattener** for reshaping pooled maps into a 169-element vector.  
- **LED-driven result output** with one-hot encoded class selection.  
- **Controller FSM** to sequence operations across the accelerator.  
- **Synthesizable SystemVerilog modules** with clean top-level integration.  
- **Target use case**: digit classification (e.g., MNIST).  

---

## Hardware Platform

The design is FPGA-agnostic and can be deployed to platforms such as Xilinx or Intel FPGAs.  
It relies only on:
- Standard FPGA primitives (BRAM, DSP slices).  
- A single top-level wrapper (`top_module.sv`) that integrates all functional modules.  
- LED I/O for classification results.  

---

## Architecture

### Components

- **Memory**
  - `bram_image_memory.sv` – Block RAM interface for storing 28×28 input images.  

- **Convolutional Pipeline**
  - `patch_extractor.sv` – Scans input and prepares 3×3 convolution patches.  
  - `conv.sv` – Performs kernel convolution over patches.  
  - `activation_unit.sv` – Applies non-linearity (ReLU).  
  - `kernel_bank.sv` – Stores convolutional kernels.  

- **Pooling**
  - `pooling_unit.sv` – Max/average pooling implementation.  
  - `pooling_window_controller.sv` – Traverses 2×2 pooling windows across the 26×26 activation map.  

- **Dense Layer**
  - `flattener.sv` – Reshapes 13×13 pooled map into a 169-element vector.  
  - `fc_layer.sv` – Fully connected layer producing 2 class scores.  

- **Classification**
  - `argmax_unit.sv` – Selects highest-scoring class index.  
  - `led_driver.sv` – Displays class result on LEDs.  

- **Control**
  - `controller_fsm.sv` – Finite state machine for orchestrating CNN stages.  
  - `top_module.sv` – System integration and I/O handling.  

---

## Processing Flow

1. **Image Load**  
   28×28 grayscale image data is written to on-chip BRAM.  

2. **Patch Extraction**  
   The FSM scans all **676 (26×26)** valid 3×3 patches.  

3. **Convolution + Activation**  
   Each patch is convolved with kernels from the kernel bank, then passed through the activation unit.  

4. **Activation Map**  
   Outputs are stored in a 26×26 feature map buffer.  

5. **Pooling**  
   Pooling reduces the activation map to a 13×13 pooled feature map.  

6. **Flattening**  
   The flattener reshapes the pooled map into a 169-element vector.  

7. **Fully Connected Layer**  
   Matrix-vector multiplication produces 2 class logits.  

8. **Argmax**  
   The highest logit determines the predicted class.  

9. **LED Output**  
   The LED driver lights the corresponding output.  

10. **Control FSM**  
    Sequences each stage, ensuring proper timing and synchronization.  

---

## Simulations

The `top_module_tb.sv` testbench validates the full accelerator:

- Generates clock/reset.  
- Loads 28×28 image data into BRAM.  
- Runs the FSM through convolution → pooling → flatten → FC → argmax.  
- Captures waveforms for intermediate signals.  
- Checks LED outputs against expected classification.  

---

## Top-Level Files

| Module                      | Purpose                                    |
|------------------------------|--------------------------------------------|
| `top_module.sv`              | Integration of all CNN modules             |
| `controller_fsm.sv`          | FSM for sequencing pipeline stages         |
| `bram_image_memory.sv`       | BRAM storage for image data                |
| `patch_extractor.sv`         | 3×3 patch extraction logic                 |
| `conv.sv`                    | Convolutional computation                  |
| `activation_unit.sv`         | ReLU non-linear activation                 |
| `kernel_bank.sv`             | Preloaded convolution kernels              |
| `pooling_unit.sv`            | Max/average pooling                        |
| `pooling_window_controller.sv` | Pooling window traversal                 |
| `flattener.sv`               | Reshape pooled map into 1D vector (169)    |
| `fc_layer.sv`                | Fully connected classification (2 outputs) |
| `argmax_unit.sv`             | Final class selection                      |
| `led_driver.sv`              | LED result output                          |

---

## Running the System

1. **Load image data** into BRAM using the testbench or FPGA programming environment.  
2. **Program FPGA** with the synthesized bitstream (`top_module.sv` as top).  
3. **Pulse the `start` signal** to begin processing.  
4. **Observe LED outputs** for classification result (LED0 = class 0, LED1 = class 1).  
5. Extend testbenches to validate intermediate layers for debugging.  

---

## Final Thoughts

This project demonstrates a full end-to-end CNN pipeline mapped to hardware, scaled from a toy 4×4 example to a real 28×28 use case.  
The modular design makes it portable and scalable across FPGA platforms.  
While simplified for digit classification (e.g., MNIST), the same architecture can be extended with multiple kernels, deeper layers, or quantized arithmetic to fit FPGA resource budgets.  
