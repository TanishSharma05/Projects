# FPGA Convolutional Neural Network Accelerator

A hardware-accelerated Convolutional Neural Network (CNN) inference engine implemented entirely in SystemVerilog.  
The system performs end-to-end image classification by sequencing convolution, activation, pooling, flattening, fully connected layers, and argmax-based classification.  
The design is modular and synthesizable for FPGA deployment, with result output via LEDs for real-time verification.

---

## Overview

This project implements a deep learning inference pipeline as a custom hardware accelerator.  
Images are loaded into on-chip memory, processed through convolutional and pooling layers, flattened, classified with a fully connected network, and reduced to a final predicted class via an argmax unit.

The project was developed in two phases:
- **Phase 1**: Core pipeline (convolution, activation, pooling, flatten, fully connected, argmax).  
- **Phase 2**: Control integration with FSM sequencing, patch extraction for convolution, and LED-driven output.

---

## Key Features

- **Fully modular CNN pipeline**: convolution, activation, pooling, flattening, fully connected, classification.  
- **Patch extractor + kernel bank** for efficient sliding-window convolution.  
- **BRAM-backed image storage** for input images and intermediate feature maps.  
- **Pooling window controller** for configurable spatial downsampling.  
- **LED-driven result output** with one-hot encoded class selection.  
- **Controller FSM** to sequence operations across the accelerator.  
- **Synthesizable SystemVerilog modules** with clean top-level integration.  
- **Target use case**: digit or small image classification (e.g., MNIST).  

---

## Hardware Platform

The design is FPGA-agnostic and can be deployed to platforms such as Xilinx or Intel FPGAs.  
It relies only on:
- Standard FPGA primitives (BRAM, DSP slices).  
- A single top-level wrapper (`top_level.sv`) that integrates all functional modules.  
- LED I/O for classification results.  

---

## Architecture

### Components

- **Memory**
  - `bram_image_memory.sv` – Block RAM interface for image storage.  
  - `image_memory.sv` – General-purpose feature map storage.  

- **Convolutional Pipeline**
  - `patch_extractor.sv` – Scans input and prepares convolution patches.  
  - `conv.sv` – Performs kernel convolution over patches.  
  - `activation_unit.sv` – Applies non-linearity (e.g., ReLU).  
  - `kernel_bank.sv` – Stores convolutional kernels.  

- **Pooling**
  - `pooling_unit.sv` – Max/average pooling implementation.  
  - `pooling_window_controller.sv` – Traverses pooling windows.  

- **Dense Layer**
  - `flattener.sv` – Reshapes feature maps into 1D vectors.  
  - `fc_layer.sv` – Fully connected layer for classification scores.  

- **Classification**
  - `argmax_unit.sv` – Selects highest-scoring class index.  
  - `led_driver.sv` – Displays class result on LEDs.  

- **Control**
  - `controller_fsm.sv` – Finite state machine for orchestrating CNN stages.  
  - `top_level.sv` – System integration and I/O handling.  

---

## Processing Flow

1. **Image Load**  
   Image data is written to on-chip BRAM.  

2. **Patch Extraction**  
   Input is scanned into overlapping convolution windows.  

3. **Convolution + Activation**  
   Each patch is convolved with kernels from the kernel bank, then passed through the activation unit.  

4. **Pooling**  
   Pooling reduces feature map resolution with configurable stride and mode.  

5. **Flattening**  
   The flattener reshapes feature maps into a flat vector.  

6. **Fully Connected Layer**  
   Matrix-vector multiplication produces class logits.  

7. **Argmax**  
   The highest logit determines the predicted class.  

8. **LED Output**  
   The LED driver lights the corresponding output.  

9. **Control FSM**  
   Sequences each stage, ensuring proper timing and synchronization.  

---

## Simulations

The `top_module_tb.sv` testbench validates the full accelerator:

- Generates clock/reset.  
- Loads image data into BRAM and initializes kernels.  
- Runs the FSM through convolution → pooling → FC → argmax.  
- Captures waveforms for intermediate signals.  
- Checks LED outputs against expected classification. 
 
--

## Top-Level Files

| Module               | Purpose                                    |
|-----------------------|--------------------------------------------|
| `top_level.sv`       | Integration of all CNN modules              |
| `controller_fsm.sv`  | FSM for sequencing pipeline stages           |
| `bram_image_memory.sv` | BRAM storage for image data               |
| `conv.sv`            | Convolutional computation                   |
| `activation_unit.sv` | Non-linear activation function              |
| `pooling_unit.sv`    | Max/average pooling                         |
| `flattener.sv`       | Reshape feature maps into 1D vector         |
| `fc_layer.sv`        | Fully connected classification              |
| `argmax_unit.sv`     | Final class selection                       |
| `led_driver.sv`      | LED result output                           |

---

## Running the System

1. **Load image data** into BRAM using the testbench or FPGA programming environment.  
2. **Program FPGA** with the synthesized bitstream (`top_level.sv` as top).  
3. **Observe LED outputs** for classification result.  
4. Extend testbenches to validate intermediate layers for debugging.  

---

## Final Thoughts

This project demonstrates a full end-to-end CNN pipeline mapped to hardware.  
The modular approach makes it portable and scalable across FPGA platforms.  
While simplified for tasks such as MNIST, the same architecture can be expanded with deeper networks, larger feature maps, or quantized arithmetic to fit FPGA resource constraints.
