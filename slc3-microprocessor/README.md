# SLC-3.2 Microprocessor in SystemVerilog

A custom 16-bit microprocessor architecture, SLC-3.2, designed and implemented to explore CPU design principles, instruction execution, and hardware interfacing. The project is built in SystemVerilog and deployed on the **RealDigital Urbana** FPGA board.

---

## Project Overview

SLC-3.2 is a reduced-instruction-set CPU modeled after the LC-3 architecture. It implements a full fetch-decode-execute pipeline, a control FSM, and supports memory-mapped I/O for real-time interaction with board components like switches, LEDs, and 7-segment displays.

This setup allowed direct deployment and testing on the Urbana board, enabling live interaction and visual output verification through physical controls and displays.

---

## Instruction Set Summary

Implemented instructions:

- **Arithmetic & Logic**: `ADD`, `ADDi`, `AND`, `ANDi`, `NOT`
- **Branch & Control**: `BR`, `JMP`, `JSR`
- **Memory**: `LDR`, `STR`
- **System**: `PAUSE`

Each is a 16-bit instruction, designed to operate with an 8-register file and standard condition codes (N, Z, P).

---

## Architecture Highlights

### Core Components

- **ALU**: 16-bit, supporting basic operations
- **Register File**: 8 general-purpose 16-bit registers
- **PC/IR/MAR/MDR/NZP**: Typical microarchitecture registers
- **Datapath**: Shared bus system managed via priority-based MUXes (no tristate logic)
- **Control Unit**: Finite state machine sequencing all operations
- **I/O Bridge**: Routes I/O through memory-mapped addresses

### RealDigital Urbana Integration

- Switches and buttons used for input
- LEDs and HEX displays provide real-time visual output
- Pause and Continue signals mapped to board controls
- Deployed and validated directly on Urbana FPGA hardware

---

## Execution Cycle

1. **Fetch**: PC → MAR, Memory → MDR → IR
2. **Decode**: IR fields decoded, control lines prepared
3. **Execute**: ALU or memory operation performed, result stored or control transferred

This cycle loops continuously unless paused by system instruction.

---

## Simulated & On-Board Programs

- **I/O Mirror**: Real-time switch state reflected on HEX display
- **Pause-Driven I/O**: User input gated with `continue` signal
- **Self-Modifying Code**: Instruction memory altered during execution
- **XOR & Multiplier**: Input-driven bitwise/arithmetic operations with results shown on display
- **Auto Counter**: Continuous software-loop-based counting
- **Bubble Sort**: Menu-based input, sort, and output routine

All programs validated both in simulation and on hardware via the Urbana board.

---

## Key Specs

| Metric           | Value           |
|------------------|-----------------|
| Clock Freq.      | ~103.7 MHz      |
| LUTs Used        | 369             |
| Flip-Flops       | 345             |
| BRAM             | 1 Block         |
| Power (Total)    | ~0.09 W         |

---

## Core Files

| Module             | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `slc3.sv`          | Processor core with I/O and memory interface                 |
| `cpu.sv`           | Datapath + FSM-based control                                 |
| `control.sv`       | Instruction-specific control logic FSM                       |
| `cpu_to_io.sv`     | Memory-mapped routing to switches, HEX, and LED I/O         |
| `memory.sv`        | Dual-mode memory with simulation and synthesis support       |
| `instantiate_ram.sv`| Initializes memory from file or predefined contents         |
| `ALU.sv`, `register.sv`, `ben.sv` | Core compute and storage logic               |

---

## Design Notes

- Tri-state buses replaced with safe, synthesisable MUX logic
- Manual wait-state insertion compensates for lack of memory ready signal
- Fully modular design for ease of testing, extending, or replacing units
- Interfacing cleanly integrated with RealDigital Urbana's hardware features

---

## Running the System

1. Synthesize the design targeting the RealDigital Urbana board
2. Initialize program memory with `.mif` or preload logic
3. Use switches/buttons for input and control
4. View results via HEX display and LEDs
5. For debugging, simulate with waveform tracing in ModelSim or equivalent

---

## Final Thoughts

SLC-3.2 delivers a functional, understandable microprocessor environment on a real FPGA platform. By keeping the architecture lean and fully testable, it serves both as a working CPU and a platform for exploring ideas like branching, self-modifying code, and user-interfaced control. The RealDigital Urbana board proved to be a reliable and responsive testbed throughout development.

