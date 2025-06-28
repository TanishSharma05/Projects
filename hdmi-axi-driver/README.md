# HDMI Text Mode Controller with AXI4 Interface

A fully functional HDMI text rendering system built around a MicroBlaze soft processor and a custom AXI4-Lite-compatible IP core. This project outputs an 80x30 character grid to a 640×480 HDMI display using embedded VRAM, font ROM, and per-character formatting. The design is implemented in SystemVerilog and deployed on the **RealDigital Urbana** FPGA development board.

---

## Overview

This system implements a character-based HDMI output interface using a memory-mapped VRAM and AXI4-Lite bus communication. A MicroBlaze processor writes character and formatting data into VRAM, and the HDMI controller continuously renders it to screen in real time.

The project was developed in two phases:
- **Phase 1**: Basic text rendering with global foreground/background colors and inversion support.
- **Phase 2**: Enhanced per-character color formatting with palette lookup and dual-port BRAM.

---

## Key Features

- **AXI4-Lite interface** for register-level control via MicroBlaze.
- **Custom IP block** for HDMI rendering logic, including font ROM and color mapper.
- **640x480 HDMI video output** rendered at 60Hz refresh.
- **80x30 text grid**, with individual glyphs mapped using ROM lookup.
- **Per-character attributes**: glyph index, inversion, foreground/background colors.
- **Dual-port BRAM VRAM** for concurrent read/write by HDMI and AXI sides.
- **Full simulation and on-board hardware verification**.

---

## Hardware Platform

All components were synthesized and tested on the **RealDigital Urbana Board**, which provides HDMI output, onboard UART, and suitable FPGA resources for soft processor + custom logic development.

---

## Architecture

### Components

- **MicroBlaze Processor**: Issues AXI transactions to populate VRAM and update control registers.
- **AXI Interconnect**: Routes transactions between MicroBlaze and peripherals.
- **HDMI Text Controller IP**:
  - AXI interface logic (address decode, handshaking)
  - Font ROM for character glyphs
  - Palette mapper for 3-bit color indices → 8-bit RGB
  - BRAM-based VRAM with 16-bit entries per character
  - VGA timing generator
  - HDMI serialization and output
- **UART Interface**: For test input and string display via serial terminal.

---

## AXI Integration

The HDMI controller IP supports:
- 600 VRAM registers (each 32-bit word holds 4 characters in Phase 1)
- Global control register for color (Phase 1) → replaced with per-character color fields (Phase 2)
- Byte-wise write support via `WSTRB`
- BRAM read pipeline and AXI-compliant handshaking
- Address decoding for both character data and palette values

---

## VRAM Format (Phase 2)

Each 16-bit character entry encodes:
- 7 bits → Glyph index  
- 1 bit → Inverse flag  
- 3 bits → Foreground color index  
- 3 bits → Background color index  
- 2 bits → Reserved  

---

## Render Flow

1. **DrawX/DrawY** pixel coordinates drive character cell index.
2. **Character index** is used to look up glyph in font ROM.
3. **Color indices** are mapped via a palette module.
4. **Inverse bit** optionally flips foreground/background.
5. Pixel value is generated and pushed to HDMI output.

---

## Simulations

- **Write/Read AXI Transactions**: Full timing-accurate verification.
- **BMP Output Snapshots**: Validated both character alignment and color output.
- **Testbench Automation**: Used `axi_write()` and `axi_read()` tasks to preload VRAM and verify palette logic.

---

## Resource Usage

| Phase       | LUTs | FFs  | BRAM | DSP | Freq (MHz) | Power (W) |
|-------------|------|------|------|-----|------------|------------|
| **Phase 1** | 15,688 | 21,028 | 32   | 3   | ~112.5     | 0.485      |
| **Phase 2** | 2,507  | 1,904  | 34   | 3   | ~112.0     | 0.456      |

---

## Top-Level Files

| Module                         | Purpose                                                                 |
|--------------------------------|-------------------------------------------------------------------------|
| `mb_usb_hdmi_top.sv`           | Integration with RealDigital I/O, HDMI output, and UART                 |
| `hdmi_text_controller_v1_0.sv` | Top-level IP for HDMI rendering with AXI interface                      |
| `hdmi_text_controller_v1_0_AXI.sv` | AXI bus logic and register decode                                    |
| `bram_mem.sv`                  | Dual-port BRAM used as video memory                                     |
| `color_mapper.sv`              | Converts character attributes to RGB values                             |
| `font_rom.sv`                  | Stores glyph bitmaps                                                    |
| `vga_controller.sv`            | VGA timing generation for 640x480@60Hz                                  |
| `instantiate_ram.sv`           | Initializes VRAM contents at boot                                       |

---

## Development Insights

- Moving from simple GPIO control to a full AXI4-Lite interface significantly improved structure and scalability.
- The AXI protocol introduced a complexity layer but allowed clean memory-mapped peripheral interaction.
- Dual-port BRAM enabled seamless concurrent access by video renderer and AXI bus.
- Per-character color attributes and palette mapping created a flexible rendering pipeline without major performance tradeoffs.

---

## Running the System

1. Synthesize and program the bitstream to the RealDigital Urbana board.
2. Use UART or MicroBlaze application to populate VRAM and control registers.
3. Observe rendered text on an HDMI display.
4. Update content live using serial communication or automation scripts.

---

## Final Thoughts

This project demonstrates a scalable approach to video system design using standard AXI communication and modular hardware components. The HDMI controller IP is portable and configurable, with room to expand into sprite-based or interactive systems. The RealDigital Urbana board provided the right platform to bridge hardware and software workflows, enabling real-time testing and visual output for every stage of development.
