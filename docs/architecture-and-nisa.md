# Architecture & NISA ISA Specification

`Transformer-Oracle` is an independent execution engine that executes arbitrary programs by compiling them into a 40-opcode RISC-V derivative instruction set architecture (**NISA**) and executing one instruction per forward pass on a 10-layer analytical transformer neural network.

---

## 1. System Architecture Overview

```
                      [ High-Level Language ]
                 (C / Rust / Lean Emitted Code)
                                │
                                ▼
                   [ RV32I Machine Target ]
               (riscv64-linux-gnu-gcc / rustc)
                                │
                                ▼
                     [ NISA Translator ]
              (RV32I -> 40-Opcode NISA IR)
                                │
                                ▼
             ┌────────────────────────────────────┐
             │   Analytical Transformer Executor  │
             │     (10-Layer Bipolar Tensor VM)    │
             └────────────────────────────────────┘
                                │
                      [ State Tensor ]
                (PC, Registers, Memory, Flags)
```

---

## 2. The NISA Instruction Set (40 Opcodes)

NISA (**Neural Instruction Set Architecture**) is a 40-opcode RISC-V-derived 32-bit ISA engineered for exact neural execution.

### Register File
* **`r0` (`x0`)**: Hardwired zero.
* **`r1`–`r29`**: General-purpose registers (`ilp32` ABI compatible).
* **`r30`, `r31` (`x30`, `x31`)**: Reserved scratch registers (`_TMP1`, `_TMP2`) used internally by the NISA translator for multi-step expansion.

### Core Instruction Categories
1. **Arithmetic & Logic**: `ADD`, `SUB`, `AND`, `OR`, `XOR`, `SLT`, `SLTU`, `MUL`, `DIV`, `REM` (Immediate & Register variants).
2. **Shifts**: `SLL`, `SRL`, `SRA` (Logical left, Logical right, Arithmetic right).
3. **Memory Access**: `LW` (Load Word), `SW` (Store Word), `LB`, `SB`, `LH`, `SH`.
4. **Control Flow**: `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`, `JAL`, `JALR`.
5. **Special**: `NOP`, `HALT`.

---

## 3. The Bipolar Tensor State Encoding

All state in the Transformer VM (registers, memory, program counter, and opcode flags) is encoded using a **bipolar representation**:

$$\text{Bit } b \in \{0, 1\} \implies s \in \{-1, +1\} \quad \text{where } s = 2b - 1$$

* $0 \to -1$
* $1 \to +1$

This encoding allows boolean logic gates (AND, OR, XOR, NOT, MUX) and multi-bit adders/shifters to be expressed as exact linear transformations (matrix multiplications) combined with sign activation functions $\text{sgn}(x)$, completely eliminating floating-point rounding errors.

---

## 4. The 10-Layer Analytical Transformer Pipeline

Every forward pass of the 10-layer transformer performs **exactly 1 CPU clock cycle** of instruction execution:

```
Cycle Step                              Layer Action
─────────────────────────────────────────────────────────────────────────────────────────────
Instruction Fetch    ──▶  Layer 1: Lookup instruction at current PC tensor.
Regfile Read         ──▶  Layer 2-3: One-hot attention selection to read rs1 & rs2 into latches.
ALU Execution        ──▶  Layer 4-6: Bipolar logic & arithmetic gates (Carry-chain / Shifter).
Memory Access        ──▶  Layer 7: Load/Store buffer addressing & SRAM tensor read/write.
Regfile Writeback    ──▶  Layer 8: MUX selection to write ALU/Mem result into rd.
PC Update            ──▶  Layer 9: Compute next PC (sequential PC+4 or branch/jump target).
Error Correction     ──▶  Layer 10: Snap continuous state values back to clean {-1, +1} bounds.
```

### Key Guarantees
* **One-Cycle Faithfulness**: 1 forward pass = 1 clock cycle.
* **Deterministic Precision**: No floating-point noise or stochastic approximations; weights are set analytically to exact integer/bipolar matrices.
* **Trace Transparency**: Every internal tensor state (registers, flags, memory) can be dumped at any cycle for debugging.
