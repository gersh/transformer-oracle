# Compiler Pipeline & Multi-Language Support

`Transformer-Oracle` provides a unified compiler pipeline that converts high-level programming languages into NISA instructions for execution on the Transformer VM.

---

## 1. Supported Front-Ends

| Language | Front-End Toolchain | Target Pipeline |
| :--- | :--- | :--- |
| **C** | `gcc-riscv64-linux-gnu` | `C` $\to$ `RV32I Assembly` $\to$ `NISA IR` $\to$ `Transformer VM` |
| **Rust** | `rustc` (LLVM) | `Rust (#![no_std])` $\to$ `RV32I Assembly` $\to$ `NISA IR` $\to$ `Transformer VM` |
| **Lean 4 Emitted C** | Lean C Emitter + Mini Runtime | `Lean` $\to$ `C` $\to$ `RV32I Assembly` $\to$ `NISA IR` $\to$ `Transformer VM` |
| **Python Subset** | Internal AST Compiler (`python_compiler.py`) | `Python AST` $\to$ `NISA IR` $\to$ `Transformer VM` |

---

## 2. C Compilation Pipeline

Use `compile_and_run` to compile and execute freestanding C code:

```python
from transformer_oracle.compiler.compiler import compile_and_run

source_c = """
int _start(void) {
    int sum = 0;
    for (int i = 1; i <= 100; i++) {
        sum += i;
    }
    return sum;
}
"""

result = compile_and_run(source_c, language="c", device="cpu")
print("Register a0 (return value):", result.reg(10))  # Output: 5050
```

### Required GCC Flags
The compiler passes specific flags to ensure clean RV32I output without unsupported ABI features:
- `-march=rv32im -mabi=ilp32`
- `-nostdlib -ffreestanding -fno-builtin -fno-stack-protector`
- `-ffixed-x30 -ffixed-x31` (reserves `x30`/`x31` for NISA translator scratch registers)

---

## 3. Rust (LLVM) Compilation Pipeline

Rust code written in `#![no_std]` mode can be compiled down to RV32I using `rustc` or Cargo with the `riscv32im-unknown-none-elf` target.

### Step 1: Rust Source (`kernel.rs`)
```rust
#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[no_mangle]
pub extern "C" fn _start() -> i32 {
    let mut a = 123456789i64;
    let mut b = 987654321i64;
    (a * b % 1000000) as i32
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
```

### Step 2: Compile to RV32I Assembly
```bash
rustc --target riscv32im-unknown-none-elf \
      --emit asm \
      -C opt-level=1 \
      -C panic=abort \
      -C target-feature=-a,-f,-d,-c \
      -C llvm-args="-fixed-x30 -fixed-x31" \
      kernel.rs -o kernel.s
```

### Step 3: Run Assembly in Python
```python
from transformer_oracle.compiler.compiler import compile_and_run

with open("kernel.s", "r") as f:
    asm_code = f.read()

result = compile_and_run(asm_code, language="asm", device="cpu")
print("Return value:", result.reg(10))
```

---

## 4. API Reference

### `transformer_oracle.compiler.compiler`
- **`compile_and_run(source, language='c', device='cpu', max_cycles=300000)`**
  Compiles source code or assembly and runs it on the Transformer VM.
  - `source`: String containing C code, Python code, or RV32I assembly.
  - `language`: `"c"`, `"asm"`, or `"python"`.
  - `device`: `"cpu"`, `"cuda"`, or `"gpu"`.
  - Returns `ExecutionResult` with `.reg(index)`, `.pc`, `.cycles`, `.memory`.

- **`compile_c(source, gcc_path='riscv64-linux-gnu-gcc')`**
  Compiles C source string to RV32I assembly string using GCC.

- **`parse_rv32i_assembly(asm_text)`**
  Parses RV32I assembly text into intermediate instruction objects.

- **`translate_program(instructions)`**
  Translates intermediate RV32I instructions to NISA opcodes.
