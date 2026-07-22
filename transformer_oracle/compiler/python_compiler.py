"""
Python-subset-to-NISA compiler.

Compiles a restricted subset of Python directly to NISA instructions
using Python's ast module. No intermediate C or RISC-V step.

Supported Python subset:
  - Integer variables and arithmetic (+, -, *, &, |, ^, ~, <<, >>)
  - Comparisons (==, !=, <, >, <=, >=)
  - Boolean operators (and, or, not) — short-circuiting
  - Assignment (x = expr, a, b = b, a+b)
  - Augmented assignment (x += 1, etc.)
  - if/elif/else
  - while loops (with break, continue)
  - for x in range(n) / range(start, stop) / range(start, stop, step)
  - Function arguments (int) and return value (int)
  - Local variables (auto-allocated to registers)
  - Constants (int literals, hex)
  - Print-like output via special register (r10 = a0 = return value)

Not supported (yet):
  - Strings, lists, dicts, objects
  - Nested functions, closures
  - Recursion (no call stack)
  - Float arithmetic
  - Global variables (use function args instead)

Usage:
    @transformer_jit
    def fib(n: int) -> int:
        a, b = 0, 1
        for i in range(n):
            a, b = b, a + b
        return a

    result = fib(10)  # → 55, executed on GPU
"""

import ast
import inspect
import textwrap
import functools
from typing import Callable, Any, Optional

from ..core.nisa import Instruction, Opcode, movi, add, sub, halt, nop
from ..runtime.gpu_executor import gpu_execute


class CompileError(Exception):
    """Error during Python → NISA compilation."""
    pass


class RegisterAllocator:
    """Simple register allocator for local variables.

    Registers 1-25 are available for variables.
    Registers 26-29 are temporaries for expression evaluation.
    Register 30-31 are reserved for the translator.
    Register 0 is hardwired zero.
    """

    def __init__(self):
        self.var_to_reg: dict[str, int] = {}
        self.next_reg = 1       # first available variable register
        self.max_var_reg = 25   # last variable register
        self.temp_base = 26     # first temp register
        self.temp_count = 4     # number of temp registers (26-29)
        self.temp_stack: list[int] = []  # stack of in-use temps

    def alloc_var(self, name: str) -> int:
        """Allocate a register for a named variable."""
        if name in self.var_to_reg:
            return self.var_to_reg[name]
        if self.next_reg > self.max_var_reg:
            raise CompileError(f"Too many local variables (max {self.max_var_reg})")
        reg = self.next_reg
        self.var_to_reg[name] = reg
        self.next_reg += 1
        return reg

    def get_var(self, name: str) -> int:
        """Get the register for an existing variable."""
        if name not in self.var_to_reg:
            raise CompileError(f"Undefined variable: '{name}'")
        return self.var_to_reg[name]

    def alloc_temp(self) -> int:
        """Allocate a temporary register for expression evaluation."""
        idx = len(self.temp_stack)
        if idx >= self.temp_count:
            raise CompileError("Expression too complex (out of temp registers)")
        reg = self.temp_base + idx
        self.temp_stack.append(reg)
        return reg

    def free_temp(self, reg: int):
        """Free a temporary register."""
        if self.temp_stack and self.temp_stack[-1] == reg:
            self.temp_stack.pop()


class PythonToNISA:
    """Compiles a Python function AST to NISA instructions."""

    def __init__(self):
        self.instrs: list[Instruction] = []
        self.regs = RegisterAllocator()
        self.loop_stack: list[tuple[str, str]] = []  # (continue_label, break_label)
        self.labels: dict[str, int] = {}
        self.fixups: list[tuple[int, str]] = []  # (instr_idx, label_name)
        self._label_counter = 0
        # Support for inlining user-defined function calls:
        self._module_globals: dict = {}          # caller's globals, to resolve callees
        self._ast_cache: dict[str, ast.FunctionDef] = {}
        self._inline_ctx: list[tuple[int, str]] = []  # stack of (return_dest_reg, end_label)
        self._max_inline_depth = 16

    def compile_function(self, func: Callable) -> list[Instruction]:
        """Compile a Python function to NISA instructions.

        Args:
            func: a Python function to compile

        Returns:
            list of NISA instructions
        """
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Remember the caller's module namespace so we can resolve (and inline)
        # calls to other user-defined functions in the same module.
        self._module_globals = getattr(func, "__globals__", {}) or {}

        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                func_def = node
                break

        if func_def is None:
            raise CompileError(f"Could not find function '{func.__name__}'")

        # Allocate registers for function arguments
        for arg in func_def.args.args:
            self.regs.alloc_var(arg.arg)

        # Compile the function body
        for stmt in func_def.body:
            self._compile_stmt(stmt)

        # Add halt at the end (in case no return)
        self.instrs.append(halt())

        # Resolve label fixups
        self._resolve_fixups()

        return self.instrs

    def _new_label(self, prefix: str = "L") -> str:
        self._label_counter += 1
        return f"{prefix}_{self._label_counter}"

    def _mark_label(self, name: str):
        """Mark current position as a label."""
        self.labels[name] = len(self.instrs)

    def _emit(self, instr: Instruction):
        self.instrs.append(instr)

    def _emit_jump(self, label: str):
        """Emit a JMP with deferred label resolution."""
        self.fixups.append((len(self.instrs), label))
        self._emit(Instruction(Opcode.JMP, a=0))

    def _emit_branch(self, opcode: Opcode, a: int, b: int, label: str):
        """Emit a conditional branch with deferred label resolution."""
        self.fixups.append((len(self.instrs), label))
        self._emit(Instruction(opcode, a=a, b=b, c=0))

    def _resolve_fixups(self):
        """Resolve all deferred label references."""
        for idx, label in self.fixups:
            if label not in self.labels:
                raise CompileError(f"Undefined label: '{label}'")
            target = self.labels[label]
            instr = self.instrs[idx]
            if instr.opcode == Opcode.JMP:
                self.instrs[idx] = Instruction(Opcode.JMP, a=target)
            else:
                self.instrs[idx] = Instruction(instr.opcode, a=instr.a,
                                                b=instr.b, c=target)

    # ── Statement compilation ──

    def _compile_stmt(self, node: ast.stmt):
        if isinstance(node, ast.Assign):
            self._compile_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._compile_augassign(node)
        elif isinstance(node, ast.Return):
            self._compile_return(node)
        elif isinstance(node, ast.If):
            self._compile_if(node)
        elif isinstance(node, ast.While):
            self._compile_while(node)
        elif isinstance(node, ast.For):
            self._compile_for(node)
        elif isinstance(node, ast.Break):
            self._compile_break()
        elif isinstance(node, ast.Continue):
            self._compile_continue()
        elif isinstance(node, ast.Expr):
            # Expression statement (e.g., function call) — evaluate and discard
            if isinstance(node.value, ast.Call):
                pass  # skip standalone function calls for now
        elif isinstance(node, ast.Pass):
            self._emit(nop())
        else:
            raise CompileError(f"Unsupported statement: {type(node).__name__}")

    def _compile_assign(self, node: ast.Assign):
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                # Simple: x = expr
                reg = self.regs.alloc_var(target.id)
                self._compile_expr_into(node.value, reg)
            elif isinstance(target, ast.Tuple):
                # Tuple unpacking: a, b = expr1, expr2
                if isinstance(node.value, ast.Tuple):
                    self._compile_tuple_assign(target, node.value)
                else:
                    raise CompileError("Tuple assignment requires tuple on right side")
            else:
                raise CompileError(f"Unsupported assignment target: {type(target).__name__}")
        else:
            raise CompileError("Multiple assignment targets not supported")

    def _compile_tuple_assign(self, targets: ast.Tuple, values: ast.Tuple):
        """Compile a, b = x, y — evaluate all values first, then assign."""
        if len(targets.elts) != len(values.elts):
            raise CompileError("Tuple assignment length mismatch")

        # Evaluate all RHS into temp registers first
        temp_regs = []
        for val in values.elts:
            tmp = self.regs.alloc_temp()
            self._compile_expr_into(val, tmp)
            temp_regs.append(tmp)

        # Then assign to target variables
        for target, tmp in zip(targets.elts, temp_regs):
            if not isinstance(target, ast.Name):
                raise CompileError("Tuple elements must be names")
            reg = self.regs.alloc_var(target.id)
            self._emit(Instruction(Opcode.MOV, a=reg, b=tmp))

        # Free temps (in reverse order)
        for tmp in reversed(temp_regs):
            self.regs.free_temp(tmp)

    def _compile_augassign(self, node: ast.AugAssign):
        """Compile x += expr, x -= expr, etc."""
        if not isinstance(node.target, ast.Name):
            raise CompileError("Augmented assignment target must be a name")

        var_reg = self.regs.get_var(node.target.id)
        tmp = self.regs.alloc_temp()
        self._compile_expr_into(node.value, tmp)

        op_map = {
            ast.Add: Opcode.ADD, ast.Sub: Opcode.SUB, ast.Mult: Opcode.MUL,
            ast.BitAnd: Opcode.AND, ast.BitOr: Opcode.OR, ast.BitXor: Opcode.XOR,
            ast.LShift: Opcode.SHL, ast.RShift: Opcode.SHR,
        }
        op_type = type(node.op)
        if op_type not in op_map:
            raise CompileError(f"Unsupported augmented assignment operator: {op_type.__name__}")

        self._emit(Instruction(op_map[op_type], a=var_reg, b=var_reg, c=tmp))
        self.regs.free_temp(tmp)

    def _compile_return(self, node: ast.Return):
        """Compile return expr. Top-level returns go to r10 (a0) + halt; returns inside
        an inlined function body write the callee's result register and jump past it."""
        if self._inline_ctx:
            dest, end_label = self._inline_ctx[-1]
            if node.value is not None:
                self._compile_expr_into(node.value, dest)
            self._emit_jump(end_label)
        else:
            if node.value is not None:
                self._compile_expr_into(node.value, 10)  # a0 = return value
            self._emit(halt())

    def _compile_if(self, node: ast.If):
        else_label = self._new_label("else")
        end_label = self._new_label("endif")

        # Compile condition — branch to else if false
        self._compile_condition_false(node.test, else_label)

        # Then body
        for stmt in node.body:
            self._compile_stmt(stmt)

        if node.orelse:
            self._emit_jump(end_label)

        self._mark_label(else_label)

        # Else body
        if node.orelse:
            for stmt in node.orelse:
                self._compile_stmt(stmt)
            self._mark_label(end_label)

    def _compile_while(self, node: ast.While):
        loop_label = self._new_label("while")
        end_label = self._new_label("endwhile")

        self.loop_stack.append((loop_label, end_label))

        self._mark_label(loop_label)

        # Condition — branch to end if false
        self._compile_condition_false(node.test, end_label)

        # Body
        for stmt in node.body:
            self._compile_stmt(stmt)

        self._emit_jump(loop_label)
        self._mark_label(end_label)

        self.loop_stack.pop()

    def _compile_for(self, node: ast.For):
        """Compile for x in range(...) loops."""
        if not isinstance(node.target, ast.Name):
            raise CompileError("For loop variable must be a name")
        if not isinstance(node.iter, ast.Call):
            raise CompileError("For loop must iterate over range()")
        if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
            raise CompileError("For loop must use range()")

        var_reg = self.regs.alloc_var(node.target.id)
        args = node.iter.args

        # Parse range arguments
        if len(args) == 1:
            # range(stop): start=0, step=1
            start_val, step_val = 0, 1
            stop_tmp = self.regs.alloc_temp()
            self._compile_expr_into(args[0], stop_tmp)
            self._emit(movi(var_reg, start_val))
        elif len(args) == 2:
            # range(start, stop): step=1
            step_val = 1
            self._compile_expr_into(args[0], var_reg)
            stop_tmp = self.regs.alloc_temp()
            self._compile_expr_into(args[1], stop_tmp)
        elif len(args) == 3:
            # range(start, stop, step)
            self._compile_expr_into(args[0], var_reg)
            stop_tmp = self.regs.alloc_temp()
            self._compile_expr_into(args[1], stop_tmp)
            # step must be a constant for now (may be a negative literal, which the AST
            # represents as UnaryOp(USub, Constant) — literal_eval folds that correctly)
            try:
                step_val = ast.literal_eval(args[2])
            except Exception:
                raise CompileError("range() step must be a constant")
            if not isinstance(step_val, int):
                raise CompileError("range() step must be an integer constant")
        else:
            raise CompileError("range() takes 1-3 arguments")

        loop_label = self._new_label("for")
        continue_label = self._new_label("for_cont")
        end_label = self._new_label("endfor")

        self.loop_stack.append((continue_label, end_label))

        self._mark_label(loop_label)

        # Condition: if var >= stop, exit (for positive step)
        if step_val > 0:
            self._emit_branch(Opcode.BGE, var_reg, stop_tmp, end_label)
        else:
            # Negative step: if var <= stop, exit → if stop >= var, exit
            self._emit_branch(Opcode.BGE, stop_tmp, var_reg, end_label)

        # Body
        for stmt in node.body:
            self._compile_stmt(stmt)

        # Increment (continue target)
        self._mark_label(continue_label)
        step_tmp = self.regs.alloc_temp()
        self._load_const(step_tmp, step_val & 0xFFFFFFFF)   # neg/large steps must not truncate
        self._emit(Instruction(Opcode.ADD, a=var_reg, b=var_reg, c=step_tmp))
        self.regs.free_temp(step_tmp)

        self._emit_jump(loop_label)
        self._mark_label(end_label)

        self.regs.free_temp(stop_tmp)
        self.loop_stack.pop()

    def _compile_break(self):
        if not self.loop_stack:
            raise CompileError("break outside loop")
        _, break_label = self.loop_stack[-1]
        self._emit_jump(break_label)

    def _compile_continue(self):
        if not self.loop_stack:
            raise CompileError("continue outside loop")
        continue_label, _ = self.loop_stack[-1]
        self._emit_jump(continue_label)

    # ── Expression compilation ──

    def _load_const(self, dest: int, val: int):
        """Load a 32-bit constant into dest. MOVI's immediate is only 21 bits in the
        packed state encoding (the matmul backends decode 21 bits, so a single MOVI with
        a wider value silently truncates there). Materialize larger constants as
        (high << 21) | low so the result is correct on EVERY backend."""
        val &= 0xFFFFFFFF
        if val < (1 << 21):
            self._emit(movi(dest, val))
        else:
            low = val & 0x1FFFFF           # low 21 bits
            high = (val >> 21) & 0x7FF      # high 11 bits
            tmp = self.regs.alloc_temp()
            self._emit(movi(dest, high))
            self._emit(movi(tmp, 21))
            self._emit(Instruction(Opcode.SHL, a=dest, b=dest, c=tmp))
            self._emit(movi(tmp, low))
            self._emit(Instruction(Opcode.OR, a=dest, b=dest, c=tmp))
            self.regs.free_temp(tmp)

    def _compile_expr_into(self, node: ast.expr, dest: int):
        """Compile an expression, placing the result in register dest."""

        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                val = int(val)
            if not isinstance(val, int):
                raise CompileError(f"Only integer constants supported, got {type(val).__name__}")
            self._load_const(dest, val & 0xFFFFFFFF)

        elif isinstance(node, ast.Name):
            src = self.regs.get_var(node.id)
            if src != dest:
                self._emit(Instruction(Opcode.MOV, a=dest, b=src))

        elif isinstance(node, ast.BinOp):
            self._compile_binop(node, dest)

        elif isinstance(node, ast.UnaryOp):
            self._compile_unaryop(node, dest)

        elif isinstance(node, ast.Compare):
            self._compile_compare_expr(node, dest)

        elif isinstance(node, ast.BoolOp):
            self._compile_boolop_expr(node, dest)

        elif isinstance(node, ast.Call):
            self._compile_call_expr(node, dest)

        elif isinstance(node, ast.IfExp):
            # x = a if cond else b
            else_label = self._new_label("ifexp_else")
            end_label = self._new_label("ifexp_end")
            self._compile_condition_false(node.test, else_label)
            self._compile_expr_into(node.body, dest)
            self._emit_jump(end_label)
            self._mark_label(else_label)
            self._compile_expr_into(node.orelse, dest)
            self._mark_label(end_label)

        else:
            raise CompileError(f"Unsupported expression: {type(node).__name__}")

    # Unsigned-comparison builtins. Ordinary Python `<`/`<=`/... compile to
    # SIGNED comparisons (SLT/BLT), which are wrong for magnitudes >= 2^31 (a
    # value with the high bit set reads as negative). These builtins emit the
    # unsigned NISA opcodes (SLTU/BLTU/BGEU) so 32-bit magnitude comparisons are
    # correct across the full [0, 2^32) range.
    UNSIGNED_CMP = ("ult", "ule", "ugt", "uge")
    # `umulh(a, b)` = high 32 bits of the unsigned 64-bit product (NISA MULHU).
    # Together with `*` (low 32 bits) this lets the subset build exact 64-bit
    # arithmetic (e.g. num*num for num up to 2^32) without silent overflow.
    BUILTINS = UNSIGNED_CMP + ("umulh",)

    def _lookup_user_func(self, name: str):
        """Return the AST FunctionDef for a same-module user function, or None."""
        if name in self._ast_cache:
            return self._ast_cache[name]
        fn = self._module_globals.get(name)
        if fn is None or not callable(fn):
            return None
        try:
            src = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            return None
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.FunctionDef) and n.name == name:
                self._ast_cache[name] = n
                return n
        return None

    def _inline_call(self, fn_def: ast.FunctionDef, arg_nodes: list, dest: int):
        """Inline a call to a user-defined function: bind parameters to fresh registers,
        compile the body with `return` redirected to `dest`, then restore scope."""
        if len(self._inline_ctx) >= self._max_inline_depth:
            raise CompileError(f"Inline depth exceeded (recursive call to '{fn_def.name}'?)")
        params = [a.arg for a in fn_def.args.args]
        if len(arg_nodes) != len(params):
            raise CompileError(f"{fn_def.name}() expects {len(params)} args, got {len(arg_nodes)}")

        saved_vars = dict(self.regs.var_to_reg)
        saved_next = self.regs.next_reg
        # Evaluate args in the CALLER's scope into fresh variable registers.
        param_regs = []
        for arg_node in arg_nodes:
            reg = self.regs.next_reg
            if reg > self.regs.max_var_reg:
                raise CompileError("Too many local variables (inlining ran out of registers)")
            self.regs.next_reg += 1
            self._compile_expr_into(arg_node, reg)
            param_regs.append(reg)
        # Switch to the callee's scope: params bound; locals allocate past the params.
        self.regs.var_to_reg = dict(zip(params, param_regs))
        end_label = self._new_label("inline_end")
        self._inline_ctx.append((dest, end_label))
        for stmt in fn_def.body:
            self._compile_stmt(stmt)
        self._inline_ctx.pop()
        self._mark_label(end_label)
        # Restore the caller's scope (freeing the inlined function's registers).
        self.regs.var_to_reg = saved_vars
        self.regs.next_reg = saved_next

    def _compile_call_expr(self, node: ast.Call, dest: int):
        """Compile a call: builtins (unsigned compares, umulh) or an inlined user function."""
        if not isinstance(node.func, ast.Name):
            raise CompileError("Only calls to named functions/builtins are supported")
        name = node.func.id
        if name not in self.BUILTINS:
            fn_def = self._lookup_user_func(name)
            if fn_def is not None:
                self._inline_call(fn_def, node.args, dest)
                return
            raise CompileError(
                f"Unsupported function call: {name}() "
                f"(supported builtins: {', '.join(self.BUILTINS)}; or a same-module def)")
        if len(node.args) != 2:
            raise CompileError(f"{name}() takes exactly 2 arguments")

        left_reg = dest
        self._compile_expr_into(node.args[0], left_reg)
        right_tmp = self.regs.alloc_temp()
        self._compile_expr_into(node.args[1], right_tmp)

        if name == "umulh":        # high 32 bits of a*b (unsigned)
            self._emit(Instruction(Opcode.MULHU, a=dest, b=left_reg, c=right_tmp))
        elif name == "ult":        # a <u b
            self._emit(Instruction(Opcode.SLTU, a=dest, b=left_reg, c=right_tmp))
        elif name == "ugt":        # a >u b  ==  b <u a
            self._emit(Instruction(Opcode.SLTU, a=dest, b=right_tmp, c=left_reg))
        else:                      # ule / uge: negate the strict form
            if name == "ule":      # a <=u b  ==  not (b <u a)
                self._emit(Instruction(Opcode.SLTU, a=dest, b=right_tmp, c=left_reg))
            else:                  # uge: a >=u b  ==  not (a <u b)
                self._emit(Instruction(Opcode.SLTU, a=dest, b=left_reg, c=right_tmp))
            tmp2 = self.regs.alloc_temp()
            self._emit(movi(tmp2, 1))
            self._emit(Instruction(Opcode.XOR, a=dest, b=dest, c=tmp2))
            self.regs.free_temp(tmp2)

        self.regs.free_temp(right_tmp)

    def _compile_binop(self, node: ast.BinOp, dest: int):
        op_map = {
            ast.Add: Opcode.ADD, ast.Sub: Opcode.SUB, ast.Mult: Opcode.MUL,
            ast.BitAnd: Opcode.AND, ast.BitOr: Opcode.OR, ast.BitXor: Opcode.XOR,
            ast.LShift: Opcode.SHL, ast.RShift: Opcode.SHR,
        }
        op_type = type(node.op)
        if op_type not in op_map:
            raise CompileError(f"Unsupported binary operator: {op_type.__name__}")

        nisa_op = op_map[op_type]

        # Compile left into dest, right into temp
        self._compile_expr_into(node.left, dest)
        tmp = self.regs.alloc_temp()
        self._compile_expr_into(node.right, tmp)
        self._emit(Instruction(nisa_op, a=dest, b=dest, c=tmp))
        self.regs.free_temp(tmp)

    def _compile_unaryop(self, node: ast.UnaryOp, dest: int):
        if isinstance(node.op, ast.Invert):
            # ~x → NOT
            self._compile_expr_into(node.operand, dest)
            self._emit(Instruction(Opcode.NOT, a=dest, b=dest))
        elif isinstance(node.op, ast.USub):
            # -x → SUB 0, x
            self._compile_expr_into(node.operand, dest)
            self._emit(Instruction(Opcode.SUB, a=dest, b=0, c=dest))
        elif isinstance(node.op, ast.Not):
            # not x → x == 0 → SLTU dest, 0, x; XOR dest, dest, 1?
            # Simpler: result = (x == 0) ? 1 : 0
            self._compile_expr_into(node.operand, dest)
            # seqz: sltiu dest, dest, 1 — but we don't have sltiu in NISA
            # Use: MOVI tmp, 1; SLTU dest, dest, tmp
            tmp = self.regs.alloc_temp()
            self._emit(movi(tmp, 1))
            self._emit(Instruction(Opcode.SLTU, a=dest, b=dest, c=tmp))
            self.regs.free_temp(tmp)
        else:
            raise CompileError(f"Unsupported unary operator: {type(node.op).__name__}")

    def _compile_compare_expr(self, node: ast.Compare, dest: int):
        """Compile a comparison expression to 0/1 result."""
        if len(node.ops) != 1:
            raise CompileError("Chained comparisons not supported")

        op = node.ops[0]
        left_reg = dest
        self._compile_expr_into(node.left, left_reg)
        right_tmp = self.regs.alloc_temp()
        self._compile_expr_into(node.comparators[0], right_tmp)

        if isinstance(op, ast.Lt):
            self._emit(Instruction(Opcode.SLT, a=dest, b=left_reg, c=right_tmp))
        elif isinstance(op, ast.Gt):
            self._emit(Instruction(Opcode.SLT, a=dest, b=right_tmp, c=left_reg))
        elif isinstance(op, ast.LtE):
            # a <= b → NOT (a > b) → NOT (b < a)
            self._emit(Instruction(Opcode.SLT, a=dest, b=right_tmp, c=left_reg))
            tmp2 = self.regs.alloc_temp()
            self._emit(movi(tmp2, 1))
            self._emit(Instruction(Opcode.XOR, a=dest, b=dest, c=tmp2))
            self.regs.free_temp(tmp2)
        elif isinstance(op, ast.GtE):
            self._emit(Instruction(Opcode.SLT, a=dest, b=left_reg, c=right_tmp))
            tmp2 = self.regs.alloc_temp()
            self._emit(movi(tmp2, 1))
            self._emit(Instruction(Opcode.XOR, a=dest, b=dest, c=tmp2))
            self.regs.free_temp(tmp2)
        elif isinstance(op, (ast.Eq, ast.NotEq)):
            # a == b → (a XOR b) == 0 → SLTU(a^b, 1)
            self._emit(Instruction(Opcode.XOR, a=dest, b=left_reg, c=right_tmp))
            tmp2 = self.regs.alloc_temp()
            self._emit(movi(tmp2, 1))
            self._emit(Instruction(Opcode.SLTU, a=dest, b=dest, c=tmp2))
            if isinstance(op, ast.NotEq):
                # Flip result
                self._emit(Instruction(Opcode.XOR, a=dest, b=dest, c=tmp2))
            self.regs.free_temp(tmp2)
        else:
            raise CompileError(f"Unsupported comparison: {type(op).__name__}")

        self.regs.free_temp(right_tmp)

    def _compile_boolop_expr(self, node: ast.BoolOp, dest: int):
        """Compile `and`/`or` to 0/1 result."""
        if isinstance(node.op, ast.And):
            false_label = self._new_label("and_false")
            end_label = self._new_label("and_end")
            for value in node.values[:-1]:
                self._compile_expr_into(value, dest)
                self._emit_branch(Opcode.BEQ, dest, 0, false_label)
            self._compile_expr_into(node.values[-1], dest)
            self._emit_jump(end_label)
            self._mark_label(false_label)
            self._emit(movi(dest, 0))
            self._mark_label(end_label)
        elif isinstance(node.op, ast.Or):
            true_label = self._new_label("or_true")
            end_label = self._new_label("or_end")
            for value in node.values[:-1]:
                self._compile_expr_into(value, dest)
                self._emit_branch(Opcode.BNE, dest, 0, true_label)
            self._compile_expr_into(node.values[-1], dest)
            self._emit_jump(end_label)
            self._mark_label(true_label)
            self._emit(movi(dest, 1))
            self._mark_label(end_label)

    # ── Condition compilation (for branches) ──

    def _compile_condition_false(self, node: ast.expr, false_label: str):
        """Compile a condition, branching to false_label if it's false."""

        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            # Optimize direct comparison → single branch instruction
            op = node.ops[0]
            left_tmp = self.regs.alloc_temp()
            right_tmp = self.regs.alloc_temp()
            self._compile_expr_into(node.left, left_tmp)
            self._compile_expr_into(node.comparators[0], right_tmp)

            # Branch if condition is FALSE
            if isinstance(op, ast.Eq):
                self._emit_branch(Opcode.BNE, left_tmp, right_tmp, false_label)
            elif isinstance(op, ast.NotEq):
                self._emit_branch(Opcode.BEQ, left_tmp, right_tmp, false_label)
            elif isinstance(op, ast.Lt):
                self._emit_branch(Opcode.BGE, left_tmp, right_tmp, false_label)
            elif isinstance(op, ast.GtE):
                self._emit_branch(Opcode.BLT, left_tmp, right_tmp, false_label)
            elif isinstance(op, ast.Gt):
                # a > b → branch if NOT (b < a) → branch if a <= b → branch if b >= a
                self._emit_branch(Opcode.BGE, right_tmp, left_tmp, false_label)
            elif isinstance(op, ast.LtE):
                # a <= b → branch if NOT (a <= b) → branch if a > b → branch if b < a
                self._emit_branch(Opcode.BLT, right_tmp, left_tmp, false_label)
            else:
                raise CompileError(f"Unsupported comparison in condition: {type(op).__name__}")

            self.regs.free_temp(right_tmp)
            self.regs.free_temp(left_tmp)

        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                # All must be true; any false → branch to false_label
                for value in node.values:
                    self._compile_condition_false(value, false_label)
            elif isinstance(node.op, ast.Or):
                # (A or B or ...): false only if ALL false. If any early value is TRUE,
                # short-circuit to the body (the fall-through point past the false-branch).
                body_label = self._new_label("or_body")
                for value in node.values[:-1]:
                    self._compile_condition_true(value, body_label)
                self._compile_condition_false(node.values[-1], false_label)
                self._mark_label(body_label)

        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            # not x → branch if x is TRUE
            self._compile_condition_true(node.operand, false_label)

        else:
            # General expression: evaluate and branch if zero
            tmp = self.regs.alloc_temp()
            self._compile_expr_into(node, tmp)
            self._emit_branch(Opcode.BEQ, tmp, 0, false_label)
            self.regs.free_temp(tmp)

    def _compile_condition_true(self, node: ast.expr, true_label: str):
        """Compile a condition, branching to true_label if it's true."""
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = node.ops[0]
            left_tmp = self.regs.alloc_temp()
            right_tmp = self.regs.alloc_temp()
            self._compile_expr_into(node.left, left_tmp)
            self._compile_expr_into(node.comparators[0], right_tmp)

            if isinstance(op, ast.Eq):
                self._emit_branch(Opcode.BEQ, left_tmp, right_tmp, true_label)
            elif isinstance(op, ast.NotEq):
                self._emit_branch(Opcode.BNE, left_tmp, right_tmp, true_label)
            elif isinstance(op, ast.Lt):
                self._emit_branch(Opcode.BLT, left_tmp, right_tmp, true_label)
            elif isinstance(op, ast.GtE):
                self._emit_branch(Opcode.BGE, left_tmp, right_tmp, true_label)
            elif isinstance(op, ast.Gt):
                self._emit_branch(Opcode.BLT, right_tmp, left_tmp, true_label)
            elif isinstance(op, ast.LtE):
                self._emit_branch(Opcode.BGE, right_tmp, left_tmp, true_label)

            self.regs.free_temp(right_tmp)
            self.regs.free_temp(left_tmp)
        else:
            tmp = self.regs.alloc_temp()
            self._compile_expr_into(node, tmp)
            self._emit_branch(Opcode.BNE, tmp, 0, true_label)
            self.regs.free_temp(tmp)


def compile_python(func: Callable) -> list[Instruction]:
    """Compile a Python function to NISA instructions."""
    compiler = PythonToNISA()
    return compiler.compile_function(func)


def transformer_jit(func: Callable = None, *, device: str = "cuda",
                    max_cycles: int = 100000):
    """Decorator that JIT-compiles a Python function to NISA and runs on GPU.

    Usage:
        @transformer_jit
        def fib(n: int) -> int:
            a, b = 0, 1
            for i in range(n):
                a, b = b, a + b
            return a

        result = fib(10)  # Returns 55
    """
    def decorator(fn: Callable):
        nisa_code = None  # lazy compilation

        @functools.wraps(fn)
        def wrapper(*args):
            nonlocal nisa_code
            if nisa_code is None:
                nisa_code = compile_python(fn)

            # Set up argument registers
            # Function args map to registers 1, 2, 3, ... in the compiler
            initial_regs = {}
            compiler = PythonToNISA()
            for i, arg in enumerate(inspect.signature(fn).parameters):
                reg = i + 1  # registers start at 1
                if i < len(args):
                    initial_regs[reg] = int(args[i]) & 0xFFFFFFFF

            result = gpu_execute(
                nisa_code,
                initial_registers=initial_regs,
                device=device,
                max_cycles=max_cycles,
            )

            # Return value is in r10 (a0)
            val = result.reg(10)
            # Convert to signed if MSB is set
            if val >= 0x80000000:
                val -= 0x100000000
            return val

        wrapper._nisa_code = lambda: nisa_code or compile_python(fn)
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
