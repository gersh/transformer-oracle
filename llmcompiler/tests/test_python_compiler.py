"""
Tests for the Python-subset-to-NISA compiler.
"""

import pytest
from ..compiler.python_compiler import compile_python, transformer_jit
from ..runtime.gpu_executor import gpu_execute


class TestBasicExpressions:

    def test_return_constant(self):
        def f():
            return 42
        nisa = compile_python(f)
        result = gpu_execute(nisa, device='cuda')
        assert result.reg(10) == 42

    def test_return_arg(self):
        def f(x):
            return x
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 99}, device='cuda')
        assert result.reg(10) == 99

    def test_add(self):
        def f(a, b):
            return a + b
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 5, 2: 3}, device='cuda')
        assert result.reg(10) == 8

    def test_sub(self):
        def f(a, b):
            return a - b
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 10, 2: 3}, device='cuda')
        assert result.reg(10) == 7

    def test_mul(self):
        def f(a, b):
            return a * b
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 7, 2: 6}, device='cuda')
        assert result.reg(10) == 42

    def test_bitwise(self):
        def f(a, b):
            return (a & b) | (a ^ b)
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 0xFF, 2: 0x0F}, device='cuda')
        assert result.reg(10) == (0xFF & 0x0F) | (0xFF ^ 0x0F)

    def test_negate(self):
        def f(x):
            return -x
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 5}, device='cuda')
        # -5 in unsigned 32-bit = 0xFFFFFFFB
        assert result.reg(10) == 0xFFFFFFFB

    def test_invert(self):
        def f(x):
            return ~x
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 0xFF}, device='cuda')
        assert result.reg(10) == 0xFFFFFF00

    def test_complex_expr(self):
        def f(a, b):
            return (a + b) * (a - b)
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 10, 2: 3}, device='cuda')
        assert result.reg(10) == (10 + 3) * (10 - 3)


class TestVariables:

    def test_local_var(self):
        def f(x):
            y = x + 1
            return y
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 41}, device='cuda')
        assert result.reg(10) == 42

    def test_multiple_vars(self):
        def f():
            a = 10
            b = 20
            c = a + b
            return c
        nisa = compile_python(f)
        result = gpu_execute(nisa, device='cuda')
        assert result.reg(10) == 30

    def test_augmented_assign(self):
        def f(x):
            x += 10
            x *= 2
            return x
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 5}, device='cuda')
        assert result.reg(10) == 30

    def test_tuple_assign(self):
        def f():
            a, b = 3, 7
            a, b = b, a
            return a
        nisa = compile_python(f)
        result = gpu_execute(nisa, device='cuda')
        assert result.reg(10) == 7

    def test_swap(self):
        def f(a, b):
            a, b = b, a
            return a
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 10, 2: 20}, device='cuda')
        assert result.reg(10) == 20


class TestControlFlow:

    def test_if_true(self):
        def f(x):
            if x > 5:
                return 1
            return 0
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 10}, device='cuda')
        assert result.reg(10) == 1

    def test_if_false(self):
        def f(x):
            if x > 5:
                return 1
            return 0
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 3}, device='cuda')
        assert result.reg(10) == 0

    def test_if_else(self):
        def f(x):
            if x == 0:
                r = 100
            else:
                r = 200
            return r
        nisa = compile_python(f)
        r1 = gpu_execute(nisa, initial_registers={1: 0}, device='cuda')
        r2 = gpu_execute(nisa, initial_registers={1: 1}, device='cuda')
        assert r1.reg(10) == 100
        assert r2.reg(10) == 200

    def test_while_loop(self):
        def f(n):
            total = 0
            i = 1
            while i <= n:
                total += i
                i += 1
            return total
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 10}, device='cuda')
        assert result.reg(10) == 55  # sum 1..10

    def test_for_range(self):
        def f(n):
            total = 0
            for i in range(n):
                total += i
            return total
        nisa = compile_python(f)
        result = gpu_execute(nisa, initial_registers={1: 10}, device='cuda')
        assert result.reg(10) == 45  # sum 0..9

    def test_for_range_start_stop(self):
        def f():
            total = 0
            for i in range(5, 10):
                total += i
            return total
        nisa = compile_python(f)
        result = gpu_execute(nisa, device='cuda')
        assert result.reg(10) == 35  # 5+6+7+8+9

    def test_break(self):
        def f():
            total = 0
            for i in range(100):
                if i >= 5:
                    break
                total += i
            return total
        nisa = compile_python(f)
        result = gpu_execute(nisa, device='cuda')
        assert result.reg(10) == 10  # 0+1+2+3+4

    def test_continue(self):
        def f():
            total = 0
            for i in range(10):
                if i == 5:
                    continue
                total += i
            return total
        nisa = compile_python(f)
        result = gpu_execute(nisa, device='cuda')
        assert result.reg(10) == 40  # 0+1+2+3+4+6+7+8+9 = 45-5

    def test_nested_loops(self):
        def f():
            total = 0
            for i in range(5):
                for j in range(5):
                    total += 1
            return total
        nisa = compile_python(f)
        result = gpu_execute(nisa, device='cuda')
        assert result.reg(10) == 25


class TestAlgorithms:

    def test_fibonacci(self):
        def fib(n):
            a, b = 0, 1
            for i in range(n):
                a, b = b, a + b
            return a
        nisa = compile_python(fib)
        result = gpu_execute(nisa, initial_registers={1: 10}, device='cuda')
        assert result.reg(10) == 55

    def test_gcd(self):
        def gcd(a, b):
            while b != 0:
                r = a
                while r >= b:
                    r -= b
                a = b
                b = r
            return a
        nisa = compile_python(gcd)
        result = gpu_execute(nisa, initial_registers={1: 48, 2: 18}, device='cuda')
        assert result.reg(10) == 6

    def test_factorial(self):
        def factorial(n):
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        nisa = compile_python(factorial)
        # fact(10) = 3628800
        result = gpu_execute(nisa, initial_registers={1: 10}, device='cuda')
        assert result.reg(10) == 3628800

    def test_is_prime(self):
        def is_prime(n):
            if n < 2:
                return 0
            i = 2
            while i * i <= n:
                # n % i via repeated subtraction
                r = n
                while r >= i:
                    r -= i
                if r == 0:
                    return 0
                i += 1
            return 1
        nisa = compile_python(is_prime)
        # 17 is prime
        r1 = gpu_execute(nisa, initial_registers={1: 17}, device='cuda')
        assert r1.reg(10) == 1
        # 15 is not prime
        r2 = gpu_execute(nisa, initial_registers={1: 15}, device='cuda')
        assert r2.reg(10) == 0

    def test_power(self):
        def power(base, exp):
            result = 1
            for i in range(exp):
                result *= base
            return result
        nisa = compile_python(power)
        result = gpu_execute(nisa, initial_registers={1: 2, 2: 10}, device='cuda')
        assert result.reg(10) == 1024


class TestTransformerJIT:
    """Test the @transformer_jit decorator."""

    def test_jit_basic(self):
        @transformer_jit
        def add_one(x: int) -> int:
            return x + 1

        assert add_one(41) == 42

    def test_jit_fibonacci(self):
        @transformer_jit
        def fib(n: int) -> int:
            a, b = 0, 1
            for i in range(n):
                a, b = b, a + b
            return a

        assert fib(10) == 55
        assert fib(20) == 6765

    def test_jit_multiple_args(self):
        @transformer_jit
        def add(a: int, b: int) -> int:
            return a + b

        assert add(5, 3) == 8
        assert add(100, 200) == 300

    def test_jit_gcd(self):
        @transformer_jit
        def gcd(a: int, b: int) -> int:
            while b != 0:
                r = a
                while r >= b:
                    r -= b
                a = b
                b = r
            return a

        assert gcd(48, 18) == 6
        assert gcd(100, 75) == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
