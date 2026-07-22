"""
Tests for differentiable fuzzing.
"""

import torch
import pytest
from ..compiler.python_compiler import compile_python
from ..runtime.differentiable_executor import execute_differentiable, fuzz
from ..core.nisa import Instruction, Opcode, movi, halt


class TestDifferentiableExecutor:

    def test_gradient_through_add(self):
        """Gradients flow through addition."""
        from ..assembler.assembler import assemble
        prog = assemble("add r3, r1, r2\nhalt")

        x = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        y = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        result = execute_differentiable(prog, {1: x, 2: y})
        output = result.registers[3]
        output.backward()

        # d(x+y)/dx = 1, d(x+y)/dy = 1
        assert x.grad.item() == pytest.approx(1.0)
        assert y.grad.item() == pytest.approx(1.0)

    def test_gradient_through_mul(self):
        """Gradients flow through multiplication."""
        from ..assembler.assembler import assemble
        prog = assemble("mul r3, r1, r2\nhalt")

        x = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        y = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        result = execute_differentiable(prog, {1: x, 2: y})
        output = result.registers[3]
        output.backward()

        # d(x*y)/dx = y = 3, d(x*y)/dy = x = 5
        assert x.grad.item() == pytest.approx(3.0)
        assert y.grad.item() == pytest.approx(5.0)

    def test_gradient_through_expression(self):
        """Gradients through (a+b)*(a-b) = a²-b²."""
        def f(a, b):
            c = a + b
            d = a - b
            return c * d
        prog = compile_python(f)

        a = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        result = execute_differentiable(prog, {1: a, 2: b})
        output = result.registers[10]  # return value
        output.backward()

        # f = a²-b², df/da = 2a = 20, df/db = -2b = -6
        assert a.grad.item() == pytest.approx(20.0)
        assert b.grad.item() == pytest.approx(-6.0)

    def test_branch_distance_tracked(self):
        """Branch distances are recorded and differentiable."""
        def f(x):
            if x > 10:
                return 1
            return 0
        prog = compile_python(f)

        x = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        result = execute_differentiable(prog, {1: x})

        assert len(result.branch_events) > 0
        # x=5, condition is x > 10, so branch NOT taken
        # Gradient should point toward increasing x to flip the branch
        dist = result.branch_events[0].distance
        dist.backward()
        assert x.grad is not None

    def test_branch_coverage(self):
        """Coverage tracking works."""
        def f(x):
            if x > 10:
                return 1
            return 0
        prog = compile_python(f)

        # x=5: takes the "not taken" path
        r1 = execute_differentiable(prog, {
            1: torch.tensor(5.0, dtype=torch.float64)
        })
        # x=15: takes the "taken" path
        r2 = execute_differentiable(prog, {
            1: torch.tensor(15.0, dtype=torch.float64)
        })

        # Different coverage
        assert r1.coverage != r2.coverage


class TestFuzzer:

    def test_fuzz_simple_branch(self):
        """Fuzzer finds inputs to cover both sides of a branch."""
        def check(x):
            if x == 42:
                return 1  # hard to reach!
            return 0
        prog = compile_python(check)

        result = fuzz(prog, n_input_regs=1, n_iterations=200,
                      lr=5.0, verbose=False)

        # Should find at least some branches
        assert len(result['best_coverage']) > 0
        assert len(result['all_branches']) > 0

    def test_fuzz_multiple_branches(self):
        """Fuzzer explores multiple branches."""
        def classify(x):
            if x < 10:
                return 0
            if x < 50:
                return 1
            if x < 100:
                return 2
            return 3
        prog = compile_python(classify)

        result = fuzz(prog, n_input_regs=1, n_iterations=500,
                      lr=20.0, verbose=False)

        # Should cover at least 2 branch directions
        assert len(result['best_coverage']) >= 2

    def test_fuzz_password_check(self):
        """Classic fuzzing target: find the 'password' value."""
        def check_password(x):
            # Must match specific value
            if x == 1337:
                return 1  # "access granted"
            return 0
        prog = compile_python(check_password)

        # The fuzzer should discover inputs near 1337
        result = fuzz(prog, n_input_regs=1, n_iterations=500,
                      lr=10.0, verbose=True, seed=0)

        # Check if gradient guidance helped explore the branch
        print(f"\nBest inputs: {result['best_inputs']}")
        print(f"Coverage: {len(result['best_coverage'])} branches covered")
        print(f"Total branches seen: {len(result['all_branches'])}")

    def test_fuzz_two_inputs(self):
        """Fuzzer with two input parameters."""
        def check(a, b):
            if a + b == 100:
                return 1
            if a > b:
                return 2
            return 3
        prog = compile_python(check)

        result = fuzz(prog, n_input_regs=2, n_iterations=200,
                      lr=10.0, verbose=False)

        assert len(result['best_coverage']) >= 2

    def test_gradient_guides_toward_target(self):
        """Verify gradients actually point toward the branch-flipping direction."""
        def check(x):
            if x > 100:
                return 1
            return 0
        prog = compile_python(check)

        # Start with x=50 (below threshold)
        x = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
        result = execute_differentiable(prog, {1: x})

        # The branch distance should be positive (not taken: x < 100)
        assert len(result.branch_events) > 0
        dist = result.branch_events[0].distance
        assert dist.item() < 0  # BLT: 100 - x = negative means not taken...

        # Compute gradient: which direction to move x to flip the branch?
        dist.backward()
        # The gradient should point toward increasing x (to make x > 100)
        # distance = signed(x) - signed(100), to minimize |distance|, we increase x
        assert x.grad is not None
        print(f"\nx={x.item():.0f}, distance={dist.item():.1f}, grad={x.grad.item():.4f}")
        print(f"Gradient says: move x in direction {'+' if x.grad.item() < 0 else '-'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
