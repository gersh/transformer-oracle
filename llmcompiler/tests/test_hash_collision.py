"""
Gradient-guided hash collision search.

Uses differentiable execution to compute ∂hash(x)/∂x, then
gradient descent to find x2 ≠ x1 where hash(x1) == hash(x2).
"""

import torch
import pytest
from ..compiler.python_compiler import compile_python
from ..runtime.differentiable_executor import execute_differentiable
from ..runtime.gpu_executor import gpu_execute


# ── FNV-1a inspired hash (32-bit, runs on transformer) ──

def fnv_hash(x):
    """Hash a 32-bit integer. FNV-1a inspired."""
    h = 2166136261  # FNV offset basis
    b0 = x & 255
    h = h ^ b0
    h = h * 16777619

    b1 = (x >> 8) & 255
    h = h ^ b1
    h = h * 16777619

    b2 = (x >> 16) & 255
    h = h ^ b2
    h = h * 16777619

    b3 = (x >> 24) & 255
    h = h ^ b3
    h = h * 16777619

    return h


# ── Simpler hash for collision finding ──

def weak_hash(x):
    """Intentionally weak hash — easier to find collisions."""
    h = x * 2654435761  # Knuth multiplicative
    h = h ^ (h >> 16)
    h = h & 0xFFFF       # truncate to 16 bits → collision guaranteed
    return h


class TestHashFunction:
    def test_fnv_correctness(self):
        nisa = compile_python(fnv_hash)
        # Different inputs should give different hashes
        hashes = set()
        for v in [0, 1, 42, 100, 255, 1337]:
            r = gpu_execute(nisa, initial_registers={1: v}, device='cuda')
            hashes.add(r.reg(10))
        assert len(hashes) >= 5  # most should be unique

    def test_weak_hash(self):
        nisa = compile_python(weak_hash)
        r = gpu_execute(nisa, initial_registers={1: 42}, device='cuda')
        assert r.reg(10) < 0x10000  # 16-bit output


class TestGradientCollisionSearch:

    def test_gradient_exists(self):
        """Verify gradients flow through the hash computation."""
        nisa = compile_python(weak_hash)
        x = torch.tensor(42.0, dtype=torch.float64, requires_grad=True)
        result = execute_differentiable(nisa, {1: x})
        h = result.registers[10]
        h.backward()
        assert x.grad is not None
        assert x.grad.item() != 0, "Gradient should be non-zero"
        print(f"\nhash(42) = {h.item():.0f}, ∂hash/∂x = {x.grad.item():.4f}")

    def test_gradient_guides_toward_target(self):
        """Gradients point toward a target hash value."""
        nisa = compile_python(weak_hash)

        # Target: hash(42)
        target_r = gpu_execute(nisa, initial_registers={1: 42}, device='cuda')
        target_hash = float(target_r.reg(10))
        print(f"\nTarget: hash(42) = {target_hash:.0f}")

        # Start from x=1000, optimize toward target
        x = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=10.0)

        distances = []
        for i in range(200):
            optimizer.zero_grad()
            result = execute_differentiable(nisa, {1: x})
            h = result.registers[10]
            loss = (h - target_hash) ** 2
            loss.backward()
            optimizer.step()
            distances.append(abs(h.item() - target_hash))

        # Should have gotten closer
        assert distances[-1] < distances[0], \
            f"Should move toward target: start={distances[0]:.0f}, end={distances[-1]:.0f}"
        print(f"  Start distance: {distances[0]:.0f}")
        print(f"  End distance:   {distances[-1]:.0f}")
        print(f"  Final x: {int(x.item())}")

    def test_find_collision_weak_hash(self):
        """Find a collision in the weak (16-bit) hash."""
        nisa = compile_python(weak_hash)

        target_input = 42
        target_r = gpu_execute(nisa, initial_registers={1: target_input}, device='cuda')
        target_hash = float(target_r.reg(10))
        print(f"\nTarget: hash({target_input}) = {target_hash:.0f}")

        # Search from multiple starting points
        best_x = None
        best_dist = float('inf')

        for start in [1000, 5000, 20000, 50000, 100000]:
            x = torch.tensor(float(start), dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.Adam([x], lr=20.0)

            for i in range(300):
                optimizer.zero_grad()
                result = execute_differentiable(nisa, {1: x})
                h = result.registers[10]
                loss = (h - target_hash) ** 2
                loss.backward()
                optimizer.step()

                xi = int(x.detach().item()) & 0xFFFFFFFF
                dist = abs(h.item() - target_hash)
                if dist < best_dist and xi != target_input:
                    best_dist = dist
                    best_x = xi

            xi = int(x.detach().item()) & 0xFFFFFFFF
            # Verify on actual executor
            r = gpu_execute(nisa, initial_registers={1: xi}, device='cuda')
            actual_dist = abs(r.reg(10) - target_r.reg(10))
            print(f"  start={start:6d} → x={xi:8d}, hash={r.reg(10):5d}, "
                  f"target={target_r.reg(10):5d}, dist={actual_dist}")
            if actual_dist == 0 and xi != target_input:
                print(f"  *** COLLISION: hash({target_input}) == hash({xi}) == {target_r.reg(10)} ***")
                best_x = xi
                best_dist = 0
                break

        if best_dist == 0:
            print(f"\nCOLLISION FOUND!")
        else:
            print(f"\nClosest: x={best_x}, distance={best_dist:.0f}")
            print(f"(16-bit hash has 65536 values — collision exists by pigeonhole)")

    def test_brute_force_vs_gradient(self):
        """Compare: gradient search vs brute force for collision finding."""
        nisa = compile_python(weak_hash)

        target = 42
        target_r = gpu_execute(nisa, initial_registers={1: target}, device='cuda')
        target_hash = target_r.reg(10)

        # Brute force
        import time
        t0 = time.time()
        brute_collision = None
        for v in range(100000):
            if v == target:
                continue
            r = gpu_execute(nisa, initial_registers={1: v}, device='cuda')
            if r.reg(10) == target_hash:
                brute_collision = v
                break
        t_brute = time.time() - t0

        # Gradient search
        t0 = time.time()
        grad_collision = None
        for start in range(0, 100000, 1000):
            x = torch.tensor(float(start), dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.Adam([x], lr=10.0)
            for _ in range(100):
                optimizer.zero_grad()
                result = execute_differentiable(nisa, {1: x})
                loss = (result.registers[10] - float(target_hash)) ** 2
                loss.backward()
                optimizer.step()

            xi = int(x.detach().item()) & 0xFFFFFFFF
            r = gpu_execute(nisa, initial_registers={1: xi}, device='cuda')
            if r.reg(10) == target_hash and xi != target:
                grad_collision = xi
                break
        t_grad = time.time() - t0

        print(f"\nBrute force: {'found '+str(brute_collision) if brute_collision else 'not found'} in {t_brute:.2f}s")
        print(f"Gradient:    {'found '+str(grad_collision) if grad_collision else 'not found'} in {t_grad:.2f}s")
        if brute_collision and grad_collision:
            print(f"Speedup: {t_brute/t_grad:.1f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
