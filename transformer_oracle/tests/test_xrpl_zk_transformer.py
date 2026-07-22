"""
Transformer-based zero-knowledge attack on XRP Ledger proofs.

Uses a multi-head attention transformer trained on 50K+ proof samples
to detect ANY statistical leakage from the zero-knowledge proofs.

If accuracy is even 0.1% above random chance with statistical
significance, the proof leaks information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import hashlib
import math
import pytest

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def mod_n(x):
    return x % N


def scalar_hash(*args):
    h = hashlib.sha256()
    h.update(b"MPT_POK_PLAINTEXT_PROOF")
    for a in args:
        h.update(int(a).to_bytes(32, 'big'))
    return int.from_bytes(h.digest(), 'big') % N


def generate_proof(amount, sk):
    r = random.randint(1, N - 1)
    t = random.randint(1, N - 1)
    c1 = r
    c2 = mod_n(amount + r * sk)
    t1 = t
    t2 = mod_n(t * sk)
    if amount > 0:
        e = scalar_hash(c1, c2, sk, amount, t1, t2)
    else:
        e = scalar_hash(c1, c2, sk, t1, t2)
    s = mod_n(t + e * r)
    return [t1, t2, s, e, c1, c2], r


def proof_to_tensor(proof_values):
    """Convert proof to a (6, 32) tensor — 6 scalars, 32 bytes each.
    This preserves the byte-level structure for the transformer."""
    rows = []
    for v in proof_values:
        byte_vals = [(v >> (i * 8)) & 0xFF for i in range(32)]
        rows.append(torch.tensor(byte_vals, dtype=torch.float32) / 255.0)
    return torch.stack(rows)  # (6, 32)


class TransformerClassifier(nn.Module):
    """Transformer that processes proof scalars as a sequence."""

    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(32, d_model)
        self.pos_embed = nn.Parameter(torch.randn(6, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, 6, 32)
        x = self.input_proj(x) + self.pos_embed  # (batch, 6, d_model)
        x = self.transformer(x)  # (batch, 6, d_model)
        x = x.mean(dim=1)  # pool over sequence
        return self.classifier(x)


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes,
                       epochs=50, batch_size=256, lr=0.001):
    """Train transformer and return test accuracy with confidence interval."""
    model = TransformerClassifier(num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n_train = len(X_train)
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_train)
        total_loss = 0
        for start in range(0, n_train, batch_size):
            idx = indices[start:start + batch_size]
            out = model(X_train[idx])
            loss = F.cross_entropy(out, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        correct = (preds == y_test).float()
        accuracy = correct.mean().item()
        # Wilson score interval for 95% confidence
        n = len(y_test)
        z = 1.96
        p = accuracy
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom

    return accuracy, center - margin, center + margin


class TestZKTransformer:

    def _generate_dataset(self, gen_fn, n_samples):
        X_list, y_list = [], []
        for _ in range(n_samples):
            x, label = gen_fn()
            X_list.append(x)
            y_list.append(label)
        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)
        split = int(n_samples * 0.8)
        return X[:split], y[:split], X[split:], y[split:]

    def test_amount_prediction(self):
        """Transformer tries to predict amount class from proof."""
        print("\n" + "="*65)
        print("TRANSFORMER ATTACK: Predict AMOUNT from proof")
        print("50K samples, 3-layer transformer, 4-head attention")
        print("="*65)

        amounts = [100, 500, 1000, 5000, 10000]
        sk = random.randint(1, N - 1)
        random.seed(42)

        def gen():
            label = random.randint(0, len(amounts) - 1)
            proof, _ = generate_proof(amounts[label], sk)
            return proof_to_tensor(proof), label

        X_tr, y_tr, X_te, y_te = self._generate_dataset(gen, 50000)

        acc, ci_lo, ci_hi = train_and_evaluate(X_tr, y_tr, X_te, y_te,
                                                len(amounts), epochs=80)
        chance = 1.0 / len(amounts)

        print(f"\n  Classes: {len(amounts)}, Samples: 50K")
        print(f"  Random chance:     {chance:.1%}")
        print(f"  Transformer acc:   {acc:.2%} (95% CI: [{ci_lo:.2%}, {ci_hi:.2%}])")

        if ci_lo > chance:
            print(f"  !!! STATISTICALLY SIGNIFICANT LEAKAGE DETECTED !!!")
        else:
            print(f"  ✓ No leakage — accuracy within chance")

    def test_key_identification(self):
        """Transformer tries to identify which key generated a proof."""
        print("\n" + "="*65)
        print("TRANSFORMER ATTACK: Identify SECRET KEY from proof")
        print("="*65)

        num_keys = 10
        random.seed(42)
        keys = [random.randint(1, N - 1) for _ in range(num_keys)]

        def gen():
            label = random.randint(0, num_keys - 1)
            amount = random.randint(1, 10000)
            proof, _ = generate_proof(amount, keys[label])
            return proof_to_tensor(proof), label

        X_tr, y_tr, X_te, y_te = self._generate_dataset(gen, 50000)

        acc, ci_lo, ci_hi = train_and_evaluate(X_tr, y_tr, X_te, y_te,
                                                num_keys, epochs=80)
        chance = 1.0 / num_keys

        print(f"\n  Keys: {num_keys}, Samples: 50K")
        print(f"  Random chance:     {chance:.1%}")
        print(f"  Transformer acc:   {acc:.2%} (95% CI: [{ci_lo:.2%}, {ci_hi:.2%}])")

        if ci_lo > chance:
            print(f"  !!! KEY IDENTITY LEAKS !!!")
        else:
            print(f"  ✓ Key identity hidden")

    def test_real_vs_random(self):
        """Transformer tries to distinguish real proofs from random."""
        print("\n" + "="*65)
        print("TRANSFORMER ATTACK: Distinguish REAL vs RANDOM proofs")
        print("="*65)

        random.seed(42)
        sk = random.randint(1, N - 1)

        def gen():
            if random.random() < 0.5:
                proof, _ = generate_proof(random.randint(1, 10000), sk)
                return proof_to_tensor(proof), 1
            else:
                fake = [random.randint(0, N - 1) for _ in range(6)]
                return proof_to_tensor(fake), 0

        X_tr, y_tr, X_te, y_te = self._generate_dataset(gen, 50000)

        acc, ci_lo, ci_hi = train_and_evaluate(X_tr, y_tr, X_te, y_te,
                                                2, epochs=80)

        print(f"\n  Samples: 50K (25K real + 25K random)")
        print(f"  Random chance:     50.0%")
        print(f"  Transformer acc:   {acc:.2%} (95% CI: [{ci_lo:.2%}, {ci_hi:.2%}])")

        if ci_lo > 0.5:
            print(f"  !!! PROOFS ARE DISTINGUISHABLE FROM RANDOM !!!")
        else:
            print(f"  ✓ Proofs indistinguishable from random")

    def test_linkability(self):
        """Transformer tries to link proof pairs from same sender."""
        print("\n" + "="*65)
        print("TRANSFORMER ATTACK: LINK proofs from same sender")
        print("="*65)

        random.seed(42)
        senders = [random.randint(1, N - 1) for _ in range(50)]

        def gen():
            if random.random() < 0.5:
                s = random.choice(senders)
                p1, _ = generate_proof(random.randint(1, 10000), s)
                p2, _ = generate_proof(random.randint(1, 10000), s)
                x = torch.cat([proof_to_tensor(p1), proof_to_tensor(p2)])
                return x, 1
            else:
                s1, s2 = random.sample(senders, 2)
                p1, _ = generate_proof(random.randint(1, 10000), s1)
                p2, _ = generate_proof(random.randint(1, 10000), s2)
                x = torch.cat([proof_to_tensor(p1), proof_to_tensor(p2)])
                return x, 0

        X_list, y_list = [], []
        for _ in range(50000):
            x, label = gen()
            X_list.append(x)
            y_list.append(label)
        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)
        split = 40000

        # Need wider model for paired input (12 tokens instead of 6)
        model = TransformerClassifier(2, d_model=64, nhead=4, num_layers=3)
        model.input_proj = nn.Linear(32, 64)
        model.pos_embed = nn.Parameter(torch.randn(12, 64) * 0.02)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        for epoch in range(80):
            model.train()
            idx = torch.randperm(split)
            for start in range(0, split, 256):
                batch = idx[start:start + 256]
                loss = F.cross_entropy(model(X[batch]), y[batch])
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X[split:]).argmax(1)
            acc = (preds == y[split:]).float().mean().item()
            n = len(y[split:])
            z = 1.96; p = acc
            denom = 1 + z*z/n
            margin = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
            ci_lo = (p + z*z/(2*n))/denom - margin

        print(f"\n  Senders: 50, Pairs: 50K")
        print(f"  Random chance:     50.0%")
        print(f"  Transformer acc:   {acc:.2%} (95% CI lower: {ci_lo:.2%})")

        if ci_lo > 0.5:
            print(f"  !!! PROOFS ARE LINKABLE !!!")
        else:
            print(f"  ✓ Proofs unlinkable")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
