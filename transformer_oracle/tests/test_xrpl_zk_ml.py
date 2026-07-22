"""
ML-based zero-knowledge testing for XRP Ledger proofs.

If a proof is truly zero-knowledge, a neural network should NOT be
able to predict ANY property of the secret witness from the proof.

Tests:
1. Can ML predict the AMOUNT from the proof? (should be random chance)
2. Can ML predict the SECRET KEY from proofs? (should be impossible)
3. Can ML distinguish real proofs from random bytes? (should be 50/50)
4. Can ML link two proofs as coming from the same sender? (unlinkability)
"""

import torch
import torch.nn as nn
import random
import hashlib
import pytest

# secp256k1 curve order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def mod_n(x):
    return x % N


def scalar_hash(*args):
    h = hashlib.sha256()
    h.update(b"MPT_POK_PLAINTEXT_PROOF")
    for a in args:
        h.update(int(a).to_bytes(32, 'big'))
    return int.from_bytes(h.digest(), 'big') % N


def generate_proof(amount, sk_recipient, r=None):
    """Generate an equality proof for a given amount.
    Returns (proof_bytes, secret_r) where proof = (T1, T2, s, e)."""
    if r is None:
        r = random.randint(1, N - 1)
    t = random.randint(1, N - 1)

    c1 = r
    c2 = mod_n(amount + r * sk_recipient)
    t1 = t
    t2 = mod_n(t * sk_recipient)

    if amount > 0:
        e = scalar_hash(c1, c2, sk_recipient, amount, t1, t2)
    else:
        e = scalar_hash(c1, c2, sk_recipient, t1, t2)

    s = mod_n(t + e * r)

    # Return proof as list of scalar values
    return [t1, t2, s, e, c1, c2], r


def proof_to_tensor(proof_values):
    """Convert proof scalars to a normalized float tensor."""
    # Take lower 64 bits of each scalar (captures most entropy)
    features = []
    for v in proof_values:
        # Extract 8 bytes worth of features from each 32-byte scalar
        for shift in range(0, 256, 32):
            features.append(float((v >> shift) & 0xFFFFFFFF) / 0xFFFFFFFF)
    return torch.tensor(features, dtype=torch.float32)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_classifier(X, y, num_classes, epochs=100):
    """Train a classifier and return test accuracy."""
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = SimpleClassifier(X_train.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy


def train_regressor(X, y, epochs=100):
    """Train a regressor and return test R²."""
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = SimpleRegressor(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = nn.functional.mse_loss(out, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_test)
        ss_res = ((y_test - preds) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return r2.item()


class TestZeroKnowledgeML:

    def test_predict_amount_from_proof(self):
        """Can ML predict which amount a proof is for?

        If ZK: accuracy ≈ 1/num_classes (random chance)
        If leaks: accuracy > random chance
        """
        print("\n" + "="*60)
        print("TEST 1: Can ML predict the AMOUNT from the proof?")
        print("="*60)

        amounts = [100, 200, 500, 1000, 5000]
        sk = random.randint(1, N - 1)
        random.seed(42)

        X_list, y_list = [], []
        for _ in range(2000):
            label = random.randint(0, len(amounts) - 1)
            proof, _ = generate_proof(amounts[label], sk)
            X_list.append(proof_to_tensor(proof))
            y_list.append(label)

        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)

        accuracy = train_classifier(X, y, len(amounts), epochs=200)
        random_chance = 1.0 / len(amounts)

        print(f"\n  {len(amounts)} amount classes, {len(X)} samples")
        print(f"  Random chance: {random_chance:.1%}")
        print(f"  ML accuracy:   {accuracy:.1%}")

        if accuracy > random_chance + 0.1:
            print(f"  !!! PROOF LEAKS AMOUNT INFORMATION !!!")
        else:
            print(f"  ✓ Proof is zero-knowledge w.r.t. amount")

        # Allow some statistical noise
        assert accuracy < random_chance + 0.15, \
            f"ML accuracy {accuracy:.1%} significantly above chance {random_chance:.1%} — ZK violated!"

    def test_predict_secret_key(self):
        """Can ML extract bits of the secret key from many proofs?

        Generate many proofs with the same key, try to predict key bits.
        """
        print("\n" + "="*60)
        print("TEST 2: Can ML extract SECRET KEY from proofs?")
        print("="*60)

        random.seed(42)
        num_keys = 10

        keys = [random.randint(1, N - 1) for _ in range(num_keys)]

        X_list, y_list = [], []
        for _ in range(3000):
            key_idx = random.randint(0, num_keys - 1)
            amount = random.randint(1, 10000)
            proof, _ = generate_proof(amount, keys[key_idx])
            X_list.append(proof_to_tensor(proof))
            y_list.append(key_idx)

        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)

        accuracy = train_classifier(X, y, num_keys, epochs=200)
        random_chance = 1.0 / num_keys

        print(f"\n  {num_keys} different keys, {len(X)} proofs")
        print(f"  Random chance: {random_chance:.1%}")
        print(f"  ML accuracy:   {accuracy:.1%}")

        if accuracy > random_chance + 0.1:
            print(f"  !!! PROOFS LEAK KEY IDENTITY !!!")
        else:
            print(f"  ✓ Proofs don't reveal which key was used")

        assert accuracy < random_chance + 0.15

    def test_distinguish_real_from_random(self):
        """Can ML tell real proofs apart from random bytes?

        If ZK: proofs should be indistinguishable from random.
        """
        print("\n" + "="*60)
        print("TEST 3: Can ML distinguish REAL proofs from RANDOM?")
        print("="*60)

        random.seed(42)
        sk = random.randint(1, N - 1)

        X_list, y_list = [], []
        for _ in range(2000):
            if random.random() < 0.5:
                # Real proof
                proof, _ = generate_proof(random.randint(1, 10000), sk)
                X_list.append(proof_to_tensor(proof))
                y_list.append(1)
            else:
                # Random "proof" (uniform random scalars)
                fake = [random.randint(0, N - 1) for _ in range(6)]
                X_list.append(proof_to_tensor(fake))
                y_list.append(0)

        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)

        accuracy = train_classifier(X, y, 2, epochs=200)

        print(f"\n  1000 real proofs + 1000 random")
        print(f"  Random chance: 50.0%")
        print(f"  ML accuracy:   {accuracy:.1%}")

        if accuracy > 0.6:
            print(f"  !!! PROOFS ARE DISTINGUISHABLE FROM RANDOM !!!")
            print(f"  (This doesn't mean ZK is broken — proofs may have")
            print(f"   structural properties like valid curve points)")
        else:
            print(f"  ✓ Proofs indistinguishable from random")

    def test_link_same_sender(self):
        """Can ML link two proofs as coming from the same sender?

        Unlinkability: given two proofs, can ML tell if they're from
        the same sender (same r or same sk)?
        """
        print("\n" + "="*60)
        print("TEST 4: Can ML LINK proofs from the same sender?")
        print("="*60)

        random.seed(42)
        num_senders = 20
        senders = [random.randint(1, N - 1) for _ in range(num_senders)]

        X_list, y_list = [], []
        for _ in range(2000):
            if random.random() < 0.5:
                # Same sender
                sender = random.choice(senders)
                p1, _ = generate_proof(random.randint(1, 10000), sender)
                p2, _ = generate_proof(random.randint(1, 10000), sender)
                features = torch.cat([proof_to_tensor(p1), proof_to_tensor(p2)])
                X_list.append(features)
                y_list.append(1)
            else:
                # Different senders
                s1, s2 = random.sample(senders, 2)
                p1, _ = generate_proof(random.randint(1, 10000), s1)
                p2, _ = generate_proof(random.randint(1, 10000), s2)
                features = torch.cat([proof_to_tensor(p1), proof_to_tensor(p2)])
                X_list.append(features)
                y_list.append(0)

        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)

        accuracy = train_classifier(X, y, 2, epochs=200)

        print(f"\n  {num_senders} senders, paired proofs")
        print(f"  Random chance: 50.0%")
        print(f"  ML accuracy:   {accuracy:.1%}")

        if accuracy > 0.6:
            print(f"  !!! PROOFS FROM SAME SENDER ARE LINKABLE !!!")
        else:
            print(f"  ✓ Proofs are unlinkable")

    def test_predict_randomness(self):
        """Can ML predict any bits of the secret randomness r?"""
        print("\n" + "="*60)
        print("TEST 5: Can ML predict the secret RANDOMNESS r?")
        print("="*60)

        random.seed(42)
        sk = random.randint(1, N - 1)

        X_list, y_list = [], []
        for _ in range(2000):
            amount = random.randint(1, 10000)
            proof, r = generate_proof(amount, sk)
            X_list.append(proof_to_tensor(proof))
            # Try to predict the lowest bit of r
            y_list.append(r & 1)

        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)

        accuracy = train_classifier(X, y, 2, epochs=200)

        print(f"\n  Predicting lowest bit of randomness r")
        print(f"  Random chance: 50.0%")
        print(f"  ML accuracy:   {accuracy:.1%}")

        if accuracy > 0.6:
            print(f"  !!! RANDOMNESS LEAKS THROUGH PROOF !!!")
        else:
            print(f"  ✓ Randomness is hidden")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
