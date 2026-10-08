# ============================================================
# QUANTUM KERNEL FOR FEDERATED UPDATE SIMILARITY
# ============================================================

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def create_fixed_projection(input_dim=256, output_dim=4, seed=123):
    """Create one deterministic 256 -> 4 random projection."""
    rng = np.random.default_rng(seed)
    return rng.normal(
        0.0,
        1.0 / np.sqrt(output_dim),
        size=(input_dim, output_dim),
    )


def project_update_profiles(profiles, projection):
    """Project 256-D profiles to a small quantum input space."""
    X = np.asarray(profiles, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D profiles, got {X.shape}")
    if X.shape[1] != projection.shape[0]:
        raise ValueError(
            f"Profile dimension {X.shape[1]} does not match "
            f"projection dimension {projection.shape[0]}"
        )

    projected = X @ projection

    # Standardize across the currently participating clients.
    mean = projected.mean(axis=0, keepdims=True)
    std = projected.std(axis=0, keepdims=True)
    projected = (projected - mean) / (std + 1e-8)

    # Keep feature-map angles bounded.
    return np.tanh(projected) * np.pi


class QuantumKernel:
    """Statevector-simulated fidelity quantum kernel."""

    def __init__(self, quantum_dim=4, reps=2, seed=123):
        self.quantum_dim = quantum_dim
        self.reps = reps
        self.seed = seed

    def feature_map(self, x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) != self.quantum_dim:
            raise ValueError(
                f"Expected {self.quantum_dim} dimensions, got {len(x)}"
            )

        qc = QuantumCircuit(self.quantum_dim)

        for _ in range(self.reps):
            for q in range(self.quantum_dim):
                qc.ry(float(x[q]), q)

            for q in range(self.quantum_dim - 1):
                qc.cx(q, q + 1)

        return qc

    def statevector(self, x):
        return Statevector.from_instruction(self.feature_map(x))

    def evaluate(self, x, y):
        """K(x,y) = |<phi(x)|phi(y)>|^2."""
        sx = self.statevector(x)
        sy = self.statevector(y)
        overlap = np.vdot(sx.data, sy.data)
        return float(np.clip(np.abs(overlap) ** 2, 0.0, 1.0))

    def matrix(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.quantum_dim:
            raise ValueError(
                f"Expected X with shape (N,{self.quantum_dim}), got {X.shape}"
            )

        states = [self.statevector(x) for x in X]
        n = len(states)
        K = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i, n):
                overlap = np.vdot(states[i].data, states[j].data)
                value = float(np.abs(overlap) ** 2)
                K[i, j] = value
                K[j, i] = value

        K = np.clip(K, 0.0, 1.0)
        np.fill_diagonal(K, 1.0)
        return K


def quantum_diversity_matrix(kernel_matrix):
    """D_ij = 1 - K_ij."""
    K = np.asarray(kernel_matrix, dtype=np.float64)
    D = np.clip(1.0 - K, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D


def client_diversity_scores(diversity_matrix):
    """Mean pairwise quantum diversity for each client."""
    D = np.asarray(diversity_matrix, dtype=np.float64)
    n = D.shape[0]
    scores = np.zeros(n, dtype=np.float64)

    if n <= 1:
        return scores

    for i in range(n):
        others = [j for j in range(n) if j != i]
        scores[i] = float(np.mean(D[i, others]))

    return scores