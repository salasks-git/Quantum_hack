"""Quantum circuit builders: data-encoding feature maps and the trainable VQC ansatz."""
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes, ZZFeatureMap

from qmed.config import FEATURE_MAP_ZZ


def build_feature_map(name: str, n_qubits: int, reps: int, entanglement: str):
    """Circuit that encodes one patient's angles into a quantum state."""
    if name == FEATURE_MAP_ZZ:
        return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
    return PauliFeatureMap(
        feature_dimension=n_qubits, reps=reps, paulis=["Z", "YY", "ZX"], entanglement=entanglement
    )


def build_ansatz(n_qubits: int, reps: int, entanglement: str):
    """Trainable Ry + CNOT circuit whose weights the VQC optimizer tunes."""
    return RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
