"""Quantum circuit builders: data-encoding feature maps and the trainable VQC ansatz."""
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes, ZZFeatureMap

from qmed.config import FEATURE_MAP_ZZ


def build_feature_map(name: str, n_qubits: int, reps: int, entanglement: str):
    """Circuit that encodes one patient's angles into a quantum state."""
    if name == FEATURE_MAP_ZZ:
        return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
    return PauliFeatureMap(
        feature_dimension=n_qubits, reps=reps, paulis=["Z", "YY", "ZX"], entanglement=entanglement
    )


def scale_feature_map(feature_map, bandwidth: float):
    """Same circuit, but every input angle x_i is replaced by bandwidth * x_i.

    Baking the scale into the circuit keeps it consistent everywhere the kernel is
    evaluated (training, test inference, live simulator) without callers having to
    remember to rescale their inputs.
    """
    if bandwidth == 1.0:
        return feature_map
    x = ParameterVector("x", feature_map.num_parameters)
    return feature_map.assign_parameters(
        {p: bandwidth * xi for p, xi in zip(feature_map.parameters, x)}
    )


def build_ansatz(n_qubits: int, reps: int, entanglement: str):
    """Trainable Ry + CNOT circuit whose weights the VQC optimizer tunes."""
    return RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
