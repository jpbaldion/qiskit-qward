"""
Variational Quantum Time Evolution (VQTE) experiment runner para QWARD

Genera variaciones automáticas de VQTE sobre Hamiltonianos sencillos (por ejemplo, modelo de Ising o Heisenberg),
ejecuta las métricas LOC, Halstead, Behavioral y Quantum Software Quality, y guarda los resultados en un CSV.
"""

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from scipy.linalg import expm
from ..metrics import BehavioralMetrics, ElementMetrics, StructuralMetrics, QuantumSpecificMetrics
import os


def create_vqte_circuit_variational(
    num_qubits,
    p,
    evolution_type="imaginary",
    ansatz_type="ising",
    delta_t=0.1,
    params_per_layer=None,
    with_measurements=False
):
    """
    Crea un circuito inspirado en VarQTE/VarQITE.

    params_per_layer: número de parámetros por capa. Si None -> use num_qubits (un parámetro por qubit por capa).
    with_measurements: si True añade medida al final.
    """
    if params_per_layer is None:
        params_per_layer = num_qubits

    total_params = p * params_per_layer
    params = ParameterVector('theta', total_params)
    qc = QuantumCircuit(num_qubits)

    # Inicialización en |+>^n
    qc.h(range(num_qubits))

    # Uso: param index base por capa
    for step in range(p):
        base = step * params_per_layer

        # si params_per_layer == 1: se usa el mismo parámetro para toda la capa
        for q in range(num_qubits):
            param_idx = base + (q if params_per_layer > 1 else 0)
            theta_q = params[param_idx]

            if ansatz_type == "ising":
                # aplico interacción ZZ entre vecinos con rotación dependiente de theta_q
                if q < num_qubits - 1:
                    qc.cx(q, q + 1)
                    qc.rz(2 * theta_q * delta_t, q + 1)
                    qc.cx(q, q + 1)

            elif ansatz_type == "heisenberg":
                if q < num_qubits - 1:
                    # XX
                    qc.cx(q, q + 1)
                    qc.rx(2 * theta_q * delta_t, q + 1)
                    qc.cx(q, q + 1)
                    # YY
                    qc.sdg(q)
                    qc.cx(q, q + 1)
                    qc.ry(2 * theta_q * delta_t, q + 1)
                    qc.cx(q, q + 1)
                    qc.s(q)
                    # ZZ
                    qc.cx(q, q + 1)
                    qc.rz(2 * theta_q * delta_t, q + 1)
                    qc.cx(q, q + 1)

            elif ansatz_type == "su2":
                # Rotaciones por qubit y entangling cz en la capa
                qc.ry(theta_q * delta_t, q)

        # entangling layer para su2
        if ansatz_type == "su2":
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)

        # single-qubit evolution según tipo
        for q in range(num_qubits):
            param_idx = base + (q if params_per_layer > 1 else 0)
            theta_q = params[param_idx]
            if evolution_type == "real":
                qc.rx(2 * theta_q * delta_t, q)
            elif evolution_type == "imaginary":
                qc.ry(2 * theta_q * delta_t, q)
            else:
                raise ValueError(f"Unknown evolution_type: {evolution_type}")

    if with_measurements:
        qc.measure_all()

    return qc

def ising_ideal_state(num_qubits, J=1.0, h=1.0, t=1.0):
    # Construir Hamiltoniano Ising 1D: H = J Σ Z_i Z_{i+1} + h Σ X_i
    paulis = []
    for i in range(num_qubits - 1):
        label = ['I'] * num_qubits
        label[i] = 'Z'
        label[i+1] = 'Z'
        paulis.append(("".join(label), J))

    for i in range(num_qubits):
        label = ['I'] * num_qubits
        label[i] = 'X'
        paulis.append(("".join(label), h))

    H = SparsePauliOp.from_list(paulis).to_matrix()
    
    # Estado inicial |+>^n
    init = Statevector.from_label('+' * num_qubits).data

    # Evolución exacta: exp(-iHt)|ψ₀⟩
    U = expm(-1j * H * t)
    evolved = U @ init

    return evolved / np.linalg.norm(evolved)

def run_experiments(
    num_instances=3,
    num_qubits_list=[5, 6, 7],
    p_list=[1, 2, 3],
    seed=123,
    output_csv="vqte_metrics_results.csv"
):
    results = []
    np.random.seed(seed)
    for n in num_qubits_list:
        for p in p_list:
            for idx in range(num_instances):

                for evo_type in ["imaginary", "real"]:
                    for ansatz in ["ising", "heisenberg", "su2"]:
                        qc = create_vqte_circuit_variational(n, p, evolution_type=evo_type, ansatz_type=ansatz)

                row = {
                    "num_qubits": n,
                    "p": p,
                    "instance_id": idx
                }
                # Element
                element_metrics = ElementMetrics(qc).get_metrics()
                row.update({f"element_{k}": v for k, v in element_metrics.model_dump().items()})
                # Structural
                structural_metrics = StructuralMetrics(qc).get_metrics()
                row.update({f"structural_{k}": v for k, v in structural_metrics.model_dump().items()})
                # Behavioral
                behavioral_metrics = BehavioralMetrics(qc).get_metrics()
                row.update({f"behavioral_{k}": v for k, v in behavioral_metrics.model_dump().items()})
                # Quantum Specific
                quantum_specific_metrics = QuantumSpecificMetrics(qc).get_metrics()
                row.update({f"quantum_specific_{k}": v for k, v in quantum_specific_metrics.model_dump().items()})
                results.append(row)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Resultados guardados en {os.path.abspath(output_csv)}")


if __name__ == "__main__":
    run_experiments()
