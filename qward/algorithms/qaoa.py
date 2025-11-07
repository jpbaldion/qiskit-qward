"""
QAOA MaxCut experiment runner for QWARD

Genera variaciones automáticas de QAOA para MaxCut sobre grafos aleatorios y diferentes valores de p,
ejecuta las métricas LOC, Halstead, Behavioral y Quantum Software Quality, y guarda los resultados en un CSV.
"""


import numpy as np
import networkx as nx
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, state_fidelity
from ..metrics import StructuralMetrics, BehavioralMetrics, ElementMetrics, QuantumSpecificMetrics
import os


def create_qaoa_maxcut_circuit(graph, p, params=None):
    n = graph.number_of_nodes()
    qc = QuantumCircuit(n)

    if params is None:
        params = ParameterVector('θ', 2 * p)

    # Superposición inicial
    qc.h(range(n))

    for i in range(p):
        gamma = params[i]
        beta = params[p + i]

        # Hamiltoniano de costo (MaxCut)
        for u, v in graph.edges():
            qc.cx(u, v)
            qc.rz(-gamma, v)
            qc.cx(u, v)

        # Hamiltoniano mezclador
        for q in range(n):
            qc.rx(2 * beta, q)

    qc.measure_all()
    return qc

def maxcut_cost(bitstring, graph):
    cost = 0
    for u, v in graph.edges():
        if bitstring[u] != bitstring[v]:
            cost += 1
    return cost

def get_optimal_bitstrings(graph):
    n = graph.number_of_nodes()
    best_cost = -1
    best_states = []
    for i in range(2**n):
        bitstr = format(i, f"0{n}b")
        cost = maxcut_cost(bitstr, graph)
        if cost > best_cost:
            best_cost = cost
            best_states = [bitstr]
        elif cost == best_cost:
            best_states.append(bitstr)
    return best_states, best_cost

def target_state_from_bitstrings(states):
    n = len(states[0])
    vec = np.zeros(2**n, dtype=complex)
    for s in states:
        vec[int(s,2)] = 1
    vec /= np.linalg.norm(vec)
    return Statevector(vec)

def get_target_state(graph):
    optimal_bitstrings, _ = get_optimal_bitstrings(graph)
    return target_state_from_bitstrings(optimal_bitstrings)

def get_qaoa_statevector(qc):
    qc = qc.remove_final_measurements(inplace=False)
    return Statevector.from_instruction(qc)

def probability_of_optimal(sv, optimal_bitstrings):
    probs = sv.probabilities_dict()
    return sum(probs.get(bs, 0) for bs in optimal_bitstrings)

def expected_cost_from_sv(sv, graph):
    probs = sv.probabilities_dict()
    return sum(p * maxcut_cost(bitstr, graph) for bitstr, p in probs.items())


def run_experiments(
    num_graphs=5,
    num_nodes_list=[1, 2, 3],
    p_list=[1, 2, 3],
    seed=42,
    output_csv="qaoa_metrics_results.csv"
):
    results = []
    np.random.seed(seed)
    for n in num_nodes_list:
        for p in p_list:
            for gidx in range(num_graphs):
                # Grafo aleatorio
                graph = nx.erdos_renyi_graph(n, 0.5, seed=seed + gidx)
                qc = create_qaoa_maxcut_circuit(graph, p)
                # Ejecutar métricas
                row = {
                    "num_nodes": n,
                    "p": p,
                    "graph_id": gidx,
                    "edges": list(graph.edges())
                }
                # Element
                element_metrics = ElementMetrics(qc).get_metrics()
                row.update({f"element_{k}": v for k, v in element_metrics.dict().items()})
                # Structural
                loc_metrics = StructuralMetrics(qc).get_metrics()
                row.update({f"structural_{k}": v for k, v in loc_metrics.dict().items()})
                # Behavioral
                behavioral_metrics = BehavioralMetrics(qc).get_metrics()
                row.update({f"behavioral_{k}": v for k, v in behavioral_metrics.dict().items()})
                # Quantum Specific
                quantum_specific_metrics = QuantumSpecificMetrics(qc).get_metrics()
                row.update({f"quantum_specific_{k}": v for k, v in quantum_specific_metrics.dict().items()})
                # Guardar resultados
                results.append(row)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Resultados guardados en {os.path.abspath(output_csv)}")


if __name__ == "__main__":
    run_experiments(3, [5, 10, 15], [5, 5, 5], seed=123, output_csv="qaoa_metrics_test_results.csv")
