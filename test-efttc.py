import numpy as np
import json

input = {
    "with_db": False,
    "solver": {
        "type": "NeptuneMinDelayAndUtilization",
        "args": {"alpha": 1, "verbose": True, "soften_step1_sol": 1.3}
    },
    "workload_coeff": 1,
    "community": "community-test",
    "namespace": "namespace-test",
    "node_names": [
        "node_a", "node_b", "node_c"
    ],
    "node_delay_matrix": [[0, 3, 2],
                          [3, 0, 4],
                          [2, 4, 0]],
    "workload_on_source_matrix": [[100, 0, 0], [1, 0, 0]],
    "node_memories": [
        100, 100, 200
    ],
    "node_cores": [
        100, 50, 50
    ],
    "gpu_node_names": [
    ],
    "gpu_node_memories": [
    ],
    "function_names": [
        "ns/fn_1", "ns/fn_2"
    ],
    "function_memories": [
        5, 5
    ],
    "function_max_delays": [
        1000, 1000
    ],
    "gpu_function_names": [
    ],
    "gpu_function_memories": [
    ],
    "actual_cpu_allocations": {
        "ns/fn_1": {
            "node_a": True,
            "node_b": True,
            "node_c": True,
        },
        "ns/fn_2": {
            "node_a": True,
            "node_b": True,
            "node_c": True,
        }
    },
    "actual_gpu_allocations": {
    },
}

input["cores_matrix"] = [[1,1,1]] * len(input["function_names"])
input["workload_on_destination_matrix"] = [[1,1,1]] * len(input["function_names"])

import numpy as np

import numpy as np

def ef_ttc(input_data):
    function_names = input_data["function_names"]
    node_names = input_data["node_names"]
    node_memories = input_data["node_memories"]
    node_cores = input_data["node_cores"]

    gpu_function_names = input_data["gpu_function_names"]
    gpu_node_names = input_data["gpu_node_names"]
    gpu_node_memories = input_data["gpu_node_memories"]

    function_memories = input_data["function_memories"]
    function_max_delays = input_data["function_max_delays"]
    cores_matrix = np.array(input_data["cores_matrix"])  # funzione × nodo
    node_delay_matrix = np.array(input_data["node_delay_matrix"])
    workload_matrix = np.array(input_data["workload_on_source_matrix"])

    # Stato risorse
    memory_used = np.zeros(len(node_names))
    cores_used = np.zeros(len(node_names))
    active_functions = {fn: set() for fn in function_names}

    # ALLOCAZIONI
    cpu_allocations = {}
    cpu_routing_rules = {node: {fn: {} for fn in function_names} for node in node_names}

    for f_idx, fn in enumerate(function_names):
        best_nodes = np.argsort(node_delay_matrix[f_idx])  # Ordina per latenza
        mem_req = function_memories[f_idx]
        max_delay = function_max_delays[f_idx]

        for n_idx in best_nodes:
            node = node_names[n_idx]
            delay_ok = node_delay_matrix[f_idx][n_idx] <= max_delay
            mem_ok = memory_used[n_idx] + mem_req <= node_memories[n_idx]
            core_ok = cores_used[n_idx] + cores_matrix[f_idx][n_idx] <= node_cores[n_idx]

            if delay_ok and mem_ok and core_ok:
                # Assegna funzione a nodo
                cpu_allocations[fn] = {node: True}
                cpu_routing_rules[node][fn] = {node: 1.0}

                # Aggiorna stato
                memory_used[n_idx] += mem_req
                cores_used[n_idx] += cores_matrix[f_idx][n_idx]
                active_functions[fn].add(node)
                break

    # GESTIONE GPU
    gpu_allocations = {}
    gpu_routing_rules = {node: {fn: {} for fn in gpu_function_names} for node in gpu_node_names}
    if gpu_function_names:
        memory_used_gpu = np.zeros(len(gpu_node_names))
        cores_gpu = [50] * len(gpu_node_names)  # esempio, puoi passarlo da input

        for f_idx, fn in enumerate(gpu_function_names):
            best_nodes = range(len(gpu_node_names))  # qui potresti ordinare per latenza
            mem_req = input_data["gpu_function_memories"][f_idx]

            for n_idx in best_nodes:
                node = gpu_node_names[n_idx]
                if memory_used_gpu[n_idx] + mem_req <= gpu_node_memories[n_idx]:
                    gpu_allocations[fn] = {node: True}
                    gpu_routing_rules[node][fn] = {node: 1.0}
                    memory_used_gpu[n_idx] += mem_req
                    break

    return {
        "cpu_allocations": cpu_allocations,
        "cpu_routing_rules": cpu_routing_rules,
        "gpu_allocations": gpu_allocations,
        "gpu_routing_rules": gpu_routing_rules,
    }


# Chiamata dell'algoritmo EF-TTC con l'input
actual_allocations = ef_ttc(input)
# print in output-efttc.json
output_file = 'output-efttc.json'
with open(output_file, 'w') as f:
    json.dump(actual_allocations, f, indent=4)

print(actual_allocations)
