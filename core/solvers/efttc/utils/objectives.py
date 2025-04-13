import numpy as np


'''def score_minimize_network_delay(data, x):
    total_delay = 0.0
    node_delay = data.node_delay_matrix
    workload = data.workload_matrix
    num_functions = len(data.functions)
    num_nodes = len(data.nodes)

    for f in range(num_functions):
        for i in range(num_nodes):
            w = workload[f, i]
            if w == 0:
                continue  # inutile iterare se non c'è carico
            for j in range(num_nodes):
                val = x[(i, f, j)]["val"]
                if val == 0:
                    continue  # inutile sommare se la variabile è zero
                total_delay += val * node_delay[i, j] * w
    return total_delay'''

def score_minimize_network_delay(data, x):
    x_array = np.zeros((len(data.nodes), len(data.functions), len(data.nodes)))
    for (i, f, j), val_dict in x.items():
        x_array[i, f, j] = val_dict["val"]

    delay = data.node_delay_matrix  # shape (i, j)
    workload = data.workload_matrix  # shape (f, i)

    # reshape workload to (i, f, j) compatibile per broadcasting
    workload_reshaped = np.transpose(workload, (1, 0))[:, :, np.newaxis]  # shape (i, f, 1)
    delay_reshaped = delay[:, np.newaxis, :]  # shape (i, 1, j)

    total = np.sum(x_array * delay_reshaped * workload_reshaped)
    return total

def score_maximize_handled_requests(data, x):
    total_handled = 0.0
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            for j in range(len(data.nodes)):
                val = x[(i, f, j)]["val"]
                workload = data.workload_matrix[f, i]
                total_handled += val * workload
    return -total_handled  # Perché usiamo minimizzazione

def score_minimize_node_utilization(data, n):
    return sum(1 for i in range(len(data.nodes)) if n[i]["val"])

import numpy as np

def score_minimize_node_delay_and_utilization(data, n, x, alpha):
    num_nodes = len(data.nodes)
    num_funcs = len(data.functions)

    # 1. Penalità nodi attivi
    node_util_score = sum(1 for i in range(num_nodes) if n[i]["val"])
    node_util_score *= alpha / num_nodes

    workload_matrix = data.workload_matrix  # shape: (f, i)
    delay_matrix = data.node_delay_matrix   # shape: (i, j)
    max_delay_array = data.max_delay_matrix  # shape: (f,)

    total_workload = np.sum(workload_matrix)
    if total_workload == 0:
        return node_util_score

    # 2. Calcolo vettoriale del max_workload_delay
    # Expand delay_matrix to (f, i, j) for broadcasting with max_delay_array
    delay_exp = np.broadcast_to(delay_matrix, (num_funcs, num_nodes, num_nodes))  # shape: (f, i, j)
    max_delay_exp = max_delay_array[:, np.newaxis, np.newaxis]  # shape: (f, 1, 1)

    # Boolean mask: delay[i, j] <= max_delay[f]
    delay_mask = delay_exp <= max_delay_exp

    # Apply mask, keep max delay ≤ max_delay[f] per (f, i)
    masked_delay = np.where(delay_mask, delay_exp, 0)
    max_delay_per_f_i = masked_delay.max(axis=2)  # shape: (f, i)

    max_workload_delay = np.sum(workload_matrix * max_delay_per_f_i)
    if max_workload_delay == 0:
        return node_util_score

    # 3. Costruisci x_matrix (i, f, j)
    x_matrix = np.zeros((num_nodes, num_funcs, num_nodes), dtype=np.float32)
    for (i, f, j), val_dict in x.items():
        x_matrix[i, f, j] = val_dict["val"]

    # workload[f, i] → reshape per broadcasting con x: (i, f, 1)
    workload_transposed = np.transpose(workload_matrix, (1, 0))[:, :, np.newaxis]  # (i, f, 1)
    delay_expanded = delay_matrix[:, np.newaxis, :]  # (i, 1, j)

    # Score = val * workload * delay
    delay_contrib = x_matrix * workload_transposed * delay_expanded
    delay_score = np.sum(delay_contrib) * (1 - alpha) / max_workload_delay

    return node_util_score + delay_score


'''
def score_minimize_node_delay_and_utilization(data, n, x, alpha):
    node_util_score = sum(1 for i in range(len(data.nodes)) if n[i]["val"])
    node_util_score *= alpha / len(data.nodes)

    total_workload = np.sum(data.workload_matrix)
    if total_workload == 0:
        return node_util_score  # Nessun workload → solo penalità nodi

    max_workload_delay = 0
    for f in range(len(data.functions)):
        max_func_delay = data.max_delay_matrix[f]
        for i in range(len(data.nodes)):
            max_node_delay = max([d for d in data.node_delay_matrix[i] if d <= max_func_delay], default=0)
            workload = data.workload_matrix[f, i]
            max_workload_delay += workload * max_node_delay

    if max_workload_delay == 0:
        return node_util_score  # Evitiamo divisione per zero

    delay_score = 0
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            workload = data.workload_matrix[f, i]
            for j in range(len(data.nodes)):
                val = x[(i, f, j)]["val"]
                delay = data.node_delay_matrix[i, j]
                delay_score += val * workload * delay * (1 - alpha) / max_workload_delay

    return node_util_score + delay_score
'''
def score_minimize_disruption(data, moved_from, moved_to, allocated, deallocated):
    w = np.ma.size(data.old_allocations_matrix)
    score = 0.0
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            score += moved_from[(f, j)]["val"] * w
            score += moved_to[(f, j)]["val"] * w
    score += allocated["val"] * (w - 1)
    score += deallocated["val"] * (w + 1)
    return score
