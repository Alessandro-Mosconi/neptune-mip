import numpy as np


def score_minimize_network_delay(data, x):
    total_delay = 0.0
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            for j in range(len(data.nodes)):
                val = x[(i, f, j)]["val"]
                delay = data.node_delay_matrix[i, j]
                workload = data.workload_matrix[f, i]
                total_delay += val * delay * workload
    return total_delay


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
    # Stampa dell'input (n)
    print("Input 'n' (allocation state per node):")
    for i in range(len(data.nodes)):
        print(f"Node {i}: {n[i]['val']}")
    return sum(1 for i in range(len(data.nodes)) if n[i]["val"])

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
