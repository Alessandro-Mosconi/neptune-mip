import itertools
import numpy as np


def constrain_moved_from(data, moved_from, c):
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            val = moved_from[(f, j)]["val"]
            expected = c[(f, j)]["val"] - data.old_allocations_matrix[f, j]
            if val < 0 or val < expected:
                return False
    return True


def constrain_moved_to(data, moved_to, c):
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            val = moved_to[(f, j)]["val"]
            expected = data.old_allocations_matrix[f, j] - c[(f, j)]["val"]
            if val < 0 or val < expected:
                return False
    return True


def constrain_migrations(data, c, allocated, deallocated):
    sum_old = np.sum(data.old_allocations_matrix)
    sum_new = sum(1 if c[(f, j)]["val"] else 0 for f in range(len(data.functions)) for j in range(len(data.nodes)))

    if allocated["val"] > 0:
        return False
    if sum_old - sum_new < allocated["val"]:
        return False
    if deallocated["val"] > 0:
        return False
    if sum_new - sum_old < deallocated["val"]:
        return False
    return True


def constrain_deletions(data, c, allocated, deallocated):
    sum_old = np.sum(data.old_allocations_matrix)
    sum_new = sum(1 if c[(f, j)]["val"] else 0 for f in range(len(data.functions)) for j in range(len(data.nodes)))

    result = deallocated["val"] + allocated["val"] + sum_old - sum_new
    return result >= 0


def constrain_creations(data, c, allocated, deallocated):
    sum_old = np.sum(data.old_allocations_matrix)
    sum_new = sum(1 if c[(f, j)]["val"] else 0 for f in range(len(data.functions)) for j in range(len(data.nodes)))

    result = deallocated["val"] + allocated["val"] - sum_old + sum_new
    return result >= 0

def constrain_network_delay(data, x, soften_step1_sol):
    lhs = sum(
        x[(i, f, j)]["val"] * data.node_delay_matrix[i, j] * data.workload_matrix[f, i]
        for i in range(len(data.nodes))
        for f in range(len(data.functions))
        for j in range(len(data.nodes))
    )
    rhs = soften_step1_sol * sum(
        data.node_delay_matrix[i, j] * data.workload_matrix[f, i] * data.prev_x[i, f, j]
        for i in range(len(data.nodes))
        for f in range(len(data.functions))
        for j in range(len(data.nodes))
    )
    return lhs <= rhs + 1e-6


def constrain_node_utilization(data, n, soften_step1_sol):
    total_used = sum(1 for i in range(len(data.nodes)) if n[i]["val"])
    return total_used <= data.max_score * soften_step1_sol + 1e-6


def constrain_score(data, x, n, alpha, soften_step1_sol):
    max_func_delay = data.max_delay_matrix
    max_node_delay = data.node_delay_matrix.max(axis=0)  # shape: (num_nodes,)
    func_matrix = np.vstack([max_func_delay for _ in range(len(data.nodes))])
    node_matrix = np.vstack([max_node_delay for _ in range(len(data.functions))])
    max_delay_matrix = np.maximum(func_matrix, node_matrix.T)

    node_score = sum((alpha / len(data.nodes)) for i in range(len(data.nodes)) if n[i]["val"])

    delay_score = 0.0
    for i in range(len(data.nodes)):
        for f in range(len(data.functions)):
            for j in range(len(data.nodes)):
                val = x[(i, f, j)]["val"]
                delay = data.node_delay_matrix[i, j]
                workload = data.workload_matrix[f, i]
                denom = max_delay_matrix[i, f] if max_delay_matrix[i, f] != 0 else 1
                delay_score += val * (1 - alpha) * workload * delay / denom

    return node_score + delay_score <= data.max_score * soften_step1_sol + 1e-6
