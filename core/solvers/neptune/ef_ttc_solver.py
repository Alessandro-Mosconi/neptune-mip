# File: ef_ttc_solver.py

import numpy as np
from .utils import init_x, init_c
from ..solver import Solver

class EFTTCStep1CPUBase(Solver):
    def __init__(self, objective="delay_utilization", alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.objective = objective
        self.alpha = alpha
        self.x = {}
        self.c = {}

    def load_data(self, data):
        self.data = data

    def init_vars(self):
        init_x(self.data, self.solver, self.x)
        init_c(self.data, self.solver, self.c)

    def init_constraints(self):
        pass

    def init_objective(self):
        pass

    def solve(self):
        self.x, self.c = ef_ttc_adapted_to_data(self.data, self.objective, self.alpha)
        return True

    def results(self):
        x, c = convert_to_full_x_and_c(self.data, self.x, self.c)
        print("Step 1 (EF-TTC) - x:", x, sep='\n')
        print("Step 1 (EF-TTC) - c:", c, sep='\n')
        self.data.prev_x = x
        self.data.prev_c = c
        return x, c

    def score(self):
        score = 0.0
        delay_matrix = self.data.delay_matrix
        cores_matrix = np.array(self.data.cores_matrix)

        for j in self.x:
            for f in self.x[j]:
                if self.x[j][f]:
                    delay = delay_matrix[f][j]
                    core = cores_matrix[f][j]

                    if self.objective == "delay":
                        score += delay
                    elif self.objective == "utilization":
                        score += core
                    elif self.objective == "delay_utilization":
                        score += self.alpha * delay + (1 - self.alpha) * core

        return score


class EF_TTC_MinDelay(EFTTCStep1CPUBase):
    def __init__(self, **kwargs):
        super().__init__(objective="delay", **kwargs)


class EF_TTC_MinUtilization(EFTTCStep1CPUBase):
    def __init__(self, **kwargs):
        super().__init__(objective="utilization", **kwargs)


class EF_TTC_MinDelayAndUtilization(EFTTCStep1CPUBase):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(objective="delay_utilization", alpha=alpha, **kwargs)


def ef_ttc_adapted_to_data(data, objective="delay_utilization", alpha=0.5):
    function_names = data.functions
    node_names = data.nodes
    node_memories = data.node_memory_matrix
    node_cores = data.node_cores_matrix
    function_memories = data.function_memory_matrix
    max_delays = data.max_delay_matrix
    cores_matrix = np.array(data.cores_matrix)

    # Fallback: workload matrix
    if not hasattr(data, "workload_on_source_matrix") or data.workload_on_source_matrix is None:
        print("[EF-TTC] workload_on_source_matrix mancante, uso uniform distribution")
        workload_matrix = np.ones((len(function_names), len(node_names)))
    else:
        workload_matrix = np.array(data.workload_on_source_matrix)

    # Fallback: node delay matrix
    if not hasattr(data, "node_delay_matrix") or data.node_delay_matrix is None:
        data.node_delay_matrix = np.array([[0 if i == j else 1 for j in range(len(node_names))] for i in range(len(node_names))])

    node_delay_matrix = np.array(data.node_delay_matrix)

    # Converti node_delay_matrix (n_nodes x n_nodes) → delay_matrix (n_functions x n_nodes)
    delay_matrix = np.zeros((len(function_names), len(node_names)))
    for f in range(len(function_names)):
        for j in range(len(node_names)):
            total_delay = 0
            total_weight = 0
            for i in range(len(node_names)):
                w = workload_matrix[f][i]
                d = node_delay_matrix[i][j]
                total_delay += w * d
                total_weight += w
            delay_matrix[f][j] = total_delay / total_weight if total_weight > 0 else 0

    data.delay_matrix = delay_matrix  # salva per uso in score()

    x = {j: {r: 0 for r in range(len(function_names))} for j in range(len(node_names))}
    c = {f: {j: 0 for j in range(len(node_names))} for f in range(len(function_names))}

    memory_used = np.zeros(len(node_names))
    cores_used = np.zeros(len(node_names))

    for f_idx, f in enumerate(function_names):
        scores = []
        for j in range(len(node_names)):
            delay = delay_matrix[f_idx][j]
            mem_util = memory_used[j] / node_memories[j] if node_memories[j] else 1
            core_util = cores_used[j] / node_cores[j] if node_cores[j] else 1
            util_score = mem_util + core_util

            if objective == "utilization":
                score = util_score
            elif objective == "delay":
                score = delay
            elif objective == "delay_utilization":
                score = alpha * delay + (1 - alpha) * util_score
            else:
                score = delay

            scores.append(score)

        preferred_nodes = np.argsort(scores)

        for j in preferred_nodes:
            if delay_matrix[f_idx][j] > max_delays[f_idx]:
                continue

            if memory_used[j] + function_memories[f_idx] > node_memories[j]:
                continue

            if cores_used[j] + cores_matrix[f_idx][j] > node_cores[j]:
                continue

            c[f_idx][j] = 1
            x[j][f_idx] = 1
            memory_used[j] += function_memories[f_idx]
            cores_used[j] += cores_matrix[f_idx][j]
            break

    return x, c


def convert_to_full_x_and_c(data, x, c):
    x_matrix = np.zeros((len(data.nodes), len(data.functions), len(data.nodes)))
    for j in x:
        for f in x[j]:
            if x[j][f]:
                for i in range(len(data.nodes)):
                    x_matrix[i][f][j] = 1.0 if i == j else 0.0

    c_matrix = np.zeros((len(data.functions), len(data.nodes)))
    for f in c:
        for j in c[f]:
            if c[f][j]:
                c_matrix[f][j] = 1.0

    return x_matrix, c_matrix
