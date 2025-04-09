import numpy as np
from core.solvers.neptune.utils import init_x, init_c
from core.solvers.solver import Solver

class EFTTCMultiPathCPUBase(Solver):
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
        self.x, self.c = ef_ttc_adapted_to_data_multi_path(self.data, self.objective, self.alpha)
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
        delay_matrix = self.data.node_delay_matrix
        cores_matrix = np.array(self.data.cores_matrix)

        if not hasattr(self.data, "workload_on_source_matrix") or self.data.workload_on_source_matrix is None:
            workload_matrix = np.ones((len(self.data.functions), len(self.data.nodes)))
        else:
            workload_matrix = np.array(self.data.workload_on_source_matrix)

        total_cores_per_node = np.zeros(len(self.data.nodes))

        for i in range(len(self.data.nodes)):
            for f in range(len(self.data.functions)):
                for j in range(len(self.data.nodes)):
                    if self.x[i][f][j]:
                        delay = delay_matrix[i][j]
                        core = cores_matrix[f][j]
                        workload = workload_matrix[f][i]

                        total_cores_per_node[j] += core * workload

                        if self.objective == "delay":
                            score += delay * workload
                        elif self.objective == "utilization":
                            score += core * workload
                        elif self.objective == "delay_utilization":
                            score += workload * (self.alpha * delay + (1 - self.alpha) * core)

        self.data.total_cores_per_node = total_cores_per_node
        return score


class EF_TTC_MultiPath_MinDelay(EFTTCMultiPathCPUBase):
    def __init__(self, **kwargs):
        super().__init__(objective="delay", **kwargs)


class EF_TTC_MultiPath_MinUtilization(EFTTCMultiPathCPUBase):
    def __init__(self, **kwargs):
        super().__init__(objective="utilization", **kwargs)


class EF_TTC_MultiPath_MinDelayAndUtilization(EFTTCMultiPathCPUBase):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(objective="delay_utilization", alpha=alpha, **kwargs)


def ef_ttc_adapted_to_data_multi_path(data, objective="delay_utilization", alpha=0.5):
    F, N = len(data.functions), len(data.nodes)
    node_memories = data.node_memory_matrix
    node_cores = data.node_cores_matrix
    function_memories = data.function_memory_matrix
    cores_matrix = np.array(data.cores_matrix)

    if not hasattr(data, "workload_on_source_matrix") or data.workload_on_source_matrix is None:
        print("[EF-TTC] workload_on_source_matrix mancante, uso uniform distribution")
        workload_matrix = np.ones((F, N))
    else:
        workload_matrix = np.array(data.workload_on_source_matrix)

    delay_matrix = np.array(data.node_delay_matrix)
    max_delays = data.max_delay_matrix

    x = {i: {f: {j: 0 for j in range(N)} for f in range(F)} for i in range(N)}
    c = {f: {j: 0 for j in range(N)} for f in range(F)}

    memory_used = np.zeros(N)
    cores_used = np.zeros(N)

    for f in range(F):
        best_j = None
        best_score = float('inf')

        for j in range(N):
            if memory_used[j] + function_memories[f] > node_memories[j]:
                continue
            if cores_used[j] + cores_matrix[f][j] > node_cores[j]:
                continue

            total_score = 0
            for i in range(N):
                delay = delay_matrix[i][j]
                if delay > max_delays[f]:
                    continue
                workload = workload_matrix[f][i]
                core = cores_matrix[f][j]

                if objective == "delay":
                    total_score += delay * workload
                elif objective == "utilization":
                    total_score += core * workload
                elif objective == "delay_utilization":
                    total_score += workload * (alpha * delay + (1 - alpha) * core)

            if total_score < best_score:
                best_score = total_score
                best_j = j

        if best_j is not None:
            c[f][best_j] = 1
            memory_used[best_j] += function_memories[f]
            cores_used[best_j] += cores_matrix[f][best_j]
            for i in range(N):
                if delay_matrix[i][best_j] <= max_delays[f]:
                    x[i][f][best_j] = 1

    return x, c


def convert_to_full_x_and_c(data, x, c):
    x_matrix = np.zeros((len(data.nodes), len(data.functions), len(data.nodes)))
    for i in x:
        for f in x[i]:
            for j in x[i][f]:
                if x[i][f][j]:
                    x_matrix[i][f][j] = 1.0

    c_matrix = np.zeros((len(data.functions), len(data.nodes)))
    for f in c:
        for j in c[f]:
            if c[f][j]:
                c_matrix[f][j] = 1.0

    return x_matrix, c_matrix
