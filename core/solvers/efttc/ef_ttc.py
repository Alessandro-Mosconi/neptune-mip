import json
from collections import defaultdict

import numpy as np

# test
class EFTTCBase:
    def __init__(self, alpha=0.5, beta=1.0, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.data = None
        self.x = None
        self.c = None

    def load_data(self, data):
        self.data = data

    def init_allocation(self):
        self.x = np.zeros((len(self.data.nodes), len(self.data.functions), len(self.data.nodes)))
        self.c = np.zeros((len(self.data.functions), len(self.data.nodes)))

    def results(self):
        return (
            self._convert_x_matrix(self.x, self.data.nodes, self.data.functions),
            self._convert_c_matrix(self.c, self.data.functions, self.data.nodes)
        )

    def _convert_x_matrix(self, matrix, nodes, functions):
        routings = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        for i, source in enumerate(nodes):
            for f, function in enumerate(functions):
                for j, destination in enumerate(nodes):
                    if matrix[i][f][j] > 0.001:
                        routings[source][function][destination] = round(float(matrix[i][f][j]), 3)
        return json.loads(json.dumps(routings))  # forza serializzabilità

    def _convert_c_matrix(self, matrix, functions, nodes):
        allocations = defaultdict(lambda : defaultdict(bool))
        for f, function in enumerate(functions):
            for j, destination in enumerate(nodes):
                if matrix[f][j] > 0.001:
                    allocations[function][destination] = True
        return json.loads(json.dumps(allocations))  # forza serializzabilità

    def score(self):
        return {
            "step1": self._score_step1(),
            "step2": self._score_step2()
        }

    def _score_step2(self):
        total = 0
        for f in range(len(self.data.functions)):
            for j in range(len(self.data.nodes)):
                if self.data.old_allocations_matrix[f][j] != self.c[f][j]:
                    total += self.beta
        return total

class EFTTCMinDelay(EFTTCBase):
    def solve(self):
        self.init_allocation()
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                workload = self.data.workload_matrix[f][i]
                if workload <= 0:
                    continue  # niente da assegnare

                best_j, best_score = None, float('inf')
                for j in range(len(self.data.nodes)):
                    delay = self.data.node_delay_matrix[i][j]
                    if delay < best_score:
                        best_j = j
                        best_score = delay

                # aggiorna assegnazione solo se c'è workload
                if best_j is not None:
                    self.x[i][f][best_j] = workload
                    self.c[f][best_j] = 1
        return True


    def _score_step1(self):
        total = 0
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                for j in range(len(self.data.nodes)):
                    total += self.x[i][f][j] * self.data.node_delay_matrix[i][j]
        return total

class EFTTCMinUtilization(EFTTCBase):
    def solve(self):
        self.init_allocation()
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                workload = self.data.workload_matrix[f][i]
                if workload <= 0:
                    continue

                best_j, best_score = None, float('inf')
                for j in range(len(self.data.nodes)):
                    cpu = self.data.core_per_req_matrix[f][j]
                    cap = self.data.node_cores_matrix[j]
                    util = cpu / cap if cap > 0 else float('inf')
                    if util < best_score:
                        best_j = j
                        best_score = util

                if best_j is not None:
                    self.x[i][f][best_j] = workload
                    self.c[f][best_j] = 1
        return True


    def _score_step1(self):
        total = 0
        for f in range(len(self.data.functions)):
            for j in range(len(self.data.nodes)):
                if self.c[f][j] > 0:
                    cpu = self.data.core_per_req_matrix[f][j]
                    cap = self.data.node_cores_matrix[j]
                    total += cpu / cap if cap > 0 else 0
        return total

class EFTTCMinDelayAndUtilization(EFTTCBase):
    def solve(self):
        self.init_allocation()
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                workload = self.data.workload_matrix[f][i]
                if workload <= 0:
                    continue

                best_j, best_score = None, float('inf')
                for j in range(len(self.data.nodes)):
                    delay = self.data.node_delay_matrix[i][j]
                    cpu = self.data.core_per_req_matrix[f][j]
                    cap = self.data.node_cores_matrix[j]
                    util = cpu / cap if cap > 0 else float('inf')
                    score = self.alpha * util + (1 - self.alpha) * delay
                    if score < best_score:
                        best_j = j
                        best_score = score

                if best_j is not None:
                    self.x[i][f][best_j] = workload
                    self.c[f][best_j] = 1
        return True


    def _score_step1(self):
        total = 0
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                for j in range(len(self.data.nodes)):
                    if self.x[i][f][j] > 0:
                        delay = self.data.node_delay_matrix[i][j]
                        cpu = self.data.core_per_req_matrix[f][j]
                        cap = self.data.node_cores_matrix[j]
                        util = cpu / cap if cap > 0 else 0
                        total += self.alpha * util + (1 - self.alpha) * delay
        return total
