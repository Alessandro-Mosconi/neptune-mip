import numpy as np

class EFTTCUnifiedBase:
    def __init__(self, beta=1.0, **kwargs):
        self.beta = beta
        self.data = None
        self.x = None
        self.c = None

    def load_data(self, data):
        self.data = data

    def init_allocation(self):
        self.x = np.zeros((len(self.data.nodes), len(self.data.functions), len(self.data.nodes)))
        self.c = np.zeros((len(self.data.functions), len(self.data.nodes)))

    def solve(self):
        raise NotImplementedError()

    def results(self):
        return self.x, self.c

    def score(self):
        return {
            "step1": self._score_step1(),
            "step2": self._score_step2()
        }

    def _score_step2(self):
        total = 0
        for f in range(len(self.data.functions)):
            for j in range(len(self.data.nodes)):
                prev = self.data.old_allocations_matrix[f][j]
                curr = self.c[f][j]
                if prev != curr:
                    total += self.beta
        return total

class EF_TTC_MinDelay(EFTTCUnifiedBase):
    def solve(self):
        self.init_allocation()
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                best_j, best_delay = None, float('inf')
                for j in range(len(self.data.nodes)):
                    delay = self.data.node_delay_matrix[i][j]
                    if delay < best_delay:
                        best_j = j
                        best_delay = delay
                self.x[i][f][best_j] = self.data.workload_matrix[f][i]
                self.c[f][best_j] = 1
        return True

    def _score_step1(self):
        total = 0
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                for j in range(len(self.data.nodes)):
                    total += self.x[i][f][j] * self.data.node_delay_matrix[i][j]
        return total

class EF_TTC_MinUtilization(EFTTCUnifiedBase):
    def solve(self):
        self.init_allocation()
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                best_j, best_util = None, float('inf')
                for j in range(len(self.data.nodes)):
                    cpu = self.data.core_per_req_matrix[f][j]
                    cap = self.data.node_cores_matrix[j]
                    util = cpu / cap if cap > 0 else float('inf')
                    if util < best_util:
                        best_j = j
                        best_util = util
                self.x[i][f][best_j] = self.data.workload_matrix[f][i]
                self.c[f][best_j] = 1
        return True

    def _score_step1(self):
        total = 0

class EF_TTC_MinDelayAndUtilization(EFTTCUnifiedBase):
    def __init__(self, alpha=0.5, beta=1.0, **kwargs):
        super().__init__(beta=beta, **kwargs)
        self.alpha = alpha

    def solve(self):
        self.init_allocation()
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
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
                self.x[i][f][best_j] = self.data.workload_matrix[f][i]
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
