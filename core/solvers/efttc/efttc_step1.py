import copy
from gc import get_objects

from .utils import *
from ..solver import Solver


class EfttcStepBase(Solver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x, self.c = {}, {}

    def init_vars(self):
        init_x(self.data, self.x)
        init_c(self.data, self.c)

    def init_constraints(self):
        pass
    def init_objective(self):
        pass

    def get_constraints(self):
        return constrain_c_according_to_x(self.data, self.c, self.x) and constrain_memory_usage(self.data, self.c)

    def solve(self, max_iterations=100):
        self.init_vars()
        best_score = float("inf")
        best_solution = None

        for _ in range(max_iterations):
            self.generate_candidate()  # TTC-style o euristica

            if self.get_constraints():  # vincoli soddisfatti?
                score = self.get_objective()  # valuta la soluzione

                if score < best_score:
                    best_score = score
                    best_solution = (
                        copy.deepcopy(self.x),
                        copy.deepcopy(self.c),
                        copy.deepcopy(getattr(self, "n", {}))  # se presente
                    )

        if best_solution:
            self.x, self.c = best_solution[0], best_solution[1]
            if hasattr(self, "n"):
                self.n = best_solution[2]
        else:
            print("⚠️ Nessuna soluzione valida trovata.")

    def generate_candidate(self):
        # reset c and x
        for key in self.c:
            self.c[key]["val"] = False
        for key in self.x:
            self.x[key]["val"] = 0.0

        for f in range(len(self.data.functions)):
            # Costrutto: [(node_id, total_delay), ...] per questa funzione
            candidates = []
            for j in range(len(self.data.nodes)):
                # Calcolo del "costo" per assegnare f al nodo j
                total_delay = 0
                for i in range(len(self.data.nodes)):
                    total_delay += self.data.workload_matrix[f, i] * self.data.node_delay_matrix[i, j]
                candidates.append((j, total_delay))

            # Ordina i nodi da migliore (delay minimo) a peggiore
            candidates.sort(key=lambda x: x[1])

            # Prova ad assegnare f al miglior nodo disponibile
            for j, _ in candidates:
                mem_required = self.data.function_memory_matrix[f]
                mem_used = sum(
                    self.data.function_memory_matrix[f2] if self.c[(f2, j)]["val"] else 0
                    for f2 in range(len(self.data.functions))
                )
                if mem_used + mem_required <= self.data.node_memory_matrix[j]:
                    self.c[(f, j)]["val"] = True
                    for i in range(len(self.data.nodes)):
                        self.x[(i, f, j)]["val"] = 1.0
                    break  # assegnato, passo alla prossima funzione

    def results(self):
        x, c = output_x_and_c(self.data, self.x, self.c)
        print("Step 1 - x:", x, sep='\n')
        print("Step 1 - c:", c, sep='\n')
        self.data.prev_x = x
        self.data.prev_c = c 
        return x, c

    def get_objective(self):
        raise NotImplementedError("Efttc must implement getObjective()")

    def score(self):
        return self.get_objective()
    

class EfttcStep1CPUBase(EfttcStepBase):

    def get_constraints(self):
        return (super().get_constraints()
                and constrain_handle_required_requests(self.data, self.x)
                and constrain_CPU_usage(self.data, self.x))

       
class EfttcStep1CPUMinUtilization(EfttcStep1CPUBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = {}

    def init_vars(self):
        super().init_vars()
        init_n(self.data, self.n)

    def get_constraints(self):
        return (super().init_constraints() and
            constrain_n_according_to_c(self.data, self.n, self.c) and
            constrain_budget(self.data, self.n))

    def get_objective(self):
        return score_minimize_node_utilization(self.data, self.n)

    def results(self):
        x, c = super().results()
        n = output_n(self.data, self.n)
        self.data.prev_n = n
        print("Step 1 - n:", n, sep='\n')
        return x, c


class EfttcStep1CPUMinDelay(EfttcStep1CPUBase):

    def get_objective(self):
        return score_minimize_network_delay(self.data, self.x)

class EfttcStep1CPUMinDelayAndUtilization(EfttcStep1CPUMinUtilization):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def load_data(self, data):
        data.alpha = self.alpha
        super().load_data(data)
    

    def get_objective(self):
        return score_minimize_node_delay_and_utilization(self.data, self.objective, self.n, self.x, self.alpha)
