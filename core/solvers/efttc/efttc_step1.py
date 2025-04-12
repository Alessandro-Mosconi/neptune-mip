import copy

from .utils import *
from ..solver import Solver


class EfttcStepBase(Solver):
    def __init__(self, **kwargs):
        self.invalid_pairs = set()
        super().__init__(**kwargs)
        self.x, self.c = {}, {}
        self.assigned_functions = set()
        self.assigned_nodes = set()
        self._score_cache = {}

    def init_vars(self):
        init_x(self.data, self.x)
        init_c(self.data, self.c)

    def init_constraints(self):
        pass  # I vincoli vengono verificati tramite get_constraints()

    def get_constraints(self):
        return True # constrain_c_according_to_x(self.data, self.c, self.x) and constrain_memory_usage(self.data, self.c)

    def snapshot_vars(self):
        return {
            "x": {k: v.copy() for k, v in self.x.items()},
            "c": {k: v.copy() for k, v in self.c.items()},
            "n": {k: v.copy() for k, v in self.n.items()} if hasattr(self, "n") else None
        }


    def restore_vars(self, snapshot):
        self.x = snapshot["x"]
        self.c = snapshot["c"]
        if snapshot["n"] is not None:
            self.n = snapshot["n"]

    def solve(self):
        self.init_vars()

        print("DELAY MATRIX")
        print(self.data.node_delay_matrix)

        print("WORKLOAD MATRIX")
        print(self.data.workload_matrix)

        remaining_functions = set(range(len(self.data.functions)))
        remaining_nodes = set(range(len(self.data.nodes)))
        tried_cycles = set()
        while remaining_functions:
            print(f"\n=== Iterazione TTC ===")
            print(f"🧱 Numero di invalid_pairs: {len(self.invalid_pairs)}")

            print("♻️ remaining_functions")
            print(remaining_functions)
            print("♻️ remaining_nodes")
            print(remaining_nodes)

            # ✅ Svuota la cache per evitare score obsoleti
            self._score_cache.clear()

            graph = self.build_preference_graph(remaining_functions, remaining_nodes)
            print("🤖 graph")
            print(graph)
            cycle = self.find_cycle(graph)
            if not cycle:
                print("❌ Nessun ciclo trovato. Allocazione terminata.")
                break
            cycle_key = tuple(sorted(cycle))
            if cycle_key in tried_cycles:
                print("⚠️ Ciclo già provato e non valido. Interrompo per evitare loop.")
                break

            # backup stato
            snapshot = self.snapshot_vars()

            print("✔️ Ciclo trovato, assegno")
            hasSuccess = self.assign_cycle(cycle)
            self.update_n_from_c()
            # print("🍀 C: ")
            # print(self.c)

            if not hasSuccess:
                print("⛔️ Alcune assegnazioni non sono valide")
                self.restore_vars(snapshot)
                tried_cycles.add(cycle_key)
                continue

            # Verifica i vincoli globali
            if self.get_constraints():
                remaining_functions -= {f for f, _ in cycle}
                for _, j in cycle:
                    mem_used = sum(
                        self.data.function_memory_matrix[f2] if self.c[(f2, j)]["val"] else 0
                        for f2 in range(len(self.data.functions))
                    )
                    if mem_used == self.data.node_memory_matrix[j]:
                        remaining_nodes.discard(j)

                    if mem_used > self.data.node_memory_matrix[j]:
                        self.restore_vars(snapshot)
                        print("⚠️ Ciclo trovato ma non ci sta nel nodo")
                        for f, j in cycle:
                            self.invalid_pairs.add((f, j))
                    else:
                        print("✅ Ciclo trovato e variabili aggiornate")
                        self.snapshot_vars()


            else:
                print("⚠️ Ciclo trovato ma viola i vincoli globali. Scartato.")
                self.restore_vars(snapshot)
                tried_cycles.add(cycle_key)
                for f, j in cycle:
                    self.invalid_pairs.add((f, j))

        print("🍀 C: ")
        print(self.c)

    def build_preference_graph(self, remaining_functions, remaining_nodes):
        graph = {}
        for f in remaining_functions:
            preferred_nodes = self.rank_nodes_for_function(f, remaining_nodes)
            if not preferred_nodes:
                continue
            j = preferred_nodes[0]  # best node for function f
            graph[f] = ~j  # store node as negative key (~j)

        for j in remaining_nodes:
            preferred_functions = self.rank_functions_for_node(j, remaining_functions)
            if not preferred_functions:
                continue
            f = preferred_functions[0]  # best function for node j
            graph[~j] = f  # node ~j prefers function f

        return graph

    def rank_nodes_for_function(self, f, node_pool):
        valid_nodes = [j for j in node_pool if (f, j) not in self.invalid_pairs]
        return sorted(valid_nodes, key=lambda j: (self.score_local(f, j), abs(j)))

    def rank_functions_for_node(self, j, function_pool):
        return sorted(function_pool, key=lambda f: (self.score_local(f, j), abs(f)))

    def find_cycle(self, graph):
        visited = set()
        for start in graph:
            if start in visited:
                continue
            path = []
            current = start
            local_visited = set()

            while current not in local_visited:
                local_visited.add(current)
                path.append(current)

                if current not in graph:
                    break

                next_node = graph[current]
                path.append(next_node)

                if next_node in local_visited:
                    # ciclo trovato
                    cycle_start = path.index(next_node)
                    cycle = [(path[i], path[i + 1]) for i in range(cycle_start, len(path) - 1)]
                    cleaned = []
                    seen = set()
                    for a, b in cycle:
                        if a >= 0 and b < 0:
                            pair = (a, ~b)
                        elif a < 0 and b >= 0:
                            pair = (b, ~a)
                        else:
                            continue  # caso anomalo
                        if pair not in seen:
                            seen.add(pair)
                            cleaned.append(pair)
                    return cleaned

                current = next_node

            visited |= local_visited
        return []

    def update_n_from_c(self):
        if not hasattr(self, "n"):
            return
        for j in range(len(self.data.nodes)):
            self.n[j]["val"] = any(
                self.c[(f, j)]["val"]
                for f in range(len(self.data.functions))
            )

    def assign_cycle(self, cycle):
        success = False  # flag per tracciare se almeno un'assegnazione è andata a buon fine
        for f, j in cycle:
            if isinstance(f, int) and isinstance(j, int):
                j = ~j if j < 0 else j  # convert node back from negative if needed
                if not self.can_assign(f, j):
                    print(f"❌ Non posso assegnare f={f} a j={j} perché non ci sta")
                    self.invalid_pairs.add((f, j))
                    continue
                self.c[(f, j)]["val"] = True
                print(f"📍 Assegno funzione {f} al nodo {j} via ciclo TTC")
                for i in range(len(self.data.nodes)):
                    self.x[(i, f, j)]["val"] = 1.0
                print(f"✅ Assegnata funzione {f} al nodo {j} via ciclo TTC")
                success = True
        return success

    def can_assign(self, f, j):
        mem_required = self.data.function_memory_matrix[f]
        mem_used = sum(
            self.data.function_memory_matrix[f2] if self.c[(f2, j)]["val"] else 0
            for f2 in range(len(self.data.functions))
        )
        return mem_used + mem_required <= self.data.node_memory_matrix[j]

    def get_objective(self):
        raise NotImplementedError("Efttc must implement getObjective()")

    def score_local(self, f, j):
        raise NotImplementedError("Efttc must implement score_local(f, j)")

    def results(self):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)


        x, c = output_x_and_c(self.data, self.x, self.c)
        print("Step 1 - x:", x, sep='\n')
        print("Step 1 - c:", c, sep='\n')
        return x, c

    def score(self):
        return self.get_objective()


class EfttcStep1CPUBase(EfttcStepBase):
    def get_constraints(self):
        ok = super().get_constraints()
        if not ok:
            print("❌ Vincolo fallito: super().get_constraints()")

        ok3 = constrain_CPU_usage(self.data, self.x)
        if not ok3:
            print("❌ Vincolo fallito: constrain_CPU_usage")

        return ok and ok3


class EfttcStep1CPUMinUtilization(EfttcStep1CPUBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = {}

    def init_vars(self):
        super().init_vars()
        init_n(self.data, self.n)

    def get_constraints(self):
        return (super().get_constraints() and
                # constrain_n_according_to_c(self.data, self.n, self.c) and
                constrain_budget(self.data, self.n))

    def get_objective(self):
        return score_minimize_node_utilization(self.data, self.n)
    '''
    def score_local(self, f, j):
        return sum(self.c[(f2, j)]["val"] for f2 in range(len(self.data.functions)))
   
    def score_local(self, f, j):
        utilizzo = sum(self.c[(f2, j)]["val"] for f2 in range(len(self.data.functions)))
        cost = self.data.node_costs[j]
        budget_left = self.data.node_budget - sum(
            self.n[k]["val"] * self.data.node_costs[k]
            for k in range(len(self.data.nodes))
        )
        return utilizzo * cost if cost <= budget_left else float('inf')  # evita nodi che sforano
    
    
    def score_local(self, f, j):
        utilizzo = sum(self.c[(f2, j)]["val"] for f2 in range(len(self.data.functions)))
        return utilizzo * self.data.node_costs[j]  # penalizza nodi costosi
        
    '''

    def score_local(self, f, j):
        #TODO considerare funzioni già allocate inizializzare c con data.actual_cpu_allocations
        utilizzo = sum(self.c[(f2, j)]["val"] for f2 in range(len(self.data.functions)))
        costo = self.data.node_costs[j]

        # Penalizza nodi poco utilizzati e costosi: più è alto lo score, peggio è
        return costo / (1 + utilizzo)

    def results(self):
        x, c = super().results()
        n = output_n(self.data, self.n)
        self.data.prev_n = n
        print("Step 1 - n:", n, sep='\n')
        return x, c


class EfttcStep1CPUMinDelay(EfttcStep1CPUBase):
    def get_objective(self):
        return score_minimize_network_delay(self.data, self.x)
    '''
    def score_local(self, f, j):
        return self.data.node_delay_matrix[:, j].dot(self.data.workload_matrix[f])
    '''
    def score_local(self, f, j):
        key = (f, j)
        if key not in self._score_cache:
            self._score_cache[key] = self.data.node_delay_matrix[:, j].dot(self.data.workload_matrix[f])
        return self._score_cache[key]


class EfttcStep1CPUMinDelayAndUtilization(EfttcStep1CPUMinUtilization):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def load_data(self, data):
        data.alpha = self.alpha
        super().load_data(data)

    def get_objective(self):
        return score_minimize_node_delay_and_utilization(self.data, self.n, self.x, self.alpha)

    def score_local(self, f, j):
        util = sum(self.c[(f2, j)]["val"] for f2 in range(len(self.data.functions)))
        delay = self.data.node_delay_matrix[:, j].dot(self.data.workload_matrix[f])
        return self.alpha * util + (1 - self.alpha) * delay


class EfttcStep1CPUMinDelayTTC(EfttcStepBase):
    def get_constraints(self):
        return (super().get_constraints()
                #and constrain_handle_required_requests(self.data, self.x)
                and constrain_CPU_usage(self.data, self.x))

    def get_objective(self):
        return score_minimize_network_delay(self.data, self.x)

    def score_local(self, f, j):
        return self.data.node_delay_matrix[:, j].dot(self.data.workload_matrix[f])