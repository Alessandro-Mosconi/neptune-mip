import copy
from datetime import datetime
from .utils import *
from ..solver import Solver


class EfttcStepBase(Solver):
    def __init__(self, **kwargs):
        self.invalid_pairs = set()
        super().__init__(**kwargs)
        self.x, self.c, self.n = {}, {}, {}
        self._score_cache = {}
        self.objective='min_delay_min_utilization'

    def init_vars(self):
        init_x(self.data, self.x)
        init_c(self.data, self.c)
        init_n(self.data, self.n)

    def init_constraints(self):
        pass

    def get_constraints(self):
        return True

    def snapshot_vars(self):
        return {
            "x": {k: {"name": v["name"], "val": v["val"]} for k, v in self.x.items()},
            "c": {k: {"name": v["name"], "val": v["val"]} for k, v in self.c.items()},
            "n": {k: {"name": v["name"], "val": v["val"]} for k, v in self.n.items()} if hasattr(self, "n") else None
        }

    def restore_vars(self, snapshot):
        self.x = snapshot["x"]
        self.c = snapshot["c"]
        if snapshot["n"] is not None:
            self.n = snapshot["n"]

    def solve(self):
        self.init_vars()
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
            #print("🤖 graph")
            #print(graph)
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
            hasSuccess = self.can_assign_cycle(cycle)

            if not hasSuccess:
                print("⛔️ Alcune assegnazioni non sono valide")
                #self.restore_vars(snapshot)
                tried_cycles.add(cycle_key)
                continue

            # Verifica i vincoli globali
            if self.get_constraints():
                self.handle_cycle(cycle, remaining_functions, remaining_nodes, snapshot)
            else:
                print("⚠️ Ciclo trovato ma viola i vincoli globali. Scartato.")
                tried_cycles.add(cycle_key)
                self.restore_vars(snapshot)
                for f, j in cycle:
                    self.invalid_pairs.add((f, j))

    def handle_cycle(self, cycle, remaining_functions, remaining_nodes, snapshot):
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
                for f, j in cycle:
                    self.invalid_pairs.add((f, j))

                if 'min_delay' in self.objective:

                    for f, _ in cycle:
                        if (self.find_best_node_by_delay_improvement(f, remaining_nodes, self.data, self.invalid_pairs) is not None):
                            print("✅ Funzione può essere migliorata")
                        else:
                            print("❌ Funzione non può essere migliorata")
                            remaining_functions.remove(f)
                else:
                    for f, _ in cycle:
                        remaining_functions.discard(f)

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

    def change_n_one(self, n, c, j):
        n[j]["val"] = any(
            c[(f, j)]["val"]
            for f in range(len(self.data.functions))
        )

    def change_x_one(self, x, c, f):
        for i in range(len(self.data.nodes)):
            active_nodes = [
                j for j in range(len(self.data.nodes))
                if c[(f, j)]["val"]
            ]

            if not active_nodes:
                continue

            delays = {j: self.data.node_delay_matrix[i][j] for j in active_nodes}
            min_delay = min(delays.values())
            best_nodes = [j for j, d in delays.items() if abs(d - min_delay) < 1e-6]

            val = 1.0 / len(best_nodes)
            for j in active_nodes:
                x[(i, f, j)]["val"] = val if j in best_nodes else 0.0

    def find_best_node_by_delay_improvement(self, f, candidate_nodes, data, invalid_pairs):
        if not candidate_nodes:
            print(f"🚫 Nessun nodo candidato per la funzione {f}")
            return None

        # Filtro nodi validi: non già assegnati e non invalidati
        useful_candidates = []
        for j in candidate_nodes:
            if self.c.get((f, j), {}).get("val", False):
                #print(f"⚠️ Nodo {j} già ha la funzione {f}, lo salto")
                continue
            if (f, j) in invalid_pairs:
                #print(f"⚠️ Coppia (f={f}, j={j}) è negli invalid_pairs, la salto")
                continue
            useful_candidates.append(j)

        if not useful_candidates:
            print(f"❌ Nessun nodo utile rimasto per la funzione {f}")
            return None

        # Calcola vettore workload per la funzione f
        workload_f = data.workload_matrix[f]  # shape (i,)
        delay_matrix = data.node_delay_matrix  # shape (i, j)

        # Trova i nodi attivi (dove f è già assegnata)
        active_nodes = [j2 for j2 in range(len(data.nodes)) if self.c.get((f, j2), {}).get("val", False)]

        # Calcola il delay attuale totale per la funzione f (weighted sum)
        current_delay_vec = np.min(
            delay_matrix[:, active_nodes], axis=1
        ) if active_nodes else np.full(len(data.nodes), np.inf)
        current_delay_score = np.sum(workload_f * current_delay_vec)

        #print(f"📊 Delay attuale della funzione {f}: {current_delay_score:.4f}")

        best_node = None
        best_delta_score = 0.0

        for j in useful_candidates:
            #print(f"🔍 Valuto il nodo {j} per la funzione {f}")

            # Calcolo nuovo vettore di delay se assegnassi f anche su j
            delay_vec_candidate = delay_matrix[:, j]
            new_delay_vec = np.minimum(current_delay_vec, delay_vec_candidate)
            new_delay_score = np.sum(workload_f * new_delay_vec)
            delta_delay = current_delay_score - new_delay_score

            #print(f"  ⏱️  Delta delay con nodo {j}: {delta_delay:.4f}")

            if self.objective == 'min_delay':
                if delta_delay > best_delta_score + 1e-6:
                    print(f"  ✅ Nodo {j} migliora il delay rispetto a precedente best (score: {delta_delay:.4f})")
                    best_delta_score = delta_delay
                    best_node = j

            elif self.objective == 'min_delay_min_utilization':
                alpha = getattr(data, "alpha", 0.5)
                node_active = self.n.get(j, {"val": False})["val"]
                delta_utilization = (1 / len(data.nodes)) if not node_active else 0
                delta_score = (1 - alpha) * delta_delay - alpha * delta_utilization

                #print(f"  🧮 Delta utilization: {delta_utilization:.4f}")
                #print(f"  🎯 Delta score combinato: {delta_score:.4f}")

                if delta_score > best_delta_score + 1e-6:
                    #print(f"  ✅ Nodo {j} migliora il punteggio combinato rispetto a precedente best (score: {delta_score:.4f})")
                    best_delta_score = delta_score
                    best_node = j

        if best_node is not None:
            print(f"🧠 Nodo migliore per f={f} → {best_node} con miglioramento ≈ {best_delta_score:.4f}")
            return best_node

        print(f"❌ Nessun nodo porta miglioramento utile per la funzione {f}")
        return None

    def can_assign_cycle(self, cycle):
        success = False  # flag per tracciare se almeno un'assegnazione è andata a buon fine
        for f, j in cycle:
            if isinstance(f, int) and isinstance(j, int):
                j = ~j if j < 0 else j  # convert node back from negative if needed
                if not self.can_assign(f, j):
                    print(f"❌ Non posso assegnare f={f} a j={j} perché non ci sta")
                    self.invalid_pairs.add((f, j))
                    continue
                self.c[(f, j)]["val"] = True
                self.change_x_one(self.x, self.c, f)
                self.change_n_one(self.n, self.c, j)
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

    def get_objective(self, n, x):
        raise NotImplementedError("Efttc must implement getObjective()")

    def score_local(self, f, j):
        raise NotImplementedError("Efttc must implement score_local(f, j)")

    def results(self):
        x, c = output_x_and_c(self.data, self.x, self.c)
        return x, c

    def score(self):
        return self.get_objective(self.n, self.x)

class EfttcStep1CPUBase(EfttcStepBase):
    def get_constraints(self):
        ok = super().get_constraints()
        if not ok:
            print("❌ Vincolo fallito: super().get_constraints()")
        '''
        ok2 = constrain_handle_required_requests(self.data, self.x)
        if not ok2:
            print("❌ Vincolo fallito: constrain_handle_required_requests")
        '''
        ok3 = constrain_CPU_usage(self.data, self.x)
        if not ok3:
            print("❌ Vincolo fallito: constrain_CPU_usage")

        return ok  and ok3


class EfttcStep1CPUMinUtilization(EfttcStep1CPUBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objective='min_utilization'

    def get_constraints(self):
        return (super().get_constraints() and
                constrain_budget(self.data, self.n))

    def get_objective(self, n, x):
        return score_minimize_node_utilization(self.data, n)

    def score_local(self, f, j):
        # planned: assegnazioni che il solver sta valutando
        planned_util = sum(
            self.c[(f2, j)]["val"]
            for f2 in range(len(self.data.functions))
        )

        # warm start bonus: se la funzione f è già attiva su j
        warm_bonus = 1.0
        actual_alloc = getattr(self.data, "old_allocations_matrix", None)

        if isinstance(actual_alloc, np.ndarray):
            warm_bonus = 0.5 if actual_alloc[f, j] else 1.0

            # opzionale: conteggio di funzioni già attive su j
            actual_util = int(np.sum(actual_alloc[:, j]))
        else:
            actual_util = 0

        total_util = planned_util + actual_util
        cost = self.data.node_costs[j]

        return (cost / (1 + total_util)) * warm_bonus

    def results(self):
        x, c = super().results()
        n = output_n(self.data, self.n)
        self.data.prev_n = n
        self.data.prev_x = x
        self.data.prev_c = c

        return x, c

class EfttcStep1CPUMinDelay(EfttcStep1CPUBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objective='min_delay'

    def get_objective(self, n, x):
        return score_minimize_network_delay(self.data, x)

    def score_local(self, f, j):
        key = (f, j)
        if key not in self._score_cache:
            self._score_cache[key] = self.data.node_delay_matrix[:, j].dot(self.data.workload_matrix[f])

        alloc_matrix = getattr(self.data, "old_allocations_matrix", None)

        if isinstance(alloc_matrix, np.ndarray):
            if alloc_matrix[f, j] == 1:
                warm_bonus = 0.5
            else:
                warm_bonus = 1.0

        return self._score_cache[key] * warm_bonus

class EfttcStep1CPUMinDelayAndUtilization(EfttcStep1CPUMinUtilization):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.objective='min_delay_min_utilization'

    def load_data(self, data):
        data.alpha = self.alpha
        super().load_data(data)

    def get_objective(self, n, x):
        return score_minimize_node_delay_and_utilization(self.data, n, x, self.alpha)

    def score_local(self, f, j):
        # Calcolo warm bonus (accesso diretto)
        warm_bonus = 1.0
        alloc_matrix = self.data.old_allocations_matrix
        if alloc_matrix is not None and alloc_matrix[f, j] == 1:
            warm_bonus = 0.5

        # Calcolo utilization per nodo j (evita chiamate inutili a len)
        F = len(self.data.functions)
        util = sum(self.c[(f2, j)]["val"] for f2 in range(F))

        # Calcolo delay con NumPy (già efficiente)
        delay = np.dot(self.data.node_delay_matrix[:, j], self.data.workload_matrix[f])

        return (self.alpha * (self.data.node_costs[j] / (1 + util)) + (1 - self.alpha) * delay) * warm_bonus


