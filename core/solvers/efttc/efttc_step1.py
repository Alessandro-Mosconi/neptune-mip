import copy
from datetime import datetime

from .utils import *
from ..solver import Solver


class EfttcStepBase(Solver):
    def __init__(self, **kwargs):
        self.invalid_pairs = set()
        super().__init__(**kwargs)
        self.x, self.c, self.n = {}, {}, {}
        self.assigned_functions = set()
        self.assigned_nodes = set()
        self._score_cache = {}
        self.objective='min_delay_min_utilization'

    def init_vars(self):
        init_x(self.data, self.x)
        init_c(self.data, self.c)
        init_n(self.data, self.n)

    def init_constraints(self):
        pass  # I vincoli vengono verificati tramite get_constraints()

    def get_constraints(self):
        return True # constrain_c_according_to_x(self.data, self.c, self.x) and constrain_memory_usage(self.data, self.c)

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

    def should_remove_function(self, f, remaining_nodes):
        current_score = self.score()
        print(f"\n🔍 Verifica se la funzione {f} deve essere rimossa (score attuale: {current_score})")

        for j in remaining_nodes:
            if (f, j) in self.invalid_pairs or self.c[(f, j)]["val"]:
                print(f"⚠️ Funzione {f} è già instanziata su nodo {j} oppure è una coppia invalida.")
                continue

            if not self.can_assign(f, j):
                print(f"❌ Nodo {j} non ha abbastanza memoria per la funzione {f}")
                continue

            snapshot = self.snapshot_vars()

            # Prova ad aggiungere f su j
            self.c[(f, j)]["val"] = True
            self.update_n_from_c()

            # Redistribuisci le richieste: per ogni nodo source i, manda verso il j* con delay minimo
            for i in range(len(self.data.nodes)):
                if self.data.workload_matrix[f, i] > 0:
                    candidates = [
                        jj for jj in range(len(self.data.nodes))
                        if self.c[(f, jj)]["val"]
                    ]
                    if candidates:
                        best_j = min(candidates, key=lambda jj: self.data.node_delay_matrix[i, jj])
                        for jj in range(len(self.data.nodes)):
                            self.x[(i, f, jj)]["val"] = 1.0 if jj == best_j else 0.0

            new_score = self.score()
            print(f"✅ Tentativo su nodo {j}: nuovo score {new_score}")

            self.restore_vars(snapshot)

            if new_score < current_score - 1e-6:  # miglioramento vero
                print(f"📈 Nuova assegnazione su nodo {j} migliora lo score di funzione {f}")
                return False

        print(f"🗑️ Nessun miglioramento possibile per f={f}, la funzione può essere rimossa")
        return True

    def solve(self):
        self.init_vars()

        #print("DELAY MATRIX")
        #print(self.data.node_delay_matrix)

        #print("WORKLOAD MATRIX")
        #print(self.data.workload_matrix)

        #print("ACTUAL ALLOCATION")
        #print(getattr(self.data, "old_allocations_matrix", {}))

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
            hasSuccess = self.assign_cycle(cycle)
            self.update_n_from_c()

            if not hasSuccess:
                print("⛔️ Alcune assegnazioni non sono valide")
                self.restore_vars(snapshot)
                tried_cycles.add(cycle_key)
                continue

            # Verifica i vincoli globali
            if self.get_constraints():

                self.update_x_from_c()

                if 'min_delay' in self.objective:

                    time = datetime.now()
                    for f, _ in cycle:
                        if(self.find_best_node_by_delay_improvement(f, remaining_nodes, self.data) is not None):
                            print("✅ Funzione può essere migliorata")
                        else:
                            print("❌ Funzione non può essere migliorata")
                            remaining_functions.remove(f)
                    #self.evaluate_cycle_removals(cycle, remaining_functions, remaining_nodes)

                    elapsedTime = datetime.now() - time
                    print(f"⏱️ Tempo per capire se rimuovere la funzione: {elapsedTime.total_seconds()} secondi")
                else:
                    for f, _ in cycle:
                        remaining_functions.discard(f)

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
                        self.snapshot_vars()


            else:
                print("⚠️ Ciclo trovato ma viola i vincoli globali. Scartato.")
                self.restore_vars(snapshot)
                tried_cycles.add(cycle_key)
                for f, j in cycle:
                    self.invalid_pairs.add((f, j))

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

    def update_x_from_c(self):
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                # Trova tutti i nodi j dove la funzione f è attiva
                active_nodes = [
                    j for j in range(len(self.data.nodes))
                    if self.c[(f, j)]["val"]
                ]

                if not active_nodes:
                    # Nessuna istanza attiva per f, ignoro questa richiesta
                    continue

                # Calcola il delay minimo per (i, f)
                delays = {j: self.data.node_delay_matrix[i][j] for j in active_nodes}
                min_delay = min(delays.values())

                # Prende tutti i nodi j con delay minimo
                best_nodes = [j for j, d in delays.items() if abs(d - min_delay) < 1e-6]

                # Distribuisci in modo uniforme tra i best nodes
                val = 1.0 / len(best_nodes)
                for j in active_nodes:
                    self.x[(i, f, j)]["val"] = val if j in best_nodes else 0.0

    def change_n_from_c(self, n, c):
        for j in range(len(self.data.nodes)):
            n[j]["val"] = any(
                c[(f, j)]["val"]
                for f in range(len(self.data.functions))
            )

    def change_x_from_c(self, x, c):
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                # Trova tutti i nodi j dove la funzione f è attiva
                active_nodes = [
                    j for j in range(len(self.data.nodes))
                    if c[(f, j)]["val"]
                ]

                if not active_nodes:
                    # Nessuna istanza attiva per f, ignoro questa richiesta
                    continue

                # Calcola il delay minimo per (i, f)
                delays = {j: self.data.node_delay_matrix[i][j] for j in active_nodes}
                min_delay = min(delays.values())

                # Prende tutti i nodi j con delay minimo
                best_nodes = [j for j, d in delays.items() if abs(d - min_delay) < 1e-6]

                # Distribuisci in modo uniforme tra i best nodes
                val = 1.0 / len(best_nodes)
                for j in active_nodes:
                    x[(i, f, j)]["val"] = val if j in best_nodes else 0.0

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

    def find_best_node_by_delay_improvement(self, f, candidate_nodes, data):
        if not candidate_nodes:
            return None

        # Converti in array NumPy (serve per indexing e maschere)
        candidate_nodes = np.array(list(candidate_nodes), dtype=int)

        # --- STEP 1: Filtra i nodi che NON hanno già f allocata usando mask boolean ---
        # Costruisci un array booleano dove mask[j] = True se self.c[(f, j)]["val"] è False
        is_unallocated = np.array([
            not self.c.get((f, j), {}).get("val", False) for j in candidate_nodes
        ])

        # Applica la maschera per ottenere solo i nodi validi
        candidate_nodes = candidate_nodes[is_unallocated]

        if candidate_nodes.size == 0:
            return None

        # --- STEP 2: Check workload della funzione ---
        workload_f = data.workload_matrix[f]
        if not np.any(workload_f):
            return None

        # --- STEP 3: Calcolo vettoriale del contributo al delay ---
        delay_matrix = data.node_delay_matrix
        candidate_delays = delay_matrix[:, candidate_nodes]

        scores = workload_f @ candidate_delays
        best_idx = np.argmin(scores)
        best_score = scores[best_idx]
        best_node = candidate_nodes[best_idx]

        if np.isclose(best_score, 0.0):
            return None

        return best_node

    def evaluate_cycle_removals(self, cycle, remaining_functions, remaining_nodes):
        current_score = self.score_param(self.n, self.x)
        print(f"\n🔍 Valutazione collettiva per rimozione funzioni (score attuale: {current_score})")

        candidates_to_keep = set()

        for f, _ in cycle:
            # Snapshot una volta sola per f
            #print time
            time = datetime.now()
            base_x = {k: {"name": v["name"], "val": v["val"]} for k, v in self.x.items()}
            base_c = {k: {"name": v["name"], "val": v["val"]} for k, v in self.c.items()}
            base_n = (
                {k: {"name": v["name"], "val": v["val"]} for k, v in self.n.items()}
                if hasattr(self, "n")
                else {k: {"name": f"n[{k}]", "val": False} for k in range(len(self.data.nodes))}
            )
            elapsedTime = datetime.now() - time
            print(f"⏱️ Tempo di snapshot: {elapsedTime.total_seconds()} secondi")



            for j in remaining_nodes:
                if (f, j) in self.invalid_pairs or self.c[(f, j)]["val"]:
                    continue
                if not self.can_assign(f, j):
                    continue

                # Copie leggere da base (solo quello che serve modificare)
                time = datetime.now()
                c = base_c.copy()
                elapsedTime = datetime.now() - time
                print(f"⏱️ Tempo per copiare c: {elapsedTime.total_seconds()} secondi")

                c[(f, j)] = {"name": c[(f, j)]["name"], "val": True}

                time = datetime.now()
                n = base_n.copy()
                self.change_n_one(n, c, j)
                elapsedTime = datetime.now() - time
                print(f"⏱️ Tempo per copiare e aggiornare n: {elapsedTime.total_seconds()} secondi")

                time = datetime.now()
                x = base_x.copy()
                self.change_x_one(x, c, f)
                elapsedTime = datetime.now() - time
                print(f"⏱️ Tempo per copiare e aggiornare x: {elapsedTime.total_seconds()} secondi")

                time = datetime.now()
                new_score = self.score_param(n, x)
                elapsedTime = datetime.now() - time
                print(f"⏱️ Tempo per ccalcolare lo score: {elapsedTime.total_seconds()} secondi")
                print(f"✅ Tentativo: funzione {f} su nodo {j} → score: {new_score}")

                if new_score < current_score - 1e-6:
                    print(f"📈 Miglioramento trovato per funzione {f}")
                    candidates_to_keep.add(f)
                    break

        for f, _ in cycle:
            if f not in candidates_to_keep:
                print(f"🗑️ Funzione {f} può essere rimossa (nessun miglioramento)")
                remaining_functions.discard(f)
            else:
                print(f"🔒 Funzione {f} mantenuta (miglioramento possibile)")
    '''

    def evaluate_cycle_removals(self, cycle, remaining_functions, remaining_nodes):
        current_score = self.score_param(self.n, self.x)
        print(f"\n🔍 [DELTA] Valutazione collettiva per rimozione funzioni (score attuale: {current_score})")

        candidates_to_keep = set()

        for f, _ in cycle:
            # Snapshot una volta per f
            base_x = {k: {"name": v["name"], "val": v["val"]} for k, v in self.x.items()}
            base_c = {k: {"name": v["name"], "val": v["val"]} for k, v in self.c.items()}
            base_n = {k: {"name": v["name"], "val": v["val"]} for k, v in self.n.items()}

            for j in remaining_nodes:
                if (f, j) in self.invalid_pairs or base_c[(f, j)]["val"]:
                    continue
                if not self.can_assign(f, j):
                    continue

                # Copie leggere e aggiornamenti locali
                c = base_c.copy()
                c[(f, j)] = {"name": c[(f, j)]["name"], "val": True}

                n = base_n.copy()
                self.change_n_one(n, c, j)

                x = base_x.copy()
                self.change_x_one(x, c, f)

                new_score = self.score_param(n, x)
                if new_score < current_score - 1e-6:
                    print(f"📈 [DELTA] Funzione {f} migliorabile su nodo {j} → score: {new_score}")
                    candidates_to_keep.add(f)
                    break

        for f, _ in cycle:
            if f not in candidates_to_keep:
                print(f"🗑️ [DELTA] Funzione {f} può essere rimossa (nessun miglioramento)")
                remaining_functions.discard(f)
    '''

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
        #print("Step 1 - x:", x, sep='\n')
        #print("Step 1 - c:", c, sep='\n')
        return x, c

    def score(self):
        return self.get_objective(self.n, self.x)

    def score_param(self, n, x):
        if(n is None):
            n = {k: {"name": f"n[{k}]", "val": False} for k in range(len(self.data.nodes))}
        return self.get_objective(n, x)


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
        self.objective='min_utilization'

    def init_vars(self):
        super().init_vars()
        init_n(self.data, self.n)

    def get_constraints(self):
        return (super().get_constraints() and
                # constrain_n_according_to_c(self.data, self.n, self.c) and
                constrain_budget(self.data, self.n))

    def get_objective(self, n, x):
        return score_minimize_node_utilization(self.data, n)
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
        #print("Step 1 - n:", n, sep='\n')
        return x, c


class EfttcStep1CPUMinDelay(EfttcStep1CPUBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objective='min_delay'

    def get_objective(self, n, x):
        return score_minimize_network_delay(self.data, x)
    '''
    def score_local(self, f, j):
        return self.data.node_delay_matrix[:, j].dot(self.data.workload_matrix[f])
    '''
    def score_local(self, f, j):
        key = (f, j)
        if key not in self._score_cache:
            self._score_cache[key] = self.data.node_delay_matrix[:, j].dot(self.data.workload_matrix[f])

        alloc_matrix = getattr(self.data, "old_allocations_matrix", None)

        if isinstance(alloc_matrix, np.ndarray):
            if alloc_matrix[f, j] == 1:
                warm_bonus = 0.01
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
        alloc_matrix = getattr(self.data, "old_allocations_matrix", None)

        if isinstance(alloc_matrix, np.ndarray):
            if alloc_matrix[f, j] == 1:
                warm_bonus = 0.5
            else:
                warm_bonus = 1.0

        util = sum(self.c[(f2, j)]["val"] for f2 in range(len(self.data.functions)))
        delay = self.data.node_delay_matrix[:, j].dot(self.data.workload_matrix[f])
        return (self.alpha * util + (1 - self.alpha) * delay ) * warm_bonus

