import os
import json
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Crea PDF

from core.solvers.efttc.utils.objectives import (
    score_minimize_node_delay_and_utilization,
    score_minimize_network_delay,
    score_minimize_node_utilization
)

from core.solvers.efttc.utils.constraints_step1 import *
from core.utils.input_to_data import data_to_solver_input

# 📁 Percorso alla cartella dei JSON
directory = "/home/alessandromosconi/Desktop/fork/neptune-mip/allocation_algorithm_test"
results = []
alloc_matrix = {}

# 🔁 Ricostruzione c, n, x da JSON e solver_input

def recreate_all_vars_from_json(data):
    cpu_allocations = data.get("cpu_allocations", {})
    cpu_routing_rules = data.get("cpu_routing_rules", {})
    input_data = data.get("input", {})

    solver_input = data_to_solver_input(input_data, with_db=False, workload_coeff=input_data.get("workload_coeff", 1))
    functions = solver_input.functions
    nodes = solver_input.nodes

    fn_to_idx = {fn: i for i, fn in enumerate(functions)}
    node_to_idx = {node: j for j, node in enumerate(nodes)}

    c = {}
    for fn, node_map in cpu_allocations.items():
        if fn not in fn_to_idx:
            continue
        f = fn_to_idx[fn]
        for node, allocated in node_map.items():
            if node not in node_to_idx:
                continue
            j = node_to_idx[node]
            c[(f, j)] = {"val": bool(allocated)}
    for f in range(len(functions)):
        for j in range(len(nodes)):
            c.setdefault((f, j), {"val": False})

    n = {}
    for j in range(len(nodes)):
        n[j] = {"val": any(c[(f, j)]["val"] for f in range(len(functions)))}

    x = {}
    for i_node in cpu_routing_rules:
        if i_node not in node_to_idx:
            continue
        i = node_to_idx[i_node]
        for fn, destinations in cpu_routing_rules[i_node].items():
            if fn not in fn_to_idx:
                continue
            f = fn_to_idx[fn]
            for j_node, val in destinations.items():
                if j_node not in node_to_idx:
                    continue
                j = node_to_idx[j_node]
                x[(i, f, j)] = {"val": round(val, 6)}
    for i in range(len(nodes)):
        for f in range(len(functions)):
            for j in range(len(nodes)):
                x.setdefault((i, f, j), {"val": 0.0})

    return c, n, x, functions, nodes, solver_input


def add_table_to_ax(ax, df, title):
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax.set_title(title, fontweight="bold", fontsize=10, loc='left')

def plot_single_suffix_page(pdf, suffix, subset, test_case_labels, score_column):
    # Prepara i pivot
    pivot_time = subset.pivot(index="test_case", columns="method", values="processing_time").sort_index(axis=1).reset_index()
    pivot_score = subset.pivot(index="test_case", columns="method", values=score_column).sort_index(axis=1).reset_index()

    # 🔁 Stampa anche a schermo
    print(f"\n[📊] Tabella tempi (ms) - Metodo {suffix}")
    print(pivot_time.to_string(index=False))

    print(f"\n[📊] Tabella score - Metodo {suffix}")
    print(pivot_score.to_string(index=False))

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.suptitle(f"📄 Report Metodo {suffix}", fontsize=16, fontweight='bold')

    # Tempi - Lineare
    ax = axes[0, 0]
    pivot_time.set_index("test_case").plot(kind="bar", ax=ax, log=False, legend=False)
    ax.set_title("Tempo di elaborazione (ms)")
    ax.set_xticks(range(len(pivot_time["test_case"])))
    ax.set_xticklabels([test_case_labels.get(tc, str(tc)) for tc in pivot_time["test_case"]], rotation=45, ha='right')
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=7)

    # Tempi - Log
    ax = axes[0, 1]
    pivot_time.set_index("test_case").plot(kind="bar", ax=ax, log=True, legend=True)
    ax.set_title("Tempo di elaborazione (ms) [log]")
    ax.set_xticks(range(len(pivot_time["test_case"])))
    ax.set_xticklabels([test_case_labels.get(tc, str(tc)) for tc in pivot_time["test_case"]], rotation=45, ha='right')
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=7)

    # Score - Lineare
    ax = axes[1, 0]
    pivot_score.set_index("test_case").plot(kind="bar", ax=ax, log=False, legend=False)
    ax.set_title("Score")
    ax.set_xticks(range(len(pivot_score["test_case"])))
    ax.set_xticklabels([test_case_labels.get(tc, str(tc)) for tc in pivot_score["test_case"]], rotation=45, ha='right')
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7)

    # Score - Log
    ax = axes[1, 1]
    pivot_score.set_index("test_case").plot(kind="bar", ax=ax, log=True, legend=True)
    ax.set_title("Score [log]")
    ax.set_xticks(range(len(pivot_score["test_case"])))
    ax.set_xticklabels([test_case_labels.get(tc, str(tc)) for tc in pivot_score["test_case"]], rotation=45, ha='right')
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7)

    # Tabelle come immagini
    add_table_to_ax(axes[2, 0], pivot_time, "Tabella tempi (ms)")
    add_table_to_ax(axes[2, 1], pivot_score, "Tabella score")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)  # Salva nel PDF
    plt.show()  # Mostra sullo schermo
    plt.close(fig)  # Chiude la figura

# 🔍 Estrazione dati dai file JSON
for file in os.listdir(directory):
    if file.startswith("output_") and file.endswith(".json"):
        method = file.split("_")[1]
        test_case = int(file.split("case")[-1].split(".")[0])
        path = os.path.join(directory, file)
        with open(path, 'r') as f:
            try:
                content = f.read().strip()
                if not content:
                    print(f"[⚠️ Vuoto] Skipping file: {file}")
                    continue
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[❌ Errore parsing] {file}: {e}")
                continue

            score1 = data.get("score", {}).get("step1", None)
            score2 = data.get("score", {}).get("step2", "X" if "score" in data and "step1" in data["score"] else None)
            cpu_allocations = data.get("cpu_allocations", {})
            total_allocated = sum(len(nodes) for nodes in cpu_allocations.values())
            processing_time = data.get("processing_time", None)
            response_time = data.get("response_time", None)

            # ➕ Calcolo dei punteggi con le tre metriche
            try:
                c, n, x, functions, nodes, solver_input = recreate_all_vars_from_json(data)

                score_delay = score_minimize_network_delay(solver_input, x)
                score_util = score_minimize_node_utilization(solver_input, n)
                score_combined = score_minimize_node_delay_and_utilization(solver_input, n, x, 0.5)

                # ✅ Verifica vincoli
                constraint_flags = {
                    "constrain_c_x": constrain_c_according_to_x(solver_input, c, x),
                    "constrain_memory": constrain_memory_usage(solver_input, c, verbose=False),
                    "constrain_handle_requests": constrain_handle_required_requests(solver_input, x),
                    "constrain_CPU": constrain_CPU_usage(solver_input, x),
                    "constrain_n_c": constrain_n_according_to_c(solver_input, n, c),
                    "constrain_budget": constrain_budget(solver_input, n),
                }
            except Exception as e:
                print(f"[⚠️ Errore calcolo punteggio o vincoli] {file}: {e}")
                score_delay = score_util = score_combined = None
                constraint_flags = {
                    "constrain_c_x": False,
                    "constrain_memory": False,
                    "constrain_handle_requests": False,
                    "constrain_CPU": False,
                    "constrain_n_c": False,
                    "constrain_budget": False,
                }

            results.append({
                "method": method,
                "test_case": test_case,
                "score_step1": score1,
                "score_step2": score2,
                "score_delay": score_delay,
                "score_utilization": score_util,
                "score_combined": score_combined,
                "num_allocated": total_allocated,
                "processing_time": processing_time,
                "response_time": response_time,
                "cpu_allocations": cpu_allocations,
                **constraint_flags
            })

            # 🧱 Salva assegnazioni funzione → nodo
            for fn, node_map in cpu_allocations.items():
                for node, allocated in node_map.items():
                    if allocated:
                        key = (method, test_case)
                        alloc_matrix.setdefault(key, {}).setdefault(fn, set()).add(node)

# 📊 Costruzione DataFrame
df = pd.DataFrame(results)
df["processing_time"] *= 1000
df["response_time"] *= 1000

# Etichette test_case
test_case_labels = {
    0: "0 - 1 nodo, 1 funzione, non allocata",
    1: "1 - 1 nodo, 1 funzione, già allocata",
    2: "2 - 1 nodo, 2 funzioni, non allocate",
    3: "3 - 1 nodo, 2 funzioni, una allocata",
    4: "4 - 1 nodo, 2 funzioni, entrambe allocate",
    5: "5 - molti nodi, molte funzioni, nessuna allocata",
    6: "6 - molti nodi, molte funzioni, tutte allocate",
    7: "7 - 50 node, 20 functions",
    7: "8 - 50 node, 5 functions",
    7: "9 -  25 node, 20 functions"
}



# Score columns
score_column_map = {
    "MinDelay": "score_delay",
    "MinUtilization": "score_utilization",
    "MinDelayAndUtilization": "score_combined",
}

# Unico ciclo per mostrare tempi e score
with PdfPages("report_finale.pdf") as pdf:
    for suffix in ["MinDelay", "MinDelayAndUtilization", "MinUtilization"]:
        print(f"\n📄 Generazione pagina PDF - Metodo {suffix}")

        # Filtra dati per suffix
        subset = df[df["method"].str.endswith(suffix)].copy()
        if subset.empty:
            continue

        # Normalizza metodo
        subset["method"] = (
            subset["method"]
            .str.replace(f"_{suffix}$", "", regex=True)
            .str.replace(f"{suffix}$", "", regex=True)
            .str.rstrip("_")
        )

        score_column = score_column_map[suffix]
        if score_column not in subset.columns:
            print(f"⚠️ Colonna {score_column} mancante per {suffix}")
            continue

        # Stampa pagina intera nel PDF
        plot_single_suffix_page(pdf, suffix, subset, test_case_labels, score_column)

# 📈 Tabella vincoli per metodo
df_constraints = df[[
    "method", "test_case",
    "constrain_c_x", "constrain_memory", "constrain_handle_requests",
    "constrain_CPU", "constrain_n_c", "constrain_budget"
]].copy()

df_constraints.replace({True: "✅", False: "❌"}, inplace=True)

for suffix in ["MinDelay", "MinDelayAndUtilization", "MinUtilization"]:
    print(f"[📋] Vincoli verificati - Metodo {suffix}")
    subset = df_constraints[df_constraints["method"].str.endswith(suffix)].sort_values(by=["test_case", "method"]).copy()
    subset["method"] = (
        subset["method"]
        .str.replace(f"{suffix}$", "", regex=True)
        .str.replace(f"_{suffix}$", "", regex=True)
        .str.rstrip("_")
    )
    print(subset.to_string(index=False))
