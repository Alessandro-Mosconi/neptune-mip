import os
import json
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from core.solvers.efttc.utils.objectives import (
    score_minimize_node_delay_and_utilization,
    score_minimize_network_delay,
    score_minimize_node_utilization
)
from core.solvers.efttc.utils.constraints_step1 import *
from core.utils.input_to_data import data_to_solver_input

# 📁 Percorso alla cartella dei JSON
base_dir = Path(__file__).resolve().parent

directory = base_dir / "simulated_test"
out_dir = base_dir / "result_simulated"
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
    # Prepare pivots
    pivot_time = (
        subset
        .pivot(index="test_case", columns="method", values="processing_time")
        .sort_index(axis=1)
        .reset_index()
    )
    pivot_score = (
        subset
        .pivot(index="test_case", columns="method", values=score_column)
        .sort_index(axis=1)
        .reset_index()
    )

    # Print to console
    print(f"\n[Processing Time Table (ms)] Method {suffix}")
    print(pivot_time.to_string(index=False))
    print(f"\n[Score Table] Method {suffix}")
    print(pivot_score.to_string(index=False))

    # Combined figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.suptitle(f"Method {suffix} Report", fontsize=16, fontweight='bold')

    # Common bar_label options
    bar_label_opts = dict(
        rotation=90,
        padding=3,
        fontsize=7,
        label_type='edge'
    )

    # 1) Processing Time – Linear
    ax = axes[0, 0]
    pivot_time.set_index("test_case").plot(
        kind="bar", ax=ax, log=False, legend=False, width=0.8
    )
    ax.set_title("Processing Time (ms)")
    ax.set_xlabel("")
    ax.set_xticks(range(len(pivot_time["test_case"])))
    ax.set_xticklabels(
        [test_case_labels.get(tc, str(tc)) for tc in pivot_time["test_case"]],
        rotation=45, ha='right'
    )
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.1f", **bar_label_opts)

    # 2) Processing Time – Log
    ax = axes[0, 1]
    pivot_time.set_index("test_case").plot(
        kind="bar", ax=ax, log=True, legend=True, width=0.8
    )
    ax.set_title("Processing Time (ms) [log]")
    ax.set_xlabel("")
    ax.set_xticks(range(len(pivot_time["test_case"])))
    ax.set_xticklabels(
        [test_case_labels.get(tc, str(tc)) for tc in pivot_time["test_case"]],
        rotation=45, ha='right'
    )
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.1f", **bar_label_opts)

    # 3) Score – Linear
    ax = axes[1, 0]
    pivot_score.set_index("test_case").plot(
        kind="bar", ax=ax, log=False, legend=False, width=0.8
    )
    ax.set_title("Score")
    ax.set_xlabel("")
    ax.set_xticks(range(len(pivot_score["test_case"])))
    ax.set_xticklabels(
        [test_case_labels.get(tc, str(tc)) for tc in pivot_score["test_case"]],
        rotation=45, ha='right'
    )
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", **bar_label_opts)

    # 4) Score – Log
    ax = axes[1, 1]
    pivot_score.set_index("test_case").plot(
        kind="bar", ax=ax, log=True, legend=True, width=0.8
    )
    ax.set_title("Score [log]")
    ax.set_xlabel("")
    ax.set_xticks(range(len(pivot_score["test_case"])))
    ax.set_xticklabels(
        [test_case_labels.get(tc, str(tc)) for tc in pivot_score["test_case"]],
        rotation=45, ha='right'
    )
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", **bar_label_opts)

    # 5+6) Tables
    add_table_to_ax(axes[2, 0], pivot_time, "Processing Time Table (ms)")
    add_table_to_ax(axes[2, 1], pivot_score, "Score Table")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # 1) Save combined figure to PDF
    pdf.savefig(fig)

    # 2) Save each of the 4 main plots as standalone PNGs
    os.makedirs(out_dir, exist_ok=True)

    def save_single(df, title, is_log, fname):
        if "score_log" in fname:
            return

        fig, ax = plt.subplots(figsize=(9, 6))

        # Plot + legenda
        df.plot(kind="bar", ax=ax, log=is_log, legend=True, width=0.8)

        # Titolo e xticks
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_xticks(range(len(df.index)))
        ax.set_xticklabels(
            [test_case_labels.get(tc, str(tc)) for tc in df.index],
            rotation=45, ha="right"
        )

        # Sposta la legenda fuori a destra
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.subplots_adjust(right=0.75)

        # ETICHETTE: se log, forzo a usare i valori raw di df
        fmt = "%.1f" if "Processing" in title else "%.2f"
        for cont, col in zip(ax.containers, df.columns):
            if is_log:
                # qui prendo la colonna originale e formatto io
                raw = df[col].tolist()
                labels = [fmt % v for v in raw]
                ax.bar_label(
                    cont,
                    labels=labels,
                    rotation=90,
                    padding=3,
                    fontsize=8,
                    label_type="edge",
                    clip_on=False
                )
            else:
                # comportamento standard
                ax.bar_label(
                    cont,
                    fmt=fmt,
                    rotation=90,
                    padding=3,
                    fontsize=8,
                    label_type="edge",
                    clip_on=False
                )

        # Allunga ylim (come prima)
        ymin, ymax = ax.get_ylim()
        scale = ((10 if "time_log" in fname else 5) if is_log else 1.2)
        ax.set_ylim(ymin, ymax * scale)

        # Layout e salvataggio
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.savefig(os.path.join(out_dir, "simulated_" + fname), dpi=300)
        plt.close(fig)

    save_single(
        pivot_time.set_index("test_case"),
        "Processing Time (ms)",
        False,
        f"{suffix}_processing_time_linear.png"
    )
    save_single(
        pivot_time.set_index("test_case"),
        "Processing Time (ms) [log]",
        True,
        f"{suffix}_processing_time_log.png"
    )
    save_single(
        pivot_score.set_index("test_case"),
        "Score",
        False,
        f"{suffix}_score_linear.png"
    )
    save_single(
        pivot_score.set_index("test_case"),
        "Score [log]",
        True,
        f"{suffix}_score_log.png"
    )

    plt.show()
    plt.close(fig)

# 🔍 Estrazione dati dai file JSON
for file in os.listdir(directory):
    if file.startswith("output_") and file.endswith(".json") and "WithEFTTC" not in file:
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
    0: "1 node, 1 function (not allocated)",
    1: "1 node, 1 function (already allocated)",
    2: "1 node, 2 functions (none allocated)",
    3: "1 node, 2 functions (one allocated)",
    4: "1 node, 2 functions (both allocated)",
    5: "20 nodes, 5 functions (none allocated)",
    6: "20 nodes, 5 functions (all allocated)",
    7: "50 nodes, 15 functions (none allocated)",
    8: "50 nodes, 5 functions (none allocated)",
    9: "25 nodes, 15 functions (none allocated)"
}


# Score columns
score_column_map = {
    "MinDelay": "score_delay",
    "MinUtilization": "score_utilization",
    "MinDelayAndUtilization": "score_combined",
}

# Unico ciclo per mostrare tempi e score
with PdfPages("simulated_report_finale.pdf") as pdf:
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
