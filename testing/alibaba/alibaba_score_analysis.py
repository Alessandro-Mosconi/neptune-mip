import os
import json
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from core.solvers.efttc.utils.objectives import (
    score_minimize_node_delay_and_utilization,
    score_minimize_network_delay,
    score_minimize_node_utilization
)
from core.solvers.efttc.utils.constraints_step1 import *
from core.utils.input_to_data import data_to_solver_input

# Percorso alla cartella dei JSON
base_dir = Path(__file__).resolve().parent

directory = base_dir / "alibaba_test"
out_dir = base_dir / "result_real"
results = []
alloc_matrix = {}

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

def format_number(x):
    """Formatta i numeri con separatore delle migliaia"""
    if isinstance(x, (int, float)):
        return f"{x:,.1f}".replace(",", " ") if x % 1 else f"{int(x):,}".replace(",", " ")
    return x

def plot_single_suffix_page(pdf, suffix, subset, test_case_labels, score_column, constraints_df):
    import os
    import matplotlib.pyplot as plt
    import pandas as pd

    # --- Prepare the pivots ---
    pivot_time = (
        subset
        .pivot(index="test_case", columns="method", values="processing_time")
        .sort_index(axis=1)
    )
    pivot_time = pivot_time.applymap(format_number)

    pivot_score = (
        subset
        .pivot(index="test_case", columns="method", values=score_column)
        .sort_index(axis=1)
    )
    pivot_score = pivot_score.applymap(format_number)

    # --- Extra score table and constraints ---
    method_constraints = constraints_df[constraints_df['method'].str.endswith(suffix)].copy()
    method_constraints['method'] = method_constraints['method'] \
        .str.replace(f'{suffix}$', '', regex=True)
    method_constraints = method_constraints.drop(columns=['test_case'])

    # --- Console output ---
    print(f"\n[📊] Combined time table - Method {suffix}")
    print(pivot_time.to_string())
    print(f"\n[📊] Main score table - Method {suffix}")
    print(pivot_score.to_string())
    print(f"\n[📋] Verified constraints - Method {suffix}")
    print(method_constraints.to_string(index=False))

    # --- Create combined figure for the PDF ---
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(f"📄 Report Method {suffix}", fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])

    # 1) Processing Time – Linear
    ax1 = fig.add_subplot(gs[0, 0])
    pivot_time_numeric = pivot_time.replace('[^0-9.]', '', regex=True).astype(float)
    pivot_time_numeric.plot(kind="bar", ax=ax1, log=False, legend=False)
    ax1.set_title("Processing Time (ms)")
    ax1.set_xticks(range(len(pivot_time.index)))
    ax1.set_xticklabels(
        [test_case_labels.get(tc, str(tc)) for tc in pivot_time.index],
        rotation=45, ha='right'
    )
    ax1.set_xlabel("")
    for c in ax1.containers:
        ax1.bar_label(c, fmt="%s", rotation=0, fontsize=7, clip_on=False)

    # 2) Processing Time – Log
    ax2 = fig.add_subplot(gs[0, 1])
    pivot_time_numeric.plot(kind="bar", ax=ax2, log=True, legend=True)
    ax2.set_title("Processing Time (ms) [log]")
    ax2.set_xticks(range(len(pivot_time.index)))
    ax2.set_xticklabels(
        [test_case_labels.get(tc, str(tc)) for tc in pivot_time.index],
        rotation=45, ha='right'
    )
    ax2.set_xlabel("")
    for c in ax2.containers:
        ax2.bar_label(c, fmt="%s", rotation=0, fontsize=7, clip_on=False)

    # 3) Objective Function Score – Linear
    ax3 = fig.add_subplot(gs[1, 0])
    pivot_score_numeric = pivot_score.replace('[^0-9.]', '', regex=True).astype(float)
    pivot_score_numeric.plot(kind="bar", ax=ax3, log=False, legend=False)
    ax3.set_title("Objective Function Score")
    ax3.set_xticks(range(len(pivot_score.index)))
    ax3.set_xticklabels(
        [test_case_labels.get(tc, str(tc)) for tc in pivot_score.index],
        rotation=45, ha='right'
    )
    ax3.set_xlabel("")
    for c in ax3.containers:
        ax3.bar_label(c, fmt="%s", rotation=0, fontsize=7, clip_on=False)

    # 4) Verified Constraints table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    mc_clean = method_constraints.drop(columns=["method"]).T
    tbl4 = ax4.table(
        cellText=mc_clean.values,
        rowLabels=mc_clean.index,
        colLabels=method_constraints["method"].values,
        cellLoc='center', loc='center'
    )
    tbl4.auto_set_font_size(False)
    tbl4.set_fontsize(8)
    tbl4.scale(1.2, 1.2)
    ax4.set_title(f"Verified Constraints - Method {suffix}",
                  fontweight="bold", fontsize=10, loc='left')

    # 5) Time table (ms/s/min)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    units = ["ms", "s", "min"]
    n_rows = len(pivot_time.index) * 3
    units_list = [units[i % len(units)] for i in range(n_rows)]
    combined_time = pd.DataFrame()
    for tc in pivot_time.index:
        row_ms = pivot_time.loc[tc].rename(f"{test_case_labels.get(tc, str(tc))} (ms)")
        row_s = (pivot_time_numeric.loc[tc] / 1000).rename(f"{test_case_labels.get(tc, str(tc))} (s)")
        row_min = (pivot_time_numeric.loc[tc] / 60000).rename(f"{test_case_labels.get(tc, str(tc))} (min)")
        combined_time = pd.concat([combined_time, pd.DataFrame([row_ms, row_s, row_min])])
    table_data = combined_time.reset_index(drop=True)
    table_data.insert(0, "Unit", units_list)
    tbl5 = ax5.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center', loc='center'
    )
    tbl5.auto_set_font_size(False)
    tbl5.set_fontsize(8)
    tbl5.scale(1.2, 1.5)
    ax5.set_title("Time Table (ms/s/min)",
                  fontweight="bold", fontsize=10, loc='left')

    # 6) Objective Function Score Table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    tbl6 = ax6.table(
        cellText=pivot_score_numeric.reset_index(drop=True).values,
        colLabels=pivot_score_numeric.columns,
        cellLoc='center', loc='center'
    )
    tbl6.auto_set_font_size(False)
    tbl6.set_fontsize(8)
    tbl6.scale(1.2, 1.5)
    ax6.set_title("Objective Function Score Table",
                  fontweight="bold", fontsize=10, loc='left')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    os.makedirs(out_dir, exist_ok=True)

    def save_single(df, title, is_log, fname):
        if "score_log" in fname:
            return
        fig, ax = plt.subplots(figsize=(9, 6))
        df.plot(kind="bar", ax=ax, log=is_log, legend=True, width=0.8)
        ax.set_title(title)
        ax.set_xlabel("")
        # remove x‐axis tick labels entirely
        ax.set_xticks(range(len(df.index)))
        ax.set_xticklabels([], visible=False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.subplots_adjust(right=0.75)

        fmt = "%.1f" if "Processing" in title else "%.2f"
        for cont, col in zip(ax.containers, df.columns):
            if is_log:
                raw = df[col].tolist()
                labels = [fmt % v for v in raw]
                ax.bar_label(
                    cont,
                    labels=labels,
                    rotation=0, padding=3, fontsize=8,
                    label_type="edge", clip_on=False
                )
            else:
                ax.bar_label(
                    cont,
                    fmt=fmt,
                    rotation=0, padding=3, fontsize=8,
                    label_type="edge", clip_on=False
                )

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "real_" + fname), dpi=300)
        plt.close(fig)

    save_single(
        pivot_time_numeric,
        "Processing Time (ms)",
        False,
        f"{suffix}_processing_time_linear.png"
    )
    save_single(
        pivot_time_numeric,
        "Processing Time (ms) [log]",
        True,
        f"{suffix}_processing_time_log.png"
    )
    save_single(
        pivot_score_numeric,
        "Score",
        False,
        f"{suffix}_score_linear.png"
    )
    save_single(
        pivot_score_numeric,
        "Score [log]",
        True,
        f"{suffix}_score_log.png"
    )

    plt.show()
    plt.close()

# Estrazione dati dai file JSON
for file in os.listdir(directory):
    if file.startswith("output_") and file.endswith(".json") and "WithEFTTC" not in file:
        method = file.split("_")[1]
        test_case = int(file.split("case")[-1].split(".")[0])
        path = os.path.join(directory, file)
        with open(path, 'r') as f:
            try:
                content = f.read().strip()
                if not content:
                    print(f"[WARNING] Skipping empty file: {file}")
                    continue
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parsing error in {file}: {e}")
                continue

            score1 = data.get("score", {}).get("step1", None)
            score2 = data.get("score", {}).get("step2", "X" if "score" in data and "step1" in data["score"] else None)
            cpu_allocations = data.get("cpu_allocations", {})
            total_allocated = sum(len(nodes) for nodes in cpu_allocations.values())
            processing_time = data.get("processing_time", None)
            response_time = data.get("response_time", None)

            try:
                c, n, x, functions, nodes, solver_input = recreate_all_vars_from_json(data)

                score_delay = score_minimize_network_delay(solver_input, x)
                score_util = score_minimize_node_utilization(solver_input, n)
                score_combined = score_minimize_node_delay_and_utilization(solver_input, n, x, 0.5)

                constraint_flags = {
                    "constrain_c_x": constrain_c_according_to_x(solver_input, c, x),
                    "constrain_memory": constrain_memory_usage(solver_input, c, verbose=False),
                    "constrain_handle_requests": constrain_handle_required_requests(solver_input, x),
                    "constrain_CPU": constrain_CPU_usage(solver_input, x),
                    "constrain_n_c": constrain_n_according_to_c(solver_input, n, c),
                    "constrain_budget": constrain_budget(solver_input, n),
                }
            except Exception as e:
                print(f"[WARNING] Error calculating scores/constraints in {file}: {e}")
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

            for fn, node_map in cpu_allocations.items():
                for node, allocated in node_map.items():
                    if allocated:
                        key = (method, test_case)
                        alloc_matrix.setdefault(key, {}).setdefault(fn, set()).add(node)

# Costruzione DataFrame
df = pd.DataFrame(results)
df["processing_time"] *= 1000  # Converti in millisecondi
df["response_time"] *= 1000    # Converti in millisecondi

# Etichette test_case
test_case_labels = {
    0: "alibaba - 100 n, 25 f, 2 f allocated"
}

# Score columns
score_column_map = {
    "MinDelay": "score_delay",
    "MinUtilization": "score_utilization",
    "MinDelayAndUtilization": "score_combined",
}

# Prepara df_constraints con SI/NO invece di emoji
df_constraints = df[[
    "method", "test_case",
    "constrain_c_x", "constrain_memory", "constrain_handle_requests",
    "constrain_CPU", "constrain_n_c", "constrain_budget"
]].copy()
df_constraints.replace({True: "SI", False: "NO"}, inplace=True)

# Generazione PDF
with PdfPages("alibaba_report_finale.pdf") as pdf:
    for suffix in ["MinDelay", "MinDelayAndUtilization", "MinUtilization"]:
        print(f"\nGenerating PDF page for method: {suffix}")

        subset = df[df["method"].str.endswith(suffix)].copy()
        if subset.empty:
            print(f"No data found for method {suffix}")
            continue

        subset["method"] = (
            subset["method"]
            .str.replace(f"_{suffix}$", "", regex=True)
            .str.replace(f"{suffix}$", "", regex=True)
            .str.rstrip("_")
        )

        score_column = score_column_map[suffix]
        if score_column not in subset.columns:
            print(f"Warning: Column {score_column} missing for {suffix}")
            continue

        plot_single_suffix_page(pdf, suffix, subset, test_case_labels, score_column, df_constraints)

# Stampa vincoli nella console
for suffix in ["MinDelay", "MinDelayAndUtilization", "MinUtilization"]:
    print(f"\n[CONSTRAINTS] Method: {suffix}")
    subset = df_constraints[df_constraints["method"].str.endswith(suffix)].sort_values(
        by=["test_case", "method"]).copy()
    subset["method"] = (
        subset["method"]
        .str.replace(f"{suffix}$", "", regex=True)
        .str.replace(f"_{suffix}$", "", regex=True)
        .str.rstrip("_")
    )
    print(subset.to_string(index=False))