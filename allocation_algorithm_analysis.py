import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict  # <--- AGGIUNGI QUESTA RIGA
import seaborn as sns

# Percorso alla tua cartella contenente i file JSON
directory = "/home/alessandromosconi/Desktop/fork/neptune-mip/allocation_algorithm_test"
results = []
alloc_matrix = {}

# Crea tabella: per ogni metodo/test_case, numero di funzioni su ciascun nodo
def generate_function_node_count_table(method, test_case):
    alloc = alloc_matrix.get((method, test_case), {})
    fn_to_nodes = defaultdict(set)
    for fn, node in alloc.items():
        fn_to_nodes[fn].add(node)
    fn_counts = {fn: len(nodes) for fn, nodes in fn_to_nodes.items()}
    return pd.DataFrame([fn_counts]).rename(index={0: f"{method}_case{test_case}"})


# Estrazione dei dati
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
            results.append({
                "method": method,
                "test_case": test_case,
                "score_step1": score1,
                "score_step2": score2,
                "num_allocated": total_allocated,
                "processing_time": processing_time,
                "response_time": response_time,
                "cpu_allocations": cpu_allocations
            })

            # Salva assegnazioni funzione → nodo
            for fn, node_map in cpu_allocations.items():
                for node, allocated in node_map.items():
                    if allocated:
                        key = (method, test_case)
                        alloc_matrix.setdefault(key, {}).setdefault(fn, set()).add(node)

# Creazione del DataFrame
df = pd.DataFrame(results)

# Converti in millisecondi
df["processing_time"] = df["processing_time"] * 1000
df["response_time"] = df["response_time"] * 1000

# Pivot delle metriche
pivot_scores_step1 = df.pivot(index="test_case", columns="method", values="score_step1").reset_index()
pivot_scores_step2 = df.pivot(index="test_case", columns="method", values="score_step2").reset_index()
pivot_allocations = df.pivot(index="test_case", columns="method", values="num_allocated").reset_index()
pivot_proc_time = df.pivot(index="test_case", columns="method", values="processing_time").reset_index()
pivot_resp_time = df.pivot(index="test_case", columns="method", values="response_time").reset_index()

# Formattazione numerica
pd.options.display.float_format = '{:,.3f}'.format

# Ordinamento colonne
def reorder_columns(df):
    ordered_cols = ['test_case']
    if 'test_case' in df.columns:
        other_cols = [col for col in df.columns if col != 'test_case']
        neptune = sorted([col for col in other_cols if col.startswith('Neptune') and not col.startswith('NeptuneWith')])
        neptune_with = sorted([col for col in other_cols if col.startswith('NeptuneWith')])
        efttc = sorted([col for col in other_cols if col.startswith('Efttc') and not col.startswith('EFTTCMultiPath')])
        efttc_multi = sorted([col for col in other_cols if col.startswith('EFTTCMultiPath')])
        ordered_cols += neptune + neptune_with + efttc + efttc_multi
        return df[ordered_cols]
    return df


pivot_scores_step1 = reorder_columns(pivot_scores_step1)
pivot_scores_step2 = reorder_columns(pivot_scores_step2)
pivot_allocations = reorder_columns(pivot_allocations)
pivot_proc_time = reorder_columns(pivot_proc_time)
pivot_resp_time = reorder_columns(pivot_resp_time)


# Visualizzazione tabelle
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("\n[📊] Score Step1")
print(pivot_scores_step1.to_string(index=False))

print("\n[📊] Score Step2")
print(pivot_scores_step2.fillna("X").to_string(index=False))

print("\n[📊] Numero di allocazioni CPU")
print(pivot_allocations.to_string(index=False))

print("\n[⏱️] Tempo di elaborazione (processing_time)")
print(pivot_proc_time.to_string(index=False))

print("\n[🌐] Tempo totale di risposta (response_time)")
print(pivot_resp_time.to_string(index=False))

import pandas as pd

def extract_allocated_nodes(cpu_allocations):
    data = []
    for fn, node_dict in cpu_allocations.items():
        allocated_nodes = sorted(int(node.split("_")[1]) for node, is_alloc in node_dict.items() if is_alloc)
        node_str = ",".join(map(str, allocated_nodes)) if allocated_nodes else "-"
        data.append({
            "Function": fn,
            "Allocated Nodes": node_str,
            "Count": len(allocated_nodes)
        })

    return pd.DataFrame(data)

import pandas as pd

def build_allocation_table_by_function(alloc_matrix, suffix):
    rows = []
    all_functions = set()

    for (method, test_case), fn_allocs in alloc_matrix.items():
        if method.endswith(suffix):
            row_key = f"{method} case_{test_case}"
            row = {"id": row_key}
            row["test_case_num"] = test_case
            row["method"] = method
            for fn_full, nodes in fn_allocs.items():
                fn = fn_full.split("/")[-1]
                all_functions.add(fn)
                node_list = sorted(nodes)
                row[fn] = ",".join(node_list) if node_list else "-"
            rows.append(row)

    df = pd.DataFrame(rows)

    # Aggiungi colonne mancanti prima di settare l'indice
    for fn in sorted(all_functions):
        if fn not in df.columns:
            df[fn] = "-"

    df = df.set_index("id")

    ordered_cols = [f"fn_{i}" for i in range(100) if f"fn_{i}" in df.columns]
    df = df[["test_case_num", "method"] + ordered_cols]
    df = df.fillna("-")

    # Ordina prima per numero di test case, poi per nome metodo
    df = df.sort_values(["test_case_num", "method"])
    df = df.drop(columns=["test_case_num", "method"])

    return df

# Raggruppamento per famiglie di metodi
for suffix in ["MinDelay", "MinDelayAndUtilization", "MinUtilization"]:
    matching_cols_step1 = [col for col in pivot_scores_step1.columns if col.endswith(suffix)]
    matching_cols_step2 = [col for col in pivot_scores_step2.columns if col.endswith(suffix)]
    matching_cols_alloc = [col for col in pivot_allocations.columns if col.endswith(suffix)]
    matching_cols_proc = [col for col in pivot_proc_time.columns if col.endswith(suffix)]
    matching_cols_resp = [col for col in pivot_resp_time.columns if col.endswith(suffix)]

    if matching_cols_step1:
        print(f"\n[📊] Score Step1 - Metodo {suffix}")
        print(pivot_scores_step1[['test_case'] + matching_cols_step1].to_string(index=False))

    if matching_cols_step2:
        print(f"\n[📊] Score Step2 - Metodo {suffix}")
        print(pivot_scores_step2[['test_case'] + matching_cols_step2].fillna("X").to_string(index=False))

    if matching_cols_alloc:
        print(f"\n[🖥️] Allocazioni CPU - Metodo {suffix}")
        print(pivot_allocations[['test_case'] + matching_cols_alloc].to_string(index=False))

    if matching_cols_proc:
        print(f"\n[⏱️] Tempo di elaborazione - Metodo {suffix}")
        print(pivot_proc_time[['test_case'] + matching_cols_proc].to_string(index=False))

    if matching_cols_resp:
        print(f"\n[🌐] Tempo di risposta - Metodo {suffix}")
        print(pivot_resp_time[['test_case'] + matching_cols_resp].to_string(index=False))

    if any(method.endswith(suffix) for (method, test_case) in alloc_matrix):
        df_fn_alloc_table = build_allocation_table_by_function(alloc_matrix, suffix)

        print(f"\n[📋] Tabella funzione → nodi per il gruppo {suffix}")
        print(df_fn_alloc_table.to_string())


# Salvataggio opzionale in CSV
pivot_scores_step1.to_csv("score_step1.csv", index=False)
pivot_scores_step2.to_csv("score_step2.csv", index=False)
pivot_allocations.to_csv("allocazioni_cpu.csv", index=False)
pivot_proc_time.to_csv("processing_time.csv", index=False)
pivot_resp_time.to_csv("response_time.csv", index=False)

# Mappa per test_case -> descrizione
test_case_labels = {
    0: "0 - 1 nodo, 1 funzione, non allocata",
    1: "1 - 1 nodo, 1 funzione, già allocata",
    2: "2 - 1 nodo, 2 funzioni, non allocate",
    3: "3 - 1 nodo, 2 funzioni, una allocata",
    4: "4 - 1 nodo, 2 funzioni, entrambe allocate",
    5: "5 - molti nodi, molte funzioni, nessuna allocata",
    6: "6 - molti nodi, molte funzioni, tutte allocate",
    7: "7 - molti molti nodi, molte molte funzioni, nessuna allocata"
}

# Colori coerenti per gruppi
color_map = {
    "Neptune": "#f90d1b",
    "NeptuneWithEFTTC": "#fde005",
    "Efttc": "#9d00fe",
    "EFTTCMultiPath": "#5d00fe"
}

# Funzione per grafici
def plot_grouped_bars(df, suffix, ylabel, title, log_scale=False):
    df = df.copy()
    df["test_case"] = df["test_case"].map(test_case_labels)
    df.set_index("test_case", inplace=True)
    df = df[[col for col in df.columns if col.endswith(suffix)]]

    colors = []
    for col in df.columns:
        if col.startswith("NeptuneWithEFTTC"):
            colors.append(color_map["NeptuneWithEFTTC"])
        elif col.startswith("EFTTCMultiPath"):
            colors.append(color_map["EFTTCMultiPath"])
        elif col.startswith("Efttc"):
            colors.append(color_map["Efttc"])
        else:
            colors.append(color_map["Neptune"])

    ax = df.plot(kind='bar', figsize=(14, 6), width=0.7, color=colors, log=log_scale)
    ax.set_title(f"{title} - {suffix}", fontsize=14)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Test case")
    ax.tick_params(axis='x', rotation=30)
    ax.legend(title="Metodo", bbox_to_anchor=(1.05, 1), loc='upper left')

    for container in ax.containers:
        labels = [f'{v.get_height():.1f} ms' if not np.isnan(v.get_height()) else '' for v in container]
        ax.bar_label(container, labels=labels, label_type='edge', fontsize=8)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_allocation_heatmap(alloc_matrix, suffix):
    df = build_allocation_table_by_function(alloc_matrix, suffix)
    df_count = df.applymap(lambda x: 0 if x == "-" else len(x.split(",")))

    plt.figure(figsize=(14, max(6, len(df_count) * 0.4)))
    sns.heatmap(df_count, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"label": "Numero di nodi"})
    plt.title(f"Heatmap: Numero di nodi per funzione - {suffix}", fontsize=14)
    plt.xlabel("Funzione")
    plt.ylabel("Metodo + Test Case")
    plt.tight_layout()
    plt.show()


# Plot per ciascun gruppo e metrica
for suffix in ["MinDelay", "MinDelayAndUtilization", "MinUtilization"]:
    #plot_allocation_heatmap(alloc_matrix, suffix)

    #plot_grouped_bars(pivot_proc_time, suffix, "Tempo di elaborazione (ms)", "Tempo di elaborazione")
    plot_grouped_bars(pivot_proc_time, suffix, "Tempo di elaborazione (ms)", "Tempo di elaborazione (scala log)", log_scale=True)
    #plot_grouped_bars(pivot_resp_time, suffix, "Tempo di risposta (ms)", "Tempo di risposta")
    #plot_grouped_bars(pivot_resp_time, suffix, "Tempo di risposta (ms)", "Tempo di risposta (scala log)", log_scale=True)

