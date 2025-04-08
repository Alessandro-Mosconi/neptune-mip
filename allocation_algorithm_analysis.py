import pandas as pd
import os
import json

# Percorso alla tua cartella contenente i file JSON
directory = "/home/alessandromosconi/Desktop/fork/neptune-mip/allocation_algorithm_test"
results = []

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
            score2 = data.get("score", {}).get("step2", None)
            cpu_allocations = data.get("cpu_allocations", {})
            total_allocated = sum(len(nodes) for nodes in cpu_allocations.values())
            results.append({
                "method": method,
                "test_case": test_case,
                "score_step1": score1,
                "score_step2": score2,
                "num_allocated": total_allocated
            })

# Creazione del DataFrame
df = pd.DataFrame(results)

# Pivot delle metriche
pivot_scores_step1 = df.pivot(index="test_case", columns="method", values="score_step1").reset_index()
pivot_scores_step2 = df.pivot(index="test_case", columns="method", values="score_step2").reset_index()
pivot_allocations = df.pivot(index="test_case", columns="method", values="num_allocated").reset_index()

# Formattazione numerica per una migliore leggibilità
pd.options.display.float_format = '{:,.3f}'.format

# Stampa dei risultati
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("\n[📊] Score Step1")
print(pivot_scores_step1.to_string(index=False))

print("\n[📊] Score Step2")
print(pivot_scores_step2.to_string(index=False))

print("\n[📊] Numero di allocazioni CPU")
print(pivot_allocations.to_string(index=False))

# Raggruppamento per famiglie di metodi
for suffix in ["MinDelay", "MinDelayAndUtilization", "MinUtilization"]:
    matching_cols_step1 = [col for col in pivot_scores_step1.columns if col.endswith(suffix)]
    matching_cols_step2 = [col for col in pivot_scores_step2.columns if col.endswith(suffix)]

    if matching_cols_step1:
        print(f"\n[📊] Score Step1 - Metodo {suffix}")
        print(pivot_scores_step1[['test_case'] + matching_cols_step1].to_string(index=False))

    if matching_cols_step2:
        print(f"\n[📊] Score Step2 - Metodo {suffix}")
        print(pivot_scores_step2[['test_case'] + matching_cols_step2].to_string(index=False))

# Salvataggio opzionale in CSV
pivot_scores_step1.to_csv("score_step1.csv", index=False)
pivot_scores_step2.to_csv("score_step2.csv", index=False)
pivot_allocations.to_csv("allocazioni_cpu.csv", index=False)
