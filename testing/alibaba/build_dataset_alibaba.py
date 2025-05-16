from pathlib import Path

import pandas as pd
import json
import time

# 📁 Percorsi file
DATASET_PATH = "/mnt/linux-storage/ALIBABA_dataset"
MACHINE_META_FILE = f"{DATASET_PATH}/machine_meta.csv"
TASK_FILE = f"{DATASET_PATH}/batch_task.csv"
INSTANCE_FILE = f"{DATASET_PATH}/batch_instance.csv"

base_dir = Path(__file__).resolve().parent

OUTPUT_JSON_PATH = base_dir / "alibaba_test_case.json"

# ⚙️ Parametri
N_NODES = 100
N_FUNCTIONS = 25
CHUNK_SIZE = 500_000
CHUNKS_TO_SCAN = 10  # Quanti chunk leggere da batch_instance

print("🔹 Inizio generazione test case Alibaba")

# 1. Carica i nodi
print("📥 Caricamento machine_meta.csv...")
machines = pd.read_csv(MACHINE_META_FILE, header=None)
machines.columns = [
    "machine_id", "time_stamp", "failure_domain_1", "failure_domain_2",
    "cpu_num", "mem_size", "status"
]
# 🔧 Seleziona i primi N_NODES machine_id distinti
machines = machines.drop_duplicates(subset="machine_id").head(N_NODES)
node_names = machines["machine_id"].tolist()
node_cores = machines["cpu_num"].astype(int).tolist()
node_memories = machines["mem_size"].astype(int).tolist()
print(f"✅ {len(node_names)} nodi unici caricati")

# 2. Trova le funzioni dai primi chunk di batch_instance
print(f"📦 Scansione dei primi {CHUNKS_TO_SCAN} chunk di batch_instance.csv...")
found_functions = set()
instances = []

for chunk_idx, chunk in enumerate(pd.read_csv(INSTANCE_FILE, chunksize=CHUNK_SIZE, header=None)):
    chunk.columns = [
        "instance_name", "task_name", "job_name", "task_type", "status",
        "start_time", "end_time", "machine_id", "seq_no", "total_seq_no",
        "cpu_avg", "cpu_max", "mem_avg", "mem_max"
    ]
    for _, row in chunk.iterrows():
        fname = (row["job_name"], row["task_name"])
        instances.append((fname, row["machine_id"]))
        found_functions.add(fname)
        if len(found_functions) >= N_FUNCTIONS:
            break
    if len(found_functions) >= N_FUNCTIONS or chunk_idx + 1 >= CHUNKS_TO_SCAN:
        break

print(f"✅ Funzioni trovate: {len(found_functions)}")

# 3. Costruisci lista finalizzata di funzioni
selected_functions = list(found_functions)[:N_FUNCTIONS]
selected_function_names = [f"{job}/{task}" for (job, task) in selected_functions]
print(f"🎯 Funzioni selezionate: {selected_function_names}")

# 4. Recupera i dati da batch_task.csv
print("📥 Caricamento batch_task.csv...")
tasks = pd.read_csv(TASK_FILE, header=None)
tasks.columns = [
    "task_name", "instance_num", "job_name", "task_type", "status",
    "start_time", "end_time", "plan_cpu", "plan_mem"
]

function_memories = []
function_max_delays = []
function_names = []

for (job, task) in selected_functions:
    match = tasks[(tasks["job_name"] == job) & (tasks["task_name"] == task)]
    if not match.empty:
        mem = match.iloc[0]["plan_mem"]
    else:
        mem = 10  # fallback se non trovato
    fname = f"{job}/{task}"
    function_names.append(fname)
    function_memories.append(mem)
    function_max_delays.append(100)

# 5. Ricostruisci le allocazioni CPU solo sui nodi caricati
actual_cpu_allocations = {}

for (fname, mid) in instances:
    if fname in selected_functions:
        if mid not in node_names:
            continue  # ignora nodi non caricati
        full_name = f"{fname[0]}/{fname[1]}"
        if full_name not in actual_cpu_allocations:
            actual_cpu_allocations[full_name] = {}
        actual_cpu_allocations[full_name][mid] = True

# 6. Crea il test case JSON
print("🧱 Costruzione struttura JSON...")
test_case = {
    "case": 200,
    "solver": {
        "type": "YourSolverType",  # ← Cambialo se serve
        "args": {"alpha": 0.0, "verbose": False}
    },
    "with_db": False,
    "community": "community-trace",
    "namespace": "namespace-trace",
    "node_names": node_names,
    "node_memories": node_memories,
    "node_cores": node_cores,
    "gpu_node_names": [],
    "gpu_node_memories": [],
    "function_names": function_names,
    "function_memories": function_memories,
    "function_max_delays": function_max_delays,
    "gpu_function_names": [],
    "gpu_function_memories": [],
    "actual_cpu_allocations": actual_cpu_allocations,
    "actual_gpu_allocations": {}
}

# 7. Mostra le allocazioni trovate
print("\n🖨️ Allocazioni CPU trovate:")
for fname in sorted(actual_cpu_allocations):
    nodi = list(actual_cpu_allocations[fname].keys())
    print(f" - {fname} → {nodi}")

# 8. Salva il JSON
print(f"\n💾 Scrittura su file JSON in {OUTPUT_JSON_PATH}...")
with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(test_case, f, indent=2)

print(f"✅ Test case salvato in {OUTPUT_JSON_PATH}")
