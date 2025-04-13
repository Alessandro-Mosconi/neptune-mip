import os
import pprint
import json
import time

import requests

folder = "allocation_algorithm_test/"

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"🗑️ File rimosso: {file_path}")

for solver_type in [
    "EfttcMinDelay",
    "EfttcMinUtilization",
    "EfttcMinDelayAndUtilization",
    #  "EFTTCMultiPathMinDelay",
    #  "EFTTCMultiPathMinUtilization",
    #  "EFTTCMultiPathMinDelayAndUtilization",
    "NeptuneWithEFTTCMinDelay",
    "NeptuneWithEFTTCMinUtilization",
    "NeptuneWithEFTTCMinDelayAndUtilization",
    "NeptuneMinDelayAndUtilization",
    "NeptuneMinDelay",
    "NeptuneMinUtilization",
    #"VSVBP",
    #"Criticality",
    #"CriticalityHeuristic",
    #"MCF"
]:
    # solver_type = "NeptuneMinDelayAndUtilization"

    inputs = [
        # Simplest input
        # One node, one function
        # No function allocated
        {
            "case": 0,
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "cpu_coeff": 1,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [
                "node_a"
            ],
            "node_memories": [
                100
            ],
            "node_cores": [
                100
            ],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [
                "ns/fn_1"
            ],
            "function_memories": [
                10
            ],
            "function_max_delays": [
                100
            ],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
                "ns/fn_1": {},
            },
            "actual_gpu_allocations": {
            }
        },
        # Simplest input
        # One node, one function
        # The function was already allocated
        {
            "case": 1,
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [
                "node_a"
            ],
            "node_memories": [
                100
            ],
            "node_cores": [
                100
            ],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [
                "ns/fn_1"
            ],
            "function_memories": [
                10
            ],
            "function_max_delays": [
                100
            ],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
                "ns/fn_1": {
                    "node_a": True,
                },
            },
            "actual_gpu_allocations": {
            }
        },
        # Simplest input
        # One node, two functions
        # Both of them were not allocated
        {
            "case": 2,
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [
                "node_a"
            ],
            "node_memories": [
                100
            ],
            "node_cores": [
                100
            ],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [
                "ns/fn_1", "ns/fn_2"
            ],
            "function_memories": [
                10, 10
            ],
            "function_max_delays": [
                100, 100
            ],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
            },
            "actual_gpu_allocations": {
            }
        },
        # Simplest input
        # One node, two functions
        # Only one of them is allocated
        {
            "case": 3,
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [
                "node_a"
            ],
            "node_memories": [
                100
            ],
            "node_cores": [
                100
            ],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [
                "ns/fn_1", "ns/fn_2"
            ],
            "function_memories": [
                10, 10
            ],
            "function_max_delays": [
                100, 100
            ],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
                "ns/fn_1": {
                    "node_a": True,
                },
            },
            "actual_gpu_allocations": {
            }
        },
        # Simplest input
        # One node, two functions
        # Both of them were allocated
        {
            "case": 4,
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [
                "node_a"
            ],
            "node_memories": [
                100
            ],
            "node_cores": [
                100
            ],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [
                "ns/fn_1", "ns/fn_2"
            ],
            "function_memories": [
                10, 10
            ],
            "function_max_delays": [
                100, 100
            ],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
                "ns/fn_1": {
                    "node_a": True,
                },
                "ns/fn_2": {
                    "node_a": True,
                },
            },
            "actual_gpu_allocations": {
            }
        },
        # Many node, many functions
        # None of them were allocated
        {
            "case": 5,
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [f"node_{i}" for i in range(20)],
            "node_memories": [100 for i in range(20)],
            "node_cores": [100 for i in range(20)],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [f"ns/fn_{i}" for i in range(5)],
            "function_memories": [30 for i in range(5)],
            "function_max_delays": [100 for i in range(5)],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
            },
            "actual_gpu_allocations": {
            }
        },
        # Many node, many functions
        # All of them were allocated
        {
            "case": 6,
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [f"node_{i}" for i in range(20)],
            "node_memories": [100 for i in range(20)],
            "node_cores": [100 for i in range(20)],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [f"ns/fn_{i}" for i in range(5)],
            "function_memories": [10 for i in range(5)],
            "function_max_delays": [100 for i in range(5)],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
                "ns/fn_0": {
                    "node_1": True,
                },
                "ns/fn_1": {
                    "node_1": True,
                },
                "ns/fn_2": {
                    "node_1": True,
                },
                "ns/fn_3": {
                    "node_1": True,
                },
                "ns/fn_4": {
                    "node_1": True,
                },
            },
            "actual_gpu_allocations": {
            }
        },
        # 50 node, 20 functions
        # None of them were allocated
        {
            "case": 7,
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [f"node_{i}" for i in range(50)],
            "node_memories": [100 for i in range(50)],
            "node_cores": [100 for i in range(50)],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [f"ns/fn_{i}" for i in range(20)],
            "function_memories": [30 for i in range(20)],
            "function_max_delays": [100 for i in range(20)],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
            },
            "actual_gpu_allocations": {
            }
        },
        # 50 node, 5 functions
        # None of them were allocated
        {
            "case": 8,
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [f"node_{i}" for i in range(50)],
            "node_memories": [100 for i in range(50)],
            "node_cores": [100 for i in range(50)],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [f"ns/fn_{i}" for i in range(5)],
            "function_memories": [30 for i in range(5)],
            "function_max_delays": [100 for i in range(5)],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
            },
            "actual_gpu_allocations": {
            }
        },
        # 25 node, 20 functions
        # None of them were allocated
        {
            "case": 8,
            "solver": {
                "type": solver_type,
                "args": {"alpha": 0.0, "verbose": False}
            },
            "with_db": False,
            "community": "community-test",
            "namespace": "namespace-test",
            "node_names": [f"node_{i}" for i in range(25)],
            "node_memories": [100 for i in range(25)],
            "node_cores": [100 for i in range(25)],
            "gpu_node_names": [],
            "gpu_node_memories": [],
            "function_names": [f"ns/fn_{i}" for i in range(20)],
            "function_memories": [30 for i in range(20)],
            "function_max_delays": [100 for i in range(20)],
            "gpu_function_names": [],
            "gpu_function_memories": [],
            "actual_cpu_allocations": {
            },
            "actual_gpu_allocations": {
            }
        },
    ]

    for i, input_request in enumerate(inputs):

        start_time = time.time()
        response = requests.request(method='get', url="http://localhost:5000/", json=input_request)
        elapsed_time = time.time() - start_time

        print("=" * 40)
        print(f"Solver: {solver_type}")
        print("Status:", response.status_code)


        output_file = f"allocation_algorithm_test/output_{solver_type}_case{i}.json"

        try:
            response_json = response.json()
            response_json["response_time"] = elapsed_time
            json.dumps(input_request)
            response_json["input"] = input_request
            pprint.pprint(response_json)
            with open(output_file, 'w') as f:
                json.dump(response_json, f, indent=4)
            print(f"✓ Risposta JSON salvata in {output_file}")
        except Exception as e:
            print("❌ Errore nel parsing JSON:", e)
            print("Contenuto grezzo della risposta:")
            print(response.text)
            with open(output_file, 'w') as f:
                f.write(response.text)
            print(f"⚠️ Risposta grezza salvata in {output_file}")
