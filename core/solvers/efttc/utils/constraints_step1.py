M = 10**6
epsilon = 10**-6

# If a function `f` is deployed on node i then c[f,i] is True
def constrain_c_according_to_x(data, c, x, M=1e6, epsilon=1e-6):
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            sum_x = sum(x[(i, f, j)]["val"] for i in range(len(data.nodes)))

            # Primo vincolo: sum_x <= c[f, j] * M
            if sum_x > (M if c[(f, j)]["val"] else 0):
                return False

            # Secondo vincolo: sum_x + epsilon >= c[f, j]
            if sum_x + epsilon < (1 if c[(f, j)]["val"] else 0):
                return False

    return True


# The sum of the memory of functions deployed on a node is less than its capacity
def constrain_memory_usage(data, c, verbose=False):
    for j in range(len(data.nodes)):
        total_memory = sum(
            (data.function_memory_matrix[f] if c[(f, j)]["val"] else 0)
            for f in range(len(data.functions))
        )
        if verbose:
            print(f"Node {j}: used {total_memory} / available {data.node_memory_matrix[j]}")
        if total_memory > data.node_memory_matrix[j]:
            return False
    return True


# All requests in a node are rerouted somewhere else
def constrain_handle_all_requests(data, x, eq=True):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            total = sum(x[(i, f, j)]["val"] for j in range(len(data.nodes)))
            if eq and not abs(total - 1) < 1e-6:
                return False
            if not eq and total > 1 + 1e-6:
                return False
    return True



# All requests except the ones managed by GPUs in a node are rerouted somewhere else
def constrain_handle_only_remaining_requests(data, x):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            total = sum(x[(i, f, j)]["val"] for j in range(len(data.nodes)))
            expected = 1 - data.prev_x[i][f].sum()
            if abs(total - expected) > 1e-6:
                return False
    return True



def constrain_handle_required_requests(data, x):
    if data.prev_x.shape == (0,):
        return constrain_handle_all_requests(data, x)
    else:
        return constrain_handle_only_remaining_requests(data, x)



# Do not overload nodes' CPUs
def constrain_CPU_usage(data, x):
    for j in range(len(data.nodes)):
        total = 0
        for f in range(len(data.functions)):
            for i in range(len(data.nodes)):
                val = x[(i, f, j)]["val"]
                total += val * data.workload_matrix[f, i] * data.core_per_req_matrix[f, j]
        if total > data.node_cores_matrix[j] + 1e-6:
            return False
    return True



# If a node i contains one or more functions then n[i] is True
def constrain_n_according_to_c(data, n, c, M=1e6, epsilon=1e-6):
    for i in range(len(data.nodes)):
        sum_c = sum(1 if c[(f, i)]["val"] else 0 for f in range(len(data.functions)))
        n_val = 1 if n[i]["val"] else 0
        if sum_c > n_val * M:
            return False
        if sum_c + epsilon < n_val:
            return False
    return True



# The sum of the memory of functions deployed on a gpu device is less than its capacity
def constrain_GPU_memory_usage(data, c):
    for j in range(len(data.nodes)):
        total = sum(
            data.gpu_function_memory_matrix[f] if c[(f, j)]["val"] else 0
            for f in range(len(data.functions))
        )
        if total > data.gpu_node_memory_matrix[j]:
            return False
    return True



# Do not overload nodes' GPUs
def constrain_GPU_usage(data, x):
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            total = sum(
                x[(i, f, j)]["val"] * data.workload_matrix[f, i] * data.response_time_matrix[f, j]
                for i in range(len(data.nodes))
            )
            if total > 1000 + 1e-6:
                return False
    return True



def constrain_budget(data, n):
    total_cost = sum(
        n[j]["val"] * data.node_costs[j] for j in range(len(data.nodes))
    )
    return total_cost <= data.node_budget + 1e-6

