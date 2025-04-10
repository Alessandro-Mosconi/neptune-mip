from .utils import *
from .efttc_step1 import *


class EfttcStep2Base(EfttcStepBase):
    def __init__(self, mode=str, soften_step1_sol=1.3, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        assert mode in ["delete", "create"]
        self.moved_from, self.moved_to = {}, {}
        self.soften_step1_sol = soften_step1_sol

    def init_vars(self):
        super().init_vars()
        init_moved_from(self.data, self.moved_from)
        init_moved_to(self.data, self.moved_to)
        self.allocated = init_allocated(self.data)
        self.deallocated = init_deallocated(self.data)

    def init_constraints(self):
        pass
    def init_objective(self):
        pass

    def get_constraints(self):
        self.log(self.data.old_allocations_matrix)
        self.log(self.data.core_per_req_matrix)
        self.log(self.data.workload_matrix)
        self.log(self.data.node_cores_matrix)

        base_constraints = (
                super().get_constraints() and
                constrain_handle_all_requests(self.data, self.x) and
                constrain_CPU_usage(self.data, self.x) and
                constrain_moved_from(self.data, self.moved_from, self.c) and
                constrain_moved_to(self.data, self.moved_to, self.c) and
                constrain_migrations(self.data, self.c, self.allocated, self.deallocated)
        )

        if self.mode == "delete":
            return base_constraints and constrain_deletions(self.data, self.c, self.allocated, self.deallocated)
        elif self.mode == "create":
            return base_constraints and constrain_creations(self.data, self.c, self.allocated, self.deallocated)
        else:
            return base_constraints

    def get_objective(self):
        return score_minimize_disruption(self.data, self.moved_from, self.moved_to, self.allocated, self.deallocated)

    def results(self):
        x, c = output_x_and_c(self.data, self.x, self.c)
        print("Step 2 - x:", x, sep='\n')
        print("Step 2 - c:", c, sep='\n')
        print("Number of delta pod deallocated")
        print(-self.deallocated["val"])
        print("Number of delta pod allocated")
        print(-self.allocated["val"])

        return x, c

class EfttcStep2MinUtilization(EfttcStep2Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = {}

    def init_vars(self):
        super().init_vars()
        init_n(self.data, self.n)

    def get_constraints(self):
        return (super().init_constraints() and
            constrain_n_according_to_c(self.data, self.n, self.c) and
            constrain_budget(self.data, self.n) and
            constrain_node_utilization(self.data, self.n, self.soften_step1_sol))

    def results(self):
        x, c = super().results()
        n = output_n(self.data, self.n)
        print("Step 2 - n:", n, sep='\n')
        return x, c


class EfttcStep2MinDelay(EfttcStep2Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_constraints(self):
        return (super().init_constraints() and
            constrain_network_delay(self.data, self.x, self.soften_step1_sol))


class EfttcStep2MinDelayAndUtilization(EfttcStep2MinUtilization):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def get_constraints(self):
        return (EfttcStep2Base.init_constraints(self) and
            constrain_n_according_to_c(self.data, self.n, self.c) and
            constrain_budget(self.data, self.n) and
            constrain_score(self.data, self.x, self.n, self.alpha, self.soften_step1_sol))
