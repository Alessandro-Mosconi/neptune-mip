from core.solvers.efttc.efttc_step1 import *
from core.solvers.efttc.utils import *


class EfttcBase(Solver):
    def __init__(self, step1=None, step2_delete=None, step2_create=None, **kwargs):
        super().__init__(**kwargs)
        self.step1 = step1
        self.step2_delete = step2_delete
        self.step2_create = step2_create

    def init_vars(self): pass
    def init_constraints(self): pass

    def solve(self):
        self.step1.load_data(self.data)
        self.step1.solve()
        self.step1_x, self.step1_c = self.step1.results()
        self.data.max_score = self.step1.score()
        self.step2_x, self.step2_c = self.step1_x, self.step1_c
        return False
    
    def results(self):
        return convert_x_matrix(self.step1_x, self.data.nodes, self.data.functions), convert_c_matrix(self.step1_c, self.data.functions, self.data.nodes)

    def score(self):
        return { "step1": self.step1.score(), "step2": -1 }

class EfttcMinDelayAndUtilization(EfttcBase):
    def __init__(self, **kwargs):
        super().__init__(
            EfttcStep1CPUMinDelayAndUtilization(**kwargs),
            **kwargs
            )

class EfttcMinDelay(EfttcBase):
    def __init__(self, **kwargs):
        super().__init__(
            EfttcStep1CPUMinDelay(**kwargs),
            **kwargs
            )

class EfttcMinUtilization(EfttcBase):
    def __init__(self, **kwargs):
        super().__init__(
            EfttcStep1CPUMinUtilization(**kwargs),
            **kwargs
            )
