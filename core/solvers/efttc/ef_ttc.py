from core.solvers.efttc.ef_ttc_solver import *
from core.solvers.neptune.utils.output import convert_x_matrix, convert_c_matrix
from ..solver import Solver

class EFTTC(Solver):
    def __init__(self, step1=None, **kwargs):
        super().__init__(**kwargs)
        self.step1 = step1
        self.solved = False

    def init_vars(self): pass
    def init_constraints(self): pass

    def solve(self):
        self.step1.load_data(self.data)
        self.solved = self.step1.solve()
        self.step1_x, self.step1_c = self.step1.results()
        return self.solved

    def results(self):
        return convert_x_matrix(self.step1_x, self.data.nodes, self.data.functions), \
               convert_c_matrix(self.step1_c, self.data.functions, self.data.nodes)

    def score(self):
        return {"step1": self.step1.score(), "step2": None}

class EFTTCMinDelay(EFTTC):
    def __init__(self, **kwargs):
        super().__init__(step1=EF_TTC_MinDelay(**kwargs), **kwargs)

class EFTTCMinUtilization(EFTTC):
    def __init__(self, **kwargs):
        super().__init__(step1=EF_TTC_MinUtilization(**kwargs), **kwargs)

class EFTTCMinDelayAndUtilization(EFTTC):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(step1=EF_TTC_MinDelayAndUtilization(alpha=alpha, **kwargs), **kwargs)
