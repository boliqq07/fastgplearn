from mgetool.tool import tt
from mgetool.tool import tt
from sklearn.datasets import load_boston

from fastgplearn.skflow import SymbolicRegressor as FSR
from gplearn.genetic import SymbolicRegressor as SR
from bgp.skflow import SymbolLearning

x, y = load_boston(return_X_y=True)

sr1 = FSR(population_size=10000, generations=3, stopping_criteria=0.95,
          store=False, p_mutate=0.2, p_crossover=0.5, select_method="tournament",
          tournament_size=5, hall_of_fame=3, store_of_fame=50,
          constant_range=(0, 1.0), constants=None, depth=(2, 5),
          function_set=('add', 'sub', 'mul', 'div'),n_jobs=1, verbose=True,
          random_state=0, method_backend='p_numpy', func_p=None,
          sci_preset="default")

sr2 = SR(population_size=100000, generations=10, stopping_criteria=0.95, p_crossover=0.5,
         tournament_size=5, function_set=('add', 'sub', 'mul', 'div'), n_jobs=8, verbose=True, random_state=0,)

sr3 = SymbolLearning(loop="MultiMutateLoop", pop=100000, gen=10, random_state=0,add_coef=False,n_jobs=4)

tt.t
sr1.fit(x, y)
tt.t
sr1.top_n(5)
# sr2.fit(x, y)
# tt.t
# sr3.fit(x, y)
# tt.t
tt.p
