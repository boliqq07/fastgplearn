import time

from mgetool.tool import tt
from numpy import random
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

from fastgplearn.skflow import SymbolicRegressor

if __name__ == "__main__":
    # data

    x, y = load_boston(return_X_y=True)
    sr = SymbolicRegressor(population_size=10000, generations=20, stopping_criteria=0.95,
                           store=False, p_mutate=0.2, p_crossover=0.5, select_method="tournament",
                           tournament_size=5, hall_of_fame=3, store_of_fame=50,
                           constant_range=(0, 1.0), constants=None, depth=(2, 5),
                           function_set=('add', 'sub', 'mul', 'div', "pow2", "pow3","exp"),
                           n_jobs=1, verbose=True, random_state=0, method_backend='p_numpy', func_p=None, sci_preset="default")
    tt.t
    sr.fit(x,y)
    res = sr.top_n(30)
    tt.t
    tt.p

    #
    # sr = SymbolicEstimator(population_size=10000, generations=20,  stopping_criteria=0.95,
    #              hall_of_fame=3, store=False, p_mutate=0.2, p_crossover=0.5,select_method="tournament",
    #              tournament_size=5,
    #              constant_range=(0, 1.0), constants=None, depth=(2, 5),
    #              function_set=('add', 'sub', 'mul', 'div', "pow2", "pow3","exp"),
    #              warm_start=False, low_memory=False,
    #              n_jobs=1, verbose=True, random_state=None, method_backend='p_numpy', func_p=None,sci_preset="default")
    # tt.t
    # sr.fit(x,y)
    # tt.t
    # tt.p

    # sr = SymbolicEstimator(population_size=10000, generations=20,  stopping_criteria=0.95,
    #              hall_of_fame=3, store=False, p_mutate=0.2, p_crossover=0.5,select_method="tournament",
    #              tournament_size=5,
    #              constant_range=(0, 1.0), constants=None, depth=(2, 5),
    #              function_set=('add', 'sub', 'mul', 'div', "pow2", "pow3","exp"),
    #              warm_start=False, low_memory=False,
    #              n_jobs=1, verbose=True, random_state=None, method_backend='c_torch', func_p=None,sci_preset="default")
    # sr.fit(x,y)


    # sr = SymbolicEstimator(population_size=10000, generations=20,  stopping_criteria=0.95,
    #              hall_of_fame=3, store=False, p_mutate=0.2, p_crossover=0.5,select_method="tournament",
    #              tournament_size=5,
    #              constant_range=(0, 1.0), constants=None, depth=(2, 5),
    #              function_set=('add', 'sub', 'mul', 'div', "pow2", "pow3","exp"),
    #              warm_start=False, low_memory=False,
    #              n_jobs=1, verbose=True, random_state=None, method_backend='p_torch', func_p=None,sci_preset="default")
    # tt.t
    # sr.fit(x,y)
    # tt.t
    # tt.p




r2_score