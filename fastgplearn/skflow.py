# -*- coding: utf-8 -*-

# @Time     : 2021/10/31 16:13
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx


from abc import ABC, abstractmethod

import numpy as np
from mgetool.tool import time_this_function
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer

from fastgplearn.backend import backend
from fastgplearn.backend.p_numpy import p_np_str_name, p_np_cal, lg, lr
from fastgplearn.gp import set_seed, generate_random, select_index, mutate_random, crossover, mutate_sci
from fastgplearn.sci_formula import usr_preset
from fastgplearn.tools import Hall, Logs

try:
    import torch
except ImportError:
    torch = None


class SymbolicEstimator(BaseEstimator, ABC):
    def __init__(self, population_size=10000, generations=20, stopping_criteria=0.95, store_of_fame=50,
                 hall_of_fame=3, store=False, p_mutate=0.2, p_crossover=0.5, select_method="tournament",
                 tournament_size=5, device="cpu", sci_preset=None,
                 constant_range=(0, 1.0), constants=None, depth=(2, 5),
                 function_set=('add', 'sub', 'mul', 'div', "pow2", "pow3"),
                 n_jobs=1, verbose=0, random_state=None, method_backend='p_numpy', func_p=None,

                 ):
        """

        Args:
            population_size (int): number of population, default 10000.
            generations (int): number of generations, default 20.
            tournament_size (int): tournament size for selection.
            stopping_criteria (float): criteria of correlation score, max 1.0.
            constant_range (tuple): floats. constant_range=(0,1.0)
            constants (tuple): floats. constants=(-1,1,2,10), if given, The parameter constant_range would be ignored.
            depth (tuple): default (2, 5), The max of depth is not more than 8.
            function_set (tuple): tuple of str. optional: ('add', 'sub', 'mul', 'div', "ln", "exp", "pow2", "pow3", "rec", "max", "min", "sin", "cos").
            n_jobs (int):n jobs to parallel.
            verbose (bool):print message.
            p_mutate: mutate probability.
            p_crossover (float): crossover probability.
            random_state (int):random state
            hall_of_fame (int): hall of frame number to add to next generation.
            store_of_fame (int): hall of frame number to return result.
            method_backend (str): optional: ("p_numpy","c_numpy","p_torch","c_torch")
            device (str): default "cpu", "cuda:0", only accessible of torch.
            func_p (np.ndarray,tuple): with shape (n_function,), probability values of each function.
            sci_preset (str,list): None, "default" or user self-defined list template, default None.


        """

        assert population_size > 100 and generations >= 1
        assert tournament_size <= 30 and hall_of_fame <= store_of_fame
        assert hall_of_fame >= 1
        assert store_of_fame >= 10
        assert depth[1] <= 8
        try:
            self.func_names, self._str_name, self._score_mp, self._score, self._cal = backend[method_backend]
        except KeyError:
            raise NotImplementedError(
                f"The {method_backend} is just accessible after be compiled, see more in 'Install'.")
        self.population_size = self.pop_n = population_size
        self.hall_of_fame = hall_of_fame
        self.generations = self.gen = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.constants = constants
        self.depth = depth
        self.select_method = select_method
        self.depth_min = depth[0]
        self.depth_max = depth[1]
        self.function_set = function_set
        self.store_of_fame = store_of_fame

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.method_backend = method_backend
        self.p_mutate = p_mutate
        self.p_crossover = p_crossover
        self.constant_n = 5
        self.device = device

        self.constant_gen = []
        self.hall = Hall(size=store_of_fame)
        self.logs = Logs()
        self.store = store

        self.all_scores = []

        if random_state is not None:
            set_seed(random_state)

        self.func_index = [self.func_names.index(f"{i}_") for i in function_set]
        self.func_num = len(function_set)
        self.xs_p = None

        if constants is None:
            self.constant_range = None
        else:
            self.constant_range = constant_range
        self.constants = constants

        if isinstance(func_p, (list, tuple)):
            func_p = list(func_p)
            self.func_p = np.array(func_p)
        else:
            self.func_p = func_p

        self.sci_preset = self.filter_sci_perset(sci_preset)
        self.xs = None
        self.y = None
        self.x_label = None

    def filter_sci_perset(self, sci_preset):
        """Get the available sci available"""
        if sci_preset is None:
            sci_preset = []
        elif sci_preset == "default":
            sci_preset = usr_preset

        temp_sci_preset = []

        func_index_ = set(tuple(self.func_index))
        for site, pre_i in sci_preset:
            if set(tuple(pre_i)) <= func_index_:
                temp_sci_preset.append([list(pre_i), [self.func_index.index(i) for i in pre_i]])

        return temp_sci_preset

    def refresh_xcs_more(self):
        """Refresh X and constant for each generation for torch."""
        xcs, xcs_p, xcs_p_shape0, y = self.refresh_xcs()

        if "torch" in self.method_backend and torch is not None:
            if self.device != "cpu":
                dev = torch.device(self.device)
                xcs = torch.from_numpy(xcs).to(device=dev)
                y = torch.from_numpy(y).to(device=dev)
            else:
                xcs = torch.from_numpy(xcs)
                y = torch.from_numpy(y)

        return xcs, xcs_p, xcs_p_shape0, y

    def refresh_xcs(self):
        """Refresh X and constant for each generation."""
        if self.constants is None and self.constant_range is None:
            temp_cons = np.zeros(self.constant_n)
            self.constant_gen.append(temp_cons)
            return self.xs, None, self.xs.shape[0], self.y

        elif self.constants is None and self.constant_range is not None:
            temp_cons = np.random.random(self.constant_range[0], self.constant_range[1], self.constant_n).astype(
                np.float32)
        else:
            temp_cons = np.array(self.constants).astype(np.float32)
            self.constant_n = temp_cons.shape[0]
        self.constant_gen.append(temp_cons)
        xcs = np.concatenate((self.xs, np.array(temp_cons).reshape(-1, 1)))

        if self.xs_p is not None:
            xcs_p = np.append(0.9 * self.xs_p, 0.1 / np.ones_like(temp_cons))
        else:
            xcs_p = None
        y = self.y

        return xcs, xcs_p, xcs_p.shape[0], y

    @time_this_function
    def fit(self, X: np.ndarray, y: np.ndarray, xs_p: np.ndarray = None, x_label=None):
        """
        Fitting.

        Args:
            X (np.ndarray): with shape (n_sample,n_fea).
            y (np.ndarray): with shape (n_sample,).
            xs_p (np.ndarray): with shape (n_fea,), probability values of each xi.
            x_label (np.ndarray): with shape (n_fea), names of xi.

        """
        self.x_label = x_label
        X, y = self._validate_data(X, y=y)

        if X.ndim == 1:
            raise IndexError("Try to reshape your date to (-1,1), X just accept 2D array.")
        else:
            pass

        self.xs = X.T.astype(np.float32)  # change to (n_sample,n_x)

        self.y = y.ravel().astype(np.float32)

        if isinstance(xs_p, (list, tuple)):
            xs_p = list(xs_p)
            assert len(xs_p) == self.xs.shape[0]
            self.xs_p = np.array(xs_p)
        else:
            self.xs_p = xs_p

        self.hall.refresh_x_num(self.xs.shape[0])
        self.run_gp()

    def single_name(self, n):
        """Get the name of n_ed expression name."""
        n_fea, x_sam = self.xs.shape
        real_names = self.x_label
        if real_names is not None:
            assert len(real_names) == n_fea, "The name of feature are not consist with X."
            xcs = real_names
        else:
            xcs = list(range(n_fea))

        ind, _gen_i, _score, const = self.hall[n]

        if np.all(ind < n_fea):
            cns = None
        else:
            cns = const.tolist()

        return self._single_name(ind.tolist(), xcs, cns=cns, y=self.y, func_index=self.func_index,
                                 real_names=real_names)

    @staticmethod
    def _single_name(vei, xns, cns=None, y=None, func_index=None, real_names=None):
        """Get the name of n_ed expression name."""
        return p_np_str_name([vei, ], xns, cns=cns, y=y, func_index=func_index, real_names=real_names)[0]

    def single_cal(self, n, new_x=None, with_coef=True):
        """Get the temp predict y of n_ed expression name (without coef and intercept),This is not the final result!
        """
        n_fea, x_sam = self.xs.shape
        if new_x is not None:
            assert with_coef is False
            n_fea2, x_sam = new_x.shape
            assert n_fea2 == n_fea, "The numbers of feature for train and test must be same."
            xs = new_x.astype(np.float32)
        else:
            xs = self.xs

        ind, _gen_i, _score, const = self.hall[n]
        if np.all(ind < n_fea):
            xcs = xs
        else:
            xcs = np.concatenate((xs, np.repeat(const.reshape(-1, 1), x_sam, axis=1)), axis=0)

        return self._single_cal(ind.tolist(), xcs, self.y, func_index=self.func_index, with_coef=with_coef)

    @abstractmethod
    def _single_cal(self, vei, xs, y, func_index, with_coef=False):
        """Return tuple(y, coef, intercept). if with coef is False, coef and intercept are all 0."""

    @abstractmethod
    def predict(self, X, y=None, n=0):
        """Return the real predicted y."""

    def score(self, X, y, scoring, n=0):
        """Score."""
        scorer = get_scorer(scoring=scoring)
        func = scorer._score_func
        sign = scorer._sign
        pre_y = self.predict(X, y=None, n=n)
        return sign * func(y, pre_y)

    def run_gp(self):
        """
        Run the GP processing.
        """
        if self.verbose:
            self.logs.print(head=True)

        xcs, xcs_p, xcs_num, y = self.refresh_xcs_more()

        # 1.generate###################################################################

        population = generate_random(self.func_num, xs_num=xcs_num, pop_size=self.pop_n,
                                     depth_min=self.depth_min, depth_max=self.depth_max,
                                     p=None, func_p=self.func_p, xs_p=xcs_p)

        for gen_i in range(1, self.gen + 1):

            # 2.score###################################################################
            population_list = population.tolist()

            score_result = self._score_mp(population_list, xs=xcs, y=y, func_index=self.func_index,
                                          n_jobs=self.n_jobs)

            if "torch" in self.method_backend:
                score_result = score_result.cpu().numpy()

            # 3.log-hall###############################

            if self.store:
                self.all_scores.append(score_result)

            self.hall.update(population, gen_i, score_result, self.constant_gen[-1])
            max_ind_score = np.max(score_result)
            self.logs.record(max_ind_score)
            if self.verbose:
                self.logs.print()
            if max_ind_score >= self.stopping_criteria:
                print("Reach the stopping criteria and terminate early at generation {}".format(gen_i))
                break

            if gen_i == self.gen:
                break

            # 4. next generation ！！！！#######################################################
            # selection and mutate,mate,migration

            if self.constants is None:
                xcs, xcs_p, xcs_num, y = self.refresh_xcs_more()

            couple_index = select_index(score_result, num_percent=self.pop_n - self.hall_of_fame,
                                        method=self.select_method,
                                        tour_num=self.tournament_size)

            couple_index = population[couple_index]

            offspring = crossover(couple_index, p_crossover=self.p_crossover)

            sp = int(0.7 * self.pop_n)

            population[:sp] = mutate_random(offspring[:sp], self.func_num, xcs_num, pop_size=self.pop_n,
                                            depth_min=self.depth_min, depth_max=self.depth_max, p_mutate=self.p_mutate,
                                            p=None, func_p=self.func_p, xs_p=xcs_p)

            population[sp:(self.pop_n - self.hall_of_fame)] = mutate_sci(self.func_num, xcs_num,
                                                                         pop_size=self.pop_n - sp - self.hall_of_fame,
                                                                         depth_min=1, depth_max=5,
                                                                         p=None, func_p=self.func_p, xs_p=xcs_p,
                                                                         sci_preset=self.sci_preset)
            if self.hall_of_fame >= 0:
                population[-self.hall_of_fame:] = self.hall.inds[:self.hall_of_fame]


#
class SymbolicRegressor(SymbolicEstimator):
    """
    A Genetic Programming symbolic regressor.

    A symbolic regressor is an estimator that begins by building a population of naive random formulas to
    represent a relationship. The formulas are represented as tree-like structures with mathematical functions
    being recursively applied to variables and constants. Each successive generation of programs is then evolved
    from the one that came before it by selecting the fittest individuals from the population to undergo genetic
    operations such as crossover, mutation or reproduction.

    The default score for find expression is R (correlation coefficient), Thus this score needs to be further calculated.

    Examples:

    >>> from fastgplearn.skflow import SymbolicRegressor
    >>> est_gp = SymbolicRegressor(population_size=5000,
    ...                     generations=20, stopping_criteria=0.01,
    ...                     p_crossover=0.7, p_mutate_=0.1,
    ...                     max_samples=0.9, verbose=1,
    ...                     random_state=0)
    >>> est_gp.fit(X_train, y_train)
    >>> est_gp.top_n()
    >>> test_score = est_gp.score(X_test,y_test)

    """

    def __init__(self, population_size=10000, generations=20, stopping_criteria=0.95, store_of_fame=50,
                 hall_of_fame=3, store=False, p_mutate=0.2, p_crossover=0.5, select_method="tournament",
                 tournament_size=5, constant_range=(0, 1.0), constants=None, depth=(2, 5),
                 function_set=('add', 'sub', 'mul', 'div', "pow2", "pow3"), sci_preset=None,
                 device="cpu", n_jobs=1, verbose=0, random_state=None, method_backend='p_numpy', func_p=None,
                 ):
        """

        Args:
            population_size (int): number of population, default 10000.
            generations (int): number of generations, default 20.
            tournament_size (int): tournament size for selection.
            stopping_criteria (float): criteria of correlation score, max 1.0.
            constant_range (tuple): floats. constant_range=(0,1.0)
            constants (tuple): floats. constants=(-1,1,2,10), if given, The parameter constant_range would be ignored.
            depth (tuple): default (2, 5), The max of depth is not more than 8.
            function_set (tuple): tuple of str. optional: ('add', 'sub', 'mul', 'div', "ln", "exp", "pow2", "pow3",
                "rec", "max", "min", "sin", "cos").
            n_jobs (int):n jobs to parallel.
            verbose (bool):print message.
            p_mutate: mutate probability.
            p_crossover (float): crossover probability.
            random_state (int):random state
            hall_of_fame (int): hall of frame number to add to next generation.
            store_of_fame (int): hall of frame number to return result.
            method_backend (str): optional: ("p_numpy","c_numpy","p_torch","c_torch")
            device (str): default "cpu", "cuda:0", only accessible of torch.
            func_p (np.ndarray): with shape (n_function,), probability values of each function.
            sci_preset (str,list): None, "default" or user self-defined list template, default None.

        """
        super(SymbolicRegressor, self).__init__(population_size=population_size, generations=generations,
                                                stopping_criteria=stopping_criteria, store=store,
                                                store_of_fame=store_of_fame, random_state=random_state,
                                                hall_of_fame=hall_of_fame, p_mutate=p_mutate, depth=depth,
                                                p_crossover=p_crossover, function_set=function_set,
                                                select_method=select_method, method_backend=method_backend,
                                                tournament_size=tournament_size, sci_preset=sci_preset,
                                                constant_range=constant_range, constants=constants,
                                                device=device, n_jobs=n_jobs, verbose=verbose, func_p=func_p,
                                                )
        self.logs = Logs("Person Corr")

    @staticmethod
    def single_coef_linear(X, y):
        """Fitting by sklearn.linear_model.LinearRegression."""
        lr.fit(X.reshape(-1, 1), y)
        return lr.coef_[0] * X + lr.intercept_, lr.coef_[0], lr.intercept_

    def _single_cal(self, vei, xs, y, func_index, with_coef=False):
        """Return (y,coef,intercept). if with coef is False, coef and intercept are all 0."""

        pre_y = p_np_cal([vei, ], xs, y, func_index)[0]

        if with_coef:
            pre_y_coef, coef_, intercept_ = self.single_coef_linear(pre_y, y)
            return pre_y_coef, coef_, intercept_
        else:
            return pre_y, 0, 0

    def predict(self, X, y=None, n=0):
        """
        Return the real predicted y.

        Args:
            X (np.ndarray): array-like of shape (n_samples, n_features).
            Input vectors, where n_samples is the number of samples and n_features is the number of features
            y (np.ndarray): array-like of shape (n_samples,).
            n (int): calculate by the n_ed expression.

        Returns:
            y (np.ndarray): array-like of shape (n_samples,).
        """
        _ = y
        X = X.T
        _, coef_, intercept_ = self.single_cal(n=n, with_coef=True)
        return self.single_cal(n=n, new_x=X, with_coef=False)[0] * coef_ + intercept_

    def score(self, X, y, scoring="r2", n=0):
        """
        Return the r2 score (default) on the given test data and labels.

        Args:
            X (np.ndarray): array-like of shape (n_samples, n_features).
            y (np.ndarray): array-like of shape (n_samples,).
            scoring (str): see also sklearn.metrics.
            n (int): calculate by the n_ed expression.

        Returns:
            score (float): Mean r2 of ``self.predict(X)`` wrt. `y`.
        """
        return super().score(X, y, scoring, n=n)

    def top_n(self, n=0, scoring="r2"):
        """Print the top n result. The best one is index 0.

        Args:
            scoring (str): see also sklearn.metrics.
            n (int): calculate by the n_ed expression.

        """
        assert self.store_of_fame >= n, "The top_n just accessible while 'hall_of_fame >= n'."
        scorer = get_scorer(scoring=scoring)
        func = scorer._score_func
        sign = scorer._sign

        self.logs.record_and_print(" ")
        self.logs.record_and_print(f"The top {n} result:")
        self.logs.record_and_print(f"Scoring by {scoring}: ( score, expression, coef, intercept )")

        for ni in range(n):
            pre_y, coef_, intercept_ = self.single_cal(n=ni, with_coef=True)
            msg = str((sign * func(self.y, pre_y), self.single_name(ni), coef_, intercept_))
            self.logs.record_and_print(msg)

    def best_expression(self, scoring="r2"):
        """Print the best expression."""
        self.top_n(n=0, scoring=scoring)


class SymbolicClassifier(SymbolicEstimator):
    """
    A Genetic Programming symbolic classifier.

    A symbolic classifier is an estimator that begins by building a population of naive random formulas
    to represent a relationship. The formulas are represented as tree-like structures with mathematical
    functions being recursively applied to variables and constants. Each successive generation of programs
    is then evolved from the one that came before it by selecting the fittest individuals from the population
    to undergo genetic operations such as crossover, mutation or reproduction.

    The default score for find expression is accuracy.

    Examples:

    >>> from fastgplearn.skflow import SymbolicRegressor
    >>> est_gp = SymbolicRegressor(population_size=5000,
    ...                     generations=20, stopping_criteria=0.01,
    ...                     p_crossover=0.7, p_mutate_=0.1,
    ...                     max_samples=0.9, verbose=1,
    ...                     random_state=0)
    >>> est_gp.fit(X_train, y_train)
    >>> est_gp.top_n()
    >>> test_score = est_gp.score(X_test,y_test)
    """

    def __init__(self, population_size=10000, generations=20, stopping_criteria=0.95, store_of_fame=50,
                 hall_of_fame=3, store=False, p_mutate=0.2, p_crossover=0.5, select_method="tournament",
                 tournament_size=5, device="cpu", sci_preset=None,
                 constant_range=(0, 1.0), constants=None, depth=(2, 5),
                 function_set=('add', 'sub', 'mul', 'div', "pow2", "pow3"),
                 n_jobs=1, verbose=0, random_state=None, method_backend='p_numpy', func_p=None,
                 ):
        """

        Args:
            population_size (int): number of population, default 10000.
            generations (int): number of generations, default 20.
            tournament_size (int): tournament size for selection.
            stopping_criteria (float): criteria of correlation score, max 1.0.
            constant_range (tuple): floats. constant_range=(0,1.0)
            constants (tuple): floats. constants=(-1,1,2,10), if given, The parameter constant_range would be ignored.
            depth (tuple): default (2, 5), The max of depth is not more than 8.
            function_set (tuple): tuple of str. optional: ('add', 'sub', 'mul', 'div', "ln", "exp", "pow2", "pow3",
                "rec", "max", "min", "sin", "cos").
            n_jobs (int):n jobs to parallel.
            verbose (bool):print message.
            p_mutate: mutate probability.
            p_crossover (float): crossover probability.
            random_state (int):random state
            hall_of_fame (int): hall of frame number to add to next generation.
            store_of_fame (int): hall of frame number to return result.
            method_backend (str): optional: ("p_numpy","c_numpy","p_torch","c_torch")
            device (str): default "cpu", "cuda:0", only accessible of torch.
            func_p (np.ndarray,tuple): with shape (n_function,), probability values of each function.
            sci_preset (str,list): None, "default" or user self-defined list template, default None.

        """
        super(SymbolicClassifier, self).__init__(population_size=population_size, generations=generations,
                                                 stopping_criteria=stopping_criteria, store=store,
                                                 store_of_fame=store_of_fame, random_state=random_state,
                                                 hall_of_fame=hall_of_fame, p_mutate=p_mutate, depth=depth,
                                                 p_crossover=p_crossover, function_set=function_set,
                                                 select_method=select_method, method_backend=method_backend,
                                                 tournament_size=tournament_size, sci_preset=sci_preset,
                                                 constant_range=constant_range, constants=constants,
                                                 device=device, n_jobs=n_jobs, verbose=verbose, func_p=func_p, )
        self.logs = Logs("Accuracy")

    def fit(self, X: np.ndarray, y: np.ndarray, xs_p: np.ndarray = None, x_label=None):
        """
        Fitting.

        Args:
            X (np.ndarray): with shape (n_sample,n_fea).
            y (np.ndarray): with shape (n_sample,).
            xs_p (np.ndarray): with shape (n_fea,), probability values of each xi.
            x_label (np.ndarray): with shape (n_fea), names of xi.

        """
        class_y = np.unique(y)
        need = {0, 1}
        if class_y.ndim != 2 and set([i for i in class_y]) != need:
            raise ValueError("Just for binary classification problem (with 2 labels: [0,1]).")
        y = y.astype(np.float32)
        super().fit(X, y, xs_p, x_label=x_label)

    def single_coef_logistic(self, X, y):
        """Fitting by sklearn.linear_model.LogisticRegression."""
        lg.fit(X.reshape(-1, 1), y)
        return lg.coef_[0] * X + lg.intercept_, lg.coef_[0], lg.intercept_

    def _single_cal(self, vei, xs, y, func_index, with_coef=False):
        """Return (y,coef,intercept). if with coef is False, coef and intercept are all 0."""

        pre_y = p_np_cal([vei, ], xs, y, func_index)[0]

        if with_coef:
            pre_y_coef, coef_, intercept_ = self.single_coef_logistic(pre_y, self.y)
            return pre_y_coef, coef_, intercept_
        else:
            return pre_y, 0, 0

    @staticmethod
    def cla(pre_y):
        """classification tool."""
        pre_y = 1.0 / (1.0 + np.exp(-pre_y))
        pre_y[np.where(pre_y >= 0.5)] = 1
        pre_y[np.where(pre_y < 0.5)] = 0
        return pre_y

    def predict(self, X, y=None, n=0):
        """
        Return the real predicted y.

        Args:
            X (np.ndarray): array-like of shape (n_samples, n_features).
            Input vectors, where n_samples is the number of samples and n_features is the number of features
            y (np.ndarray): array-like of shape (n_samples,).
            n:

        Returns:
            y (np.ndarray): array-like of shape (n_samples,).
        """

        _ = y

        X = X.T
        _, coef_, intercept_ = self.single_cal(n=n, with_coef=True)
        return self.cla(self.single_cal(n=n, new_x=X, with_coef=False)[0] * coef_ + intercept_)

    def score(self, X, y, scoring="accuracy", n=0):
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X (np.ndarray): array-like of shape (n_samples, n_features).
            y (np.ndarray): array-like of shape (n_samples,).
            scoring (str): see also sklearn.metrics.
            n (int): calculate by the n_ed expression.

        Returns:
            score (float): Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        return super().score(X, y, scoring, n=n)

    def top_n(self, n=0, scoring="accuracy"):
        """Print the top n result. The best one is index 0.

        Args:
            scoring (str): see also sklearn.metrics.
            n (int): calculate by the n_ed expression.

        """

        assert self.store_of_fame >= n, "The top_n just accessible while 'hall_of_fame >= n'."
        scorer = get_scorer(scoring=scoring)
        func = scorer._score_func
        sign = scorer._sign

        self.logs.record_and_print(f"The top {n} result:")
        self.logs.record_and_print(f"Scoring by {scoring}: ( score, expression, coef, intercept )")

        for ni in range(n):
            pre_y, coef_, intercept_ = self.single_cal(n=ni, with_coef=True)
            msg = str((sign * func(self.y, self.cla(pre_y)), self.single_name(ni), coef_, intercept_))
            self.logs.record_and_print(msg)

    def best_expression(self, scoring="accuracy"):
        """Print the best expression."""
        self.top_n(n=0, scoring=scoring)
