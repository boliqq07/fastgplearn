Regression
==============

Prepare the data
::

>>> from sklearn.datasets import load_boston
>>> x, y = load_boston(return_X_y=True)

Fitting
::

>>> from skflow import SymbolicRegressor
>>> sr = SymbolicRegressor(population_size=1000, generations=10, stopping_criteria=0.95,
>>>                         store=False, p_mutate=0.2, p_crossover=0.5, select_method="tournament",
>>>                         tournament_size=5, hall_of_fame=3, store_of_fame=50,
>>>                         constant_range=(0, 1.0), constants=None, depth=(2, 5),
>>>                         function_set=('add', 'sub', 'mul', 'div'),
>>>                         n_jobs=1, verbose=True, random_state=0, method_backend='p_numpy', func_p=None,
>>>                         sci_preset="default")
>>> sr.fit(x, y)



Result
::

For result, you can specify the number of expression to calculate or score.

>>> sr.top_n(n = 10)
>>> res0 = sr.score(x, y, n=0)
>>> pre_y = sr.predict(x, y=None, n=0)
