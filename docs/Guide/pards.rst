Parameters details
===================

For both :py:class:`fastgplearn.skflow.SymbolicRegressor` and :py:class:`fastgplearn.skflow.SymbolicClassifier` .

Init Parameters:
::::::::::::::::::::::

================== =============== ========= =======================  ======================================================
Parameters name    Type            Default   Suggest Range            Definition
------------------ --------------- --------- -----------------------  ------------------------------------------------------
population_size       (int)        10000     [50, 1000000]            number of population
generations           (int)        20        [10,...]                 number of generations
tournament_size       (int)        20        [5,20]                   number of for each turn of tournament
stopping_criteria    (float)       0.95      [0,1]                    early stopping criteria
constant_range     tuple of float  (0,1.0)   /                        constants would choice from range randomly
constants          tuple of float  None      /                        if given, the parameter constant_range would be ignored, and just use the constants offered
depth              tuple of float  (2,5)     1st [1,...], 2ed [2,8)   (min_depth,max_depth), keep the max of depth is not more than 8 !
function_set       tuple of string (+-*/)    /                        optional: ('add', 'sub', 'mul', 'div', "ln", "exp", "pow2", "pow3", "rec", "max", "min", "sin", "cos")
n_jobs                (int)        1         [1,...]                  n jobs to parallel
verbose               (bool)       True      True,False               print message
p_mutate             (float)       0.5       (0,1)                    mutate probability
p_crossover          (float)       0.5       (0,1)                    crossover probability
random_state          (int)        None      /                        random state
hall_of_fame          (int)        3         [0,10]                   hall of frame number to add to next generation
store_of_fame         (int)        3         [0,10]                   hall of frame number to return result
method_backend       (string)      "p_numpy" /                        optional: ("p_numpy","c_numpy","p_torch","c_torch")
device               (string)      "cpu"     /                        optional: ("cpu","cuda:0", ...) depend on your computer device
func_p              (np.ndarray)   None      /                        specific the probability values of each function
sci_template          list, str    "default" /                        user self-defined list template or "default" or  None
================== =============== ========= =======================  ======================================================


Fit Parameters:
::::::::::::::::::::::

Fit parameters in ``SymbolicRegressor().fit()`` or ``SymbolicClassifier().fit()`` method.

================== =============== ========= =======================  ======================================================
Parameters name    Type            Default   Suggest Range            Definition
------------------ --------------- --------- -----------------------  ------------------------------------------------------
X                  (np.ndarray)    /         /                        with shape (n_sample, n_fea)
y                  (np.ndarray)    /         /                        with shape (n_sample,)
xs_p               (np.ndarray)    None      /                        specific the probability values of each feature
x_label            list of string  None      /                        specific the name values of each feature
================== =============== ========= =======================  ======================================================


Other Parameters:
::::::::::::::::::::::

Other parameters present in ``predict()`` or ``score()``, or ``top_n()`` method.

================== =============== ========= =======================  ======================================================
Parameters name    Type            Default   Suggest Range            Definition
------------------ --------------- --------- -----------------------  ------------------------------------------------------
X                  (np.ndarray)    /         /                        with shape (n_sample, n_fea)
y                  (np.ndarray)    /         /                        with shape (n_sample,)
n                     (int)        0         0                        specify the number of expression to calculate or score
scoring               (str)        /         /                        for regression, default is "r2", for classification is "accuracy"
================== =============== ========= =======================  ======================================================