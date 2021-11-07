import unittest
from sklearn.datasets import load_boston, load_iris
from skflow import SymbolicRegressor, SymbolicClassifier


class MyTestCase(unittest.TestCase):
    def setUp(self):
        x, y = load_iris(return_X_y=True)

        self.x=x[y<2]
        self.x[49,:]=4
        self.y=y[y<2]

    def test_something(self):

        sr = SymbolicClassifier(population_size=1000, generations=10, stopping_criteria=0.95,
                               store=False, p_mutate=0.2, p_crossover=0.5, select_method="tournament",
                               tournament_size=5, hall_of_fame=3, store_of_fame=50,
                               constant_range=(0, 1.0), constants=None, depth=(2, 5),
                               function_set=('add', 'sub', 'mul', 'div', "pow2", "pow3", "exp"),
                               n_jobs=1, verbose=True, random_state=0, method_backend='p_numpy', func_p=None,
                               sci_preset="default")
        x,y =self.x,self.y
        sr.fit(x, y)
        res = sr.top_n(30)
        res0 = sr.score(x, y, n=0)
        assert res0>0.5


if __name__ == '__main__':
    unittest.main()
