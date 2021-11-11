import unittest

import numpy as np

from fastgplearn.skflow import SymbolicRegressor

    # func_index = (0,1,2,3,4,5,6,7,8,9,10,11,12)
    # deps = generate_random(func_num=13, xs_num=10, pop_size=1000, depth_min=1, depth_max=3, p=None, func_p=None, xs_p=None)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        x = np.random.random(size=(100, 10), )
        x = x + 1
        x[:, 0] = 5*x[:, 0]
        x[:, 2] = 5*x[:, 2]
        y = np.random.random(size=100) * 0.01 + np.exp((x[:, 0]+x[:, 2]) / x[:, 1],)
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        self.x = x
        self.y = y

    def test_something(self):
        sr = SymbolicRegressor(population_size=10000, generations=30, stopping_criteria=1.0,
                                store=False, p_mutate=0.2, p_crossover=0.5, select_method="tournament",
                                tournament_size=5, hall_of_fame=3, store_of_fame=50,
                                constant_range=(0, 1.0), constants=None, depth=(2, 4),
                                function_set=('add', 'sub', 'mul', 'div',"exp"),
                                n_jobs=1, verbose=True, random_state=0, method_backend='p_numpy', func_p=None,
                                # sci_template="default")
                                sci_template=None)
        x, y = self.x, self.y
        sr.fit(x, y)
        sr.top_n(30)
        res0 = sr.score(x, y, n=0)



if __name__ == '__main__':
    unittest.main()
