# -*- coding: utf-8 -*-

# @Time     : 2021/11/2 23:06
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
# -*- coding: utf-8 -*-

# @Time     : 2021/11/1 9:20
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

import numpy as np
from mgetool.tool import tt
from numpy.random import choice
from fastgplearn.gp import generate_random, mutate_random, mutate_sci, sub_science, csub_science

if __name__ == "__main__":
    np.random.seed(0)

    x = np.random.random(size = (10,100),)
    y = np.random.random(size = 100,)
    x= x.astype(np.float32)
    y=y.astype(np.float32)
    # tt.t
    # pop_np = generate_random(func_num=13, xs_num=10, pop_size=10, depth_min=1, depth_max=3, p=None, func_p=None, xs_p=None)
    # tt.t
    # deps2 = mutate_random(pop_np[:3],func_num=13, xs_num=10,pop_size=10, depth_min=1, depth_max=3, p_mutate=0.001, p=None,func_p=None,
    #                       xs_p=None)
    tt.t

    new_pop = generate_random(func_num=13, xs_num=10, pop_size=10, depth_min=1, depth_max=5,
                              )

    tt.t
    new_pop = csub_science(new_pop, [[[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]]])
    tt.t

    tt.p



