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
import torch
from fastgplearn.backend.c_torch import c_torch_score, c_torch_score_mp

from fastgplearn.backend.p_torch import p_torch_score, p_torch_score_mp

from fastgplearn.backend.c_numpy import c_np_score, c_np_score_mp

from fastgplearn.backend.p_numpy import p_np_score, p_np_score_mp

from mgetool.tool import tt
from numpy.random import choice

from fastgplearn.gp import generate_random

if __name__ == "__main__":
    np.random.seed(0)

    x = np.random.random(size = (10,100),)
    y = np.random.random(size = 100,)
    x= x.astype(np.float32)
    y=y.astype(np.float32)
    func_index = (0,1,2,3,4,5,6,7,8,9,10,11,12)

    x2 = torch.from_numpy(x)
    y2 = torch.from_numpy(y)

    tt.t1
    deps = generate_random(func_num=13, xs_num=10, pop_size=100000, depth_min=1, depth_max=3, p=None, func_p=None, xs_p=None)
    tt.t2
    deps = deps.tolist()

    tt.t3
    rs = p_np_score(deps, x, y, func_index=func_index)
    tt.t4
    rs = p_np_score_mp(deps, x, y, func_index=func_index, n_jobs=4)
    tt.t5

    rs = c_np_score(deps, x, y, func_index=func_index)
    tt.t6
    rs = c_np_score_mp(deps, x, y, func_index=func_index, n_jobs=4)
    tt.t7

    rs1 = p_torch_score(deps, x2, y2, func_index=func_index)
    tt.t8
    rs2 = p_torch_score_mp(deps, x2, y2, func_index=func_index, n_jobs=4)
    tt.t9
    rs3 = c_torch_score(deps, x2, y2, func_index=func_index,return_numpy=True)
    tt.t10
    rs4 = c_torch_score_mp(deps, x2, y2, func_index=func_index, n_jobs=4,return_numpy=True)
    tt.t11
    tt.p

