# -*- coding: utf-8 -*-

# @Time     : 2021/10/31 21:43
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

import pathlib
from multiprocessing import Pool

import numpy as np
import torch
from mgetool.draft import TorchJit

# second
path = pathlib.Path(__file__)
file = path.parent.parent / "source" / "torch_tool.cpp"
file = str(file.absolute())
bd = TorchJit(file, temps="torch_temp", warm_start=True, log_print=False)
c_torch_backend_tool = bd.quick_import(build=False, suffix=None)
c_torch_score0 = c_torch_backend_tool.c_torch_score
c_torch_cal0 = c_torch_backend_tool.c_torch_cal


def c_torch_score(ve, xs, y, func_index, return_numpy=False, clf=False, single_start=6):
    """Batch score."""
    res = c_torch_score0(ve, xs, y, func_index, clf, single_start)
    if return_numpy:
        return res.numpy()
    else:
        return res


def c_torch_score_temp(ve, xs, y, func_index, clf=False, single_start=6):
    """Batch score."""
    return c_torch_score0(ve, xs, y, func_index, clf, single_start)


def c_torch_cal(ve, xs, y, func_index, clf=False, single_start=6):
    """Batch calculate."""
    res = c_torch_cal0(ve, xs, y, func_index, clf, single_start)
    return res


def c_torch_score_mp(ve, xs, y, func_index, n_jobs=1, return_numpy=False, clf=False, single_start=6):
    """Batch score with n_jobs."""
    if isinstance(ve, np.ndarray):
        ve = ve.tolist()
    if n_jobs == 1:
        return c_torch_score(ve, xs, y, func_index, return_numpy, clf, single_start=single_start)
    else:
        for i in range(3):
            print("For torch with c++ (c_torch), with n_jobs>1, this function is very slow!")

        pool = Pool(n_jobs)

        left = int(len(ve) % n_jobs)

        if left > 0:
            bs = int(len(ve) // n_jobs)
            nve = [ve[bs * (i - 1):i * bs] for i in range(1, n_jobs + 1)]
            nve.append(ve[-left:])
        else:
            bs = int(len(ve) // n_jobs)
            nve = [ve[bs * (i - 1):i * bs] for i in range(1, n_jobs + 1)]

        res = []
        for nvei in nve:
            ret = pool.apply(c_torch_score_temp, (nvei, xs, y, func_index, clf, single_start))
            res.append(ret)
        pool.close()
        pool.join()

        if not return_numpy:
            res = torch.cat(res)
        else:
            res = np.concatenate(res)
        return res
