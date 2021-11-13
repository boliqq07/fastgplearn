# -*- coding: utf-8 -*-

# @Time     : 2021/11/2 17:23
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import functools
import pathlib
from multiprocessing import Pool

import numpy as np
from mgetool.draft import DraftPyx

# second
path = pathlib.Path(__file__)
file = path.parent.parent / "source" / "np_tool.pyx"
file = str(file.absolute())
bd = DraftPyx(file, language="c++", temps="np_temps", warm_start=True, log_print=False)
c_numpy_backend_tool = bd.quick_import(build=False)
c_np_score = c_numpy_backend_tool.c_np_score
c_np_cal = c_numpy_backend_tool.c_np_cal
find_add_mask_all = c_numpy_backend_tool.find_add_mask_all
sci_subs = c_numpy_backend_tool.sci_subs


def c_np_score_mp(ve, xs, y, func_index=None, n_jobs=1, clf=False, single_start=6):
    """Batch score with n_jobs."""
    if isinstance(ve, np.ndarray):
        ve = ve.tolist()
    if n_jobs == 1:
        return c_np_score(ve, xs, y, func_index, clf=clf, single_start=single_start)
    else:

        pool = Pool(n_jobs)

        if func_index is not None:
            func_index = tuple(func_index)

        left = int(len(ve) % n_jobs)

        if left > 0:
            bs = int(len(ve) // n_jobs)
            nve = [ve[bs * (i - 1):i * bs] for i in range(1, n_jobs + 1)]
            nve.append(ve[-left:])
        else:
            bs = int(len(ve) // n_jobs)
            nve = [ve[bs * (i - 1):i * bs] for i in range(1, n_jobs + 1)]

        func = functools.partial(c_np_score, xs=tuple(xs), y=y, func_index=func_index, clf=clf,
                                 single_start=single_start)

        res = [i for i in pool.map_async(func, nve).get()]
        pool.close()
        pool.join()

        res = np.concatenate(res)

        return res
