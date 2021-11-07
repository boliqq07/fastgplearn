# -*- coding: utf-8 -*-

# @Time     : 2021/11/1 11:07
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

from fastgplearn.backend.p_numpy import p_np_str_name, p_np_cal, p_np_score_mp, p_np_score
from fastgplearn.backend.p_torch import p_torch_score_mp, p_torch_score, p_torch_cal

__all__ = ["c_numpy","c_torch","p_torch","p_numpy"]

func_names = ["add_", "sub_", "mul_", "div_", "ln_", "exp_", "pow2_", "pow3_", "rec_", "max_", "min_", "sin_", "cos_"]

backend = {
    "p_numpy":(func_names, p_np_str_name,p_np_score_mp,p_np_score,p_np_cal),
    "p_torch":(func_names, p_np_str_name,p_torch_score_mp, p_torch_score, p_torch_cal),}
try:
    from fastgplearn.backend.c_numpy import c_np_score_mp, c_np_score, c_np_cal
    backend.update({"c_numpy": (func_names, p_np_str_name, c_np_score_mp, c_np_score, c_np_cal)})
except BaseException:
    pass
try:
    from fastgplearn.backend.c_torch import c_torch_cal, c_torch_score, c_torch_score_mp
    backend.update({"c_torch":(func_names, p_np_str_name,c_torch_score_mp, c_torch_score, c_torch_cal),})
except BaseException:
    pass




