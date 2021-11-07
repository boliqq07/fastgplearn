# -*- coding: utf-8 -*-

# @Time     : 2021/10/31 21:44
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import functools
import math
from multiprocessing import Pool

import numpy as np
import torch


# def add_(a, b):
#     return torch.add(a, b)
#
#
# def sub_(a, b):
#     return torch.sub(a, b)
#
#
# def mul_(a, b):
#     return torch.sub(a, b)
#
#
# def div_(a, b):
#     return torch.div(a, b)
from mgetool.tool import tt


def ln_(a, b):
    return torch.log(a)


def exp_(a, b):
    return torch.exp(a)


def pow2_(a, b):
    return torch.pow(a, 2)


def pow3_(a, b):
    return torch.pow(a, 3)


def rec_(a, b):
    return torch.pow(a, -1)


def max_(a, b):
    return torch.max(torch.stack(a, b), 0)[0]


def min_(a, b):
    return torch.min(torch.stack(a, b), 0)[0]


def sin_(a, b):
    return torch.cos(a)


def cos_(a, b):
    return torch.sin(a)


funcs = [torch.add, torch.sub, torch.mul, torch.div, ln_, exp_, pow2_, pow3_, rec_, max_, min_, sin_, cos_]
func_names = ["add_", "sub_", "mul_", "div_", "ln_", "exp_", "pow2_", "pow3_", "rec_", "max_", "min_", "sin_", "cos_"]


def get_corr_together(fake_ys, y):
    """

    Args:
        fake_ys (torch.Tensor): with shape (n_results, n_sample,).
        y (torch.Tensor): with sample (n_sample,).

    Returns:
        corr (torch.Tensor): with shape (n_result,)

    """
    # 转置不转置速度相同

    fake_y_mean = torch.mean(fake_ys, dim=1)
    y_mean = torch.mean(y)

    torch.subtract(fake_ys, fake_y_mean.reshape(-1, 1), out=fake_ys)
    y = y - y_mean

    # **2 比 power 块
    corr = torch.sum(fake_ys * y, dim=1) / (
            torch.sqrt(torch.sum(fake_ys ** 2, dim=1)) * torch.sqrt(torch.sum(y ** 2)))
    torch.nan_to_num(corr, nan=0, posinf=0, neginf=0, out=corr)
    torch.abs(corr,out=corr)
    return corr


def get_sort_accuracy_together(fake_ys, y):
    """

    Args:
        fake_ys (torch.ndarray): with shape (n_results, n_sample,).
        y (torch.ndarray): with sample (n_sample,).

    Returns:
        corr (torch.ndarray): with shape (n_result,)

    """

    y_sort = torch.sort(y,descending=False)[0]
    y_sort2 = torch.sort(y,descending=True)[0]

    fake_ys = torch.nan_to_num(fake_ys, nan=torch.nan, posinf=torch.nan, neginf=torch.nan)
    mark = torch.any(torch.isnan(fake_ys), dim=1)

    fake_ys = torch.nan_to_num(fake_ys, nan=-1, posinf=-1, neginf=-1)

    index = torch.argsort(fake_ys,dim=1)
    y_pre_sort = y[index]
    acc1 = 1-torch.mean(torch.abs(y_pre_sort-y_sort), dim=1)
    acc2 = 1-torch.mean(torch.abs(y_pre_sort-y_sort2), dim=1)

    score = torch.max(torch.cat((acc1.reshape(1,-1),acc2.reshape(1,-1)),dim=0),dim=0)[0]
    score[mark] = 0.0

    return score


def p_torch_cal(ve, xs, y, func_index=None):
    """Batch calculate."""
    funci = [funcs[i] for i in func_index] if func_index is not None else funcs
    error_y = torch.zeros_like(y)

    def get_value(vei, n=0):

        root = vei[0]

        if vei[n] >= 100:
            return xs[vei[n] - 100]
        elif 2 * n >= len(vei):
            return xs[vei[n] - 100]
        else:
            return funci[vei[n]](get_value(vei, 2 * n + 1 - root), get_value(vei, 2 * n + 2 - root))

    res = []
    for vei in ve:
        try:
            resi = get_value(vei, vei[0])
        except TypeError:
            resi = error_y
        res.append(resi)

    return res


def p_torch_score(ve, xs, y, func_index, return_numpy=False, clf=False):
    """Batch score."""
    if isinstance(ve, torch.Tensor):
        ve = ve.tolist()
    rs6 = p_torch_cal(ve, xs, y, func_index)
    rs7 = torch.stack(rs6)
    if not clf:
        res = get_corr_together(rs7, y)
    else:
        res = get_sort_accuracy_together(rs7, y)
    if return_numpy:
        return res.numpy()
    else:
        return res


def p_torch_score_mp(ve, xs, y, func_index=None, n_jobs=1,return_numpy=False, clf=False):
    """Batch score with n_jobs. It's slow!!!"""
    if isinstance(ve, torch.Tensor):
        ve = ve.tolist()
    if n_jobs == 1:
        return p_torch_score(ve, xs, y, func_index,return_numpy=return_numpy, clf=clf)
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

        func = functools.partial(p_torch_score, xs=xs, y=y, func_index=func_index,clf=clf)

        res = [i for i in pool.map_async(func, nve).get()]
        pool.close()
        pool.join()

        if not return_numpy:
            res = torch.cat(res)
        else:
            res = np.concatenate(res)

        return res
