# cython: language_level=3
# -*- coding: utf-8 -*-

# @Time     : 2021/11/1 11:08
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

import warnings

import numpy as np

warnings.filterwarnings("ignore")


# def add_(a, b):
#     return np.add(a, b)
#
#
# def sub_(a, b):
#
#     return np.subtract(a, b)
#
#
# def mul_(a, b):
#     return np.multiply(a, b)
#
#
# def div_(a, b):
#     return np.divide(a, b)


def ln_(a, b):
    return np.log(a)


def exp_(a, b):
    return np.exp(a)


def pow2_(a, b):
    return a ** 2


def pow3_(a, b):
    return a ** 3


def rec_(a, b):
    return 1 / a


def max_(a, b):
    return np.max(np.stack((a, b)), 0)


def min_(a, b):
    return np.min(np.stack((a, b)), 0)


def sin_(a, b):
    return np.cos(a)


def cos_(a, b):
    return np.sin(a)


funcs = [np.add, np.subtract, np.multiply, np.divide, ln_, exp_, pow2_, pow3_, rec_, max_, min_, sin_, cos_]
func_names = ["add_", "sub_", "mul_", "div_", "ln_", "exp_", "pow2_", "pow3_", "rec_", "max_", "min_", "sin_", "cos_"]


def get_corr_together(fake_ys, y):
    """


    Args:
        fake_ys (np.ndarray): with shape (n_results, n_sample,).
        y (np.ndarray): with sample (n_sample,).

    Returns:
        corr (np.ndarray): with shape (n_result,)

    """
    # 转置不转置速度相同

    fake_y_mean = np.mean(fake_ys, axis=1)
    y_mean = np.mean(y)

    np.subtract(fake_ys, fake_y_mean.reshape(-1, 1), out=fake_ys)
    y = y - y_mean

    # **2 比 power 块
    corr = np.sum(fake_ys * y, axis=1) / (
            np.sqrt(np.sum(fake_ys ** 2, axis=1)) * np.sqrt(np.sum(y ** 2)))

    return np.abs(np.nan_to_num(corr,posinf=0, neginf=0))


def c_np_cal(ve, xs, y, func_index=None):
    """Batch calculation."""
    funci = [funcs[i] for i in func_index] if func_index is not None else funcs
    error_y = np.zeros_like(y)

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

    # res = [get_value(vei, vei[0]) for vei in ve]

    return res


def get_sort_accuracy_together(fake_ys, y):
    """

    Args:
        fake_ys (np.ndarray): with shape (n_results, n_sample,).
        y (np.ndarray): with sample (n_sample,).

    Returns:
        corr (np.ndarray): with shape (n_result,)

    """

    y_sort = np.sort(y)
    y_sort2 = np.sort(y)[::-1]

    fake_ys = np.nan_to_num(fake_ys, nan=np.nan, posinf=np.nan, neginf=np.nan)
    mark = np.any(np.isnan(fake_ys), axis=1)

    fake_ys = np.nan_to_num(fake_ys, nan=-1, posinf=-1, neginf=-1)

    index = np.argsort(fake_ys, axis=1)
    y_pre_sort = y[index]

    acc1 = 1-np.mean(np.abs(y_pre_sort-y_sort), axis=1)
    acc2 = 1-np.mean(np.abs(y_pre_sort-y_sort2), axis=1)

    score = np.max(np.concatenate((acc1.reshape(1,-1),acc2.reshape(1,-1)),axis=0),axis=0)
    score[mark] = 0.0

    return score


def c_np_score(ve, xs, y, func_index,clf=False):
    """Batch score."""
    if isinstance(ve, np.ndarray):
        ve = ve.tolist()
    rs6 = c_np_cal(ve, xs, y, func_index)
    rs7 = np.array(rs6)
    if not clf:
        return get_corr_together(rs7, y)
    else:
        return get_sort_accuracy_together(rs7, y)


cpdef sci_subs(int [:,:] pop,list s,list v,int root,int n,int half_prim_n):
    # """sci subs (pyx version)."""
    cdef float m = 0.5
    cdef int ind
    cdef int site
    for si, vi in zip(s[1:], v[1:]):
        m = -m
        site = int(2 * si + m + 1.5 + root)
        if site >= half_prim_n:
            break
        elif vi != -1:
            ind = int(2 * si + m + 1.5 + root)
            pop[n, ind] = vi


