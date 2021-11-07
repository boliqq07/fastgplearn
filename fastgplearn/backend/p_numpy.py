# -*- coding: utf-8 -*-

# @Time     : 2021/11/1 11:08
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import functools
import warnings
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

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
    return np.power(a, 3)


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


# def get_corr_single(fake_y, y):
#     fake_y_mean = np.mean(fake_y)
#     y_mean = np.mean(y)
#     corr = (np.sum((fake_y - fake_y_mean) * (y - y_mean))) / (
#             np.sqrt(np.sum(np.power((fake_y - fake_y_mean), 2))) * np.sqrt(np.sum(np.power((y - y_mean), 2))))
#     return corr


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

    return np.abs(np.nan_to_num(corr, posinf=0, neginf=0))


# def get_r2_together(fake_ys, y):
#     """
#     Now not use
#
#
#     Args:
#         fake_ys (np.ndarray): with shape (n_results, n_sample,).
#         y (np.ndarray): with sample (n_sample,).
#
#     Returns:
#         corr (np.ndarray): with shape (n_result,)
#
#     """
#
#     # 转置不转置速度相同
#     y_pred = fake_ys
#     y_true = y
#
#     numerator = ((y_true - y_pred) ** 2).sum(axis=1,dtype=np.float32)
#     denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0,
#                                                           dtype=np.float32)
#
#     r2 = 1 - numerator/denominator
#
#     return np.nan_to_num(r2,nan=0, posinf=0, neginf=0)


# def get_accuracy_together(fake_ys, y):
#     """
#
#     Args:
#         fake_ys (np.ndarray): with shape (n_results, n_sample,).
#         y (np.ndarray): with sample (n_sample,).
#
#     Returns:
#         corr (np.ndarray): with shape (n_result,)
#
#     """
#     # 转置不转置速度相同
#
#     fake_ys = 1.0 / (1.0 + np.exp(-fake_ys))
#
#     fake_ys = np.nan_to_num(fake_ys, nan=np.nan, posinf=np.nan, neginf=np.nan)
#
#     fake_ys[fake_ys >= 0.5] = 1
#     fake_ys[fake_ys < 0.5] = 0
#
#     shape = y.shape[0]
#     dis = np.abs(fake_ys - y.reshape(-1, 1))
#     scores = (shape - np.sum(dis, axis=1)) / shape
#
#     return np.nan_to_num(scores, 0, 0, 0, )


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

    acc1 = 1 - np.mean(np.abs(y_pre_sort - y_sort), axis=1)
    acc2 = 1 - np.mean(np.abs(y_pre_sort - y_sort2), axis=1)

    score = np.max(np.concatenate((acc1.reshape(1, -1), acc2.reshape(1, -1)), axis=0), axis=0)
    score[mark] = 0.0

    return score


def p_np_cal(ve, xs, y, func_index=None):
    """Batch calculate."""
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


def p_np_score(ve, xs, y, func_index, clf=False):
    """Batch score."""
    if isinstance(ve, np.ndarray):
        ve = ve.tolist()
    rs6 = p_np_cal(ve, xs, y, func_index)
    rs7 = np.array(rs6)
    if not clf:
        return get_corr_together(rs7, y)
    else:
        return get_sort_accuracy_together(rs7, y)


def p_np_score_mp(ve, xs, y, func_index=None, n_jobs=1, clf=False):
    """Batch score with n_jobs."""
    if isinstance(ve, np.ndarray):
        ve = ve.tolist()
    if n_jobs == 1:
        return p_np_score(ve, xs, y, func_index, clf=clf)
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

        func = functools.partial(p_np_score, xs=tuple(xs), y=y, func_index=func_index, clf=clf)

        res = [i for i in pool.map_async(func, nve).get()]
        pool.close()
        pool.join()

        res = np.concatenate(res)

        return res


def p_np_str_name(ve, xns, cns=None, y=None, func_index=None, real_names=None):
    """Batch get name of expression,(without coef and intercept)."""
    _ = y
    func_namesi = [func_names[i] for i in func_index] if func_index is not None else func_names

    if real_names is None:
        xns = [f"x{i}" for i in range(len(xns))]
    else:
        assert len(xns) == real_names
        xns = real_names

    if cns is not None:
        cns = ["{:+.2f}".format(i) for i in cns]
        xns.extend(cns)

    def get_str(vei, n):

        root = vei[0]
        if vei[n] >= 100:
            return xns[vei[n] - 100]
        elif 2 * n >= len(vei):
            return xns[vei[n] - 100]
        else:
            return f"{func_namesi[vei[n]]}({get_str(vei, 2 * n + 1 - root)},{get_str(vei, 2 * n + 2 - root)})"

    res = []
    for vei in ve:
        try:
            resi = get_str(vei, vei[0])
        except TypeError:
            resi = ""
        res.append(resi)

    return res


lr = LinearRegression(fit_intercept=True)
lg = LogisticRegression(fit_intercept=True, penalty='none', dual=False, tol=1e-4, intercept_scaling=1,
                        class_weight=None, random_state=0, solver='lbfgs', max_iter=100,
                        multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
