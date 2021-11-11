# -*- coding: utf-8 -*-

# @Time     : 2021/11/1 9:20
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os

import numpy as np
from numpy.random import choice
from numpy import random as nprdm


try:
    from fastgplearn.backend.c_numpy import c_numpy_backend_tool

    csci_subs = c_numpy_backend_tool.sci_subs
except BaseException:
    csci_subs = None


def set_seed(seed):
    """Set random seed."""
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except BaseException:
        pass
    try:
        from torch.backends import cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except BaseException:
        pass


def select_index(score, num_percent=0.3, method="tournament", tour_num=3):
    """
    Selection.

    Args:
        score (np.ndarray): score with shape (n_res,)/
        num_percent (int,float): number or percent of population.
        method (str):  "tournament" or "k_best".
        tour_num (int): tournament size .

    Returns:
        index (np.ndarray): index of selection to population.
    """

    size = score.shape[0]
    num_percent = num_percent if num_percent > 1 else int(size * num_percent)

    if method == "k_best":
        return np.argsort(score)[-num_percent:][::-1]
    elif method == "tournament":
        indexx = nprdm.choice(np.arange(size), (tour_num, num_percent))
        scores = score[indexx]
        temp_index = np.argmax(scores, axis=0)[::-1]
        index = indexx[temp_index, np.arange(num_percent)]
        return index
    else:
        raise NotImplementedError


def generate_random(func_num, xs_num, pop_size=10, depth_min=1, depth_max=5, p=None, func_p=None, xs_p=None):
    """
    Generate the first population.
    Each individual with ordered: [mark,1,2,3,4,5,6,7, 103,102,100,102,102,103,102,100]
    1.First part: mark index of root.
    2.second part: index of x gene and f gen.
    3.Third part: protect index of x gene.

    Args:
        func_num (int): func number.
        xs_num (int): x number (n_fea).
        pop_size (int): population size.
        depth_min (int): min depth of expression.
        depth_max (max): max depth of expression.
        p (None): (just for test).
        func_p (np.ndarray): with shape of (n_func), probability,.
        xs_p (np.ndarray): with shape of (n_fea), probability.

    Returns:
        pop (np.ndarray):with shape (n_pop,2**depth_max), population .

    """

    assert depth_max <= 8 and xs_num < 125 and depth_min > 0, "Out of limitation !!!"

    if p is None:
        func_p = np.full(func_num, 0.8 / func_num) if func_p is None else func_p
        xs_p = np.full(xs_num, 0.2 / xs_num) if xs_p is None else xs_p
        p = np.append(func_p, xs_p)

    deps = choice(np.append(np.arange(func_num, dtype=np.uint8), np.arange(100, 100 + xs_num, dtype=np.uint8)),
                  size=(pop_size, 2 ** depth_max), replace=True, p=p)

    deps2 = choice(np.arange(100, 100 + xs_num, dtype=np.uint8), size=(pop_size, 2 ** (depth_max - 1)), replace=True,
                   p=xs_p / np.sum(xs_p))
    deps[:, int(deps.shape[1] / 2):] = deps2

    marks = 1 + 2 ** (depth_max - 1) - 2 ** np.arange(depth_min, depth_max, dtype=np.uint8)
    mark = choice(marks, size=pop_size, replace=True)
    deps[:, 0] = mark

    return deps


def mutate(mutate_pop, func_num, xs_num, depth_min=1, depth_max=5, p_mutate=0.8, p=None, func_p=None, xs_p=None):
    """
    Mutate.
    Each individual with ordered: [mark,1,2,3,4,5,6,7, 103,102,100,102,102,103,102,100]
    1.First part: mark index of root.
    2.second part: index of x gene and f gen.
    3.Third part: protect index of x gene.

    Args:
        func_num (int): func number.
        mutate_pop (np.ndarray): with shape (n_pop,2**depth_max),population.
        xs_num (int): x number (n_fea).
        depth_min (int): min depth of expression.
        depth_max (max): max depth of expression.
        p (None): (just for test).
        func_p (np.ndarray): with shape of (n_func), probability,.
        xs_p (np.ndarray): with shape of (n_fea), probability.
        p_mutate (flaot): probability for mutate.

    Returns:
        pop (np.ndarray): population with shape (n_pop,2**depth_max).

    """
    pop_n, prim_n = mutate_pop.shape
    pii = 1 - (1 - p_mutate) ** (1 / prim_n)
    sub_pop = generate_random(func_num, xs_num, pop_size=pop_n, depth_min=depth_min, depth_max=depth_max,
                              p=p, func_p=func_p, xs_p=xs_p)
    select_mark = choice((0, 1), size=(pop_n, prim_n), replace=True, p=(1 - pii, pii))
    mutate_pop = mutate_pop * (1 - select_mark) + sub_pop * select_mark
    return mutate_pop


def csub_science(pop, sci_template):
    """
    This would change the init pop!!!

    pyx version for sci substitute.
    """
    now_pop_n, prim_n = pop.shape

    le_scis = len(sci_template)

    def get_real_index(n, prei=-1):
        s, v = sci_template[prei]

        root = pop[n, 0]
        pop[n, root] = v[0]
        csci_subs(pop, s, v, root, n)

    index = nprdm.randint(0, high=le_scis, size=now_pop_n)

    for n, i in enumerate(index):
        get_real_index(n=n, prei=i)

    return pop


def sub_science(pop, sci_template):
    """
    This would change the init pop!!!
    sci substitute.
    """
    now_pop_n, prim_n = pop.shape
    half_prim_n = prim_n / 2

    le_scis = len(sci_template)

    def get_real_index(n, prei=-1):
        s, v = sci_template[prei]

        root = pop[n, 0]
        pop[n, root] = v[0]
        m = 0.5
        for si, vi in zip(s[1:], v[1:]):
            m = -m
            site = int(2 * si + m + 1.5 + root)
            if vi != -1 and site < half_prim_n:
                pop[n, 2 * si + int(m + 1.5) + root] = vi

    index = nprdm.randint(0, high=le_scis, size=now_pop_n)

    for n, i in enumerate(index):
        get_real_index(n=n, prei=i)

    return pop


def mutate_random(pop_np, func_num, xs_num, pop_size=10, depth_min=1, depth_max=5, p_mutate=0.8, p=None, func_p=None,
                  xs_p=None):
    """
    Mutate.
    Each individual with ordered: [mark,1,2,3,4,5,6,7, 103,102,100,102,102,103,102,100]
    1.First part: mark index of root.
    2.second part: index of x gene and f gen.
    3.Third part: protect index of x gene.

    Args:
        func_num (int): func number.
        pop_size (int): population size.
        pop_np (np.ndarray): with shape (n_pop,2**depth_max),population.
        xs_num (int): x number (n_fea).
        depth_min (int): min depth of expression.
        depth_max (max): max depth of expression.
        p (None): (just for test).
        func_p (np.ndarray): with shape of (n_func), probability,.
        xs_p (np.ndarray): with shape of (n_fea), probability.
        p_mutate (flaot): probability for mutate.

    Returns:
        pop (np.ndarray): population with shape (n_pop,2**depth_max).

    """
    _ = pop_size

    now_pop_n, prim_n = pop_np.shape
    index = np.arange(now_pop_n)
    nprdm.shuffle(index)
    mutate_pop = pop_np[index]

    mutate_pop = mutate(mutate_pop, func_num, xs_num, depth_min=depth_min, depth_max=depth_max, p_mutate=p_mutate,
                        p=p, func_p=func_p, xs_p=xs_p)

    return mutate_pop


def mutate_sci(func_num, xs_num, pop_size=10, depth_min=1, depth_max=5, p=None, func_p=None,
               xs_p=None, sci_template=None):
    """
    Mutate.
    Each individual with ordered: [mark,1,2,3,4,5,6,7, 103,102,100,102,102,103,102,100]
    1.First part: mark index of root.
    2.second part: index of x gene and f gen.
    3.Third part: protect index of x gene.

    Args:
        func_num (int): func number.
        pop_size (int): population size.

        xs_num (int): x number (n_fea).
        depth_min (int): min depth of expression.
        depth_max (max): max depth of expression.
        p (None): (just for test).
        func_p (np.ndarray): with shape of (n_func), probability,.
        xs_p (np.ndarray): with shape of (n_fea), probability.
        sci_template (list of list): the science expression templates.

    Returns:
        pop (np.ndarray): population with shape (n_pop,2**depth_max).

    """
    new_pop = generate_random(func_num, xs_num, pop_size=pop_size, depth_min=depth_min, depth_max=depth_max,
                              p=p, func_p=func_p, xs_p=xs_p)
    if sci_template is None or sci_template == []:
        pass
    else:

        new_pop = sub_science(new_pop, sci_template)
        # if csci_subs is None:
        #     new_pop = sub_science(new_pop, sci_template)
        # else:
        #     new_pop = csub_science(new_pop, sci_template)

    return new_pop


def crossover(pop_np, p_crossover=0.5):
    """
    Corssover.

    Args:
        pop_np (np.ndarray): population
        p_crossover (float): probability for crossover.

    Returns:
        pop (np.ndarray): population with shape (n_pop,2**depth_max).

    """
    now_pop_n, prim_n = pop_np.shape

    whe = int(prim_n / 2)
    need_num = int(now_pop_n * p_crossover / 2)
    need_num2 = need_num * 2

    pop_np[need_num:need_num2, :whe], pop_np[0:need_num, :whe] = pop_np[0:need_num:, :whe], pop_np[need_num:need_num2,
                                                                                            :whe]
    return pop_np
