# std library
import os
import sys
import cProfile
import pstats

# 3rd party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties as u
import uncertainties.unumpy as unp

# own
from labtool_ import labtool as lt

# lt.plt_latex()
# lt.chdir()

#plt.plot([i for i in range(10)], [i**2 for i in range(10)])
#plt.title("LaTeX test ß €")
# plt.show()

# print(os.getcwd())
# with lt.cdContextManager("C:/Users/andre"):
#     print(os.getcwd())
#     print(lt.Student.t_df)
# print(os.getcwd())


def profile(func):
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            func(*args, **kwargs)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats("profiling.snakeviz")  # for snakeviz
    return wrapper


lst_n = [i/1.1 for i in range(10, 30)]
lst_s = [i/11 for i in range(10, 30)]


@profile
def test():
    var = unp.core.uncert_core.Variable(12.8, 0.23)
    uarr = unp.uarray([12.8], [.23])
    print(var)
    print(uarr)


@profile
def test2():
    vec_func = (np.vectorize(
        lambda v, s: unp.core.uncert_core.Variable(v, s), otypes=[object]))
    vec_func_2 = (np.vectorize(
        lambda v, s: u.ufloat(v, s), otypes=[object]))
    uarr = vec_func_2(lst_n, lst_s)
    print(uarr)
    print(type(vec_func_2))


@profile
def test3():
    uarr = unp.uarray(lst_n, lst_s)
    print(uarr)
    print(type(uarr))
    return np.asarray(uarr, dtype=object)


@profile
def test4():
    #print(map(lambda v, s: unp.core.uncert_core.Variable(v, s), zip(lst_n, lst_s)))
    pass


@profile
def test5():
    lst = []
    for n, s in zip(lst_n, lst_s):
        lst.append(u.ufloat(n, s))
    print(lst)


def test6():
    uarr = lt.unp.uarray(lst_n, lst_s)
    lst = [f"{float(x.n)}+/-{float(x.s)}" for x in uarr]
    print(f"original:\n{uarr}\n\ncomprehension:\n{lst}")


def test7():
    uf = lt.u.ufloat(3, 2.2)
    print(uf)


test7()

print(lt.Student.t_df)
