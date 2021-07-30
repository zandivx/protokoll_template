import core
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import inspect
import uncertainties as u
import uncertainties.unumpy as unp
import sys
import cProfile
import pstats

# core.plt_latex()
# core.chdir()

#plt.plot([i for i in range(10)], [i**2 for i in range(10)])
#plt.title("LaTeX test ß €")
# plt.show()

# print(os.getcwd())
# with core.cdContextManager("C:/Users/andre"):
#     print(os.getcwd())
#     print(core.Student.t_df)
# print(os.getcwd())


# def profile(func):
#     def wrapper(*args, **kwargs):
#         with cProfile.Profile() as pr:
#             func(*args, **kwargs)
#         stats = pstats.Stats(pr)
#         stats.sort_stats(pstats.SortKey.TIME)
#         stats.dump_stats("profiling.snakeviz")  # for snakeviz
#     return wrapper


lst_n = [i/1.1 for i in range(10, 30)]
lst_s = [i/11 for i in range(10, 30)]


# @profile
# def test():
#     var = unp.core.uncert_core.Variable(12.8, 0.23)
#     uarr = unp.uarray([12.8], [.23])
#     print(var)
#     print(uarr)


# @profile
# def test2():
#     vec_func = (np.vectorize(
#         lambda v, s: unp.core.uncert_core.Variable(v, s), otypes=[object]))
#     vec_func_2 = (np.vectorize(
#         lambda v, s: u.ufloat(v, s), otypes=[object]))

#     uarr = vec_func_2(lst_n, lst_s)
#     print(uarr)
#     print(type(vec_func_2))


# @profile
def test3():
    print(unp.uarray(lst_n, lst_s))


# @profile
# def test4():
#     #print(map(lambda v, s: unp.core.uncert_core.Variable(v, s), zip(lst_n, lst_s)))
#     pass


# @profile
# def test5():
#     lst = []
#     for n, s in zip(lst_n, lst_s):
#         lst.append(u.ufloat(n, s))
#     print(lst)

test3()
