# std library
import os

# 3rd party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties as u
import uncertainties.unumpy as unp

# own
import labtool as lt


def test0():
    lt.plt_latex()
    lt.cd()
    lt.plt.plot(list(range(10)), [i**2 for i in range(10)])
    lt.plt.title("LaTeX test ß €")
    plt.show()
    print(os.getcwd())
    with lt.CDContxt("C:/Users/andre"):
        print(os.getcwd())
        print(lt.Student.t_df)
    print(os.getcwd())


lst_n = [i/1.1 for i in range(10, 30)]
lst_s = [i/11 for i in range(10, 30)]


@lt.profile
def test1():
    var = unp.core.uncert_core.Variable(12.8, 0.23)
    uarr = unp.uarray([12.8], [.23])
    print(var)
    print(uarr)


@lt.profile
def test2():
    vec_func = (np.vectorize(
        lambda v, s: unp.core.uncert_core.Variable(v, s), otypes=[object]))
    vec_func_2 = (np.vectorize(
        lambda v, s: u.ufloat(v, s), otypes=[object]))
    uarr = vec_func_2(lst_n, lst_s)
    print(uarr)
    print(type(vec_func_2))


@lt.profile
def test3():
    uarr = unp.uarray(lst_n, lst_s)
    print(uarr)
    print(type(uarr))
    return np.asarray(uarr, dtype=object)


@lt.profile
def test4():
    print(map(lambda v, s: unp.core.uncert_core.Variable(
        v, s), zip(lst_n, lst_s)))  # type: ignore


@lt.profile
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


def test8():
    x = np.linspace(-np.pi, np.pi, 1000)
    y = np.sin(x**2)

    fit = lt.Interpolation(x, y)
    print(fit)
    fit.plot(style_in="o", style_out=".--")


def test9():
    print(lt.Student.t_df)
    st = lt.Student(list(range(50)))
    print(f"t: {st.t}\n{st.mean}")
    print(st)


def test10():
    x = lt.np.linspace(-lt.np.pi, lt.np.pi, 1000)
    y = lt.np.sin(2*x)
    def func(x, a): return lt.np.sin(a*x)
    fit = lt.CurveFit(func, x, y)

    # fit.plot(style_in=".")
    print(fit)


def test11():
    x = lt.u.ufloat(1, 3)
    y = lt.u.ufloat(1, 1.2e-4)
    print(x*y)  # type: ignore
    print(lt.u.ufloat(3, 0.24))


def test12():
    uf = lt.u.ufloat(2.978, 0.2)
    print(uf.format(".5f"))  # type: ignore


def test13():
    print(lt.u.ufloat(2245345743767e-7, 5345234563456e-9))


if __name__ == "__main__":
    test13()
