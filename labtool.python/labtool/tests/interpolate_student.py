# type: ignore[attr-defined]
from numpy import dtype
import labtool as lt
from scipy.interpolate import interp1d, splrep, splev

df = lt.Student._t_df_old


def fit(sigma: str):
    def func2(x, n, a): return a/x**n + 1
    student = lt.CurveFit(func2, df["N"], df[sigma], bounds=(
        (1, 0), (2, 10)))
    print(student)
    student.plot(xlim=(0, 30))


def interpolate(sigma: str):
    func_1 = interp1d(df["N"], df[sigma])
    func_2 = interp1d(df["N"], df[sigma], kind="cubic")

    x = lt.np.arange(2, 200).reshape(-1, 1)
    y_1 = lt.np.round(func_1(x), 3)
    y_2 = lt.np.round(func_2(x), 3)

    dataf_1 = lt.pd.DataFrame(y_1, index=x.astype("int64").flatten())
    dataf_2 = lt.pd.DataFrame(y_2, index=x.astype("int64").flatten())

    dataf_1.to_csv(f"interpolation_{sigma}_linear.txt")
    dataf_1.to_csv(
        f"interpolation_{sigma}_linear-copy.txt", line_terminator=", ", index=False)

    dataf_2.to_csv(f"interpolation_{sigma}_cubic.txt")
    dataf_2.to_csv(
        f"interpolation_{sigma}_cubic-copy.txt", line_terminator=", ", index=False)

    lt.plt.plot(df["N"], df[sigma], "o-", label="input")
    lt.plt.plot(x, y_1, ".-", label="linear")
    lt.plt.plot(x, y_2, ".--", label="cubic")
    lt.plt.legend()
    lt.plt.show()


def spline(sigma: str):
    tck = splrep(df["N"], df[sigma], s=0)
    x = lt.np.arange(2, 200)
    y = lt.np.round(splev(x, tck, der=0), 3)

    data = lt.pd.DataFrame(y)

    data.to_csv(f"spline_{sigma}.txt", line_terminator=", ", index=False)

    lt.plt.plot(df["N"], df[sigma], "o", x, y, ".--")
    lt.plt.legend(["input", "spline"])
    lt.plt.show()


interpolate("1")
spline("1")
fit("1")
