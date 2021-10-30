import labtool as lt


def fit():
    def func(x, a, b, c): return a * lt.np.exp(-b * x) + c

    xdata = lt.np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    rng = lt.np.random.default_rng()
    y_noise = 0.2 * rng.normal(size=xdata.size)
    ydata = y + y_noise
    #lt.plt.plot(xdata, ydata, 'b-', label='data')

    fit = lt.CurveFit(func, xdata, ydata)
    print(fit)
    fit.save("saves/fit.csv")
    fit.plot()


def student():
    data = [28.89, 28.85, 28.92, 28.93, 28.98, 28.90, 28.85, 28.98, 28.88,
            28.91, 28.84, 28.86, 28.90, 28.87, 28.86, 28.91, 28.93, 28.86, 28.89, 28.89]

    series = lt.Student(data, 1)

    print(series)
    series.save("saves/series.csv")


def interpolate():
    data = [28.89, 28.85, 28.92, 28.93, 28.98, 28.90, 28.85, 28.98, 28.88,
            28.91, 28.84, 28.86, 28.90, 28.87, 28.86, 28.91, 28.93, 28.86, 28.89, 28.89]

    interp = lt.Interpolation(range(len(data)), data)
    print(interp)
    interp.save("saves/interp.csv")
    interp.plot(style_in="o")


def student_array():
    lst = []
    for i in range(5):
        if i == 0:
            x = lt.pd.read_csv(f"data/{i}.csv", names=["x", "0"]).iloc[:, 0]
        lst.append(lt.pd.read_csv(
            f"data/{i}.csv", names=["x", f"{i}"]).iloc[:, 1])

    sa = lt.StudentArray(lst)
    print(lt.np.array(sa))
    print(sa.n)
    lt.plt_uplot(x, sa, band=True)  # type: ignore
    lt.plt.show()


if __name__ == "__main__":
    student_array()
