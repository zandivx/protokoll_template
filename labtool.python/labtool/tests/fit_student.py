# type: ignore[attr-defined]
from labtool_ import labtool as lt


df = lt.Student.t_values
def func2(x, a, d, n): return a/x**n + d


student = lt.Fit(func2, df["N"], df["1"], bounds=(
    (1, 0, 1), (lt.np.inf, 100, 4)))
print(student)
student.plot()
