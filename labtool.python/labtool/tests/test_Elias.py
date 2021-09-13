# type: ignore
import labtool as lt
lt.monkeypatch_uncertainties.init()
lt.monkeypatch_uncertainties.display()

ufloat = lt.u.ufloat

x = ufloat(1.23987, 0)
y = ufloat(2.852, 0)
print(x+y)  # should be 4.092 | but prints 4.1
# yields same result
x = ufloat(12398.7e-4, 0)
y = ufloat(285.2e-2, 0)
print(x+y)  # should be 4.092 | but prints 4.1

i = ufloat(1.23, 0)
j = ufloat(1.45, 0)
print(i*j)  # ! should be 1.78 | but prints 1.68
