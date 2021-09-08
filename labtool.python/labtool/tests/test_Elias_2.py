# type: ignore
import sys
import labtool as lt
from uncertainties import ufloat
init = lt.monkeypatch_uncertainties.init
display = lt.monkeypatch_uncertainties.display
undo = lt.monkeypatch_uncertainties.undo


def test(_0):
    x = ufloat(1.23987, 0 if _0 else 2e-5)
    y = ufloat(2.852, 0 if _0 else 2e-3)
    print(f"{x} + {y} = {x+y}")

    # yields same result
    a = ufloat(12398.7e-4, 0 if _0 else 2e-5)
    b = ufloat(285.2e-2, 0 if _0 else 2e-3)
    print(f"{a} + {b} = {a+b}")

    i = ufloat(1.23, 0 if _0 else 2e-2)
    j = ufloat(1.45, 0 if _0 else 2e-2)
    print(f"{i} * {j} = {i*j}")


def tests(undo=False, _0=True):
    print("nothing:")
    test(_0)

    # apply init
    init()
    print("\ninit:")
    test(_0)

    # apply display
    display()
    print("\ndisplay:")
    test(_0)

    # undo
    if undo:
        undo()
        print("\nundo:")
        test(_0)


tests(_0=False)
