from decimal import Decimal
import labtool as lt

func = lt.src.monkeypatch_uncertainties._digits_exponent_std_dev

print(func(Decimal(1e-14)))  # type: ignore
print(Decimal("1e-14"))
