from monkeypatch import MonkeyPatch
import uncertainties as u

MonkeyPatch().uncertainties_rounding(msg=True)

print(u.ufloat(3.2, 0.25))
