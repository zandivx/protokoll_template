import contextlib
import sys
with contextlib.nullcontext():
    sys.path.append("../..")
    import labtool  # type: ignore

del contextlib
del sys
