import contextlib
import sys
with contextlib.nullcontext():
    sys.path.append("../src")
    import core  # type:ignore
