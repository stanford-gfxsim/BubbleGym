"""Centralised thread caps for bempp-cl / BLAS / OpenCL CPU backends."""

from __future__ import annotations

import os


def limit_threads(n: int, *, set_bempp_cpu_workers: bool = True) -> int:
    """
    Cap parallel threads **before** importing bempp-cl or heavy linear algebra.

    Returns the clamped thread count actually applied.
    """
    n_clamped = max(1, int(n))
    s = str(n_clamped)
    for var in (
        "BEMPP_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = s
    os.environ.setdefault("POCL_MAX_PTHREAD_COUNT", s)
    os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "0")

    if set_bempp_cpu_workers:
        try:
            from .bempp_io import _import_bempp_api

            api = _import_bempp_api()
            gp = getattr(api, "GLOBAL_PARAMETERS", None)
            asm = getattr(gp, "assembly", None) if gp is not None else None
            if asm is not None and hasattr(asm, "cpu_workers"):
                asm.cpu_workers = n_clamped
        except Exception:
            pass
    return n_clamped


def default_thread_count() -> int:
    """``max(1, cpu_count - 2)`` by default, leaving two cores free for the OS."""
    try:
        ncpu = os.cpu_count() or 4
    except Exception:
        ncpu = 4
    return max(1, ncpu - 2)


__all__ = ["limit_threads", "default_thread_count"]
