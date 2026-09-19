"""
Mixed Galerkin BEM for the Minnaert-frequency / capacitance problem (bempp-cl).

Exterior Laplace Dirichlet on a closed bubble surface Gamma whose normal points
from the bubble into the fluid: ``Delta phi = 0`` outside Gamma,
``phi|_Gamma = 1``, ``phi -> 0`` at infinity. In bempp-cl's sign convention the
direct boundary integral equation is ``V q = (K - 1/2 I) phi`` on Gamma, and the
capacitance ``C = -(1/(4 pi)) int_Gamma q dS`` comes out positive for an
outward-oriented surface with no absolute-value fix-up.

``trial_pair`` picks the discretization: ``"P1-DP0"`` (conforming, the default),
``"DP0-DP0"``, or ``"P1-DP1"``. Batch drivers must call
:func:`python.bem._threading.limit_threads` before bempp-cl is imported --
unbounded MKL/OpenCL threads can freeze the machine.
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from .geometry_utils import _flip_faces
from .bempp_io import (
    _compute_volume,
    _import_bempp_api,
    _load_triangle_mesh,
    _make_bempp_grid,
    _minnaert_constant,
    _signed_volume,
)

MeshLike = Union[str, Path, Tuple[np.ndarray, np.ndarray]]

_TRIAL_PAIRS = {
    "DP0-DP0": ("DP", 0, "DP", 0),
    "P1-DP0":  ("P",  1, "DP", 0),
    "P1-DP1":  ("P",  1, "DP", 1),
}

_PRECOND_CHOICES = ("none", "mass", "calderon")
_SOLVER_CHOICES = ("gmres", "dense")
_NONCONVERGENCE_CHOICES = ("raise", "warn", "ignore")


class GmresNotConvergedError(RuntimeError):
    """GMRES stopped without reaching ``gmres_tol``.

    The frequency computed from such a solve is finite and plausible-looking,
    which is exactly why it must not pass silently into a ground-truth column.
    ``result`` holds the full result dict of the unconverged solve, so a batch
    driver can still log its residual inputs and timings.
    """

    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result


class GmresNotConvergedWarning(RuntimeWarning):
    """Issued instead of :class:`GmresNotConvergedError` under ``on_nonconvergence="warn"``."""


def _prepare_mesh(mesh: MeshLike, *, rescale: bool) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(V, F, V0)`` ready for BEM.

    Implements the mesh-preparation half of the paper's Algorithm 1: load, flip
    the face winding when the signed volume is negative (so the normals point
    out of the bubble), and -- when ``rescale`` -- centre the mesh at its
    centroid and scale it to unit volume.
    """
    if isinstance(mesh, (tuple, list)):
        v = np.asarray(mesh[0], dtype=np.float64)
        f = np.asarray(mesh[1], dtype=np.int64)
    else:
        v, f = _load_triangle_mesh(str(mesh))
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)
    if v.size == 0 or f.size == 0:
        raise ValueError(f"Empty mesh: {mesh!r}")

    # Algorithm 1, orientation step: a consistently wound closed surface has
    # positive signed volume exactly when its normals point outward, so a
    # negative signed volume is fixed by swapping two indices per face.
    if _signed_volume(v, f) < 0.0:
        f = _flip_faces(f)

    if rescale:
        v = v - v.mean(axis=0)
        vol = _compute_volume(v, f)
        if vol <= 1e-12:
            raise ValueError(f"Degenerate (non-positive) volume for mesh: {mesh!r}")
        v = v * (1.0 / vol) ** (1.0 / 3.0)

    v0 = _compute_volume(v, f)
    if v0 <= 1e-12:
        raise ValueError("Input mesh has non-positive volume.")
    return v, f, float(v0)


def _set_quadrature_orders(bempp_api, regular: int, singular: int):
    """Set global quadrature orders on bempp-cl; return a ``restore()`` callback."""
    gp = getattr(bempp_api, "GLOBAL_PARAMETERS", None)
    if gp is None:
        return lambda: None
    q = getattr(gp, "quadrature", None)
    if q is None:
        return lambda: None

    prev = {}

    def _try_set(attr: str, value: int):
        if hasattr(q, attr):
            try:
                prev[attr] = getattr(q, attr)
                setattr(q, attr, int(value))
            except Exception:
                pass

    _try_set("regular", regular)
    _try_set("singular", singular)
    _try_set("double_singular", singular)
    for a in ("near", "medium", "far"):
        _try_set(a, regular)

    def restore():
        for a, val in prev.items():
            try:
                setattr(q, a, val)
            except Exception:
                pass

    return restore


def _integral_of_grid_function(bempp_api, q_gf, space) -> float:
    """
    Compute ``int_Gamma q dS`` robustly for any scalar space.

    Uses the mass matrix of ``space`` and the coefficients of the constant-one
    function (which is ``np.ones(...)`` for DP0 and for any partition-of-unity
    basis like DP1 / P1):

        <q, 1>_L2  =  q_coef^T  ( M @ ones )
    """
    ident = bempp_api.operators.boundary.sparse.identity(space, space, space)
    M = ident.weak_form()
    ones = np.ones(space.global_dof_count, dtype=np.float64)
    Mones = np.asarray(M.matvec(ones) if hasattr(M, "matvec") else M @ ones)
    return float(np.dot(np.asarray(q_gf.coefficients, dtype=np.float64), Mones))


def _configure_dense_assembly(bempp_api) -> None:
    """Ask bempp-cl to assemble boundary operators as dense matrices."""
    gp = getattr(bempp_api, "GLOBAL_PARAMETERS", None)
    if gp is None:
        return
    asm = getattr(gp, "assembly", None)
    if asm is None:
        return
    for attr in ("boundary_operator_assembly_type", "operator_assembly_type"):
        if hasattr(asm, attr):
            try:
                setattr(asm, attr, "dense")
            except Exception:
                pass


def _as_dense(weak_form) -> np.ndarray:
    """Materialise a bempp-cl discrete operator as a dense float64 matrix."""
    bempp_api = _import_bempp_api()
    if hasattr(bempp_api, "as_matrix"):
        try:
            return np.asarray(bempp_api.as_matrix(weak_form), dtype=np.float64)
        except Exception:
            pass
    if hasattr(weak_form, "A"):
        try:
            A = weak_form.A
            if A is not None:
                return np.asarray(A, dtype=np.float64)
        except Exception:
            pass
    raise RuntimeError(
        "Could not materialise a dense operator matrix from bempp-cl. "
        "Ensure _configure_dense_assembly() ran before operator assembly, "
        "or upgrade bempp-cl so weak_form.A is populated."
    )


class _IterationCounter:
    """Tiny GMRES callback that counts calls."""

    def __init__(self):
        self.n = 0

    def __call__(self, *_args, **_kwargs):
        self.n += 1


def solve_minnaert_frequency_galerkin(
    mesh: MeshLike,
    *,
    trial_pair: str = "P1-DP0",
    precond: str = "mass",
    solver: str = "gmres",
    gmres_tol: float = 1e-12,
    gmres_maxiter: int = 4000,
    gmres_restart: int = 300,
    quadrature_regular: int = 6,
    quadrature_singular: int = 6,
    gamma: float = 1.4,
    p0: float = 101325.0,
    rho: float = 1000.0,
    skip_volume_rescale: bool = False,
    rescale_to_unit_volume: bool | None = None,
    rhs_orientation_tol: float | None = 1e-2,
    on_nonconvergence: str = "raise",
    verbose: bool = False,
) -> dict:
    """
    Solve ``V q = (K - 1/2 I) phi`` in conforming mixed Galerkin form and return
    capacitance + Minnaert frequency.

    ``mesh`` is a path to a closed triangle mesh or a pre-built ``(V, F)``
    tuple; in either case a negatively wound surface is flipped outward before
    solving. ``trial_pair`` is ``"DP0-DP0"``, ``"P1-DP0"`` (default) or
    ``"P1-DP1"`` (Dirichlet-trace space first). ``precond`` is ``"none"``,
    ``"mass"`` (``use_strong_form=True``, the default) or ``"calderon"``
    (hypersingular preconditioner, ``P1-DP0`` only). ``solver`` is ``"gmres"``
    or ``"dense"`` (``np.linalg.solve``; small meshes only). ``gamma``, ``p0``
    and ``rho`` convert capacitance to a frequency.

    Parameters
    ----------
    rescale_to_unit_volume :
        Centre the mesh and rescale it to unit volume before solving. ``None``
        (the default) keeps the historical behaviour: a path is rescaled unless
        ``skip_volume_rescale`` is set, a ``(V, F)`` tuple is used as-is. Pass
        ``True``/``False`` to state the intent explicitly -- in particular so a
        tuple caller can opt into the rescaling.
    rhs_orientation_tol :
        Tolerance on the right-hand-side identity, default ``1e-2``. With
        ``phi = 1`` on a closed, consistently wound surface the solid-angle
        identity makes ``b = (K - 1/2 I) phi`` the constant ``-1`` exactly, so
        ``max|b + 1|`` is a direct measure of surface integrity: it is
        ``~3e-5`` for a sound mesh and ``>5e-2`` once the surface has holes or
        mixed winding. Raises ``ValueError`` above the tolerance, because those
        meshes return a plausible but wrong capacitance rather than failing.
        The residual is always reported as ``rhs_orientation_residual`` in the
        result. Pass ``None`` to disable the check.
    on_nonconvergence :
        What to do when GMRES stops without reaching ``gmres_tol`` (a non-zero
        ``info``: positive when it ran out of iterations, negative on breakdown
        or illegal input). ``"raise"`` (the default) raises
        :class:`GmresNotConvergedError`, whose ``result`` attribute carries the
        unconverged result; ``"warn"`` issues :class:`GmresNotConvergedWarning`
        and returns; ``"ignore"`` returns silently. In every mode the result
        records ``gmres_converged``. Raising is the default for the same reason
        as the orientation check: an unconverged solve returns a finite,
        plausible frequency, and nothing downstream can tell it apart. Both
        batch drivers catch the error per mesh and record the row as failed.

    Returns
    -------
    dict with keys
        frequency, capacitance, capacitance_raw, v0, gmres_info,
        gmres_iterations, gmres_converged, n_dirichlet_dofs, n_neumann_dofs,
        n_triangles, rhs_orientation_residual, wall_time_assemble_s,
        wall_time_solve_s, trial_pair, solver, precond.
    """
    if trial_pair not in _TRIAL_PAIRS:
        raise ValueError(
            f"Unknown trial_pair {trial_pair!r}; choose from {list(_TRIAL_PAIRS)}."
        )
    if precond not in _PRECOND_CHOICES:
        raise ValueError(f"Unknown precond {precond!r}; choose from {_PRECOND_CHOICES}.")
    if solver not in _SOLVER_CHOICES:
        raise ValueError(f"Unknown solver {solver!r}; choose from {_SOLVER_CHOICES}.")
    if on_nonconvergence not in _NONCONVERGENCE_CHOICES:
        raise ValueError(
            f"Unknown on_nonconvergence {on_nonconvergence!r}; "
            f"choose from {_NONCONVERGENCE_CHOICES}."
        )

    bempp_api = _import_bempp_api()

    if rescale_to_unit_volume is None:
        do_rescale = not (isinstance(mesh, (tuple, list)) or skip_volume_rescale)
    else:
        do_rescale = bool(rescale_to_unit_volume)
        if do_rescale and skip_volume_rescale:
            raise ValueError(
                "rescale_to_unit_volume=True contradicts skip_volume_rescale=True."
            )
    V_, F_, v0 = _prepare_mesh(mesh, rescale=do_rescale)

    if verbose:
        print(
            f"[galerkin] mesh: {V_.shape[0]} vertices, {F_.shape[0]} triangles, V0 = {v0:.6g}"
        )

    restore_quad = _set_quadrature_orders(
        bempp_api, quadrature_regular, quadrature_singular
    )
    try:
        grid = _make_bempp_grid(V_, F_)

        dir_kind, dir_order, neu_kind, neu_order = _TRIAL_PAIRS[trial_pair]
        dirichlet_space = bempp_api.function_space(grid, dir_kind, dir_order)
        neumann_space = bempp_api.function_space(grid, neu_kind, neu_order)

        n_dir = int(dirichlet_space.global_dof_count)
        n_neu = int(neumann_space.global_dof_count)
        if verbose:
            print(
                f"[galerkin] trial_pair={trial_pair}  "
                f"n_dirichlet_dofs={n_dir}  n_neumann_dofs={n_neu}"
            )

        # Signature convention: boundary_operator(domain, range, dual_to_range).
        # We use range == dual_to_range == neumann_space for every operator, so
        # that (a) slp's weak form is square in the Neumann DOFs, and (b) the
        # implicit mass-matrix inversion used by ``op * gridfunction`` is
        # well-posed (its mass matrix is a square neumann_space x neumann_space
        # matrix). For the Dirichlet trace we only need it as the *domain* of
        # dlp / ident; the rhs is returned as a GridFunction in neumann_space,
        # which is exactly the space slp solves into.
        if solver == "dense":
            _configure_dense_assembly(bempp_api)
        t0 = time.perf_counter()
        slp = bempp_api.operators.boundary.laplace.single_layer(
            neumann_space, neumann_space, neumann_space
        )
        dlp = bempp_api.operators.boundary.laplace.double_layer(
            dirichlet_space, neumann_space, neumann_space
        )
        ident = bempp_api.operators.boundary.sparse.identity(
            dirichlet_space, neumann_space, neumann_space
        )

        phi = bempp_api.GridFunction(
            dirichlet_space,
            coefficients=np.ones(n_dir, dtype=np.float64),
        )

        rhs = (dlp - 0.5 * ident) * phi

        # Surface-integrity check. For a CLOSED surface with outward normals the
        # solid-angle identity gives (K phi)(x) = -1/2 for x on Gamma when
        # phi = 1, so b = (K - 1/2 I) phi is the constant -1 exactly, for any
        # geometry. Discretely only quadrature error remains, so max|b+1| is a
        # direct measure of how well the mesh satisfies the hypothesis.
        #
        # Global inward orientation is already handled: the winding flip in
        # _prepare_mesh fires on negative signed volume, and for a consistently
        # wound closed surface positive signed volume implies outward normals.
        # What this catches is the residue -- holes and inconsistent winding --
        # which the flip cannot fix and which return a plausible but wrong
        # capacitance rather than an error (the "C < 0" safeguard below does
        # not see them).
        #
        # Measured on exhale15 bubble meshes (dense solver, quad 6/6):
        #   closed and well wound        2.6e-05 .. 2.0e-04  (n = 24)
        #   1% of faces removed          4.3e-01   -> f off by -31%
        #   1% of faces wound inward     5.6e-02   -> f off by +30%
        #   50% of faces wound inward    6.1e-01   -> f off by -70%
        # The 1e-2 default therefore sits ~50x above the worst good mesh and
        # ~5x below the mildest broken one.
        rhs_vals = np.asarray(rhs.coefficients, dtype=np.float64)
        rhs_residual = float(np.max(np.abs(rhs_vals + 1.0))) if rhs_vals.size else float("inf")
        if verbose:
            print(f"[galerkin] rhs orientation check: max|b+1| = {rhs_residual:.3e}")
        if rhs_orientation_tol is not None:
            worst = rhs_residual
            if not np.isfinite(worst) or worst > rhs_orientation_tol:
                raise ValueError(
                    "right-hand side is not the constant -1 "
                    f"(max|b+1| = {worst:.3e} > {rhs_orientation_tol:g}). "
                    "b == -1 is exact for a closed, consistently wound surface, "
                    "so this mesh is open or has faces wound both ways. Both "
                    "return a plausible but wrong capacitance -- on a test "
                    "bubble, removing 1% of faces shifted the frequency by 31% "
                    "with no other symptom. Repair the mesh (or screen it out) "
                    "rather than raising the tolerance."
                )

        slp.weak_form()  # force SLP assembly; solve timer measures iterations only
        t_assemble = time.perf_counter() - t0

        t0 = time.perf_counter()
        gmres_iterations = 0
        info: int = 0

        if solver == "dense":
            V_mat = _as_dense(slp.weak_form())  # reuses cached weak form assembled above
            rhs_vec = np.asarray(rhs.projections(neumann_space), dtype=np.float64)
            q_coef = np.linalg.solve(V_mat, rhs_vec)
            q_gf = bempp_api.GridFunction(neumann_space, coefficients=q_coef)
        else:
            counter = _IterationCounter()
            gmres_kwargs = dict(
                tol=gmres_tol,
                maxiter=gmres_maxiter,
                restart=gmres_restart,
            )
            if precond == "mass":
                gmres_kwargs["use_strong_form"] = True
            elif precond == "calderon":
                if trial_pair != "P1-DP0":
                    raise NotImplementedError(
                        "Calderon preconditioning is only wired up for trial_pair='P1-DP0' "
                        "(needs DUAL0 / P1 stable pairings); please pick precond='mass' for "
                        "the other trial pairs."
                    )
                dual0 = bempp_api.function_space(grid, "DUAL", 0)
                hyp = bempp_api.operators.boundary.laplace.hypersingular(
                    dual0, neumann_space, dual0
                )
                gmres_kwargs["use_strong_form"] = True
                gmres_kwargs["preconditioner"] = hyp
            # bempp-cl builds differ in which keywords gmres() accepts. Drop the
            # iteration callback first -- it is diagnostic only, so losing it
            # costs nothing but the `gmres_iterations` field. Losing the
            # preconditioner changes how the system is solved, so that one is
            # reported rather than swallowed.
            try:
                q_gf, info = bempp_api.linalg.gmres(
                    slp, rhs, callback=counter, **gmres_kwargs
                )
            except TypeError:
                try:
                    q_gf, info = bempp_api.linalg.gmres(slp, rhs, **gmres_kwargs)
                except TypeError as exc:
                    if "preconditioner" in gmres_kwargs:
                        warnings.warn(
                            "bempp-cl's gmres() rejected the 'preconditioner' "
                            f"keyword ({exc}); retrying without it, so "
                            f"precond={precond!r} is NOT applied and the solve "
                            "falls back to the mass-preconditioned system. "
                            "Expect more iterations; upgrade bempp-cl to use "
                            "Calderon preconditioning.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    gmres_kwargs.pop("preconditioner", None)
                    q_gf, info = bempp_api.linalg.gmres(slp, rhs, **gmres_kwargs)
            gmres_iterations = int(counter.n)

        t_solve = time.perf_counter() - t0

        integral_q = _integral_of_grid_function(bempp_api, q_gf, neumann_space)
        c_raw = -integral_q / (4.0 * math.pi)

        if c_raw < 0.0:
            if verbose:
                print(
                    f"[galerkin] WARNING: raw capacitance is negative ({c_raw:.6g}); "
                    "this now most likely indicates an inward-facing mesh winding. "
                    "Using |C| for frequency, but the input mesh should be fixed."
                )
            c_val = -c_raw
        else:
            c_val = c_raw

        try:
            freq = math.sqrt(4.0 * math.pi * gamma * p0 * c_val / (rho * v0)) / (2.0 * math.pi)
        except ValueError:
            freq = float("nan")

        if verbose:
            print(
                f"[galerkin] C_raw = {c_raw:.9g}   C = {c_val:.9g}   f = {freq:.6g} Hz   "
                f"gmres_info = {info}   gmres_it = {gmres_iterations}   "
                f"t_assemble = {t_assemble:.3f}s   t_solve = {t_solve:.3f}s"
            )

        gmres_info = 0 if info is None else int(info)
        # The dense path solves directly and always reports info = 0.
        converged = gmres_info == 0
        result = {
            "frequency": float(freq),
            "capacitance": float(c_val),
            "capacitance_raw": float(c_raw),
            "v0": float(v0),
            "gmres_info": gmres_info,
            "gmres_iterations": int(gmres_iterations),
            "gmres_converged": bool(converged),
            "n_dirichlet_dofs": n_dir,
            "n_neumann_dofs": n_neu,
            "n_triangles": int(F_.shape[0]),
            "rhs_orientation_residual": float(rhs_residual),
            "wall_time_assemble_s": float(t_assemble),
            "wall_time_solve_s": float(t_solve),
            "trial_pair": trial_pair,
            "solver": solver,
            "precond": precond,
        }

        if not converged and on_nonconvergence != "ignore":
            if gmres_info > 0:
                why = (
                    f"it stopped after {gmres_iterations or gmres_info} iterations "
                    f"without reaching tol={gmres_tol:g} "
                    f"(maxiter={gmres_maxiter}, restart={gmres_restart})"
                )
                fix = (
                    "Raise gmres_maxiter or gmres_restart, or check the mesh: "
                    "badly shaped triangles slow convergence sharply."
                )
            else:
                why = "it broke down or was given illegal input"
                fix = "Check the mesh and the solver arguments."
            message = (
                f"GMRES did not converge (info={gmres_info}): {why}. The "
                f"returned frequency, {freq:.6g} Hz, is not a converged "
                f"solution and must not be used as ground truth. {fix}"
            )
            if on_nonconvergence == "raise":
                raise GmresNotConvergedError(message, result)
            warnings.warn(message, GmresNotConvergedWarning, stacklevel=2)

        return result
    finally:
        restore_quad()


__all__ = [
    "GmresNotConvergedError",
    "GmresNotConvergedWarning",
    "solve_minnaert_frequency_galerkin",
    "_TRIAL_PAIRS",
]


def _main():
    import argparse

    from ._threading import default_thread_count, limit_threads

    p = argparse.ArgumentParser(
        description="Accurate Galerkin Minnaert-frequency solver (bempp-cl)."
    )
    p.add_argument("mesh_file")
    p.add_argument(
        "--trial-pair",
        default="P1-DP0",
        choices=list(_TRIAL_PAIRS),
        help="Function-space pair: %(default)s by default.",
    )
    p.add_argument(
        "--precond", default="mass", choices=_PRECOND_CHOICES,
        help="GMRES preconditioner.",
    )
    p.add_argument(
        "--solver", default="gmres", choices=_SOLVER_CHOICES,
        help="Linear solver.",
    )
    p.add_argument("--gmres-tol", type=float, default=1e-12)
    p.add_argument("--gmres-maxiter", type=int, default=4000)
    p.add_argument("--gmres-restart", type=int, default=300)
    p.add_argument(
        "--on-nonconvergence", default="raise", choices=_NONCONVERGENCE_CHOICES,
        help="If GMRES stops short of --gmres-tol: raise an error (default), "
             "warn and print the unconverged result, or ignore it.",
    )
    p.add_argument("--quad-regular", type=int, default=6)
    p.add_argument("--quad-singular", type=int, default=6)
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Cap BLAS/OpenCL/bempp threads (default: max(1, cpu_count - 2)).",
    )
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--p0", type=float, default=101325.0)
    p.add_argument("--rho", type=float, default=1000.0)
    p.add_argument(
        "--skip-volume-rescale", action="store_true",
        help="Do NOT rescale the input mesh to unit volume.",
    )
    args = p.parse_args()

    n_threads = (
        int(args.threads)
        if args.threads is not None
        else default_thread_count()
    )
    limit_threads(n_threads)

    out = solve_minnaert_frequency_galerkin(
        args.mesh_file,
        trial_pair=args.trial_pair,
        precond=args.precond,
        solver=args.solver,
        gmres_tol=args.gmres_tol,
        gmres_maxiter=args.gmres_maxiter,
        gmres_restart=args.gmres_restart,
        quadrature_regular=args.quad_regular,
        quadrature_singular=args.quad_singular,
        gamma=args.gamma,
        p0=args.p0,
        rho=args.rho,
        skip_volume_rescale=args.skip_volume_rescale,
        on_nonconvergence=args.on_nonconvergence,
        verbose=True,
    )

    v0 = out["v0"]
    if abs(v0 - 1.0) < 1e-6:
        r_eq = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        f_ana = _minnaert_constant() / r_eq
        c_ana = r_eq
        print(
            "\n  Analytical unit-volume sphere reference: "
            f"C = {c_ana:.6g}, f = {f_ana:.6g} Hz"
        )
        print(
            f"  |C - C_ana|/C_ana = {abs(out['capacitance'] - c_ana)/c_ana*100:.4f}%"
        )
        print(
            f"  |f - f_ana|/f_ana = {abs(out['frequency'] - f_ana)/f_ana*100:.4f}%"
        )


if __name__ == "__main__":
    _main()
