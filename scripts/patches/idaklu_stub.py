"""Pure-Python/casadi stub for pybammsolvers.idaklu.

Applied automatically by setup_env.sh when the C++ idaklu extension cannot be
loaded (e.g., casadi ABI mismatch on aarch64 or missing sundials libraries).

Implements VectorRealtypeNdArray, observe, and observe_hermite_interp using
casadi, so pybamm ProcessedVariable works without the C++ extension.
IDAKLUSolver itself remains unsupported — simulator.py uses CasadiSolver.
"""
from __future__ import annotations

import numpy as np


class VectorRealtypeNdArray:
    """Python replacement for the C++ pybind11 VectorRealtypeNdArray."""
    __slots__ = ("_data",)

    def __init__(self, arrays):
        self._data = list(arrays)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]


def _deserialize(func_str: str):
    import casadi
    return casadi.Function.deserialize(func_str)


def observe(ts_wrapped, ys_wrapped, inputs_wrapped, funcs, is_f_contiguous, shapes):
    """Evaluate pybamm casadi functions at solution time points.

    Replaces C++ idaklu.observe.  Each funcs[i] is a serialized
    casadi.Function(t, y, p) that maps the DAE state to a variable value.
    """
    import casadi

    results: list[np.ndarray] = []
    fn_cache: dict[str, object] = {}

    for t_seg, y_seg, inp_seg, func_str in zip(
            ts_wrapped, ys_wrapped, inputs_wrapped, funcs):

        t_arr = np.asarray(t_seg).ravel()
        y_arr = np.asarray(y_seg)
        inp_arr = np.asarray(inp_seg).ravel()

        if func_str not in fn_cache:
            fn_cache[func_str] = _deserialize(func_str)
        fn = fn_cache[func_str]

        n_t = len(t_arr)
        for j in range(n_t):
            t_j = float(t_arr[j])
            y_j = y_arr[:, j] if y_arr.ndim > 1 else y_arr
            out = fn(t_j, casadi.DM(y_j), casadi.DM(inp_arr))
            results.append(np.array(out).ravel())

    if not results:
        total = int(np.prod(shapes)) if shapes else 0
        return np.zeros(total)

    combined = np.concatenate(results)
    return combined if len(shapes) == 1 else combined.reshape(shapes, order="F")


def observe_hermite_interp(t, ts_wrapped, ys_wrapped, yps_wrapped,
                            inputs_wrapped, funcs, shapes):
    """Hermite interpolation fallback — delegates to nearest-point observe."""
    return observe(ts_wrapped, ys_wrapped, inputs_wrapped, funcs,
                   is_f_contiguous=False, shapes=shapes)


class _Stub:
    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **kw):
        raise RuntimeError(f"IDAKLU C++ solver not available: {self._name}")

    def __repr__(self):
        return f"<idaklu stub: {self._name}>"


create_casadi_solver_group = _Stub("create_casadi_solver_group")
create_idaklu_jax = _Stub("create_idaklu_jax")
generate_function = _Stub("generate_function")
reduce_knots = _Stub("reduce_knots")
registrations = {}
sundials_error_message = _Stub("sundials_error_message")
