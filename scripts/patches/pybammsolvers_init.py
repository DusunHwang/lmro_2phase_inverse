import importlib.util as il

from .version import __version__

idaklu_spec = il.find_spec("pybammsolvers.idaklu")
try:
    idaklu = il.module_from_spec(idaklu_spec)
    idaklu_spec.loader.exec_module(idaklu)
except (ImportError, OSError):
    # IDAKLU C++ solver unavailable (e.g., missing sundials libs or casadi ABI mismatch).
    # idaklu.py Python stub will be used instead for ProcessedVariable observation.
    idaklu = None
