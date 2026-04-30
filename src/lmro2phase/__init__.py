"""lmro2phase: LMR 2-phase inverse modeling pipeline."""

__version__ = "0.1.0"

PHASES = ["rhombohedral_R3m", "monoclinic_C2m"]

phase_map = {
    "Primary": "rhombohedral_R3m",
    "Secondary": "monoclinic_C2m",
}
