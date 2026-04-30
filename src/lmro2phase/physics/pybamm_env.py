"""PyBaMM 환경 확인 및 초기화 유틸리티."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def check_pybamm() -> dict:
    """PyBaMM 설치 여부 및 버전 확인."""
    try:
        import pybamm
        info = {
            "installed": True,
            "version": pybamm.__version__,
        }
        log.info(f"PyBaMM {pybamm.__version__} 감지")
        return info
    except ImportError:
        log.error("PyBaMM가 설치되어 있지 않습니다. setup_env.sh를 실행하세요.")
        return {"installed": False, "version": None}


def set_pybamm_threads(n: int = 1) -> None:
    """PyBaMM 내부 BLAS/OMP 스레드 수 설정."""
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)


def check_torch() -> dict:
    """PyTorch 설치 및 CUDA 가용성 확인."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": cuda,
            "device": device,
        }
        if cuda:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        log.info(f"PyTorch {torch.__version__}, CUDA={cuda}, device={device}")
        return info
    except ImportError:
        return {"installed": False}


def get_torch_device(device_str: str = "auto"):
    """env.yaml의 device 설정으로 torch.device 반환."""
    import torch

    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
