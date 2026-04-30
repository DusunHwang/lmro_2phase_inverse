#!/usr/bin/env bash
# ============================================================
# setup_env.sh  —  시스템별 환경 설치 스크립트
# sudo 없이도 동작합니다.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
TORCH_INDEX_URL=""   # CUDA 사용 시 변경: https://download.pytorch.org/whl/cu121

# ─── 색상 출력 ────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── Python 확인 ─────────────────────────────────────────
info "Python 버전 확인..."
PYTHON=$(command -v python3 || command -v python || error "Python3를 찾을 수 없습니다.")
PY_VER=$("$PYTHON" -c "import sys; print(sys.version_info[:2])")
info "  사용 Python: $PYTHON ($PY_VER)"

# ─── pip 확인/설치 ────────────────────────────────────────
ensure_pip() {
    if "$PYTHON" -m pip --version &>/dev/null; then
        info "  pip 이미 사용 가능"
        return 0
    fi

    warn "pip를 찾을 수 없습니다. get-pip.py로 설치합니다..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$PYTHON" /tmp/get-pip.py --user
    rm -f /tmp/get-pip.py
    export PATH="$HOME/.local/bin:$PATH"

    if "$PYTHON" -m pip --version &>/dev/null; then
        info "  pip 설치 완료"
    else
        error "pip 설치에 실패했습니다."
    fi
}

# ─── 가상환경 생성 ────────────────────────────────────────
create_venv() {
    info "=== [1/4] 가상환경 생성 ==="

    if [ -d "$VENV_DIR" ]; then
        info "  이미 존재: $VENV_DIR (skip)"
        return 0
    fi

    # 방법 1: python -m venv (ensurepip 필요)
    if "$PYTHON" -m venv "$VENV_DIR" &>/dev/null; then
        info "  python -m venv 성공: $VENV_DIR"
        return 0
    fi

    warn "  python -m venv 실패 (ensurepip 없음?). 대안 시도..."

    # 방법 2: virtualenv (pip로 설치)
    ensure_pip
    if "$PYTHON" -m pip install virtualenv --user -q; then
        "$PYTHON" -m virtualenv "$VENV_DIR"
        info "  virtualenv 성공: $VENV_DIR"
        return 0
    fi

    # 방법 3: conda 환경 사용
    if command -v conda &>/dev/null; then
        warn "  conda 환경을 사용합니다."
        cat <<'MSG'
[안내] conda 환경에서는 아래 명령으로 직접 설치하세요:
  conda create -n lmro2phase python=3.11 -y
  conda activate lmro2phase
  pip install -e ".[dev]"
MSG
        exit 0
    fi

    # 방법 4: --user 직접 설치 (venv 없이)
    warn "  가상환경 생성 불가. --user 모드로 직접 설치합니다."
    warn "  (가상환경 없이 사용자 로컬에 설치됩니다)"
    VENV_DIR=""
}

create_venv

# ─── pip 업그레이드 ───────────────────────────────────────
info "=== [2/4] pip 업그레이드 ==="
if [ -n "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip wheel setuptools -q
else
    "$PYTHON" -m pip install --upgrade pip wheel setuptools --user -q
    PIP="$PYTHON -m pip"
fi

# ─── PyTorch 설치 ─────────────────────────────────────────
info "=== [3/4] PyTorch 설치 ==="
if [ -n "$TORCH_INDEX_URL" ]; then
    pip install torch --index-url "$TORCH_INDEX_URL" -q
    info "  PyTorch (CUDA) 설치 완료"
else
    pip install torch -q
    info "  PyTorch (CPU) 설치 완료"
fi

# ─── 패키지 설치 ──────────────────────────────────────────
info "=== [4/4] lmro2phase 패키지 설치 (editable) ==="
pip install -e ".[dev]" -q

# ─── 확인 ────────────────────────────────────────────────
echo ""
info "=== 설치 완료 확인 ==="
python -c "import pybamm; print(f'  PyBaMM    : {pybamm.__version__}')"
python -c "import torch;   print(f'  PyTorch   : {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "import lmro2phase; print(f'  lmro2phase: {lmro2phase.__version__}')"

echo ""
if [ -n "$VENV_DIR" ]; then
    info "환경 활성화 명령: source $VENV_DIR/bin/activate"
else
    warn "가상환경 없이 설치됨. 터미널 재시작 또는 PATH 업데이트 필요할 수 있습니다."
fi
