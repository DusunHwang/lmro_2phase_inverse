#!/usr/bin/env bash
# ============================================================
# setup_env.sh  —  시스템별 환경 설치 스크립트
# 다른 시스템에서 처음 세팅할 때 실행합니다.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- configs/env.yaml에서 설정 읽기 ---
# 필요 시 python -c "import yaml; ..." 으로 파싱 가능
# 여기서는 간단히 하드코딩 fallback 사용
VENV_DIR=".venv"
TORCH_INDEX_URL=""   # CUDA 사용 시: https://download.pytorch.org/whl/cu121

echo "=== [1/4] 가상환경 생성 ==="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
else
    echo "  Already exists: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "=== [2/4] pip 업그레이드 ==="
pip install --upgrade pip wheel setuptools -q

echo "=== [3/4] PyTorch 설치 ==="
if [ -n "$TORCH_INDEX_URL" ]; then
    pip install torch --index-url "$TORCH_INDEX_URL" -q
else
    pip install torch -q
fi

echo "=== [4/4] 패키지 설치 (editable) ==="
pip install -e ".[dev]" -q

echo ""
echo "=== 설치 완료 ==="
python -c "import pybamm; print(f'PyBaMM: {pybamm.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import lmro2phase; print(f'lmro2phase: {lmro2phase.__version__}')"
echo ""
echo "활성화 명령: source $VENV_DIR/bin/activate"
