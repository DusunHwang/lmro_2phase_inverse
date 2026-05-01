#!/usr/bin/env bash
# ============================================================
# build_idaklu_deps.sh
# aarch64 환경에서 SUNDIALS + SuiteSparse를 소스 빌드하여
# pybammsolvers의 IDAKLU solver를 사용 가능하게 합니다.
# 설치 대상: <project>/.idaklu/
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR/.idaklu"
BUILD_TMP="$SCRIPT_DIR/.idaklu_build"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

SUITESPARSE_VER="7.7.0"
SUNDIALS_VER="6.7.0"
NPROC=$(nproc)

info "빌드 디렉토리: $BUILD_TMP"
info "설치 디렉토리: $INSTALL_DIR"
info "병렬 빌드: $NPROC 코어"
mkdir -p "$BUILD_TMP" "$INSTALL_DIR"

# ─── BLAS/LAPACK 확인 ─────────────────────────────────────
info "=== BLAS/LAPACK 확인 ==="
BLAS_LIB=$(find /usr -name "libblas.so*" -o -name "libopenblas.so*" 2>/dev/null | head -1)
LAPACK_LIB=$(find /usr -name "liblapack.so*" 2>/dev/null | head -1)
if [ -z "$BLAS_LIB" ]; then
    warn "BLAS를 찾을 수 없습니다. 기본 경로 사용."
fi
info "  BLAS: ${BLAS_LIB:-시스템 기본값}"
info "  LAPACK: ${LAPACK_LIB:-시스템 기본값}"

# ─── SuiteSparse 빌드 ────────────────────────────────────
info "=== [1/3] SuiteSparse $SUITESPARSE_VER 빌드 ==="

SS_SRC="$BUILD_TMP/SuiteSparse-$SUITESPARSE_VER"
SS_TAR="$BUILD_TMP/SuiteSparse-$SUITESPARSE_VER.tar.gz"

if [ ! -d "$SS_SRC" ]; then
    info "  다운로드 중..."
    curl -fsSL \
        "https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v${SUITESPARSE_VER}.tar.gz" \
        -o "$SS_TAR"
    tar -xzf "$SS_TAR" -C "$BUILD_TMP"
    rm -f "$SS_TAR"
fi

SS_BUILD_DIR="$BUILD_TMP/SuiteSparse-build"
mkdir -p "$SS_BUILD_DIR"

cmake -S "$SS_SRC" -B "$SS_BUILD_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;btf;colamd;camd;ccolamd;klu" \
    -DSUITESPARSE_USE_CUDA=OFF \
    -DSUITESPARSE_USE_OPENMP=OFF \
    -DBLA_VENDOR=Generic \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -Wno-dev

cmake --build "$SS_BUILD_DIR" --config Release -j "$NPROC"
cmake --install "$SS_BUILD_DIR"
info "  SuiteSparse 설치 완료: $INSTALL_DIR"

# ─── SUNDIALS 빌드 ───────────────────────────────────────
info "=== [2/3] SUNDIALS $SUNDIALS_VER 빌드 ==="

SD_SRC="$BUILD_TMP/sundials-$SUNDIALS_VER"
SD_TAR="$BUILD_TMP/sundials-$SUNDIALS_VER.tar.gz"

if [ ! -d "$SD_SRC" ]; then
    info "  다운로드 중..."
    curl -fsSL \
        "https://github.com/LLNL/sundials/releases/download/v${SUNDIALS_VER}/sundials-${SUNDIALS_VER}.tar.gz" \
        -o "$SD_TAR"
    tar -xzf "$SD_TAR" -C "$BUILD_TMP"
    rm -f "$SD_TAR"
fi

SD_BUILD_DIR="$BUILD_TMP/sundials-build"
mkdir -p "$SD_BUILD_DIR"

cmake -S "$SD_SRC" -B "$SD_BUILD_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_KLU=ON \
    -DKLU_INCLUDE_DIR="$INSTALL_DIR/include/suitesparse" \
    -DKLU_LIBRARY_DIR="$INSTALL_DIR/lib" \
    -DENABLE_LAPACK=ON \
    -DENABLE_OPENMP=ON \
    -DBUILD_TESTING=OFF \
    -DEXAMPLES_ENABLE_C=OFF \
    -DEXAMPLES_ENABLE_CXX=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build "$SD_BUILD_DIR" --config Release -j "$NPROC"
cmake --install "$SD_BUILD_DIR"
info "  SUNDIALS 설치 완료: $INSTALL_DIR"

# ─── pybammsolvers 빌드 ───────────────────────────────────
info "=== [3/3] pybammsolvers 빌드 (GitHub 소스 사용) ==="
# PyPI 0.6.0 tarball에 헤더 파일 3개 누락 버그 → GitHub tag v0.6.0 소스 사용

VENV_DIR="$SCRIPT_DIR/.venv"
source "$VENV_DIR/bin/activate"

PYBAMM_SRC="/tmp/pybammsolvers-git"
if [ ! -d "$PYBAMM_SRC" ]; then
    info "  GitHub에서 pybammsolvers v0.6.0 clone 중..."
    git clone --depth=1 --branch v0.6.0 \
        https://github.com/pybamm-team/pybammsolvers.git "$PYBAMM_SRC" 2>&1
fi

export LD_LIBRARY_PATH="$INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"
export SUNDIALS_ROOT="$INSTALL_DIR"
export SuiteSparse_ROOT="$INSTALL_DIR"
export CMAKE_PREFIX_PATH="$INSTALL_DIR:${CMAKE_PREFIX_PATH:-}"

# pip >= 23: --build-option deprecated → setup.py 직접 호출
cd "$PYBAMM_SRC"
python setup.py bdist_wheel \
    --suitesparse-root="$INSTALL_DIR" \
    --sundials-root="$INSTALL_DIR" \
    2>&1 | tail -20

# 생성된 wheel 설치
WHEEL_FILE=$(ls dist/*.whl 2>/dev/null | head -1)
if [ -n "$WHEEL_FILE" ]; then
    info "  wheel 설치: $WHEEL_FILE"
    pip install "$WHEEL_FILE" --force-reinstall -q
else
    # wheel 없으면 직접 소스 설치
    warn "  wheel 생성 실패, pip install 직접 시도..."
    pip install "$PYBAMM_SRC" \
        --config-settings="--build-option=--suitesparse-root=$INSTALL_DIR" \
        --config-settings="--build-option=--sundials-root=$INSTALL_DIR" \
        -q
fi
cd "$SCRIPT_DIR"

info "=== 설치 확인 ==="
python -c "from pybammsolvers import idaklu; print('  pybammsolvers: OK')"
python -c "import pybamm; print(f'  PyBaMM: {pybamm.__version__}')"

info "완료! LD_LIBRARY_PATH에 추가 필요:"
info "  export LD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\$LD_LIBRARY_PATH\""
