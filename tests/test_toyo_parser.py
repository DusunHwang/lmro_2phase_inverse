"""TOYO ASCII 파서 단위 테스트."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CONFIG = ROOT / "configs" / "toyo_ascii.yaml"
SAMPLE_CSV = ROOT.parent / "Toyo_LMR_HalfCell_Sample_50cycles.csv"


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="실제 데이터 파일 없음")
def test_parse_real_file():
    from lmro2phase.io.toyo_ascii import ToyoAsciiParser

    parser = ToyoAsciiParser(CONFIG)
    profile = parser.parse(SAMPLE_CSV)

    assert len(profile) > 0
    assert profile.voltage_v is not None
    assert profile.current_a is not None
    assert profile.time_s is not None
    # 전압 범위 확인
    assert profile.voltage_v.min() >= 1.5
    assert profile.voltage_v.max() <= 5.5
    # 사이클 감지
    cycles = profile.get_cycles()
    assert len(cycles) > 0


def test_parse_synthetic_ascii():
    """인메모리 합성 CSV 파일로 파서 테스트."""
    from lmro2phase.io.toyo_ascii import ToyoAsciiParser

    csv_content = (
        "Data_Point,Cycle_No,Step_No,Mode,Time(s),Voltage(V),Current(mA),Capacity(mAh)\n"
        "1,1,1,CC-Chg,0,3.2,0.25,0.0\n"
        "2,1,1,CC-Chg,1800,4.0,0.25,0.125\n"
        "3,1,2,Rest,3600,3.9,0.0,0.125\n"
        "4,1,3,CC-Dchg,5400,3.5,-0.25,0.0\n"
    )

    # 임시 파일에 저장
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        parser = ToyoAsciiParser(CONFIG)
        profile = parser.parse(tmp_path)

        assert len(profile) == 4
        assert profile.current_a is not None
        # mA → A 변환 확인
        assert abs(profile.current_a[0] - 0.25e-3) < 1e-6
        # Rest 구간 전류 0 확인
        assert abs(profile.current_a[2]) < 1e-4
        # cycle 감지
        assert 1 in profile.get_cycles()
    finally:
        os.unlink(tmp_path)


def test_parse_japanese_header():
    """일본어 헤더 파일 파싱 테스트."""
    from lmro2phase.io.toyo_ascii import ToyoAsciiParser

    csv_content = "時間,電圧,電流\n0,3.2,0.25\n1800,4.0,0.25\n"

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        parser = ToyoAsciiParser(CONFIG)
        profile = parser.parse(tmp_path)
        assert profile.voltage_v is not None
        assert len(profile) == 2
    finally:
        os.unlink(tmp_path)


def test_segment_mode_detection():
    """StepMode 자동 감지 테스트."""
    from lmro2phase.io.profile_schema import StepMode
    from lmro2phase.io.toyo_ascii import _infer_step_mode

    cfg = {"rest_current_threshold_a": 1e-4,
           "cc_std_ratio_threshold": 0.05,
           "cv_voltage_std_threshold_v": 0.002}

    # CC 충전
    i_cc = np.full(50, 0.25e-3)
    v_cc = np.linspace(3.2, 4.5, 50)
    assert _infer_step_mode(i_cc, v_cc, cfg) == StepMode.CC_CHARGE

    # Rest
    i_rest = np.zeros(50)
    v_rest = np.full(50, 3.9)
    assert _infer_step_mode(i_rest, v_rest, cfg) == StepMode.REST
