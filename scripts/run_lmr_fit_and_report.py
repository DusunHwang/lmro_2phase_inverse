"""Run LMR 2-phase fitting and generate its report.

This wrapper forwards fitting arguments to run_lmr_2phase_fit.py, captures the
timestamped output directory printed by that script, then runs
generate_lmr_fit_report.py for the same directory.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _extract_fit_dir(output_lines: list[str]) -> Path | None:
    pattern = re.compile(r"출력 경로:\s*(.+?)\s*$")
    for line in reversed(output_lines):
        match = pattern.search(line)
        if match:
            return Path(match.group(1)).expanduser()
    return None


def _arg_value(args: list[str], flag: str, default: str | None = None) -> str | None:
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def _run_streaming(cmd: list[str]) -> tuple[int, list[str]]:
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))
    return proc.wait(), lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LMR 2-phase fitting, then generate 피팅_결과_리포트.md",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Only run fitting and skip report generation",
    )
    parser.add_argument(
        "--report-subsample-plot",
        type=int,
        default=200,
        help="subsample-plot passed to generate_lmr_fit_report.py",
    )
    parser.add_argument(
        "--report-data-csv",
        default=None,
        help="data CSV for report. Defaults to fitting --data-csv",
    )
    wrapper_args, fit_args = parser.parse_known_args()

    fit_cmd = [sys.executable, str(HERE / "run_lmr_2phase_fit.py"), *fit_args]
    print("fitting 실행:")
    print("  " + " ".join(fit_cmd), flush=True)
    code, lines = _run_streaming(fit_cmd)
    if code != 0:
        return code

    fit_dir = _extract_fit_dir(lines)
    if fit_dir is None:
        print("ERROR: fitting 출력에서 결과 폴더를 찾지 못했습니다.", file=sys.stderr)
        return 2
    if not fit_dir.is_absolute():
        fit_dir = ROOT / fit_dir

    if wrapper_args.skip_report:
        print(f"\nreport 생략. fitting 결과: {fit_dir}")
        return 0

    data_csv = wrapper_args.report_data_csv or _arg_value(
        fit_args,
        "--data-csv",
        "data/raw/toyo/lmr_dfn_2phase_sample/Toyo_LMR_DFN_2phase_0p1C_0p33C_0p5C_1C.csv",
    )
    report_cmd = [
        sys.executable,
        str(HERE / "generate_lmr_fit_report.py"),
        "--fit-dir",
        str(fit_dir),
        "--data-csv",
        str(data_csv),
        "--subsample-plot",
        str(wrapper_args.report_subsample_plot),
    ]
    print("\nreport 생성:")
    print("  " + " ".join(report_cmd), flush=True)
    report_code, _report_lines = _run_streaming(report_cmd)
    if report_code != 0:
        return report_code

    print(f"\n완료: {fit_dir / '피팅_결과_리포트.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
