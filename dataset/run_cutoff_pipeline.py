# -*- coding: utf-8 -*-
"""
실험 작업공간들에 대해 전체 파이프라인을 순차 실행한다. 분리 프로세스로 띄우기 위한 것.

    python dataset/run_cutoff_pipeline.py <작업공간1> [<작업공간2> ...] [--M 100]

    Phase 1 (main.py) -> HMM (run_pipeline.py) -> 시나리오 선정 (--mode a)
    -> 기존 9개와 매칭 -> scenario_main -> results_main (--M)

각 단계 로그는 <작업공간>/run_<단계>.log, 전체 진행은 첫 작업공간의 부모 디렉터리에
cutoff_pipeline.log. 한 작업공간이 실패해도 다음으로 넘어간다.
Windows 이면 끝나고 절전 설정을 5분으로 되돌린다(실행 전 0으로 꺼 두었을 경우를 위해).

다른 머신에서 (Windows 예시):
    git clone <repo> ETF_code && cd ETF_code
    python -m venv etf && etf\\Scripts\\pip install -r requirements.txt
    python dataset\\build_cutoff_workspace.py 2025-12-31 ..\\ETF_code_ext_2512
    python dataset\\build_cutoff_workspace.py 2026-03-31 ..\\ETF_code_ext_2603
    start /b python -u dataset\\run_cutoff_pipeline.py ..\\ETF_code_ext_2512 ..\\ETF_code_ext_2603
Linux/mac 이면 venv 경로와 start 대신 nohup ... & 를 쓰면 된다.
"""
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PY = sys.executable
REPO = Path(__file__).resolve().parent.parent
MATCH = REPO / "analysis" / "match_scenarios.py"


def log(master, msg):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with master.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(master, root, name, args):
    t0 = time.time()
    log(master, f"{root.name} | {name} 시작")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with (root / f"run_{name}.log").open("w", encoding="utf-8") as out:
        r = subprocess.run([PY, "-u", *args], cwd=root, stdout=out, stderr=subprocess.STDOUT, env=env)
    log(master, f"{root.name} | {name} 종료 exit={r.returncode} ({(time.time()-t0)/60:.1f}분)")
    if r.returncode != 0:
        raise RuntimeError(f"{root.name} {name} 실패 — {root / f'run_{name}.log'} 참조")


def main(roots, m_repeat):
    roots = [Path(r).resolve() for r in roots]
    master = roots[0].parent / "cutoff_pipeline.log"
    log(master, "=" * 60)
    log(master, f"파이프라인 시작: {[r.name for r in roots]}  M={m_repeat}  python={PY}")
    for root in roots:
        try:
            run(master, root, "phase1",   ["simulator/main.py"])
            run(master, root, "preproc",  ["preprocessing/run_pipeline.py"])
            run(master, root, "scensel",  ["analysis/1_scenario_selection.py", "--mode", "a"])
            run(master, root, "match",    [str(root / "analysis" / "match_scenarios.py"), str(root)])
            ids = (root / "results" / "scenario_selection" / "matched_ids.txt").read_text(encoding="utf-8").strip()
            if not ids:
                raise RuntimeError("매칭된 시나리오가 없다")
            log(master, f"{root.name} | 매칭 ID: {ids}")
            run(master, root, "scenario", ["simulator/scenario_main.py", "--scenarios", ids])
            run(master, root, "results",  ["simulator/results_main.py", "--M", m_repeat, "--scenarios", ids])
            log(master, f"{root.name} | 완료")
        except Exception as e:
            log(master, f"{root.name} | 오류: {e} — 다음 작업공간으로")
    if platform.system() == "Windows":
        log(master, "절전 설정 복원 (5분)")
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", "5"])
        subprocess.run(["powercfg", "/change", "hibernate-timeout-ac", "5"])
    log(master, "전체 완료")


if __name__ == "__main__":
    argv = sys.argv[1:]
    m = "100"
    if "--M" in argv:
        i = argv.index("--M"); m = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if not argv:
        print(__doc__); sys.exit(1)
    main(argv, m)
