# -*- coding: utf-8 -*-
"""
이 리포에서 격리된 실험 작업공간을 만든다. 코드를 복사하고 dataset/*_ext 를
dataset/{raw,train,train_ver1} 자리에 넣은 뒤, 지정 날짜까지 절단한다.

    python dataset/build_cutoff_workspace.py <절단일|full> <출력 경로> [--ctrl]

    python dataset/build_cutoff_workspace.py full        ../ETF_code_ext        # 확장판 전체
    python dataset/build_cutoff_workspace.py 2025-12-31  ../ETF_code_ext_2512   # 충격 이전
    python dataset/build_cutoff_workspace.py 2026-03-31  ../ETF_code_ext_2603   # 충격 초입
    --ctrl 을 주면 대조군(raw_ext_ctrl 계열, 트렌드 기존값 보존)을 쓴다.

값은 재계산하지 않고 행만 잘라내므로 겹치는 구간은 확장판과 bit-exact로 같다.
results/, 가상환경(etf/), 로그는 복사하지 않는다. 대부분의 스크립트가
Path(__file__).parent.parent 로 루트를 잡으므로 출력 경로에서 그대로 실행된다.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CODE_DIRS = ["simulator", "preprocessing", "analysis", "correlation", "eda", "plot", "trend"]
CODE_FILES = ["settings.py", "requirements.txt"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.log", "*.err", "*.pyc")


def main(cutoff: str, dst: str, ctrl: bool = False) -> int:
    dst = Path(dst).resolve()
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for d in CODE_DIRS:
        if (REPO / d).exists():
            shutil.copytree(REPO / d, dst / d, ignore=IGNORE)
    for f in CODE_FILES:
        if (REPO / f).exists():
            shutil.copy2(REPO / f, dst / f)

    suffix = "_ext_ctrl" if ctrl else "_ext"
    mapping = {"raw": f"raw{suffix}", "train": f"train{suffix}", "train_ver1": f"train_ver1{suffix}"}
    (dst / "dataset").mkdir()
    for target, src in mapping.items():
        shutil.copytree(REPO / "dataset" / src, dst / "dataset" / target, ignore=IGNORE)
    shutil.rmtree(dst / "dataset" / "raw" / "_cache", ignore_errors=True)
    # 원본 raw 에만 있는 보조 파일 (make_train_data_ver1.py 용, 파이프라인은 안 씀)
    for aux in ["KOSPI Volatility_history.csv", "bitcoin_all_ver1.csv", "bitcoin_kr_ver1.csv"]:
        s = REPO / "dataset" / "raw" / aux
        if s.exists():
            shutil.copy2(s, dst / "dataset" / "raw" / aux)
    shutil.copytree(REPO / "dataset" / "baseline_scenarios", dst / "dataset" / "baseline_scenarios")

    cut = None if cutoff.lower() == "full" else pd.Timestamp(cutoff)
    print(f"[{cutoff}] {'대조군' if ctrl else '앵커보정판'} -> {dst}")
    for target in mapping:
        for f in sorted((dst / "dataset" / target).glob("*.csv")):
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "Date" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"])
            n0 = len(df)
            if cut is not None:
                df = df[df["Date"] <= cut]
            df.to_csv(f, index=False, encoding="utf-8" if target == "raw" else "utf-8-sig")
            print(f"  {target}/{f.name}: {n0} -> {len(df)}행  (~{df['Date'].max().date()})")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(args[0], args[1], ctrl="--ctrl" in sys.argv))
