# -*- coding: utf-8 -*-
"""
기존 9개 시나리오와 새 시나리오 집합을 축 해상도 변화를 감안해 대응시킨다.

    python analysis/match_scenarios.py <새 작업공간 루트>

규칙
  - 축의 수준 집합이 기존과 같으면 정확 일치만 인정
  - 기존 2수준(High/Low)이 새 3수준(High/Mid/Low)으로 쪼개졌으면
    High -> {High, Mid}, Low -> {Low, Mid} 로 호환 매칭
출력
  <루트>/results/scenario_selection/matched_ids.txt   쉼표 구분 ID (scenario_main 인자용)
  <루트>/results/scenario_selection/match_table.csv   대응표
"""
import sys
from pathlib import Path

import pandas as pd

AX = ["Bitcoin_RV", "VKOSPI", "KR_Volume", "KR_SVI", "Global_SVI"]
# 기존(342일) 시나리오 정의. results/는 gitignore라 dataset/baseline_scenarios/에 추적 사본을 둔다.
# 작업공간에서 실행되면 그 작업공간의 사본을, 리포에서 실행되면 리포의 사본을 읽는다.
ORIG = Path(__file__).resolve().parent.parent / "dataset" / "baseline_scenarios"


def main(root: str) -> int:
    root = Path(root)
    old = pd.read_csv(ORIG / "final_scenarios_latest.csv")
    new = pd.read_csv(root / "results" / "scenario_selection" / "final_scenarios_latest.csv")
    oldt = pd.read_csv(ORIG / "regime_classified_table.csv")
    newt = pd.read_csv(root / "results" / "regime_classified_table.csv")

    rows = []
    for _, o in old.iterrows():
        for _, n in new.iterrows():
            ok = True
            for a in AX:
                lo, ln = set(oldt[a].astype(str)), set(newt[a].astype(str))
                if lo == ln:
                    ok &= (str(o[a]) == str(n[a]))
                elif len(lo) == 2 and len(ln) == 3:
                    comp = {"High": {"High", "Mid"}, "Low": {"Low", "Mid"}}
                    ok &= str(n[a]) in comp.get(str(o[a]), {str(o[a])})
                else:
                    ok &= (str(o[a]) == str(n[a]))
                if not ok:
                    break
            if ok:
                rows.append({
                    "old": o["Scenario_ID"], "old_combo": "-".join(str(o[a]) for a in AX),
                    "new": n["Scenario_ID"], "new_combo": "-".join(str(n[a]) for a in AX),
                    "new_freq": n["joint_frequency"],
                    "exact": all(str(o[a]) == str(n[a]) for a in AX),
                })
    m = pd.DataFrame(rows)
    out = root / "results" / "scenario_selection"
    m.to_csv(out / "match_table.csv", index=False, encoding="utf-8-sig")
    ids = sorted(m["new"].unique()) if len(m) else []
    (out / "matched_ids.txt").write_text(",".join(ids), encoding="utf-8")

    print(f"축별 수준: " + ", ".join(f"{a}={oldt[a].nunique()}->{newt[a].nunique()}" for a in AX))
    print(f"기존 9개 중 대응물 있는 것: {m['old'].nunique() if len(m) else 0}")
    print(f"매칭 ID ({len(ids)}개): {','.join(ids)}")
    if len(m):
        print(m.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
