"""
Train 데이터(nav_train, gap_train, kp_train)에 대한 탐색적 데이터 분석(EDA)
기존 raw 데이터 EDA(eda_analysis.py)는 그대로 두고, 결과는 results/eda_train_results 에 저장
"""

import pandas as pd
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import settings
from eda.eda_analysis import perform_eda


def main():
    train_dir = settings.TRAIN_DATASET_DIR
    results_dir = settings.get_eda_train_results_path()
    eda_config = settings.get_eda_config()

    settings.ensure_directories()

    file_pattern = eda_config["file_pattern"]
    csv_files = sorted(train_dir.glob(file_pattern)) if train_dir.exists() else []

    if not csv_files:
        print(f"{train_dir} 폴더에 {file_pattern} 패턴의 파일이 없습니다.")
        return

    print(f"Train EDA: {train_dir} → {results_dir}")
    print(f"총 {len(csv_files)}개 CSV 분석: {[f.name for f in csv_files]}")

    all_results = {}
    output_suffix = eda_config["output_suffix"]
    encoding = eda_config["encoding"]
    json_indent = eda_config["json_indent"]

    for csv_file in csv_files:
        print(f"\n분석 중: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            eda_result = perform_eda(df, csv_file.name)
            all_results[csv_file.stem] = eda_result

            output_file = results_dir / f"{csv_file.stem}{output_suffix}"
            with open(output_file, "w", encoding=encoding) as f:
                json.dump(eda_result, f, ensure_ascii=False, indent=json_indent)
            print(f"  [완료] {output_file}")
        except Exception as e:
            print(f"  [오류] {csv_file.name}: {e}")
            all_results[csv_file.stem] = {"file_name": csv_file.name, "error": str(e)}

    summary_file = results_dir / "all_eda_train_results.json"
    with open(summary_file, "w", encoding=encoding) as f:
        json.dump(all_results, f, ensure_ascii=False, indent=json_indent)

    print(f"\n전체 결과 요약: {summary_file}")
    print("Train EDA 완료.")


if __name__ == "__main__":
    main()
