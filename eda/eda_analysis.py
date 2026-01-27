"""
데이터셋의 모든 CSV 파일에 대한 탐색적 데이터 분석(EDA) 수행
결과를 JSON 형식으로 저장
"""

import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import sys

# 프로젝트 루트를 경로에 추가하여 settings 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import settings


def perform_eda(df, file_name):
    """
    데이터프레임에 대한 EDA 수행
    
    Args:
        df: pandas DataFrame
        file_name: 파일 이름
    
    Returns:
        dict: EDA 결과를 담은 딕셔너리
    """
    eda_results = {
        "file_name": file_name,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "basic_info": {
            "shape": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1])
            },
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        },
        "missing_values": {
            "total_missing": int(df.isnull().sum().sum()),
            "missing_per_column": {col: int(count) for col, count in df.isnull().sum().items()},
            "missing_percentage": {col: float(percent) for col, percent in (df.isnull().sum() / len(df) * 100).items()}
        },
        "statistics": {}
    }
    
    # 수치형 컬럼에 대한 통계
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()
        eda_results["statistics"]["numeric"] = {}
        for col in numeric_cols:
            eda_results["statistics"]["numeric"][col] = {
                "count": float(numeric_stats.loc['count', col]),
                "mean": float(numeric_stats.loc['mean', col]),
                "std": float(numeric_stats.loc['std', col]),
                "min": float(numeric_stats.loc['min', col]),
                "25%": float(numeric_stats.loc['25%', col]),
                "50%": float(numeric_stats.loc['50%', col]),
                "75%": float(numeric_stats.loc['75%', col]),
                "max": float(numeric_stats.loc['max', col])
            }
    
    # 범주형/문자열 컬럼에 대한 통계
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if len(categorical_cols) > 0:
        eda_results["statistics"]["categorical"] = {}
        top_values_count = settings.get_eda_config()["top_values_count"]
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(top_values_count).to_dict()
            eda_results["statistics"]["categorical"][col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in value_counts.items()}
            }
    
    # 상관관계 (수치형 컬럼이 설정된 최소 개수 이상일 때)
    analysis_options = settings.get_eda_analysis_options()
    min_corr_cols = analysis_options["min_correlation_cols"]
    if len(numeric_cols) >= min_corr_cols and analysis_options["include_correlation"]:
        correlation_matrix = df[numeric_cols].corr()
        eda_results["correlation"] = {
            col: {other_col: float(corr) for other_col, corr in correlation_matrix[col].items()}
            for col in correlation_matrix.columns
        }
    
    # 중복 행 확인
    if analysis_options["include_duplicates"]:
        eda_results["duplicates"] = {
            "total_duplicates": int(df.duplicated().sum()),
            "duplicate_percentage": float(df.duplicated().sum() / len(df) * 100)
        }
    
    return eda_results


def main():
    """
    메인 실행 함수
    """
    # 설정에서 경로 가져오기
    dataset_dir = settings.get_dataset_path()
    results_dir = settings.get_eda_results_path()
    eda_config = settings.get_eda_config()
    
    # 결과 디렉토리 생성
    settings.ensure_directories()
    
    # dataset 폴더의 모든 CSV 파일 찾기
    file_pattern = eda_config["file_pattern"]
    csv_files = list(dataset_dir.glob(file_pattern))
    
    if not csv_files:
        print(f"{dataset_dir} 폴더에 {file_pattern} 패턴의 파일이 없습니다.")
        return
    
    print(f"총 {len(csv_files)}개의 CSV 파일을 찾았습니다.")
    
    all_results = {}
    output_suffix = eda_config["output_suffix"]
    encoding = eda_config["encoding"]
    json_indent = eda_config["json_indent"]
    
    # 각 CSV 파일에 대해 EDA 수행
    for csv_file in csv_files:
        print(f"\n분석 중: {csv_file.name}")
        try:
            # CSV 파일 읽기
            df = pd.read_csv(csv_file)
            
            # EDA 수행
            eda_result = perform_eda(df, csv_file.name)
            all_results[csv_file.stem] = eda_result
            
            # 개별 파일 결과 저장
            output_file = results_dir / f"{csv_file.stem}{output_suffix}"
            with open(output_file, 'w', encoding=encoding) as f:
                json.dump(eda_result, f, ensure_ascii=False, indent=json_indent)
            
            print(f"  [완료] {output_file}")
            
        except Exception as e:
            print(f"  [오류] ({csv_file.name}): {str(e)}")
            all_results[csv_file.stem] = {
                "file_name": csv_file.name,
                "error": str(e)
            }
    
    # 전체 결과를 하나의 파일로 저장
    summary_filename = eda_config["summary_filename"]
    summary_file = results_dir / summary_filename
    with open(summary_file, 'w', encoding=encoding) as f:
        json.dump(all_results, f, ensure_ascii=False, indent=json_indent)
    
    print(f"\n전체 결과 요약 저장: {summary_file}")
    print("\nEDA 분석이 완료되었습니다!")


if __name__ == "__main__":
    main()
