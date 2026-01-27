"""
프로젝트 설정 관리
경로, 파일명, 분석 옵션 등을 중앙에서 관리
"""

from pathlib import Path
from typing import Dict, Any


class Settings:
    """프로젝트 전역 설정 클래스"""
    
    # 프로젝트 루트 디렉토리
    PROJECT_ROOT = Path(__file__).parent
    
    # ========== 경로 설정 ==========
    DATASET_DIR = PROJECT_ROOT / "dataset"
    RESULTS_DIR = PROJECT_ROOT / "results"
    EDA_RESULTS_DIR = RESULTS_DIR / "eda_results"
    VARIABLES_RESULTS_DIR = RESULTS_DIR / "variables_results"
    VARIABLES_DIR = PROJECT_ROOT / "variables"
    EDA_DIR = PROJECT_ROOT / "eda"
    
    # ========== 파일명 설정 ==========
    # 데이터 파일
    NAV_VARIABLES_FILE = "nav_variables.csv"
    Y_TRUE_VARIABLES_FILE = "y_true_variables.csv"
    
    # 결과 파일
    MEAN_SELECTED_FILE = "mean_selected.csv"
    VAR_SELECTED_FILE = "var_selected.csv"
    RESULTS_JSON_FILE = "variable_selection_results.json"
    
    # ========== 분석 설정 ==========
    # Lag 설정
    MIN_LAG = 1
    MAX_LAG = 5
    
    # ARIMAX 설정
    ARIMAX_CONFIG: Dict[str, Any] = {
        "max_p": 3,  # AR 최대 차수
        "max_q": 3,  # MA 최대 차수
        "criterion": "bic",  # 모델 선택 기준
        "univariate_threshold": 0.1,  # 단변량 스크리닝 p-value 임계값
        "multivariate_method": "bic",  # 다변량 정리 방법: "bic" or "backward"
    }
    
    # GARCHX 설정
    GARCHX_CONFIG: Dict[str, Any] = {
        "garch_p": 1,
        "garch_q": 1,
        "dist": "t",  # 분포: "t" (t-distribution)
        "univariate_threshold": 0.1,  # 단변량 스크리닝 p-value 임계값
        "multivariate_method": "bic",  # 다변량 정리 방법: "bic" or "backward"
    }
    
    # 다중검정 보정 설정
    MULTIPLE_TESTING: Dict[str, Any] = {
        "method": "holm",  # "holm" or "bonferroni"
        "alpha": 0.05,  # 유의수준
    }
    
    # EDA 설정
    EDA_CONFIG: Dict[str, Any] = {
        "file_pattern": "*.csv",  # 분석할 파일 패턴
        "output_suffix": "_eda.json",  # 출력 파일 접미사
        "summary_filename": "all_eda_results.json",  # 전체 결과 요약 파일명
        "top_values_count": 10,  # 범주형 변수 상위 값 개수
        "json_indent": 2,  # JSON 출력 들여쓰기
        "encoding": "utf-8",  # 파일 인코딩
    }
    
    # EDA 분석 옵션
    EDA_ANALYSIS_OPTIONS: Dict[str, Any] = {
        "include_correlation": True,  # 상관관계 분석 포함 여부
        "include_duplicates": True,  # 중복 행 분석 포함 여부
        "min_correlation_cols": 2,  # 상관관계 분석 최소 컬럼 수
    }
    
    # ========== 경로 반환 메서드 ==========
    @classmethod
    def get_dataset_path(cls) -> Path:
        """데이터셋 디렉토리 경로 반환"""
        return cls.DATASET_DIR
    
    @classmethod
    def get_nav_variables_path(cls) -> Path:
        """nav_variables.csv 파일 경로 반환"""
        return cls.DATASET_DIR / cls.NAV_VARIABLES_FILE
    
    @classmethod
    def get_y_true_variables_path(cls) -> Path:
        """y_true_variables.csv 파일 경로 반환"""
        return cls.DATASET_DIR / cls.Y_TRUE_VARIABLES_FILE
    
    @classmethod
    def get_eda_results_path(cls) -> Path:
        """EDA 결과 디렉토리 경로 반환"""
        return cls.EDA_RESULTS_DIR
    
    @classmethod
    def get_variables_results_path(cls) -> Path:
        """변수 선별 결과 디렉토리 경로 반환"""
        return cls.VARIABLES_RESULTS_DIR
    
    @classmethod
    def get_variables_dir(cls) -> Path:
        """variables 디렉토리 경로 반환"""
        return cls.VARIABLES_DIR
    
    @classmethod
    def ensure_directories(cls):
        """필요한 디렉토리 생성"""
        cls.EDA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VARIABLES_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VARIABLES_DIR.mkdir(parents=True, exist_ok=True)
        cls.EDA_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========== 설정 반환 메서드 ==========
    @classmethod
    def get_arimax_config(cls) -> Dict[str, Any]:
        """ARIMAX 설정 반환"""
        return cls.ARIMAX_CONFIG.copy()
    
    @classmethod
    def get_garchx_config(cls) -> Dict[str, Any]:
        """GARCHX 설정 반환"""
        return cls.GARCHX_CONFIG.copy()
    
    @classmethod
    def get_multiple_testing_config(cls) -> Dict[str, Any]:
        """다중검정 보정 설정 반환"""
        return cls.MULTIPLE_TESTING.copy()
    
    @classmethod
    def get_eda_config(cls) -> Dict[str, Any]:
        """EDA 설정 반환"""
        return cls.EDA_CONFIG.copy()
    
    @classmethod
    def get_eda_analysis_options(cls) -> Dict[str, Any]:
        """EDA 분석 옵션 반환"""
        return cls.EDA_ANALYSIS_OPTIONS.copy()


# 전역 설정 인스턴스
settings = Settings()
