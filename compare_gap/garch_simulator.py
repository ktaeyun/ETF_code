"""
GARCH 모델 시뮬레이터
변화형 모델: z_t = mu + phi(L)z_{t-1} + eps_t, eps_t = sigma_t * e_t
sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2
"""

import numpy as np
import pandas as pd
from arch import arch_model


class GARCHSimulator:
    """
    GARCH 시뮬레이터 클래스
    변화형 모델로 z_t = Δy_t를 직접 생성
    """
    
    def __init__(self, fitted_garch_model, dist='normal'):
        """
        Args:
            fitted_garch_model: 적합된 GARCH 모델 (arch 라이브러리)
            dist: 오차항 분포 ('normal' 또는 't')
        """
        self.fitted_model = fitted_garch_model
        self.dist = dist
    
    def simulate_changes(self, T, seed=None):
        """
        변화량 z_t = Δy_t 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 변화량 z_t
        """
        if seed is not None:
            np.random.seed(seed)
        
        fitted = self.fitted_model
        
        # GARCH 파라미터 추출
        mu = fitted.params.get('mu', 0.0)
        omega = fitted.params['omega']
        alpha = fitted.params.get('alpha[1]', 0.1)
        beta = fitted.params.get('beta[1]', 0.8)
        
        # AR 파라미터 (있는 경우)
        ar_params = []
        for i in range(1, 6):  # 최대 AR(5)
            param_name = f'ar.L{i}'
            if param_name in fitted.params.index:
                ar_params.append(fitted.params[param_name])
            else:
                break
        
        # 분포 파라미터
        if 'nu' in fitted.params.index:
            nu = fitted.params['nu']
        else:
            nu = None
        
        # 초기값 설정
        initial_var = np.var(fitted.resid) if hasattr(fitted, 'resid') else 1.0
        sigma2 = initial_var
        z = np.zeros(T)
        eps = np.zeros(T)
        
        # 초기값 (AR을 위한)
        z_history = [0.0] * max(len(ar_params), 1)
        
        for t in range(T):
            # AR 항 계산
            ar_term = mu
            for i, ar_coef in enumerate(ar_params):
                if t - i - 1 >= 0:
                    ar_term += ar_coef * z[t - i - 1]
                else:
                    ar_term += ar_coef * z_history[-(i+1)]
            
            # 분산 업데이트
            if t == 0:
                sigma2_t = sigma2
            else:
                sigma2_t = omega + alpha * (eps[t-1]**2) + beta * sigma2
            
            # 오차항 생성
            if nu is not None and self.dist == 't':
                # t-분포
                e_t = np.random.standard_t(nu)
            else:
                # 정규분포
                e_t = np.random.normal(0, 1)
            
            eps[t] = np.sqrt(sigma2_t) * e_t
            
            # 변화량 생성
            z[t] = ar_term + eps[t]
            
            # 다음 반복을 위한 분산 업데이트
            sigma2 = sigma2_t
        
        return pd.Series(z)


def fit_garch_model(z_series, garch_p=1, garch_q=1, ar_order=0, dist='normal'):
    """
    실제 변화량 데이터로부터 GARCH 모델 적합
    
    Args:
        z_series: 변화량 시계열
        garch_p: GARCH p 파라미터
        garch_q: GARCH q 파라미터
        ar_order: AR 차수 (0이면 AR 없음)
        dist: 분포 ('normal' 또는 't')
    
    Returns:
        fitted GARCH 모델
    """
    print(f"\nGARCH({garch_p},{garch_q})-{dist} 모델 적합 중...")
    if ar_order > 0:
        print(f"  AR({ar_order}) 포함")
    
    try:
        # arch 라이브러리는 백분율 단위를 선호하지만, 여기서는 원래 단위 사용
        model = arch_model(
            z_series * 100,  # 백분율로 변환 (arch 라이브러리 호환성)
            mean='AR' if ar_order > 0 else 'Constant',
            lags=ar_order if ar_order > 0 else 0,
            vol='GARCH',
            p=garch_p,
            q=garch_q,
            dist=dist
        )
        fitted = model.fit(disp='off')
        
        print(f"  - 모델 적합 완료")
        print(f"  - BIC: {fitted.bic:.4f}")
        print(f"  - Log Likelihood: {fitted.loglikelihood:.4f}")
        
        return fitted
        
    except Exception as e:
        print(f"  - GARCH 모델 적합 실패: {str(e)}")
        raise


def create_garch_simulator(z_series, garch_p=1, garch_q=1, ar_order=0, dist='normal'):
    """
    GARCH 시뮬레이터 생성
    
    Args:
        z_series: 변화량 시계열
        garch_p: GARCH p 파라미터
        garch_q: GARCH q 파라미터
        ar_order: AR 차수
        dist: 분포
    
    Returns:
        tuple: (시뮬레이터 인스턴스, fitted 모델)
    """
    fitted_model = fit_garch_model(z_series, garch_p, garch_q, ar_order, dist)
    simulator = GARCHSimulator(fitted_model, dist=dist)
    return simulator, fitted_model
