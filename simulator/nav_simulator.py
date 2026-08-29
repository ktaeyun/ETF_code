"""
NAV 시뮬레이터 메인 모듈
Poisson-Gaussian 또는 Poisson-GARCH 시뮬레이터
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist
from arch import arch_model


class NAVSimulator:
    """
    NAV 시뮬레이터 클래스
    """
    
    def __init__(self, jump_params, continuous_params, lambda_param):
        """
        Args:
            jump_params: 점프 분포 파라미터 (dict)
            continuous_params: 연속 성분 파라미터 (dict)
            lambda_param: 점프 강도 λ
        """
        self.jump_params = jump_params
        self.continuous_params = continuous_params
        self.lambda_param = lambda_param
    
    def simulate_returns(self, T, seed=None):
        """
        수익률 시뮬레이션: r = r_cont + sum(Jumps)
        
        Args:
            T: 시뮬레이션 기간 (일수)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 수익률
        """
        if seed is not None:
            np.random.seed(seed)
        
        returns = np.zeros(T)
        
        # 연속 성분 생성
        if self.continuous_params['model_type'] == 'normal':
            mu = self.continuous_params['mu']
            sigma = self.continuous_params['sigma']
            r_cont = np.random.normal(mu, sigma, T)
        
        elif self.continuous_params['model_type'] == 'garch':
            # GARCH 모델에서 시뮬레이션
            r_cont = self._simulate_garch(T)
        
        else:
            raise ValueError(f"지원하지 않는 연속 성분 모델: {self.continuous_params['model_type']}")
        
        # 점프 생성
        jumps = self._simulate_jumps(T)
        
        # 최종 수익률: r = r_cont + sum(Jumps)
        returns = r_cont + jumps
        
        return pd.Series(returns)
    
    def _simulate_garch(self, T):
        """
        GARCH 모델에서 수익률 시뮬레이션
        """
        fitted = self.continuous_params['fitted_model']
        
        # GARCH 파라미터 추출
        omega = fitted.params['omega']
        alpha = fitted.params.get('alpha[1]', 0.1)
        beta = fitted.params.get('beta[1]', 0.8)
        
        # 분포 파라미터
        if self.continuous_params['dist'] == 't':
            nu = fitted.params.get('nu', 10.0)
        else:
            nu = None
        
        # 초기값
        sigma2 = np.var(fitted.resid) / 10000  # 백분율 단위 조정
        returns = np.zeros(T)
        
        for t in range(T):
            # 분산 업데이트
            if t == 0:
                sigma2_t = sigma2
            else:
                sigma2_t = omega + alpha * (returns[t-1] ** 2) + beta * sigma2
            
            # 수익률 생성
            if nu is not None:
                # t-분포
                z = np.random.standard_t(nu)
            else:
                # 정규분포
                z = np.random.normal(0, 1)
            
            returns[t] = np.sqrt(sigma2_t) * z / 100  # 백분율 단위 조정
        
        return returns
    
    def _simulate_jumps(self, T):
        """
        점프 생성: Poisson 과정으로 점프 발생 시점 결정, 점프 크기는 분포에서 샘플링
        """
        jumps = np.zeros(T)
        
        # Poisson 과정으로 점프 발생 시점 결정
        jump_times = []
        t = 0
        while t < T:
            # 다음 점프까지의 간격 (지수분포)
            interval = np.random.exponential(1 / self.lambda_param)
            t += interval
            if t < T:
                jump_times.append(int(t))
        
        # 점프 크기 샘플링
        if len(jump_times) > 0:
            if self.jump_params['dist_type'] == 'normal':
                mu_j = self.jump_params['mu']
                sigma_j = self.jump_params['sigma']
                jump_sizes = np.random.normal(mu_j, sigma_j, len(jump_times))
            
            elif self.jump_params['dist_type'] == 't':
                mu_j = self.jump_params['mu']
                sigma_j = self.jump_params['sigma']
                nu_j = self.jump_params['nu']
                # t-분포 샘플링 (위치-스케일 변환)
                z = np.random.standard_t(nu_j)
                jump_sizes = mu_j + sigma_j * z
            
            # 점프 할당
            for i, t_jump in enumerate(jump_times):
                jumps[t_jump] += jump_sizes[i]  # 여러 점프가 같은 시점에 발생할 수 있음
        
        return jumps
    
    def simulate_nav_path(self, S0, T, seed=None):
        """
        NAV 경로 시뮬레이션: S_{t+1} = S_t * exp(r_t)
        
        Args:
            S0: 초기 NAV
            T: 시뮬레이션 기간 (일수)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 NAV 경로
        """
        returns = self.simulate_returns(T, seed=seed)
        
        # NAV 경로 생성
        nav_path = np.zeros(T + 1)
        nav_path[0] = S0
        
        for t in range(T):
            nav_path[t + 1] = nav_path[t] * np.exp(returns.iloc[t])
        
        return pd.Series(nav_path[1:], index=range(T))


def create_simulator(jump_detection_result, continuous_fit_result):
    """
    시뮬레이터 생성
    
    Args:
        jump_detection_result: 점프 감지 결과
        continuous_fit_result: 연속 성분 적합 결과
    
    Returns:
        NAVSimulator 인스턴스
    """
    # 점프 분포 추정
    from simulator.jump_detector import estimate_jump_distribution
    
    jump_returns = jump_detection_result['jump_returns']
    jump_params = estimate_jump_distribution(jump_returns, dist_type='normal')
    
    # 시뮬레이터 생성
    simulator = NAVSimulator(
        jump_params=jump_params,
        continuous_params=continuous_fit_result,
        lambda_param=jump_detection_result['lambda']
    )
    
    return simulator
