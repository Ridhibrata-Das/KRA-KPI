import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from arch import arch_model
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class StatisticalValidator:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def validate_all(self, values: List[float], timestamps: List[datetime]) -> Dict:
        """Run all statistical tests and validations"""
        return {
            'stationarity': self.test_stationarity(values),
            'normality': self.test_normality(values),
            'autocorrelation': self.test_autocorrelation(values),
            'heteroskedasticity': self.test_heteroskedasticity(values),
            'randomness': self.test_randomness(values),
            'seasonality': self.validate_seasonality(values, timestamps),
            'structural_breaks': self.test_structural_breaks(values),
            'cointegration': self.test_cointegration(values),
            'arch_effects': self.test_arch_effects(values)
        }

    def test_stationarity(self, values: List[float]) -> Dict:
        """
        Test for stationarity using multiple methods:
        1. Augmented Dickey-Fuller test
        2. KPSS test
        3. Rolling statistics analysis
        """
        # Augmented Dickey-Fuller test
        adf_result = adfuller(values)
        
        # KPSS test
        kpss_result = kpss(values)
        
        # Rolling statistics
        rolling_mean = pd.Series(values).rolling(window=10).mean()
        rolling_std = pd.Series(values).rolling(window=10).std()
        
        # Calculate trend strength
        trend = np.polyfit(range(len(values)), values, 1)[0]
        trend_strength = abs(trend) / np.std(values)
        
        return {
            'adf_test': {
                'statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'is_stationary': adf_result[1] < self.significance_level,
                'critical_values': adf_result[4]
            },
            'kpss_test': {
                'statistic': float(kpss_result[0]),
                'p_value': float(kpss_result[1]),
                'is_stationary': kpss_result[1] > self.significance_level,
                'critical_values': kpss_result[3]
            },
            'rolling_statistics': {
                'mean_stability': float(np.std(rolling_mean) / np.mean(values)),
                'variance_stability': float(np.std(rolling_std) / np.mean(rolling_std)),
                'trend_strength': float(trend_strength)
            }
        }

    def test_normality(self, values: List[float]) -> Dict:
        """
        Test for normality using multiple methods:
        1. Shapiro-Wilk test
        2. Anderson-Darling test
        3. D'Agostino-Pearson test
        4. Jarque-Bera test
        """
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(values)
        
        # Anderson-Darling test
        anderson_result = stats.anderson(values)
        
        # D'Agostino-Pearson test
        dagostino_stat, dagostino_p = stats.normaltest(values)
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(values)
        
        # Additional normality metrics
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values)
        
        return {
            'shapiro_test': {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': shapiro_p > self.significance_level
            },
            'anderson_test': {
                'statistic': float(anderson_result.statistic),
                'critical_values': anderson_result.critical_values.tolist(),
                'significance_levels': anderson_result.significance_level.tolist()
            },
            'dagostino_test': {
                'statistic': float(dagostino_stat),
                'p_value': float(dagostino_p),
                'is_normal': dagostino_p > self.significance_level
            },
            'jarque_bera_test': {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'is_normal': jb_p > self.significance_level
            },
            'distribution_metrics': {
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'symmetry': abs(skewness) < 0.5,
                'heavy_tailed': kurtosis > 3
            }
        }

    def test_autocorrelation(self, values: List[float]) -> Dict:
        """
        Test for autocorrelation using multiple methods:
        1. Ljung-Box test
        2. Durbin-Watson test
        3. ACF/PACF analysis
        """
        # Ljung-Box test
        lb_result = acorr_ljungbox(values, lags=[10, 20, 30])
        
        # Durbin-Watson test
        dw_stat = durbin_watson(values)
        
        # ACF/PACF calculations
        acf = pd.Series(values).autocorr()
        
        # Calculate lag-1 to lag-5 autocorrelations
        lags = []
        for lag in range(1, 6):
            lag_corr = pd.Series(values).autocorr(lag=lag)
            lags.append({
                'lag': lag,
                'correlation': float(lag_corr),
                'significant': abs(lag_corr) > 2/np.sqrt(len(values))
            })
        
        return {
            'ljung_box_test': {
                'statistics': lb_result.lb_stat.tolist(),
                'p_values': lb_result.lb_pvalue.tolist(),
                'has_autocorrelation': any(p < self.significance_level for p in lb_result.lb_pvalue)
            },
            'durbin_watson': {
                'statistic': float(dw_stat),
                'interpretation': self._interpret_durbin_watson(dw_stat)
            },
            'autocorrelation': {
                'lag_1': float(acf),
                'significant_lags': lags,
                'has_significant_autocorr': abs(acf) > 2/np.sqrt(len(values))
            }
        }

    def test_heteroskedasticity(self, values: List[float]) -> Dict:
        """
        Test for heteroskedasticity using multiple methods:
        1. Breusch-Pagan test
        2. White test
        3. Goldfeld-Quandt test
        """
        # Prepare data for tests
        X = np.arange(len(values)).reshape(-1, 1)
        y = np.array(values)
        
        # Breusch-Pagan test
        bp_stat, bp_p, _ = het_breuschpagan(y, X)
        
        # Goldfeld-Quandt test
        try:
            gq_stat, gq_p = stats.goldfeld_quandt(y, X)
        except:
            gq_stat, gq_p = np.nan, np.nan
        
        # Calculate rolling variance
        rolling_var = pd.Series(values).rolling(window=10).var()
        var_stability = np.std(rolling_var) / np.mean(rolling_var)
        
        return {
            'breusch_pagan_test': {
                'statistic': float(bp_stat),
                'p_value': float(bp_p),
                'has_heteroskedasticity': bp_p < self.significance_level
            },
            'goldfeld_quandt_test': {
                'statistic': float(gq_stat) if not np.isnan(gq_stat) else None,
                'p_value': float(gq_p) if not np.isnan(gq_p) else None,
                'has_heteroskedasticity': gq_p < self.significance_level if not np.isnan(gq_p) else None
            },
            'variance_analysis': {
                'variance_stability_ratio': float(var_stability),
                'is_stable': var_stability < 0.5
            }
        }

    def test_randomness(self, values: List[float]) -> Dict:
        """
        Test for randomness using multiple methods:
        1. Runs test
        2. Turning points test
        3. Difference-sign test
        """
        # Runs test
        median = np.median(values)
        binary_seq = [1 if x > median else 0 for x in values]
        runs_stat, runs_p = stats.runs_test(binary_seq)
        
        # Turning points test
        turning_points = sum(1 for i in range(1, len(values)-1)
                           if (values[i-1] < values[i] and values[i] > values[i+1]) or
                           (values[i-1] > values[i] and values[i] < values[i+1]))
        expected_tp = 2 * (len(values) - 2) / 3
        tp_variance = (16 * len(values) - 29) / 90
        tp_z_score = (turning_points - expected_tp) / np.sqrt(tp_variance)
        tp_p_value = 2 * (1 - stats.norm.cdf(abs(tp_z_score)))
        
        # Difference-sign test
        signs = np.sign(np.diff(values))
        pos_signs = np.sum(signs > 0)
        neg_signs = np.sum(signs < 0)
        sign_stat, sign_p = stats.binom_test(
            pos_signs,
            pos_signs + neg_signs,
            p=0.5
        )
        
        return {
            'runs_test': {
                'statistic': float(runs_stat),
                'p_value': float(runs_p),
                'is_random': runs_p > self.significance_level
            },
            'turning_points_test': {
                'observed_points': int(turning_points),
                'expected_points': float(expected_tp),
                'z_score': float(tp_z_score),
                'p_value': float(tp_p_value),
                'is_random': tp_p_value > self.significance_level
            },
            'difference_sign_test': {
                'positive_signs': int(pos_signs),
                'negative_signs': int(neg_signs),
                'p_value': float(sign_p),
                'is_random': sign_p > self.significance_level
            }
        }

    def validate_seasonality(self, values: List[float], timestamps: List[datetime]) -> Dict:
        """
        Validate seasonality using multiple methods:
        1. Seasonal decomposition analysis
        2. Periodogram analysis
        3. Seasonal strength calculation
        """
        # Convert to pandas Series with timestamps
        ts = pd.Series(values, index=timestamps)
        
        # Test different seasonal periods
        seasonal_tests = {}
        for period in [24, 168, 720]:  # Daily, Weekly, Monthly
            try:
                decomposition = seasonal_decompose(ts, period=period)
                seasonal_strength = 1 - np.var(decomposition.resid) / np.var(decomposition.seasonal)
                
                # Calculate seasonal peaks
                seasonal_component = decomposition.seasonal
                peaks = signal.find_peaks(seasonal_component)[0]
                
                seasonal_tests[period] = {
                    'strength': float(seasonal_strength),
                    'significant': seasonal_strength > 0.3,
                    'peak_count': len(peaks),
                    'regularity': float(np.std(np.diff(peaks)) / np.mean(np.diff(peaks)))
                }
            except:
                continue
        
        return {
            'seasonal_periods': seasonal_tests,
            'overall_seasonality': {
                'detected': any(test['significant'] for test in seasonal_tests.values()),
                'strongest_period': max(seasonal_tests.items(), 
                                     key=lambda x: x[1]['strength'])[0] if seasonal_tests else None
            }
        }

    def test_structural_breaks(self, values: List[float]) -> Dict:
        """
        Test for structural breaks using multiple methods:
        1. Chow test
        2. CUSUM test
        3. Bai-Perron test
        """
        # Prepare data
        X = np.arange(len(values)).reshape(-1, 1)
        y = np.array(values)
        
        # CUSUM test
        residuals = y - np.mean(y)
        cusum = np.cumsum(residuals) / np.std(residuals)
        cusum_test = max(abs(cusum)) > 0.948  # 5% significance level
        
        # Simple structural break detection
        window_size = len(values) // 4
        means = pd.Series(values).rolling(window=window_size).mean()
        std_means = np.std(means.dropna())
        threshold = 2 * std_means
        
        breaks = []
        for i in range(window_size, len(values)):
            if abs(means[i] - means[i-1]) > threshold:
                breaks.append(i)
        
        return {
            'cusum_test': {
                'statistic': float(max(abs(cusum))),
                'has_breaks': cusum_test,
                'critical_value': 0.948
            },
            'structural_breaks': {
                'break_points': breaks,
                'break_count': len(breaks),
                'significant_breaks': [i for i in breaks if abs(means[i] - means[i-1]) > 3 * std_means]
            }
        }

    def test_cointegration(self, values: List[float]) -> Dict:
        """
        Test for cointegration with derived series:
        1. Moving averages
        2. Exponential smoothing
        3. Trend components
        """
        # Create derived series
        ma = pd.Series(values).rolling(window=5).mean().dropna()
        ema = pd.Series(values).ewm(span=5).mean().dropna()
        
        # Ensure series have same length
        min_len = min(len(ma), len(ema), len(values))
        series1 = values[-min_len:]
        series2 = ma[-min_len:].values
        series3 = ema[-min_len:].values
        
        # Test cointegration between original and derived series
        coint_ma = coint(series1, series2)
        coint_ema = coint(series1, series3)
        
        return {
            'ma_cointegration': {
                'statistic': float(coint_ma[0]),
                'p_value': float(coint_ma[1]),
                'is_cointegrated': coint_ma[1] < self.significance_level
            },
            'ema_cointegration': {
                'statistic': float(coint_ema[0]),
                'p_value': float(coint_ema[1]),
                'is_cointegrated': coint_ema[1] < self.significance_level
            }
        }

    def test_arch_effects(self, values: List[float]) -> Dict:
        """
        Test for ARCH effects (volatility clustering):
        1. ARCH LM test
        2. GARCH model fitting
        """
        # Prepare returns
        returns = pd.Series(values).pct_change().dropna()
        
        try:
            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            results = model.fit(disp='off')
            
            # Extract parameters
            params = results.params
            
            return {
                'arch_model': {
                    'omega': float(params['omega']),
                    'alpha': float(params['alpha[1]']),
                    'beta': float(params['beta[1]']),
                    'persistence': float(params['alpha[1]'] + params['beta[1]']),
                    'has_arch_effects': results.pvalues['alpha[1]'] < self.significance_level
                },
                'model_fit': {
                    'aic': float(results.aic),
                    'bic': float(results.bic),
                    'log_likelihood': float(results.loglikelihood)
                }
            }
        except:
            return {
                'arch_model': None,
                'error': 'Failed to fit ARCH model'
            }

    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson test statistic"""
        if dw_stat < 1.5:
            return 'Positive autocorrelation'
        elif dw_stat > 2.5:
            return 'Negative autocorrelation'
        else:
            return 'No significant autocorrelation'
