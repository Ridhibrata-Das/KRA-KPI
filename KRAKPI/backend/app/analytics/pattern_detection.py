import numpy as np
from scipy import stats, signal
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from dtaidistance import dtw
from scipy.stats import linregress

class PatternDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def detect_all_patterns(self, values: List[float], timestamps: List[datetime]) -> Dict:
        """Detect multiple types of patterns in the time series"""
        results = {
            'seasonality': self.detect_seasonality(values, timestamps),
            'trends': self.detect_trends(values),
            'cycles': self.detect_cycles(values),
            'outliers': self.detect_outliers(values),
            'change_points': self.detect_change_points(values),
            'volatility_clusters': self.detect_volatility_clusters(values),
            'repeated_patterns': self.detect_repeated_patterns(values)
        }
        return results

    def detect_seasonality(self, values: List[float], timestamps: List[datetime]) -> Dict:
        """Detect multiple types of seasonality (daily, weekly, monthly)"""
        df = pd.DataFrame({
            'value': values,
            'timestamp': timestamps
        })
        df.set_index('timestamp', inplace=True)
        
        seasonal_patterns = {}
        for period in [24, 168, 720]:  # Daily (24h), Weekly (168h), Monthly (720h)
            try:
                if len(values) >= period * 2:
                    decomposition = seasonal_decompose(
                        df['value'],
                        period=period,
                        extrapolate_trend='freq'
                    )
                    seasonal_strength = 1 - np.var(decomposition.resid) / np.var(decomposition.seasonal)
                    seasonal_patterns[period] = {
                        'strength': seasonal_strength,
                        'pattern': decomposition.seasonal.tolist(),
                        'is_significant': seasonal_strength > 0.3
                    }
            except:
                continue
                
        return seasonal_patterns

    def detect_trends(self, values: List[float]) -> Dict:
        """Detect various types of trends"""
        x = np.arange(len(values))
        
        # Linear trend
        slope, intercept, r_value, p_value, std_err = linregress(x, values)
        
        # Polynomial trends
        poly_trends = {}
        for degree in [2, 3]:
            coeffs = np.polyfit(x, values, degree)
            poly_fit = np.polyval(coeffs, x)
            residuals = values - poly_fit
            r_squared = 1 - (np.sum(residuals**2) / np.sum((values - np.mean(values))**2))
            poly_trends[f'degree_{degree}'] = {
                'coefficients': coeffs.tolist(),
                'r_squared': r_squared
            }
        
        return {
            'linear': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            },
            'polynomial': poly_trends
        }

    def detect_cycles(self, values: List[float]) -> Dict:
        """Detect cyclical patterns using spectral analysis"""
        if len(values) < 4:
            return {'cycles_detected': False}
            
        # Perform FFT
        fft_values = np.fft.fft(values)
        frequencies = np.fft.fftfreq(len(values))
        
        # Find dominant frequencies
        power_spectrum = np.abs(fft_values)**2
        dominant_freq_idx = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies
        
        cycles = []
        for idx in dominant_freq_idx:
            if frequencies[idx] > 0:  # Only positive frequencies
                period = 1 / frequencies[idx]
                amplitude = np.abs(fft_values[idx]) / len(values)
                cycles.append({
                    'period': period,
                    'amplitude': amplitude,
                    'power': power_spectrum[idx]
                })
        
        return {
            'cycles_detected': len(cycles) > 0,
            'dominant_cycles': sorted(cycles, key=lambda x: x['power'], reverse=True)
        }

    def detect_outliers(self, values: List[float]) -> Dict:
        """Detect outliers using multiple methods"""
        # Z-score method
        z_scores = stats.zscore(values)
        z_score_outliers = np.where(np.abs(z_scores) > 3)[0]
        
        # IQR method
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        iqr_outliers = np.where((values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR))[0]
        
        # Modified Z-score method
        median = np.median(values)
        mad = stats.median_abs_deviation(values)
        modified_z_scores = 0.6745 * (values - median) / mad
        modified_z_outliers = np.where(np.abs(modified_z_scores) > 3.5)[0]
        
        return {
            'z_score': z_score_outliers.tolist(),
            'iqr': iqr_outliers.tolist(),
            'modified_z_score': modified_z_outliers.tolist(),
            'consensus_outliers': list(set(z_score_outliers) & set(iqr_outliers) & set(modified_z_outliers))
        }

    def detect_change_points(self, values: List[float]) -> Dict:
        """Detect points where the time series characteristics change"""
        if len(values) < 10:
            return {'change_points': []}
            
        # Detect mean shifts
        window_size = max(5, len(values) // 10)
        means = pd.Series(values).rolling(window=window_size).mean()
        mean_diff = np.abs(means.diff())
        mean_changes = signal.find_peaks(mean_diff, height=np.std(values))[0]
        
        # Detect variance changes
        variances = pd.Series(values).rolling(window=window_size).std()
        var_diff = np.abs(variances.diff())
        variance_changes = signal.find_peaks(var_diff, height=np.std(values))[0]
        
        return {
            'mean_changes': mean_changes.tolist(),
            'variance_changes': variance_changes.tolist(),
            'all_changes': sorted(list(set(mean_changes) | set(variance_changes)))
        }

    def detect_volatility_clusters(self, values: List[float]) -> Dict:
        """Detect periods of high volatility"""
        if len(values) < 4:
            return {'clusters': []}
            
        # Calculate rolling volatility
        returns = np.diff(values) / values[:-1]
        volatility = pd.Series(returns).rolling(window=5).std()
        
        # Identify high volatility periods
        high_vol_threshold = np.mean(volatility) + 2 * np.std(volatility)
        is_high_vol = volatility > high_vol_threshold
        
        # Find clusters
        clusters = []
        start_idx = None
        
        for i, is_high in enumerate(is_high_vol):
            if is_high and start_idx is None:
                start_idx = i
            elif not is_high and start_idx is not None:
                clusters.append({
                    'start': start_idx,
                    'end': i,
                    'duration': i - start_idx,
                    'avg_volatility': float(np.mean(volatility[start_idx:i]))
                })
                start_idx = None
        
        return {
            'clusters': clusters,
            'avg_volatility': float(np.mean(volatility)),
            'volatility_series': volatility.tolist()
        }

    def detect_repeated_patterns(self, values: List[float]) -> Dict:
        """Detect recurring patterns using dynamic time warping"""
        if len(values) < 10:
            return {'patterns': []}
            
        window_size = min(10, len(values) // 4)
        step_size = max(1, window_size // 2)
        
        # Extract windows
        windows = []
        for i in range(0, len(values) - window_size, step_size):
            windows.append(values[i:i + window_size])
        
        # Compare windows using DTW
        patterns = []
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                distance = dtw.distance(windows[i], windows[j])
                similarity = 1 / (1 + distance)
                
                if similarity > 0.8:  # High similarity threshold
                    patterns.append({
                        'window1_start': i * step_size,
                        'window1_end': i * step_size + window_size,
                        'window2_start': j * step_size,
                        'window2_end': j * step_size + window_size,
                        'similarity': float(similarity)
                    })
        
        return {
            'patterns': sorted(patterns, key=lambda x: x['similarity'], reverse=True),
            'pattern_count': len(patterns)
        }

    def get_pattern_summary(self, values: List[float], timestamps: List[datetime]) -> Dict:
        """Generate a comprehensive summary of all detected patterns"""
        all_patterns = self.detect_all_patterns(values, timestamps)
        
        summary = {
            'primary_trend': self._get_primary_trend(all_patterns['trends']),
            'seasonality': self._summarize_seasonality(all_patterns['seasonality']),
            'significant_cycles': self._summarize_cycles(all_patterns['cycles']),
            'outlier_count': len(all_patterns['outliers']['consensus_outliers']),
            'volatility_status': self._summarize_volatility(all_patterns['volatility_clusters']),
            'pattern_complexity': self._calculate_pattern_complexity(all_patterns)
        }
        
        return summary

    def _get_primary_trend(self, trend_data: Dict) -> str:
        linear_trend = trend_data['linear']
        if not linear_trend['is_significant']:
            return 'No significant trend'
        
        slope = linear_trend['slope']
        if abs(slope) < 0.01:
            return 'Stable'
        elif slope > 0:
            return f'Increasing (slope: {slope:.3f})'
        else:
            return f'Decreasing (slope: {slope:.3f})'

    def _summarize_seasonality(self, seasonality_data: Dict) -> List[str]:
        summary = []
        period_names = {24: 'Daily', 168: 'Weekly', 720: 'Monthly'}
        
        for period, data in seasonality_data.items():
            if data['is_significant']:
                summary.append(f"{period_names.get(period, 'Unknown')} seasonality detected")
        
        return summary if summary else ['No significant seasonality detected']

    def _summarize_cycles(self, cycle_data: Dict) -> List[Dict]:
        if not cycle_data['cycles_detected']:
            return []
        
        return [
            {
                'period': cycle['period'],
                'strength': cycle['amplitude']
            }
            for cycle in cycle_data['dominant_cycles'][:3]  # Top 3 cycles
        ]

    def _summarize_volatility(self, volatility_data: Dict) -> str:
        clusters = volatility_data['clusters']
        if not clusters:
            return 'Stable'
        
        total_volatile_periods = sum(c['duration'] for c in clusters)
        volatility_percentage = total_volatile_periods / len(clusters) * 100
        
        if volatility_percentage < 10:
            return 'Low volatility'
        elif volatility_percentage < 30:
            return 'Moderate volatility'
        else:
            return 'High volatility'

    def _calculate_pattern_complexity(self, all_patterns: Dict) -> str:
        complexity_score = 0
        
        # Add points for each type of pattern
        if all_patterns['trends']['linear']['is_significant']:
            complexity_score += 1
        
        for season in all_patterns['seasonality'].values():
            if season['is_significant']:
                complexity_score += 2
        
        if all_patterns['cycles']['cycles_detected']:
            complexity_score += len(all_patterns['cycles']['dominant_cycles'])
        
        complexity_score += len(all_patterns['change_points']['all_changes']) * 0.5
        complexity_score += len(all_patterns['volatility_clusters']['clusters'])
        
        if complexity_score < 3:
            return 'Simple'
        elif complexity_score < 6:
            return 'Moderate'
        else:
            return 'Complex'
