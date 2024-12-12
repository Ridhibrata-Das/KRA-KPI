import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Tuple

class KPIPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def prepare_time_features(self, dates: List[datetime]) -> np.ndarray:
        """Convert datetime features into numerical values for ML model"""
        features = []
        for date in dates:
            features.append([
                date.timestamp(),
                date.month,
                date.weekday(),
                date.hour if hasattr(date, 'hour') else 0
            ])
        return np.array(features)
    
    def train(self, dates: List[datetime], values: List[float]):
        """Train the forecasting model"""
        X = self.prepare_time_features(dates)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, values)
    
    def predict(self, future_dates: List[datetime]) -> List[float]:
        """Predict future KPI values"""
        X = self.prepare_time_features(future_dates)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class AnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def detect(self, values: List[float]) -> List[bool]:
        """Detect anomalies in KPI values"""
        values_reshaped = np.array(values).reshape(-1, 1)
        predictions = self.model.fit_predict(values_reshaped)
        # Convert predictions to boolean (True for anomalies)
        return [pred == -1 for pred in predictions]

class CorrelationAnalyzer:
    @staticmethod
    def analyze(kpi_data: Dict[str, List[float]]) -> pd.DataFrame:
        """Analyze correlations between different KPIs"""
        df = pd.DataFrame(kpi_data)
        correlation_matrix = df.corr()
        return correlation_matrix
    
    @staticmethod
    def get_top_correlations(correlation_matrix: pd.DataFrame, threshold: float = 0.5) -> List[Dict]:
        """Get top correlated KPI pairs"""
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                value = correlation_matrix.iloc[i, j]
                if abs(value) >= threshold:
                    correlations.append({
                        'kpi1': correlation_matrix.columns[i],
                        'kpi2': correlation_matrix.columns[j],
                        'correlation': value
                    })
        return sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)

def generate_forecast_report(
    kpi_name: str,
    historical_data: List[Tuple[datetime, float]],
    forecast_data: List[Tuple[datetime, float]],
    anomalies: List[bool]
) -> Dict:
    """Generate a comprehensive forecast report"""
    return {
        'kpi_name': kpi_name,
        'historical_data': {
            'dates': [str(date) for date, _ in historical_data],
            'values': [value for _, value in historical_data],
            'anomalies': anomalies
        },
        'forecast_data': {
            'dates': [str(date) for date, _ in forecast_data],
            'values': [value for _, value in forecast_data]
        },
        'statistics': {
            'current_value': historical_data[-1][1] if historical_data else None,
            'avg_value': np.mean([value for _, value in historical_data]) if historical_data else None,
            'trend': 'increasing' if forecast_data[-1][1] > historical_data[-1][1] else 'decreasing',
            'anomaly_count': sum(anomalies)
        }
    }
