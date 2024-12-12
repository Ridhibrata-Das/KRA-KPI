from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from ..analytics.ml_models import KPIPredictor, AnomalyDetector, CorrelationAnalyzer
from ..models.kpi import KPI, KPIValue
import pandas as pd
import numpy as np

class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        self.predictor = KPIPredictor()
        self.anomaly_detector = AnomalyDetector()
        self.correlation_analyzer = CorrelationAnalyzer()

    async def generate_kpi_forecast(
        self,
        kpi_id: int,
        forecast_days: int = 30
    ) -> Dict:
        """Generate forecast for a specific KPI"""
        # Get historical data
        kpi = self.db.query(KPI).filter(KPI.id == kpi_id).first()
        if not kpi:
            raise ValueError("KPI not found")

        values = self.db.query(KPIValue).filter(
            KPIValue.kpi_id == kpi_id
        ).order_by(KPIValue.timestamp).all()

        if not values:
            raise ValueError("No historical data available for forecasting")

        # Prepare data for prediction
        historical_dates = [v.timestamp for v in values]
        historical_values = [v.value for v in values]

        # Train predictor
        self.predictor.train(historical_dates, historical_values)

        # Generate future dates
        last_date = historical_dates[-1]
        future_dates = [
            last_date + timedelta(days=i)
            for i in range(1, forecast_days + 1)
        ]

        # Generate predictions
        predictions = self.predictor.predict(future_dates)

        # Detect anomalies in historical data
        anomalies = self.anomaly_detector.detect(historical_values)

        # Prepare forecast report
        historical_data = list(zip(historical_dates, historical_values))
        forecast_data = list(zip(future_dates, predictions))

        return {
            'kpi_name': kpi.name,
            'forecast': {
                'dates': [str(d) for d in future_dates],
                'values': predictions.tolist()
            },
            'historical': {
                'dates': [str(d) for d in historical_dates],
                'values': historical_values,
                'anomalies': anomalies
            },
            'metadata': {
                'forecast_days': forecast_days,
                'last_updated': datetime.utcnow().isoformat()
            }
        }

    async def analyze_correlations(
        self,
        kpi_ids: List[int],
        time_window_days: int = 30
    ) -> Dict:
        """Analyze correlations between multiple KPIs"""
        kpi_data = {}
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)

        for kpi_id in kpi_ids:
            values = self.db.query(KPIValue).filter(
                KPIValue.kpi_id == kpi_id,
                KPIValue.timestamp >= cutoff_date
            ).order_by(KPIValue.timestamp).all()

            if values:
                kpi = self.db.query(KPI).filter(KPI.id == kpi_id).first()
                kpi_data[kpi.name] = [v.value for v in values]

        if not kpi_data:
            raise ValueError("No data available for correlation analysis")

        correlation_matrix = self.correlation_analyzer.analyze(kpi_data)
        top_correlations = self.correlation_analyzer.get_top_correlations(
            correlation_matrix
        )

        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'top_correlations': top_correlations,
            'metadata': {
                'time_window_days': time_window_days,
                'kpi_count': len(kpi_ids),
                'last_updated': datetime.utcnow().isoformat()
            }
        }

    async def export_analytics_report(
        self,
        kpi_id: int,
        format: str = 'pdf'
    ) -> bytes:
        """Generate exportable analytics report"""
        # Get KPI data
        forecast_data = await self.generate_kpi_forecast(kpi_id)
        
        if format.lower() == 'pdf':
            # Generate PDF report
            # Implementation depends on your PDF generation library
            pass
        elif format.lower() == 'excel':
            # Generate Excel report
            df = pd.DataFrame({
                'Date': forecast_data['forecast']['dates'],
                'Forecasted Value': forecast_data['forecast']['values'],
                'Historical Date': forecast_data['historical']['dates'],
                'Historical Value': forecast_data['historical']['values'],
                'Is Anomaly': forecast_data['historical']['anomalies']
            })
            
            # Save to bytes buffer
            buffer = pd.io.excel.ExcelWriter(
                path='temp.xlsx',
                engine='openpyxl'
            )
            df.to_excel(buffer)
            return buffer.book
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def detect_real_time_anomalies(
        self,
        kpi_id: int,
        window_size: int = 24
    ) -> Dict:
        """Detect anomalies in real-time KPI data"""
        # Get recent values
        recent_values = self.db.query(KPIValue).filter(
            KPIValue.kpi_id == kpi_id
        ).order_by(
            KPIValue.timestamp.desc()
        ).limit(window_size).all()

        if not recent_values:
            raise ValueError("No recent data available for anomaly detection")

        values = [v.value for v in recent_values]
        anomalies = self.anomaly_detector.detect(values)

        return {
            'timestamps': [str(v.timestamp) for v in recent_values],
            'values': values,
            'anomalies': anomalies,
            'metadata': {
                'window_size': window_size,
                'anomaly_count': sum(anomalies),
                'last_updated': datetime.utcnow().isoformat()
            }
        }
