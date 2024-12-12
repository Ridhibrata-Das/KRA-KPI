from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from .pattern_prediction import PatternPredictor
from .prediction_visualizer import PredictionVisualizer
from .prediction_alert_system import PredictionAlertSystem, AlertType, AlertSeverity, PredictionAlert
from .statistical_validation import StatisticalValidator

@dataclass
class AnalysisResult:
    kpi_name: str
    timestamp: datetime
    predictions: Dict
    alerts: List[PredictionAlert]
    accuracy_metrics: Dict
    statistical_tests: Dict
    metadata: Dict

    def to_dict(self) -> Dict:
        """Convert analysis result to dictionary format"""
        return {
            'kpi_name': self.kpi_name,
            'timestamp': self.timestamp.isoformat(),
            'predictions': self.predictions,
            'alerts': [alert.to_dict() for alert in self.alerts],
            'accuracy_metrics': self.accuracy_metrics,
            'statistical_tests': self.statistical_tests,
            'metadata': self.metadata
        }

class UnifiedAnalytics:
    def __init__(self,
                 output_dir: str = './analytics_output',
                 config: Optional[Dict] = None):
        """Initialize the unified analytics system"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.predictor = PatternPredictor(
            sequence_length=config.get('sequence_length', 30) if config else 30,
            forecast_horizon=config.get('forecast_horizon', 7) if config else 7
        )
        self.visualizer = PredictionVisualizer(
            theme=config.get('theme', 'plotly_white') if config else 'plotly_white'
        )
        self.alert_system = PredictionAlertSystem(
            threshold_config=config.get('threshold_config', {}) if config else {},
            trend_sensitivity=config.get('trend_sensitivity', 0.1) if config else 0.1,
            uncertainty_threshold=config.get('uncertainty_threshold', 0.2) if config else 0.2
        )
        self.statistical_validator = StatisticalValidator()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / 'analytics.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def analyze_kpi(self,
                   kpi_name: str,
                   values: List[float],
                   timestamps: List[datetime],
                   metadata: Optional[Dict] = None) -> AnalysisResult:
        """Perform comprehensive KPI analysis"""
        self.logger.info(f"Starting analysis for KPI: {kpi_name}")
        
        try:
            # Validate input data
            if len(values) != len(timestamps):
                raise ValueError("Length of values and timestamps must match")
            if len(values) < self.predictor.sequence_length:
                raise ValueError(f"Insufficient data points. Need at least {self.predictor.sequence_length}")

            # Step 1: Statistical Validation
            self.logger.info("Performing statistical validation")
            statistical_tests = self.statistical_validator.validate_timeseries(
                values, timestamps
            )

            # Step 2: Train Prediction Models
            self.logger.info("Training prediction models")
            training_results = self.predictor.train_models(values, timestamps)

            # Step 3: Generate Predictions
            self.logger.info("Generating predictions")
            future_timestamps = [
                timestamps[-1] + timedelta(days=i+1)
                for i in range(self.predictor.forecast_horizon)
            ]
            predictions = self.predictor.predict_patterns(
                values, timestamps
            )

            # Step 4: Analyze Prediction Accuracy
            self.logger.info("Analyzing prediction accuracy")
            accuracy_metrics = self.predictor.analyze_prediction_accuracy(
                values[-self.predictor.forecast_horizon:],
                predictions
            )

            # Step 5: Generate Alerts
            self.logger.info("Generating alerts")
            alerts = self.alert_system.analyze_predictions(
                kpi_name,
                predictions,
                values,
                future_timestamps
            )

            # Step 6: Create Visualizations
            self.logger.info("Creating visualizations")
            dashboard = self.visualizer.create_prediction_dashboard(
                values,
                timestamps,
                predictions,
                future_timestamps,
                accuracy_metrics,
                title=f"KPI Analysis Dashboard: {kpi_name}"
            )

            # Export results
            self._export_results(
                kpi_name,
                dashboard,
                predictions,
                alerts,
                accuracy_metrics,
                statistical_tests,
                metadata
            )

            # Create analysis result
            result = AnalysisResult(
                kpi_name=kpi_name,
                timestamp=datetime.now(),
                predictions=predictions,
                alerts=alerts,
                accuracy_metrics=accuracy_metrics,
                statistical_tests=statistical_tests,
                metadata={
                    'training_results': training_results,
                    'user_metadata': metadata or {},
                    'analysis_parameters': {
                        'sequence_length': self.predictor.sequence_length,
                        'forecast_horizon': self.predictor.forecast_horizon
                    }
                }
            )

            self.logger.info(f"Analysis completed for KPI: {kpi_name}")
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing KPI {kpi_name}: {str(e)}", exc_info=True)
            raise

    def _export_results(self,
                       kpi_name: str,
                       dashboard: 'plotly.graph_objects.Figure',
                       predictions: Dict,
                       alerts: List[PredictionAlert],
                       accuracy_metrics: Dict,
                       statistical_tests: Dict,
                       metadata: Optional[Dict] = None):
        """Export analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.output_dir / kpi_name / timestamp
        base_path.mkdir(parents=True, exist_ok=True)

        # Export dashboard
        dashboard.write_html(base_path / "dashboard.html")

        # Export predictions
        with open(base_path / "predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2)

        # Export alerts
        self.alert_system.export_alerts(base_path / "alerts.json")

        # Export metrics and tests
        with open(base_path / "analysis_results.json", 'w') as f:
            json.dump({
                'accuracy_metrics': accuracy_metrics,
                'statistical_tests': statistical_tests,
                'metadata': metadata
            }, f, indent=2)

    def get_historical_analysis(self,
                              kpi_name: str,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[AnalysisResult]:
        """Retrieve historical analysis results"""
        kpi_dir = self.output_dir / kpi_name
        if not kpi_dir.exists():
            return []

        results = []
        for timestamp_dir in kpi_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            try:
                # Parse timestamp from directory name
                timestamp = datetime.strptime(timestamp_dir.name, "%Y%m%d_%H%M%S")
                
                # Apply time filters
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue

                # Load analysis results
                with open(timestamp_dir / "analysis_results.json", 'r') as f:
                    analysis_data = json.load(f)

                # Load predictions
                with open(timestamp_dir / "predictions.json", 'r') as f:
                    predictions = json.load(f)

                # Load alerts
                with open(timestamp_dir / "alerts.json", 'r') as f:
                    alerts_data = json.load(f)
                    alerts = [
                        PredictionAlert(**alert_dict)
                        for alert_dict in alerts_data
                    ]

                result = AnalysisResult(
                    kpi_name=kpi_name,
                    timestamp=timestamp,
                    predictions=predictions,
                    alerts=alerts,
                    accuracy_metrics=analysis_data['accuracy_metrics'],
                    statistical_tests=analysis_data['statistical_tests'],
                    metadata=analysis_data['metadata']
                )
                results.append(result)

            except Exception as e:
                self.logger.error(
                    f"Error loading historical analysis from {timestamp_dir}: {str(e)}",
                    exc_info=True
                )

        return sorted(results, key=lambda x: x.timestamp)

    def get_alerts_summary(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         severity: Optional[AlertSeverity] = None,
                         alert_type: Optional[AlertType] = None) -> Dict:
        """Get a summary of alerts across all KPIs"""
        alerts = self.alert_system.get_alert_history(
            start_time=start_time,
            end_time=end_time,
            severity=severity,
            alert_type=alert_type
        )

        summary = {
            'total_alerts': len(alerts),
            'alerts_by_severity': {},
            'alerts_by_type': {},
            'alerts_by_kpi': {},
            'recent_alerts': sorted(
                alerts,
                key=lambda x: x.timestamp,
                reverse=True
            )[:10]
        }

        # Count alerts by severity
        for severity in AlertSeverity:
            severity_alerts = [a for a in alerts if a.severity == severity]
            summary['alerts_by_severity'][severity.value] = len(severity_alerts)

        # Count alerts by type
        for alert_type in AlertType:
            type_alerts = [a for a in alerts if a.alert_type == alert_type]
            summary['alerts_by_type'][alert_type.value] = len(type_alerts)

        # Count alerts by KPI
        for alert in alerts:
            if alert.kpi_name not in summary['alerts_by_kpi']:
                summary['alerts_by_kpi'][alert.kpi_name] = 0
            summary['alerts_by_kpi'][alert.kpi_name] += 1

        return summary

    def update_config(self, config: Dict):
        """Update configuration for all components"""
        if 'sequence_length' in config or 'forecast_horizon' in config:
            self.predictor = PatternPredictor(
                sequence_length=config.get('sequence_length', self.predictor.sequence_length),
                forecast_horizon=config.get('forecast_horizon', self.predictor.forecast_horizon)
            )

        if 'theme' in config:
            self.visualizer = PredictionVisualizer(theme=config['theme'])

        if 'threshold_config' in config:
            self.alert_system.set_threshold_config(config['threshold_config'])

        if 'trend_sensitivity' in config or 'uncertainty_threshold' in config:
            self.alert_system.set_sensitivity(
                trend_sensitivity=config.get('trend_sensitivity'),
                uncertainty_threshold=config.get('uncertainty_threshold')
            )
