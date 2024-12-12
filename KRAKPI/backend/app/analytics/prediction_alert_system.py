from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    THRESHOLD_BREACH = "threshold_breach"
    TREND_CHANGE = "trend_change"
    PATTERN_DEVIATION = "pattern_deviation"
    ANOMALY = "anomaly"
    UNCERTAINTY_HIGH = "uncertainty_high"

@dataclass
class PredictionAlert:
    alert_id: str
    timestamp: datetime
    kpi_name: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    predicted_value: float
    threshold_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert alert to dictionary format"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'kpi_name': self.kpi_name,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'predicted_value': float(self.predicted_value),
            'threshold_value': float(self.threshold_value) if self.threshold_value else None,
            'confidence_interval': [float(x) for x in self.confidence_interval] if self.confidence_interval else None,
            'metadata': self.metadata
        }

class PredictionAlertSystem:
    def __init__(self,
                 threshold_config: Optional[Dict] = None,
                 trend_sensitivity: float = 0.1,
                 uncertainty_threshold: float = 0.2):
        """Initialize the alert system with configuration"""
        self.threshold_config = threshold_config or {}
        self.trend_sensitivity = trend_sensitivity
        self.uncertainty_threshold = uncertainty_threshold
        self.alert_history: List[PredictionAlert] = []

    def analyze_predictions(self,
                          kpi_name: str,
                          predictions: Dict,
                          historical_values: List[float],
                          timestamps: List[datetime]) -> List[PredictionAlert]:
        """Analyze predictions and generate alerts based on various criteria"""
        alerts = []
        
        # Get ensemble predictions
        ensemble_pred = predictions['ensemble']
        pred_values = ensemble_pred['values']
        uncertainty = ensemble_pred['uncertainty']
        
        # Check each type of potential alert
        alerts.extend(self._check_threshold_breaches(
            kpi_name, pred_values, timestamps, uncertainty
        ))
        alerts.extend(self._check_trend_changes(
            kpi_name, pred_values, historical_values, timestamps
        ))
        alerts.extend(self._check_pattern_deviations(
            kpi_name, predictions, timestamps
        ))
        alerts.extend(self._check_uncertainty_levels(
            kpi_name, uncertainty, timestamps
        ))
        
        # Store alerts in history
        self.alert_history.extend(alerts)
        
        return alerts

    def _check_threshold_breaches(self,
                                kpi_name: str,
                                predictions: List[float],
                                timestamps: List[datetime],
                                uncertainty: Dict) -> List[PredictionAlert]:
        """Check for threshold breaches in predictions"""
        alerts = []
        
        if kpi_name not in self.threshold_config:
            return alerts
            
        config = self.threshold_config[kpi_name]
        
        for i, (pred, timestamp) in enumerate(zip(predictions, timestamps)):
            # Check critical thresholds
            if 'critical' in config:
                if pred > config['critical']['upper']:
                    alerts.append(PredictionAlert(
                        alert_id=f"{kpi_name}_threshold_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        kpi_name=kpi_name,
                        alert_type=AlertType.THRESHOLD_BREACH,
                        severity=AlertSeverity.CRITICAL,
                        message=f"Predicted value ({pred:.2f}) exceeds critical threshold ({config['critical']['upper']:.2f})",
                        predicted_value=pred,
                        threshold_value=config['critical']['upper'],
                        confidence_interval=(uncertainty['lower'][i], uncertainty['upper'][i])
                    ))
                elif pred < config['critical']['lower']:
                    alerts.append(PredictionAlert(
                        alert_id=f"{kpi_name}_threshold_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        kpi_name=kpi_name,
                        alert_type=AlertType.THRESHOLD_BREACH,
                        severity=AlertSeverity.CRITICAL,
                        message=f"Predicted value ({pred:.2f}) below critical threshold ({config['critical']['lower']:.2f})",
                        predicted_value=pred,
                        threshold_value=config['critical']['lower'],
                        confidence_interval=(uncertainty['lower'][i], uncertainty['upper'][i])
                    ))
            
            # Check warning thresholds
            if 'warning' in config:
                if pred > config['warning']['upper']:
                    alerts.append(PredictionAlert(
                        alert_id=f"{kpi_name}_threshold_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        kpi_name=kpi_name,
                        alert_type=AlertType.THRESHOLD_BREACH,
                        severity=AlertSeverity.HIGH,
                        message=f"Predicted value ({pred:.2f}) exceeds warning threshold ({config['warning']['upper']:.2f})",
                        predicted_value=pred,
                        threshold_value=config['warning']['upper'],
                        confidence_interval=(uncertainty['lower'][i], uncertainty['upper'][i])
                    ))
                elif pred < config['warning']['lower']:
                    alerts.append(PredictionAlert(
                        alert_id=f"{kpi_name}_threshold_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        kpi_name=kpi_name,
                        alert_type=AlertType.THRESHOLD_BREACH,
                        severity=AlertSeverity.HIGH,
                        message=f"Predicted value ({pred:.2f}) below warning threshold ({config['warning']['lower']:.2f})",
                        predicted_value=pred,
                        threshold_value=config['warning']['lower'],
                        confidence_interval=(uncertainty['lower'][i], uncertainty['upper'][i])
                    ))
        
        return alerts

    def _check_trend_changes(self,
                           kpi_name: str,
                           predictions: List[float],
                           historical_values: List[float],
                           timestamps: List[datetime]) -> List[PredictionAlert]:
        """Detect significant trend changes in predictions"""
        alerts = []
        
        # Calculate historical trend
        historical_trend = np.polyfit(
            range(len(historical_values[-10:])),
            historical_values[-10:],
            1
        )[0]
        
        # Calculate prediction trend
        pred_trend = np.polyfit(
            range(len(predictions)),
            predictions,
            1
        )[0]
        
        # Check for significant trend changes
        trend_change = abs(pred_trend - historical_trend)
        if trend_change > self.trend_sensitivity:
            severity = (AlertSeverity.HIGH if trend_change > 2 * self.trend_sensitivity
                       else AlertSeverity.MEDIUM)
            
            alerts.append(PredictionAlert(
                alert_id=f"{kpi_name}_trend_{timestamps[0].isoformat()}",
                timestamp=timestamps[0],
                kpi_name=kpi_name,
                alert_type=AlertType.TREND_CHANGE,
                severity=severity,
                message=f"Significant trend change detected (change: {trend_change:.2f})",
                predicted_value=predictions[0],
                metadata={
                    'historical_trend': float(historical_trend),
                    'prediction_trend': float(pred_trend),
                    'trend_change': float(trend_change)
                }
            ))
        
        return alerts

    def _check_pattern_deviations(self,
                                kpi_name: str,
                                predictions: Dict,
                                timestamps: List[datetime]) -> List[PredictionAlert]:
        """Detect significant deviations between different model predictions"""
        alerts = []
        
        # Get all model predictions except ensemble
        model_predictions = {
            k: v['values'] for k, v in predictions.items()
            if k != 'ensemble'
        }
        
        # Calculate mean and std of predictions
        pred_array = np.array(list(model_predictions.values()))
        mean_pred = np.mean(pred_array, axis=0)
        std_pred = np.std(pred_array, axis=0)
        
        # Check for high deviation between models
        for i, (timestamp, std) in enumerate(zip(timestamps, std_pred)):
            cv = std / abs(mean_pred[i]) if mean_pred[i] != 0 else float('inf')
            
            if cv > self.uncertainty_threshold:
                alerts.append(PredictionAlert(
                    alert_id=f"{kpi_name}_deviation_{timestamp.isoformat()}",
                    timestamp=timestamp,
                    kpi_name=kpi_name,
                    alert_type=AlertType.PATTERN_DEVIATION,
                    severity=AlertSeverity.MEDIUM,
                    message=f"High deviation between model predictions (CV: {cv:.2f})",
                    predicted_value=mean_pred[i],
                    metadata={
                        'coefficient_of_variation': float(cv),
                        'standard_deviation': float(std),
                        'model_predictions': {
                            k: float(v[i]) for k, v in model_predictions.items()
                        }
                    }
                ))
        
        return alerts

    def _check_uncertainty_levels(self,
                                kpi_name: str,
                                uncertainty: Dict,
                                timestamps: List[datetime]) -> List[PredictionAlert]:
        """Check for high uncertainty in predictions"""
        alerts = []
        
        lower = np.array(uncertainty['lower'])
        upper = np.array(uncertainty['upper'])
        
        # Calculate relative uncertainty
        mean_pred = (upper + lower) / 2
        uncertainty_range = upper - lower
        relative_uncertainty = uncertainty_range / np.abs(mean_pred)
        
        for i, (timestamp, rel_uncert) in enumerate(zip(timestamps, relative_uncertainty)):
            if rel_uncert > self.uncertainty_threshold:
                alerts.append(PredictionAlert(
                    alert_id=f"{kpi_name}_uncertainty_{timestamp.isoformat()}",
                    timestamp=timestamp,
                    kpi_name=kpi_name,
                    alert_type=AlertType.UNCERTAINTY_HIGH,
                    severity=AlertSeverity.MEDIUM,
                    message=f"High prediction uncertainty detected ({rel_uncert:.2f})",
                    predicted_value=mean_pred[i],
                    confidence_interval=(float(lower[i]), float(upper[i])),
                    metadata={
                        'relative_uncertainty': float(rel_uncert),
                        'uncertainty_range': float(uncertainty_range[i])
                    }
                ))
        
        return alerts

    def get_alert_history(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         severity: Optional[AlertSeverity] = None,
                         alert_type: Optional[AlertType] = None) -> List[PredictionAlert]:
        """Retrieve filtered alert history"""
        filtered_alerts = self.alert_history
        
        if start_time:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.timestamp >= start_time
            ]
        
        if end_time:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.timestamp <= end_time
            ]
        
        if severity:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.severity == severity
            ]
        
        if alert_type:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.alert_type == alert_type
            ]
        
        return filtered_alerts

    def export_alerts(self, filename: str):
        """Export alerts to a JSON file"""
        alerts_dict = [alert.to_dict() for alert in self.alert_history]
        with open(filename, 'w') as f:
            json.dump(alerts_dict, f, indent=2)

    def set_threshold_config(self, config: Dict):
        """Update threshold configuration"""
        self.threshold_config = config

    def set_sensitivity(self,
                       trend_sensitivity: Optional[float] = None,
                       uncertainty_threshold: Optional[float] = None):
        """Update sensitivity parameters"""
        if trend_sensitivity is not None:
            self.trend_sensitivity = trend_sensitivity
        if uncertainty_threshold is not None:
            self.uncertainty_threshold = uncertainty_threshold
