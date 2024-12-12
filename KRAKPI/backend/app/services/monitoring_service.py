from typing import List, Dict, Optional
from datetime import datetime, timedelta
from fastapi import WebSocket
import asyncio
import json
from ..models.kpi import KPI, KPIValue
from ..analytics.advanced_models import AdvancedKPIPredictor
from sqlalchemy.orm import Session
import numpy as np

class AlertManager:
    def __init__(self):
        self.connections: Dict[int, List[WebSocket]] = {}
        self.alert_thresholds: Dict[int, Dict] = {}
    
    async def connect(self, websocket: WebSocket, kpi_id: int):
        await websocket.accept()
        if kpi_id not in self.connections:
            self.connections[kpi_id] = []
        self.connections[kpi_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, kpi_id: int):
        if kpi_id in self.connections:
            self.connections[kpi_id].remove(websocket)
    
    async def broadcast_alert(self, kpi_id: int, alert: Dict):
        if kpi_id in self.connections:
            for connection in self.connections[kpi_id]:
                try:
                    await connection.send_json(alert)
                except:
                    await self.disconnect(connection, kpi_id)
    
    def set_alert_threshold(self, kpi_id: int, config: Dict):
        self.alert_thresholds[kpi_id] = config

class MonitoringService:
    def __init__(self, db: Session):
        self.db = db
        self.alert_manager = AlertManager()
        self.predictor = AdvancedKPIPredictor('prophet')
        self.monitoring_tasks = {}
    
    async def start_monitoring(self, kpi_id: int):
        """Start real-time monitoring for a KPI"""
        if kpi_id in self.monitoring_tasks:
            return
        
        task = asyncio.create_task(self._monitor_kpi(kpi_id))
        self.monitoring_tasks[kpi_id] = task
    
    async def stop_monitoring(self, kpi_id: int):
        """Stop monitoring a KPI"""
        if kpi_id in self.monitoring_tasks:
            self.monitoring_tasks[kpi_id].cancel()
            del self.monitoring_tasks[kpi_id]
    
    async def _monitor_kpi(self, kpi_id: int):
        """Monitor KPI values and generate alerts"""
        while True:
            try:
                # Get latest KPI values
                recent_values = self.db.query(KPIValue).filter(
                    KPIValue.kpi_id == kpi_id,
                    KPIValue.timestamp >= datetime.utcnow() - timedelta(hours=24)
                ).order_by(KPIValue.timestamp.desc()).all()
                
                if recent_values:
                    latest_value = recent_values[0].value
                    kpi = self.db.query(KPI).get(kpi_id)
                    
                    # Check for anomalies
                    values = [v.value for v in recent_values]
                    dates = [v.timestamp for v in recent_values]
                    
                    # Train predictor on recent data
                    self.predictor.train(dates, values)
                    
                    # Get prediction for latest timestamp
                    prediction = self.predictor.predict([dates[0]])
                    predicted_value = prediction['predictions'][0]
                    lower_bound = prediction['lower_bound'][0]
                    upper_bound = prediction['upper_bound'][0]
                    
                    # Generate alerts based on different conditions
                    alerts = []
                    
                    # Anomaly detection
                    if latest_value < lower_bound or latest_value > upper_bound:
                        alerts.append({
                            'type': 'anomaly',
                            'severity': 'warning',
                            'message': f'Anomaly detected: Current value ({latest_value:.2f}) is outside expected range ({lower_bound:.2f} - {upper_bound:.2f})',
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    
                    # Threshold violations
                    if kpi.min_threshold and latest_value < kpi.min_threshold:
                        alerts.append({
                            'type': 'threshold',
                            'severity': 'error',
                            'message': f'Value below minimum threshold: {latest_value:.2f} < {kpi.min_threshold}',
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    
                    if kpi.max_threshold and latest_value > kpi.max_threshold:
                        alerts.append({
                            'type': 'threshold',
                            'severity': 'error',
                            'message': f'Value above maximum threshold: {latest_value:.2f} > {kpi.max_threshold}',
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    
                    # Trend detection
                    if len(values) >= 5:
                        recent_trend = np.polyfit(range(5), values[:5], 1)[0]
                        if abs(recent_trend) > 0.1 * np.mean(values[:5]):
                            trend_direction = "increasing" if recent_trend > 0 else "decreasing"
                            alerts.append({
                                'type': 'trend',
                                'severity': 'info',
                                'message': f'Significant {trend_direction} trend detected',
                                'timestamp': datetime.utcnow().isoformat()
                            })
                    
                    # Broadcast alerts
                    for alert in alerts:
                        await self.alert_manager.broadcast_alert(kpi_id, alert)
            
            except Exception as e:
                error_alert = {
                    'type': 'system',
                    'severity': 'error',
                    'message': f'Monitoring error: {str(e)}',
                    'timestamp': datetime.utcnow().isoformat()
                }
                await self.alert_manager.broadcast_alert(kpi_id, error_alert)
            
            # Wait before next check
            await asyncio.sleep(60)  # Check every minute
    
    async def configure_alerts(self, kpi_id: int, config: Dict):
        """Configure alert thresholds and rules"""
        self.alert_manager.set_alert_threshold(kpi_id, config)
    
    async def get_alert_history(self, kpi_id: int, hours: int = 24) -> List[Dict]:
        """Get historical alerts for a KPI"""
        # Implementation depends on how you store alert history
        pass
