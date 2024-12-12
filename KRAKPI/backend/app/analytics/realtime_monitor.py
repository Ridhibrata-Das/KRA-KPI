from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import threading
import asyncio
import websockets
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import time
from queue import Queue
import schedule

from .unified_analytics import UnifiedAnalytics, AnalysisResult
from .prediction_alert_system import AlertSeverity, AlertType, PredictionAlert

@dataclass
class MonitoringConfig:
    update_interval: int = 300  # seconds
    window_size: int = 1000     # data points
    alert_buffer_size: int = 100
    websocket_port: int = 8765
    enable_websocket: bool = True
    alert_callbacks: Dict[AlertSeverity, List[Callable]] = None
    storage_path: str = './monitoring_data'

class KPIBuffer:
    def __init__(self, window_size: int):
        """Initialize KPI data buffer"""
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.metadata = {}
        self.last_update = None

    def add_point(self,
                 value: float,
                 timestamp: datetime,
                 metadata: Optional[Dict] = None):
        """Add a new data point to the buffer"""
        self.values.append(value)
        self.timestamps.append(timestamp)
        if metadata:
            self.metadata.update(metadata)
        self.last_update = datetime.now()

    def get_data(self) -> Dict:
        """Get all data from the buffer"""
        return {
            'values': list(self.values),
            'timestamps': list(self.timestamps),
            'metadata': self.metadata,
            'last_update': self.last_update
        }

class RealtimeMonitor:
    def __init__(self,
                 analytics: UnifiedAnalytics,
                 config: Optional[MonitoringConfig] = None):
        """Initialize the real-time monitoring system"""
        self.analytics = analytics
        self.config = config or MonitoringConfig()
        
        # Initialize components
        self.kpi_buffers: Dict[str, KPIBuffer] = {}
        self.alert_buffer = deque(maxlen=self.config.alert_buffer_size)
        self.alert_callbacks = self.config.alert_callbacks or {}
        
        # Setup storage
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize threading components
        self.analysis_queue = Queue()
        self.is_running = False
        self.analysis_thread = None
        self.websocket_server = None
        self.connected_clients = set()
        
        # Schedule periodic tasks
        self._schedule_tasks()

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.storage_path / 'monitoring.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def start(self):
        """Start the monitoring system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting real-time monitoring system")
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analysis_worker)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        # Start WebSocket server if enabled
        if self.config.enable_websocket:
            asyncio.get_event_loop().run_until_complete(self._start_websocket_server())
        
        # Start scheduled tasks
        self._start_scheduler()

    def stop(self):
        """Stop the monitoring system"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping real-time monitoring system")
        self.is_running = False
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
        
        # Stop analysis thread
        self.analysis_queue.put(None)
        if self.analysis_thread:
            self.analysis_thread.join()
        
        # Save state
        self._save_state()

    def add_data_point(self,
                      kpi_name: str,
                      value: float,
                      timestamp: Optional[datetime] = None,
                      metadata: Optional[Dict] = None):
        """Add a new data point for a KPI"""
        timestamp = timestamp or datetime.now()
        
        # Create buffer if not exists
        if kpi_name not in self.kpi_buffers:
            self.kpi_buffers[kpi_name] = KPIBuffer(self.config.window_size)
        
        # Add data point
        self.kpi_buffers[kpi_name].add_point(value, timestamp, metadata)
        
        # Queue analysis if enough data
        buffer = self.kpi_buffers[kpi_name]
        if len(buffer.values) >= self.config.window_size:
            self.analysis_queue.put((kpi_name, buffer.get_data()))

    def get_latest_alerts(self,
                         severity: Optional[AlertSeverity] = None,
                         limit: Optional[int] = None) -> List[PredictionAlert]:
        """Get latest alerts with optional filtering"""
        alerts = list(self.alert_buffer)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if limit:
            alerts = alerts[-limit:]
        
        return alerts

    def get_kpi_status(self, kpi_name: str) -> Optional[Dict]:
        """Get current status of a KPI"""
        if kpi_name not in self.kpi_buffers:
            return None
        
        buffer = self.kpi_buffers[kpi_name]
        data = buffer.get_data()
        
        return {
            'last_value': data['values'][-1] if data['values'] else None,
            'last_update': data['last_update'],
            'data_points': len(data['values']),
            'metadata': data['metadata']
        }

    def add_alert_callback(self,
                          callback: Callable[[PredictionAlert], Any],
                          severity: Optional[AlertSeverity] = None):
        """Add callback for alerts"""
        if severity:
            if severity not in self.alert_callbacks:
                self.alert_callbacks[severity] = []
            self.alert_callbacks[severity].append(callback)
        else:
            for sev in AlertSeverity:
                if sev not in self.alert_callbacks:
                    self.alert_callbacks[sev] = []
                self.alert_callbacks[sev].append(callback)

    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        async def handler(websocket, path):
            self.connected_clients.add(websocket)
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    message = await websocket.recv()
                    # Process any client messages if needed
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.connected_clients.remove(websocket)

        self.websocket_server = await websockets.serve(
            handler,
            "localhost",
            self.config.websocket_port
        )

    async def _broadcast_update(self, data: Dict):
        """Broadcast update to all connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps(data)
        await asyncio.gather(
            *[client.send(message) for client in self.connected_clients]
        )

    def _analysis_worker(self):
        """Worker thread for analyzing KPI data"""
        while self.is_running:
            try:
                # Get next analysis task
                task = self.analysis_queue.get()
                if task is None:
                    break
                
                kpi_name, data = task
                
                # Perform analysis
                result = self.analytics.analyze_kpi(
                    kpi_name=kpi_name,
                    values=data['values'],
                    timestamps=data['timestamps'],
                    metadata=data['metadata']
                )
                
                # Process alerts
                self._process_alerts(result.alerts)
                
                # Broadcast update
                if self.config.enable_websocket:
                    asyncio.run(self._broadcast_update({
                        'type': 'analysis_update',
                        'kpi_name': kpi_name,
                        'timestamp': datetime.now().isoformat(),
                        'result': result.to_dict()
                    }))
                
                # Save results
                self._save_analysis_result(result)
                
            except Exception as e:
                self.logger.error(f"Error in analysis worker: {str(e)}", exc_info=True)

    def _process_alerts(self, alerts: List[PredictionAlert]):
        """Process new alerts and trigger callbacks"""
        for alert in alerts:
            # Add to buffer
            self.alert_buffer.append(alert)
            
            # Trigger callbacks
            callbacks = self.alert_callbacks.get(alert.severity, [])
            for callback in callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(
                        f"Error in alert callback: {str(e)}",
                        exc_info=True
                    )
            
            # Broadcast alert
            if self.config.enable_websocket:
                asyncio.run(self._broadcast_update({
                    'type': 'new_alert',
                    'alert': alert.to_dict()
                }))

    def _save_analysis_result(self, result: AnalysisResult):
        """Save analysis result to storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = (
            self.storage_path /
            result.kpi_name /
            f"analysis_{timestamp}.json"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def _save_state(self):
        """Save current state to storage"""
        state = {
            'kpi_buffers': {
                name: buffer.get_data()
                for name, buffer in self.kpi_buffers.items()
            },
            'alerts': [alert.to_dict() for alert in self.alert_buffer],
            'timestamp': datetime.now().isoformat()
        }
        
        state_file = self.storage_path / 'monitor_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load state from storage"""
        state_file = self.storage_path / 'monitor_state.json'
        if not state_file.exists():
            return
        
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Restore KPI buffers
        for kpi_name, data in state['kpi_buffers'].items():
            buffer = KPIBuffer(self.config.window_size)
            for value, timestamp in zip(data['values'], data['timestamps']):
                buffer.add_point(
                    value,
                    datetime.fromisoformat(timestamp),
                    data['metadata']
                )
            self.kpi_buffers[kpi_name] = buffer
        
        # Restore alerts
        for alert_data in state['alerts']:
            self.alert_buffer.append(PredictionAlert(**alert_data))

    def _schedule_tasks(self):
        """Schedule periodic tasks"""
        # Save state every hour
        schedule.every().hour.do(self._save_state)
        
        # Cleanup old data daily
        schedule.every().day.at("00:00").do(self._cleanup_old_data)

    def _start_scheduler(self):
        """Start the task scheduler"""
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()

    def _cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old analysis results"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for kpi_dir in self.storage_path.iterdir():
            if not kpi_dir.is_dir():
                continue
            
            for file in kpi_dir.glob('analysis_*.json'):
                try:
                    # Parse timestamp from filename
                    timestamp = datetime.strptime(
                        file.stem.split('_')[1],
                        "%Y%m%d%H%M%S"
                    )
                    
                    if timestamp < cutoff_date:
                        file.unlink()
                
                except Exception as e:
                    self.logger.error(
                        f"Error cleaning up file {file}: {str(e)}",
                        exc_info=True
                    )
