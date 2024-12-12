"""Alert system for monitoring and security events."""
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import asyncio
import aiohttp
from pydantic import BaseModel
from ..utils.logging import alert_logger
from ..config import settings

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(str, Enum):
    """Types of alerts."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    ERROR = "error"
    SYSTEM = "system"
    BUSINESS = "business"

class Alert(BaseModel):
    """Alert model."""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

class AlertThreshold(BaseModel):
    """Alert threshold configuration."""
    metric_name: str
    operator: str
    value: float
    duration: int  # seconds
    severity: AlertSeverity

class AlertChannel(BaseModel):
    """Alert notification channel."""
    type: str
    config: Dict[str, str]
    enabled: bool = True

class AlertManager:
    """Alert management system."""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_channels: Dict[str, AlertChannel] = {}
        self.thresholds: Dict[str, AlertThreshold] = {}
        
        # Initialize default thresholds
        self._setup_default_thresholds()
        
        # Start alert processor
        asyncio.create_task(self._process_alerts())
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        self.thresholds = {
            "cpu_usage": AlertThreshold(
                metric_name="system_cpu_usage_percent",
                operator=">=",
                value=80.0,
                duration=300,  # 5 minutes
                severity=AlertSeverity.WARNING
            ),
            "memory_usage": AlertThreshold(
                metric_name="system_memory_usage_bytes",
                operator=">=",
                value=85.0,
                duration=300,
                severity=AlertSeverity.WARNING
            ),
            "error_rate": AlertThreshold(
                metric_name="error_count_total",
                operator=">=",
                value=10.0,
                duration=60,
                severity=AlertSeverity.ERROR
            ),
            "response_time": AlertThreshold(
                metric_name="http_request_duration_seconds",
                operator=">=",
                value=2.0,
                duration=60,
                severity=AlertSeverity.WARNING
            )
        }
    
    async def add_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        description: str,
        metadata: Dict[str, Any]
    ):
        """Add new alert."""
        alert = Alert(
            id=f"alert_{datetime.now().timestamp()}",
            type=alert_type,
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        alert_logger.warning(
            f"New alert: {title}",
            extra={
                "alert_id": alert.id,
                "type": alert_type,
                "severity": severity,
                "metadata": metadata,
                "timestamp": alert.timestamp.isoformat()
            }
        )
        
        # Send notifications
        await self._notify_alert(alert)
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolution_notes: Optional[str] = None
    ):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            alert.resolution_notes = resolution_notes
            
            # Remove from active alerts
            self.active_alerts.pop(alert_id)
            
            # Log resolution
            alert_logger.info(
                f"Alert resolved: {alert.title}",
                extra={
                    "alert_id": alert.id,
                    "resolution_notes": resolution_notes,
                    "timestamp": alert.resolved_at.isoformat()
                }
            )
    
    async def _notify_alert(self, alert: Alert):
        """Send alert notifications."""
        for channel in self.alert_channels.values():
            if not channel.enabled:
                continue
            
            try:
                if channel.type == "slack":
                    await self._send_slack_alert(alert, channel.config)
                elif channel.type == "email":
                    await self._send_email_alert(alert, channel.config)
                elif channel.type == "teams":
                    await self._send_teams_alert(alert, channel.config)
            except Exception as e:
                alert_logger.error(
                    f"Failed to send alert notification: {str(e)}",
                    extra={
                        "alert_id": alert.id,
                        "channel": channel.type,
                        "timestamp": datetime.now().isoformat()
                    }
                )
    
    async def _send_slack_alert(
        self,
        alert: Alert,
        config: Dict[str, str]
    ):
        """Send alert to Slack."""
        webhook_url = config.get("webhook_url")
        if not webhook_url:
            return
        
        payload = {
            "text": f"*{alert.severity.upper()}: {alert.title}*\n{alert.description}",
            "attachments": [{
                "fields": [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in alert.metadata.items()
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=payload)
    
    async def _send_email_alert(
        self,
        alert: Alert,
        config: Dict[str, str]
    ):
        """Send alert via email."""
        # Implement email sending logic
        pass
    
    async def _send_teams_alert(
        self,
        alert: Alert,
        config: Dict[str, str]
    ):
        """Send alert to Microsoft Teams."""
        webhook_url = config.get("webhook_url")
        if not webhook_url:
            return
        
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": alert.title,
            "themeColor": self._get_severity_color(alert.severity),
            "title": f"{alert.severity.upper()}: {alert.title}",
            "sections": [{
                "text": alert.description,
                "facts": [
                    {"name": k, "value": str(v)}
                    for k, v in alert.metadata.items()
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=payload)
    
    def _get_severity_color(self, severity: AlertSeverity) -> str:
        """Get color code for severity level."""
        return {
            AlertSeverity.INFO: "0076D7",
            AlertSeverity.WARNING: "FFA500",
            AlertSeverity.ERROR: "FF0000",
            AlertSeverity.CRITICAL: "8B0000"
        }.get(severity, "000000")
    
    async def _process_alerts(self):
        """Process and manage alerts."""
        while True:
            try:
                # Check thresholds
                for name, threshold in self.thresholds.items():
                    # Implement threshold checking logic
                    pass
                
                # Auto-resolve old alerts
                current_time = datetime.now()
                for alert_id, alert in list(self.active_alerts.items()):
                    if (current_time - alert.timestamp).total_seconds() > settings.ALERT_AUTO_RESOLVE_HOURS * 3600:
                        await self.resolve_alert(
                            alert_id,
                            "Auto-resolved due to age"
                        )
                
            except Exception as e:
                alert_logger.error(
                    f"Alert processing error: {str(e)}",
                    extra={"timestamp": datetime.now().isoformat()}
                )
            
            await asyncio.sleep(60)  # Check every minute

# Initialize alert manager
alert_manager = AlertManager()
