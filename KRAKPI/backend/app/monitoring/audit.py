"""Audit logging system for tracking security and business events."""
from typing import Optional, Dict, Any
from datetime import datetime
import json
from enum import Enum
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from ..config import settings
from ..utils.logging import audit_logger

class AuditEventType(str, Enum):
    """Types of audit events."""
    # Security events
    LOGIN = "login"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"
    
    # Data events
    DATA_CREATE = "data_create"
    DATA_READ = "data_read"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    
    # KPI events
    KPI_CREATE = "kpi_create"
    KPI_UPDATE = "kpi_update"
    KPI_DELETE = "kpi_delete"
    KPI_ASSIGNMENT = "kpi_assignment"
    
    # System events
    CONFIG_CHANGE = "config_change"
    SYSTEM_ERROR = "system_error"
    MAINTENANCE = "maintenance"

class AuditEvent(BaseModel):
    """Audit event model."""
    id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    ip_address: Optional[str]
    resource_id: Optional[str]
    resource_type: Optional[str]
    action: str
    status: str
    details: Dict[str, Any]
    metadata: Dict[str, Any]

class AuditLogger:
    """Audit logging system."""
    
    def __init__(self):
        self.db_client: Optional[AsyncIOMotorClient] = None
        self.collection_name = "audit_logs"
        self._setup_database()
    
    def _setup_database(self):
        """Setup database connection."""
        try:
            self.db_client = AsyncIOMotorClient(settings.MONGODB_URL)
            self.db = self.db_client[settings.DATABASE_NAME]
        except Exception as e:
            audit_logger.error(
                f"Failed to setup audit database: {str(e)}",
                extra={"timestamp": datetime.now().isoformat()}
            )
    
    async def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        status: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an audit event."""
        try:
            event = AuditEvent(
                id=f"audit_{datetime.now().timestamp()}",
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                ip_address=ip_address,
                resource_id=resource_id,
                resource_type=resource_type,
                action=action,
                status=status,
                details=details or {},
                metadata=metadata or {}
            )
            
            # Log to database
            await self._store_event(event)
            
            # Log to file
            self._log_to_file(event)
            
            # Check for critical events
            await self._check_critical_event(event)
            
        except Exception as e:
            audit_logger.error(
                f"Failed to log audit event: {str(e)}",
                extra={
                    "event_type": event_type,
                    "action": action,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def _store_event(self, event: AuditEvent):
        """Store event in database."""
        if not self.db_client:
            return
        
        try:
            await self.db[self.collection_name].insert_one(
                event.dict()
            )
        except Exception as e:
            audit_logger.error(
                f"Failed to store audit event: {str(e)}",
                extra={
                    "event_id": event.id,
                    "timestamp": event.timestamp.isoformat()
                }
            )
    
    def _log_to_file(self, event: AuditEvent):
        """Log event to file."""
        audit_logger.info(
            f"Audit event: {event.action}",
            extra={
                "event_id": event.id,
                "event_type": event.event_type,
                "user_id": event.user_id,
                "resource_id": event.resource_id,
                "status": event.status,
                "timestamp": event.timestamp.isoformat(),
                "details": json.dumps(event.details)
            }
        )
    
    async def _check_critical_event(self, event: AuditEvent):
        """Check for critical events that need immediate attention."""
        critical_events = {
            AuditEventType.ACCESS_DENIED,
            AuditEventType.SYSTEM_ERROR,
            AuditEventType.PERMISSION_CHANGE
        }
        
        if event.event_type in critical_events:
            from .alerts import alert_manager, AlertType, AlertSeverity
            
            await alert_manager.add_alert(
                alert_type=AlertType.SECURITY,
                severity=AlertSeverity.WARNING,
                title=f"Critical audit event: {event.event_type}",
                description=f"Action: {event.action}\nStatus: {event.status}",
                metadata={
                    "event_id": event.id,
                    "user_id": event.user_id,
                    "resource_id": event.resource_id,
                    "details": event.details
                }
            )
    
    async def query_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ):
        """Query audit events."""
        if not self.db_client:
            return []
        
        query = {}
        if event_type:
            query["event_type"] = event_type
        if user_id:
            query["user_id"] = user_id
        if resource_id:
            query["resource_id"] = resource_id
        if start_time or end_time:
            query["timestamp"] = {}
            if start_time:
                query["timestamp"]["$gte"] = start_time
            if end_time:
                query["timestamp"]["$lte"] = end_time
        
        try:
            cursor = self.db[self.collection_name].find(
                query
            ).sort(
                "timestamp", -1
            ).limit(limit)
            
            return await cursor.to_list(length=limit)
            
        except Exception as e:
            audit_logger.error(
                f"Failed to query audit events: {str(e)}",
                extra={"timestamp": datetime.now().isoformat()}
            )
            return []
    
    async def cleanup_old_events(self, days: int = 90):
        """Clean up old audit events."""
        if not self.db_client:
            return
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            result = await self.db[self.collection_name].delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            audit_logger.info(
                f"Cleaned up {result.deleted_count} old audit events",
                extra={"timestamp": datetime.now().isoformat()}
            )
            
        except Exception as e:
            audit_logger.error(
                f"Failed to cleanup old audit events: {str(e)}",
                extra={"timestamp": datetime.now().isoformat()}
            )

# Initialize audit logger
audit_logger_instance = AuditLogger()
