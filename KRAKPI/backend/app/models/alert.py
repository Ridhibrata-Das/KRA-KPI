from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from ..database import Base

class AlertSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(str, enum.Enum):
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    IN_APP = "in_app"

class AlertRule(Base):
    __tablename__ = "alert_rules"

    id = Column(Integer, primary_key=True, index=True)
    kpi_id = Column(Integer, ForeignKey("kpis.id"))
    name = Column(String, nullable=False)
    description = Column(String)
    condition_type = Column(String, nullable=False)  # threshold, trend, anomaly, pattern
    condition_params = Column(JSON, nullable=False)  # Flexible JSON for different condition types
    severity = Column(Enum(AlertSeverity), nullable=False)
    channels = Column(JSON, nullable=False)  # List of notification channels
    is_active = Column(Boolean, default=True)
    cooldown_minutes = Column(Integer, default=60)  # Minimum time between alerts
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    kpi = relationship("KPI", back_populates="alert_rules")
    alerts = relationship("Alert", back_populates="rule")
    created_by = relationship("User", back_populates="alert_rules")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(Integer, ForeignKey("alert_rules.id"))
    kpi_id = Column(Integer, ForeignKey("kpis.id"))
    severity = Column(Enum(AlertSeverity), nullable=False)
    message = Column(String, nullable=False)
    details = Column(JSON)  # Additional context about the alert
    triggered_value = Column(Float)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    rule = relationship("AlertRule", back_populates="alerts")
    kpi = relationship("KPI", back_populates="alerts")
    acknowledged_by = relationship("User", back_populates="acknowledged_alerts")

class AlertNotification(Base):
    __tablename__ = "alert_notifications"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"))
    channel = Column(Enum(AlertChannel), nullable=False)
    recipient = Column(String, nullable=False)  # Email, phone number, webhook URL, etc.
    status = Column(String)  # sent, delivered, failed
    error_message = Column(String)
    sent_at = Column(DateTime(timezone=True), server_default=func.now())
    delivered_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    alert = relationship("Alert", backref="notifications")
