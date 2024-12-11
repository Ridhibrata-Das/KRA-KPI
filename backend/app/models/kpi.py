from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from typing import Optional
from datetime import datetime

from ..database import Base

class TimeperiodEnum(str, enum.Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class KPI(Base):
    __tablename__ = "kpis"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    target_value = Column(Float, nullable=False)
    min_threshold = Column(Float)
    max_threshold = Column(Float)
    time_period = Column(Enum(TimeperiodEnum), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    created_by = relationship("User", back_populates="kpis")
    values = relationship("KPIValue", back_populates="kpi")
    assignments = relationship("KPIAssignment", back_populates="kpi")

class KPIValue(Base):
    __tablename__ = "kpi_values"

    id = Column(Integer, primary_key=True, index=True)
    kpi_id = Column(Integer, ForeignKey("kpis.id"))
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    kpi = relationship("KPI", back_populates="values")

class KPIAssignment(Base):
    __tablename__ = "kpi_assignments"

    id = Column(Integer, primary_key=True, index=True)
    kpi_id = Column(Integer, ForeignKey("kpis.id"))
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    kpi = relationship("KPI", back_populates="assignments")
    team = relationship("Team", back_populates="kpi_assignments")
    project = relationship("Project", back_populates="kpi_assignments")
