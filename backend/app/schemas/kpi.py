from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class TimeperiodEnum(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class KPIBase(BaseModel):
    name: str = Field(..., description="Name of the KPI")
    description: Optional[str] = Field(None, description="Description of the KPI")
    target_value: float = Field(..., description="Target value for the KPI")
    min_threshold: Optional[float] = Field(None, description="Minimum threshold value")
    max_threshold: Optional[float] = Field(None, description="Maximum threshold value")
    time_period: TimeperiodEnum = Field(..., description="Time period for KPI measurement")

class KPICreate(KPIBase):
    pass

class KPIUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    target_value: Optional[float] = None
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    time_period: Optional[TimeperiodEnum] = None

class KPIInDB(KPIBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]
    created_by_id: int

    class Config:
        from_attributes = True

class KPIValueBase(BaseModel):
    value: float
    timestamp: datetime

class KPIValueCreate(KPIValueBase):
    kpi_id: int

class KPIValueInDB(KPIValueBase):
    id: int
    kpi_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class KPIWithValues(KPIInDB):
    values: List[KPIValueInDB] = []
