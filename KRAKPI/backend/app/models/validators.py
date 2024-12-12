from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, validator, Field
from .kpi_schema import (
    KPIDefinition,
    AssignmentType,
    AssignmentRole,
    DetailedAssignment,
    NotificationPreference
)

class AssignmentValidator(BaseModel):
    @validator('valid_from', 'valid_until')
    def validate_dates(cls, v: Optional[datetime], values: Dict[str, Any]) -> Optional[datetime]:
        if v and 'valid_from' in values and values['valid_from']:
            if v < values['valid_from']:
                raise ValueError("valid_until must be after valid_from")
        return v

    @validator('weight')
    def validate_weight(cls, v: float) -> float:
        if v <= 0 or v > 1:
            raise ValueError("weight must be between 0 and 1")
        return v

    @validator('notifications')
    def validate_notifications(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if v.get('preference') == NotificationPreference.CUSTOM and not v.get('custom_schedule'):
            raise ValueError("custom_schedule is required when preference is CUSTOM")
        return v

class KPIValidator(BaseModel):
    @validator('thresholds')
    def validate_thresholds(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if v['min_value'] >= v['max_value']:
            raise ValueError("min_value must be less than max_value")
        if v['target_value'] < v['min_value'] or v['target_value'] > v['max_value']:
            raise ValueError("target_value must be between min_value and max_value")
        if v['warning_threshold'] <= v['critical_threshold']:
            raise ValueError("warning_threshold must be greater than critical_threshold")
        return v

    @validator('time_config')
    def validate_time_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if v.get('end_date') and v['start_date'] >= v['end_date']:
            raise ValueError("start_date must be before end_date")
        return v

    @validator('calculation_method')
    def validate_calculation_method(cls, v: str) -> str:
        valid_methods = {'SUM', 'AVG', 'COUNT', 'MIN', 'MAX', 'CUSTOM'}
        method = v.split('(')[0].upper()
        if method not in valid_methods:
            raise ValueError(f"Invalid calculation method. Must be one of: {valid_methods}")
        return v

def validate_kpi_definition(kpi: KPIDefinition) -> None:
    """Comprehensive KPI validation"""
    # Validate basic structure
    KPIValidator(**kpi.dict())
    
    # Validate assignments
    for assignment in kpi.assignment.get_all_assignments().values():
        AssignmentValidator(**assignment.dict())
    
    # Validate dependencies
    _validate_dependencies(kpi)

def _validate_dependencies(kpi: KPIDefinition) -> None:
    """Validate KPI dependencies"""
    visited = set()
    path = []

    def check_circular_deps(kpi_id: str) -> None:
        if kpi_id in path:
            cycle = path[path.index(kpi_id):] + [kpi_id]
            raise ValueError(f"Circular dependency detected: {' -> '.join(cycle)}")
        
        if kpi_id in visited:
            return
        
        visited.add(kpi_id)
        path.append(kpi_id)
        
        for dep_id in kpi.dependencies:
            check_circular_deps(dep_id)
        
        path.pop()

    for dep_id in kpi.dependencies:
        check_circular_deps(dep_id)
