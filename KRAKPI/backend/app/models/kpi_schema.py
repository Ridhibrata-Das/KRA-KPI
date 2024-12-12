from typing import List, Dict, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class TimeUnit(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class KPIType(str, Enum):
    NUMERIC = "numeric"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    RATIO = "ratio"
    BOOLEAN = "boolean"

class KPITemplate(str, Enum):
    SALES_TARGET = "sales_target"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    EMPLOYEE_PERFORMANCE = "employee_performance"
    PROJECT_COMPLETION = "project_completion"
    REVENUE_GROWTH = "revenue_growth"
    COST_REDUCTION = "cost_reduction"
    MARKET_SHARE = "market_share"
    QUALITY_METRIC = "quality_metric"

class ThresholdConfig(BaseModel):
    min_value: float = Field(..., description="Minimum acceptable value")
    max_value: float = Field(..., description="Maximum acceptable value")
    warning_threshold: float = Field(..., description="Warning threshold value")
    critical_threshold: float = Field(..., description="Critical threshold value")
    target_value: float = Field(..., description="Target value to achieve")

class TimeConfig(BaseModel):
    unit: TimeUnit
    frequency: int = Field(1, description="Frequency of measurement")
    start_date: datetime
    end_date: Optional[datetime] = None

class AssignmentRole(str, Enum):
    OWNER = "owner"
    CONTRIBUTOR = "contributor"
    VIEWER = "viewer"
    APPROVER = "approver"
    STAKEHOLDER = "stakeholder"

class AssignmentType(str, Enum):
    TEAM = "team"
    PROJECT = "project"
    USER = "user"
    DEPARTMENT = "department"
    BUSINESS_UNIT = "business_unit"

class NotificationPreference(str, Enum):
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_CHANGE = "on_change"
    ON_THRESHOLD = "on_threshold"
    CUSTOM = "custom"

class AssignmentPermission(BaseModel):
    can_view: bool = True
    can_edit: bool = False
    can_delete: bool = False
    can_assign: bool = False
    can_approve: bool = False
    can_set_targets: bool = False
    can_modify_thresholds: bool = False

class AssignmentNotification(BaseModel):
    preference: NotificationPreference
    threshold_alerts: bool = True
    progress_updates: bool = True
    summary_reports: bool = True
    custom_schedule: Optional[str] = None
    email_enabled: bool = True
    slack_enabled: bool = False
    teams_enabled: bool = False
    custom_webhook: Optional[str] = None

class DetailedAssignment(BaseModel):
    id: str
    type: AssignmentType
    role: AssignmentRole
    permissions: AssignmentPermission
    notifications: AssignmentNotification
    assigned_by: str
    assigned_at: datetime = Field(default_factory=datetime.now)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    delegation_allowed: bool = False
    delegated_to: Optional[List[str]] = None
    weight: float = 1.0  # For weighted responsibility distribution
    tags: List[str] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)

class KPIAssignment(BaseModel):
    team_assignments: Dict[str, DetailedAssignment] = Field(
        default_factory=dict,
        description="Team assignments with detailed configuration"
    )
    project_assignments: Dict[str, DetailedAssignment] = Field(
        default_factory=dict,
        description="Project assignments with detailed configuration"
    )
    user_assignments: Dict[str, DetailedAssignment] = Field(
        default_factory=dict,
        description="User assignments with detailed configuration"
    )
    department_assignments: Dict[str, DetailedAssignment] = Field(
        default_factory=dict,
        description="Department assignments with detailed configuration"
    )
    business_unit_assignments: Dict[str, DetailedAssignment] = Field(
        default_factory=dict,
        description="Business unit assignments with detailed configuration"
    )

    def get_all_assignments(self) -> Dict[str, DetailedAssignment]:
        """Get all assignments combined"""
        return {
            **self.team_assignments,
            **self.project_assignments,
            **self.user_assignments,
            **self.department_assignments,
            **self.business_unit_assignments
        }

    def get_assignments_by_type(self, type: AssignmentType) -> Dict[str, DetailedAssignment]:
        """Get assignments for a specific type"""
        type_map = {
            AssignmentType.TEAM: self.team_assignments,
            AssignmentType.PROJECT: self.project_assignments,
            AssignmentType.USER: self.user_assignments,
            AssignmentType.DEPARTMENT: self.department_assignments,
            AssignmentType.BUSINESS_UNIT: self.business_unit_assignments
        }
        return type_map.get(type, {})

    def get_assignments_by_role(self, role: AssignmentRole) -> Dict[str, DetailedAssignment]:
        """Get all assignments with a specific role"""
        return {
            id: assignment
            for id, assignment in self.get_all_assignments().items()
            if assignment.role == role
        }

    def get_active_assignments(self) -> Dict[str, DetailedAssignment]:
        """Get currently active assignments"""
        now = datetime.now()
        return {
            id: assignment
            for id, assignment in self.get_all_assignments().items()
            if (not assignment.valid_from or assignment.valid_from <= now) and
               (not assignment.valid_until or assignment.valid_until >= now)
        }

class AssignmentHistory(BaseModel):
    assignment_id: str
    action: str  # assigned, updated, removed, delegated
    timestamp: datetime
    performed_by: str
    previous_state: Optional[Dict] = None
    new_state: Dict
    reason: Optional[str] = None

class KPIMetadata(BaseModel):
    created_by: str
    created_at: datetime = Field(default_factory=datetime.now)
    modified_by: Optional[str] = None
    modified_at: Optional[datetime] = None
    version: int = Field(1, description="Version number of the KPI configuration")

class KPIDefinition(BaseModel):
    id: str = Field(..., description="Unique identifier for the KPI")
    name: str = Field(..., description="Name of the KPI")
    description: str = Field(..., description="Detailed description of the KPI")
    type: KPIType
    template: Optional[KPITemplate] = None
    thresholds: ThresholdConfig
    time_config: TimeConfig
    assignment: KPIAssignment
    assignment_history: List[AssignmentHistory] = Field(
        default_factory=list,
        description="History of assignment changes"
    )
    metadata: KPIMetadata
    data_source: Dict = Field(..., description="Configuration for data source")
    calculation_method: str = Field(..., description="Method to calculate the KPI value")
    visualization_preferences: Dict = Field(
        default_factory=dict,
        description="Preferred visualization settings"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    status: str = Field("active", description="Current status of the KPI")
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of other KPIs this one depends on"
    )

    class Config:
        schema_extra = {
            "example": {
                "id": "sales_q1_2024",
                "name": "Q1 2024 Sales Target",
                "description": "Quarterly sales target for Q1 2024",
                "type": "CURRENCY",
                "template": "SALES_TARGET",
                "thresholds": {
                    "min_value": 100000,
                    "max_value": 500000,
                    "warning_threshold": 150000,
                    "critical_threshold": 120000,
                    "target_value": 250000
                },
                "time_config": {
                    "unit": "QUARTERLY",
                    "frequency": 1,
                    "start_date": "2024-01-01T00:00:00",
                    "end_date": "2024-03-31T23:59:59"
                },
                "assignment": {
                    "team_assignments": {
                        "sales_team_1": {
                            "id": "sales_team_1",
                            "type": "TEAM",
                            "role": "OWNER",
                            "permissions": {
                                "can_view": True,
                                "can_edit": True,
                                "can_delete": True,
                                "can_assign": True,
                                "can_approve": True,
                                "can_set_targets": True,
                                "can_modify_thresholds": True
                            },
                            "notifications": {
                                "preference": "DAILY",
                                "threshold_alerts": True,
                                "progress_updates": True,
                                "summary_reports": True
                            },
                            "assigned_by": "admin",
                            "assigned_at": "2023-12-12T00:00:00",
                            "valid_from": "2024-01-01T00:00:00",
                            "valid_until": "2024-03-31T23:59:59",
                            "delegation_allowed": False,
                            "delegated_to": None,
                            "weight": 1.0,
                            "tags": ["sales", "quarterly"],
                            "metadata": {}
                        }
                    },
                    "project_assignments": {},
                    "user_assignments": {},
                    "department_assignments": {},
                    "business_unit_assignments": {}
                },
                "assignment_history": [],
                "metadata": {
                    "created_by": "admin",
                    "created_at": "2023-12-12T00:00:00",
                    "version": 1
                },
                "data_source": {
                    "type": "database",
                    "table": "sales_transactions",
                    "aggregation": "sum"
                },
                "calculation_method": "SUM(sales_amount) WHERE date BETWEEN start_date AND end_date",
                "visualization_preferences": {
                    "chart_type": "line",
                    "color_scheme": "blue",
                    "show_target_line": True
                },
                "tags": ["sales", "quarterly", "revenue"],
                "status": "active",
                "dependencies": []
            }
        }
