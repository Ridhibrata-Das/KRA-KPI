from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import uuid
from .kpi_schema import (
    KPIDefinition,
    KPITemplate,
    KPIType,
    TimeUnit,
    ThresholdConfig,
    TimeConfig,
    KPIAssignment,
    KPIMetadata,
    DetailedAssignment,
    AssignmentType,
    AssignmentRole,
    AssignmentPermission,
    AssignmentNotification,
    NotificationPreference,
    AssignmentHistory
)

class KPIManager:
    def __init__(self, database_connection):
        """Initialize KPI Manager with database connection"""
        self.db = database_connection
        self.template_configs = self._load_template_configs()

    def _load_template_configs(self) -> Dict[KPITemplate, Dict[str, Any]]:
        """Load predefined KPI template configurations"""
        return {
            KPITemplate.SALES_TARGET: {
                "type": KPIType.CURRENCY,
                "thresholds": {
                    "min_value": 0,
                    "max_value": 1000000,
                    "warning_threshold": 0.8,
                    "critical_threshold": 0.6,
                    "target_value": 100000
                },
                "calculation_method": "SUM(sales_amount)",
                "visualization_preferences": {
                    "chart_type": "line",
                    "show_target_line": True,
                    "color_scheme": "blue"
                }
            },
            KPITemplate.CUSTOMER_SATISFACTION: {
                "type": KPIType.PERCENTAGE,
                "thresholds": {
                    "min_value": 0,
                    "max_value": 100,
                    "warning_threshold": 70,
                    "critical_threshold": 60,
                    "target_value": 85
                },
                "calculation_method": "AVG(satisfaction_score)",
                "visualization_preferences": {
                    "chart_type": "gauge",
                    "color_scheme": "green"
                }
            },
            # Add more templates as needed
        }

    async def create_kpi(self, kpi_data: Dict[str, Any], user_id: str) -> KPIDefinition:
        """Create a new KPI definition"""
        # Generate unique ID
        kpi_id = str(uuid.uuid4())
        
        # If using template, merge template configs with provided data
        if "template" in kpi_data:
            template_config = self.template_configs.get(kpi_data["template"], {})
            kpi_data = {**template_config, **kpi_data}
        
        # Create metadata
        metadata = KPIMetadata(
            created_by=user_id,
            created_at=datetime.now()
        )
        
        # Create full KPI definition
        kpi_definition = KPIDefinition(
            id=kpi_id,
            metadata=metadata,
            **kpi_data
        )
        
        # Validate and store in database
        await self._store_kpi(kpi_definition)
        
        return kpi_definition

    async def update_kpi(self, kpi_id: str, updates: Dict[str, Any], user_id: str) -> KPIDefinition:
        """Update an existing KPI definition"""
        # Get existing KPI
        existing_kpi = await self.get_kpi(kpi_id)
        if not existing_kpi:
            raise ValueError(f"KPI with ID {kpi_id} not found")
        
        # Update metadata
        updates["metadata"] = KPIMetadata(
            created_by=existing_kpi.metadata.created_by,
            created_at=existing_kpi.metadata.created_at,
            modified_by=user_id,
            modified_at=datetime.now(),
            version=existing_kpi.metadata.version + 1
        )
        
        # Create updated KPI definition
        updated_kpi = KPIDefinition(
            **{**existing_kpi.dict(), **updates}
        )
        
        # Validate and store
        await self._store_kpi(updated_kpi)
        
        return updated_kpi

    async def get_kpi(self, kpi_id: str) -> Optional[KPIDefinition]:
        """Retrieve a KPI definition by ID"""
        kpi_data = await self.db.kpis.find_one({"id": kpi_id})
        return KPIDefinition(**kpi_data) if kpi_data else None

    async def list_kpis(self, 
                       filters: Optional[Dict[str, Any]] = None,
                       sort_by: str = "name",
                       ascending: bool = True) -> List[KPIDefinition]:
        """List KPIs with optional filtering and sorting"""
        query = filters or {}
        cursor = self.db.kpis.find(query)
        
        # Apply sorting
        cursor = cursor.sort(sort_by, 1 if ascending else -1)
        
        return [KPIDefinition(**kpi_data) for kpi_data in await cursor.to_list(None)]

    async def delete_kpi(self, kpi_id: str) -> bool:
        """Delete a KPI definition"""
        result = await self.db.kpis.delete_one({"id": kpi_id})
        return result.deleted_count > 0

    async def assign_kpi(self, 
                        kpi_id: str,
                        team_ids: Optional[List[str]] = None,
                        project_ids: Optional[List[str]] = None,
                        user_ids: Optional[List[str]] = None) -> KPIDefinition:
        """Assign a KPI to teams, projects, or users"""
        kpi = await self.get_kpi(kpi_id)
        if not kpi:
            raise ValueError(f"KPI with ID {kpi_id} not found")
        
        # Update assignments
        new_assignment = KPIAssignment(
            team_ids=team_ids or kpi.assignment.team_ids,
            project_ids=project_ids or kpi.assignment.project_ids,
            user_ids=user_ids or kpi.assignment.user_ids
        )
        
        # Update KPI with new assignment
        return await self.update_kpi(
            kpi_id,
            {"assignment": new_assignment.dict()},
            "system"  # Using system as the modifier
        )

    async def get_kpi_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available KPI templates with their default configurations"""
        return {
            template.value: config
            for template, config in self.template_configs.items()
        }

    async def _store_kpi(self, kpi: KPIDefinition):
        """Store KPI definition in database"""
        await self.db.kpis.replace_one(
            {"id": kpi.id},
            kpi.dict(),
            upsert=True
        )

    async def validate_kpi_access(self, user_id: str, kpi_id: str) -> bool:
        """Validate user's access to a KPI"""
        kpi = await self.get_kpi(kpi_id)
        if not kpi:
            return False
        
        # Check if user is assigned to the KPI
        if user_id in kpi.assignment.user_ids:
            return True
        
        # TODO: Add more complex access validation (e.g., team membership, role-based access)
        return False

    async def get_kpi_dependencies(self, kpi_id: str) -> List[KPIDefinition]:
        """Get all KPIs that this KPI depends on"""
        kpi = await self.get_kpi(kpi_id)
        if not kpi:
            return []
        
        dependencies = []
        for dep_id in kpi.dependencies:
            dep_kpi = await self.get_kpi(dep_id)
            if dep_kpi:
                dependencies.append(dep_kpi)
        
        return dependencies

    async def validate_kpi_definition(self, kpi: KPIDefinition) -> bool:
        """Validate a KPI definition"""
        # Check for circular dependencies
        visited = set()
        async def check_circular_deps(kpi_id: str) -> bool:
            if kpi_id in visited:
                return False
            visited.add(kpi_id)
            curr_kpi = await self.get_kpi(kpi_id)
            if not curr_kpi:
                return True
            for dep_id in curr_kpi.dependencies:
                if not await check_circular_deps(dep_id):
                    return False
            visited.remove(kpi_id)
            return True
        
        # Validate dependencies
        for dep_id in kpi.dependencies:
            if not await check_circular_deps(dep_id):
                return False
        
        # Validate thresholds
        thresholds = kpi.thresholds
        if not (thresholds.min_value <= thresholds.target_value <= thresholds.max_value):
            return False
        
        # Validate time configuration
        time_config = kpi.time_config
        if time_config.end_date and time_config.start_date > time_config.end_date:
            return False
        
        return True

    async def create_assignment(self,
                              kpi_id: str,
                              assignment_type: AssignmentType,
                              target_id: str,
                              role: AssignmentRole,
                              assigned_by: str,
                              permissions: Optional[AssignmentPermission] = None,
                              notifications: Optional[AssignmentNotification] = None,
                              valid_from: Optional[datetime] = None,
                              valid_until: Optional[datetime] = None,
                              delegation_allowed: bool = False,
                              weight: float = 1.0,
                              tags: List[str] = None,
                              metadata: Dict = None) -> KPIDefinition:
        """Create a new assignment for a KPI"""
        kpi = await self.get_kpi(kpi_id)
        if not kpi:
            raise ValueError(f"KPI with ID {kpi_id} not found")

        # Create detailed assignment
        assignment = DetailedAssignment(
            id=str(uuid.uuid4()),
            type=assignment_type,
            role=role,
            permissions=permissions or AssignmentPermission(),
            notifications=notifications or AssignmentNotification(
                preference=NotificationPreference.ON_THRESHOLD
            ),
            assigned_by=assigned_by,
            assigned_at=datetime.now(),
            valid_from=valid_from,
            valid_until=valid_until,
            delegation_allowed=delegation_allowed,
            weight=weight,
            tags=tags or [],
            metadata=metadata or {}
        )

        # Update KPI assignments based on type
        if assignment_type == AssignmentType.TEAM:
            kpi.assignment.team_assignments[target_id] = assignment
        elif assignment_type == AssignmentType.PROJECT:
            kpi.assignment.project_assignments[target_id] = assignment
        elif assignment_type == AssignmentType.USER:
            kpi.assignment.user_assignments[target_id] = assignment
        elif assignment_type == AssignmentType.DEPARTMENT:
            kpi.assignment.department_assignments[target_id] = assignment
        elif assignment_type == AssignmentType.BUSINESS_UNIT:
            kpi.assignment.business_unit_assignments[target_id] = assignment

        # Record assignment history
        history_entry = AssignmentHistory(
            assignment_id=assignment.id,
            action="assigned",
            timestamp=datetime.now(),
            performed_by=assigned_by,
            new_state=assignment.dict()
        )
        kpi.assignment_history.append(history_entry)

        # Update KPI in database
        await self._store_kpi(kpi)
        return kpi

    async def update_assignment(self,
                              kpi_id: str,
                              assignment_id: str,
                              updates: Dict[str, Any],
                              updated_by: str) -> KPIDefinition:
        """Update an existing assignment"""
        kpi = await self.get_kpi(kpi_id)
        if not kpi:
            raise ValueError(f"KPI with ID {kpi_id} not found")

        # Find the assignment
        all_assignments = kpi.assignment.get_all_assignments()
        if assignment_id not in all_assignments:
            raise ValueError(f"Assignment with ID {assignment_id} not found")

        assignment = all_assignments[assignment_id]
        previous_state = assignment.dict()

        # Update assignment fields
        for key, value in updates.items():
            if hasattr(assignment, key):
                setattr(assignment, key, value)

        # Record update in history
        history_entry = AssignmentHistory(
            assignment_id=assignment_id,
            action="updated",
            timestamp=datetime.now(),
            performed_by=updated_by,
            previous_state=previous_state,
            new_state=assignment.dict()
        )
        kpi.assignment_history.append(history_entry)

        # Update KPI in database
        await self._store_kpi(kpi)
        return kpi

    async def remove_assignment(self,
                              kpi_id: str,
                              assignment_id: str,
                              removed_by: str,
                              reason: Optional[str] = None) -> KPIDefinition:
        """Remove an assignment from a KPI"""
        kpi = await self.get_kpi(kpi_id)
        if not kpi:
            raise ValueError(f"KPI with ID {kpi_id} not found")

        # Find and remove the assignment
        assignment_found = False
        previous_state = None
        
        for assignments in [
            kpi.assignment.team_assignments,
            kpi.assignment.project_assignments,
            kpi.assignment.user_assignments,
            kpi.assignment.department_assignments,
            kpi.assignment.business_unit_assignments
        ]:
            if assignment_id in assignments:
                previous_state = assignments[assignment_id].dict()
                del assignments[assignment_id]
                assignment_found = True
                break

        if not assignment_found:
            raise ValueError(f"Assignment with ID {assignment_id} not found")

        # Record removal in history
        history_entry = AssignmentHistory(
            assignment_id=assignment_id,
            action="removed",
            timestamp=datetime.now(),
            performed_by=removed_by,
            previous_state=previous_state,
            reason=reason
        )
        kpi.assignment_history.append(history_entry)

        # Update KPI in database
        await self._store_kpi(kpi)
        return kpi

    async def delegate_assignment(self,
                                kpi_id: str,
                                assignment_id: str,
                                delegate_to: List[str],
                                delegated_by: str) -> KPIDefinition:
        """Delegate an assignment to other users"""
        kpi = await self.get_kpi(kpi_id)
        if not kpi:
            raise ValueError(f"KPI with ID {kpi_id} not found")

        # Find the assignment
        all_assignments = kpi.assignment.get_all_assignments()
        if assignment_id not in all_assignments:
            raise ValueError(f"Assignment with ID {assignment_id} not found")

        assignment = all_assignments[assignment_id]
        if not assignment.delegation_allowed:
            raise ValueError("Delegation is not allowed for this assignment")

        previous_state = assignment.dict()
        assignment.delegated_to = delegate_to

        # Record delegation in history
        history_entry = AssignmentHistory(
            assignment_id=assignment_id,
            action="delegated",
            timestamp=datetime.now(),
            performed_by=delegated_by,
            previous_state=previous_state,
            new_state=assignment.dict()
        )
        kpi.assignment_history.append(history_entry)

        # Update KPI in database
        await self._store_kpi(kpi)
        return kpi

    async def get_assignment_history(self,
                                   kpi_id: str,
                                   assignment_id: Optional[str] = None) -> List[AssignmentHistory]:
        """Get assignment history for a KPI"""
        kpi = await self.get_kpi(kpi_id)
        if not kpi:
            raise ValueError(f"KPI with ID {kpi_id} not found")

        if assignment_id:
            return [
                entry for entry in kpi.assignment_history
                if entry.assignment_id == assignment_id
            ]
        return kpi.assignment_history

    async def get_user_assignments(self,
                                 user_id: str,
                                 role: Optional[AssignmentRole] = None,
                                 active_only: bool = True) -> List[Tuple[KPIDefinition, DetailedAssignment]]:
        """Get all KPIs assigned to a user"""
        all_kpis = await self.list_kpis()
        user_assignments = []

        for kpi in all_kpis:
            assignments = kpi.assignment.user_assignments
            if user_id in assignments:
                assignment = assignments[user_id]
                if (not role or assignment.role == role) and \
                   (not active_only or self._is_assignment_active(assignment)):
                    user_assignments.append((kpi, assignment))

        return user_assignments

    def _is_assignment_active(self, assignment: DetailedAssignment) -> bool:
        """Check if an assignment is currently active"""
        now = datetime.now()
        return (not assignment.valid_from or assignment.valid_from <= now) and \
               (not assignment.valid_until or assignment.valid_until >= now)
