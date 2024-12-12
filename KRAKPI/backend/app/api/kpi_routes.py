from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from ..models.kpi_schema import (
    KPIDefinition,
    KPITemplate,
    AssignmentType,
    AssignmentRole,
    AssignmentPermission,
    AssignmentNotification,
    DetailedAssignment,
    AssignmentHistory
)
from ..models.kpi_manager import KPIManager
from ..auth.auth_utils import get_current_user

router = APIRouter(prefix="/api/kpis", tags=["KPIs"])

async def get_kpi_manager():
    """Dependency to get KPI manager instance"""
    # TODO: Initialize with proper database connection
    return KPIManager(None)

@router.post("/", response_model=KPIDefinition)
async def create_kpi(
    kpi_data: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Create a new KPI"""
    try:
        return await kpi_manager.create_kpi(kpi_data, current_user)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/templates")
async def get_kpi_templates(
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Get available KPI templates"""
    return await kpi_manager.get_kpi_templates()

@router.get("/", response_model=List[KPIDefinition])
async def list_kpis(
    type: Optional[str] = None,
    status: Optional[str] = None,
    team_id: Optional[str] = None,
    project_id: Optional[str] = None,
    sort_by: str = Query("name", description="Field to sort by"),
    ascending: bool = Query(True, description="Sort in ascending order"),
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """List KPIs with optional filtering"""
    filters = {}
    if type:
        filters["type"] = type
    if status:
        filters["status"] = status
    if team_id:
        filters["assignment.team_ids"] = team_id
    if project_id:
        filters["assignment.project_ids"] = project_id
    
    return await kpi_manager.list_kpis(filters, sort_by, ascending)

@router.get("/{kpi_id}", response_model=KPIDefinition)
async def get_kpi(
    kpi_id: str,
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Get a specific KPI by ID"""
    # Validate access
    if not await kpi_manager.validate_kpi_access(current_user, kpi_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    kpi = await kpi_manager.get_kpi(kpi_id)
    if not kpi:
        raise HTTPException(status_code=404, detail="KPI not found")
    
    return kpi

@router.put("/{kpi_id}", response_model=KPIDefinition)
async def update_kpi(
    kpi_id: str,
    updates: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Update an existing KPI"""
    # Validate access
    if not await kpi_manager.validate_kpi_access(current_user, kpi_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        return await kpi_manager.update_kpi(kpi_id, updates, current_user)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{kpi_id}")
async def delete_kpi(
    kpi_id: str,
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Delete a KPI"""
    # Validate access
    if not await kpi_manager.validate_kpi_access(current_user, kpi_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if await kpi_manager.delete_kpi(kpi_id):
        return {"message": "KPI deleted successfully"}
    raise HTTPException(status_code=404, detail="KPI not found")

@router.post("/{kpi_id}/assign")
async def assign_kpi(
    kpi_id: str,
    team_ids: Optional[List[str]] = None,
    project_ids: Optional[List[str]] = None,
    user_ids: Optional[List[str]] = None,
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Assign a KPI to teams, projects, or users"""
    # Validate access
    if not await kpi_manager.validate_kpi_access(current_user, kpi_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        kpi = await kpi_manager.assign_kpi(kpi_id, team_ids, project_ids, user_ids)
        return {"message": "KPI assigned successfully", "kpi": kpi}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{kpi_id}/dependencies")
async def get_kpi_dependencies(
    kpi_id: str,
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Get KPIs that this KPI depends on"""
    # Validate access
    if not await kpi_manager.validate_kpi_access(current_user, kpi_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    dependencies = await kpi_manager.get_kpi_dependencies(kpi_id)
    return {"dependencies": dependencies}

@router.post("/{kpi_id}/assignments")
async def create_kpi_assignment(
    kpi_id: str,
    assignment_type: AssignmentType,
    target_id: str,
    role: AssignmentRole,
    permissions: Optional[AssignmentPermission] = None,
    notifications: Optional[AssignmentNotification] = None,
    valid_from: Optional[datetime] = None,
    valid_until: Optional[datetime] = None,
    delegation_allowed: bool = False,
    weight: float = Query(1.0, gt=0, le=1),
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Create a new KPI assignment"""
    try:
        return await kpi_manager.create_assignment(
            kpi_id=kpi_id,
            assignment_type=assignment_type,
            target_id=target_id,
            role=role,
            assigned_by=current_user,
            permissions=permissions,
            notifications=notifications,
            valid_from=valid_from,
            valid_until=valid_until,
            delegation_allowed=delegation_allowed,
            weight=weight,
            tags=tags,
            metadata=metadata
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/{kpi_id}/assignments/{assignment_id}")
async def update_kpi_assignment(
    kpi_id: str,
    assignment_id: str,
    updates: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Update an existing KPI assignment"""
    try:
        return await kpi_manager.update_assignment(
            kpi_id=kpi_id,
            assignment_id=assignment_id,
            updates=updates,
            updated_by=current_user
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{kpi_id}/assignments/{assignment_id}")
async def remove_kpi_assignment(
    kpi_id: str,
    assignment_id: str,
    reason: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Remove a KPI assignment"""
    try:
        return await kpi_manager.remove_assignment(
            kpi_id=kpi_id,
            assignment_id=assignment_id,
            removed_by=current_user,
            reason=reason
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{kpi_id}/assignments/{assignment_id}/delegate")
async def delegate_kpi_assignment(
    kpi_id: str,
    assignment_id: str,
    delegate_to: List[str],
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Delegate a KPI assignment to other users"""
    try:
        return await kpi_manager.delegate_assignment(
            kpi_id=kpi_id,
            assignment_id=assignment_id,
            delegate_to=delegate_to,
            delegated_by=current_user
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{kpi_id}/assignments/history")
async def get_kpi_assignment_history(
    kpi_id: str,
    assignment_id: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Get assignment history for a KPI"""
    try:
        return await kpi_manager.get_assignment_history(
            kpi_id=kpi_id,
            assignment_id=assignment_id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/assignments/user/{user_id}")
async def get_user_kpi_assignments(
    user_id: str,
    role: Optional[AssignmentRole] = None,
    active_only: bool = True,
    current_user: str = Depends(get_current_user),
    kpi_manager: KPIManager = Depends(get_kpi_manager)
):
    """Get all KPIs assigned to a user"""
    try:
        return await kpi_manager.get_user_assignments(
            user_id=user_id,
            role=role,
            active_only=active_only
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
