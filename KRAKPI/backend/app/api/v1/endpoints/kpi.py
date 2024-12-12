from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ....db.session import get_db
from ....crud import crud_kpi
from ....schemas.kpi import (
    KPICreate,
    KPIUpdate,
    KPIInDB,
    KPIWithValues,
    KPIValueCreate
)
from ....core.auth import get_current_user
from ....models.user import User

router = APIRouter()

@router.post("/", response_model=KPIInDB)
async def create_kpi(
    *,
    db: Session = Depends(get_db),
    kpi_in: KPICreate,
    current_user: User = Depends(get_current_user)
):
    """Create new KPI"""
    kpi = crud_kpi.create_with_owner(
        db=db, obj_in=kpi_in, owner_id=current_user.id
    )
    return kpi

@router.get("/", response_model=List[KPIInDB])
async def list_kpis(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Retrieve KPIs"""
    kpis = crud_kpi.get_multi_by_owner(
        db=db, owner_id=current_user.id, skip=skip, limit=limit
    )
    return kpis

@router.get("/{kpi_id}", response_model=KPIWithValues)
async def get_kpi(
    *,
    db: Session = Depends(get_db),
    kpi_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get KPI by ID"""
    kpi = crud_kpi.get(db=db, id=kpi_id)
    if not kpi:
        raise HTTPException(status_code=404, detail="KPI not found")
    if kpi.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return kpi

@router.put("/{kpi_id}", response_model=KPIInDB)
async def update_kpi(
    *,
    db: Session = Depends(get_db),
    kpi_id: int,
    kpi_in: KPIUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update KPI"""
    kpi = crud_kpi.get(db=db, id=kpi_id)
    if not kpi:
        raise HTTPException(status_code=404, detail="KPI not found")
    if kpi.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    kpi = crud_kpi.update(db=db, db_obj=kpi, obj_in=kpi_in)
    return kpi

@router.delete("/{kpi_id}")
async def delete_kpi(
    *,
    db: Session = Depends(get_db),
    kpi_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete KPI"""
    kpi = crud_kpi.get(db=db, id=kpi_id)
    if not kpi:
        raise HTTPException(status_code=404, detail="KPI not found")
    if kpi.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    crud_kpi.remove(db=db, id=kpi_id)
    return {"success": True}

@router.post("/{kpi_id}/values", response_model=KPIWithValues)
async def add_kpi_value(
    *,
    db: Session = Depends(get_db),
    kpi_id: int,
    value_in: KPIValueCreate,
    current_user: User = Depends(get_current_user)
):
    """Add KPI value"""
    kpi = crud_kpi.get(db=db, id=kpi_id)
    if not kpi:
        raise HTTPException(status_code=404, detail="KPI not found")
    if kpi.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    kpi = crud_kpi.add_value(db=db, kpi=kpi, value_in=value_in)
    return kpi
