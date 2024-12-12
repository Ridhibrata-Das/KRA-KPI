from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from sqlalchemy.orm import Session
from ....db.session import get_db
from ....services.analytics_service import AnalyticsService
from ....core.auth import get_current_user
from fastapi.responses import StreamingResponse
import io

router = APIRouter()

@router.get("/{kpi_id}/forecast")
async def get_kpi_forecast(
    kpi_id: int,
    forecast_days: int = Query(default=30, gt=0, le=365),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Generate forecast for a specific KPI"""
    try:
        analytics_service = AnalyticsService(db)
        forecast = await analytics_service.generate_kpi_forecast(
            kpi_id=kpi_id,
            forecast_days=forecast_days
        )
        return forecast
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/correlations")
async def analyze_correlations(
    kpi_ids: List[int] = Query(...),
    time_window_days: int = Query(default=30, gt=0, le=365),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Analyze correlations between multiple KPIs"""
    try:
        analytics_service = AnalyticsService(db)
        correlations = await analytics_service.analyze_correlations(
            kpi_ids=kpi_ids,
            time_window_days=time_window_days
        )
        return correlations
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{kpi_id}/export")
async def export_analytics_report(
    kpi_id: int,
    format: str = Query(default="pdf", regex="^(pdf|excel)$"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Export analytics report in specified format"""
    try:
        analytics_service = AnalyticsService(db)
        report_data = await analytics_service.export_analytics_report(
            kpi_id=kpi_id,
            format=format
        )
        
        if format == 'pdf':
            media_type = 'application/pdf'
            filename = f'kpi_report_{kpi_id}.pdf'
        else:
            media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename = f'kpi_report_{kpi_id}.xlsx'
        
        return StreamingResponse(
            io.BytesIO(report_data),
            media_type=media_type,
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{kpi_id}/anomalies")
async def detect_anomalies(
    kpi_id: int,
    window_size: int = Query(default=24, gt=0, le=168),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Detect anomalies in KPI data"""
    try:
        analytics_service = AnalyticsService(db)
        anomalies = await analytics_service.detect_real_time_anomalies(
            kpi_id=kpi_id,
            window_size=window_size
        )
        return anomalies
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
