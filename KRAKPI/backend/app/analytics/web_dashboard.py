from typing import List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import uvicorn

from .unified_analytics import UnifiedAnalytics, AnalysisResult
from .realtime_monitor import RealtimeMonitor, MonitoringConfig
from .batch_analyzer import BatchAnalyzer, BatchConfig
from .prediction_alert_system import AlertSeverity, AlertType
from .advanced_visualizations import AdvancedVisualizer

class AnalyticsDashboard:
    def __init__(self,
                 analytics: UnifiedAnalytics,
                 monitor: RealtimeMonitor,
                 batch_analyzer: BatchAnalyzer,
                 static_dir: str = './static',
                 template_dir: str = './templates'):
        """Initialize the web dashboard"""
        self.analytics = analytics
        self.monitor = monitor
        self.batch_analyzer = batch_analyzer
        self.visualizer = AdvancedVisualizer()
        
        # Setup FastAPI
        self.app = FastAPI(title="KPI Analytics Dashboard")
        self.setup_routes()
        
        # Setup static files and templates
        self.static_dir = Path(static_dir)
        self.template_dir = Path(template_dir)
        self.setup_static_files()
        self.templates = Jinja2Templates(directory=str(self.template_dir))
        
        # Setup WebSocket management
        self.active_connections: Dict[str, List[WebSocket]] = defaultdict(list)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def setup_static_files(self):
        """Setup static files and create default directories"""
        # Create directories if they don't exist
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default CSS
        self._create_default_css()
        
        # Create default templates
        self._create_default_templates()
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")

    def _create_default_css(self):
        """Create default CSS file"""
        css_content = """
        /* Modern Dashboard Theme */
        :root {
            --primary-color: #2C3E50;
            --secondary-color: #3498DB;
            --accent-color: #E74C3C;
            --background-color: #ECF0F1;
            --text-color: #2C3E50;
            --card-background: #FFFFFF;
            --border-radius: 8px;
            --spacing-unit: 20px;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--background-color);
            color: var(--text-color);
        }

        .dashboard-container {
            display: grid;
            grid-template-columns: 250px 1fr;
            min-height: 100vh;
        }

        .sidebar {
            background-color: var(--primary-color);
            color: white;
            padding: var(--spacing-unit);
            position: fixed;
            height: 100vh;
            width: 250px;
            overflow-y: auto;
        }

        .sidebar h2 {
            margin: 0 0 var(--spacing-unit) 0;
            padding-bottom: var(--spacing-unit);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .nav-link {
            color: white;
            text-decoration: none;
            padding: 10px;
            display: block;
            border-radius: var(--border-radius);
            transition: background-color 0.3s;
        }

        .nav-link:hover,
        .nav-link.active {
            background-color: rgba(255, 255, 255, 0.1);
        }

        .kpi-selector {
            margin-top: var(--spacing-unit);
            padding-top: var(--spacing-unit);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .kpi-item {
            margin: 5px 0;
            padding: 5px;
            display: flex;
            align-items: center;
        }

        .kpi-item input[type="checkbox"] {
            margin-right: 10px;
        }

        .main-content {
            margin-left: 250px;
            padding: var(--spacing-unit);
        }

        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--spacing-unit);
        }

        .time-range-selector,
        .refresh-control {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        select,
        input[type="datetime-local"],
        button {
            padding: 8px 12px;
            border: 1px solid #BDC3C7;
            border-radius: var(--border-radius);
            background-color: white;
            font-size: 14px;
        }

        button {
            background-color: var(--secondary-color);
            color: white;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #2980B9;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: var(--spacing-unit);
            margin-bottom: var(--spacing-unit);
        }

        .metric-card {
            background: var(--card-background);
            border-radius: var(--border-radius);
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--secondary-color);
        }

        .metric-label {
            font-size: 14px;
            color: #7F8C8D;
            margin-top: 5px;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: var(--spacing-unit);
        }

        .dashboard-item {
            min-height: 400px;
        }

        .dashboard-item.wide {
            grid-column: span 2;
        }

        .card {
            background: var(--card-background);
            border-radius: var(--border-radius);
            padding: var(--spacing-unit);
            height: 100%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .card h3 {
            margin: 0 0 15px 0;
            color: var(--primary-color);
        }

        .chart-container {
            width: 100%;
            height: calc(100% - 40px);
            min-height: 300px;
        }

        .alerts-panel {
            max-height: 300px;
            overflow-y: auto;
        }

        .alert {
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                transform: translateX(-100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        .alert-critical {
            background-color: #FADBD8;
            border-left: 4px solid #E74C3C;
        }

        .alert-high {
            background-color: #FCF3CF;
            border-left: 4px solid #F1C40F;
        }

        .alert-medium {
            background-color: #D5F5E3;
            border-left: 4px solid #2ECC71;
        }

        .alert-low {
            background-color: #EBF5FB;
            border-left: 4px solid #3498DB;
        }

        @media (max-width: 1200px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .dashboard-item.wide {
                grid-column: auto;
            }
        }

        @media (max-width: 768px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }
            
            .sidebar {
                position: static;
                width: 100%;
                height: auto;
            }
            
            .main-content {
                margin-left: 0;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
        """
        
        css_file = self.static_dir / 'css' / 'dashboard.css'
        css_file.parent.mkdir(parents=True, exist_ok=True)
        css_file.write_text(css_content)

    def _create_default_templates(self):
        """Create default HTML templates"""
        # Base template
        base_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{% block title %}KPI Analytics Dashboard{% endblock %}</title>
            <link rel="stylesheet" href="{{ url_for('static', path='css/dashboard.css') }}">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            {% block extra_head %}{% endblock %}
        </head>
        <body>
            <div class="dashboard-container">
                <nav class="sidebar">
                    <h2>Analytics Dashboard</h2>
                    <ul>
                        <li><a href="/" class="nav-link active">Overview</a></li>
                        <li><a href="/kpis" class="nav-link">KPI Analysis</a></li>
                        <li><a href="/alerts" class="nav-link">Alerts</a></li>
                        <li><a href="/batch" class="nav-link">Batch Analysis</a></li>
                        <li><a href="/reports" class="nav-link">Reports</a></li>
                    </ul>
                    <div class="kpi-selector">
                        <h3>Select KPIs</h3>
                        <div id="kpi-list">
                            <!-- Dynamically populated -->
                        </div>
                    </div>
                </nav>
                <main class="main-content">
                    {% block content %}{% endblock %}
                </main>
            </div>
            {% block scripts %}{% endblock %}
        </body>
        </html>
        """
        
        # Dashboard template
        dashboard_template = """
        {% extends "base.html" %}

        {% block content %}
        <div class="dashboard-header">
            <div class="time-range-selector">
                <select id="timeRange">
                    <option value="1h">Last Hour</option>
                    <option value="24h" selected>Last 24 Hours</option>
                    <option value="7d">Last 7 Days</option>
                    <option value="30d">Last 30 Days</option>
                    <option value="custom">Custom Range</option>
                </select>
                <div id="customRange" style="display: none;">
                    <input type="datetime-local" id="startTime">
                    <input type="datetime-local" id="endTime">
                </div>
            </div>
            <div class="refresh-control">
                <select id="refreshInterval">
                    <option value="0">Manual Refresh</option>
                    <option value="5000">5 seconds</option>
                    <option value="30000" selected>30 seconds</option>
                    <option value="60000">1 minute</option>
                </select>
                <button id="refreshButton">Refresh Now</button>
            </div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="totalKpis">{{ total_kpis }}</div>
                <div class="metric-label">Active KPIs</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="totalAlerts">{{ total_alerts }}</div>
                <div class="metric-label">Active Alerts</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="predictionAccuracy">{{ prediction_accuracy }}%</div>
                <div class="metric-label">Prediction Accuracy</div>
            </div>
        </div>

        <div class="dashboard-grid">
            <!-- Real-time KPI Monitoring -->
            <div class="dashboard-item wide">
                <div class="card">
                    <h3>Real-time KPI Monitoring</h3>
                    <div id="realtimeChart" class="chart-container"></div>
                </div>
            </div>

            <!-- Alerts Panel -->
            <div class="dashboard-item">
                <div class="card">
                    <h3>Recent Alerts</h3>
                    <div id="alertsContainer" class="alerts-panel">
                        {% for alert in recent_alerts %}
                        <div class="alert alert-{{ alert.severity }}">
                            <strong>{{ alert.kpi_name }}</strong>: {{ alert.message }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            <!-- Pattern Heatmap -->
            <div class="dashboard-item">
                <div class="card">
                    <h3>Pattern Analysis</h3>
                    <div id="patternHeatmap" class="chart-container"></div>
                </div>
            </div>

            <!-- Distribution Analysis -->
            <div class="dashboard-item">
                <div class="card">
                    <h3>Value Distribution</h3>
                    <div id="distributionPlot" class="chart-container"></div>
                </div>
            </div>

            <!-- Correlation Network -->
            <div class="dashboard-item wide">
                <div class="card">
                    <h3>KPI Correlation Network</h3>
                    <div id="correlationNetwork" class="chart-container"></div>
                </div>
            </div>

            <!-- Model Performance Matrix -->
            <div class="dashboard-item">
                <div class="card">
                    <h3>Model Performance</h3>
                    <div id="performanceMatrix" class="chart-container"></div>
                </div>
            </div>

            <!-- Forecast Comparison -->
            <div class="dashboard-item wide">
                <div class="card">
                    <h3>Forecast Comparison</h3>
                    <div id="forecastComparison" class="chart-container"></div>
                </div>
            </div>
        </div>
        {% endblock %}

        {% block scripts %}
        <script>
            // WebSocket connection
            const ws = new WebSocket("ws://localhost:8765/ws");
            let selectedKpis = new Set();
            let refreshInterval = null;
            
            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                initializeKpiSelector();
                initializeTimeRange();
                initializeRefreshControl();
                loadAllVisualizations();
            });

            function initializeKpiSelector() {
                fetch('/api/kpis')
                    .then(response => response.json())
                    .then(kpis => {
                        const kpiList = document.getElementById('kpi-list');
                        Object.keys(kpis).forEach(kpi => {
                            const div = document.createElement('div');
                            div.className = 'kpi-item';
                            div.innerHTML = `
                                <input type="checkbox" id="kpi-${kpi}" value="${kpi}">
                                <label for="kpi-${kpi}">${kpi}</label>
                            `;
                            kpiList.appendChild(div);
                            
                            div.querySelector('input').addEventListener('change', function(e) {
                                if (e.target.checked) {
                                    selectedKpis.add(kpi);
                                } else {
                                    selectedKpis.delete(kpi);
                                }
                                refreshVisualizations();
                            });
                        });
                    });
            }

            function initializeTimeRange() {
                const timeRange = document.getElementById('timeRange');
                const customRange = document.getElementById('customRange');
                
                timeRange.addEventListener('change', function() {
                    if (this.value === 'custom') {
                        customRange.style.display = 'block';
                    } else {
                        customRange.style.display = 'none';
                    }
                    refreshVisualizations();
                });
            }

            function initializeRefreshControl() {
                const refreshSelect = document.getElementById('refreshInterval');
                const refreshButton = document.getElementById('refreshButton');
                
                refreshSelect.addEventListener('change', function() {
                    if (refreshInterval) {
                        clearInterval(refreshInterval);
                    }
                    
                    const interval = parseInt(this.value);
                    if (interval > 0) {
                        refreshInterval = setInterval(refreshVisualizations, interval);
                    }
                });
                
                refreshButton.addEventListener('click', refreshVisualizations);
            }

            function getTimeRange() {
                const timeRange = document.getElementById('timeRange').value;
                const now = new Date();
                
                switch(timeRange) {
                    case '1h':
                        return {
                            start: new Date(now - 3600000).toISOString(),
                            end: now.toISOString()
                        };
                    case '24h':
                        return {
                            start: new Date(now - 86400000).toISOString(),
                            end: now.toISOString()
                        };
                    case '7d':
                        return {
                            start: new Date(now - 604800000).toISOString(),
                            end: now.toISOString()
                        };
                    case '30d':
                        return {
                            start: new Date(now - 2592000000).toISOString(),
                            end: now.toISOString()
                        };
                    case 'custom':
                        return {
                            start: document.getElementById('startTime').value,
                            end: document.getElementById('endTime').value
                        };
                    default:
                        return {
                            start: new Date(now - 86400000).toISOString(),
                            end: now.toISOString()
                        };
                }
            }

            async function loadAllVisualizations() {
                const timeRange = getTimeRange();
                const kpis = Array.from(selectedKpis);
                
                if (kpis.length === 0) return;
                
                // Load each visualization
                await Promise.all([
                    loadRealtimeChart(kpis, timeRange),
                    loadPatternHeatmap(kpis[0], timeRange),
                    loadDistributionPlot(kpis[0]),
                    loadCorrelationNetwork(),
                    loadPerformanceMatrix(),
                    loadForecastComparison(kpis[0], timeRange)
                ]);
            }

            async function loadRealtimeChart(kpis, timeRange) {
                const promises = kpis.map(kpi =>
                    fetch(`/api/visualizations/${kpi}/realtime?start_time=${timeRange.start}&end_time=${timeRange.end}`)
                        .then(response => response.json())
                );
                
                const data = await Promise.all(promises);
                Plotly.newPlot('realtimeChart', data);
            }

            // Similar functions for other visualizations...
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            function updateDashboard(data) {
                if (data.type === 'new_alert') {
                    addAlert(data.alert);
                } else if (data.type === 'kpi_update') {
                    updateRealtimeChart(data);
                }
            }
            
            function addAlert(alert) {
                const alertsContainer = document.getElementById('alertsContainer');
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert alert-${alert.severity}`;
                alertDiv.innerHTML = `<strong>${alert.kpi_name}</strong>: ${alert.message}`;
                alertsContainer.insertBefore(alertDiv, alertsContainer.firstChild);
                
                // Remove old alerts if too many
                while (alertsContainer.children.length > 10) {
                    alertsContainer.removeChild(alertsContainer.lastChild);
                }
            }
            
            function updateRealtimeChart(data) {
                const update = {
                    x: [[data.timestamp]],
                    y: [[data.value]]
                };
                
                Plotly.extendTraces('realtimeChart', update, [data.kpi_index]);
            }
            
            // Refresh all visualizations
            function refreshVisualizations() {
                loadAllVisualizations();
            }
        </script>
        {% endblock %}
        """
        
        # Save templates
        (self.template_dir / 'base.html').write_text(base_template)
        (self.template_dir / 'dashboard.html').write_text(dashboard_template)

    def setup_routes(self):
        """Setup FastAPI routes"""
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request):
            """Main dashboard view"""
            # Get current statistics
            total_kpis = len(self.monitor.kpi_buffers)
            total_alerts = len(self.monitor.alert_buffer)
            recent_alerts = self.monitor.get_latest_alerts(limit=5)
            
            # Calculate average prediction accuracy
            accuracy = self._calculate_average_accuracy()
            
            return self.templates.TemplateResponse(
                "dashboard.html",
                {
                    "request": request,
                    "total_kpis": total_kpis,
                    "total_alerts": total_alerts,
                    "prediction_accuracy": accuracy,
                    "recent_alerts": recent_alerts
                }
            )

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.active_connections["dashboard"].append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Process any incoming messages if needed
            except WebSocketDisconnect:
                self.active_connections["dashboard"].remove(websocket)

        @self.app.get("/api/kpis")
        async def get_kpis():
            """Get all monitored KPIs"""
            return {
                name: self.monitor.get_kpi_status(name)
                for name in self.monitor.kpi_buffers
            }

        @self.app.get("/api/kpi/{kpi_name}")
        async def get_kpi(kpi_name: str):
            """Get specific KPI data"""
            status = self.monitor.get_kpi_status(kpi_name)
            if not status:
                raise HTTPException(status_code=404, detail="KPI not found")
            return status

        @self.app.get("/api/alerts")
        async def get_alerts(
            severity: Optional[str] = None,
            limit: Optional[int] = None
        ):
            """Get alerts with optional filtering"""
            severity_enum = (
                AlertSeverity[severity.upper()]
                if severity else None
            )
            return [
                alert.to_dict()
                for alert in self.monitor.get_latest_alerts(
                    severity=severity_enum,
                    limit=limit
                )
            ]

        @self.app.post("/api/batch/analyze")
        async def run_batch_analysis(data: Dict):
            """Run batch analysis on multiple KPIs"""
            try:
                result = self.batch_analyzer.analyze_batch(
                    data['kpi_data'],
                    metadata=data.get('metadata')
                )
                return result
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Batch analysis failed: {str(e)}"
                )

        @self.app.get("/api/visualizations/{kpi_name}/{viz_type}")
        async def get_visualization(
            kpi_name: str,
            viz_type: str,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None
        ):
            """Get specific visualization for a KPI"""
            try:
                # Get KPI data
                kpi_data = self.monitor.get_kpi_data(kpi_name, start_time, end_time)
                if not kpi_data:
                    raise HTTPException(status_code=404, detail="KPI data not found")
                
                # Create visualization based on type
                if viz_type == "pattern_heatmap":
                    fig = self.visualizer.create_pattern_heatmap(
                        kpi_data['values'],
                        kpi_data['timestamps']
                    )
                elif viz_type == "anomaly_scatter":
                    fig = self.visualizer.create_anomaly_scatter(
                        kpi_data['values'],
                        kpi_data['timestamps'],
                        kpi_data['anomalies']
                    )
                elif viz_type == "distribution":
                    fig = self.visualizer.create_distribution_plot(
                        kpi_data['values']
                    )
                elif viz_type == "decomposition":
                    fig = self.visualizer.create_decomposition_plot(
                        kpi_data['values'],
                        kpi_data['timestamps']
                    )
                elif viz_type == "forecast_comparison":
                    fig = self.visualizer.create_forecast_comparison(
                        kpi_data['values'],
                        kpi_data['predictions'],
                        kpi_data['timestamps']
                    )
                elif viz_type == "realtime":
                    fig = self.visualizer.create_realtime_plot(
                        kpi_data['values'],
                        kpi_data['timestamps']
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported visualization type: {viz_type}"
                    )
                
                return JSONResponse(content=fig.to_json())
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Visualization generation failed: {str(e)}"
                )

        @self.app.get("/api/correlation_network")
        async def get_correlation_network(
            threshold: float = 0.5
        ):
            """Get correlation network for all KPIs"""
            try:
                # Get data for all KPIs
                kpi_data = {
                    name: self.monitor.get_kpi_data(name)['values']
                    for name in self.monitor.kpi_buffers
                }
                
                fig = self.visualizer.create_correlation_network(
                    kpi_data,
                    threshold=threshold
                )
                
                return JSONResponse(content=fig.to_json())
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Correlation network generation failed: {str(e)}"
                )

        @self.app.get("/api/performance_matrix")
        async def get_performance_matrix():
            """Get performance comparison matrix for all models"""
            try:
                metrics = self.analytics.get_model_metrics()
                fig = self.visualizer.create_performance_matrix(metrics)
                return JSONResponse(content=fig.to_json())
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Performance matrix generation failed: {str(e)}"
                )

    def _calculate_average_accuracy(self) -> float:
        """Calculate average prediction accuracy across all KPIs"""
        accuracies = []
        for kpi_name in self.monitor.kpi_buffers:
            status = self.monitor.get_kpi_status(kpi_name)
            if status and 'accuracy' in status:
                accuracies.append(status['accuracy'])
        
        return round(sum(accuracies) / len(accuracies), 2) if accuracies else 0.0

    def run(self, host: str = "localhost", port: int = 8000):
        """Run the dashboard server"""
        uvicorn.run(self.app, host=host, port=port)
