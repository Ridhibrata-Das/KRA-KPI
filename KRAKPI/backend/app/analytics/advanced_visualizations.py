from typing import List, Dict, Optional, Union, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from datetime import datetime
import pandas as pd
from scipy import stats
import json

class AdvancedVisualizer:
    def __init__(self, theme: str = 'plotly_white'):
        """Initialize advanced visualization module"""
        self.theme = theme
        self.colors = px.colors.qualitative.Set3

    def create_pattern_heatmap(self,
                             values: List[float],
                             timestamps: List[datetime],
                             window_size: int = 24,
                             overlap: int = 12) -> go.Figure:
        """Create a pattern heatmap showing recurring patterns"""
        # Prepare data
        data = np.array(values)
        n_patterns = len(data) - window_size + 1
        pattern_matrix = np.zeros((n_patterns, window_size))
        
        for i in range(n_patterns):
            pattern_matrix[i] = data[i:i+window_size]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pattern_matrix,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title='Pattern Heatmap Analysis',
            xaxis_title='Time Window',
            yaxis_title='Pattern Sequence',
            template=self.theme
        )
        
        return fig

    def create_correlation_network(self,
                                 kpi_data: Dict[str, List[float]],
                                 threshold: float = 0.5) -> go.Figure:
        """Create an interactive correlation network visualization"""
        # Calculate correlations
        df = pd.DataFrame(kpi_data)
        corr_matrix = df.corr()
        
        # Create network
        edges_x = []
        edges_y = []
        edge_colors = []
        
        nodes_x = []
        nodes_y = []
        node_labels = []
        
        # Layout nodes in a circle
        n_nodes = len(corr_matrix)
        for i, kpi in enumerate(corr_matrix.index):
            angle = 2 * np.pi * i / n_nodes
            x = np.cos(angle)
            y = np.sin(angle)
            nodes_x.append(x)
            nodes_y.append(y)
            node_labels.append(kpi)
        
        # Create edges for correlations above threshold
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    edges_x.extend([nodes_x[i], nodes_x[j], None])
                    edges_y.extend([nodes_y[i], nodes_y[j], None])
                    edge_colors.append(corr_matrix.iloc[i, j])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edges_x,
            y=edges_y,
            mode='lines',
            line=dict(
                color=edge_colors,
                colorscale='RdBu',
                width=2
            ),
            hoverinfo='none'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=nodes_x,
            y=nodes_y,
            mode='markers+text',
            marker=dict(size=20, color='skyblue'),
            text=node_labels,
            textposition='top center'
        ))
        
        fig.update_layout(
            title='KPI Correlation Network',
            showlegend=False,
            template=self.theme
        )
        
        return fig

    def create_anomaly_scatter(self,
                             values: List[float],
                             timestamps: List[datetime],
                             anomalies: List[bool]) -> go.Figure:
        """Create scatter plot highlighting anomalies"""
        fig = go.Figure()
        
        # Normal points
        normal_mask = ~np.array(anomalies)
        fig.add_trace(go.Scatter(
            x=np.array(timestamps)[normal_mask],
            y=np.array(values)[normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=8)
        ))
        
        # Anomaly points
        anomaly_mask = np.array(anomalies)
        fig.add_trace(go.Scatter(
            x=np.array(timestamps)[anomaly_mask],
            y=np.array(values)[anomaly_mask],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=12, symbol='x')
        ))
        
        fig.update_layout(
            title='Anomaly Detection Visualization',
            xaxis_title='Timestamp',
            yaxis_title='Value',
            template=self.theme
        )
        
        return fig

    def create_distribution_plot(self,
                               values: List[float],
                               bins: int = 30) -> go.Figure:
        """Create an interactive distribution plot with statistical overlay"""
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3])
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=bins,
                name='Distribution'
            ),
            row=1, col=1
        )
        
        # Kernel density estimation
        kde_x = np.linspace(min(values), max(values), 100)
        kde = stats.gaussian_kde(values)
        kde_y = kde(kde_x)
        
        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde_y,
                name='KDE',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                x=values,
                name='Box Plot'
            ),
            row=2, col=1
        )
        
        # Add statistical annotations
        stats_text = f"""
        Mean: {np.mean(values):.2f}
        Median: {np.median(values):.2f}
        Std Dev: {np.std(values):.2f}
        Skewness: {stats.skew(values):.2f}
        Kurtosis: {stats.kurtosis(values):.2f}
        """
        
        fig.add_annotation(
            xref='paper',
            yref='paper',
            x=1.1,
            y=0.5,
            text=stats_text,
            showarrow=False,
            font=dict(size=12)
        )
        
        fig.update_layout(
            title='Value Distribution Analysis',
            template=self.theme,
            showlegend=True,
            height=800
        )
        
        return fig

    def create_decomposition_plot(self,
                                values: List[float],
                                timestamps: List[datetime],
                                period: Optional[int] = None) -> go.Figure:
        """Create time series decomposition plot"""
        # Convert to pandas series
        ts = pd.Series(values, index=timestamps)
        
        # Detect period if not provided
        if period is None:
            period = self._detect_seasonality(values)
        
        # Perform decomposition
        decomposition = stats.seasonal_decompose(
            ts,
            period=period,
            extrapolate_trend='freq'
        )
        
        # Create figure
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                'Original',
                'Trend',
                'Seasonal',
                'Residual'
            )
        )
        
        # Add components
        components = [
            (ts, 'Original'),
            (decomposition.trend, 'Trend'),
            (decomposition.seasonal, 'Seasonal'),
            (decomposition.resid, 'Residual')
        ]
        
        for i, (data, name) in enumerate(components, 1):
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=data,
                    name=name
                ),
                row=i,
                col=1
            )
        
        fig.update_layout(
            height=1000,
            title='Time Series Decomposition',
            template=self.theme
        )
        
        return fig

    def create_forecast_comparison(self,
                                 actual: List[float],
                                 predictions: Dict[str, List[float]],
                                 timestamps: List[datetime]) -> go.Figure:
        """Create interactive forecast comparison plot"""
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=actual,
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Add predictions from each model
        for i, (model_name, pred_values) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=pred_values,
                name=f'{model_name} Forecast',
                line=dict(color=self.colors[i % len(self.colors)])
            ))
        
        # Add buttons for different time ranges
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig.data)}],
                            label="All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [True] + [False] * (len(fig.data)-1)}],
                            label="Actual Only",
                            method="restyle"
                        )
                    ]),
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )
        
        fig.update_layout(
            title='Forecast Model Comparison',
            xaxis_title='Time',
            yaxis_title='Value',
            template=self.theme,
            height=600
        )
        
        return fig

    def create_performance_matrix(self,
                                metrics: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create performance metrics comparison matrix"""
        # Prepare data
        models = list(metrics.keys())
        metric_types = list(metrics[models[0]].keys())
        
        z_data = []
        for metric in metric_types:
            row = [metrics[model][metric] for model in models]
            z_data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=models,
            y=metric_types,
            colorscale='RdYlBu_r'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison Matrix',
            xaxis_title='Model',
            yaxis_title='Metric',
            template=self.theme
        )
        
        return fig

    def _detect_seasonality(self, values: List[float]) -> int:
        """Detect seasonality period using autocorrelation"""
        n = len(values)
        if n < 4:
            return 1
            
        acf = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
        acf = acf[n-1:]
        
        # Find first peak after lag 0
        peaks = np.where((acf[1:] > acf[:-1])[:-1] & (acf[1:] > acf[2:]))[0] + 1
        
        if len(peaks) > 0:
            return int(peaks[0])
        return 1

    def export_to_html(self,
                      figures: Dict[str, go.Figure],
                      filename: str = 'dashboard.html'):
        """Export multiple figures to an interactive HTML dashboard"""
        with open(filename, 'w') as f:
            f.write("""
            <html>
            <head>
                <title>KPI Analytics Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    .chart-container {
                        width: 100%;
                        margin-bottom: 20px;
                    }
                </style>
            </head>
            <body>
            """)
            
            for name, fig in figures.items():
                f.write(f'<div class="chart-container" id="{name}"></div>')
                f.write(f'<script>{fig.to_json()}</script>')
                f.write(f'''
                <script>
                    Plotly.newPlot("{name}", 
                        {fig.to_json()}.data,
                        {fig.to_json()}.layout);
                </script>
                ''')
            
            f.write("</body></html>")
