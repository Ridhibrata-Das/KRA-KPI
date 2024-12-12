import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd

class PredictionVisualizer:
    def __init__(self, theme: str = 'plotly_white'):
        """Initialize the visualizer with a theme"""
        self.theme = theme
        self.colors = {
            'actual': '#2C3E50',
            'attention_lstm': '#E74C3C',
            'conv_lstm': '#3498DB',
            'bidirectional_gru': '#2ECC71',
            'prophet': '#9B59B6',
            'sarima': '#F1C40F',
            'xgboost': '#1ABC9C',
            'ensemble': '#34495E',
            'uncertainty': 'rgba(149, 165, 166, 0.2)'
        }

    def plot_predictions(self,
                        historical_values: List[float],
                        historical_timestamps: List[datetime],
                        predictions: Dict,
                        future_timestamps: List[datetime],
                        title: str = "KPI Predictions with Uncertainty",
                        height: int = 800) -> go.Figure:
        """Create an interactive plot of predictions with uncertainty bands"""
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                "Predictions with Uncertainty Bands",
                "Model-wise Prediction Comparison"
            ),
            vertical_spacing=0.15
        )

        # Plot historical data
        fig.add_trace(
            go.Scatter(
                x=historical_timestamps,
                y=historical_values,
                name="Historical",
                line=dict(color=self.colors['actual']),
                showlegend=True
            ),
            row=1, col=1
        )

        # Plot predictions for each model
        for model_name, pred_data in predictions.items():
            # Skip ensemble for main plot
            if model_name != 'ensemble':
                fig.add_trace(
                    go.Scatter(
                        x=future_timestamps,
                        y=pred_data['values'],
                        name=f"{model_name.replace('_', ' ').title()}",
                        line=dict(color=self.colors[model_name]),
                        showlegend=True
                    ),
                    row=2, col=1
                )

        # Plot ensemble prediction with uncertainty
        ensemble_data = predictions['ensemble']
        
        # Add uncertainty band
        fig.add_trace(
            go.Scatter(
                x=future_timestamps + future_timestamps[::-1],
                y=ensemble_data['uncertainty']['upper'] + 
                  ensemble_data['uncertainty']['lower'][::-1],
                fill='toself',
                fillcolor=self.colors['uncertainty'],
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True
            ),
            row=1, col=1
        )

        # Add ensemble prediction line
        fig.add_trace(
            go.Scatter(
                x=future_timestamps,
                y=ensemble_data['values'],
                name='Ensemble Prediction',
                line=dict(
                    color=self.colors['ensemble'],
                    width=3,
                    dash='dash'
                ),
                showlegend=True
            ),
            row=1, col=1
        )

        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme,
            height=height,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)

        return fig

    def plot_model_performance(self,
                             accuracy_metrics: Dict,
                             title: str = "Model Performance Comparison",
                             height: int = 600) -> go.Figure:
        """Create an interactive plot comparing model performance metrics"""
        
        # Prepare data
        models = []
        mse = []
        mae = []
        mape = []
        coverage = []

        for model, metrics in accuracy_metrics.items():
            models.append(model.replace('_', ' ').title())
            mse.append(metrics['mse'])
            mae.append(metrics['mae'])
            mape.append(metrics['mape'])
            coverage.append(metrics.get('interval_coverage', None))

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Mean Squared Error",
                "Mean Absolute Error",
                "Mean Absolute Percentage Error",
                "Prediction Interval Coverage"
            )
        )

        # Add bars for each metric
        fig.add_trace(
            go.Bar(x=models, y=mse, name="MSE",
                  marker_color=list(self.colors.values())),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=models, y=mae, name="MAE",
                  marker_color=list(self.colors.values())),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=models, y=mape, name="MAPE (%)",
                  marker_color=list(self.colors.values())),
            row=2, col=1
        )
        
        # Add coverage plot only for models with uncertainty estimates
        valid_coverage = [(m, c) for m, c in zip(models, coverage) if c is not None]
        if valid_coverage:
            models_with_coverage, coverage_values = zip(*valid_coverage)
            fig.add_trace(
                go.Bar(
                    x=list(models_with_coverage),
                    y=list(coverage_values),
                    name="Coverage (%)",
                    marker_color=list(self.colors.values())[:len(models_with_coverage)]
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme,
            height=height,
            showlegend=False,
            hovermode='x unified'
        )

        return fig

    def create_prediction_dashboard(self,
                                  historical_values: List[float],
                                  historical_timestamps: List[datetime],
                                  predictions: Dict,
                                  future_timestamps: List[datetime],
                                  accuracy_metrics: Dict,
                                  title: str = "KPI Prediction Dashboard") -> go.Figure:
        """Create a comprehensive dashboard combining predictions and performance metrics"""
        
        # Create main prediction plot
        pred_fig = self.plot_predictions(
            historical_values,
            historical_timestamps,
            predictions,
            future_timestamps
        )
        
        # Create performance metrics plot
        perf_fig = self.plot_model_performance(accuracy_metrics)
        
        # Combine plots into a dashboard
        dashboard = go.Figure()
        
        # Add prediction plot
        for trace in pred_fig.data:
            dashboard.add_trace(trace)
            
        # Add performance metrics plot below
        for trace in perf_fig.data:
            dashboard.add_trace(trace)
        
        # Update layout
        dashboard.update_layout(
            title=title,
            template=self.theme,
            height=1200,
            grid=dict(
                rows=3,
                columns=1,
                pattern='independent',
                roworder='top to bottom'
            )
        )
        
        return dashboard

    def export_dashboard(self,
                        dashboard: go.Figure,
                        filename: str = "prediction_dashboard.html"):
        """Export the dashboard to an HTML file"""
        dashboard.write_html(filename)
