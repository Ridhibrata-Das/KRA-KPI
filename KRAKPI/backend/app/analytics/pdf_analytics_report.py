from typing import List, Dict, Optional
from datetime import datetime
import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

from .prediction_alert_system import AlertSeverity, AlertType, PredictionAlert

class PDFAnalyticsReport:
    def __init__(self, output_dir: str = './reports'):
        """Initialize the PDF report generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12
        ))
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=10
        ))

    def generate_report(self,
                       analysis_result: 'AnalysisResult',
                       include_plots: bool = True) -> str:
        """Generate a comprehensive PDF report for a KPI analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"kpi_report_{analysis_result.kpi_name}_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(
            str(filename),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Build the document content
        story = []
        
        # Title
        story.append(Paragraph(
            f"KPI Analysis Report: {analysis_result.kpi_name}",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.extend(self._create_executive_summary(analysis_result))
        
        # Statistical Analysis
        story.extend(self._create_statistical_analysis(analysis_result))
        
        # Predictions
        story.extend(self._create_predictions_section(analysis_result))
        
        # Alerts
        story.extend(self._create_alerts_section(analysis_result))
        
        # Performance Metrics
        story.extend(self._create_performance_section(analysis_result))
        
        # Plots
        if include_plots:
            story.extend(self._create_visualization_section(analysis_result))
        
        # Build the PDF
        doc.build(story)
        return str(filename)

    def _create_executive_summary(self, result: 'AnalysisResult') -> List:
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph('Executive Summary', self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Summary text
        summary = [
            f"Analysis performed on: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total alerts generated: {len(result.alerts)}",
            f"Critical alerts: {len([a for a in result.alerts if a.severity == AlertSeverity.CRITICAL])}",
            f"Statistical validity: {'✓' if result.statistical_tests.get('is_valid', False) else '✗'}"
        ]
        
        for line in summary:
            elements.append(Paragraph(line, self.styles['Normal']))
            elements.append(Spacer(1, 6))
        
        elements.append(Spacer(1, 12))
        return elements

    def _create_statistical_analysis(self, result: 'AnalysisResult') -> List:
        """Create statistical analysis section"""
        elements = []
        
        elements.append(Paragraph('Statistical Analysis', self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Create table of statistical tests
        data = [['Test', 'Result', 'P-Value']]
        for test_name, test_result in result.statistical_tests.items():
            if isinstance(test_result, dict):
                data.append([
                    test_name,
                    '✓' if test_result.get('result', False) else '✗',
                    f"{test_result.get('p_value', 'N/A'):.4f}"
                ])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        return elements

    def _create_predictions_section(self, result: 'AnalysisResult') -> List:
        """Create predictions section"""
        elements = []
        
        elements.append(Paragraph('Predictions Analysis', self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Ensemble predictions summary
        ensemble_pred = result.predictions['ensemble']
        pred_values = ensemble_pred['values']
        
        summary = [
            f"Forecast Horizon: {len(pred_values)} periods",
            f"Average Predicted Value: {np.mean(pred_values):.2f}",
            f"Prediction Range: {min(pred_values):.2f} to {max(pred_values):.2f}",
            f"Average Uncertainty: {np.mean(np.array(ensemble_pred['uncertainty']['upper']) - np.array(ensemble_pred['uncertainty']['lower'])):.2f}"
        ]
        
        for line in summary:
            elements.append(Paragraph(line, self.styles['Normal']))
            elements.append(Spacer(1, 6))
        
        elements.append(Spacer(1, 12))
        return elements

    def _create_alerts_section(self, result: 'AnalysisResult') -> List:
        """Create alerts section"""
        elements = []
        
        elements.append(Paragraph('Alert Analysis', self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Group alerts by severity
        alerts_by_severity = {}
        for alert in result.alerts:
            if alert.severity not in alerts_by_severity:
                alerts_by_severity[alert.severity] = []
            alerts_by_severity[alert.severity].append(alert)
        
        # Create alert summary table
        data = [['Severity', 'Count', 'Latest Alert']]
        for severity in AlertSeverity:
            alerts = alerts_by_severity.get(severity, [])
            latest = max(alerts, key=lambda x: x.timestamp).message if alerts else 'N/A'
            data.append([severity.value, len(alerts), latest])
        
        table = Table(data, colWidths=[1*inch, 0.8*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        return elements

    def _create_performance_section(self, result: 'AnalysisResult') -> List:
        """Create performance metrics section"""
        elements = []
        
        elements.append(Paragraph('Model Performance', self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Create performance metrics table
        data = [['Model', 'MSE', 'MAE', 'MAPE']]
        for model, metrics in result.accuracy_metrics.items():
            data.append([
                model,
                f"{metrics['mse']:.4f}",
                f"{metrics['mae']:.4f}",
                f"{metrics['mape']:.2f}%"
            ])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        return elements

    def _create_visualization_section(self, result: 'AnalysisResult') -> List:
        """Create visualization section"""
        elements = []
        
        elements.append(Paragraph('Visualizations', self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Create prediction plot
        plt.figure(figsize=(10, 6))
        for model, pred_data in result.predictions.items():
            if model != 'ensemble':
                plt.plot(pred_data['values'], label=model)
        plt.title('Model Predictions Comparison')
        plt.xlabel('Time Period')
        plt.ylabel('Value')
        plt.legend()
        
        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add plot to PDF
        img = Image(buffer)
        img.drawHeight = 4*inch
        img.drawWidth = 6*inch
        elements.append(img)
        elements.append(Spacer(1, 20))
        
        return elements

    def batch_generate_reports(self,
                             analysis_results: List['AnalysisResult'],
                             include_plots: bool = True) -> List[str]:
        """Generate reports for multiple analysis results"""
        return [
            self.generate_report(result, include_plots)
            for result in analysis_results
        ]
