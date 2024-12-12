from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from io import BytesIO
import base64
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
        
    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for the report"""
        custom_styles = {}
        
        # Header style
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f497d')
        )
        custom_styles['Header'] = header_style
        
        # Subheader style
        subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#4f81bd')
        )
        custom_styles['SubHeader'] = subheader_style
        
        # Section style
        section_style = ParagraphStyle(
            'CustomSection',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            leading=16
        )
        custom_styles['Section'] = section_style
        
        # Recommendation style
        recommendation_style = ParagraphStyle(
            'CustomRecommendation',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            leftIndent=20,
            textColor=colors.HexColor('#006400')
        )
        custom_styles['Recommendation'] = recommendation_style
        
        return custom_styles

    def generate_pdf_report(self, 
                          report_data: Dict[str, Any], 
                          output_path: str,
                          kpi_values: List[float],
                          timestamps: List[datetime]) -> None:
        """Generate a PDF report with all analyses and visualizations"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build the story (content) of the PDF
        story = []
        
        # Add title
        story.append(Paragraph(
            f"Statistical Analysis Report - {report_data['summary']['title']}",
            self.custom_styles['Header']
        ))
        story.append(Spacer(1, 30))
        
        # Add executive summary
        story.extend(self._create_executive_summary(report_data['summary']))
        story.append(Spacer(1, 20))
        
        # Add main visualizations
        story.extend(self._create_visualizations(kpi_values, timestamps))
        story.append(Spacer(1, 20))
        
        # Add statistical analysis sections
        story.extend(self._create_statistical_analysis(report_data))
        story.append(Spacer(1, 20))
        
        # Add recommendations
        story.extend(self._create_recommendations(report_data['recommendations']))
        
        # Build the PDF
        doc.build(story)

    def _create_executive_summary(self, summary_data: Dict) -> List:
        """Create executive summary section"""
        elements = []
        
        # Add summary header
        elements.append(Paragraph('Executive Summary', self.custom_styles['SubHeader']))
        
        # Add key findings
        for finding in summary_data['key_findings']:
            elements.append(Paragraph(
                f"• {finding}",
                self.custom_styles['Section']
            ))
        
        # Add quality scores
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(
            f"Data Quality Score: {summary_data['data_quality']}/10",
            self.custom_styles['Section']
        ))
        elements.append(Paragraph(
            f"Reliability Score: {summary_data['reliability_score']}/10",
            self.custom_styles['Section']
        ))
        
        return elements

    def _create_visualizations(self, 
                             values: List[float], 
                             timestamps: List[datetime]) -> List:
        """Create visualization section with charts"""
        elements = []
        
        # Add visualization header
        elements.append(Paragraph('Data Visualizations', self.custom_styles['SubHeader']))
        
        # Time Series Chart
        drawing = Drawing(500, 200)
        chart = HorizontalLineChart()
        chart.x = 50
        chart.y = 50
        chart.height = 125
        chart.width = 400
        
        # Add data to chart
        chart.data = [values]
        chart.categoryAxis.categoryNames = [t.strftime('%Y-%m-%d') for t in timestamps]
        chart.valueAxis.valueMin = min(values)
        chart.valueAxis.valueMax = max(values)
        
        drawing.add(chart)
        elements.append(drawing)
        elements.append(Spacer(1, 20))
        
        # Distribution Chart
        dist_drawing = Drawing(500, 200)
        hist = VerticalBarChart()
        hist.x = 50
        hist.y = 50
        hist.height = 125
        hist.width = 400
        
        # Calculate histogram data
        hist_values, _ = np.histogram(values, bins=20)
        hist.data = [hist_values.tolist()]
        
        dist_drawing.add(hist)
        elements.append(dist_drawing)
        
        return elements

    def _create_statistical_analysis(self, report_data: Dict) -> List:
        """Create statistical analysis section"""
        elements = []
        
        # Add analysis header
        elements.append(Paragraph('Statistical Analysis', self.custom_styles['SubHeader']))
        
        # Stationarity Analysis
        elements.append(Paragraph('Stationarity Analysis', self.styles['Heading3']))
        elements.append(Paragraph(
            f"ADF Test Result: {report_data['stationarity']['adf_test']['result']}",
            self.custom_styles['Section']
        ))
        elements.append(Paragraph(
            f"Confidence: {report_data['stationarity']['adf_test']['confidence']}",
            self.custom_styles['Section']
        ))
        
        # Distribution Analysis
        elements.append(Paragraph('Distribution Analysis', self.styles['Heading3']))
        elements.append(Paragraph(
            f"Distribution Type: {report_data['distribution']['distribution_type']}",
            self.custom_styles['Section']
        ))
        
        # Seasonality Analysis
        elements.append(Paragraph('Seasonality Analysis', self.styles['Heading3']))
        if report_data['seasonality']['has_seasonality']:
            elements.append(Paragraph(
                f"Primary Period: {report_data['seasonality']['primary_period']} units",
                self.custom_styles['Section']
            ))
        else:
            elements.append(Paragraph(
                "No significant seasonality detected",
                self.custom_styles['Section']
            ))
        
        return elements

    def _create_recommendations(self, recommendations: List[Dict]) -> List:
        """Create recommendations section"""
        elements = []
        
        # Add recommendations header
        elements.append(Paragraph('Recommendations', self.custom_styles['SubHeader']))
        
        # Add each recommendation
        for rec in recommendations:
            elements.append(Paragraph(
                f"Category: {rec['category']}",
                self.styles['Heading4']
            ))
            elements.append(Paragraph(
                rec['recommendation'],
                self.custom_styles['Recommendation']
            ))
            elements.append(Paragraph(
                f"Priority: {rec['priority']} | Expected Impact: {rec['impact']}",
                self.custom_styles['Section']
            ))
            elements.append(Spacer(1, 10))
        
        return elements

    def add_watermark(self, pdf_path: str) -> None:
        """Add watermark to the PDF"""
        from PyPDF2 import PdfReader, PdfWriter
        
        # Create watermark
        watermark = Drawing(600, 800)
        watermark.add(String(300, 400, "CONFIDENTIAL", textAnchor='middle'))
        
        # Add watermark to each page
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        for page in reader.pages:
            page.merge_page(watermark)
            writer.add_page(page)
        
        # Save watermarked PDF
        with open(pdf_path, 'wb') as file:
            writer.write(file)

    def _create_table(self, data: List[List[str]], style: List = None) -> Table:
        """Create a formatted table"""
        table = Table(data)
        
        if style is None:
            style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f81bd')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]
            
        table.setStyle(TableStyle(style))
        return table
