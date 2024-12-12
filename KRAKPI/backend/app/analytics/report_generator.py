from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from .statistical_validation import StatisticalValidator
from .pattern_detection import PatternDetector

class StatisticalReportGenerator:
    def __init__(self, kpi_name: str = "KPI"):
        self.kpi_name = kpi_name
        self.validator = StatisticalValidator()
        self.pattern_detector = PatternDetector()
        
    def generate_complete_report(self, 
                               values: List[float], 
                               timestamps: List[datetime],
                               output_format: str = 'html') -> str:
        """Generate a complete statistical analysis report"""
        # Gather all statistical results
        stats_results = self.validator.validate_all(values, timestamps)
        pattern_results = self.pattern_detector.detect_all_patterns(values, timestamps)
        
        # Generate visualizations
        plots = self._generate_plots(values, timestamps, stats_results, pattern_results)
        
        # Create report sections
        sections = {
            'summary': self._generate_summary_section(stats_results, pattern_results),
            'stationarity': self._generate_stationarity_section(stats_results['stationarity']),
            'patterns': self._generate_patterns_section(pattern_results),
            'distribution': self._generate_distribution_section(stats_results['normality']),
            'seasonality': self._generate_seasonality_section(stats_results['seasonality']),
            'anomalies': self._generate_anomalies_section(stats_results, pattern_results),
            'recommendations': self._generate_recommendations(stats_results, pattern_results)
        }
        
        if output_format == 'html':
            return self._generate_html_report(sections, plots)
        elif output_format == 'json':
            return self._generate_json_report(sections)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _generate_plots(self, 
                       values: List[float], 
                       timestamps: List[datetime],
                       stats_results: Dict,
                       pattern_results: Dict) -> Dict:
        """Generate all visualizations for the report"""
        plots = {}
        
        # Time series plot with trends and patterns
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Time Series with Patterns', 'Distribution Analysis'))
        
        # Main time series
        fig.add_trace(
            go.Scatter(x=timestamps, y=values, name='Original Data',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add trend line if exists
        if 'trends' in pattern_results:
            trend = np.polyval(np.polyfit(range(len(values)), values, 1), range(len(values)))
            fig.add_trace(
                go.Scatter(x=timestamps, y=trend, name='Trend',
                          line=dict(color='red', dash='dash')),
                row=1, col=1
            )
        
        # Distribution plot
        fig.add_trace(
            go.Histogram(y=values, name='Distribution',
                        nbinsy=30, orientation='h'),
            row=2, col=1
        )
        
        plots['main_plot'] = fig.to_html(full_html=False)
        
        # Generate additional specialized plots
        plots.update(self._generate_specialized_plots(values, timestamps, stats_results))
        
        return plots

    def _generate_specialized_plots(self, 
                                  values: List[float], 
                                  timestamps: List[datetime],
                                  stats_results: Dict) -> Dict:
        """Generate specialized statistical plots"""
        plots = {}
        
        # ACF/PACF plot
        fig_acf = make_subplots(rows=2, cols=1, 
                               subplot_titles=('Autocorrelation', 'Partial Autocorrelation'))
        
        # Calculate ACF/PACF
        acf_values = pd.Series(values).autocorr()
        pacf_values = pd.Series(values).corr(pd.Series(values).shift(1))
        
        fig_acf.add_trace(
            go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'),
            row=1, col=1
        )
        fig_acf.add_trace(
            go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF'),
            row=2, col=1
        )
        
        plots['acf_pacf_plot'] = fig_acf.to_html(full_html=False)
        
        # QQ Plot
        fig_qq = go.Figure()
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(values)))
        sorted_values = np.sort(values)
        
        fig_qq.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_values, mode='markers',
                      name='Q-Q Plot')
        )
        
        plots['qq_plot'] = fig_qq.to_html(full_html=False)
        
        return plots

    def _generate_summary_section(self, 
                                stats_results: Dict, 
                                pattern_results: Dict) -> Dict:
        """Generate executive summary of the analysis"""
        return {
            'title': f'Statistical Analysis Summary for {self.kpi_name}',
            'key_findings': [
                self._get_stationarity_summary(stats_results['stationarity']),
                self._get_distribution_summary(stats_results['normality']),
                self._get_pattern_summary(pattern_results),
                self._get_seasonality_summary(stats_results['seasonality'])
            ],
            'data_quality': self._assess_data_quality(stats_results),
            'reliability_score': self._calculate_reliability_score(stats_results)
        }

    def _generate_stationarity_section(self, stationarity_results: Dict) -> Dict:
        """Generate detailed stationarity analysis section"""
        return {
            'title': 'Stationarity Analysis',
            'adf_test': {
                'result': 'Stationary' if stationarity_results['adf_test']['is_stationary'] 
                         else 'Non-stationary',
                'confidence': f"{(1 - stationarity_results['adf_test']['p_value']) * 100:.2f}%",
                'interpretation': self._interpret_stationarity_test(stationarity_results['adf_test'])
            },
            'kpss_test': {
                'result': 'Stationary' if stationarity_results['kpss_test']['is_stationary']
                         else 'Non-stationary',
                'confidence': f"{(1 - stationarity_results['kpss_test']['p_value']) * 100:.2f}%"
            },
            'trend_analysis': self._analyze_trend_strength(stationarity_results['rolling_statistics'])
        }

    def _generate_patterns_section(self, pattern_results: Dict) -> Dict:
        """Generate pattern analysis section"""
        return {
            'title': 'Pattern Analysis',
            'trends': self._analyze_trends(pattern_results.get('trends', {})),
            'cycles': self._analyze_cycles(pattern_results.get('cycles', {})),
            'repeated_patterns': self._analyze_repeated_patterns(
                pattern_results.get('repeated_patterns', {})
            )
        }

    def _generate_distribution_section(self, normality_results: Dict) -> Dict:
        """Generate distribution analysis section"""
        return {
            'title': 'Distribution Analysis',
            'normality': {
                'is_normal': normality_results['shapiro_test']['is_normal'],
                'confidence': f"{(1 - normality_results['shapiro_test']['p_value']) * 100:.2f}%",
                'skewness': normality_results['distribution_metrics']['skewness'],
                'kurtosis': normality_results['distribution_metrics']['kurtosis']
            },
            'distribution_type': self._determine_distribution_type(normality_results),
            'recommendations': self._get_distribution_recommendations(normality_results)
        }

    def _generate_seasonality_section(self, seasonality_results: Dict) -> Dict:
        """Generate seasonality analysis section"""
        return {
            'title': 'Seasonality Analysis',
            'has_seasonality': seasonality_results['overall_seasonality']['detected'],
            'primary_period': seasonality_results['overall_seasonality']['strongest_period'],
            'seasonal_patterns': self._analyze_seasonal_patterns(
                seasonality_results['seasonal_periods']
            )
        }

    def _generate_anomalies_section(self, 
                                  stats_results: Dict, 
                                  pattern_results: Dict) -> Dict:
        """Generate anomalies analysis section"""
        return {
            'title': 'Anomalies Analysis',
            'outliers': self._analyze_outliers(pattern_results.get('outliers', {})),
            'structural_breaks': self._analyze_structural_breaks(
                stats_results['structural_breaks']
            ),
            'volatility': self._analyze_volatility(stats_results['arch_effects'])
        }

    def _generate_recommendations(self, 
                                stats_results: Dict, 
                                pattern_results: Dict) -> List[Dict]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Stationarity recommendations
        if not stats_results['stationarity']['adf_test']['is_stationary']:
            recommendations.append({
                'category': 'Stationarity',
                'recommendation': 'Consider differencing or detrending the data',
                'priority': 'High',
                'impact': 'Improved forecasting accuracy'
            })
        
        # Distribution recommendations
        if not stats_results['normality']['shapiro_test']['is_normal']:
            recommendations.append({
                'category': 'Distribution',
                'recommendation': 'Use non-parametric statistical methods',
                'priority': 'Medium',
                'impact': 'More reliable statistical inference'
            })
        
        # Seasonality recommendations
        if stats_results['seasonality']['overall_seasonality']['detected']:
            recommendations.append({
                'category': 'Seasonality',
                'recommendation': 'Implement seasonal adjustment in analysis',
                'priority': 'High',
                'impact': 'Better trend identification'
            })
        
        return recommendations

    def _generate_html_report(self, sections: Dict, plots: Dict) -> str:
        """Generate HTML report using template"""
        template = Template('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Statistical Analysis Report - {{ sections.summary.title }}</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }
                .section { margin-bottom: 30px; }
                .plot { margin: 20px 0; }
                .recommendation { 
                    padding: 10px; 
                    margin: 10px 0; 
                    border-left: 4px solid #007bff; 
                }
                .high-priority { border-color: #dc3545; }
                .medium-priority { border-color: #ffc107; }
                .low-priority { border-color: #28a745; }
            </style>
        </head>
        <body>
            <h1>{{ sections.summary.title }}</h1>
            
            <!-- Executive Summary -->
            <div class="section">
                <h2>Executive Summary</h2>
                <ul>
                {% for finding in sections.summary.key_findings %}
                    <li>{{ finding }}</li>
                {% endfor %}
                </ul>
                <p>Data Quality Score: {{ sections.summary.data_quality }}/10</p>
                <p>Reliability Score: {{ sections.summary.reliability_score }}/10</p>
            </div>
            
            <!-- Main Plots -->
            <div class="section">
                <h2>Time Series Analysis</h2>
                <div class="plot">{{ plots.main_plot }}</div>
            </div>
            
            <!-- Statistical Analysis -->
            <div class="section">
                <h2>Statistical Analysis</h2>
                
                <!-- Stationarity -->
                <h3>Stationarity Analysis</h3>
                <p>{{ sections.stationarity.adf_test.interpretation }}</p>
                
                <!-- Distribution -->
                <h3>Distribution Analysis</h3>
                <p>Distribution Type: {{ sections.distribution.distribution_type }}</p>
                <div class="plot">{{ plots.qq_plot }}</div>
                
                <!-- Seasonality -->
                <h3>Seasonality Analysis</h3>
                {% if sections.seasonality.has_seasonality %}
                <p>Primary Seasonal Period: {{ sections.seasonality.primary_period }} units</p>
                {% else %}
                <p>No significant seasonality detected</p>
                {% endif %}
            </div>
            
            <!-- Recommendations -->
            <div class="section">
                <h2>Recommendations</h2>
                {% for rec in sections.recommendations %}
                <div class="recommendation {{ rec.priority.lower() }}-priority">
                    <h4>{{ rec.category }}</h4>
                    <p>{{ rec.recommendation }}</p>
                    <p><strong>Priority:</strong> {{ rec.priority }}</p>
                    <p><strong>Expected Impact:</strong> {{ rec.impact }}</p>
                </div>
                {% endfor %}
            </div>
        </body>
        </html>
        ''')
        
        return template.render(sections=sections, plots=plots)

    def _generate_json_report(self, sections: Dict) -> str:
        """Generate JSON format report"""
        return json.dumps(sections, indent=2)

    def _assess_data_quality(self, stats_results: Dict) -> int:
        """Assess overall data quality on a scale of 1-10"""
        score = 10
        
        # Deduct points for various quality issues
        if not stats_results['stationarity']['adf_test']['is_stationary']:
            score -= 2
        
        if not stats_results['normality']['shapiro_test']['is_normal']:
            score -= 1
        
        if stats_results['heteroskedasticity']['breusch_pagan_test']['has_heteroskedasticity']:
            score -= 2
        
        if not stats_results['randomness']['runs_test']['is_random']:
            score -= 1
        
        return max(1, score)

    def _calculate_reliability_score(self, stats_results: Dict) -> int:
        """Calculate reliability score on a scale of 1-10"""
        score = 10
        
        # Adjust score based on statistical properties
        if stats_results['arch_effects'].get('arch_model'):
            if stats_results['arch_effects']['arch_model']['has_arch_effects']:
                score -= 2
        
        if stats_results['structural_breaks']['cusum_test']['has_breaks']:
            score -= 2
        
        return max(1, score)

    def _determine_distribution_type(self, normality_results: Dict) -> str:
        """Determine the type of distribution based on statistical tests"""
        if normality_results['shapiro_test']['is_normal']:
            return 'Normal Distribution'
        
        skewness = normality_results['distribution_metrics']['skewness']
        kurtosis = normality_results['distribution_metrics']['kurtosis']
        
        if abs(skewness) > 1:
            if skewness > 0:
                return 'Right-skewed Distribution'
            else:
                return 'Left-skewed Distribution'
        
        if kurtosis > 3:
            return 'Heavy-tailed Distribution'
        elif kurtosis < 3:
            return 'Light-tailed Distribution'
        
        return 'Non-normal Distribution'

    def _get_distribution_recommendations(self, normality_results: Dict) -> List[str]:
        """Get recommendations based on distribution analysis"""
        recommendations = []
        
        if not normality_results['shapiro_test']['is_normal']:
            recommendations.append(
                'Consider using non-parametric statistical methods'
            )
        
        if abs(normality_results['distribution_metrics']['skewness']) > 1:
            recommendations.append(
                'Data transformation might be necessary to handle skewness'
            )
        
        if normality_results['distribution_metrics']['kurtosis'] > 3:
            recommendations.append(
                'Consider robust statistical methods to handle heavy tails'
            )
        
        return recommendations
