import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from scipy import stats
import plotly.express as px
from scipy.signal import find_peaks

class InteractiveVisualizer:
    def __init__(self, theme: str = 'plotly'):
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set3

    def create_all_visualizations(self, 
                                values: List[float], 
                                timestamps: List[datetime],
                                stats_results: Dict,
                                pattern_results: Dict) -> Dict[str, go.Figure]:
        """Create all interactive visualizations"""
        return {
            'time_series': self.create_time_series_plot(values, timestamps, pattern_results),
            'distribution': self.create_distribution_plot(values, stats_results),
            'seasonality': self.create_seasonality_plot(values, timestamps, pattern_results),
            'correlation': self.create_correlation_plot(values),
            'anomalies': self.create_anomaly_plot(values, timestamps, pattern_results),
            'patterns': self.create_pattern_plot(values, timestamps, pattern_results)
        }

    def create_time_series_plot(self, 
                              values: List[float], 
                              timestamps: List[datetime],
                              pattern_results: Dict) -> go.Figure:
        """Create interactive time series plot with multiple components"""
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Time Series Analysis', 'Components'),
            vertical_spacing=0.15
        )

        # Main time series
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                name='Original',
                line=dict(color='blue'),
                hovertemplate='%{y:.2f}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add trend if available
        if 'trends' in pattern_results:
            trend = pattern_results['trends']['linear']
            trend_line = np.polyval([trend['slope'], trend['intercept']], range(len(values)))
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=trend_line,
                    name='Trend',
                    line=dict(color='red', dash='dash'),
                    hovertemplate='%{y:.2f}<br>%{x}<extra></extra>'
                ),
                row=1, col=1
            )

        # Add moving average
        ma_window = min(30, len(values) // 10)
        ma = pd.Series(values).rolling(window=ma_window).mean()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=ma,
                name=f'{ma_window}-period MA',
                line=dict(color='green', dash='dot'),
                hovertemplate='%{y:.2f}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add volatility
        volatility = pd.Series(values).rolling(window=ma_window).std()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=volatility,
                name='Volatility',
                line=dict(color='orange'),
                hovertemplate='%{y:.2f}<br>%{x}<extra></extra>'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title='Interactive Time Series Analysis',
            showlegend=True,
            hovermode='x unified',
            height=800,
            template=self.theme
        )

        # Add range slider
        fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

        return fig

    def create_distribution_plot(self, 
                               values: List[float],
                               stats_results: Dict) -> go.Figure:
        """Create interactive distribution analysis plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution Histogram',
                'Q-Q Plot',
                'Box Plot',
                'Violin Plot'
            )
        )

        # Histogram with KDE
        hist_values, hist_bins = np.histogram(values, bins='auto', density=True)
        kde_x = np.linspace(min(values), max(values), 100)
        kde = stats.gaussian_kde(values)
        
        fig.add_trace(
            go.Histogram(
                x=values,
                name='Histogram',
                histnorm='probability density',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde(kde_x),
                name='KDE',
                line=dict(color='red')
            ),
            row=1, col=1
        )

        # Q-Q Plot
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(values))
        )
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=np.sort(values),
                mode='markers',
                name='Q-Q Plot'
            ),
            row=1, col=2
        )

        # Box Plot
        fig.add_trace(
            go.Box(
                y=values,
                name='Box Plot',
                boxpoints='outliers'
            ),
            row=2, col=1
        )

        # Violin Plot
        fig.add_trace(
            go.Violin(
                y=values,
                name='Violin Plot',
                box_visible=True,
                meanline_visible=True
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title='Distribution Analysis',
            showlegend=True,
            height=800,
            template=self.theme
        )

        return fig

    def create_seasonality_plot(self,
                              values: List[float],
                              timestamps: List[datetime],
                              pattern_results: Dict) -> go.Figure:
        """Create interactive seasonality analysis plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Seasonal Decomposition',
                'Periodogram',
                'Seasonal Pattern',
                'Seasonal Strength'
            )
        )

        # Seasonal Decomposition
        if 'seasonality' in pattern_results:
            seasonal_data = pattern_results['seasonality']
            for period, data in seasonal_data.items():
                if data['is_significant']:
                    fig.add_trace(
                        go.Scatter(
                            y=data['pattern'],
                            name=f'Period-{period}',
                            line=dict(color=self.color_palette[0])
                        ),
                        row=1, col=1
                    )

        # Periodogram
        frequencies = np.fft.fftfreq(len(values))
        power_spectrum = np.abs(np.fft.fft(values))**2
        
        fig.add_trace(
            go.Scatter(
                x=frequencies[1:len(frequencies)//2],
                y=power_spectrum[1:len(frequencies)//2],
                name='Periodogram',
                line=dict(color=self.color_palette[1])
            ),
            row=1, col=2
        )

        # Seasonal Pattern
        if 'seasonality' in pattern_results:
            strongest_period = pattern_results['seasonality']['strongest_period']
            if strongest_period:
                seasonal_pattern = np.array(values).reshape(-1, strongest_period).mean(axis=0)
                fig.add_trace(
                    go.Scatter(
                        y=seasonal_pattern,
                        name='Average Pattern',
                        line=dict(color=self.color_palette[2])
                    ),
                    row=2, col=1
                )

        # Seasonal Strength
        if 'seasonality' in pattern_results:
            strengths = [
                (period, data['strength'])
                for period, data in pattern_results['seasonality'].items()
            ]
            if strengths:
                periods, strength_values = zip(*strengths)
                fig.add_trace(
                    go.Bar(
                        x=[str(p) for p in periods],
                        y=strength_values,
                        name='Seasonal Strength',
                        marker_color=self.color_palette[3]
                    ),
                    row=2, col=2
                )

        # Update layout
        fig.update_layout(
            title='Seasonality Analysis',
            showlegend=True,
            height=800,
            template=self.theme
        )

        return fig

    def create_correlation_plot(self, values: List[float]) -> go.Figure:
        """Create interactive correlation analysis plot"""
        # Calculate lagged correlations
        max_lag = min(50, len(values) // 4)
        lags = range(1, max_lag + 1)
        
        # ACF
        acf_values = [pd.Series(values).autocorr(lag=lag) for lag in lags]
        
        # PACF
        pacf_values = []
        for lag in lags:
            model = np.polyfit(values[:-lag], values[lag:], 1)
            pacf_values.append(model[0])

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)'),
            vertical_spacing=0.15
        )

        # ACF plot
        fig.add_trace(
            go.Bar(
                x=lags,
                y=acf_values,
                name='ACF',
                marker_color=self.color_palette[0]
            ),
            row=1, col=1
        )

        # PACF plot
        fig.add_trace(
            go.Bar(
                x=lags,
                y=pacf_values,
                name='PACF',
                marker_color=self.color_palette[1]
            ),
            row=2, col=1
        )

        # Add confidence intervals
        ci = 1.96 / np.sqrt(len(values))
        for row in [1, 2]:
            fig.add_hline(y=ci, line_dash="dash", line_color="red",
                         annotation_text="95% CI", row=row, col=1)
            fig.add_hline(y=-ci, line_dash="dash", line_color="red",
                         row=row, col=1)

        # Update layout
        fig.update_layout(
            title='Correlation Analysis',
            showlegend=True,
            height=800,
            template=self.theme
        )

        return fig

    def create_anomaly_plot(self,
                           values: List[float],
                           timestamps: List[datetime],
                           pattern_results: Dict) -> go.Figure:
        """Create interactive anomaly detection plot"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Anomaly Detection', 'Local Outlier Factors'),
            vertical_spacing=0.15
        )

        # Main time series with anomalies
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                name='Original',
                line=dict(color='blue'),
                hovertemplate='%{y:.2f}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )

        if 'outliers' in pattern_results:
            outliers = pattern_results['outliers']['consensus_outliers']
            if outliers:
                outlier_x = [timestamps[i] for i in outliers]
                outlier_y = [values[i] for i in outliers]
                fig.add_trace(
                    go.Scatter(
                        x=outlier_x,
                        y=outlier_y,
                        mode='markers',
                        name='Anomalies',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='x'
                        ),
                        hovertemplate='%{y:.2f}<br>%{x}<extra>Anomaly</extra>'
                    ),
                    row=1, col=1
                )

        # Local Outlier Factors
        if 'outliers' in pattern_results:
            z_scores = stats.zscore(values)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=np.abs(z_scores),
                    name='|Z-Score|',
                    line=dict(color='orange'),
                    hovertemplate='|Z-Score|: %{y:.2f}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )

            # Add threshold line
            fig.add_hline(
                y=3,
                line_dash="dash",
                line_color="red",
                annotation_text="3σ Threshold",
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title='Anomaly Detection Analysis',
            showlegend=True,
            height=800,
            template=self.theme,
            hovermode='x unified'
        )

        return fig

    def create_pattern_plot(self,
                          values: List[float],
                          timestamps: List[datetime],
                          pattern_results: Dict) -> go.Figure:
        """Create interactive pattern analysis plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Pattern Recognition',
                'Change Points',
                'Volatility Clusters',
                'Pattern Similarity'
            )
        )

        # Pattern Recognition
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                name='Original',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        if 'repeated_patterns' in pattern_results:
            patterns = pattern_results['repeated_patterns']['patterns']
            for i, pattern in enumerate(patterns[:3]):  # Show top 3 patterns
                pattern_x = timestamps[pattern['window1_start']:pattern['window1_end']]
                pattern_y = values[pattern['window1_start']:pattern['window1_end']]
                fig.add_trace(
                    go.Scatter(
                        x=pattern_x,
                        y=pattern_y,
                        name=f'Pattern {i+1}',
                        line=dict(color=self.color_palette[i])
                    ),
                    row=1, col=1
                )

        # Change Points
        if 'change_points' in pattern_results:
            changes = pattern_results['change_points']['all_changes']
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    name='Data',
                    line=dict(color='blue')
                ),
                row=1, col=2
            )
            
            if changes:
                change_x = [timestamps[i] for i in changes]
                change_y = [values[i] for i in changes]
                fig.add_trace(
                    go.Scatter(
                        x=change_x,
                        y=change_y,
                        mode='markers',
                        name='Change Points',
                        marker=dict(color='red', size=10)
                    ),
                    row=1, col=2
                )

        # Volatility Clusters
        if 'volatility_clusters' in pattern_results:
            clusters = pattern_results['volatility_clusters']['clusters']
            volatility = pattern_results['volatility_clusters']['volatility_series']
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=volatility,
                    name='Volatility',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )

            if clusters:
                for cluster in clusters:
                    cluster_x = timestamps[cluster['start']:cluster['end']]
                    cluster_y = [cluster['avg_volatility']] * len(cluster_x)
                    fig.add_trace(
                        go.Scatter(
                            x=cluster_x,
                            y=cluster_y,
                            name=f'Cluster (vol={cluster["avg_volatility"]:.2f})',
                            line=dict(color='red', width=2)
                        ),
                        row=2, col=1
                    )

        # Pattern Similarity Matrix
        if 'repeated_patterns' in pattern_results:
            patterns = pattern_results['repeated_patterns']['patterns']
            similarity_matrix = np.zeros((len(patterns), len(patterns)))
            
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    similarity_matrix[i, j] = patterns[i]['similarity']
            
            fig.add_trace(
                go.Heatmap(
                    z=similarity_matrix,
                    colorscale='Viridis',
                    name='Similarity'
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title='Pattern Analysis',
            showlegend=True,
            height=1000,
            template=self.theme,
            hovermode='x unified'
        )

        return fig
