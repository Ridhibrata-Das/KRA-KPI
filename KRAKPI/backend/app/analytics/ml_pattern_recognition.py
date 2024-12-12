import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Conv1D, MaxPooling1D, UpSampling1D, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from scipy.signal import find_peaks
import tensorflow as tf
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
import warnings
warnings.filterwarnings('ignore')

class MLPatternRecognizer:
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all ML models"""
        # Autoencoder for pattern extraction
        self.models['autoencoder'] = self._build_autoencoder()
        
        # LSTM for sequence prediction
        self.models['lstm'] = self._build_lstm()
        
        # CNN for pattern classification
        self.models['cnn'] = self._build_cnn()

    def _build_autoencoder(self) -> Model:
        """Build convolutional autoencoder for pattern extraction"""
        input_layer = Input(shape=(self.sequence_length, 1))
        
        # Encoder
        x = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(16, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        
        # Decoder
        x = Conv1D(16, 3, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(32, 3, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        decoded = Conv1D(1, 3, activation='linear', padding='same')(x)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder

    def _build_lstm(self) -> Model:
        """Build LSTM model for sequence prediction"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, 1)),
            LSTM(50, return_sequences=False),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model

    def _build_cnn(self) -> Model:
        """Build CNN for pattern classification"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                  input_shape=(self.sequence_length, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def detect_patterns(self, 
                       values: List[float], 
                       timestamps: List[datetime]) -> Dict:
        """Detect patterns using multiple ML techniques"""
        # Prepare data
        scaled_values = self.scaler.fit_transform(np.array(values).reshape(-1, 1))
        sequences = self._create_sequences(scaled_values)
        
        return {
            'learned_patterns': self._detect_learned_patterns(sequences),
            'temporal_patterns': self._detect_temporal_patterns(sequences),
            'anomalous_patterns': self._detect_anomalous_patterns(sequences),
            'recurring_patterns': self._detect_recurring_patterns(sequences),
            'trend_patterns': self._detect_trend_patterns(scaled_values, timestamps),
            'pattern_summary': self._generate_pattern_summary(sequences)
        }

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for pattern detection"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequence = data[i:(i + self.sequence_length)]
            sequences.append(sequence)
        return np.array(sequences)

    def _detect_learned_patterns(self, sequences: np.ndarray) -> Dict:
        """Detect patterns using autoencoder"""
        # Train autoencoder
        self.models['autoencoder'].fit(
            sequences, sequences,
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Get reconstructions
        reconstructions = self.models['autoencoder'].predict(sequences)
        reconstruction_errors = np.mean(np.square(sequences - reconstructions), axis=(1,2))
        
        # Find significant patterns
        threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
        pattern_indices = np.where(reconstruction_errors > threshold)[0]
        
        return {
            'pattern_indices': pattern_indices.tolist(),
            'reconstruction_errors': reconstruction_errors.tolist(),
            'significant_patterns': [
                {
                    'start_idx': int(idx),
                    'end_idx': int(idx + self.sequence_length),
                    'error': float(reconstruction_errors[idx])
                }
                for idx in pattern_indices
            ]
        }

    def _detect_temporal_patterns(self, sequences: np.ndarray) -> Dict:
        """Detect temporal patterns using LSTM"""
        # Prepare data for LSTM
        X = sequences[:-1]
        y = sequences[1:, -1, 0]
        
        # Train LSTM
        self.models['lstm'].fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Get predictions
        predictions = self.models['lstm'].predict(X)
        prediction_errors = np.abs(predictions - y)
        
        # Find significant temporal patterns
        threshold = np.mean(prediction_errors) + 2 * np.std(prediction_errors)
        pattern_indices = np.where(prediction_errors > threshold)[0]
        
        return {
            'pattern_indices': pattern_indices.tolist(),
            'prediction_errors': prediction_errors.tolist(),
            'temporal_patterns': [
                {
                    'start_idx': int(idx),
                    'end_idx': int(idx + self.sequence_length),
                    'error': float(prediction_errors[idx])
                }
                for idx in pattern_indices
            ]
        }

    def _detect_anomalous_patterns(self, sequences: np.ndarray) -> Dict:
        """Detect anomalous patterns using Isolation Forest"""
        # Flatten sequences for Isolation Forest
        flattened_sequences = sequences.reshape(sequences.shape[0], -1)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(flattened_sequences)
        
        # Find anomalous patterns
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        
        return {
            'anomaly_indices': anomaly_indices.tolist(),
            'anomalous_patterns': [
                {
                    'start_idx': int(idx),
                    'end_idx': int(idx + self.sequence_length),
                    'score': float(iso_forest.score_samples([flattened_sequences[idx]])[0])
                }
                for idx in anomaly_indices
            ]
        }

    def _detect_recurring_patterns(self, sequences: np.ndarray) -> Dict:
        """Detect recurring patterns using Time Series K-Means"""
        # Use DTW-based k-means
        n_clusters = min(5, len(sequences) // 10)  # Adaptive number of clusters
        ts_kmeans = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",
            random_state=42
        )
        cluster_labels = ts_kmeans.fit_predict(sequences)
        
        # Analyze clusters
        clusters = []
        for i in range(n_clusters):
            cluster_sequences = sequences[cluster_labels == i]
            if len(cluster_sequences) > 0:
                centroid = ts_kmeans.cluster_centers_[i]
                distances = [dtw(seq.flatten(), centroid.flatten()) 
                           for seq in cluster_sequences]
                
                clusters.append({
                    'cluster_id': int(i),
                    'size': int(len(cluster_sequences)),
                    'avg_distance': float(np.mean(distances)),
                    'pattern_indices': np.where(cluster_labels == i)[0].tolist()
                })
        
        return {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'clusters': clusters
        }

    def _detect_trend_patterns(self, 
                             values: np.ndarray, 
                             timestamps: List[datetime]) -> Dict:
        """Detect trend patterns using CNN"""
        # Prepare sequences for trend detection
        sequences = self._create_sequences(values)
        
        # Calculate trend labels (1 for upward trend, 0 for downward)
        trend_labels = np.array([
            1 if np.polyfit(range(len(seq)), seq.flatten(), 1)[0] > 0 else 0
            for seq in sequences
        ])
        
        # Train CNN
        self.models['cnn'].fit(
            sequences, trend_labels,
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Get trend predictions
        trend_probs = self.models['cnn'].predict(sequences)
        
        # Find significant trend changes
        trend_changes = np.where(np.abs(np.diff(trend_probs.flatten())) > 0.5)[0]
        
        return {
            'trend_probabilities': trend_probs.flatten().tolist(),
            'trend_changes': trend_changes.tolist(),
            'trend_patterns': [
                {
                    'start_idx': int(idx),
                    'end_idx': int(idx + self.sequence_length),
                    'trend_probability': float(trend_probs[idx][0])
                }
                for idx in trend_changes
            ]
        }

    def _generate_pattern_summary(self, sequences: np.ndarray) -> Dict:
        """Generate a comprehensive summary of detected patterns"""
        # Calculate pattern complexity
        pca = PCA()
        flattened_sequences = sequences.reshape(sequences.shape[0], -1)
        pca.fit(flattened_sequences)
        complexity = sum(np.cumsum(pca.explained_variance_ratio_) < 0.95)
        
        # Calculate pattern diversity
        kmeans = KMeans(n_clusters=min(10, len(sequences) // 5))
        cluster_labels = kmeans.fit_predict(flattened_sequences)
        diversity = len(np.unique(cluster_labels))
        
        # Calculate pattern stability
        stability = 1.0 - np.mean(np.std(sequences, axis=0))
        
        return {
            'complexity': int(complexity),
            'diversity': int(diversity),
            'stability': float(stability),
            'pattern_types': {
                'learned': len(self._detect_learned_patterns(sequences)['pattern_indices']),
                'temporal': len(self._detect_temporal_patterns(sequences)['pattern_indices']),
                'anomalous': len(self._detect_anomalous_patterns(sequences)['anomaly_indices']),
                'recurring': len(self._detect_recurring_patterns(sequences)['clusters'])
            }
        }

    def analyze_pattern_significance(self, 
                                   pattern_results: Dict,
                                   confidence_level: float = 0.95) -> Dict:
        """Analyze the statistical significance of detected patterns"""
        significant_patterns = {}
        
        # Analyze learned patterns
        learned = pattern_results['learned_patterns']['significant_patterns']
        significant_learned = [
            p for p in learned
            if p['error'] > np.quantile(
                pattern_results['learned_patterns']['reconstruction_errors'],
                confidence_level
            )
        ]
        
        # Analyze temporal patterns
        temporal = pattern_results['temporal_patterns']['temporal_patterns']
        significant_temporal = [
            p for p in temporal
            if p['error'] > np.quantile(
                pattern_results['temporal_patterns']['prediction_errors'],
                confidence_level
            )
        ]
        
        # Analyze anomalous patterns
        anomalous = pattern_results['anomalous_patterns']['anomalous_patterns']
        significant_anomalous = [
            p for p in anomalous
            if p['score'] < np.quantile(
                [p['score'] for p in anomalous],
                1 - confidence_level
            )
        ]
        
        return {
            'significant_learned': significant_learned,
            'significant_temporal': significant_temporal,
            'significant_anomalous': significant_anomalous,
            'confidence_level': confidence_level,
            'total_significant': len(significant_learned) + 
                               len(significant_temporal) + 
                               len(significant_anomalous)
        }

    def get_pattern_recommendations(self, pattern_results: Dict) -> List[Dict]:
        """Generate recommendations based on pattern analysis"""
        recommendations = []
        
        # Check pattern complexity
        complexity = pattern_results['pattern_summary']['complexity']
        if complexity > 5:
            recommendations.append({
                'type': 'complexity',
                'message': 'High pattern complexity detected. Consider decomposing the KPI into sub-components.',
                'priority': 'high'
            })
        
        # Check pattern diversity
        diversity = pattern_results['pattern_summary']['diversity']
        if diversity > 7:
            recommendations.append({
                'type': 'diversity',
                'message': 'High pattern diversity observed. Consider implementing pattern-specific monitoring.',
                'priority': 'medium'
            })
        
        # Check stability
        stability = pattern_results['pattern_summary']['stability']
        if stability < 0.5:
            recommendations.append({
                'type': 'stability',
                'message': 'Low pattern stability detected. Consider implementing more robust thresholds.',
                'priority': 'high'
            })
        
        return recommendations
