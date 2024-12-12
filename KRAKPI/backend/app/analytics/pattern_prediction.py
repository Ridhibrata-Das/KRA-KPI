import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Conv1D, GRU, Bidirectional
from tensorflow.keras.layers import Dropout, BatchNormalization, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class PatternPredictor:
    def __init__(self, sequence_length: int = 30, forecast_horizon: int = 7):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all prediction models"""
        # Deep Learning Models
        self.models['attention_lstm'] = self._build_attention_lstm()
        self.models['conv_lstm'] = self._build_conv_lstm()
        self.models['bidirectional_gru'] = self._build_bidirectional_gru()
        
        # Traditional Models
        self.models['prophet'] = None  # Will be initialized during training
        self.models['sarima'] = None   # Will be initialized during training
        self.models['xgboost'] = None  # Will be initialized during training

    def _build_attention_lstm(self) -> Model:
        """Build LSTM model with attention mechanism"""
        inputs = Input(shape=(self.sequence_length, 1))
        
        # Multi-head attention layer
        attention = MultiHeadAttention(num_heads=4, key_dim=8)(inputs, inputs)
        x = BatchNormalization()(attention)
        
        # LSTM layers
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(32, activation='relu')(x)
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model

    def _build_conv_lstm(self) -> Model:
        """Build Convolutional LSTM model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu',
                  input_shape=(self.sequence_length, 1)),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model

    def _build_bidirectional_gru(self) -> Model:
        """Build Bidirectional GRU model"""
        model = Sequential([
            Bidirectional(GRU(128, return_sequences=True),
                         input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            Bidirectional(GRU(64)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model

    def train_models(self,
                    values: List[float],
                    timestamps: List[datetime],
                    validation_split: float = 0.2) -> Dict:
        """Train all models on the provided data"""
        # Prepare data
        scaled_values = self.scalers['standard'].fit_transform(
            np.array(values).reshape(-1, 1)
        )
        X, y = self._prepare_sequences(scaled_values)
        
        # Train deep learning models
        training_results = {}
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        for model_name in ['attention_lstm', 'conv_lstm', 'bidirectional_gru']:
            history = self.models[model_name].fit(
                X, y,
                epochs=100,
                batch_size=32,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            training_results[model_name] = {
                'final_loss': float(history.history['loss'][-1]),
                'best_val_loss': float(min(history.history['val_loss'])),
                'epochs_trained': len(history.history['loss'])
            }
        
        # Train Prophet
        df = pd.DataFrame({
            'ds': timestamps,
            'y': values
        })
        self.models['prophet'] = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative'
        )
        self.models['prophet'].fit(df)
        
        # Train SARIMA
        try:
            self.models['sarima'] = SARIMAX(
                values,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12)
            ).fit(disp=False)
        except:
            self.models['sarima'] = None
        
        # Train XGBoost
        X_xgb, y_xgb = self._prepare_xgboost_data(scaled_values)
        self.models['xgboost'] = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100
        )
        self.models['xgboost'].fit(X_xgb, y_xgb)
        
        return training_results

    def predict_patterns(self,
                        values: List[float],
                        timestamps: List[datetime],
                        confidence_level: float = 0.95) -> Dict:
        """Generate predictions using all models with uncertainty estimates"""
        # Prepare data
        scaled_values = self.scalers['standard'].transform(
            np.array(values).reshape(-1, 1)
        )
        last_sequence = scaled_values[-self.sequence_length:]
        
        predictions = {}
        
        # Deep Learning predictions
        dl_models = ['attention_lstm', 'conv_lstm', 'bidirectional_gru']
        for model_name in dl_models:
            pred = self.models[model_name].predict(
                last_sequence.reshape(1, self.sequence_length, 1)
            )
            predictions[model_name] = {
                'values': self.scalers['standard'].inverse_transform(
                    pred.reshape(-1, 1)
                ).flatten().tolist(),
                'uncertainty': self._estimate_dl_uncertainty(
                    self.models[model_name],
                    last_sequence,
                    n_samples=100
                )
            }
        
        # Prophet predictions
        future_dates = pd.DataFrame({
            'ds': [timestamps[-1] + timedelta(days=i+1)
                  for i in range(self.forecast_horizon)]
        })
        prophet_forecast = self.models['prophet'].predict(future_dates)
        predictions['prophet'] = {
            'values': prophet_forecast['yhat'].values.tolist(),
            'uncertainty': {
                'lower': prophet_forecast['yhat_lower'].values.tolist(),
                'upper': prophet_forecast['yhat_upper'].values.tolist()
            }
        }
        
        # SARIMA predictions
        if self.models['sarima'] is not None:
            sarima_forecast = self.models['sarima'].forecast(self.forecast_horizon)
            sarima_conf = self.models['sarima'].get_forecast(
                self.forecast_horizon
            ).conf_int(alpha=1-confidence_level)
            predictions['sarima'] = {
                'values': sarima_forecast.values.tolist(),
                'uncertainty': {
                    'lower': sarima_conf.iloc[:, 0].values.tolist(),
                    'upper': sarima_conf.iloc[:, 1].values.tolist()
                }
            }
        
        # XGBoost predictions
        X_xgb = self._prepare_xgboost_prediction(scaled_values)
        xgb_pred = self.models['xgboost'].predict(X_xgb)
        predictions['xgboost'] = {
            'values': self.scalers['standard'].inverse_transform(
                xgb_pred.reshape(-1, 1)
            ).flatten().tolist(),
            'uncertainty': self._estimate_xgboost_uncertainty(
                self.models['xgboost'],
                X_xgb
            )
        }
        
        # Ensemble predictions
        predictions['ensemble'] = self._create_ensemble_prediction(predictions)
        
        return predictions

    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[(i + self.sequence_length):(i + self.sequence_length + self.forecast_horizon)])
        return np.array(X), np.array(y)

    def _prepare_xgboost_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for XGBoost training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i:(i + self.sequence_length)].flatten())
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def _prepare_xgboost_prediction(self, data: np.ndarray) -> np.ndarray:
        """Prepare data for XGBoost prediction"""
        return data[-self.sequence_length:].reshape(1, -1)

    def _estimate_dl_uncertainty(self,
                               model: Model,
                               sequence: np.ndarray,
                               n_samples: int = 100) -> Dict:
        """Estimate uncertainty for deep learning models using Monte Carlo Dropout"""
        predictions = []
        sequence = sequence.reshape(1, self.sequence_length, 1)
        
        for _ in range(n_samples):
            pred = model(sequence, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return {
            'lower': self.scalers['standard'].inverse_transform(
                (mean_pred - 1.96 * std_pred).reshape(-1, 1)
            ).flatten().tolist(),
            'upper': self.scalers['standard'].inverse_transform(
                (mean_pred + 1.96 * std_pred).reshape(-1, 1)
            ).flatten().tolist()
        }

    def _estimate_xgboost_uncertainty(self,
                                    model: xgb.XGBRegressor,
                                    X: np.ndarray) -> Dict:
        """Estimate uncertainty for XGBoost predictions"""
        # Use quantile regression for uncertainty estimation
        lower_model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.025
        )
        upper_model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.975
        )
        
        lower_model.fit(X, np.zeros(X.shape[0]))  # Dummy fit
        upper_model.fit(X, np.zeros(X.shape[0]))  # Dummy fit
        
        lower_pred = lower_model.predict(X)
        upper_pred = upper_model.predict(X)
        
        return {
            'lower': self.scalers['standard'].inverse_transform(
                lower_pred.reshape(-1, 1)
            ).flatten().tolist(),
            'upper': self.scalers['standard'].inverse_transform(
                upper_pred.reshape(-1, 1)
            ).flatten().tolist()
        }

    def _create_ensemble_prediction(self, predictions: Dict) -> Dict:
        """Create ensemble prediction by combining all model predictions"""
        model_predictions = []
        model_weights = {
            'attention_lstm': 0.25,
            'conv_lstm': 0.20,
            'bidirectional_gru': 0.20,
            'prophet': 0.15,
            'sarima': 0.10,
            'xgboost': 0.10
        }
        
        # Collect all valid predictions
        for model_name, weight in model_weights.items():
            if model_name in predictions:
                pred = np.array(predictions[model_name]['values'])
                model_predictions.append(pred * weight)
        
        # Calculate weighted average
        ensemble_pred = np.sum(model_predictions, axis=0)
        
        # Calculate uncertainty
        all_lower = []
        all_upper = []
        for model_name in predictions:
            if 'uncertainty' in predictions[model_name]:
                all_lower.append(predictions[model_name]['uncertainty']['lower'])
                all_upper.append(predictions[model_name]['uncertainty']['upper'])
        
        # Combine uncertainties
        lower = np.min(all_lower, axis=0)
        upper = np.max(all_upper, axis=0)
        
        return {
            'values': ensemble_pred.tolist(),
            'uncertainty': {
                'lower': lower.tolist(),
                'upper': upper.tolist()
            }
        }

    def analyze_prediction_accuracy(self,
                                  true_values: List[float],
                                  predictions: Dict) -> Dict:
        """Analyze prediction accuracy for all models"""
        accuracy_metrics = {}
        
        for model_name, pred_data in predictions.items():
            pred_values = np.array(pred_data['values'])
            true = np.array(true_values)
            
            # Calculate metrics
            mse = np.mean((true - pred_values) ** 2)
            mae = np.mean(np.abs(true - pred_values))
            mape = np.mean(np.abs((true - pred_values) / true)) * 100
            
            # Calculate prediction intervals accuracy
            if 'uncertainty' in pred_data:
                lower = np.array(pred_data['uncertainty']['lower'])
                upper = np.array(pred_data['uncertainty']['upper'])
                coverage = np.mean((true >= lower) & (true <= upper)) * 100
            else:
                coverage = None
            
            accuracy_metrics[model_name] = {
                'mse': float(mse),
                'mae': float(mae),
                'mape': float(mape),
                'interval_coverage': coverage
            }
        
        return accuracy_metrics
