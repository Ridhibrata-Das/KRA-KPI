import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime, timedelta

class AdvancedKPIPredictor:
    def __init__(self, model_type: str = 'prophet'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        if model_type == 'prophet':
            self.model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                seasonality_mode='multiplicative'
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1
            )
        elif model_type == 'lstm':
            self.model = self._build_lstm_model()
    
    def _build_lstm_model(self, sequence_length: int = 10):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _prepare_lstm_data(self, values: List[float], sequence_length: int = 10):
        X, y = [], []
        for i in range(len(values) - sequence_length):
            X.append(values[i:(i + sequence_length)])
            y.append(values[i + sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, dates: List[datetime], values: List[float]):
        """Train the selected forecasting model"""
        if self.model_type == 'prophet':
            df = pd.DataFrame({
                'ds': dates,
                'y': values
            })
            self.model.fit(df)
        
        elif self.model_type == 'xgboost':
            X = self._create_features(dates)
            y = np.array(values)
            self.model.fit(X, y)
        
        elif self.model_type == 'lstm':
            scaled_values = self.scaler.fit_transform(np.array(values).reshape(-1, 1))
            X, y = self._prepare_lstm_data(scaled_values.flatten())
            X = X.reshape((X.shape[0], X.shape[1], 1))
            self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    def _create_features(self, dates: List[datetime]) -> np.ndarray:
        """Create time-based features for ML models"""
        features = []
        for date in dates:
            features.append([
                date.timestamp(),
                date.month,
                date.day,
                date.weekday(),
                date.hour if hasattr(date, 'hour') else 0,
                date.minute if hasattr(date, 'minute') else 0,
                np.sin(2 * np.pi * date.month / 12),  # Yearly seasonality
                np.cos(2 * np.pi * date.month / 12),
                np.sin(2 * np.pi * date.weekday() / 7),  # Weekly seasonality
                np.cos(2 * np.pi * date.weekday() / 7),
            ])
        return np.array(features)
    
    def predict(self, future_dates: List[datetime]) -> Dict:
        """Generate predictions with uncertainty estimates"""
        if self.model_type == 'prophet':
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = self.model.predict(future_df)
            return {
                'predictions': forecast['yhat'].values,
                'lower_bound': forecast['yhat_lower'].values,
                'upper_bound': forecast['yhat_upper'].values
            }
        
        elif self.model_type == 'xgboost':
            X_future = self._create_features(future_dates)
            predictions = self.model.predict(X_future)
            
            # Generate prediction intervals using quantile regression
            quantiles = []
            for alpha in [0.1, 0.9]:
                self.model.set_params(objective=f'reg:quantileerror@{alpha}')
                quantiles.append(self.model.predict(X_future))
            
            return {
                'predictions': predictions,
                'lower_bound': quantiles[0],
                'upper_bound': quantiles[1]
            }
        
        elif self.model_type == 'lstm':
            # Use last sequence_length values for initial prediction
            sequence_length = 10
            last_sequence = self.scaler.transform(
                np.array(values[-sequence_length:]).reshape(-1, 1)
            )
            
            predictions = []
            for _ in range(len(future_dates)):
                X = last_sequence[-sequence_length:].reshape(1, sequence_length, 1)
                pred = self.model.predict(X)[0]
                predictions.append(pred)
                last_sequence = np.append(last_sequence[1:], pred)
            
            predictions = self.scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            )
            
            # Simple uncertainty estimation using historical std
            std = np.std(values)
            return {
                'predictions': predictions.flatten(),
                'lower_bound': predictions.flatten() - 2 * std,
                'upper_bound': predictions.flatten() + 2 * std
            }

class EnsemblePredictor:
    def __init__(self):
        self.models = {
            'prophet': AdvancedKPIPredictor('prophet'),
            'xgboost': AdvancedKPIPredictor('xgboost'),
            'lstm': AdvancedKPIPredictor('lstm')
        }
    
    def train(self, dates: List[datetime], values: List[float]):
        """Train all models in the ensemble"""
        for model in self.models.values():
            model.train(dates, values)
    
    def predict(self, future_dates: List[datetime]) -> Dict:
        """Generate ensemble predictions with uncertainty estimates"""
        all_predictions = []
        all_lower_bounds = []
        all_upper_bounds = []
        
        for model in self.models.values():
            preds = model.predict(future_dates)
            all_predictions.append(preds['predictions'])
            all_lower_bounds.append(preds['lower_bound'])
            all_upper_bounds.append(preds['upper_bound'])
        
        # Ensemble predictions using weighted average
        weights = [0.4, 0.3, 0.3]  # Prophet, XGBoost, LSTM
        ensemble_predictions = np.average(all_predictions, axis=0, weights=weights)
        ensemble_lower = np.average(all_lower_bounds, axis=0, weights=weights)
        ensemble_upper = np.average(all_upper_bounds, axis=0, weights=weights)
        
        return {
            'predictions': ensemble_predictions,
            'lower_bound': ensemble_lower,
            'upper_bound': ensemble_upper,
            'individual_predictions': {
                'prophet': all_predictions[0],
                'xgboost': all_predictions[1],
                'lstm': all_predictions[2]
            }
        }
