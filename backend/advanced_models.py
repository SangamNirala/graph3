"""
Advanced Time Series Forecasting Models
State-of-the-art ML models optimized for CPU deployment with balanced performance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
import xgboost as xgb
import optuna
import joblib
from tqdm import tqdm
import warnings
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DLinearModel(nn.Module):
    """
    DLinear: Simple yet effective linear model for time series forecasting
    Decomposition + Linear layers approach
    """
    
    def __init__(self, seq_len: int, pred_len: int, individual: bool = False):
        super(DLinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        
        # Linear layers for trend and seasonal components
        if individual:
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(1)
            ])
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(1)
            ])
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights to prevent NaN losses"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
            
    def forward(self, x):
        # Decomposition
        seasonal_init, trend_init = self.decompose(x)
        
        if self.individual:
            seasonal_output = self.Linear_Seasonal[0](seasonal_init)
            trend_output = self.Linear_Trend[0](trend_init)
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            
        return seasonal_output + trend_output
    
    def decompose(self, x):
        """Simple moving average decomposition"""
        # Moving average for trend
        kernel_size = 25
        if x.shape[-1] < kernel_size:
            kernel_size = x.shape[-1] // 2 + 1
            
        # Simple moving average
        trend = torch.nn.functional.avg_pool1d(
            x.transpose(-1, -2), 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2
        ).transpose(-1, -2)
        
        # Ensure same shape
        if trend.shape[-1] != x.shape[-1]:
            trend = torch.nn.functional.interpolate(
                trend.transpose(-1, -2), 
                size=x.shape[-1], 
                mode='linear', 
                align_corners=False
            ).transpose(-1, -2)
        
        seasonal = x - trend
        return seasonal, trend


class NBeatsBlock(nn.Module):
    """N-BEATS block implementation"""
    
    def __init__(self, input_size: int, output_size: int, theta_size: int, 
                 basis_function: str = 'generic', layers: int = 4, layer_size: int = 512):
        super(NBeatsBlock, self).__init__()
        self.layers = layers
        self.layer_size = layer_size
        self.basis_function = basis_function
        self.input_size = input_size
        self.output_size = output_size
        self.theta_size = theta_size
        
        # FC layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else layer_size, layer_size)
            for i in range(layers)
        ])
        
        # Theta layer
        self.theta_layer = nn.Linear(layer_size, theta_size)
        
        # Basis functions
        if basis_function == 'trend':
            self.basis = self.trend_basis
        elif basis_function == 'seasonal':
            self.basis = self.seasonal_basis
        else:
            self.basis = self.generic_basis
            
    def forward(self, x):
        # FC stack
        for layer in self.fc_layers:
            x = torch.relu(layer(x))
            
        # Theta
        theta = self.theta_layer(x)
        
        # Basis function
        backcast, forecast = self.basis(theta)
        
        return backcast, forecast
    
    def generic_basis(self, theta):
        """Generic basis function"""
        backcast = theta[:, :self.input_size]
        forecast = theta[:, self.input_size:self.input_size + self.output_size]
        return backcast, forecast
    
    def trend_basis(self, theta):
        """Trend basis function"""
        t_b = torch.arange(self.input_size).float().to(theta.device)
        t_f = torch.arange(self.input_size, self.input_size + self.output_size).float().to(theta.device)
        
        # Polynomial basis
        T_b = torch.stack([t_b ** i for i in range(self.theta_size)], dim=0).T
        T_f = torch.stack([t_f ** i for i in range(self.theta_size)], dim=0).T
        
        backcast = torch.matmul(T_b, theta.T).T
        forecast = torch.matmul(T_f, theta.T).T
        
        return backcast, forecast
    
    def seasonal_basis(self, theta):
        """Seasonal basis function"""
        t_b = torch.arange(self.input_size).float().to(theta.device)
        t_f = torch.arange(self.input_size, self.input_size + self.output_size).float().to(theta.device)
        
        # Fourier basis
        T_b = torch.stack([torch.cos(2 * np.pi * i * t_b / self.input_size) 
                          for i in range(self.theta_size // 2)], dim=0).T
        T_b = torch.cat([T_b, torch.stack([torch.sin(2 * np.pi * i * t_b / self.input_size) 
                                          for i in range(self.theta_size // 2)], dim=0).T], dim=1)
        
        T_f = torch.stack([torch.cos(2 * np.pi * i * t_f / self.input_size) 
                          for i in range(self.theta_size // 2)], dim=0).T
        T_f = torch.cat([T_f, torch.stack([torch.sin(2 * np.pi * i * t_f / self.input_size) 
                                          for i in range(self.theta_size // 2)], dim=0).T], dim=1)
        
        backcast = torch.matmul(T_b, theta.T).T
        forecast = torch.matmul(T_f, theta.T).T
        
        return backcast, forecast


class NBeatsModel(nn.Module):
    """N-BEATS model implementation"""
    
    def __init__(self, input_size: int, output_size: int, stacks: int = 2, 
                 layers: int = 4, layer_size: int = 512):
        super(NBeatsModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Stacks
        self.stacks = nn.ModuleList([
            NBeatsBlock(input_size, output_size, 
                       theta_size=layer_size//4, 
                       basis_function='trend' if i == 0 else 'seasonal' if i == 1 else 'generic',
                       layers=layers, layer_size=layer_size)
            for i in range(stacks)
        ])
        
    def forward(self, x):
        residual = x
        forecast = 0
        
        for stack in self.stacks:
            backcast, f = stack(residual)
            residual = residual - backcast
            forecast = forecast + f
            
        return forecast


class LightweightLSTM(nn.Module):
    """Lightweight LSTM for CPU optimization"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(LightweightLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class AdvancedTimeSeriesForecaster:
    """
    Advanced Time Series Forecasting class with multiple state-of-the-art models
    Optimized for CPU deployment with balanced performance
    """
    
    def __init__(self, seq_len: int = 50, pred_len: int = 30, model_type: str = 'dlinear'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.training_history = []
        
        # Performance metrics
        self.metrics = {
            'rmse': None,
            'mae': None,
            'r2': None,
            'mape': None
        }
        
    def create_sequences(self, data: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for supervised learning"""
        X, y = [], []
        
        # Check if we have enough data
        min_required_length = seq_len + pred_len
        if len(data) < min_required_length:
            raise ValueError(f"Dataset too small: {len(data)} samples, need at least {min_required_length} samples for seq_len={seq_len} and pred_len={pred_len}")
        
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:(i + seq_len)])
            y.append(data[i + seq_len:i + seq_len + pred_len])
            
        return np.array(X), np.array(y)
    
    def prepare_data(self, data: pd.DataFrame, time_col: str, target_col: str) -> Dict[str, Any]:
        """Enhanced data preparation with feature engineering"""
        # Sort by time and handle missing values
        data = data.sort_values(time_col)
        data = data.dropna()
        
        # Extract target values
        target_values = data[target_col].values
        
        # Normalize data
        target_values_scaled = self.scaler.fit_transform(target_values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.create_sequences(target_values_scaled, self.seq_len, self.pred_len)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'original_data': target_values,
            'scaled_data': target_values_scaled
        }
    
    def build_model(self) -> nn.Module:
        """Build the specified model"""
        if self.model_type == 'dlinear':
            return DLinearModel(self.seq_len, self.pred_len)
        elif self.model_type == 'nbeats':
            return NBeatsModel(self.seq_len, self.pred_len)
        elif self.model_type == 'lstm':
            return LightweightLSTM(1, 64, 2, self.pred_len)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_pytorch_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray, 
                           epochs: int = 100, batch_size: int = 32, lr: float = 0.001) -> Dict[str, Any]:
        """Train PyTorch model with early stopping"""
        
        # Reshape for LSTM if needed
        if self.model_type == 'lstm':
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = self.build_model()
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                if self.model_type == 'lstm':
                    output = self.model(batch_x)
                else:
                    output = self.model(batch_x)
                
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    if self.model_type == 'lstm':
                        output = self.model(batch_x)
                    else:
                        output = self.model(batch_x)
                    
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model with unique filename
                model_path = f'/tmp/best_model_{self.model_type}_{id(self)}.pth'
                torch.save(self.model.state_dict(), model_path)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Load best model with error handling
        try:
            model_path = f'/tmp/best_model_{self.model_type}_{id(self)}.pth'
            if os.path.exists(model_path):
                state_dict = torch.load(model_path)
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"Warning: Missing keys in state_dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                    
                self.fitted = True
            else:
                print(f"Model file {model_path} not found, using current model state")
                self.fitted = True
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Using current model state instead of best model")
            self.fitted = True
        
        return {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'epochs_trained': epoch + 1
        }
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray, 
                               X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train gradient boosting models (LightGBM, XGBoost)"""
        
        # Reshape data for gradient boosting
        X_train_gb = X_train.reshape(X_train.shape[0], -1)
        X_test_gb = X_test.reshape(X_test.shape[0], -1)
        
        # For multi-step prediction, we need to flatten y for gradient boosting
        # We'll use the first step of multi-step prediction as target
        if len(y_train.shape) > 1:
            y_train_gb = y_train[:, 0]  # Use first step as target
            y_test_gb = y_test[:, 0]
        else:
            y_train_gb = y_train
            y_test_gb = y_test
        
        models = {}
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_gb, y_train_gb)
        models['lightgbm'] = lgb_model
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train_gb, y_train_gb)
        models['xgboost'] = xgb_model
        
        # Ensemble
        lgb_pred = lgb_model.predict(X_test_gb)
        xgb_pred = xgb_model.predict(X_test_gb)
        ensemble_pred = (lgb_pred + xgb_pred) / 2
        
        self.model = models
        self.fitted = True
        
        return {
            'models': models,
            'ensemble_prediction': ensemble_pred
        }
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.fitted:
            raise ValueError("Model must be trained first")
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Calculate metrics
        y_test_flat = y_test.flatten()
        pred_flat = predictions.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y_test_flat) | np.isnan(pred_flat))
        y_test_clean = y_test_flat[valid_mask]
        pred_clean = pred_flat[valid_mask]
        
        if len(y_test_clean) == 0:
            return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf'), 'mape': float('inf')}
        
        rmse = np.sqrt(mean_squared_error(y_test_clean, pred_clean))
        mae = mean_absolute_error(y_test_clean, pred_clean)
        r2 = r2_score(y_test_clean, pred_clean)
        
        # MAPE calculation
        mape = np.mean(np.abs((y_test_clean - pred_clean) / (y_test_clean + 1e-8))) * 100
        
        self.metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model must be trained first")
        
        if isinstance(self.model, dict):  # Gradient boosting ensemble
            lgb_pred = self.model['lightgbm'].predict(X.reshape(X.shape[0], -1))
            xgb_pred = self.model['xgboost'].predict(X.reshape(X.shape[0], -1))
            predictions = (lgb_pred + xgb_pred) / 2
        else:  # PyTorch model
            self.model.eval()
            
            with torch.no_grad():
                if self.model_type == 'lstm':
                    X_tensor = torch.FloatTensor(X.reshape(X.shape[0], X.shape[1], 1))
                else:
                    X_tensor = torch.FloatTensor(X)
                
                predictions = self.model(X_tensor).numpy()
        
        return predictions
    
    def predict_next_steps(self, last_sequence: np.ndarray, steps: int = 30) -> np.ndarray:
        """
        Predict next steps using the model with improved pattern-based prediction
        and bias correction to prevent downward drift
        """
        if not self.fitted:
            raise ValueError("Model must be trained first")
        
        # Ensure we have enough data
        if len(last_sequence) < self.seq_len:
            # Pad with the last value
            padding = np.full(self.seq_len - len(last_sequence), last_sequence[-1])
            last_sequence = np.concatenate([padding, last_sequence])
        
        # Take the last seq_len points
        input_sequence = last_sequence[-self.seq_len:]
        
        # Analyze historical patterns for bias correction
        historical_mean = np.mean(last_sequence)
        historical_std = np.std(last_sequence)
        recent_trend = np.polyfit(np.arange(len(input_sequence)), input_sequence, 1)[0]
        
        # Calculate local statistics for stabilization
        local_mean = np.mean(input_sequence)
        local_std = np.std(input_sequence)
        
        # Scale input
        input_scaled = self.scaler.transform(input_sequence.reshape(-1, 1)).flatten()
        
        predictions = []
        current_input = input_scaled.copy()
        original_input = input_sequence.copy()  # Keep original for pattern analysis
        
        # Multi-step prediction with error correction
        for step in range(steps):
            # Prepare input for prediction
            if isinstance(self.model, dict):  # Gradient boosting
                pred_input = current_input.reshape(1, -1)
                pred = self.model['lightgbm'].predict(pred_input)[0]
                if len(pred) > 0:
                    next_pred = pred[0] if hasattr(pred, '__len__') else pred
                else:
                    next_pred = current_input[-1]
            else:  # PyTorch model
                self.model.eval()
                with torch.no_grad():
                    if self.model_type == 'lstm':
                        pred_input = torch.FloatTensor(current_input.reshape(1, -1, 1))
                    else:
                        pred_input = torch.FloatTensor(current_input.reshape(1, -1))
                    
                    pred = self.model(pred_input).numpy()
                    next_pred = pred[0, 0] if pred.shape[1] > 0 else current_input[-1]
            
            # Apply bias correction and stabilization
            next_pred = self._apply_bias_correction(
                next_pred, step, historical_mean, historical_std, 
                recent_trend, local_mean, local_std, original_input
            )
            
            predictions.append(next_pred)
            
            # Update input sequence with controlled feedback
            # Mix predicted value with pattern-based estimate to reduce error accumulation
            pattern_estimate = self._estimate_pattern_based_value(
                original_input, step, recent_trend, local_mean
            )
            
            # Weighted combination to reduce feedback loop bias
            feedback_weight = max(0.3, 1.0 - (step * 0.02))  # Reduce weight as we go further
            pattern_weight = 1.0 - feedback_weight
            
            combined_pred = feedback_weight * next_pred + pattern_weight * pattern_estimate
            current_input = np.append(current_input[1:], combined_pred)
        
        # Inverse transform predictions
        predictions_array = np.array(predictions)
        predictions_unscaled = self.scaler.inverse_transform(predictions_array.reshape(-1, 1)).flatten()
        
        # Apply final smoothing to maintain historical patterns
        predictions_unscaled = self._apply_final_smoothing(
            predictions_unscaled, last_sequence, historical_mean, recent_trend
        )
        
        return predictions_unscaled
    
    def _apply_bias_correction(self, prediction, step, historical_mean, historical_std, 
                              recent_trend, local_mean, local_std, original_input):
        """Apply enhanced bias correction to prevent accumulated drift and maintain historical patterns"""
        # Multi-level mean reversion with historical context
        global_mean_reversion = (historical_mean - prediction) * 0.02 * (1 + step * 0.003)
        local_mean_reversion = (local_mean - prediction) * 0.05 * (1 + step * 0.002)
        
        # Enhanced trend continuation with adaptive decay based on trend strength
        trend_strength = abs(recent_trend) / (historical_std + 1e-8)
        trend_decay = 0.97 ** step if trend_strength > 0.1 else 0.93 ** step
        trend_component = recent_trend * trend_decay
        
        # Multi-level volatility constraint with historical context
        historical_range = np.max(original_input) - np.min(original_input)
        recent_range = np.max(original_input[-min(10, len(original_input)):]) - np.min(original_input[-min(10, len(original_input)):])
        
        # Adaptive volatility bounds based on recent vs historical patterns
        volatility_bound = min(2.0 * local_std, 0.3 * historical_range)
        if abs(prediction - local_mean) > volatility_bound:
            volatility_correction = np.sign(local_mean - prediction) * volatility_bound * 0.4
        else:
            volatility_correction = 0
        
        # Enhanced pattern-based correction with multi-scale analysis
        pattern_correction = self._calculate_enhanced_pattern_correction(
            prediction, step, original_input, local_mean, historical_mean
        )
        
        # Advanced momentum correction with acceleration awareness
        momentum_correction = self._calculate_advanced_momentum_correction(
            prediction, step, original_input, recent_trend, historical_std
        )
        
        # Range preservation correction - ensure predictions stay within realistic bounds
        range_correction = self._calculate_range_preservation_correction(
            prediction, step, original_input, historical_mean, historical_std
        )
        
        # Pattern consistency correction - maintain identified patterns
        consistency_correction = self._calculate_pattern_consistency_correction(
            prediction, step, original_input, recent_trend
        )
        
        corrected_prediction = (prediction + 
                              global_mean_reversion + local_mean_reversion + 
                              trend_component + volatility_correction + 
                              pattern_correction + momentum_correction + 
                              range_correction + consistency_correction)
        
        return corrected_prediction
    
    def _calculate_enhanced_pattern_correction(self, prediction, step, original_input, 
                                             local_mean, historical_mean):
        """Calculate enhanced pattern-based correction with multi-scale analysis"""
        pattern_corrections = []
        
        # Short-term pattern correction (last 3-5 values)
        if len(original_input) >= 4:
            short_pattern = original_input[-4:]
            short_changes = np.diff(short_pattern)
            if len(short_changes) > 0:
                expected_change = np.mean(short_changes)
                expected_next = original_input[-1] + expected_change
                short_correction = (expected_next - prediction) * 0.4
                pattern_corrections.append(short_correction)
        
        # Medium-term pattern correction (last 6-10 values)
        if len(original_input) >= 8:
            medium_pattern = original_input[-8:]
            medium_changes = np.diff(medium_pattern)
            if len(medium_changes) > 0:
                # Use weighted average of changes (more weight to recent)
                weights = np.exp(np.linspace(-1, 0, len(medium_changes)))
                weights /= np.sum(weights)
                weighted_change = np.sum(medium_changes * weights)
                expected_next = original_input[-1] + weighted_change
                medium_correction = (expected_next - prediction) * 0.3
                pattern_corrections.append(medium_correction)
        
        # Long-term pattern correction (overall trend)
        if len(original_input) >= 12:
            long_trend = np.polyfit(np.arange(len(original_input)), original_input, 1)[0]
            expected_next = original_input[-1] + long_trend
            long_correction = (expected_next - prediction) * 0.2
            pattern_corrections.append(long_correction)
        
        # Combine pattern corrections
        if pattern_corrections:
            return np.mean(pattern_corrections)
        else:
            return 0.0
    
    def _calculate_advanced_momentum_correction(self, prediction, step, original_input, 
                                              recent_trend, historical_std):
        """Calculate advanced momentum correction with acceleration awareness"""
        if len(original_input) >= 4:
            # Calculate multi-level momentum
            recent_changes = np.diff(original_input[-4:])
            recent_momentum = np.mean(recent_changes)
            
            # Calculate acceleration (change in momentum)
            if len(recent_changes) >= 2:
                acceleration = np.mean(np.diff(recent_changes))
            else:
                acceleration = 0
            
            # Adaptive momentum correction based on trend consistency
            momentum_strength = abs(recent_momentum) / (historical_std + 1e-8)
            
            # Strong momentum should be preserved
            if momentum_strength > 0.2:
                momentum_correction = recent_momentum * 0.5 * (1 / (1 + step * 0.05))
                
                # Add acceleration component for smooth transitions
                acceleration_correction = acceleration * 0.2 * (1 / (1 + step * 0.1))
                
                return momentum_correction + acceleration_correction
            else:
                return 0.0
        
        return 0.0
    
    def _calculate_range_preservation_correction(self, prediction, step, original_input, 
                                               historical_mean, historical_std):
        """Calculate correction to keep predictions within realistic historical ranges"""
        # Calculate historical percentiles for realistic bounds
        historical_min = np.percentile(original_input, 5)   # 5th percentile
        historical_max = np.percentile(original_input, 95)  # 95th percentile
        historical_range = historical_max - historical_min
        
        # Calculate recent percentiles for adaptive bounds
        recent_portion = min(20, len(original_input))
        recent_data = original_input[-recent_portion:]
        recent_min = np.percentile(recent_data, 10)
        recent_max = np.percentile(recent_data, 90)
        
        # Adaptive bounds that expand slightly over time but stay reasonable
        expansion_factor = 1.0 + (step * 0.02)  # Gradual expansion
        lower_bound = max(historical_min - 0.5 * historical_std, 
                         recent_min - 0.3 * historical_std * expansion_factor)
        upper_bound = min(historical_max + 0.5 * historical_std, 
                         recent_max + 0.3 * historical_std * expansion_factor)
        
        # Apply soft bounds correction
        if prediction < lower_bound:
            return (lower_bound - prediction) * 0.6
        elif prediction > upper_bound:
            return (upper_bound - prediction) * 0.6
        else:
            return 0.0
    
    def _calculate_pattern_consistency_correction(self, prediction, step, original_input, recent_trend):
        """Calculate correction to maintain pattern consistency"""
        if len(original_input) >= 6:
            # Analyze pattern consistency in recent data
            pattern_window = min(6, len(original_input))
            recent_pattern = original_input[-pattern_window:]
            
            # Calculate pattern regularity
            changes = np.diff(recent_pattern)
            if len(changes) > 2:
                # Check for consistent directional patterns
                positive_changes = np.sum(changes > 0)
                negative_changes = np.sum(changes < 0)
                
                # If there's a strong directional pattern, enforce it
                if positive_changes > len(changes) * 0.7:  # Strong upward pattern
                    expected_change = np.mean(changes[changes > 0])
                    expected_next = original_input[-1] + expected_change * 0.7
                    return (expected_next - prediction) * 0.3
                elif negative_changes > len(changes) * 0.7:  # Strong downward pattern
                    expected_change = np.mean(changes[changes < 0])
                    expected_next = original_input[-1] + expected_change * 0.7
                    return (expected_next - prediction) * 0.3
        
        return 0.0
    
    def _calculate_momentum_correction(self, prediction, step, original_input, recent_trend):
        """Calculate momentum correction to maintain historical directional patterns"""
        if len(original_input) >= 3:
            # Calculate recent momentum
            recent_momentum = np.mean(np.diff(original_input[-3:]))
            
            # If prediction goes against recent momentum, apply correction
            if recent_trend > 0.01 and prediction < original_input[-1]:
                # Upward trend should continue
                momentum_correction = recent_momentum * 0.4 * (1 / (1 + step * 0.1))
            elif recent_trend < -0.01 and prediction > original_input[-1]:
                # Downward trend should continue
                momentum_correction = recent_momentum * 0.4 * (1 / (1 + step * 0.1))
            else:
                momentum_correction = 0.0
            
            return momentum_correction
        
        return 0.0
    
    def _estimate_pattern_based_value(self, original_input, step, recent_trend, local_mean):
        """Estimate next value based on advanced historical pattern analysis"""
        if len(original_input) >= 3:
            # Multi-scale pattern analysis
            
            # 1. Short-term pattern (last 3-5 values)
            short_window = min(5, len(original_input))
            short_pattern = original_input[-short_window:]
            short_changes = np.diff(short_pattern)
            short_trend = np.mean(short_changes) if len(short_changes) > 0 else 0
            
            # 2. Medium-term pattern (last 6-10 values)
            medium_window = min(10, len(original_input))
            medium_pattern = original_input[-medium_window:]
            medium_changes = np.diff(medium_pattern)
            medium_trend = np.mean(medium_changes) if len(medium_changes) > 0 else 0
            
            # 3. Adaptive trend component with pattern-aware decay
            trend_consistency = self._calculate_pattern_trend_consistency(original_input)
            
            # Use trend consistency to determine decay rate
            if trend_consistency > 0.7:  # Strong consistent trend
                trend_decay = 0.98 ** step
            elif trend_consistency > 0.4:  # Moderate trend
                trend_decay = 0.95 ** step
            else:  # Weak or inconsistent trend
                trend_decay = 0.90 ** step
            
            # Weighted combination of trends
            weighted_trend = (short_trend * 0.5 + medium_trend * 0.3 + recent_trend * 0.2) * trend_decay
            
            # 4. Enhanced cyclical component with pattern learning
            cyclical_component = self._calculate_adaptive_cyclical_component(
                original_input, step, short_pattern
            )
            
            # 5. Pattern-based base value calculation
            base_value = original_input[-1] + weighted_trend + cyclical_component
            
            # 6. Adaptive mean reversion with historical context
            # Use recent mean for short-term, historical mean for long-term
            short_mean = np.mean(short_pattern)
            historical_mean = np.mean(original_input)
            
            # Adaptive weighting based on step distance
            recent_weight = 1.0 / (1.0 + step * 0.05)
            historical_weight = 1.0 - recent_weight
            
            target_mean = recent_weight * short_mean + historical_weight * historical_mean
            mean_reversion = (target_mean - base_value) * 0.03 * (1 + step * 0.002)
            
            # 7. Volatility-aware adjustment
            volatility_adjustment = self._calculate_volatility_aware_adjustment(
                original_input, base_value, step
            )
            
            final_value = base_value + mean_reversion + volatility_adjustment
            
            # 8. Ensure value is within reasonable bounds
            historical_std = np.std(original_input)
            final_value = np.clip(final_value, 
                                 historical_mean - 3 * historical_std, 
                                 historical_mean + 3 * historical_std)
            
            return final_value
        
        return original_input[-1]
    
    def _calculate_pattern_trend_consistency(self, original_input):
        """Calculate how consistent the trend is across different segments"""
        if len(original_input) < 8:
            return 0.5
        
        # Split into segments and calculate trends
        segment_size = len(original_input) // 3
        trends = []
        
        for i in range(0, len(original_input) - segment_size, segment_size):
            segment = original_input[i:i + segment_size]
            if len(segment) >= 3:
                trend = np.polyfit(np.arange(len(segment)), segment, 1)[0]
                trends.append(trend)
        
        if len(trends) > 1:
            # Calculate consistency as inverse of standard deviation
            trend_std = np.std(trends)
            trend_mean = np.mean(np.abs(trends))
            consistency = 1.0 / (1.0 + trend_std / (trend_mean + 1e-8))
            return max(0.0, min(1.0, consistency))
        
        return 0.5
    
    def _calculate_adaptive_cyclical_component(self, original_input, step, short_pattern):
        """Calculate adaptive cyclical component based on detected patterns"""
        if len(original_input) >= 8:
            # Look for cyclical patterns in the data
            autocorr_values = []
            for lag in range(1, min(6, len(original_input) // 2)):
                if len(original_input) >= lag * 2:
                    x1 = original_input[:-lag]
                    x2 = original_input[lag:]
                    if len(x1) > 0 and len(x2) > 0:
                        corr = np.corrcoef(x1, x2)[0, 1]
                        if not np.isnan(corr):
                            autocorr_values.append((lag, corr))
            
            # Find the strongest cyclical pattern
            if autocorr_values:
                best_lag, best_corr = max(autocorr_values, key=lambda x: abs(x[1]))
                
                if abs(best_corr) > 0.3:  # Significant correlation
                    # Calculate cyclical component based on the pattern
                    cycle_phase = (step % best_lag) / best_lag * 2 * np.pi
                    cycle_amplitude = np.std(short_pattern) * 0.3 * abs(best_corr)
                    cyclical_component = cycle_amplitude * np.sin(cycle_phase)
                    
                    # Decay the cyclical component over time
                    decay_factor = 0.97 ** step
                    return cyclical_component * decay_factor
        
        return 0.0
    
    def _calculate_volatility_aware_adjustment(self, original_input, base_value, step):
        """Calculate volatility-aware adjustment to maintain realistic variability"""
        if len(original_input) >= 5:
            # Calculate historical volatility
            historical_changes = np.diff(original_input)
            historical_volatility = np.std(historical_changes)
            
            # Calculate recent volatility
            recent_changes = np.diff(original_input[-5:])
            recent_volatility = np.std(recent_changes) if len(recent_changes) > 0 else historical_volatility
            
            # Target volatility (blend of historical and recent)
            target_volatility = 0.7 * recent_volatility + 0.3 * historical_volatility
            
            # Add controlled variability to maintain realistic patterns
            variability_factor = target_volatility * 0.1 * (1 / (1 + step * 0.05))
            
            # Random component with appropriate scale
            import random
            random_component = random.gauss(0, variability_factor)
            
            return random_component
        
        return 0.0
    
    def _apply_final_smoothing(self, predictions, last_sequence, historical_mean, recent_trend):
        """Apply final smoothing to maintain consistency with historical patterns"""
        # Smooth transitions between predictions
        smoothed_predictions = predictions.copy()
        
        # Apply moving average smoothing
        window_size = min(3, len(predictions))
        if window_size > 1:
            for i in range(window_size, len(smoothed_predictions)):
                smoothed_predictions[i] = np.mean(smoothed_predictions[i-window_size:i+1])
        
        # Ensure predictions don't drift too far from historical characteristics
        prediction_mean = np.mean(smoothed_predictions)
        if abs(prediction_mean - historical_mean) > 2 * np.std(last_sequence):
            # Apply global correction
            correction = (historical_mean - prediction_mean) * 0.3
            smoothed_predictions += correction
        
        return smoothed_predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metrics"""
        return {
            'model_type': self.model_type,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'fitted': self.fitted,
            'metrics': self.metrics,
            'training_history': self.training_history
        }


class ModelEnsemble:
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, models: List[AdvancedTimeSeriesForecaster], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.fitted = False
        
    def fit(self, data: pd.DataFrame, time_col: str, target_col: str) -> Dict[str, Any]:
        """Fit all models in the ensemble"""
        results = []
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}: {model.model_type}")
            
            # Prepare data for this model
            data_dict = model.prepare_data(data, time_col, target_col)
            
            # Train model
            if model.model_type in ['dlinear', 'nbeats', 'lstm']:
                result = model.train_pytorch_model(
                    data_dict['X_train'], data_dict['y_train'],
                    data_dict['X_test'], data_dict['y_test']
                )
            else:
                result = model.train_gradient_boosting(
                    data_dict['X_train'], data_dict['y_train'],
                    data_dict['X_test'], data_dict['y_test']
                )
            
            # Evaluate model
            metrics = model.evaluate_model(data_dict['X_test'], data_dict['y_test'])
            result['metrics'] = metrics
            results.append(result)
            
            print(f"Model {i+1} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
        
        self.fitted = True
        return results
    
    def predict(self, last_sequence: np.ndarray, steps: int = 30) -> Dict[str, Any]:
        """Make ensemble predictions"""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted first")
        
        predictions = []
        model_predictions = {}
        
        for i, model in enumerate(self.models):
            try:
                pred = model.predict_next_steps(last_sequence, steps)
                predictions.append(pred * self.weights[i])
                model_predictions[f'model_{i}_{model.model_type}'] = pred
            except Exception as e:
                print(f"Error predicting with model {i}: {e}")
                # Use fallback prediction
                fallback_pred = np.full(steps, last_sequence[-1])
                predictions.append(fallback_pred * self.weights[i])
                model_predictions[f'model_{i}_{model.model_type}'] = fallback_pred
        
        # Ensemble prediction
        ensemble_prediction = np.sum(predictions, axis=0)
        
        # Calculate prediction confidence based on agreement between models
        model_preds_array = np.array(list(model_predictions.values()))
        prediction_std = np.std(model_preds_array, axis=0)
        confidence = np.maximum(0, 100 - (prediction_std / np.abs(ensemble_prediction + 1e-8)) * 100)
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': model_predictions,
            'prediction_confidence': confidence,
            'prediction_std': prediction_std
        }
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get metrics for all models in the ensemble"""
        ensemble_metrics = {}
        
        for i, model in enumerate(self.models):
            ensemble_metrics[f'model_{i}_{model.model_type}'] = model.metrics
        
        return ensemble_metrics


def create_advanced_forecasting_models(seq_len: int = 50, pred_len: int = 30) -> List[AdvancedTimeSeriesForecaster]:
    """Create a list of advanced forecasting models"""
    models = []
    
    # DLinear model
    models.append(AdvancedTimeSeriesForecaster(seq_len, pred_len, 'dlinear'))
    
    # N-BEATS model
    models.append(AdvancedTimeSeriesForecaster(seq_len, pred_len, 'nbeats'))
    
    # LSTM model
    models.append(AdvancedTimeSeriesForecaster(seq_len, pred_len, 'lstm'))
    
    return models


def optimize_hyperparameters(model_type: str, data: pd.DataFrame, time_col: str, 
                           target_col: str, n_trials: int = 50) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna"""
    
    def objective(trial):
        # Suggest hyperparameters
        seq_len = trial.suggest_int('seq_len', 20, 100)
        pred_len = trial.suggest_int('pred_len', 10, 50)
        
        # Create model
        model = AdvancedTimeSeriesForecaster(seq_len, pred_len, model_type)
        
        # Prepare data
        data_dict = model.prepare_data(data, time_col, target_col)
        
        try:
            # Train model
            if model_type in ['dlinear', 'nbeats', 'lstm']:
                lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
                model.train_pytorch_model(
                    data_dict['X_train'], data_dict['y_train'],
                    data_dict['X_test'], data_dict['y_test'],
                    epochs=50, batch_size=batch_size, lr=lr
                )
            else:
                model.train_gradient_boosting(
                    data_dict['X_train'], data_dict['y_train'],
                    data_dict['X_test'], data_dict['y_test']
                )
            
            # Evaluate
            metrics = model.evaluate_model(data_dict['X_test'], data_dict['y_test'])
            return metrics['rmse']
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    # Create study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials)
    }