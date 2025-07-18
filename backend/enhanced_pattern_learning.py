"""
Enhanced Pattern Learning for Time Series Forecasting
Advanced algorithms for learning and preserving historical patterns in predictions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """Advanced pattern analysis for time series data"""
    
    def __init__(self):
        self.patterns = {}
        self.pattern_features = {}
        self.historical_stats = {}
        
    def analyze_patterns(self, data: np.ndarray, window_sizes: List[int] = [3, 5, 7, 10, 15]) -> Dict[str, Any]:
        """
        Comprehensive pattern analysis including:
        - Trend patterns
        - Seasonal patterns
        - Cyclical patterns
        - Volatility patterns
        - Structural patterns
        """
        results = {
            'trends': self._analyze_trends(data),
            'seasonality': self._analyze_seasonality(data),
            'cycles': self._analyze_cycles(data),
            'volatility': self._analyze_volatility(data),
            'structural': self._analyze_structural_patterns(data, window_sizes),
            'local_patterns': self._analyze_local_patterns(data, window_sizes),
            'statistical_props': self._analyze_statistical_properties(data)
        }
        
        # Store pattern features for later use
        self.pattern_features = results
        self.historical_stats = results['statistical_props']
        
        return results
    
    def _analyze_trends(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze trend patterns at multiple scales"""
        trends = {}
        
        # Multi-scale trend analysis
        for window in [5, 10, 20, len(data)//2, len(data)]:
            if window >= len(data):
                window = len(data)
            
            # Linear trend with error handling
            x = np.arange(window)
            if window <= len(data):
                segment = data[-window:]
                try:
                    trend_coef = np.polyfit(x, segment, 1)[0]
                    trends[f'trend_{window}'] = float(trend_coef)
                except np.linalg.LinAlgError:
                    # If SVD doesn't converge, use simple difference
                    if len(segment) > 1:
                        trends[f'trend_{window}'] = float((segment[-1] - segment[0]) / (len(segment) - 1))
                    else:
                        trends[f'trend_{window}'] = 0.0
                except Exception:
                    trends[f'trend_{window}'] = 0.0
            
        # Trend direction consistency
        short_trend = trends.get('trend_5', 0)
        medium_trend = trends.get('trend_10', 0)
        long_trend = trends.get('trend_20', 0)
        
        trends['consistency'] = float(np.mean([
            np.sign(short_trend) == np.sign(medium_trend),
            np.sign(medium_trend) == np.sign(long_trend),
            np.sign(short_trend) == np.sign(long_trend)
        ]))
        
        # Trend strength
        trend_values = [abs(t) for t in trends.values() if isinstance(t, (int, float))]
        trends['strength'] = float(np.std(trend_values)) if trend_values else 0.0
        
        return trends
    
    def _analyze_seasonality(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        seasonality = {}
        
        # Auto-correlation based seasonality detection
        max_lag = min(len(data) // 3, 50)
        if max_lag > 3:
            autocorr = []
            for lag in range(1, max_lag):
                if len(data) > lag:
                    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr.append(corr)
                    else:
                        autocorr.append(0)
            
            if autocorr:
                seasonality['autocorr'] = autocorr
                seasonality['dominant_period'] = np.argmax(autocorr) + 1
                seasonality['strength'] = max(autocorr)
        
        # Fourier-based period detection
        if len(data) > 10:
            fft = np.fft.fft(data - np.mean(data))
            freqs = np.fft.fftfreq(len(data))
            power = np.abs(fft) ** 2
            
            # Find dominant frequencies
            peak_indices = signal.find_peaks(power[1:len(power)//2], height=np.max(power)*0.1)[0]
            if len(peak_indices) > 0:
                dominant_freq = freqs[peak_indices[0] + 1]
                seasonality['dominant_frequency'] = dominant_freq
                seasonality['period'] = 1 / abs(dominant_freq) if dominant_freq != 0 else len(data)
        
        return seasonality
    
    def _analyze_cycles(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze cyclical patterns"""
        cycles = {}
        
        # Detect peaks and troughs
        peaks, _ = signal.find_peaks(data, distance=max(1, len(data)//10))
        troughs, _ = signal.find_peaks(-data, distance=max(1, len(data)//10))
        
        cycles['peaks'] = peaks
        cycles['troughs'] = troughs
        cycles['peak_values'] = data[peaks] if len(peaks) > 0 else []
        cycles['trough_values'] = data[troughs] if len(troughs) > 0 else []
        
        # Cycle characteristics
        if len(peaks) > 1:
            cycles['average_cycle_length'] = np.mean(np.diff(peaks))
            cycles['cycle_amplitude'] = np.mean(cycles['peak_values']) - np.mean(cycles['trough_values']) if len(cycles['trough_values']) > 0 else 0
        
        return cycles
    
    def _analyze_volatility(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility patterns"""
        volatility = {}
        
        # Multi-scale volatility
        for window in [3, 5, 10, 20]:
            if window < len(data):
                rolling_std = []
                for i in range(window, len(data)):
                    rolling_std.append(np.std(data[i-window:i]))
                volatility[f'volatility_{window}'] = np.mean(rolling_std) if rolling_std else 0
        
        # Volatility clustering
        changes = np.diff(data)
        high_vol_threshold = np.percentile(np.abs(changes), 75)
        high_vol_periods = np.abs(changes) > high_vol_threshold
        
        if len(high_vol_periods) > 0:
            volatility['clustering'] = np.mean(high_vol_periods)
            volatility['volatility_persistence'] = np.mean([
                high_vol_periods[i] and high_vol_periods[i-1] 
                for i in range(1, len(high_vol_periods))
            ])
        
        return volatility
    
    def _analyze_structural_patterns(self, data: np.ndarray, window_sizes: List[int]) -> Dict[str, Any]:
        """Analyze structural patterns using multiple window sizes"""
        structural = {}
        
        # Pattern templates with error handling
        def safe_linear_check(x):
            try:
                if len(x) < 2:
                    return False
                corr = np.corrcoef(np.arange(len(x)), x)[0, 1]
                return abs(corr) > 0.8 if not np.isnan(corr) else False
            except:
                return False
        
        patterns = {
            'increasing': lambda x: np.all(np.diff(x) > 0) if len(x) > 1 else False,
            'decreasing': lambda x: np.all(np.diff(x) < 0) if len(x) > 1 else False,
            'u_shaped': lambda x: np.argmin(x) == len(x) // 2 if len(x) > 2 else False,
            'inverted_u': lambda x: np.argmax(x) == len(x) // 2 if len(x) > 2 else False,
            'linear': safe_linear_check,
            'stable': lambda x: np.std(x) < 0.1 * np.mean(np.abs(x)) if len(x) > 0 and np.mean(np.abs(x)) > 0 else False
        }
        
        # Analyze patterns at different scales
        for window_size in window_sizes:
            if window_size >= len(data):
                continue
                
            pattern_counts = {pattern: 0 for pattern in patterns.keys()}
            
            for i in range(len(data) - window_size + 1):
                segment = data[i:i + window_size]
                for pattern_name, pattern_func in patterns.items():
                    try:
                        if pattern_func(segment):
                            pattern_counts[pattern_name] += 1
                    except Exception:
                        pass
            
            # Normalize counts
            total_windows = len(data) - window_size + 1
            if total_windows > 0:
                for pattern_name in pattern_counts:
                    pattern_counts[pattern_name] = float(pattern_counts[pattern_name] / total_windows)
            
            structural[f'window_{window_size}'] = pattern_counts
        
        return structural
    
    def _analyze_local_patterns(self, data: np.ndarray, window_sizes: List[int]) -> Dict[str, Any]:
        """Analyze local patterns and their characteristics"""
        local = {}
        
        # Local extrema patterns
        for window_size in window_sizes:
            if window_size >= len(data):
                continue
            
            local_maxima = []
            local_minima = []
            local_trends = []
            
            for i in range(window_size, len(data) - window_size):
                segment = data[i-window_size:i+window_size+1]
                center_idx = window_size
                
                # Check if center is local maximum/minimum
                if segment[center_idx] == np.max(segment):
                    local_maxima.append(i)
                elif segment[center_idx] == np.min(segment):
                    local_minima.append(i)
                
                # Calculate local trend
                local_trend = np.polyfit(range(len(segment)), segment, 1)[0]
                local_trends.append(local_trend)
            
            local[f'maxima_{window_size}'] = local_maxima
            local[f'minima_{window_size}'] = local_minima
            local[f'trends_{window_size}'] = local_trends
        
        return local
    
    def _analyze_statistical_properties(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze statistical properties"""
        stats = {}
        
        # Basic statistics
        stats['mean'] = float(np.mean(data))
        stats['std'] = float(np.std(data))
        stats['min'] = float(np.min(data))
        stats['max'] = float(np.max(data))
        stats['range'] = float(stats['max'] - stats['min'])
        stats['median'] = float(np.median(data))
        
        # Distribution properties
        stats['skewness'] = float(self._calculate_skewness(data))
        stats['kurtosis'] = float(self._calculate_kurtosis(data))
        
        # Variability measures
        stats['coefficient_of_variation'] = float(stats['std'] / stats['mean'] if stats['mean'] != 0 else 0)
        stats['iqr'] = float(np.percentile(data, 75) - np.percentile(data, 25))
        
        # Change characteristics
        changes = np.diff(data)
        stats['change_mean'] = float(np.mean(changes))
        stats['change_std'] = float(np.std(changes))
        stats['change_magnitude'] = float(np.mean(np.abs(changes)))
        
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3


class PatternAwareLSTM(nn.Module):
    """LSTM model with pattern awareness"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, pattern_features: Dict[str, Any]):
        super(PatternAwareLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pattern_features = pattern_features
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Pattern-aware attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Pattern context layers
        self.pattern_context = nn.Linear(hidden_size, hidden_size)
        
        # Output layers
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Pattern context
        pattern_out = self.pattern_context(attn_out)
        pattern_out = torch.tanh(pattern_out)
        
        # Final output
        out = self.fc(self.dropout(pattern_out[:, -1, :]))
        return out


class EnhancedPatternPredictor:
    """Enhanced predictor that learns and preserves historical patterns"""
    
    def __init__(self, seq_len: int = 20, hidden_size: int = 64, num_layers: int = 2):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.scaler = MinMaxScaler()
        self.pattern_analyzer = PatternAnalyzer()
        self.model = None
        self.fitted = False
        
        # Pattern preservation parameters
        self.pattern_weights = {
            'trend': 0.3,
            'seasonality': 0.2,
            'cycles': 0.2,
            'volatility': 0.1,
            'local_patterns': 0.2
        }
    
    def fit(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Fit the pattern-aware model
        """
        # Analyze patterns in the data
        patterns = self.pattern_analyzer.analyze_patterns(data)
        
        # Prepare data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Initialize model
        self.model = PatternAwareLSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            pattern_features=patterns
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)
        
        # Training loop
        training_losses = []
        self.model.train()
        
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
            if epoch % 20 == 0:
                logger.info(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
        
        self.fitted = True
        
        return {
            'training_losses': training_losses,
            'final_loss': training_losses[-1],
            'patterns': patterns,
            'model_info': {
                'seq_len': self.seq_len,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers
            }
        }
    
    def predict_next_steps(self, data: np.ndarray, steps: int = 30) -> np.ndarray:
        """
        Generate pattern-aware predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Analyze current patterns
        current_patterns = self.pattern_analyzer.analyze_patterns(data)
        
        # Scale input data
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Get initial sequence
        if len(scaled_data) < self.seq_len:
            # Pad with last value
            padded_data = np.concatenate([
                np.full(self.seq_len - len(scaled_data), scaled_data[-1]),
                scaled_data
            ])
        else:
            padded_data = scaled_data[-self.seq_len:]
        
        predictions = []
        current_sequence = padded_data.copy()
        
        self.model.eval()
        with torch.no_grad():
            for step in range(steps):
                # Prepare input
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1)
                
                # Get model prediction
                pred = self.model(input_tensor).item()
                
                # Apply pattern-aware corrections
                corrected_pred = self._apply_pattern_corrections(
                    pred, step, current_sequence, data, current_patterns
                )
                
                predictions.append(corrected_pred)
                
                # Update sequence
                current_sequence = np.append(current_sequence[1:], corrected_pred)
        
        # Inverse transform
        predictions_array = np.array(predictions)
        predictions_unscaled = self.scaler.inverse_transform(predictions_array.reshape(-1, 1)).flatten()
        
        return predictions_unscaled
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        X, y = [], []
        
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i + self.seq_len])
            y.append(data[i + self.seq_len])
        
        return np.array(X), np.array(y)
    
    def _apply_pattern_corrections(self, prediction: float, step: int, 
                                 current_sequence: np.ndarray, 
                                 original_data: np.ndarray,
                                 current_patterns: Dict[str, Any]) -> float:
        """
        Apply pattern-aware corrections to maintain historical characteristics
        """
        corrections = []
        
        # Trend correction
        trend_correction = self._calculate_trend_correction(
            prediction, step, current_sequence, original_data, current_patterns
        )
        corrections.append(trend_correction * self.pattern_weights['trend'])
        
        # Seasonality correction
        seasonality_correction = self._calculate_seasonality_correction(
            prediction, step, current_sequence, original_data, current_patterns
        )
        corrections.append(seasonality_correction * self.pattern_weights['seasonality'])
        
        # Cyclical correction
        cycle_correction = self._calculate_cycle_correction(
            prediction, step, current_sequence, original_data, current_patterns
        )
        corrections.append(cycle_correction * self.pattern_weights['cycles'])
        
        # Volatility correction
        volatility_correction = self._calculate_volatility_correction(
            prediction, step, current_sequence, original_data, current_patterns
        )
        corrections.append(volatility_correction * self.pattern_weights['volatility'])
        
        # Local pattern correction
        local_correction = self._calculate_local_pattern_correction(
            prediction, step, current_sequence, original_data, current_patterns
        )
        corrections.append(local_correction * self.pattern_weights['local_patterns'])
        
        # Apply corrections
        total_correction = sum(corrections)
        corrected_prediction = prediction + total_correction
        
        return corrected_prediction
    
    def _calculate_trend_correction(self, prediction: float, step: int,
                                  current_sequence: np.ndarray,
                                  original_data: np.ndarray,
                                  patterns: Dict[str, Any]) -> float:
        """Calculate trend-based correction"""
        trends = patterns.get('trends', {})
        
        # Get dominant trend
        trend_strength = trends.get('strength', 0)
        if trend_strength < 0.01:  # Very weak trend
            return 0
        
        # Apply trend with decay
        recent_trend = trends.get('trend_5', 0)
        decay_factor = 0.98 ** step
        
        # Expected next value based on trend
        last_value = current_sequence[-1]
        expected_change = recent_trend * decay_factor
        expected_value = last_value + expected_change
        
        # Correction towards expected trend
        correction = (expected_value - prediction) * 0.3
        
        return correction
    
    def _calculate_seasonality_correction(self, prediction: float, step: int,
                                        current_sequence: np.ndarray,
                                        original_data: np.ndarray,
                                        patterns: Dict[str, Any]) -> float:
        """Calculate seasonality-based correction"""
        seasonality = patterns.get('seasonality', {})
        
        # Check if there's significant seasonality
        strength = seasonality.get('strength', 0)
        if strength < 0.3:  # Weak seasonality
            return 0
        
        # Get period
        period = seasonality.get('dominant_period', 1)
        if period <= 1:
            return 0
        
        # Find corresponding historical point
        total_length = len(original_data)
        historical_index = total_length - (step % period)
        
        if historical_index >= 0 and historical_index < total_length:
            historical_value = original_data[historical_index]
            
            # Scale to current sequence
            scaled_historical = self.scaler.transform([[historical_value]])[0][0]
            
            # Correction towards seasonal pattern
            correction = (scaled_historical - prediction) * strength * 0.2
            return correction
        
        return 0
    
    def _calculate_cycle_correction(self, prediction: float, step: int,
                                  current_sequence: np.ndarray,
                                  original_data: np.ndarray,
                                  patterns: Dict[str, Any]) -> float:
        """Calculate cycle-based correction"""
        cycles = patterns.get('cycles', {})
        
        # Get cycle characteristics
        avg_cycle_length = cycles.get('average_cycle_length', 0)
        if avg_cycle_length <= 0:
            return 0
        
        # Estimate position in cycle
        cycle_position = step % avg_cycle_length
        cycle_progress = cycle_position / avg_cycle_length
        
        # Apply sinusoidal correction based on cycle
        amplitude = cycles.get('cycle_amplitude', 0)
        if amplitude > 0:
            # Simple sinusoidal approximation
            cycle_value = amplitude * np.sin(2 * np.pi * cycle_progress)
            correction = cycle_value * 0.1  # Light correction
            return correction
        
        return 0
    
    def _calculate_volatility_correction(self, prediction: float, step: int,
                                       current_sequence: np.ndarray,
                                       original_data: np.ndarray,
                                       patterns: Dict[str, Any]) -> float:
        """Calculate volatility-based correction"""
        volatility = patterns.get('volatility', {})
        
        # Get expected volatility
        expected_vol = volatility.get('volatility_5', 0)
        if expected_vol == 0:
            return 0
        
        # Calculate recent volatility
        recent_changes = np.diff(current_sequence[-5:])
        recent_vol = np.std(recent_changes) if len(recent_changes) > 0 else 0
        
        # If volatility is too low, add some variation
        if recent_vol < expected_vol * 0.5:
            # Add controlled noise
            noise = np.random.normal(0, expected_vol * 0.2)
            return noise
        
        return 0
    
    def _calculate_local_pattern_correction(self, prediction: float, step: int,
                                          current_sequence: np.ndarray,
                                          original_data: np.ndarray,
                                          patterns: Dict[str, Any]) -> float:
        """Calculate local pattern correction"""
        local = patterns.get('local_patterns', {})
        
        # Analyze recent local pattern
        if len(current_sequence) >= 5:
            recent_pattern = current_sequence[-5:]
            recent_changes = np.diff(recent_pattern)
            
            # Check for consistent local patterns
            if len(recent_changes) > 0:
                avg_change = np.mean(recent_changes)
                expected_value = current_sequence[-1] + avg_change
                
                # Correction towards local pattern
                correction = (expected_value - prediction) * 0.15
                return correction
        
        return 0