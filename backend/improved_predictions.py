"""
Improved prediction algorithms for better pattern recognition and accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def analyze_historical_patterns(data: pd.DataFrame, time_col: str, target_col: str) -> Dict[str, Any]:
    """Analyze historical data patterns for better prediction"""
    historical_values = data[target_col].values
    timestamps = pd.to_datetime(data[time_col])
    
    # Basic statistics
    mean_val = np.mean(historical_values)
    std_val = np.std(historical_values)
    min_val = np.min(historical_values)
    max_val = np.max(historical_values)
    
    # Trend analysis using linear regression
    trend_coef = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
    trend_direction = 'increasing' if trend_coef > 0 else 'decreasing' if trend_coef < 0 else 'stable'
    
    # Seasonality detection (simple autocorrelation-based)
    seasonal_period = 0
    if len(historical_values) > 10:
        # Look for periodic patterns using autocorrelation
        autocorr = np.correlate(historical_values, historical_values, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (potential seasonal periods)
        if len(autocorr) > 5:
            # Look for the strongest correlation after lag 1
            potential_periods = []
            for i in range(2, min(len(autocorr), len(historical_values)//3)):
                if i < len(autocorr) and autocorr[i] > 0.5 * autocorr[0]:
                    potential_periods.append(i)
            
            if potential_periods:
                seasonal_period = potential_periods[0]
    
    # Volatility analysis
    volatility = np.std(np.diff(historical_values)) / mean_val if mean_val != 0 else 0
    
    # Recent trend (last 20% of data)
    recent_data_points = max(3, len(historical_values) // 5)
    recent_values = historical_values[-recent_data_points:]
    recent_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
    
    # Pattern consistency score
    consistency_score = 1.0 - (volatility / mean_val if mean_val != 0 else 1.0)
    consistency_score = max(0, min(1, consistency_score))
    
    # Smooth factor based on data characteristics
    smooth_factor = 0.1 + (0.4 * consistency_score)
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'trend_coefficient': trend_coef,
        'trend_direction': trend_direction,
        'seasonal_period': seasonal_period,
        'volatility': volatility,
        'recent_trend': recent_trend,
        'data_length': len(historical_values),
        'consistency_score': consistency_score,
        'smooth_factor': smooth_factor
    }

def smooth_predictions(predictions: np.ndarray, patterns: Dict[str, Any], historical_last_value: float) -> np.ndarray:
    """Apply smoothing to predictions to avoid unrealistic jumps"""
    if len(predictions) == 0:
        return predictions
    
    smoothed = np.copy(predictions)
    mean_val = patterns['mean']
    smooth_factor = patterns['smooth_factor']
    max_change_per_step = patterns['volatility'] * mean_val * 2
    
    # Ensure first prediction is not too far from last historical value
    if abs(smoothed[0] - historical_last_value) > max_change_per_step:
        smoothed[0] = historical_last_value + np.sign(smoothed[0] - historical_last_value) * max_change_per_step
    
    # Apply smoothing to subsequent predictions
    for i in range(1, len(smoothed)):
        max_change = max_change_per_step * (1 + smooth_factor)
        if abs(smoothed[i] - smoothed[i-1]) > max_change:
            smoothed[i] = smoothed[i-1] + np.sign(smoothed[i] - smoothed[i-1]) * max_change
    
    return smoothed

def apply_pattern_continuity(predictions: np.ndarray, patterns: Dict[str, Any]) -> np.ndarray:
    """Apply pattern continuity to make predictions follow historical patterns"""
    if len(predictions) == 0:
        return predictions
    
    adjusted = np.copy(predictions)
    
    # Apply recent trend continuation
    if abs(patterns['recent_trend']) > 0.01:  # Significant recent trend
        trend_influence = min(0.5, patterns['consistency_score'])  # Stronger influence for consistent data
        trend_adjustment = np.array([patterns['recent_trend'] * i * trend_influence for i in range(len(predictions))])
        adjusted = adjusted + trend_adjustment
    
    # Apply seasonality if detected
    if patterns['seasonal_period'] > 0 and patterns['seasonal_period'] < len(predictions):
        seasonal_influence = 0.2 * patterns['consistency_score']
        for i in range(len(predictions)):
            cycle_position = i % patterns['seasonal_period']
            seasonal_adjustment = np.sin(2 * np.pi * cycle_position / patterns['seasonal_period']) * patterns['std'] * seasonal_influence
            adjusted[i] += seasonal_adjustment
    
    return adjusted

def apply_realistic_bounds(predictions: np.ndarray, patterns: Dict[str, Any]) -> np.ndarray:
    """Apply realistic bounds to predictions based on historical patterns"""
    if len(predictions) == 0:
        return predictions
    
    mean_val = patterns['mean']
    std_val = patterns['std']
    
    # Dynamic bounds based on historical data
    upper_bound = mean_val + 3 * std_val
    lower_bound = mean_val - 3 * std_val
    
    # Allow for some extrapolation beyond historical bounds if trend is strong
    if abs(patterns['trend_coefficient']) > 0.1:
        trend_allowance = abs(patterns['trend_coefficient']) * len(predictions) * 0.5
        if patterns['trend_coefficient'] > 0:
            upper_bound += trend_allowance
        else:
            lower_bound -= trend_allowance
    
    # Apply bounds
    bounded = np.clip(predictions, lower_bound, upper_bound)
    
    return bounded

def generate_improved_prophet_predictions(model, data: pd.DataFrame, time_col: str, target_col: str, 
                                        steps: int, patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Generate improved Prophet predictions with pattern awareness"""
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    
    # Get predictions
    predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
    prediction_values = predictions['yhat'].values
    
    # Get last historical value for continuity
    historical_last_value = data[target_col].iloc[-1]
    
    # Apply pattern-aware improvements
    prediction_values = apply_pattern_continuity(prediction_values, patterns)
    prediction_values = smooth_predictions(prediction_values, patterns, historical_last_value)
    prediction_values = apply_realistic_bounds(prediction_values, patterns)
    
    # Adjust confidence intervals
    confidence_intervals = []
    for i, (_, row) in enumerate(predictions.iterrows()):
        lower = max(row['yhat_lower'], patterns['mean'] - 3 * patterns['std'])
        upper = min(row['yhat_upper'], patterns['mean'] + 3 * patterns['std'])
        
        # Adjust based on prediction value
        center = prediction_values[i]
        interval_width = (upper - lower) * 0.8  # Slightly narrower intervals
        confidence_intervals.append({
            'lower': center - interval_width / 2,
            'upper': center + interval_width / 2
        })
    
    result = {
        'timestamps': predictions['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'predictions': prediction_values.tolist(),
        'confidence_intervals': confidence_intervals
    }
    
    return result

def generate_improved_arima_predictions(model, data: pd.DataFrame, time_col: str, target_col: str, 
                                      steps: int, patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Generate improved ARIMA predictions with pattern awareness"""
    
    # Generate ARIMA predictions
    forecast = model.forecast(steps=steps)
    prediction_values = forecast.values
    
    # Get last historical value for continuity
    historical_last_value = data[target_col].iloc[-1]
    
    # Apply pattern-aware improvements
    prediction_values = apply_pattern_continuity(prediction_values, patterns)
    prediction_values = smooth_predictions(prediction_values, patterns, historical_last_value)
    prediction_values = apply_realistic_bounds(prediction_values, patterns)
    
    # Create timestamps
    if time_col in data.columns:
        last_timestamp = pd.to_datetime(data[time_col].iloc[-1])
        time_series = pd.to_datetime(data[time_col])
        freq = pd.infer_freq(time_series)
    else:
        last_timestamp = pd.to_datetime(data.index[-1])
        freq = pd.infer_freq(pd.to_datetime(data.index))
    
    if freq is None:
        freq = 'D'
    
    try:
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(days=1), 
            periods=steps, 
            freq=freq
        )
    except:
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(days=1), 
            periods=steps, 
            freq='D'
        )
    
    result = {
        'timestamps': future_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'predictions': prediction_values.tolist(),
        'confidence_intervals': None
    }
    
    return result

def train_improved_prophet_model(data: pd.DataFrame, time_col: str, target_col: str, params: Dict[str, Any]):
    """Train Prophet model with improved pattern recognition"""
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_data = data.rename(columns={time_col: 'ds', target_col: 'y'})
    
    # Analyze historical patterns for better configuration
    patterns = analyze_historical_patterns(data, time_col, target_col)
    data_length = patterns['data_length']
    
    # Determine seasonality based on data characteristics
    weekly_seasonality = data_length > 14  # At least 2 weeks of data
    yearly_seasonality = data_length > 365  # At least 1 year of data
    daily_seasonality = data_length > 10   # At least 10 days of data
    
    # Analyze trend strength
    trend_strength = abs(patterns['trend_coefficient'])
    
    # Create and train model with adaptive parameters
    model = Prophet(
        seasonality_mode=params.get('seasonality_mode', 'additive'),
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=min(0.05 + trend_strength * 0.1, 0.5),  # Adaptive changepoint sensitivity
        seasonality_prior_scale=10.0,  # Strong seasonality influence
        holidays_prior_scale=10.0,
        interval_width=0.8,
        growth='linear'
    )
    
    model.fit(prophet_data)
    return model

def train_improved_arima_model(data: pd.DataFrame, time_col: str, target_col: str, params: Dict[str, Any]):
    """Train ARIMA model with improved pattern recognition"""
    # Prepare data
    ts_data = data.set_index(time_col)[target_col]
    
    # Analyze historical patterns for better ARIMA configuration
    patterns = analyze_historical_patterns(data, time_col, target_col)
    
    # Auto-detect best ARIMA parameters if not provided
    if 'order' not in params:
        data_length = patterns['data_length']
        
        # Check for trend (first differencing needed)
        trend_strength = abs(patterns['trend_coefficient'])
        d = 1 if trend_strength > 0.1 else 0
        
        # AR parameter - based on data characteristics
        p = min(3, max(1, data_length // 10))  # Adaptive AR order
        
        # MA parameter - based on moving average characteristics
        q = min(2, max(1, data_length // 20))  # Adaptive MA order
        
        order = (p, d, q)
    else:
        order = params.get('order', (2, 1, 2))
    
    # Create and train ARIMA model
    model = ARIMA(ts_data, order=order)
    fitted_model = model.fit()
    
    return fitted_model