"""
Enhanced Time Series Prediction System
Designed for smooth, historically-consistent predictions across any time series pattern
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

class EnhancedTimeSeriesPredictor:
    """
    Enhanced prediction system that maintains historical statistical properties
    and generates smooth, trend-following predictions for any time series pattern
    """
    
    def __init__(self, smoothing_factor=0.3, trend_weight=0.4, seasonality_weight=0.2, noise_weight=0.1):
        self.smoothing_factor = smoothing_factor
        self.trend_weight = trend_weight
        self.seasonality_weight = seasonality_weight
        self.noise_weight = noise_weight
        
        # Pattern analysis results
        self.pattern_analysis = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def analyze_comprehensive_patterns(self, data, time_col=None, target_col=None, max_seasonality_period=50):
        """
        Comprehensive pattern analysis for any time series data
        """
        try:
            # Prepare data
            if isinstance(data, pd.DataFrame):
                if target_col:
                    values = data[target_col].values
                else:
                    # Use the first numeric column
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    values = data[numeric_cols[0]].values
                
                if time_col and time_col in data.columns:
                    time_index = pd.to_datetime(data[time_col])
                else:
                    time_index = pd.date_range(start='2023-01-01', periods=len(values), freq='D')
            else:
                values = np.array(data)
                time_index = pd.date_range(start='2023-01-01', periods=len(values), freq='D')
            
            # Remove NaN values
            valid_mask = ~np.isnan(values)
            values = values[valid_mask]
            time_index = time_index[valid_mask]
            
            if len(values) < 10:
                raise ValueError("Need at least 10 data points for pattern analysis")
            
            # Create time series
            ts = pd.Series(values, index=time_index)
            
            # 1. Basic statistical properties
            stats = self._calculate_statistical_properties(values)
            
            # 2. Trend analysis (multiple methods)
            trend_analysis = self._analyze_trend_patterns(values)
            
            # 3. Seasonality and cyclical patterns
            seasonality_analysis = self._analyze_seasonality_patterns(ts, max_seasonality_period)
            
            # 4. Volatility and noise analysis
            volatility_analysis = self._analyze_volatility_patterns(values)
            
            # 5. Autocorrelation and persistence
            autocorr_analysis = self._analyze_autocorrelation_patterns(values)
            
            # 6. Change point detection
            change_points = self._detect_change_points(values)
            
            # 7. Pattern stability and predictability
            stability_analysis = self._analyze_pattern_stability(values)
            
            # Combine all analyses
            pattern_analysis = {
                'statistics': stats,
                'trend': trend_analysis,
                'seasonality': seasonality_analysis,
                'volatility': volatility_analysis,
                'autocorrelation': autocorr_analysis,
                'change_points': change_points,
                'stability': stability_analysis,
                'raw_values': values,
                'time_index': time_index
            }
            
            self.pattern_analysis = pattern_analysis
            self.is_fitted = True
            
            return pattern_analysis
            
        except Exception as e:
            print(f"Error in pattern analysis: {e}")
            return None
    
    def _calculate_statistical_properties(self, values):
        """Calculate comprehensive statistical properties"""
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'variance': np.var(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'skewness': self._calculate_skewness(values),
            'kurtosis': self._calculate_kurtosis(values),
            'cv': np.std(values) / (np.mean(values) + 1e-8)  # Coefficient of variation
        }
    
    def _calculate_skewness(self, values):
        """Calculate skewness"""
        mean = np.mean(values)
        std = np.std(values)
        return np.mean(((values - mean) / std) ** 3) if std > 0 else 0
    
    def _calculate_kurtosis(self, values):
        """Calculate kurtosis"""
        mean = np.mean(values)
        std = np.std(values)
        return np.mean(((values - mean) / std) ** 4) - 3 if std > 0 else 0
    
    def _analyze_trend_patterns(self, values):
        """Analyze trend patterns using multiple methods"""
        n = len(values)
        x = np.arange(n)
        
        # Linear trend
        linear_coef = np.polyfit(x, values, 1)
        linear_trend = linear_coef[0]
        
        # Polynomial trend (2nd degree)
        poly_coef = np.polyfit(x, values, min(2, n-1))
        
        # Moving average trends
        ma_short = pd.Series(values).rolling(window=min(5, n//4)).mean()
        ma_long = pd.Series(values).rolling(window=min(10, n//2)).mean()
        
        # Trend strength using different window sizes
        trend_strength = self._calculate_trend_strength(values)
        
        # Recent trend (last 30% of data)
        recent_portion = max(3, int(n * 0.3))
        recent_values = values[-recent_portion:]
        recent_x = np.arange(len(recent_values))
        recent_trend = np.polyfit(recent_x, recent_values, 1)[0] if len(recent_values) > 1 else 0
        
        # Trend direction consistency
        direction_consistency = self._calculate_trend_consistency(values)
        
        return {
            'linear_slope': linear_trend,
            'polynomial_coeffs': poly_coef,
            'recent_trend': recent_trend,
            'trend_strength': trend_strength,
            'direction_consistency': direction_consistency,
            'ma_short': ma_short.values,
            'ma_long': ma_long.values,
            'trend_acceleration': poly_coef[0] if len(poly_coef) > 2 else 0
        }
    
    def _calculate_trend_strength(self, values):
        """Calculate trend strength"""
        if len(values) < 3:
            return 0
        
        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        return r_squared * abs(slope)
    
    def _calculate_trend_consistency(self, values):
        """Calculate trend direction consistency"""
        if len(values) < 3:
            return 0
        
        # Calculate local trends
        window_size = max(3, len(values) // 5)
        local_trends = []
        
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            x = np.arange(len(window))
            slope = np.polyfit(x, window, 1)[0]
            local_trends.append(slope)
        
        # Calculate consistency as correlation between consecutive trends
        if len(local_trends) < 2:
            return 0
        
        # Count how many trends have the same direction
        signs = np.sign(local_trends)
        consistency = np.sum(signs == np.median(signs)) / len(signs)
        
        return consistency
    
    def _analyze_seasonality_patterns(self, ts, max_period=50):
        """Analyze seasonality and cyclical patterns"""
        try:
            values = ts.values
            n = len(values)
            
            # Only analyze seasonality if we have enough data
            if n < 20:
                return {
                    'has_seasonality': False,
                    'seasonal_period': None,
                    'seasonal_strength': 0,
                    'seasonal_component': np.zeros(n),
                    'residual_component': values - np.mean(values)
                }
            
            # Try to detect seasonality using autocorrelation
            seasonal_period = self._detect_seasonal_period(values, max_period)
            
            # If we found a seasonal period, decompose the series
            if seasonal_period and seasonal_period > 2:
                try:
                    # Use seasonal decomposition
                    decomposition = seasonal_decompose(ts, model='additive', period=min(seasonal_period, n//2))
                    
                    seasonal_component = decomposition.seasonal.fillna(0).values
                    trend_component = decomposition.trend.fillna(method='bfill').fillna(method='ffill').values
                    residual_component = decomposition.resid.fillna(0).values
                    
                    # Calculate seasonal strength
                    seasonal_strength = np.var(seasonal_component) / (np.var(values) + 1e-8)
                    
                    return {
                        'has_seasonality': True,
                        'seasonal_period': seasonal_period,
                        'seasonal_strength': seasonal_strength,
                        'seasonal_component': seasonal_component,
                        'trend_component': trend_component,
                        'residual_component': residual_component
                    }
                except:
                    pass
            
            # Fallback: no seasonality detected
            return {
                'has_seasonality': False,
                'seasonal_period': None,
                'seasonal_strength': 0,
                'seasonal_component': np.zeros(n),
                'trend_component': values,
                'residual_component': np.zeros(n)
            }
            
        except Exception as e:
            print(f"Error in seasonality analysis: {e}")
            return {
                'has_seasonality': False,
                'seasonal_period': None,
                'seasonal_strength': 0,
                'seasonal_component': np.zeros(len(values)),
                'trend_component': values,
                'residual_component': np.zeros(len(values))
            }
    
    def _detect_seasonal_period(self, values, max_period=50):
        """Detect seasonal period using autocorrelation"""
        if len(values) < 10:
            return None
        
        # Calculate autocorrelation
        autocorr = np.correlate(values, values, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr[1:min(max_period, len(autocorr))], height=0.1)
        
        if len(peaks) > 0:
            # Return the period of the highest peak
            return peaks[np.argmax(autocorr[peaks + 1])] + 1
        
        return None
    
    def _analyze_volatility_patterns(self, values):
        """Analyze volatility and noise patterns"""
        if len(values) < 3:
            return {'volatility': 0, 'noise_level': 0}
        
        # Calculate returns/differences
        returns = np.diff(values)
        
        # Volatility measures
        volatility = np.std(returns)
        
        # Noise level using high-frequency fluctuations
        if len(values) > 5:
            # Smooth the series and calculate residuals
            smoothed = pd.Series(values).rolling(window=3).mean().fillna(method='bfill').fillna(method='ffill')
            noise = values - smoothed.values
            noise_level = np.std(noise)
        else:
            noise_level = volatility
        
        return {
            'volatility': volatility,
            'noise_level': noise_level,
            'returns': returns,
            'return_mean': np.mean(returns),
            'return_std': np.std(returns)
        }
    
    def _analyze_autocorrelation_patterns(self, values):
        """Analyze autocorrelation and persistence"""
        if len(values) < 5:
            return {'autocorr_lag1': 0, 'persistence': 0}
        
        # Calculate autocorrelation at different lags
        autocorr_lags = []
        max_lag = min(10, len(values) // 2)
        
        for lag in range(1, max_lag + 1):
            if len(values) > lag:
                corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                autocorr_lags.append(corr if not np.isnan(corr) else 0)
        
        # Persistence (autocorrelation at lag 1)
        persistence = autocorr_lags[0] if len(autocorr_lags) > 0 else 0
        
        return {
            'autocorr_lag1': persistence,
            'persistence': persistence,
            'autocorr_lags': autocorr_lags
        }
    
    def _detect_change_points(self, values):
        """Detect structural change points in the data"""
        if len(values) < 10:
            return []
        
        # Simple change point detection using variance changes
        change_points = []
        window_size = max(5, len(values) // 10)
        
        for i in range(window_size, len(values) - window_size):
            before = values[i-window_size:i]
            after = values[i:i+window_size]
            
            # Test for significant difference in means
            mean_diff = abs(np.mean(before) - np.mean(after))
            combined_std = np.sqrt((np.var(before) + np.var(after)) / 2)
            
            if combined_std > 0 and mean_diff > 2 * combined_std:
                change_points.append(i)
        
        return change_points
    
    def _analyze_pattern_stability(self, values):
        """Analyze pattern stability and predictability"""
        if len(values) < 10:
            return {'stability_score': 0, 'predictability': 0}
        
        # Calculate stability using rolling statistics
        window_size = max(5, len(values) // 4)
        rolling_means = pd.Series(values).rolling(window=window_size).mean()
        rolling_stds = pd.Series(values).rolling(window=window_size).std()
        
        # Stability score based on consistency of rolling statistics
        mean_stability = 1 - (np.std(rolling_means.dropna()) / (np.mean(rolling_means.dropna()) + 1e-8))
        std_stability = 1 - (np.std(rolling_stds.dropna()) / (np.mean(rolling_stds.dropna()) + 1e-8))
        
        stability_score = (mean_stability + std_stability) / 2
        stability_score = max(0, min(1, stability_score))
        
        # Predictability based on autocorrelation
        if len(values) > 3:
            lag1_autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
            predictability = abs(lag1_autocorr) if not np.isnan(lag1_autocorr) else 0
        else:
            predictability = 0
        
        return {
            'stability_score': stability_score,
            'predictability': predictability,
            'rolling_mean_stability': mean_stability,
            'rolling_std_stability': std_stability
        }
    
    def generate_enhanced_predictions(self, steps=30, confidence_level=0.95):
        """
        Generate enhanced predictions that maintain historical statistical properties
        """
        if not self.is_fitted or self.pattern_analysis is None:
            raise ValueError("Must call analyze_comprehensive_patterns first")
        
        patterns = self.pattern_analysis
        values = patterns['raw_values']
        
        # Generate predictions using multiple approaches and combine them
        predictions = []
        
        # 1. Trend-based predictions
        trend_predictions = self._generate_trend_predictions(patterns, steps)
        
        # 2. Seasonal-based predictions
        seasonal_predictions = self._generate_seasonal_predictions(patterns, steps)
        
        # 3. Autoregressive predictions
        ar_predictions = self._generate_autoregressive_predictions(patterns, steps)
        
        # 4. Statistical continuation predictions
        stat_predictions = self._generate_statistical_continuation(patterns, steps)
        
        # Combine predictions using weighted ensemble
        combined_predictions = self._combine_predictions(
            trend_predictions, seasonal_predictions, ar_predictions, stat_predictions, patterns
        )
        
        # Apply smoothing to ensure continuity
        smoothed_predictions = self._apply_advanced_smoothing(values, combined_predictions, patterns)
        
        # Generate confidence intervals
        confidence_intervals = self._generate_confidence_intervals(
            smoothed_predictions, patterns, confidence_level
        )
        
        # Ensure statistical properties are maintained
        final_predictions = self._ensure_statistical_consistency(
            smoothed_predictions, patterns, values
        )
        
        return {
            'predictions': final_predictions,
            'confidence_intervals': confidence_intervals,
            'prediction_quality': self._assess_prediction_quality(final_predictions, patterns),
            'method_weights': self._get_method_weights(patterns)
        }
    
    def _generate_trend_predictions(self, patterns, steps):
        """Generate predictions based on trend analysis"""
        trend_info = patterns['trend']
        stats = patterns['statistics']
        values = patterns['raw_values']
        
        # Use the most appropriate trend method based on data characteristics
        if trend_info['direction_consistency'] > 0.7:
            # Strong consistent trend - use linear extrapolation
            slope = trend_info['recent_trend']
            predictions = [values[-1] + slope * (i + 1) for i in range(steps)]
        elif trend_info['trend_strength'] > 0.5:
            # Moderate trend - use polynomial extrapolation
            coeffs = trend_info['polynomial_coeffs']
            n = len(values)
            predictions = []
            for i in range(steps):
                pred = sum(coeffs[j] * (n + i) ** (len(coeffs) - 1 - j) for j in range(len(coeffs)))
                predictions.append(pred)
        else:
            # Weak trend - use moving average continuation
            ma_short = trend_info['ma_short']
            last_ma = ma_short[-1] if not np.isnan(ma_short[-1]) else values[-1]
            predictions = [last_ma] * steps
        
        return np.array(predictions)
    
    def _generate_seasonal_predictions(self, patterns, steps):
        """Generate predictions based on seasonal patterns"""
        seasonal_info = patterns['seasonality']
        values = patterns['raw_values']
        
        if seasonal_info['has_seasonality'] and seasonal_info['seasonal_period']:
            # Use seasonal pattern
            seasonal_component = seasonal_info['seasonal_component']
            trend_component = seasonal_info['trend_component']
            
            period = seasonal_info['seasonal_period']
            
            # Extend seasonal pattern
            seasonal_extension = []
            for i in range(steps):
                seasonal_idx = (len(seasonal_component) - period + (i % period)) % len(seasonal_component)
                seasonal_extension.append(seasonal_component[seasonal_idx])
            
            # Extend trend
            trend_extension = []
            trend_slope = np.polyfit(np.arange(len(trend_component)), trend_component, 1)[0]
            for i in range(steps):
                trend_extension.append(trend_component[-1] + trend_slope * (i + 1))
            
            predictions = np.array(seasonal_extension) + np.array(trend_extension)
        else:
            # No seasonality - use simple continuation
            predictions = np.full(steps, values[-1])
        
        return predictions
    
    def _generate_autoregressive_predictions(self, patterns, steps):
        """Generate predictions using autoregressive approach"""
        values = patterns['raw_values']
        autocorr_info = patterns['autocorrelation']
        
        # Use autocorrelation to predict next values
        persistence = autocorr_info['persistence']
        
        predictions = []
        current_value = values[-1]
        
        # Simple AR(1) model
        mean_value = patterns['statistics']['mean']
        
        for i in range(steps):
            # AR(1): X_t = φ * X_{t-1} + (1-φ) * μ + ε
            next_value = persistence * current_value + (1 - persistence) * mean_value
            predictions.append(next_value)
            current_value = next_value
        
        return np.array(predictions)
    
    def _generate_statistical_continuation(self, patterns, steps):
        """Generate predictions that maintain statistical properties"""
        stats = patterns['statistics']
        values = patterns['raw_values']
        volatility = patterns['volatility']
        
        # Generate predictions that maintain mean and variance
        predictions = []
        current_value = values[-1]
        
        # Calculate mean reversion strength
        mean_reversion = 0.1  # How quickly to revert to mean
        
        for i in range(steps):
            # Mean reversion with noise
            mean_pull = (stats['mean'] - current_value) * mean_reversion
            
            # Add controlled noise
            noise = np.random.normal(0, volatility['noise_level'] * 0.1)
            
            next_value = current_value + mean_pull + noise
            predictions.append(next_value)
            current_value = next_value
        
        return np.array(predictions)
    
    def _combine_predictions(self, trend_pred, seasonal_pred, ar_pred, stat_pred, patterns):
        """Combine predictions using weighted ensemble"""
        # Calculate weights based on data characteristics
        weights = self._calculate_method_weights(patterns)
        
        # Combine predictions
        combined = (
            weights['trend'] * trend_pred +
            weights['seasonal'] * seasonal_pred +
            weights['autoregressive'] * ar_pred +
            weights['statistical'] * stat_pred
        )
        
        return combined
    
    def _calculate_method_weights(self, patterns):
        """Calculate weights for different prediction methods"""
        trend_strength = patterns['trend']['trend_strength']
        seasonal_strength = patterns['seasonality']['seasonal_strength']
        persistence = patterns['autocorrelation']['persistence']
        stability = patterns['stability']['stability_score']
        
        # Base weights
        weights = {
            'trend': 0.25,
            'seasonal': 0.25,
            'autoregressive': 0.25,
            'statistical': 0.25
        }
        
        # Adjust weights based on data characteristics
        if trend_strength > 0.5:
            weights['trend'] *= (1 + trend_strength)
        
        if seasonal_strength > 0.3:
            weights['seasonal'] *= (1 + seasonal_strength)
        
        if persistence > 0.5:
            weights['autoregressive'] *= (1 + persistence)
        
        if stability > 0.7:
            weights['statistical'] *= (1 + stability)
        
        # Normalize weights
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        return weights
    
    def _apply_advanced_smoothing(self, historical_values, predictions, patterns):
        """Apply advanced smoothing for continuity"""
        # Ensure smooth transition from historical to predicted
        transition_length = min(5, len(predictions))
        
        # Create transition weights
        transition_weights = np.linspace(0, 1, transition_length)
        
        # Calculate expected next value from historical data
        if len(historical_values) >= 3:
            # Use local trend for transition
            recent_trend = np.polyfit(np.arange(3), historical_values[-3:], 1)[0]
            expected_next = historical_values[-1] + recent_trend
        else:
            expected_next = historical_values[-1]
        
        # Apply transition smoothing
        smoothed_predictions = predictions.copy()
        for i in range(min(transition_length, len(predictions))):
            weight = transition_weights[i]
            smoothed_predictions[i] = (1 - weight) * expected_next + weight * predictions[i]
        
        # Apply additional smoothing using moving average
        window_size = min(3, len(smoothed_predictions))
        if window_size > 1:
            for i in range(window_size, len(smoothed_predictions)):
                smoothed_predictions[i] = np.mean(smoothed_predictions[i-window_size:i+1])
        
        return smoothed_predictions
    
    def _generate_confidence_intervals(self, predictions, patterns, confidence_level):
        """Generate confidence intervals for predictions"""
        volatility = patterns['volatility']['volatility']
        stats = patterns['statistics']
        
        # Calculate confidence interval width
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        # Increasing uncertainty with distance
        intervals = []
        for i, pred in enumerate(predictions):
            uncertainty = volatility * np.sqrt(i + 1)  # Increasing uncertainty
            margin = z_score * uncertainty
            
            intervals.append({
                'lower': pred - margin,
                'upper': pred + margin,
                'uncertainty': uncertainty
            })
        
        return intervals
    
    def _ensure_statistical_consistency(self, predictions, patterns, historical_values):
        """Ensure predictions maintain statistical consistency"""
        stats = patterns['statistics']
        
        # Check if predictions maintain reasonable bounds
        historical_range = stats['max'] - stats['min']
        reasonable_bounds = {
            'min': stats['min'] - 0.5 * historical_range,
            'max': stats['max'] + 0.5 * historical_range
        }
        
        # Apply soft bounds
        adjusted_predictions = []
        for pred in predictions:
            if pred < reasonable_bounds['min']:
                # Soft lower bound
                adjusted_predictions.append(reasonable_bounds['min'] + 0.1 * (pred - reasonable_bounds['min']))
            elif pred > reasonable_bounds['max']:
                # Soft upper bound
                adjusted_predictions.append(reasonable_bounds['max'] + 0.1 * (pred - reasonable_bounds['max']))
            else:
                adjusted_predictions.append(pred)
        
        return np.array(adjusted_predictions)
    
    def _assess_prediction_quality(self, predictions, patterns):
        """Assess the quality of generated predictions"""
        stats = patterns['statistics']
        
        # Calculate various quality metrics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        # Mean preservation (how close prediction mean is to historical mean)
        mean_preservation = 1 - abs(pred_mean - stats['mean']) / (stats['std'] + 1e-8)
        mean_preservation = max(0, min(1, mean_preservation))
        
        # Variance preservation
        variance_preservation = 1 - abs(pred_std - stats['std']) / (stats['std'] + 1e-8)
        variance_preservation = max(0, min(1, variance_preservation))
        
        # Smoothness (low volatility in predictions)
        if len(predictions) > 1:
            pred_volatility = np.std(np.diff(predictions))
            hist_volatility = patterns['volatility']['volatility']
            smoothness = 1 - abs(pred_volatility - hist_volatility) / (hist_volatility + 1e-8)
            smoothness = max(0, min(1, smoothness))
        else:
            smoothness = 1.0
        
        # Overall quality score
        quality_score = (mean_preservation + variance_preservation + smoothness) / 3
        
        return {
            'overall_quality': quality_score,
            'mean_preservation': mean_preservation,
            'variance_preservation': variance_preservation,
            'smoothness': smoothness
        }
    
    def _get_method_weights(self, patterns):
        """Get the weights used for different prediction methods"""
        return self._calculate_method_weights(patterns)