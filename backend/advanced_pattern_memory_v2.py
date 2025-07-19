"""
Advanced Pattern Memory System v2
Enhanced pattern learning and memory system for superior historical pattern following
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdvancedPatternMemoryV2:
    """
    Advanced Pattern Memory System that learns, stores, and recalls multiple types of patterns
    for superior historical pattern following in predictions
    """
    
    def __init__(self, max_patterns: int = 100, pattern_similarity_threshold: float = 0.7):
        self.max_patterns = max_patterns
        self.pattern_similarity_threshold = pattern_similarity_threshold
        
        # Pattern storage
        self.stored_patterns = {
            'cyclical': [],
            'trending': [],
            'seasonal': [],
            'volatility': [],
            'local': [],
            'complex': []
        }
        
        # Pattern analysis results
        self.pattern_analysis_cache = {}
        self.pattern_weights = {}
        self.pattern_performance = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_rate = 0.05
        self.pattern_decay_rate = 0.02
        self.confidence_threshold = 0.6
        
        # Enhanced parameters for better pattern following
        self.variability_preservation_factor = 0.9
        self.trend_momentum_factor = 0.8
        self.cyclical_strength_factor = 0.85
        self.noise_injection_factor = 0.1
        
    def learn_patterns_from_data(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive pattern learning from historical data
        
        Args:
            data: Historical time series data
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Dictionary containing learned patterns and their characteristics
        """
        try:
            # Ensure data is 1D
            if data.ndim > 1:
                data = data.flatten()
            
            # Remove NaN and infinite values
            data = data[np.isfinite(data)]
            
            if len(data) < 10:
                logger.warning("Insufficient data for pattern learning")
                return self._get_minimal_patterns(data)
            
            learned_patterns = {}
            
            # 1. Learn cyclical patterns
            learned_patterns['cyclical'] = self._learn_cyclical_patterns(data)
            
            # 2. Learn trending patterns
            learned_patterns['trending'] = self._learn_trending_patterns(data)
            
            # 3. Learn seasonal patterns
            learned_patterns['seasonal'] = self._learn_seasonal_patterns(data, timestamps)
            
            # 4. Learn volatility patterns
            learned_patterns['volatility'] = self._learn_volatility_patterns(data)
            
            # 5. Learn local patterns
            learned_patterns['local'] = self._learn_local_patterns(data)
            
            # 6. Learn complex patterns (combinations)
            learned_patterns['complex'] = self._learn_complex_patterns(data)
            
            # 7. Calculate pattern weights and confidence
            pattern_confidence = self._calculate_pattern_confidence(data, learned_patterns)
            
            # 8. Store patterns in memory
            self._store_patterns_in_memory(learned_patterns, pattern_confidence)
            
            # 9. Calculate pattern characteristics for future use
            pattern_characteristics = self._calculate_pattern_characteristics(data, learned_patterns)
            
            return {
                'learned_patterns': learned_patterns,
                'pattern_confidence': pattern_confidence,
                'pattern_characteristics': pattern_characteristics,
                'data_statistics': self._calculate_data_statistics(data)
            }
            
        except Exception as e:
            logger.error(f"Error in pattern learning: {e}")
            return self._get_minimal_patterns(data)
    
    def _learn_cyclical_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Learn cyclical patterns using FFT and autocorrelation"""
        try:
            cyclical_patterns = {}
            
            # 1. FFT-based frequency analysis
            fft_result = fft(data)
            frequencies = fftfreq(len(data))
            magnitude = np.abs(fft_result)
            
            # Find dominant frequencies
            dominant_freq_indices = find_peaks(magnitude[1:len(magnitude)//2], height=np.max(magnitude) * 0.1)[0] + 1
            dominant_frequencies = frequencies[dominant_freq_indices]
            dominant_magnitudes = magnitude[dominant_freq_indices]
            
            cyclical_patterns['frequencies'] = dominant_frequencies.tolist()
            cyclical_patterns['magnitudes'] = dominant_magnitudes.tolist()
            
            # 2. Autocorrelation analysis
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find peaks in autocorrelation (periodicity)
            autocorr_peaks = find_peaks(autocorr[1:], height=0.3, distance=5)[0] + 1
            cyclical_patterns['periods'] = autocorr_peaks.tolist()
            cyclical_patterns['period_strengths'] = autocorr[autocorr_peaks].tolist()
            
            # 3. Identify cyclical components
            cyclical_components = []
            for period in autocorr_peaks[:3]:  # Top 3 periods
                if period < len(data) // 3:  # Reasonable period length
                    # Extract cyclical component
                    phase_data = []
                    for i in range(0, len(data), period):
                        if i + period <= len(data):
                            phase_data.append(data[i:i+period])
                    
                    if phase_data:
                        avg_cycle = np.mean(phase_data, axis=0)
                        cyclical_components.append({
                            'period': period,
                            'average_cycle': avg_cycle.tolist(),
                            'amplitude': np.std(avg_cycle),
                            'phase_shift': 0
                        })
            
            cyclical_patterns['components'] = cyclical_components
            cyclical_patterns['overall_cyclical_strength'] = np.mean(autocorr[autocorr_peaks]) if autocorr_peaks.size > 0 else 0.0
            
            return cyclical_patterns
            
        except Exception as e:
            logger.error(f"Error in cyclical pattern learning: {e}")
            return {'components': [], 'overall_cyclical_strength': 0.0}
    
    def _learn_trending_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Learn trending patterns at multiple timescales"""
        try:
            trending_patterns = {}
            
            # 1. Overall trend analysis
            x = np.arange(len(data))
            
            # Linear trend
            linear_coeff = np.polyfit(x, data, 1)
            linear_trend = linear_coeff[0]
            linear_r2 = r2_score(data, np.polyval(linear_coeff, x))
            
            # Quadratic trend
            quad_coeff = np.polyfit(x, data, 2)
            quad_r2 = r2_score(data, np.polyval(quad_coeff, x))
            
            trending_patterns['linear'] = {
                'slope': linear_trend,
                'intercept': linear_coeff[1],
                'r2': linear_r2
            }
            
            trending_patterns['quadratic'] = {
                'coefficients': quad_coeff.tolist(),
                'r2': quad_r2
            }
            
            # 2. Multi-scale trend analysis
            scales = [len(data)//8, len(data)//4, len(data)//2]
            multiscale_trends = []
            
            for scale in scales:
                if scale >= 3:
                    # Sliding window trend analysis
                    trends = []
                    for i in range(0, len(data) - scale + 1, scale//2):
                        window_data = data[i:i+scale]
                        window_x = np.arange(len(window_data))
                        if len(window_data) >= 3:
                            slope = np.polyfit(window_x, window_data, 1)[0]
                            trends.append(slope)
                    
                    if trends:
                        multiscale_trends.append({
                            'scale': scale,
                            'trends': trends,
                            'avg_trend': np.mean(trends),
                            'trend_variability': np.std(trends)
                        })
            
            trending_patterns['multiscale'] = multiscale_trends
            
            # 3. Trend change points
            # Use simple derivative to find trend changes
            smoothed_data = savgol_filter(data, min(21, len(data)//2 if len(data) > 4 else 3), 2)
            derivative = np.gradient(smoothed_data)
            second_derivative = np.gradient(derivative)
            
            # Find significant changes in derivative
            change_points = find_peaks(np.abs(second_derivative), height=np.std(second_derivative))[0]
            
            trending_patterns['change_points'] = change_points.tolist()
            trending_patterns['trend_stability'] = 1.0 / (1.0 + len(change_points) / len(data))
            
            return trending_patterns
            
        except Exception as e:
            logger.error(f"Error in trending pattern learning: {e}")
            return {'linear': {'slope': 0, 'r2': 0}, 'trend_stability': 1.0}
    
    def _learn_seasonal_patterns(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Learn seasonal patterns if timestamps are available"""
        try:
            seasonal_patterns = {}
            
            if timestamps is not None and len(timestamps) == len(data):
                # Convert timestamps if needed
                if not isinstance(timestamps[0], (pd.Timestamp, datetime)):
                    timestamps = pd.to_datetime(timestamps)
                
                # Extract time features
                df = pd.DataFrame({
                    'value': data,
                    'timestamp': timestamps
                })
                
                df['hour'] = df['timestamp'].dt.hour
                df['day'] = df['timestamp'].dt.day
                df['month'] = df['timestamp'].dt.month
                df['weekday'] = df['timestamp'].dt.weekday
                
                # Hourly patterns
                if len(df['hour'].unique()) > 1:
                    hourly_means = df.groupby('hour')['value'].mean().to_dict()
                    seasonal_patterns['hourly'] = hourly_means
                
                # Daily patterns
                if len(df['day'].unique()) > 1:
                    daily_means = df.groupby('day')['value'].mean().to_dict()
                    seasonal_patterns['daily'] = daily_means
                
                # Weekly patterns
                if len(df['weekday'].unique()) > 1:
                    weekly_means = df.groupby('weekday')['value'].mean().to_dict()
                    seasonal_patterns['weekly'] = weekly_means
                
                # Monthly patterns
                if len(df['month'].unique()) > 1:
                    monthly_means = df.groupby('month')['value'].mean().to_dict()
                    seasonal_patterns['monthly'] = monthly_means
                    
            else:
                # If no timestamps, try to infer seasonal patterns from data structure
                # Assume regular intervals and look for repeating patterns
                data_length = len(data)
                
                # Check for common seasonal periods
                for period in [24, 168, 720]:  # hourly, weekly, monthly patterns
                    if data_length >= period * 2:
                        seasonal_component = []
                        for i in range(period):
                            values = data[i::period]
                            if len(values) > 1:
                                seasonal_component.append(np.mean(values))
                            else:
                                seasonal_component.append(data[i] if i < len(data) else np.mean(data))
                        
                        seasonal_patterns[f'period_{period}'] = {
                            'values': seasonal_component,
                            'strength': np.std(seasonal_component) / (np.std(data) + 1e-10)
                        }
            
            return seasonal_patterns
            
        except Exception as e:
            logger.error(f"Error in seasonal pattern learning: {e}")
            return {}
    
    def _learn_volatility_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Learn volatility and variability patterns"""
        try:
            volatility_patterns = {}
            
            # 1. Calculate returns/changes
            returns = np.diff(data)
            abs_returns = np.abs(returns)
            
            # 2. Volatility statistics
            volatility_patterns['overall_volatility'] = np.std(returns)
            volatility_patterns['mean_absolute_change'] = np.mean(abs_returns)
            volatility_patterns['max_change'] = np.max(abs_returns)
            volatility_patterns['volatility_of_volatility'] = np.std(abs_returns)
            
            # 3. Volatility clustering (GARCH-like analysis)
            # Rolling volatility
            window_size = min(10, len(returns)//3)
            if window_size >= 2:
                rolling_volatility = []
                for i in range(len(returns) - window_size + 1):
                    window_vol = np.std(returns[i:i+window_size])
                    rolling_volatility.append(window_vol)
                
                volatility_patterns['volatility_clustering'] = {
                    'rolling_volatility': rolling_volatility,
                    'avg_rolling_volatility': np.mean(rolling_volatility),
                    'volatility_persistence': np.corrcoef(rolling_volatility[:-1], rolling_volatility[1:])[0, 1] if len(rolling_volatility) > 1 else 0
                }
            
            # 4. Distribution analysis
            volatility_patterns['returns_skewness'] = stats.skew(returns)
            volatility_patterns['returns_kurtosis'] = stats.kurtosis(returns)
            
            # 5. Extreme value analysis
            threshold_95 = np.percentile(abs_returns, 95)
            extreme_events = abs_returns > threshold_95
            volatility_patterns['extreme_event_frequency'] = np.sum(extreme_events) / len(returns)
            
            return volatility_patterns
            
        except Exception as e:
            logger.error(f"Error in volatility pattern learning: {e}")
            return {'overall_volatility': np.std(data)}
    
    def _learn_local_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Learn local patterns and motifs"""
        try:
            local_patterns = {}
            
            # 1. Local extrema patterns
            peaks, _ = find_peaks(data, distance=3)
            troughs, _ = find_peaks(-data, distance=3)
            
            local_patterns['peaks'] = {
                'indices': peaks.tolist(),
                'values': data[peaks].tolist() if len(peaks) > 0 else [],
                'frequency': len(peaks) / len(data)
            }
            
            local_patterns['troughs'] = {
                'indices': troughs.tolist(),
                'values': data[troughs].tolist() if len(troughs) > 0 else [],
                'frequency': len(troughs) / len(data)
            }
            
            # 2. Local patterns around extrema
            pattern_length = min(10, len(data)//5)
            if pattern_length >= 3:
                peak_patterns = []
                for peak in peaks:
                    start = max(0, peak - pattern_length//2)
                    end = min(len(data), peak + pattern_length//2 + 1)
                    pattern = data[start:end]
                    if len(pattern) >= 3:
                        peak_patterns.append(pattern.tolist())
                
                trough_patterns = []
                for trough in troughs:
                    start = max(0, trough - pattern_length//2)
                    end = min(len(data), trough + pattern_length//2 + 1)
                    pattern = data[start:end]
                    if len(pattern) >= 3:
                        trough_patterns.append(pattern.tolist())
                
                local_patterns['peak_patterns'] = peak_patterns
                local_patterns['trough_patterns'] = trough_patterns
            
            # 3. Consecutive rising/falling patterns
            diff_data = np.diff(data)
            rising_streaks = []
            falling_streaks = []
            
            current_streak = 0
            for i, diff in enumerate(diff_data):
                if diff > 0:
                    if current_streak >= 0:
                        current_streak += 1
                    else:
                        if current_streak < -1:
                            falling_streaks.append(-current_streak)
                        current_streak = 1
                elif diff < 0:
                    if current_streak <= 0:
                        current_streak -= 1
                    else:
                        if current_streak > 1:
                            rising_streaks.append(current_streak)
                        current_streak = -1
                else:
                    if current_streak > 1:
                        rising_streaks.append(current_streak)
                    elif current_streak < -1:
                        falling_streaks.append(-current_streak)
                    current_streak = 0
            
            # Add final streak
            if current_streak > 1:
                rising_streaks.append(current_streak)
            elif current_streak < -1:
                falling_streaks.append(-current_streak)
            
            local_patterns['streaks'] = {
                'rising': rising_streaks,
                'falling': falling_streaks,
                'avg_rising_length': np.mean(rising_streaks) if rising_streaks else 0,
                'avg_falling_length': np.mean(falling_streaks) if falling_streaks else 0
            }
            
            return local_patterns
            
        except Exception as e:
            logger.error(f"Error in local pattern learning: {e}")
            return {}
    
    def _learn_complex_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Learn complex patterns that combine multiple pattern types"""
        try:
            complex_patterns = {}
            
            # 1. Trend + Cyclical combinations
            # Detrend data and analyze residual patterns
            x = np.arange(len(data))
            trend_line = np.polyval(np.polyfit(x, data, 1), x)
            detrended = data - trend_line
            
            # Analyze detrended data for cyclical patterns
            detrended_fft = fft(detrended)
            detrended_magnitude = np.abs(detrended_fft)
            dominant_cycles = find_peaks(detrended_magnitude[1:len(detrended_magnitude)//2], 
                                       height=np.max(detrended_magnitude) * 0.1)[0] + 1
            
            complex_patterns['trend_cycle_combination'] = {
                'trend_component': trend_line.tolist(),
                'cyclical_component': detrended.tolist(),
                'dominant_cycle_frequencies': dominant_cycles.tolist()
            }
            
            # 2. Volatility clustering with trend
            returns = np.diff(data)
            abs_returns = np.abs(returns)
            
            # Correlation between trend and volatility
            if len(returns) > 10:
                trend_returns = np.diff(trend_line)
                vol_trend_corr = np.corrcoef(abs_returns, trend_returns)[0, 1] if len(trend_returns) == len(abs_returns) else 0
                complex_patterns['volatility_trend_correlation'] = vol_trend_corr
            
            # 3. Multi-scale decomposition
            # Simple multi-scale analysis
            scales = [3, 7, 15]
            multiscale_components = {}
            
            for scale in scales:
                if scale < len(data)//2:
                    # Simple moving average decomposition
                    smoothed = np.convolve(data, np.ones(scale)/scale, mode='valid')
                    
                    # Pad to match original length
                    pad_left = (len(data) - len(smoothed)) // 2
                    pad_right = len(data) - len(smoothed) - pad_left
                    smoothed = np.pad(smoothed, (pad_left, pad_right), mode='edge')
                    
                    residual = data - smoothed
                    
                    multiscale_components[f'scale_{scale}'] = {
                        'smooth_component': smoothed.tolist(),
                        'residual_component': residual.tolist(),
                        'residual_energy': np.sum(residual**2)
                    }
            
            complex_patterns['multiscale_decomposition'] = multiscale_components
            
            return complex_patterns
            
        except Exception as e:
            logger.error(f"Error in complex pattern learning: {e}")
            return {}
    
    def _calculate_pattern_confidence(self, data: np.ndarray, patterns: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for different pattern types"""
        try:
            confidence_scores = {}
            
            # Cyclical pattern confidence
            cyclical = patterns.get('cyclical', {})
            cyclical_strength = cyclical.get('overall_cyclical_strength', 0)
            confidence_scores['cyclical'] = min(1.0, cyclical_strength * 2)
            
            # Trending pattern confidence
            trending = patterns.get('trending', {})
            linear_r2 = trending.get('linear', {}).get('r2', 0)
            confidence_scores['trending'] = max(0, linear_r2)
            
            # Volatility pattern confidence (based on consistency)
            volatility = patterns.get('volatility', {})
            vol_persistence = volatility.get('volatility_clustering', {}).get('volatility_persistence', 0)
            confidence_scores['volatility'] = max(0, vol_persistence)
            
            # Local pattern confidence (based on pattern richness)
            local = patterns.get('local', {})
            peak_freq = local.get('peaks', {}).get('frequency', 0)
            trough_freq = local.get('troughs', {}).get('frequency', 0)
            pattern_richness = (peak_freq + trough_freq) * 10  # Scale up for reasonable confidence
            confidence_scores['local'] = min(1.0, pattern_richness)
            
            # Seasonal pattern confidence
            seasonal = patterns.get('seasonal', {})
            seasonal_count = len(seasonal)
            confidence_scores['seasonal'] = min(1.0, seasonal_count * 0.25)
            
            # Complex pattern confidence (average of components)
            complex_patterns = patterns.get('complex', {})
            if complex_patterns:
                vol_trend_corr = abs(complex_patterns.get('volatility_trend_correlation', 0))
                confidence_scores['complex'] = min(1.0, vol_trend_corr)
            else:
                confidence_scores['complex'] = 0.0
            
            # Overall confidence (weighted average)
            weights = {'cyclical': 0.2, 'trending': 0.25, 'volatility': 0.15, 
                      'local': 0.2, 'seasonal': 0.1, 'complex': 0.1}
            
            overall_confidence = sum(confidence_scores.get(k, 0) * weights[k] for k in weights)
            confidence_scores['overall'] = overall_confidence
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Error in pattern confidence calculation: {e}")
            return {'overall': 0.5}
    
    def _store_patterns_in_memory(self, patterns: Dict[str, Any], confidence: Dict[str, float]):
        """Store learned patterns in memory for future use"""
        try:
            for pattern_type, pattern_data in patterns.items():
                if pattern_type in self.stored_patterns:
                    # Add pattern with timestamp and confidence
                    pattern_entry = {
                        'pattern_data': pattern_data,
                        'confidence': confidence.get(pattern_type, 0.5),
                        'timestamp': datetime.now().isoformat(),
                        'usage_count': 0
                    }
                    
                    self.stored_patterns[pattern_type].append(pattern_entry)
                    
                    # Limit memory size
                    if len(self.stored_patterns[pattern_type]) > self.max_patterns:
                        # Remove oldest patterns
                        self.stored_patterns[pattern_type] = self.stored_patterns[pattern_type][-self.max_patterns:]
            
        except Exception as e:
            logger.error(f"Error storing patterns in memory: {e}")
    
    def _calculate_pattern_characteristics(self, data: np.ndarray, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall pattern characteristics for future prediction use"""
        try:
            characteristics = {}
            
            # Basic statistics
            characteristics['statistical'] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'range': float(np.max(data) - np.min(data)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data))
            }
            
            # Pattern-specific characteristics
            cyclical = patterns.get('cyclical', {})
            if cyclical.get('components'):
                dominant_period = cyclical['components'][0]['period']
                dominant_amplitude = cyclical['components'][0]['amplitude']
                characteristics['dominant_cycle'] = {
                    'period': dominant_period,
                    'amplitude': dominant_amplitude
                }
            
            trending = patterns.get('trending', {})
            linear_trend = trending.get('linear', {})
            characteristics['trend'] = {
                'slope': linear_trend.get('slope', 0),
                'strength': linear_trend.get('r2', 0)
            }
            
            volatility = patterns.get('volatility', {})
            characteristics['volatility'] = {
                'overall': volatility.get('overall_volatility', np.std(np.diff(data))),
                'clustering': volatility.get('volatility_clustering', {}).get('volatility_persistence', 0)
            }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error calculating pattern characteristics: {e}")
            return {'statistical': {'mean': float(np.mean(data)), 'std': float(np.std(data))}}
    
    def _calculate_data_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate basic data statistics"""
        try:
            return {
                'length': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'range': float(np.max(data) - np.min(data)),
                'cv': float(np.std(data) / (np.mean(data) + 1e-10))  # Coefficient of variation
            }
        except Exception as e:
            logger.error(f"Error calculating data statistics: {e}")
            return {'length': len(data), 'mean': 0.0, 'std': 1.0}
    
    def _get_minimal_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Get minimal patterns when full analysis fails"""
        if len(data) == 0:
            return {
                'learned_patterns': {},
                'pattern_confidence': {'overall': 0.0},
                'pattern_characteristics': {'statistical': {'mean': 0.0, 'std': 1.0}},
                'data_statistics': {'length': 0, 'mean': 0.0, 'std': 1.0}
            }
        
        return {
            'learned_patterns': {
                'trending': {'linear': {'slope': 0, 'r2': 0}},
                'volatility': {'overall_volatility': float(np.std(data))}
            },
            'pattern_confidence': {'overall': 0.3},
            'pattern_characteristics': self._calculate_pattern_characteristics(data, {}),
            'data_statistics': self._calculate_data_statistics(data)
        }
    
    def generate_pattern_based_predictions(self, historical_data: np.ndarray, steps: int = 30,
                                         pattern_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate predictions based on learned patterns
        
        Args:
            historical_data: Historical time series data
            steps: Number of prediction steps
            pattern_weights: Optional weights for different pattern types
            
        Returns:
            Pattern-based predictions with confidence scores
        """
        try:
            # Learn patterns from current data
            pattern_analysis = self.learn_patterns_from_data(historical_data)
            patterns = pattern_analysis['learned_patterns']
            confidence = pattern_analysis['pattern_confidence']
            
            # Set default pattern weights if not provided
            if pattern_weights is None:
                pattern_weights = {
                    'cyclical': 0.3,
                    'trending': 0.25,
                    'volatility': 0.15,
                    'local': 0.2,
                    'seasonal': 0.05,
                    'complex': 0.05
                }
            
            # Generate predictions using different pattern types
            predictions = {}
            
            # 1. Cyclical predictions
            predictions['cyclical'] = self._generate_cyclical_predictions(
                historical_data, steps, patterns.get('cyclical', {})
            )
            
            # 2. Trending predictions
            predictions['trending'] = self._generate_trending_predictions(
                historical_data, steps, patterns.get('trending', {})
            )
            
            # 3. Volatility-aware predictions
            predictions['volatility'] = self._generate_volatility_predictions(
                historical_data, steps, patterns.get('volatility', {})
            )
            
            # 4. Local pattern predictions
            predictions['local'] = self._generate_local_predictions(
                historical_data, steps, patterns.get('local', {})
            )
            
            # 5. Seasonal predictions
            predictions['seasonal'] = self._generate_seasonal_predictions(
                historical_data, steps, patterns.get('seasonal', {})
            )
            
            # 6. Complex pattern predictions
            predictions['complex'] = self._generate_complex_predictions(
                historical_data, steps, patterns.get('complex', {})
            )
            
            # 7. Combine predictions using weighted ensemble
            final_predictions = self._combine_pattern_predictions(
                predictions, pattern_weights, confidence
            )
            
            # 8. Apply pattern-based enhancements
            enhanced_predictions = self._apply_pattern_enhancements(
                final_predictions, historical_data, patterns
            )
            
            return {
                'predictions': enhanced_predictions.tolist(),
                'pattern_analysis': pattern_analysis,
                'individual_predictions': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                         for k, v in predictions.items()},
                'pattern_weights': pattern_weights,
                'quality_score': self._calculate_prediction_quality(enhanced_predictions, historical_data, patterns)
            }
            
        except Exception as e:
            logger.error(f"Error in pattern-based prediction generation: {e}")
            return self._generate_fallback_predictions(historical_data, steps)
    
    def _generate_cyclical_predictions(self, data: np.ndarray, steps: int, cyclical_patterns: Dict) -> np.ndarray:
        """Generate predictions based on cyclical patterns"""
        try:
            if not cyclical_patterns or not cyclical_patterns.get('components'):
                return np.full(steps, np.mean(data))
            
            predictions = np.zeros(steps)
            base_value = np.mean(data[-min(10, len(data)):])  # Recent average
            
            # Use the most dominant cyclical component
            dominant_component = cyclical_patterns['components'][0]
            period = dominant_component['period']
            cycle_pattern = np.array(dominant_component['average_cycle'])
            amplitude = dominant_component['amplitude']
            
            # Generate cyclical predictions
            for i in range(steps):
                cycle_position = i % period
                if cycle_position < len(cycle_pattern):
                    cycle_value = cycle_pattern[cycle_position]
                    # Normalize and scale
                    cycle_contribution = (cycle_value - np.mean(cycle_pattern)) * amplitude
                    predictions[i] = base_value + cycle_contribution * 0.8  # Reduce amplitude slightly
                else:
                    predictions[i] = base_value
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in cyclical prediction generation: {e}")
            return np.full(steps, np.mean(data))
    
    def _generate_trending_predictions(self, data: np.ndarray, steps: int, trending_patterns: Dict) -> np.ndarray:
        """Generate predictions based on trending patterns"""
        try:
            if not trending_patterns:
                return np.full(steps, np.mean(data))
            
            linear_trend = trending_patterns.get('linear', {})
            slope = linear_trend.get('slope', 0)
            r2 = linear_trend.get('r2', 0)
            
            # Start from last value
            start_value = data[-1]
            predictions = np.zeros(steps)
            
            # Apply trend with decay for stability
            for i in range(steps):
                # Apply trend decay to prevent unlimited extrapolation
                trend_decay = np.exp(-i * 0.02)  # Gentle decay
                trend_contribution = slope * (i + 1) * trend_decay * r2
                predictions[i] = start_value + trend_contribution
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in trending prediction generation: {e}")
            return np.full(steps, np.mean(data))
    
    def _generate_volatility_predictions(self, data: np.ndarray, steps: int, volatility_patterns: Dict) -> np.ndarray:
        """Generate predictions incorporating volatility patterns"""
        try:
            if not volatility_patterns:
                return np.full(steps, np.mean(data))
            
            overall_vol = volatility_patterns.get('overall_volatility', np.std(np.diff(data)))
            
            # Start from last value
            start_value = data[-1]
            predictions = np.zeros(steps)
            predictions[0] = start_value
            
            # Generate random walk with historical volatility
            for i in range(1, steps):
                # Add controlled random noise based on historical volatility
                noise = np.random.normal(0, overall_vol * 0.5)  # Reduce volatility for stability
                predictions[i] = predictions[i-1] + noise
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in volatility prediction generation: {e}")
            return np.full(steps, np.mean(data))
    
    def _generate_local_predictions(self, data: np.ndarray, steps: int, local_patterns: Dict) -> np.ndarray:
        """Generate predictions based on local patterns"""
        try:
            if not local_patterns:
                return np.full(steps, np.mean(data))
            
            # Use recent local patterns
            recent_data = data[-min(20, len(data)):]
            predictions = np.zeros(steps)
            
            # Simple continuation of recent pattern
            if len(recent_data) >= 3:
                # Use last few values to establish local trend
                last_trend = np.mean(np.diff(recent_data[-3:]))
                start_value = data[-1]
                
                for i in range(steps):
                    # Apply local trend with decay
                    trend_decay = np.exp(-i * 0.05)
                    predictions[i] = start_value + last_trend * (i + 1) * trend_decay
            else:
                predictions[:] = np.mean(data)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in local prediction generation: {e}")
            return np.full(steps, np.mean(data))
    
    def _generate_seasonal_predictions(self, data: np.ndarray, steps: int, seasonal_patterns: Dict) -> np.ndarray:
        """Generate predictions based on seasonal patterns"""
        try:
            if not seasonal_patterns:
                return np.full(steps, np.mean(data))
            
            predictions = np.full(steps, np.mean(data))
            
            # Use the strongest seasonal pattern
            for pattern_name, pattern_data in seasonal_patterns.items():
                if isinstance(pattern_data, dict) and 'values' in pattern_data:
                    values = pattern_data['values']
                    strength = pattern_data.get('strength', 0.5)
                    
                    # Apply seasonal pattern
                    for i in range(steps):
                        season_index = i % len(values)
                        seasonal_contribution = (values[season_index] - np.mean(values)) * strength
                        predictions[i] += seasonal_contribution * 0.5  # Reduce impact
                    break
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in seasonal prediction generation: {e}")
            return np.full(steps, np.mean(data))
    
    def _generate_complex_predictions(self, data: np.ndarray, steps: int, complex_patterns: Dict) -> np.ndarray:
        """Generate predictions based on complex patterns"""
        try:
            if not complex_patterns:
                return np.full(steps, np.mean(data))
            
            # Use trend-cycle combination if available
            trend_cycle = complex_patterns.get('trend_cycle_combination', {})
            if trend_cycle:
                trend_component = np.array(trend_cycle.get('trend_component', []))
                cyclical_component = np.array(trend_cycle.get('cyclical_component', []))
                
                if len(trend_component) > 0 and len(cyclical_component) > 0:
                    # Extrapolate trend
                    trend_slope = np.polyfit(np.arange(len(trend_component)), trend_component, 1)[0]
                    last_trend = trend_component[-1]
                    
                    # Extrapolate cyclical pattern
                    cycle_period = len(cyclical_component) // 3 if len(cyclical_component) > 9 else len(cyclical_component)
                    
                    predictions = np.zeros(steps)
                    for i in range(steps):
                        # Trend component
                        trend_value = last_trend + trend_slope * (i + 1) * 0.5  # Reduce trend strength
                        
                        # Cyclical component
                        cycle_index = i % cycle_period
                        cycle_value = cyclical_component[cycle_index] if cycle_index < len(cyclical_component) else 0
                        
                        predictions[i] = trend_value + cycle_value * 0.3  # Reduce cycle impact
                    
                    return predictions
            
            return np.full(steps, np.mean(data))
            
        except Exception as e:
            logger.error(f"Error in complex prediction generation: {e}")
            return np.full(steps, np.mean(data))
    
    def _combine_pattern_predictions(self, predictions: Dict[str, np.ndarray], 
                                   weights: Dict[str, float], 
                                   confidence: Dict[str, float]) -> np.ndarray:
        """Combine predictions from different pattern types using weighted ensemble"""
        try:
            # Adjust weights by confidence
            adjusted_weights = {}
            for pattern_type, weight in weights.items():
                pattern_confidence = confidence.get(pattern_type, 0.5)
                adjusted_weights[pattern_type] = weight * pattern_confidence
            
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
            else:
                # Equal weights if no confidence
                adjusted_weights = {k: 1.0/len(weights) for k in weights.keys()}
            
            # Combine predictions
            combined = None
            for pattern_type, prediction in predictions.items():
                if isinstance(prediction, np.ndarray) and len(prediction) > 0:
                    weight = adjusted_weights.get(pattern_type, 0)
                    if combined is None:
                        combined = prediction * weight
                    else:
                        combined += prediction * weight
            
            if combined is None:
                # Fallback
                steps = max(len(pred) for pred in predictions.values() if isinstance(pred, np.ndarray))
                combined = np.zeros(steps)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in combining pattern predictions: {e}")
            return np.zeros(30)
    
    def _apply_pattern_enhancements(self, predictions: np.ndarray, 
                                  historical_data: np.ndarray, 
                                  patterns: Dict[str, Any]) -> np.ndarray:
        """Apply pattern-based enhancements to improve prediction quality"""
        try:
            enhanced = predictions.copy()
            
            # 1. Variability preservation
            historical_std = np.std(historical_data)
            prediction_std = np.std(enhanced)
            
            if prediction_std < 0.5 * historical_std:
                # Enhance variability
                enhancement_factor = (historical_std * 0.8) / (prediction_std + 1e-10)
                prediction_mean = np.mean(enhanced)
                enhanced = prediction_mean + (enhanced - prediction_mean) * enhancement_factor
            
            # 2. Boundary preservation
            historical_min = np.min(historical_data)
            historical_max = np.max(historical_data)
            historical_range = historical_max - historical_min
            
            # Soft boundary constraints
            buffer = historical_range * 0.1
            lower_bound = historical_min - buffer
            upper_bound = historical_max + buffer
            
            enhanced = np.clip(enhanced, lower_bound, upper_bound)
            
            # 3. Continuity preservation
            if len(historical_data) > 0:
                last_value = historical_data[-1]
                first_prediction = enhanced[0]
                
                # Ensure smooth transition
                if abs(first_prediction - last_value) > historical_std * 2:
                    # Apply smoothing
                    transition_steps = min(5, len(enhanced))
                    for i in range(transition_steps):
                        blend_factor = i / transition_steps
                        enhanced[i] = last_value * (1 - blend_factor) + enhanced[i] * blend_factor
            
            # 4. Pattern consistency check
            # Ensure predictions follow dominant patterns
            cyclical_patterns = patterns.get('cyclical', {})
            if cyclical_patterns.get('overall_cyclical_strength', 0) > 0.5:
                # Apply cyclical smoothing
                period = cyclical_patterns.get('components', [{}])[0].get('period', 10)
                if period > 0 and period < len(enhanced):
                    smoothed = savgol_filter(enhanced, min(period, len(enhanced)//2*2-1), 2)
                    enhanced = enhanced * 0.7 + smoothed * 0.3
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in applying pattern enhancements: {e}")
            return predictions
    
    def _calculate_prediction_quality(self, predictions: np.ndarray, 
                                    historical_data: np.ndarray, 
                                    patterns: Dict[str, Any]) -> float:
        """Calculate quality score for predictions based on pattern consistency"""
        try:
            quality_scores = []
            
            # 1. Variability consistency
            hist_std = np.std(historical_data)
            pred_std = np.std(predictions)
            variability_score = 1 - abs(hist_std - pred_std) / (hist_std + 1e-10)
            quality_scores.append(max(0, variability_score))
            
            # 2. Range consistency
            hist_range = np.max(historical_data) - np.min(historical_data)
            pred_range = np.max(predictions) - np.min(predictions)
            range_score = 1 - abs(hist_range - pred_range) / (hist_range + 1e-10)
            quality_scores.append(max(0, range_score))
            
            # 3. Pattern consistency
            # Check if predictions maintain cyclical patterns
            cyclical_patterns = patterns.get('cyclical', {})
            cyclical_strength = cyclical_patterns.get('overall_cyclical_strength', 0)
            
            if cyclical_strength > 0.3 and len(predictions) > 10:
                # Check for cyclical behavior in predictions
                autocorr = np.correlate(predictions, predictions, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]
                
                if len(autocorr) > 5:
                    pred_cyclical_strength = np.max(autocorr[1:min(len(autocorr), 20)])
                    cyclical_score = min(1, pred_cyclical_strength / cyclical_strength)
                    quality_scores.append(cyclical_score)
            
            # 4. Trend consistency
            trending_patterns = patterns.get('trending', {})
            trend_r2 = trending_patterns.get('linear', {}).get('r2', 0)
            
            if trend_r2 > 0.3 and len(predictions) > 5:
                # Check trend in predictions
                x = np.arange(len(predictions))
                pred_trend_r2 = r2_score(predictions, np.polyval(np.polyfit(x, predictions, 1), x))
                trend_score = min(1, pred_trend_r2 / trend_r2)
                quality_scores.append(trend_score)
            
            # 5. Continuity score
            if len(historical_data) > 0:
                transition_diff = abs(predictions[0] - historical_data[-1])
                historical_typical_diff = np.mean(np.abs(np.diff(historical_data[-10:])))
                continuity_score = max(0, 1 - transition_diff / (historical_typical_diff * 3 + 1e-10))
                quality_scores.append(continuity_score)
            
            # Overall quality score
            overall_quality = np.mean(quality_scores) if quality_scores else 0.5
            return float(overall_quality)
            
        except Exception as e:
            logger.error(f"Error in calculating prediction quality: {e}")
            return 0.5
    
    def _generate_fallback_predictions(self, data: np.ndarray, steps: int) -> Dict[str, Any]:
        """Generate simple fallback predictions when pattern analysis fails"""
        try:
            if len(data) == 0:
                predictions = np.zeros(steps)
            else:
                # Simple trend extrapolation
                recent_data = data[-min(10, len(data)):]
                if len(recent_data) >= 2:
                    trend = np.polyfit(np.arange(len(recent_data)), recent_data, 1)[0]
                    start_value = data[-1]
                    predictions = np.array([start_value + trend * (i + 1) * 0.5 for i in range(steps)])
                else:
                    predictions = np.full(steps, np.mean(data))
                
                # Add small amount of realistic noise
                noise_std = np.std(data) * 0.1 if len(data) > 1 else 0.1
                noise = np.random.normal(0, noise_std, steps)
                predictions += noise
            
            return {
                'predictions': predictions.tolist(),
                'pattern_analysis': {'learned_patterns': {}, 'pattern_confidence': {'overall': 0.3}},
                'individual_predictions': {},
                'pattern_weights': {},
                'quality_score': 0.3
            }
            
        except Exception as e:
            logger.error(f"Error in fallback prediction generation: {e}")
            return {
                'predictions': np.zeros(steps).tolist(),
                'pattern_analysis': {'learned_patterns': {}, 'pattern_confidence': {'overall': 0.1}},
                'individual_predictions': {},
                'pattern_weights': {},
                'quality_score': 0.1
            }