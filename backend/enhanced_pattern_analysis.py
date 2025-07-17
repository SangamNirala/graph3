"""
Enhanced Pattern Analysis Module for Real-time Graph Prediction System
Focus on pattern accuracy and historical trend preservation
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdvancedPatternAnalyzer:
    """
    Advanced pattern analysis for time series data with focus on accuracy and trend preservation
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pattern_cache = {}
        self.statistical_properties = {}
        
    def analyze_comprehensive_patterns(self, data: np.ndarray, 
                                     timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        Comprehensive pattern analysis focusing on accuracy and trend preservation
        """
        try:
            # Ensure data is clean
            data = self._clean_data(data)
            
            # 1. Basic statistical properties
            stats_props = self._extract_statistical_properties(data)
            
            # 2. Trend analysis with multiple methods
            trend_analysis = self._analyze_trends(data)
            
            # 3. Seasonal decomposition
            seasonal_analysis = self._analyze_seasonality(data, timestamps)
            
            # 4. Cyclical pattern detection
            cyclical_analysis = self._detect_cyclical_patterns(data)
            
            # 5. Autocorrelation analysis
            autocorr_analysis = self._analyze_autocorrelation(data)
            
            # 6. Spectral analysis
            spectral_analysis = self._analyze_spectral_properties(data)
            
            # 7. Volatility and stability analysis
            volatility_analysis = self._analyze_volatility(data)
            
            # 8. Pattern similarity and clustering
            pattern_similarity = self._analyze_pattern_similarity(data)
            
            # 9. Change point detection
            change_points = self._detect_change_points(data)
            
            # 10. Predictability assessment
            predictability = self._assess_predictability(data)
            
            # Combine all analyses
            comprehensive_patterns = {
                'statistical_properties': stats_props,
                'trend_analysis': trend_analysis,
                'seasonal_analysis': seasonal_analysis,
                'cyclical_analysis': cyclical_analysis,
                'autocorrelation_analysis': autocorr_analysis,
                'spectral_analysis': spectral_analysis,
                'volatility_analysis': volatility_analysis,
                'pattern_similarity': pattern_similarity,
                'change_points': change_points,
                'predictability': predictability,
                'data_length': len(data),
                'quality_score': self._calculate_pattern_quality_score(data)
            }
            
            # Store for future reference
            self.pattern_cache = comprehensive_patterns
            self.statistical_properties = stats_props
            
            return comprehensive_patterns
            
        except Exception as e:
            logger.error(f"Error in comprehensive pattern analysis: {e}")
            return self._get_fallback_patterns(data)
    
    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """Clean and preprocess data"""
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        # Remove outliers using IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing to maintain data length
        data = np.clip(data, lower_bound, upper_bound)
        
        return data
    
    def _extract_statistical_properties(self, data: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive statistical properties"""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'var': float(np.var(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
            'range': float(np.max(data) - np.min(data)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data)),
            'cv': float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0,
            'entropy': float(stats.entropy(np.histogram(data, bins=20)[0] + 1e-10))
        }
    
    def _analyze_trends(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze trends using multiple methods"""
        x = np.arange(len(data))
        
        # Linear trend
        linear_coef = np.polyfit(x, data, 1)
        linear_trend = linear_coef[0]
        
        # Polynomial trends
        poly2_coef = np.polyfit(x, data, 2)
        poly3_coef = np.polyfit(x, data, 3) if len(data) > 5 else [0, 0, 0, 0]
        
        # Moving average trends
        window_sizes = [5, 10, 20]
        ma_trends = {}
        for window in window_sizes:
            if len(data) >= window:
                ma = pd.Series(data).rolling(window=window).mean()
                ma_trend = np.polyfit(x[window-1:], ma[window-1:], 1)[0]
                ma_trends[f'ma_{window}'] = float(ma_trend)
        
        # Exponential smoothing trend
        alpha = 0.3
        exp_smooth = [data[0]]
        for i in range(1, len(data)):
            exp_smooth.append(alpha * data[i] + (1 - alpha) * exp_smooth[-1])
        exp_trend = np.polyfit(x, exp_smooth, 1)[0]
        
        # Recent vs historical trend comparison
        recent_portion = max(5, len(data) // 4)
        recent_data = data[-recent_portion:]
        recent_x = np.arange(len(recent_data))
        recent_trend = np.polyfit(recent_x, recent_data, 1)[0]
        
        historical_data = data[:-recent_portion] if len(data) > recent_portion else data
        historical_x = np.arange(len(historical_data))
        historical_trend = np.polyfit(historical_x, historical_data, 1)[0] if len(historical_data) > 1 else 0
        
        return {
            'linear_trend': float(linear_trend),
            'poly2_trend': float(poly2_coef[0]),
            'poly3_trend': float(poly3_coef[0]),
            'ma_trends': ma_trends,
            'exponential_trend': float(exp_trend),
            'recent_trend': float(recent_trend),
            'historical_trend': float(historical_trend),
            'trend_acceleration': float(recent_trend - historical_trend),
            'trend_consistency': self._calculate_trend_consistency(data),
            'trend_strength': self._calculate_trend_strength(data)
        }
    
    def _analyze_seasonality(self, data: np.ndarray, 
                           timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        try:
            if len(data) < 10:
                return {'seasonal_strength': 0.0, 'seasonal_patterns': {}}
            
            # STL decomposition if enough data
            if len(data) >= 20:
                series = pd.Series(data)
                stl = STL(series, seasonal=7 if len(data) >= 14 else 5)
                result = stl.fit()
                
                seasonal_strength = np.var(result.seasonal) / np.var(data)
                trend_strength = np.var(result.trend) / np.var(data)
                
                return {
                    'seasonal_strength': float(seasonal_strength),
                    'trend_strength': float(trend_strength),
                    'seasonal_component': result.seasonal.tolist(),
                    'trend_component': result.trend.tolist(),
                    'residual_component': result.resid.tolist()
                }
            else:
                # Simple seasonal detection for small datasets
                seasonal_strength = self._simple_seasonal_detection(data)
                return {
                    'seasonal_strength': seasonal_strength,
                    'seasonal_patterns': {}
                }
                
        except Exception as e:
            logger.warning(f"Seasonality analysis failed: {e}")
            return {'seasonal_strength': 0.0, 'seasonal_patterns': {}}
    
    def _detect_cyclical_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect cyclical patterns in the data"""
        try:
            # Autocorrelation-based cycle detection
            max_lag = min(len(data) // 2, 50)
            autocorr = acf(data, nlags=max_lag, fft=True)
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.1, distance=3)
            
            cycles = []
            for peak in peaks:
                cycle_length = peak + 1
                cycle_strength = autocorr[peak + 1]
                cycles.append({
                    'length': int(cycle_length),
                    'strength': float(cycle_strength)
                })
            
            # Dominant cycle
            dominant_cycle = max(cycles, key=lambda x: x['strength']) if cycles else None
            
            return {
                'detected_cycles': cycles,
                'dominant_cycle': dominant_cycle,
                'cycle_regularity': self._calculate_cycle_regularity(data)
            }
            
        except Exception as e:
            logger.warning(f"Cyclical pattern detection failed: {e}")
            return {'detected_cycles': [], 'dominant_cycle': None}
    
    def _analyze_autocorrelation(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze autocorrelation structure"""
        try:
            max_lag = min(len(data) // 2, 20)
            
            # Autocorrelation function
            autocorr = acf(data, nlags=max_lag, fft=True)
            
            # Partial autocorrelation function
            partial_autocorr = pacf(data, nlags=max_lag)
            
            # Significant lags
            significant_lags = []
            for i, val in enumerate(autocorr[1:], 1):
                if abs(val) > 2 / np.sqrt(len(data)):  # 95% confidence
                    significant_lags.append({'lag': i, 'value': float(val)})
            
            return {
                'autocorr_values': autocorr.tolist(),
                'partial_autocorr_values': partial_autocorr.tolist(),
                'significant_lags': significant_lags,
                'max_autocorr': float(np.max(autocorr[1:])),
                'autocorr_decay_rate': self._calculate_autocorr_decay_rate(autocorr)
            }
            
        except Exception as e:
            logger.warning(f"Autocorrelation analysis failed: {e}")
            return {'autocorr_values': [], 'significant_lags': []}
    
    def _analyze_spectral_properties(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral properties using FFT"""
        try:
            # FFT analysis
            fft_values = fft(data)
            freqs = fftfreq(len(data))
            
            # Power spectral density
            psd = np.abs(fft_values)**2
            
            # Dominant frequencies
            dominant_freq_indices = np.argsort(psd)[::-1][:5]
            dominant_frequencies = []
            
            for idx in dominant_freq_indices:
                if freqs[idx] > 0:  # Only positive frequencies
                    dominant_frequencies.append({
                        'frequency': float(freqs[idx]),
                        'power': float(psd[idx]),
                        'period': float(1 / freqs[idx]) if freqs[idx] != 0 else np.inf
                    })
            
            return {
                'dominant_frequencies': dominant_frequencies,
                'spectral_entropy': float(stats.entropy(psd + 1e-10)),
                'spectral_centroid': float(np.sum(freqs * psd) / np.sum(psd)),
                'spectral_rolloff': self._calculate_spectral_rolloff(freqs, psd)
            }
            
        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")
            return {'dominant_frequencies': [], 'spectral_entropy': 0.0}
    
    def _analyze_volatility(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility and stability"""
        # Rolling statistics
        windows = [5, 10, 20]
        rolling_stats = {}
        
        for window in windows:
            if len(data) >= window:
                rolling_std = pd.Series(data).rolling(window=window).std()
                rolling_mean = pd.Series(data).rolling(window=window).mean()
                
                rolling_stats[f'window_{window}'] = {
                    'avg_volatility': float(np.nanmean(rolling_std)),
                    'volatility_trend': float(np.polyfit(range(len(rolling_std)), 
                                                       rolling_std.fillna(0), 1)[0]),
                    'mean_stability': float(np.nanstd(rolling_mean))
                }
        
        # Volatility clustering
        returns = np.diff(data)
        volatility_clustering = self._calculate_volatility_clustering(returns)
        
        return {
            'overall_volatility': float(np.std(data)),
            'volatility_of_volatility': float(np.std(np.diff(data))),
            'rolling_volatility': rolling_stats,
            'volatility_clustering': volatility_clustering,
            'stability_score': self._calculate_stability_score(data)
        }
    
    def _analyze_pattern_similarity(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze pattern similarity and repetition"""
        try:
            # Sliding window pattern analysis
            window_size = min(10, len(data) // 3)
            if window_size < 3:
                return {'pattern_repetition': 0.0, 'similar_patterns': []}
            
            patterns = []
            for i in range(len(data) - window_size + 1):
                pattern = data[i:i + window_size]
                patterns.append(pattern)
            
            # Calculate pattern similarities
            similarities = []
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    similarity = np.corrcoef(patterns[i], patterns[j])[0, 1]
                    if not np.isnan(similarity):
                        similarities.append(similarity)
            
            return {
                'pattern_repetition': float(np.mean(similarities)) if similarities else 0.0,
                'max_similarity': float(np.max(similarities)) if similarities else 0.0,
                'pattern_diversity': float(np.std(similarities)) if similarities else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Pattern similarity analysis failed: {e}")
            return {'pattern_repetition': 0.0, 'similar_patterns': []}
    
    def _detect_change_points(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect change points in the data"""
        try:
            # Simple change point detection using variance
            if len(data) < 10:
                return {'change_points': [], 'change_point_strength': 0.0}
            
            window_size = max(5, len(data) // 10)
            change_points = []
            
            for i in range(window_size, len(data) - window_size):
                before = data[i - window_size:i]
                after = data[i:i + window_size]
                
                # Test for change in mean
                t_stat, p_value = stats.ttest_ind(before, after)
                
                if p_value < 0.05:  # Significant change
                    change_points.append({
                        'index': i,
                        'p_value': float(p_value),
                        'magnitude': float(abs(np.mean(after) - np.mean(before)))
                    })
            
            return {
                'change_points': change_points,
                'change_point_strength': float(len(change_points) / len(data))
            }
            
        except Exception as e:
            logger.warning(f"Change point detection failed: {e}")
            return {'change_points': [], 'change_point_strength': 0.0}
    
    def _assess_predictability(self, data: np.ndarray) -> Dict[str, Any]:
        """Assess how predictable the data is"""
        try:
            # Entropy-based predictability
            hist, _ = np.histogram(data, bins=min(20, len(data) // 2))
            entropy = stats.entropy(hist + 1e-10)
            
            # Hurst exponent (simplified)
            hurst = self._calculate_hurst_exponent(data)
            
            # Predictability score
            predictability_score = 1 - (entropy / np.log(len(hist)))
            
            return {
                'entropy': float(entropy),
                'hurst_exponent': float(hurst),
                'predictability_score': float(predictability_score),
                'randomness_test': self._randomness_test(data)
            }
            
        except Exception as e:
            logger.warning(f"Predictability assessment failed: {e}")
            return {'predictability_score': 0.5, 'entropy': 0.0}
    
    def _calculate_pattern_quality_score(self, data: np.ndarray) -> float:
        """Calculate overall pattern quality score"""
        try:
            # Combine multiple factors
            factors = []
            
            # Data length factor
            length_factor = min(1.0, len(data) / 50)
            factors.append(length_factor)
            
            # Trend consistency factor
            trend_consistency = self._calculate_trend_consistency(data)
            factors.append(trend_consistency)
            
            # Noise level factor (inverse of noise)
            noise_level = np.std(np.diff(data)) / np.std(data)
            noise_factor = 1 / (1 + noise_level)
            factors.append(noise_factor)
            
            # Predictability factor
            predictability = self._assess_predictability(data)['predictability_score']
            factors.append(predictability)
            
            return float(np.mean(factors))
            
        except Exception as e:
            logger.warning(f"Pattern quality calculation failed: {e}")
            return 0.5
    
    # Helper methods
    def _simple_seasonal_detection(self, data: np.ndarray) -> float:
        """Simple seasonal detection for small datasets"""
        if len(data) < 6:
            return 0.0
        
        # Test for periodicity using autocorrelation
        periods = [2, 3, 4, 5, 6, 7]
        max_seasonal = 0.0
        
        for period in periods:
            if len(data) >= period * 2:
                autocorr = np.corrcoef(data[:-period], data[period:])[0, 1]
                if not np.isnan(autocorr):
                    max_seasonal = max(max_seasonal, autocorr)
        
        return float(max_seasonal)
    
    def _calculate_trend_consistency(self, data: np.ndarray) -> float:
        """Calculate trend consistency"""
        if len(data) < 10:
            return 0.5
        
        # Calculate trends for different segments
        segment_size = len(data) // 3
        trends = []
        
        for i in range(0, len(data) - segment_size, segment_size):
            segment = data[i:i + segment_size]
            trend = np.polyfit(range(len(segment)), segment, 1)[0]
            trends.append(trend)
        
        if len(trends) > 1:
            consistency = 1 - (np.std(trends) / (np.mean(np.abs(trends)) + 1e-10))
            return float(max(0, min(1, consistency)))
        
        return 0.5
    
    def _calculate_trend_strength(self, data: np.ndarray) -> float:
        """Calculate trend strength"""
        if len(data) < 3:
            return 0.0
        
        x = np.arange(len(data))
        trend_coef = np.polyfit(x, data, 1)[0]
        data_std = np.std(data)
        
        return float(abs(trend_coef) / (data_std + 1e-10))
    
    def _calculate_cycle_regularity(self, data: np.ndarray) -> float:
        """Calculate cycle regularity"""
        try:
            # Find local minima and maxima
            minima = signal.argrelmin(data)[0]
            maxima = signal.argrelmax(data)[0]
            
            # Calculate cycle lengths
            if len(minima) > 1:
                cycle_lengths = np.diff(minima)
                regularity = 1 - (np.std(cycle_lengths) / (np.mean(cycle_lengths) + 1e-10))
                return float(max(0, min(1, regularity)))
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def _calculate_autocorr_decay_rate(self, autocorr: np.ndarray) -> float:
        """Calculate autocorrelation decay rate"""
        try:
            # Fit exponential decay to autocorrelation
            x = np.arange(len(autocorr))
            pos_autocorr = np.abs(autocorr)
            
            # Simple exponential fit
            log_autocorr = np.log(pos_autocorr + 1e-10)
            decay_rate = -np.polyfit(x, log_autocorr, 1)[0]
            
            return float(decay_rate)
            
        except Exception as e:
            return 0.0
    
    def _calculate_spectral_rolloff(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Calculate spectral rolloff"""
        try:
            cumulative_psd = np.cumsum(psd)
            total_power = cumulative_psd[-1]
            rolloff_threshold = 0.85 * total_power
            
            rolloff_index = np.where(cumulative_psd >= rolloff_threshold)[0]
            if len(rolloff_index) > 0:
                return float(freqs[rolloff_index[0]])
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def _calculate_volatility_clustering(self, returns: np.ndarray) -> float:
        """Calculate volatility clustering measure"""
        try:
            if len(returns) < 2:
                return 0.0
            
            # Calculate squared returns
            squared_returns = returns**2
            
            # Autocorrelation of squared returns
            autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            
            return float(autocorr) if not np.isnan(autocorr) else 0.0
            
        except Exception as e:
            return 0.0
    
    def _calculate_stability_score(self, data: np.ndarray) -> float:
        """Calculate stability score"""
        try:
            # Coefficient of variation
            cv = np.std(data) / (np.mean(data) + 1e-10)
            
            # Stability score (inverse of CV)
            stability = 1 / (1 + cv)
            
            return float(stability)
            
        except Exception as e:
            return 0.5
    
    def _calculate_hurst_exponent(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent (simplified)"""
        try:
            if len(data) < 10:
                return 0.5
            
            # Rescaled range analysis
            n = len(data)
            mean_data = np.mean(data)
            y = np.cumsum(data - mean_data)
            
            R = np.max(y) - np.min(y)
            S = np.std(data)
            
            if S == 0:
                return 0.5
            
            hurst = np.log(R / S) / np.log(n)
            return float(max(0, min(1, hurst)))
            
        except Exception as e:
            return 0.5
    
    def _randomness_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Test for randomness in the data"""
        try:
            # Runs test
            median = np.median(data)
            runs = 0
            current_run = data[0] > median
            
            for i in range(1, len(data)):
                if (data[i] > median) != current_run:
                    runs += 1
                    current_run = not current_run
            
            expected_runs = 2 * np.sum(data > median) * np.sum(data <= median) / len(data) + 1
            
            return {
                'runs': runs,
                'expected_runs': float(expected_runs),
                'is_random': abs(runs - expected_runs) < 2 * np.sqrt(expected_runs)
            }
            
        except Exception as e:
            return {'runs': 0, 'expected_runs': 0, 'is_random': True}
    
    def _get_fallback_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Fallback patterns for error cases"""
        return {
            'statistical_properties': {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'last_value': float(data[-1])
            },
            'trend_analysis': {
                'linear_trend': float(np.polyfit(range(len(data)), data, 1)[0]),
                'recent_trend': float(np.polyfit(range(len(data[-5:])), data[-5:], 1)[0])
            },
            'quality_score': 0.5,
            'predictability': {'predictability_score': 0.5}
        }