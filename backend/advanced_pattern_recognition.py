"""
Advanced Pattern Recognition System for Real-time Continuous Prediction
Industry-level pattern detection and analysis using advanced mathematical techniques
"""

import numpy as np
import pandas as pd
from scipy import signal, fft, optimize, stats
from scipy.signal import find_peaks, savgol_filter, periodogram, welch
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class IndustryLevelPatternRecognition:
    """
    Advanced pattern recognition system using state-of-the-art mathematical techniques
    for identifying and analyzing complex time series patterns
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.pattern_cache = {}
        self.pattern_history = []
        self.frequency_analyzer = FrequencyDomainAnalyzer()
        self.harmonic_analyzer = HarmonicPatternAnalyzer()
        self.trend_analyzer = MultiScaleTrendAnalyzer()
        self.seasonal_analyzer = AdvancedSeasonalAnalyzer()
        self.pattern_classifier = PatternClassifier()
        
    def analyze_comprehensive_patterns(self, data: np.ndarray, 
                                     timestamps: Optional[pd.DatetimeIndex] = None,
                                     sampling_rate: float = 1.0) -> Dict[str, Any]:
        """
        Perform comprehensive pattern analysis using advanced mathematical techniques
        
        Args:
            data: Time series data
            timestamps: Optional timestamps for the data
            sampling_rate: Sampling rate of the data
            
        Returns:
            Comprehensive pattern analysis results
        """
        try:
            # Ensure data is clean and properly formatted
            data = self._preprocess_data(data)
            
            # 1. Frequency Domain Analysis
            frequency_analysis = self.frequency_analyzer.analyze(data, sampling_rate)
            
            # 2. Harmonic Pattern Analysis
            harmonic_analysis = self.harmonic_analyzer.analyze(data, frequency_analysis)
            
            # 3. Multi-Scale Trend Analysis
            trend_analysis = self.trend_analyzer.analyze(data, timestamps)
            
            # 4. Advanced Seasonal Analysis
            seasonal_analysis = self.seasonal_analyzer.analyze(data, timestamps)
            
            # 5. Pattern Classification
            pattern_class = self.pattern_classifier.classify(data, {
                'frequency': frequency_analysis,
                'harmonic': harmonic_analysis,
                'trend': trend_analysis,
                'seasonal': seasonal_analysis
            })
            
            # 6. Predictability Assessment
            predictability = self._assess_predictability(data, pattern_class)
            
            # 7. Noise Analysis
            noise_analysis = self._analyze_noise_characteristics(data)
            
            # 8. Structural Break Detection
            structural_breaks = self._detect_structural_breaks(data)
            
            # 9. Volatility Clustering Analysis
            volatility_analysis = self._analyze_volatility_clustering(data)
            
            # 10. Pattern Stability Analysis
            stability_analysis = self._analyze_pattern_stability(data)
            
            # Compile comprehensive results
            comprehensive_patterns = {
                'frequency_analysis': frequency_analysis,
                'harmonic_analysis': harmonic_analysis,
                'trend_analysis': trend_analysis,
                'seasonal_analysis': seasonal_analysis,
                'pattern_classification': pattern_class,
                'predictability': predictability,
                'noise_analysis': noise_analysis,
                'structural_breaks': structural_breaks,
                'volatility_analysis': volatility_analysis,
                'stability_analysis': stability_analysis,
                'data_quality': self._assess_data_quality(data),
                'pattern_strength': self._calculate_pattern_strength(data, pattern_class),
                'complexity_score': self._calculate_complexity_score(data),
                'confidence_score': self._calculate_confidence_score(data, pattern_class),
                'metadata': {
                    'data_length': len(data),
                    'sampling_rate': sampling_rate,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            # Cache for future reference
            self.pattern_cache = comprehensive_patterns
            self.pattern_history.append(comprehensive_patterns)
            
            return comprehensive_patterns
            
        except Exception as e:
            logger.error(f"Error in comprehensive pattern analysis: {e}")
            return self._get_fallback_analysis(data)
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Advanced data preprocessing with outlier detection and noise reduction"""
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        # Detect and handle outliers using multiple methods
        data = self._handle_outliers(data)
        
        # Apply detrending if necessary
        if self._needs_detrending(data):
            data = signal.detrend(data)
        
        # Apply noise reduction
        data = self._apply_noise_reduction(data)
        
        return data
    
    def _handle_outliers(self, data: np.ndarray) -> np.ndarray:
        """Handle outliers using multiple detection methods"""
        # Method 1: Modified Z-score
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask1 = np.abs(modified_z_scores) > 3.5
        
        # Method 2: IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask2 = (data < lower_bound) | (data > upper_bound)
        
        # Combine outlier masks
        outlier_mask = outlier_mask1 | outlier_mask2
        
        # Handle outliers by capping instead of removing
        data_clean = data.copy()
        if np.any(outlier_mask):
            data_clean[outlier_mask] = np.where(
                data[outlier_mask] > np.median(data),
                upper_bound,
                lower_bound
            )
        
        return data_clean
    
    def _needs_detrending(self, data: np.ndarray) -> bool:
        """Determine if data needs detrending"""
        # Perform Augmented Dickey-Fuller test
        try:
            result = adfuller(data)
            p_value = result[1]
            return p_value > 0.05  # Non-stationary data needs detrending
        except:
            return False
    
    def _apply_noise_reduction(self, data: np.ndarray) -> np.ndarray:
        """Apply adaptive noise reduction"""
        if len(data) < 5:
            return data
            
        # Use Savitzky-Golay filter for noise reduction
        window_length = min(len(data), 5)
        if window_length % 2 == 0:
            window_length -= 1
            
        if window_length >= 3:
            return savgol_filter(data, window_length, 2)
        else:
            return data
    
    def _assess_predictability(self, data: np.ndarray, pattern_class: Dict) -> Dict[str, Any]:
        """Assess predictability using multiple metrics"""
        try:
            # Approximate entropy
            approx_entropy = self._calculate_approximate_entropy(data)
            
            # Hurst exponent
            hurst_exponent = self._calculate_hurst_exponent(data)
            
            # Lyapunov exponent (simplified)
            lyapunov_exponent = self._calculate_lyapunov_exponent(data)
            
            # Pattern-based predictability
            pattern_predictability = self._calculate_pattern_predictability(pattern_class)
            
            # Overall predictability score
            predictability_score = (
                (1 - approx_entropy) * 0.3 +
                abs(hurst_exponent - 0.5) * 2 * 0.3 +
                max(0, -lyapunov_exponent) * 0.2 +
                pattern_predictability * 0.2
            )
            
            return {
                'approximate_entropy': float(approx_entropy),
                'hurst_exponent': float(hurst_exponent),
                'lyapunov_exponent': float(lyapunov_exponent),
                'pattern_predictability': float(pattern_predictability),
                'predictability_score': float(np.clip(predictability_score, 0, 1))
            }
            
        except Exception as e:
            logger.warning(f"Predictability assessment failed: {e}")
            return {'predictability_score': 0.5}
    
    def _calculate_approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate approximate entropy"""
        if r is None:
            r = 0.2 * np.std(data)
        
        n = len(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(n - m + 1)])
            C = np.zeros(n - m + 1)
            
            for i in range(n - m + 1):
                template = patterns[i]
                for j in range(n - m + 1):
                    if _maxdist(template, patterns[j], m) <= r:
                        C[i] += 1
            
            phi = np.mean(np.log(C / (n - m + 1)))
            return phi
        
        return _phi(m) - _phi(m + 1)
    
    def _calculate_hurst_exponent(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        try:
            n = len(data)
            if n < 10:
                return 0.5
            
            # Calculate mean
            mean_data = np.mean(data)
            
            # Calculate cumulative deviation
            cumulative_deviation = np.cumsum(data - mean_data)
            
            # Calculate range
            R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
            
            # Calculate standard deviation
            S = np.std(data)
            
            if S == 0:
                return 0.5
            
            # Calculate Hurst exponent
            hurst = np.log(R / S) / np.log(n)
            
            return float(np.clip(hurst, 0, 1))
            
        except Exception as e:
            return 0.5
    
    def _calculate_lyapunov_exponent(self, data: np.ndarray) -> float:
        """Calculate largest Lyapunov exponent (simplified)"""
        try:
            if len(data) < 10:
                return 0.0
            
            # Embed the data
            embedding_dim = 3
            delay = 1
            
            embedded = np.array([data[i:i + embedding_dim * delay:delay] 
                               for i in range(len(data) - embedding_dim * delay + 1)])
            
            # Calculate divergence
            divergences = []
            for i in range(len(embedded) - 1):
                distances = np.linalg.norm(embedded[i+1:] - embedded[i], axis=1)
                if len(distances) > 0 and np.min(distances) > 0:
                    divergences.append(np.log(np.min(distances)))
            
            if len(divergences) > 1:
                return float(np.mean(np.diff(divergences)))
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def _calculate_pattern_predictability(self, pattern_class: Dict) -> float:
        """Calculate predictability based on pattern classification"""
        try:
            pattern_type = pattern_class.get('primary_pattern', 'unknown')
            pattern_strength = pattern_class.get('pattern_strength', 0.5)
            
            # Different pattern types have different predictability
            predictability_map = {
                'linear': 0.9,
                'exponential': 0.8,
                'logarithmic': 0.8,
                'polynomial': 0.7,
                'sinusoidal': 0.9,
                'periodic': 0.9,
                'seasonal': 0.8,
                'cyclical': 0.7,
                'trending': 0.6,
                'random_walk': 0.2,
                'white_noise': 0.1,
                'unknown': 0.5
            }
            
            base_predictability = predictability_map.get(pattern_type, 0.5)
            
            # Adjust based on pattern strength
            return base_predictability * pattern_strength
            
        except Exception as e:
            return 0.5
    
    def _analyze_noise_characteristics(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze noise characteristics in the data"""
        try:
            # Estimate noise level
            noise_level = self._estimate_noise_level(data)
            
            # Analyze noise distribution
            noise_distribution = self._analyze_noise_distribution(data)
            
            # Signal-to-noise ratio
            snr = self._calculate_snr(data)
            
            # Noise autocorrelation
            noise_autocorr = self._analyze_noise_autocorrelation(data)
            
            return {
                'noise_level': float(noise_level),
                'noise_distribution': noise_distribution,
                'signal_to_noise_ratio': float(snr),
                'noise_autocorrelation': noise_autocorr
            }
            
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
            return {'noise_level': 0.1, 'signal_to_noise_ratio': 10.0}
    
    def _estimate_noise_level(self, data: np.ndarray) -> float:
        """Estimate noise level using robust methods"""
        # Use median absolute deviation of first differences
        differences = np.diff(data)
        noise_level = np.median(np.abs(differences)) / 0.6745
        return noise_level
    
    def _analyze_noise_distribution(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of noise in the data"""
        try:
            # Estimate noise by detrending
            detrended = signal.detrend(data)
            
            # Test for normality
            shapiro_stat, shapiro_p = stats.shapiro(detrended[:5000])  # Limit for performance
            
            # Calculate skewness and kurtosis
            skewness = stats.skew(detrended)
            kurtosis = stats.kurtosis(detrended)
            
            return {
                'is_normal': float(shapiro_p > 0.05),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'shapiro_statistic': float(shapiro_stat),
                'shapiro_p_value': float(shapiro_p)
            }
            
        except Exception as e:
            return {'is_normal': True, 'skewness': 0.0, 'kurtosis': 0.0}
    
    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            # Estimate signal power
            signal_power = np.var(data)
            
            # Estimate noise power
            noise_power = np.var(np.diff(data)) / 2
            
            if noise_power == 0:
                return float('inf')
            
            snr = 10 * np.log10(signal_power / noise_power)
            return float(snr)
            
        except Exception as e:
            return 10.0
    
    def _analyze_noise_autocorrelation(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze autocorrelation of noise"""
        try:
            # Estimate noise
            noise = signal.detrend(data)
            
            # Calculate autocorrelation
            autocorr = np.correlate(noise, noise, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find significant lags
            significant_lags = []
            for i in range(1, min(len(autocorr), 20)):
                if abs(autocorr[i]) > 2 / np.sqrt(len(data)):
                    significant_lags.append({'lag': i, 'correlation': float(autocorr[i])})
            
            return {
                'significant_lags': significant_lags,
                'max_autocorr': float(np.max(np.abs(autocorr[1:20]))) if len(autocorr) > 1 else 0.0
            }
            
        except Exception as e:
            return {'significant_lags': [], 'max_autocorr': 0.0}
    
    def _detect_structural_breaks(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect structural breaks in the data"""
        try:
            if len(data) < 20:
                return {'breaks': [], 'break_count': 0}
            
            # Use CUSUM test for break detection
            breaks = []
            window_size = max(10, len(data) // 10)
            
            for i in range(window_size, len(data) - window_size):
                # Test for break at position i
                before = data[i-window_size:i]
                after = data[i:i+window_size]
                
                # T-test for difference in means
                t_stat, p_value = stats.ttest_ind(before, after)
                
                if p_value < 0.01:  # Significant break
                    breaks.append({
                        'position': i,
                        'p_value': float(p_value),
                        'magnitude': float(np.mean(after) - np.mean(before))
                    })
            
            return {
                'breaks': breaks,
                'break_count': len(breaks)
            }
            
        except Exception as e:
            return {'breaks': [], 'break_count': 0}
    
    def _analyze_volatility_clustering(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility clustering in the data"""
        try:
            # Calculate returns
            returns = np.diff(data) / data[:-1]
            
            # Calculate squared returns
            squared_returns = returns ** 2
            
            # Test for ARCH effects
            arch_lags = min(5, len(squared_returns) // 4)
            arch_test_stat = self._arch_test(squared_returns, arch_lags)
            
            # Calculate volatility clustering measure
            volatility_clustering = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            
            return {
                'volatility_clustering': float(volatility_clustering) if not np.isnan(volatility_clustering) else 0.0,
                'arch_test_statistic': float(arch_test_stat),
                'has_volatility_clustering': bool(volatility_clustering > 0.1)
            }
            
        except Exception as e:
            return {'volatility_clustering': 0.0, 'has_volatility_clustering': False}
    
    def _arch_test(self, squared_returns: np.ndarray, lags: int) -> float:
        """Perform ARCH test"""
        try:
            # Simple ARCH test implementation
            y = squared_returns[lags:]
            X = np.column_stack([squared_returns[i:i+len(y)] for i in range(lags)])
            
            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            fitted = X @ beta
            residuals = y - fitted
            
            # Calculate R-squared
            tss = np.sum((y - np.mean(y))**2)
            rss = np.sum(residuals**2)
            r_squared = 1 - rss/tss
            
            # Test statistic
            n = len(y)
            test_stat = n * r_squared
            
            return test_stat
            
        except Exception as e:
            return 0.0
    
    def _analyze_pattern_stability(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze stability of patterns over time"""
        try:
            if len(data) < 30:
                return {'stability_score': 0.5, 'is_stable': True}
            
            # Divide data into segments
            segment_size = len(data) // 3
            segments = [
                data[:segment_size],
                data[segment_size:2*segment_size],
                data[2*segment_size:]
            ]
            
            # Calculate statistics for each segment
            segment_stats = []
            for segment in segments:
                if len(segment) > 0:
                    stats_dict = {
                        'mean': np.mean(segment),
                        'std': np.std(segment),
                        'trend': np.polyfit(range(len(segment)), segment, 1)[0]
                    }
                    segment_stats.append(stats_dict)
            
            # Calculate stability metrics
            if len(segment_stats) >= 2:
                mean_stability = 1 - np.std([s['mean'] for s in segment_stats]) / np.mean([s['std'] for s in segment_stats])
                trend_stability = 1 - np.std([s['trend'] for s in segment_stats]) / (np.mean([abs(s['trend']) for s in segment_stats]) + 1e-10)
                
                stability_score = (mean_stability + trend_stability) / 2
                stability_score = np.clip(stability_score, 0, 1)
                
                return {
                    'stability_score': float(stability_score),
                    'is_stable': bool(stability_score > 0.7),
                    'segment_statistics': segment_stats
                }
            else:
                return {'stability_score': 0.5, 'is_stable': True}
                
        except Exception as e:
            return {'stability_score': 0.5, 'is_stable': True}
    
    def _assess_data_quality(self, data: np.ndarray) -> Dict[str, Any]:
        """Assess overall data quality"""
        try:
            # Missing values
            missing_ratio = np.sum(np.isnan(data)) / len(data)
            
            # Duplicate values
            unique_ratio = len(np.unique(data)) / len(data)
            
            # Outlier ratio
            outlier_ratio = self._calculate_outlier_ratio(data)
            
            # Data length adequacy
            length_adequacy = min(1.0, len(data) / 50)
            
            # Overall quality score
            quality_score = (
                (1 - missing_ratio) * 0.3 +
                unique_ratio * 0.2 +
                (1 - outlier_ratio) * 0.2 +
                length_adequacy * 0.3
            )
            
            return {
                'missing_ratio': float(missing_ratio),
                'unique_ratio': float(unique_ratio),
                'outlier_ratio': float(outlier_ratio),
                'length_adequacy': float(length_adequacy),
                'quality_score': float(quality_score)
            }
            
        except Exception as e:
            return {'quality_score': 0.5}
    
    def _calculate_outlier_ratio(self, data: np.ndarray) -> float:
        """Calculate ratio of outliers in the data"""
        try:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (data < lower_bound) | (data > upper_bound)
            return np.sum(outliers) / len(data)
            
        except Exception as e:
            return 0.0
    
    def _calculate_pattern_strength(self, data: np.ndarray, pattern_class: Dict) -> float:
        """Calculate overall pattern strength"""
        try:
            # Get pattern type and characteristics
            pattern_type = pattern_class.get('primary_pattern', 'unknown')
            pattern_confidence = pattern_class.get('confidence', 0.5)
            
            # Calculate R-squared for pattern fit
            x = np.arange(len(data))
            
            if pattern_type == 'linear':
                coeffs = np.polyfit(x, data, 1)
                fitted = np.polyval(coeffs, x)
            elif pattern_type == 'exponential':
                # Fit exponential (log-linear)
                try:
                    log_data = np.log(np.abs(data) + 1e-10)
                    coeffs = np.polyfit(x, log_data, 1)
                    fitted = np.exp(np.polyval(coeffs, x))
                except:
                    fitted = np.full_like(data, np.mean(data))
            elif pattern_type == 'sinusoidal':
                # Fit sinusoidal
                try:
                    def sine_func(x, A, B, C, D):
                        return A * np.sin(B * x + C) + D
                    
                    popt, _ = optimize.curve_fit(sine_func, x, data, maxfev=1000)
                    fitted = sine_func(x, *popt)
                except:
                    fitted = np.full_like(data, np.mean(data))
            else:
                # Default to linear fit
                coeffs = np.polyfit(x, data, 1)
                fitted = np.polyval(coeffs, x)
            
            # Calculate R-squared
            ss_res = np.sum((data - fitted) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            
            if ss_tot == 0:
                r_squared = 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # Combine with pattern confidence
            pattern_strength = r_squared * pattern_confidence
            
            return float(np.clip(pattern_strength, 0, 1))
            
        except Exception as e:
            return 0.5
    
    def _calculate_complexity_score(self, data: np.ndarray) -> float:
        """Calculate complexity score of the data"""
        try:
            # Lempel-Ziv complexity (simplified)
            lz_complexity = self._lempel_ziv_complexity(data)
            
            # Fractal dimension
            fractal_dim = self._calculate_fractal_dimension(data)
            
            # Spectral complexity
            spectral_complexity = self._calculate_spectral_complexity(data)
            
            # Combine measures
            complexity_score = (lz_complexity + fractal_dim + spectral_complexity) / 3
            
            return float(np.clip(complexity_score, 0, 1))
            
        except Exception as e:
            return 0.5
    
    def _lempel_ziv_complexity(self, data: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity (simplified)"""
        try:
            # Binarize the data
            median = np.median(data)
            binary_data = (data > median).astype(int)
            
            # Convert to string
            binary_string = ''.join(map(str, binary_data))
            
            # Calculate complexity
            i = 0
            c = 1
            l = 1
            k = 1
            k_max = 1
            n = len(binary_string)
            
            while l + k <= n:
                if binary_string[i + k - 1] == binary_string[l + k - 1]:
                    k += 1
                    if k > k_max:
                        k_max = k
                else:
                    if k_max == k:
                        i += 1
                        if i == l:
                            c += 1
                            l += k_max
                            if i > l:
                                i = l
                            k = 1
                            k_max = 1
                        else:
                            k = 1
                    else:
                        k = k_max
            
            if l <= n:
                c += 1
            
            # Normalize
            normalized_complexity = c / (n / np.log2(n))
            
            return float(np.clip(normalized_complexity, 0, 1))
            
        except Exception as e:
            return 0.5
    
    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using box counting"""
        try:
            # Normalize data
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            
            # Box counting
            scales = np.logspace(0.01, 0.5, num=10)
            counts = []
            
            for scale in scales:
                # Grid size
                grid_size = int(1 / scale)
                if grid_size < 2:
                    continue
                    
                # Count boxes
                boxes = np.zeros((grid_size, grid_size))
                for i, val in enumerate(normalized_data):
                    x = int(i * grid_size / len(normalized_data))
                    y = int(val * grid_size)
                    if x < grid_size and y < grid_size:
                        boxes[x, y] = 1
                
                count = np.sum(boxes)
                counts.append(count)
            
            if len(counts) > 1:
                # Calculate fractal dimension
                log_scales = np.log(scales[:len(counts)])
                log_counts = np.log(counts)
                
                # Fit line
                slope, _ = np.polyfit(log_scales, log_counts, 1)
                fractal_dim = -slope
                
                return float(np.clip(fractal_dim / 2, 0, 1))  # Normalize to [0, 1]
            else:
                return 0.5
                
        except Exception as e:
            return 0.5
    
    def _calculate_spectral_complexity(self, data: np.ndarray) -> float:
        """Calculate spectral complexity"""
        try:
            # Power spectral density
            freqs, psd = welch(data, nperseg=min(len(data)//4, 256))
            
            # Normalize PSD
            psd_norm = psd / np.sum(psd)
            
            # Spectral entropy
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            
            # Normalize
            max_entropy = np.log2(len(psd_norm))
            normalized_entropy = spectral_entropy / max_entropy
            
            return float(np.clip(normalized_entropy, 0, 1))
            
        except Exception as e:
            return 0.5
    
    def _calculate_confidence_score(self, data: np.ndarray, pattern_class: Dict) -> float:
        """Calculate confidence score for the analysis"""
        try:
            # Data quality factors
            data_quality = self._assess_data_quality(data)['quality_score']
            
            # Pattern strength
            pattern_strength = self._calculate_pattern_strength(data, pattern_class)
            
            # Noise level (inverse)
            noise_analysis = self._analyze_noise_characteristics(data)
            snr = noise_analysis['signal_to_noise_ratio']
            noise_factor = 1 / (1 + np.exp(-snr/10))  # Sigmoid transform
            
            # Stability
            stability = self._analyze_pattern_stability(data)['stability_score']
            
            # Combine factors
            confidence_score = (
                data_quality * 0.3 +
                pattern_strength * 0.3 +
                noise_factor * 0.2 +
                stability * 0.2
            )
            
            return float(np.clip(confidence_score, 0, 1))
            
        except Exception as e:
            return 0.5
    
    def _get_fallback_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Fallback analysis for error cases"""
        return {
            'frequency_analysis': {'dominant_frequency': 0.0, 'power_spectrum': []},
            'harmonic_analysis': {'harmonics': [], 'fundamental_frequency': 0.0},
            'trend_analysis': {'trend_slope': float(np.polyfit(range(len(data)), data, 1)[0])},
            'seasonal_analysis': {'seasonal_strength': 0.0, 'period': 0},
            'pattern_classification': {'primary_pattern': 'unknown', 'confidence': 0.5},
            'predictability': {'predictability_score': 0.5},
            'confidence_score': 0.3
        }


class FrequencyDomainAnalyzer:
    """Advanced frequency domain analysis for pattern recognition"""
    
    def analyze(self, data: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, Any]:
        """Perform comprehensive frequency domain analysis"""
        try:
            # Power spectral density
            freqs, psd = welch(data, fs=sampling_rate, nperseg=min(len(data)//4, 256))
            
            # Find dominant frequencies
            dominant_freq_indices = find_peaks(psd, height=np.max(psd) * 0.1)[0]
            dominant_frequencies = []
            
            for idx in dominant_freq_indices:
                dominant_frequencies.append({
                    'frequency': float(freqs[idx]),
                    'power': float(psd[idx]),
                    'relative_power': float(psd[idx] / np.sum(psd))
                })
            
            # Sort by power
            dominant_frequencies.sort(key=lambda x: x['power'], reverse=True)
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            
            # Spectral rolloff
            cumulative_psd = np.cumsum(psd)
            rolloff_idx = np.where(cumulative_psd >= 0.85 * cumulative_psd[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            # Bandwidth
            bandwidth = freqs[np.where(psd >= np.max(psd) * 0.5)[0]]
            spectral_bandwidth = bandwidth[-1] - bandwidth[0] if len(bandwidth) > 1 else 0
            
            return {
                'dominant_frequencies': dominant_frequencies[:5],  # Top 5
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'spectral_bandwidth': float(spectral_bandwidth),
                'power_spectrum': psd.tolist(),
                'frequency_bins': freqs.tolist(),
                'total_power': float(np.sum(psd))
            }
            
        except Exception as e:
            logger.warning(f"Frequency domain analysis failed: {e}")
            return {'dominant_frequencies': [], 'spectral_centroid': 0.0}


class HarmonicPatternAnalyzer:
    """Advanced harmonic pattern analysis"""
    
    def analyze(self, data: np.ndarray, frequency_analysis: Dict) -> Dict[str, Any]:
        """Analyze harmonic patterns in the data"""
        try:
            # Get dominant frequencies
            dominant_freqs = frequency_analysis.get('dominant_frequencies', [])
            
            if not dominant_freqs:
                return {'harmonics': [], 'fundamental_frequency': 0.0}
            
            # Find fundamental frequency
            fundamental_freq = dominant_freqs[0]['frequency']
            
            # Detect harmonics
            harmonics = []
            for freq_info in dominant_freqs:
                freq = freq_info['frequency']
                if freq > 0 and fundamental_freq > 0:
                    harmonic_ratio = freq / fundamental_freq
                    # Check if it's close to an integer ratio
                    closest_integer = round(harmonic_ratio)
                    if abs(harmonic_ratio - closest_integer) < 0.1:
                        harmonics.append({
                            'frequency': freq,
                            'harmonic_number': closest_integer,
                            'power': freq_info['power'],
                            'phase': self._calculate_phase(data, freq)
                        })
            
            # Calculate harmonic distortion
            harmonic_distortion = self._calculate_harmonic_distortion(harmonics)
            
            return {
                'harmonics': harmonics,
                'fundamental_frequency': float(fundamental_freq),
                'harmonic_distortion': float(harmonic_distortion),
                'harmonic_count': len(harmonics)
            }
            
        except Exception as e:
            logger.warning(f"Harmonic pattern analysis failed: {e}")
            return {'harmonics': [], 'fundamental_frequency': 0.0}
    
    def _calculate_phase(self, data: np.ndarray, frequency: float) -> float:
        """Calculate phase of a frequency component"""
        try:
            # Simple phase calculation using FFT
            fft_data = fft.fft(data)
            freq_bin = int(frequency * len(data))
            
            if freq_bin < len(fft_data):
                complex_val = fft_data[freq_bin]
                phase = np.angle(complex_val)
                return float(phase)
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def _calculate_harmonic_distortion(self, harmonics: List[Dict]) -> float:
        """Calculate total harmonic distortion"""
        try:
            if len(harmonics) < 2:
                return 0.0
            
            # Find fundamental (harmonic number 1)
            fundamental_power = 0
            harmonic_power = 0
            
            for harmonic in harmonics:
                if harmonic['harmonic_number'] == 1:
                    fundamental_power = harmonic['power']
                elif harmonic['harmonic_number'] > 1:
                    harmonic_power += harmonic['power']
            
            if fundamental_power == 0:
                return 0.0
            
            thd = np.sqrt(harmonic_power) / np.sqrt(fundamental_power)
            return float(thd)
            
        except Exception as e:
            return 0.0


class MultiScaleTrendAnalyzer:
    """Multi-scale trend analysis"""
    
    def analyze(self, data: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """Perform multi-scale trend analysis"""
        try:
            # Different time scales
            scales = [
                ('short', 0.1),    # 10% of data
                ('medium', 0.3),   # 30% of data
                ('long', 0.7),     # 70% of data
                ('full', 1.0)      # Full data
            ]
            
            trend_analysis = {}
            
            for scale_name, scale_ratio in scales:
                # Calculate window size
                window_size = max(5, int(len(data) * scale_ratio))
                
                # Get data for this scale
                if scale_name == 'full':
                    scale_data = data
                else:
                    scale_data = data[-window_size:]
                
                # Fit different trend models
                x = np.arange(len(scale_data))
                
                # Linear trend
                linear_coef = np.polyfit(x, scale_data, 1)
                linear_trend = linear_coef[0]
                linear_r2 = self._calculate_r2(scale_data, np.polyval(linear_coef, x))
                
                # Exponential trend
                exp_trend, exp_r2 = self._fit_exponential_trend(scale_data)
                
                # Polynomial trend
                poly_coef = np.polyfit(x, scale_data, 2)
                poly_trend = poly_coef[0]
                poly_r2 = self._calculate_r2(scale_data, np.polyval(poly_coef, x))
                
                trend_analysis[scale_name] = {
                    'linear_trend': float(linear_trend),
                    'linear_r2': float(linear_r2),
                    'exponential_trend': float(exp_trend),
                    'exponential_r2': float(exp_r2),
                    'polynomial_trend': float(poly_trend),
                    'polynomial_r2': float(poly_r2),
                    'best_fit': self._select_best_trend(linear_r2, exp_r2, poly_r2)
                }
            
            # Overall trend characteristics
            overall_trend = self._calculate_overall_trend(data)
            
            return {
                'multi_scale_analysis': trend_analysis,
                'overall_trend': overall_trend,
                'trend_consistency': self._calculate_trend_consistency(trend_analysis),
                'trend_acceleration': self._calculate_trend_acceleration(data)
            }
            
        except Exception as e:
            logger.warning(f"Multi-scale trend analysis failed: {e}")
            return {'overall_trend': {'type': 'linear', 'slope': 0.0}}
    
    def _calculate_r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared"""
        try:
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            
            if ss_tot == 0:
                return 0.0
            
            return 1 - (ss_res / ss_tot)
            
        except Exception as e:
            return 0.0
    
    def _fit_exponential_trend(self, data: np.ndarray) -> Tuple[float, float]:
        """Fit exponential trend"""
        try:
            # Transform to log space
            positive_data = np.abs(data) + 1e-10
            log_data = np.log(positive_data)
            
            x = np.arange(len(data))
            coef = np.polyfit(x, log_data, 1)
            
            # Exponential trend coefficient
            exp_trend = coef[0]
            
            # Calculate R-squared
            fitted_log = np.polyval(coef, x)
            fitted_exp = np.exp(fitted_log)
            r2 = self._calculate_r2(data, fitted_exp)
            
            return exp_trend, r2
            
        except Exception as e:
            return 0.0, 0.0
    
    def _select_best_trend(self, linear_r2: float, exp_r2: float, poly_r2: float) -> str:
        """Select the best fitting trend model"""
        r2_values = {'linear': linear_r2, 'exponential': exp_r2, 'polynomial': poly_r2}
        return max(r2_values, key=r2_values.get)
    
    def _calculate_overall_trend(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate overall trend characteristics"""
        try:
            x = np.arange(len(data))
            
            # Linear trend
            linear_coef = np.polyfit(x, data, 1)
            linear_slope = linear_coef[0]
            linear_r2 = self._calculate_r2(data, np.polyval(linear_coef, x))
            
            # Exponential trend
            exp_slope, exp_r2 = self._fit_exponential_trend(data)
            
            # Select best trend
            if exp_r2 > linear_r2 and exp_r2 > 0.8:
                trend_type = 'exponential'
                trend_slope = exp_slope
                trend_r2 = exp_r2
            else:
                trend_type = 'linear'
                trend_slope = linear_slope
                trend_r2 = linear_r2
            
            return {
                'type': trend_type,
                'slope': float(trend_slope),
                'r2': float(trend_r2),
                'strength': float(abs(trend_slope) / np.std(data))
            }
            
        except Exception as e:
            return {'type': 'linear', 'slope': 0.0, 'r2': 0.0, 'strength': 0.0}
    
    def _calculate_trend_consistency(self, trend_analysis: Dict) -> float:
        """Calculate trend consistency across scales"""
        try:
            trends = []
            for scale_data in trend_analysis.values():
                trends.append(scale_data['linear_trend'])
            
            if len(trends) < 2:
                return 0.5
            
            # Calculate coefficient of variation
            cv = np.std(trends) / (np.mean(np.abs(trends)) + 1e-10)
            consistency = 1 / (1 + cv)
            
            return float(consistency)
            
        except Exception as e:
            return 0.5
    
    def _calculate_trend_acceleration(self, data: np.ndarray) -> float:
        """Calculate trend acceleration"""
        try:
            # Second derivative
            second_derivative = np.diff(data, n=2)
            acceleration = np.mean(second_derivative)
            
            return float(acceleration)
            
        except Exception as e:
            return 0.0


class AdvancedSeasonalAnalyzer:
    """Advanced seasonal pattern analysis"""
    
    def analyze(self, data: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """Perform advanced seasonal analysis"""
        try:
            if len(data) < 10:
                return {'seasonal_strength': 0.0, 'period': 0}
            
            # Seasonal decomposition
            seasonal_decomp = self._seasonal_decomposition(data)
            
            # Periodogram analysis
            periodogram_analysis = self._periodogram_analysis(data)
            
            # Autocorrelation-based seasonality
            autocorr_seasonality = self._autocorrelation_seasonality(data)
            
            # Combine results
            return {
                'seasonal_decomposition': seasonal_decomp,
                'periodogram_analysis': periodogram_analysis,
                'autocorrelation_seasonality': autocorr_seasonality,
                'seasonal_strength': self._calculate_seasonal_strength(seasonal_decomp),
                'dominant_period': self._find_dominant_period(periodogram_analysis, autocorr_seasonality)
            }
            
        except Exception as e:
            logger.warning(f"Seasonal analysis failed: {e}")
            return {'seasonal_strength': 0.0, 'period': 0}
    
    def _seasonal_decomposition(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform seasonal decomposition"""
        try:
            # STL decomposition
            series = pd.Series(data)
            
            # Determine period
            period = self._estimate_period(data)
            
            if period > 1 and len(data) >= 2 * period:
                stl = STL(series, seasonal=period, period=period)
                result = stl.fit()
                
                return {
                    'seasonal_component': result.seasonal.tolist(),
                    'trend_component': result.trend.tolist(),
                    'residual_component': result.resid.tolist(),
                    'period': period,
                    'seasonal_variance': float(np.var(result.seasonal)),
                    'trend_variance': float(np.var(result.trend)),
                    'residual_variance': float(np.var(result.resid))
                }
            else:
                return {'seasonal_component': [], 'period': 0}
                
        except Exception as e:
            return {'seasonal_component': [], 'period': 0}
    
    def _periodogram_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze periodogram for seasonal patterns"""
        try:
            # Calculate periodogram
            freqs, power = periodogram(data)
            
            # Find peaks
            peaks, properties = find_peaks(power, height=np.max(power) * 0.1)
            
            # Convert to periods
            periods = []
            for peak in peaks:
                if freqs[peak] > 0:
                    period = 1 / freqs[peak]
                    periods.append({
                        'period': float(period),
                        'power': float(power[peak]),
                        'frequency': float(freqs[peak])
                    })
            
            # Sort by power
            periods.sort(key=lambda x: x['power'], reverse=True)
            
            return {
                'detected_periods': periods[:5],  # Top 5
                'power_spectrum': power.tolist(),
                'frequencies': freqs.tolist()
            }
            
        except Exception as e:
            return {'detected_periods': []}
    
    def _autocorrelation_seasonality(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect seasonality using autocorrelation"""
        try:
            # Calculate autocorrelation
            max_lag = min(len(data) // 2, 100)
            autocorr = acf(data, nlags=max_lag, fft=True)
            
            # Find peaks in autocorrelation
            peaks, _ = find_peaks(autocorr[1:], height=0.1, distance=2)
            peaks = peaks + 1  # Adjust for slicing
            
            # Convert to periods and strengths
            seasonal_periods = []
            for peak in peaks:
                seasonal_periods.append({
                    'period': int(peak),
                    'strength': float(autocorr[peak])
                })
            
            # Sort by strength
            seasonal_periods.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'autocorr_periods': seasonal_periods[:5],  # Top 5
                'autocorr_values': autocorr.tolist()
            }
            
        except Exception as e:
            return {'autocorr_periods': []}
    
    def _estimate_period(self, data: np.ndarray) -> int:
        """Estimate the dominant period"""
        try:
            # Try different methods
            
            # Method 1: Autocorrelation
            max_lag = min(len(data) // 3, 50)
            autocorr = acf(data, nlags=max_lag, fft=True)
            
            # Find the first significant peak
            for lag in range(2, len(autocorr)):
                if autocorr[lag] > 0.3:  # Significant correlation
                    return lag
            
            # Method 2: FFT
            fft_data = fft.fft(data)
            freqs = fft.fftfreq(len(data))
            
            # Find dominant frequency
            power = np.abs(fft_data) ** 2
            dominant_idx = np.argmax(power[1:len(power)//2]) + 1
            
            if freqs[dominant_idx] > 0:
                period = int(1 / freqs[dominant_idx])
                if 2 <= period <= len(data) // 3:
                    return period
            
            # Default
            return min(12, len(data) // 4)
            
        except Exception as e:
            return 7  # Default weekly period
    
    def _calculate_seasonal_strength(self, seasonal_decomp: Dict) -> float:
        """Calculate seasonal strength"""
        try:
            if 'seasonal_variance' not in seasonal_decomp:
                return 0.0
            
            seasonal_var = seasonal_decomp['seasonal_variance']
            residual_var = seasonal_decomp.get('residual_variance', 1.0)
            
            if residual_var == 0:
                return 1.0
            
            strength = seasonal_var / (seasonal_var + residual_var)
            return float(strength)
            
        except Exception as e:
            return 0.0
    
    def _find_dominant_period(self, periodogram_analysis: Dict, autocorr_seasonality: Dict) -> int:
        """Find the dominant seasonal period"""
        try:
            # From periodogram
            periodogram_periods = periodogram_analysis.get('detected_periods', [])
            
            # From autocorrelation
            autocorr_periods = autocorr_seasonality.get('autocorr_periods', [])
            
            # Combine and find best
            all_periods = []
            
            for period_info in periodogram_periods:
                period = period_info['period']
                if 2 <= period <= 100:  # Reasonable range
                    all_periods.append((period, period_info['power']))
            
            for period_info in autocorr_periods:
                period = period_info['period']
                if 2 <= period <= 100:
                    all_periods.append((period, period_info['strength']))
            
            if all_periods:
                # Sort by strength/power
                all_periods.sort(key=lambda x: x[1], reverse=True)
                return int(all_periods[0][0])
            else:
                return 0
                
        except Exception as e:
            return 0


class PatternClassifier:
    """Pattern classification system"""
    
    def classify(self, data: np.ndarray, analysis_results: Dict) -> Dict[str, Any]:
        """Classify the pattern type based on analysis results"""
        try:
            # Extract features from analysis results
            features = self._extract_features(data, analysis_results)
            
            # Pattern classification rules
            pattern_scores = self._calculate_pattern_scores(features)
            
            # Select primary pattern
            primary_pattern = max(pattern_scores, key=pattern_scores.get)
            
            # Calculate confidence
            confidence = self._calculate_classification_confidence(pattern_scores)
            
            return {
                'primary_pattern': primary_pattern,
                'pattern_scores': pattern_scores,
                'confidence': float(confidence),
                'pattern_strength': float(pattern_scores[primary_pattern]),
                'features': features
            }
            
        except Exception as e:
            logger.warning(f"Pattern classification failed: {e}")
            return {'primary_pattern': 'unknown', 'confidence': 0.5}
    
    def _extract_features(self, data: np.ndarray, analysis_results: Dict) -> Dict[str, Any]:
        """Extract features for pattern classification"""
        try:
            features = {}
            
            # Trend features
            trend_analysis = analysis_results.get('trend', {})
            if trend_analysis:
                features['trend_slope'] = trend_analysis.get('overall_trend', {}).get('slope', 0.0)
                features['trend_r2'] = trend_analysis.get('overall_trend', {}).get('r2', 0.0)
                features['trend_type'] = trend_analysis.get('overall_trend', {}).get('type', 'linear')
            
            # Frequency features
            frequency_analysis = analysis_results.get('frequency', {})
            if frequency_analysis:
                dominant_freqs = frequency_analysis.get('dominant_frequencies', [])
                features['dominant_frequency'] = dominant_freqs[0]['frequency'] if dominant_freqs else 0.0
                features['spectral_centroid'] = frequency_analysis.get('spectral_centroid', 0.0)
                features['spectral_bandwidth'] = frequency_analysis.get('spectral_bandwidth', 0.0)
            
            # Seasonal features
            seasonal_analysis = analysis_results.get('seasonal', {})
            if seasonal_analysis:
                features['seasonal_strength'] = seasonal_analysis.get('seasonal_strength', 0.0)
                features['seasonal_period'] = seasonal_analysis.get('dominant_period', 0)
            
            # Harmonic features
            harmonic_analysis = analysis_results.get('harmonic', {})
            if harmonic_analysis:
                features['harmonic_count'] = harmonic_analysis.get('harmonic_count', 0)
                features['harmonic_distortion'] = harmonic_analysis.get('harmonic_distortion', 0.0)
                features['fundamental_frequency'] = harmonic_analysis.get('fundamental_frequency', 0.0)
            
            # Statistical features
            features['mean'] = float(np.mean(data))
            features['std'] = float(np.std(data))
            features['skewness'] = float(stats.skew(data))
            features['kurtosis'] = float(stats.kurtosis(data))
            
            # Variability features
            features['coefficient_of_variation'] = features['std'] / (features['mean'] + 1e-10)
            features['range'] = float(np.max(data) - np.min(data))
            
            return features
            
        except Exception as e:
            return {}
    
    def _calculate_pattern_scores(self, features: Dict) -> Dict[str, float]:
        """Calculate scores for different pattern types including advanced pattern detection"""
        try:
            scores = {}
            
            # Basic pattern types
            scores['linear'] = self._score_linear_pattern(features)
            scores['exponential'] = self._score_exponential_pattern(features)
            scores['sinusoidal'] = self._score_sinusoidal_pattern(features)
            scores['seasonal'] = self._score_seasonal_pattern(features)
            scores['periodic'] = self._score_periodic_pattern(features)
            scores['trending'] = self._score_trending_pattern(features)
            scores['random_walk'] = self._score_random_walk_pattern(features)
            scores['white_noise'] = self._score_white_noise_pattern(features)
            
            # Advanced pattern types for comprehensive pattern learning
            scores['quadratic'] = self._score_quadratic_pattern(features)
            scores['cubic'] = self._score_cubic_pattern(features)
            scores['polynomial'] = self._score_polynomial_pattern(features)
            scores['spline'] = self._score_spline_pattern(features)
            scores['custom_shape'] = self._score_custom_shape_pattern(features)
            scores['composite'] = self._score_composite_pattern(features)
            
            return scores
            
        except Exception as e:
            return {'unknown': 0.5}
    
    def _score_linear_pattern(self, features: Dict) -> float:
        """Score linear pattern"""
        try:
            trend_r2 = features.get('trend_r2', 0.0)
            trend_type = features.get('trend_type', 'linear')
            seasonal_strength = features.get('seasonal_strength', 0.0)
            
            score = 0.0
            
            if trend_type == 'linear':
                score += 0.5
            
            score += trend_r2 * 0.4
            score -= seasonal_strength * 0.1  # Linear patterns have low seasonality
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            return 0.0
    
    def _score_exponential_pattern(self, features: Dict) -> float:
        """Score exponential pattern"""
        try:
            trend_type = features.get('trend_type', 'linear')
            trend_r2 = features.get('trend_r2', 0.0)
            coefficient_of_variation = features.get('coefficient_of_variation', 0.0)
            
            score = 0.0
            
            if trend_type == 'exponential':
                score += 0.6
            
            score += trend_r2 * 0.3
            score += min(coefficient_of_variation, 1.0) * 0.1  # Exponential patterns have increasing variance
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            return 0.0
    
    def _score_sinusoidal_pattern(self, features: Dict) -> float:
        """Score sinusoidal pattern"""
        try:
            harmonic_count = features.get('harmonic_count', 0)
            fundamental_frequency = features.get('fundamental_frequency', 0.0)
            harmonic_distortion = features.get('harmonic_distortion', 0.0)
            seasonal_strength = features.get('seasonal_strength', 0.0)
            
            score = 0.0
            
            if harmonic_count > 0:
                score += 0.3
            
            if fundamental_frequency > 0:
                score += 0.3
            
            score += seasonal_strength * 0.2
            score -= harmonic_distortion * 0.2  # Pure sinusoidal patterns have low distortion
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            return 0.0
    
    def _score_seasonal_pattern(self, features: Dict) -> float:
        """Score seasonal pattern"""
        try:
            seasonal_strength = features.get('seasonal_strength', 0.0)
            seasonal_period = features.get('seasonal_period', 0)
            
            score = 0.0
            
            score += seasonal_strength * 0.7
            
            if seasonal_period > 0:
                score += 0.3
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            return 0.0
    
    def _score_periodic_pattern(self, features: Dict) -> float:
        """Score periodic pattern"""
        try:
            dominant_frequency = features.get('dominant_frequency', 0.0)
            spectral_centroid = features.get('spectral_centroid', 0.0)
            seasonal_strength = features.get('seasonal_strength', 0.0)
            
            score = 0.0
            
            if dominant_frequency > 0:
                score += 0.4
            
            if spectral_centroid > 0:
                score += 0.3
            
            score += seasonal_strength * 0.3
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            return 0.0
    
    def _score_trending_pattern(self, features: Dict) -> float:
        """Score trending pattern"""
        try:
            trend_slope = abs(features.get('trend_slope', 0.0))
            trend_r2 = features.get('trend_r2', 0.0)
            
            score = 0.0
            
            if trend_slope > 0:
                score += 0.5
            
            score += trend_r2 * 0.5
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            return 0.0
    
    def _score_random_walk_pattern(self, features: Dict) -> float:
        """Score random walk pattern"""
        try:
            trend_r2 = features.get('trend_r2', 0.0)
            seasonal_strength = features.get('seasonal_strength', 0.0)
            coefficient_of_variation = features.get('coefficient_of_variation', 0.0)
            
            score = 0.0
            
            score += (1 - trend_r2) * 0.4  # Random walks have low trend fit
            score += (1 - seasonal_strength) * 0.3  # Random walks have low seasonality
            score += min(coefficient_of_variation, 1.0) * 0.3  # Random walks have high variability
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            return 0.0
    
    def _score_white_noise_pattern(self, features: Dict) -> float:
        """Score white noise pattern"""
        try:
            trend_r2 = features.get('trend_r2', 0.0)
            seasonal_strength = features.get('seasonal_strength', 0.0)
            dominant_frequency = features.get('dominant_frequency', 0.0)
            
            score = 0.0
            
            score += (1 - trend_r2) * 0.4  # White noise has no trend
            score += (1 - seasonal_strength) * 0.3  # White noise has no seasonality
            score += (1 - min(dominant_frequency, 1.0)) * 0.3  # White noise has no dominant frequency
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            return 0.0
    
    def _calculate_classification_confidence(self, pattern_scores: Dict) -> float:
        """Calculate confidence in classification"""
        try:
            scores = list(pattern_scores.values())
            scores.sort(reverse=True)
            
            if len(scores) < 2:
                return 0.5
            
            # Confidence based on separation between top two scores
            confidence = (scores[0] - scores[1]) / (scores[0] + scores[1] + 1e-10)
            
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            return 0.5