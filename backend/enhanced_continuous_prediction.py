"""
Enhanced Continuous Prediction System
Advanced real-time prediction system that maintains historical patterns
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedContinuousPredictionSystem:
    """
    Enhanced continuous prediction system that maintains historical patterns
    and prevents bias accumulation over time
    """
    
    def __init__(self):
        self.historical_memory = {}
        self.prediction_memory = []
        self.pattern_memory = {}
        self.bias_tracking = []
        self.variability_tracking = []
        
        # Enhanced parameters for better pattern following
        self.pattern_preservation_strength = 0.8  # Increased from 0.7
        self.variability_preservation_strength = 0.9  # Increased from 0.8
        self.bias_correction_strength = 0.5  # Increased from 0.3
        self.continuity_strength = 0.6  # Increased from 0.5
        
        # Pattern following parameters
        self.trend_momentum_preservation = 0.7
        self.cyclical_pattern_preservation = 0.8
        self.volatility_pattern_preservation = 0.75
        
        # Bias prevention parameters
        self.bias_accumulation_threshold = 0.1
        self.bias_correction_decay = 0.02  # Slower decay for better correction
        self.max_bias_correction = 0.3
        
    def generate_enhanced_continuous_predictions(self, 
                                               historical_data: np.ndarray,
                                               steps: int = 30,
                                               prediction_offset: int = 0,
                                               previous_predictions: Optional[List] = None) -> Dict[str, Any]:
        """
        Generate enhanced continuous predictions that maintain historical patterns
        
        Args:
            historical_data: Historical time series data
            steps: Number of prediction steps
            prediction_offset: Offset for continuous predictions
            previous_predictions: Previous predictions for continuity
            
        Returns:
            Enhanced predictions with strong pattern following
        """
        try:
            # Analyze historical patterns
            patterns = self._analyze_comprehensive_patterns(historical_data)
            
            # Store pattern memory for continuity
            self.pattern_memory = patterns
            
            # Generate base predictions using multiple methods
            base_predictions = self._generate_pattern_aware_predictions(
                historical_data, steps, patterns, prediction_offset
            )
            
            # Apply enhanced continuous corrections
            enhanced_predictions = self._apply_enhanced_continuous_corrections(
                base_predictions, historical_data, patterns, previous_predictions
            )
            
            # Ensure pattern preservation
            preserved_predictions = self._ensure_enhanced_pattern_preservation(
                enhanced_predictions, historical_data, patterns
            )
            
            # Apply bias prevention
            bias_prevented_predictions = self._apply_bias_prevention(
                preserved_predictions, historical_data, patterns
            )
            
            # Calculate enhanced quality metrics
            quality_metrics = self._calculate_enhanced_quality_metrics(
                bias_prevented_predictions, historical_data, patterns
            )
            
            # Update memory systems
            self._update_prediction_memory(bias_prevented_predictions, patterns)
            
            return {
                'predictions': bias_prevented_predictions.tolist(),
                'quality_metrics': quality_metrics,
                'pattern_analysis': patterns,
                'pattern_following_score': quality_metrics.get('pattern_following_score', 0.5),
                'variability_preservation': quality_metrics.get('variability_preservation', 0.5),
                'bias_prevention_score': quality_metrics.get('bias_prevention_score', 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced continuous prediction: {e}")
            return self._generate_fallback_predictions(historical_data, steps)
    
    def _analyze_comprehensive_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze comprehensive patterns for enhanced prediction"""
        try:
            patterns = {}
            
            # 1. Statistical patterns
            patterns['statistical'] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
            
            # 2. Trend patterns (multiple timescales)
            patterns['trends'] = self._analyze_multi_scale_trends(data)
            
            # 3. Cyclical patterns
            patterns['cyclical'] = self._analyze_cyclical_patterns(data)
            
            # 4. Volatility patterns
            patterns['volatility'] = self._analyze_volatility_patterns(data)
            
            # 5. Local patterns
            patterns['local'] = self._analyze_local_patterns(data)
            
            # 6. Frequency domain patterns
            patterns['frequency'] = self._analyze_frequency_patterns(data)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return self._get_default_patterns(data)
    
    def _generate_pattern_aware_predictions(self, data: np.ndarray, steps: int, 
                                          patterns: Dict, offset: int = 0) -> np.ndarray:
        """Generate predictions that are aware of historical patterns"""
        try:
            # Multiple prediction methods
            predictions = {}
            
            # 1. Trend-based prediction with pattern awareness
            predictions['trend'] = self._generate_trend_aware_predictions(data, steps, patterns)
            
            # 2. Cyclical pattern continuation
            predictions['cyclical'] = self._generate_cyclical_predictions(data, steps, patterns)
            
            # 3. Volatility-aware prediction
            predictions['volatility'] = self._generate_volatility_aware_predictions(data, steps, patterns)
            
            # 4. Local pattern continuation
            predictions['local'] = self._generate_local_pattern_predictions(data, steps, patterns)
            
            # 5. Frequency domain based prediction
            predictions['frequency'] = self._generate_frequency_based_predictions(data, steps, patterns)
            
            # 6. Adaptive weighted combination
            combined_predictions = self._combine_predictions_adaptively(predictions, patterns)
            
            return combined_predictions
            
        except Exception as e:
            logger.error(f"Error in pattern-aware prediction generation: {e}")
            return self._generate_simple_predictions(data, steps)
    
    def _apply_enhanced_continuous_corrections(self, predictions: np.ndarray,
                                             data: np.ndarray, patterns: Dict,
                                             previous_predictions: Optional[List] = None) -> np.ndarray:
        """Apply enhanced corrections for continuous predictions"""
        try:
            corrected_predictions = predictions.copy()
            
            # 1. Variability preservation correction
            corrected_predictions = self._apply_variability_preservation(
                corrected_predictions, data, patterns
            )
            
            # 2. Trend momentum preservation
            corrected_predictions = self._apply_trend_momentum_preservation(
                corrected_predictions, data, patterns
            )
            
            # 3. Cyclical pattern preservation
            corrected_predictions = self._apply_cyclical_preservation(
                corrected_predictions, data, patterns
            )
            
            # 4. Continuity preservation (with previous predictions)
            if previous_predictions is not None:
                corrected_predictions = self._apply_continuity_preservation(
                    corrected_predictions, data, previous_predictions
                )
            
            # 5. Boundary preservation
            corrected_predictions = self._apply_boundary_preservation(
                corrected_predictions, data, patterns
            )
            
            return corrected_predictions
            
        except Exception as e:
            logger.error(f"Error in enhanced continuous corrections: {e}")
            return predictions
    
    def _ensure_enhanced_pattern_preservation(self, predictions: np.ndarray,
                                            data: np.ndarray, patterns: Dict) -> np.ndarray:
        """Ensure enhanced pattern preservation in predictions"""
        try:
            preserved_predictions = predictions.copy()
            
            # 1. Statistical characteristic preservation
            preserved_predictions = self._preserve_statistical_characteristics(
                preserved_predictions, data, patterns
            )
            
            # 2. Trend characteristic preservation
            preserved_predictions = self._preserve_trend_characteristics(
                preserved_predictions, data, patterns
            )
            
            # 3. Cyclical characteristic preservation
            preserved_predictions = self._preserve_cyclical_characteristics(
                preserved_predictions, data, patterns
            )
            
            # 4. Volatility characteristic preservation
            preserved_predictions = self._preserve_volatility_characteristics(
                preserved_predictions, data, patterns
            )
            
            return preserved_predictions
            
        except Exception as e:
            logger.error(f"Error in enhanced pattern preservation: {e}")
            return predictions
    
    def _apply_bias_prevention(self, predictions: np.ndarray,
                              data: np.ndarray, patterns: Dict) -> np.ndarray:
        """Apply bias prevention to avoid accumulation over time"""
        try:
            bias_prevented_predictions = predictions.copy()
            
            # 1. Detect potential bias
            bias_metrics = self._detect_bias(predictions, data, patterns)
            
            # 2. Apply bias correction if needed
            if bias_metrics['bias_detected']:
                bias_prevented_predictions = self._apply_bias_correction(
                    bias_prevented_predictions, data, patterns, bias_metrics
                )
            
            # 3. Track bias for future corrections
            self.bias_tracking.append(bias_metrics)
            
            # 4. Apply accumulated bias prevention
            if len(self.bias_tracking) > 3:
                bias_prevented_predictions = self._apply_accumulated_bias_prevention(
                    bias_prevented_predictions, data
                )
            
            return bias_prevented_predictions
            
        except Exception as e:
            logger.error(f"Error in bias prevention: {e}")
            return predictions
    
    def _apply_variability_preservation(self, predictions: np.ndarray,
                                      data: np.ndarray, patterns: Dict) -> np.ndarray:
        """Apply enhanced variability preservation"""
        try:
            historical_std = patterns['statistical']['std']
            prediction_std = np.std(predictions)
            
            # Target variability should be close to historical
            target_std = historical_std * self.variability_preservation_strength
            
            if prediction_std < 0.3 * target_std:
                # Predictions too flat - enhance variability
                enhancement_factor = min(3.0, target_std / (prediction_std + 1e-10))
                prediction_mean = np.mean(predictions)
                predictions = prediction_mean + (predictions - prediction_mean) * enhancement_factor
                
            elif prediction_std > 2.0 * target_std:
                # Predictions too volatile - reduce variability
                reduction_factor = max(0.5, target_std / prediction_std)
                prediction_mean = np.mean(predictions)
                predictions = prediction_mean + (predictions - prediction_mean) * reduction_factor
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in variability preservation: {e}")
            return predictions
    
    def _apply_trend_momentum_preservation(self, predictions: np.ndarray,
                                         data: np.ndarray, patterns: Dict) -> np.ndarray:
        """Apply trend momentum preservation"""
        try:
            historical_trend = patterns['trends'].get('recent_trend', 0)
            
            # Calculate prediction trend
            if len(predictions) > 1:
                x = np.arange(len(predictions))
                prediction_trend = np.polyfit(x, predictions, 1)[0]
                
                # Apply trend momentum preservation
                trend_difference = historical_trend - prediction_trend
                if abs(trend_difference) > 0.1 * patterns['statistical']['std']:
                    # Apply gradual trend correction
                    trend_correction = trend_difference * self.trend_momentum_preservation
                    correction_weights = np.linspace(0, 1, len(predictions))
                    predictions += trend_correction * correction_weights
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in trend momentum preservation: {e}")
            return predictions
    
    def _apply_cyclical_preservation(self, predictions: np.ndarray,
                                   data: np.ndarray, patterns: Dict) -> np.ndarray:
        """Apply cyclical pattern preservation"""
        try:
            cyclical_patterns = patterns.get('cyclical', {})
            
            if cyclical_patterns.get('dominant_period'):
                period = cyclical_patterns['dominant_period']
                amplitude = cyclical_patterns.get('amplitude', 0.1 * patterns['statistical']['std'])
                
                # Apply cyclical correction
                t = np.arange(len(predictions))
                cyclical_component = amplitude * np.sin(2 * np.pi * t / period)
                
                # Blend with predictions
                blend_factor = self.cyclical_pattern_preservation * cyclical_patterns.get('strength', 0.5)
                predictions += blend_factor * cyclical_component
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in cyclical preservation: {e}")
            return predictions
    
    def _detect_bias(self, predictions: np.ndarray, data: np.ndarray, patterns: Dict) -> Dict[str, Any]:
        """Detect potential bias in predictions"""
        try:
            bias_metrics = {
                'bias_detected': False,
                'mean_bias': 0.0,
                'trend_bias': 0.0,
                'variability_bias': 0.0
            }
            
            # 1. Mean bias detection
            historical_mean = patterns['statistical']['mean']
            prediction_mean = np.mean(predictions)
            mean_bias = prediction_mean - historical_mean
            bias_metrics['mean_bias'] = mean_bias
            
            # 2. Trend bias detection
            historical_trend = patterns['trends'].get('recent_trend', 0)
            if len(predictions) > 1:
                x = np.arange(len(predictions))
                prediction_trend = np.polyfit(x, predictions, 1)[0]
                trend_bias = prediction_trend - historical_trend
                bias_metrics['trend_bias'] = trend_bias
            
            # 3. Variability bias detection
            historical_std = patterns['statistical']['std']
            prediction_std = np.std(predictions)
            variability_bias = (prediction_std - historical_std) / historical_std
            bias_metrics['variability_bias'] = variability_bias
            
            # Determine if bias correction is needed
            if (abs(mean_bias) > self.bias_accumulation_threshold * historical_std or
                abs(bias_metrics['trend_bias']) > 0.1 * historical_std or
                abs(variability_bias) > 0.5):
                bias_metrics['bias_detected'] = True
            
            return bias_metrics
            
        except Exception as e:
            logger.error(f"Error in bias detection: {e}")
            return {'bias_detected': False, 'mean_bias': 0.0, 'trend_bias': 0.0, 'variability_bias': 0.0}
    
    def _apply_bias_correction(self, predictions: np.ndarray, data: np.ndarray,
                              patterns: Dict, bias_metrics: Dict) -> np.ndarray:
        """Apply bias correction to predictions"""
        try:
            corrected_predictions = predictions.copy()
            
            # 1. Mean bias correction
            if abs(bias_metrics['mean_bias']) > 0.05 * patterns['statistical']['std']:
                mean_correction = -bias_metrics['mean_bias'] * self.bias_correction_strength
                corrected_predictions += mean_correction
            
            # 2. Trend bias correction
            if abs(bias_metrics['trend_bias']) > 0.05 * patterns['statistical']['std']:
                trend_correction = -bias_metrics['trend_bias'] * self.bias_correction_strength
                correction_weights = np.linspace(0, 1, len(predictions))
                corrected_predictions += trend_correction * correction_weights
            
            # 3. Variability bias correction
            if abs(bias_metrics['variability_bias']) > 0.3:
                target_std = patterns['statistical']['std']
                current_std = np.std(corrected_predictions)
                if current_std > 0:
                    correction_factor = 1.0 + (target_std - current_std) / current_std * 0.3
                    prediction_mean = np.mean(corrected_predictions)
                    corrected_predictions = prediction_mean + (corrected_predictions - prediction_mean) * correction_factor
            
            return corrected_predictions
            
        except Exception as e:
            logger.error(f"Error in bias correction: {e}")
            return predictions
    
    def _calculate_enhanced_quality_metrics(self, predictions: np.ndarray,
                                          data: np.ndarray, patterns: Dict) -> Dict[str, Any]:
        """Calculate enhanced quality metrics for pattern following"""
        try:
            metrics = {}
            
            # 1. Pattern following score
            metrics['pattern_following_score'] = self._calculate_pattern_following_score(
                predictions, data, patterns
            )
            
            # 2. Variability preservation score
            metrics['variability_preservation'] = self._calculate_variability_preservation_score(
                predictions, data, patterns
            )
            
            # 3. Trend preservation score
            metrics['trend_preservation'] = self._calculate_trend_preservation_score(
                predictions, data, patterns
            )
            
            # 4. Bias prevention score
            metrics['bias_prevention_score'] = self._calculate_bias_prevention_score(
                predictions, data, patterns
            )
            
            # 5. Overall quality score
            metrics['overall_quality_score'] = np.mean([
                metrics['pattern_following_score'],
                metrics['variability_preservation'],
                metrics['trend_preservation'],
                metrics['bias_prevention_score']
            ])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in quality metrics calculation: {e}")
            return {'overall_quality_score': 0.5}
    
    def _calculate_pattern_following_score(self, predictions: np.ndarray,
                                         data: np.ndarray, patterns: Dict) -> float:
        """Calculate pattern following score"""
        try:
            score = 0.0
            components = []
            
            # 1. Statistical similarity
            pred_mean = np.mean(predictions)
            hist_mean = patterns['statistical']['mean']
            mean_similarity = 1.0 - min(1.0, abs(pred_mean - hist_mean) / (patterns['statistical']['std'] + 1e-10))
            components.append(mean_similarity)
            
            # 2. Variability similarity
            pred_std = np.std(predictions)
            hist_std = patterns['statistical']['std']
            std_similarity = 1.0 - min(1.0, abs(pred_std - hist_std) / (hist_std + 1e-10))
            components.append(std_similarity)
            
            # 3. Trend similarity
            if len(predictions) > 1:
                x = np.arange(len(predictions))
                pred_trend = np.polyfit(x, predictions, 1)[0]
                hist_trend = patterns['trends'].get('recent_trend', 0)
                trend_similarity = 1.0 - min(1.0, abs(pred_trend - hist_trend) / (hist_std + 1e-10))
                components.append(trend_similarity)
            
            # 4. Range similarity
            pred_range = np.max(predictions) - np.min(predictions)
            hist_range = patterns['statistical']['range']
            range_similarity = 1.0 - min(1.0, abs(pred_range - hist_range) / (hist_range + 1e-10))
            components.append(range_similarity)
            
            score = np.mean(components)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error in pattern following score calculation: {e}")
            return 0.5
    
    def _update_prediction_memory(self, predictions: np.ndarray, patterns: Dict):
        """Update prediction memory for future use"""
        try:
            memory_entry = {
                'predictions': predictions.tolist(),
                'patterns': patterns,
                'timestamp': datetime.now(),
                'quality_score': self._calculate_pattern_following_score(predictions, np.array([]), patterns)
            }
            
            self.prediction_memory.append(memory_entry)
            
            # Keep only recent memory
            if len(self.prediction_memory) > 10:
                self.prediction_memory = self.prediction_memory[-10:]
                
        except Exception as e:
            logger.error(f"Error in updating prediction memory: {e}")
    
    # Helper methods for pattern analysis
    def _analyze_multi_scale_trends(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze trends at multiple time scales"""
        try:
            trends = {}
            
            # Recent trend (last 5 points)
            if len(data) >= 5:
                recent_data = data[-5:]
                x = np.arange(len(recent_data))
                trends['recent_trend'] = np.polyfit(x, recent_data, 1)[0]
            
            # Medium trend (last 10 points)
            if len(data) >= 10:
                medium_data = data[-10:]
                x = np.arange(len(medium_data))
                trends['medium_trend'] = np.polyfit(x, medium_data, 1)[0]
            
            # Long trend (all data)
            if len(data) >= 3:
                x = np.arange(len(data))
                trends['long_trend'] = np.polyfit(x, data, 1)[0]
            
            return trends
            
        except Exception as e:
            logger.error(f"Error in multi-scale trend analysis: {e}")
            return {'recent_trend': 0, 'medium_trend': 0, 'long_trend': 0}
    
    def _analyze_cyclical_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze cyclical patterns in data"""
        try:
            cyclical = {}
            
            # Autocorrelation analysis
            if len(data) > 10:
                autocorr = np.correlate(data, data, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                # Find peaks
                peaks, _ = signal.find_peaks(autocorr[:min(len(data)//2, 50)], height=0.3 * np.max(autocorr))
                
                if len(peaks) > 0:
                    cyclical['dominant_period'] = peaks[0]
                    cyclical['strength'] = autocorr[peaks[0]] / np.max(autocorr)
                    cyclical['amplitude'] = np.std(data) * cyclical['strength']
                else:
                    cyclical['dominant_period'] = None
                    cyclical['strength'] = 0.0
                    cyclical['amplitude'] = 0.0
            
            return cyclical
            
        except Exception as e:
            logger.error(f"Error in cyclical pattern analysis: {e}")
            return {'dominant_period': None, 'strength': 0.0, 'amplitude': 0.0}
    
    def _analyze_volatility_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility patterns in data"""
        try:
            volatility = {}
            
            # Rolling volatility
            if len(data) >= 5:
                window_size = max(3, len(data) // 5)
                volatility_values = []
                
                for i in range(window_size, len(data)):
                    window_data = data[i-window_size:i]
                    volatility_values.append(np.std(window_data))
                
                if volatility_values:
                    volatility['mean_volatility'] = np.mean(volatility_values)
                    volatility['volatility_trend'] = np.polyfit(np.arange(len(volatility_values)), volatility_values, 1)[0]
                else:
                    volatility['mean_volatility'] = np.std(data)
                    volatility['volatility_trend'] = 0.0
            else:
                volatility['mean_volatility'] = np.std(data)
                volatility['volatility_trend'] = 0.0
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error in volatility pattern analysis: {e}")
            return {'mean_volatility': 0.0, 'volatility_trend': 0.0}
    
    def _analyze_local_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze local patterns in data"""
        try:
            local = {}
            
            # Recent behavior
            if len(data) >= 5:
                recent_data = data[-5:]
                local['recent_mean'] = np.mean(recent_data)
                local['recent_std'] = np.std(recent_data)
                local['recent_trend'] = np.polyfit(np.arange(len(recent_data)), recent_data, 1)[0]
                
                # Change points detection
                changes = np.diff(recent_data)
                local['change_frequency'] = np.sum(np.abs(changes) > 0.5 * np.std(data))
                local['change_magnitude'] = np.mean(np.abs(changes))
            else:
                local['recent_mean'] = np.mean(data)
                local['recent_std'] = np.std(data)
                local['recent_trend'] = 0.0
                local['change_frequency'] = 0
                local['change_magnitude'] = 0.0
            
            return local
            
        except Exception as e:
            logger.error(f"Error in local pattern analysis: {e}")
            return {'recent_mean': 0.0, 'recent_std': 0.0, 'recent_trend': 0.0, 'change_frequency': 0, 'change_magnitude': 0.0}
    
    def _analyze_frequency_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency domain patterns"""
        try:
            frequency = {}
            
            # FFT analysis
            if len(data) >= 8:
                fft_data = np.fft.fft(data)
                frequencies = np.fft.fftfreq(len(data))
                
                # Find dominant frequencies
                power_spectrum = np.abs(fft_data)
                dominant_freq_idx = np.argmax(power_spectrum[1:len(data)//2]) + 1
                
                frequency['dominant_frequency'] = frequencies[dominant_freq_idx]
                frequency['dominant_power'] = power_spectrum[dominant_freq_idx]
                frequency['total_power'] = np.sum(power_spectrum)
                frequency['frequency_concentration'] = frequency['dominant_power'] / frequency['total_power']
            else:
                frequency['dominant_frequency'] = 0.0
                frequency['dominant_power'] = 0.0
                frequency['total_power'] = 0.0
                frequency['frequency_concentration'] = 0.0
            
            return frequency
            
        except Exception as e:
            logger.error(f"Error in frequency pattern analysis: {e}")
            return {'dominant_frequency': 0.0, 'dominant_power': 0.0, 'total_power': 0.0, 'frequency_concentration': 0.0}
    
    def _get_default_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Get default patterns when analysis fails"""
        try:
            return {
                'statistical': {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'range': np.max(data) - np.min(data),
                    'skewness': 0.0,
                    'kurtosis': 0.0
                },
                'trends': {'recent_trend': 0.0, 'medium_trend': 0.0, 'long_trend': 0.0},
                'cyclical': {'dominant_period': None, 'strength': 0.0, 'amplitude': 0.0},
                'volatility': {'mean_volatility': np.std(data), 'volatility_trend': 0.0},
                'local': {'recent_mean': np.mean(data), 'recent_std': np.std(data), 'recent_trend': 0.0, 'change_frequency': 0, 'change_magnitude': 0.0},
                'frequency': {'dominant_frequency': 0.0, 'dominant_power': 0.0, 'total_power': 0.0, 'frequency_concentration': 0.0}
            }
        except Exception as e:
            logger.error(f"Error in default patterns: {e}")
            return {}

    def _generate_trend_aware_predictions(self, data: np.ndarray, steps: int, patterns: Dict) -> np.ndarray:
        """Generate trend-aware predictions"""
        try:
            # Get trend information
            recent_trend = patterns['trends'].get('recent_trend', 0)
            medium_trend = patterns['trends'].get('medium_trend', 0)
            long_trend = patterns['trends'].get('long_trend', 0)
            
            # Weighted trend combination
            trend_weights = [0.5, 0.3, 0.2]  # Recent, medium, long
            combined_trend = (recent_trend * trend_weights[0] + 
                            medium_trend * trend_weights[1] + 
                            long_trend * trend_weights[2])
            
            # Generate predictions
            predictions = []
            last_value = data[-1]
            
            for i in range(steps):
                # Apply trend with decay
                trend_decay = np.exp(-0.1 * i)  # Gradual trend decay
                next_value = last_value + combined_trend * (i + 1) * trend_decay
                predictions.append(next_value)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in trend-aware predictions: {e}")
            return np.full(steps, data[-1])
    
    def _generate_cyclical_predictions(self, data: np.ndarray, steps: int, patterns: Dict) -> np.ndarray:
        """Generate cyclical predictions"""
        try:
            cyclical_patterns = patterns.get('cyclical', {})
            
            if cyclical_patterns.get('dominant_period') and cyclical_patterns.get('strength', 0) > 0.3:
                period = cyclical_patterns['dominant_period']
                amplitude = cyclical_patterns['amplitude']
                
                # Generate cyclical predictions
                predictions = []
                last_value = data[-1]
                
                for i in range(steps):
                    phase = 2 * np.pi * i / period
                    cyclical_component = amplitude * np.sin(phase)
                    predictions.append(last_value + cyclical_component)
                
                return np.array(predictions)
            else:
                # No strong cyclical pattern, use simple continuation
                return np.full(steps, data[-1])
                
        except Exception as e:
            logger.error(f"Error in cyclical predictions: {e}")
            return np.full(steps, data[-1])
    
    def _generate_volatility_aware_predictions(self, data: np.ndarray, steps: int, patterns: Dict) -> np.ndarray:
        """Generate volatility-aware predictions"""
        try:
            volatility_patterns = patterns.get('volatility', {})
            mean_volatility = volatility_patterns.get('mean_volatility', np.std(data))
            
            # Generate predictions with appropriate volatility
            predictions = []
            last_value = data[-1]
            
            for i in range(steps):
                # Add volatility-based noise
                noise = np.random.normal(0, mean_volatility * 0.5)
                # Apply volatility decay
                volatility_decay = np.exp(-0.05 * i)
                next_value = last_value + noise * volatility_decay
                predictions.append(next_value)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in volatility-aware predictions: {e}")
            return np.full(steps, data[-1])
    
    def _generate_local_pattern_predictions(self, data: np.ndarray, steps: int, patterns: Dict) -> np.ndarray:
        """Generate predictions based on local patterns"""
        try:
            local_patterns = patterns.get('local', {})
            recent_trend = local_patterns.get('recent_trend', 0)
            change_magnitude = local_patterns.get('change_magnitude', 0)
            
            # Generate predictions based on recent behavior
            predictions = []
            last_value = data[-1]
            
            for i in range(steps):
                # Apply local trend with some randomness
                local_change = recent_trend + np.random.normal(0, change_magnitude * 0.3)
                next_value = last_value + local_change * (i + 1)
                predictions.append(next_value)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in local pattern predictions: {e}")
            return np.full(steps, data[-1])
    
    def _generate_frequency_based_predictions(self, data: np.ndarray, steps: int, patterns: Dict) -> np.ndarray:
        """Generate predictions based on frequency analysis"""
        try:
            frequency_patterns = patterns.get('frequency', {})
            dominant_freq = frequency_patterns.get('dominant_frequency', 0)
            concentration = frequency_patterns.get('frequency_concentration', 0)
            
            if abs(dominant_freq) > 1e-6 and concentration > 0.3:
                # Generate frequency-based predictions
                predictions = []
                last_value = data[-1]
                
                for i in range(steps):
                    # Apply frequency component
                    freq_component = np.sin(2 * np.pi * dominant_freq * i) * np.std(data) * 0.3
                    predictions.append(last_value + freq_component)
                
                return np.array(predictions)
            else:
                # No strong frequency pattern
                return np.full(steps, data[-1])
                
        except Exception as e:
            logger.error(f"Error in frequency-based predictions: {e}")
            return np.full(steps, data[-1])
    
    def _combine_predictions_adaptively(self, predictions: Dict[str, np.ndarray], patterns: Dict) -> np.ndarray:
        """Combine predictions adaptively based on pattern strength"""
        try:
            # Calculate weights based on pattern strength
            weights = {}
            total_weight = 0
            
            # Trend weight
            trend_strength = max(abs(patterns['trends'].get('recent_trend', 0)), 
                               abs(patterns['trends'].get('medium_trend', 0)))
            weights['trend'] = min(0.4, trend_strength / (patterns['statistical']['std'] + 1e-10))
            
            # Cyclical weight
            cyclical_strength = patterns['cyclical'].get('strength', 0)
            weights['cyclical'] = min(0.3, cyclical_strength)
            
            # Volatility weight
            weights['volatility'] = 0.2
            
            # Local weight
            local_change = patterns['local'].get('change_magnitude', 0)
            weights['local'] = min(0.2, local_change / (patterns['statistical']['std'] + 1e-10))
            
            # Frequency weight
            freq_concentration = patterns['frequency'].get('frequency_concentration', 0)
            weights['frequency'] = min(0.2, freq_concentration)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for key in weights:
                    weights[key] /= total_weight
            else:
                # Equal weights if no pattern detected
                weights = {key: 1.0/len(predictions) for key in predictions}
            
            # Combine predictions
            combined = np.zeros_like(predictions['trend'])
            for method, pred in predictions.items():
                if method in weights:
                    combined += weights[method] * pred
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in adaptive prediction combination: {e}")
            return predictions.get('trend', np.zeros(30))
    
    def _generate_simple_predictions(self, data: np.ndarray, steps: int) -> np.ndarray:
        """Generate simple predictions as fallback"""
        try:
            if len(data) >= 2:
                # Simple linear extrapolation
                trend = data[-1] - data[-2]
                predictions = [data[-1] + trend * (i + 1) for i in range(steps)]
                return np.array(predictions)
            else:
                return np.full(steps, data[-1] if len(data) > 0 else 0.0)
                
        except Exception as e:
            logger.error(f"Error in simple predictions: {e}")
            return np.zeros(steps)
    
    def _apply_continuity_preservation(self, predictions: np.ndarray, data: np.ndarray, 
                                     previous_predictions: List) -> np.ndarray:
        """Apply continuity preservation with previous predictions"""
        try:
            if not previous_predictions:
                return predictions
            
            # Calculate expected continuation based on previous predictions
            prev_array = np.array(previous_predictions)
            if len(prev_array) >= 2:
                prev_trend = prev_array[-1] - prev_array[-2]
                expected_first = prev_array[-1] + prev_trend
                
                # Apply continuity correction
                actual_first = predictions[0]
                continuity_gap = expected_first - actual_first
                
                # Apply gap correction with decay
                if abs(continuity_gap) > 0.1 * np.std(data):
                    correction_weights = np.exp(-0.2 * np.arange(len(predictions)))
                    predictions += continuity_gap * correction_weights * self.continuity_strength
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in continuity preservation: {e}")
            return predictions
    
    def _apply_boundary_preservation(self, predictions: np.ndarray, data: np.ndarray, 
                                   patterns: Dict) -> np.ndarray:
        """Apply boundary preservation to keep predictions within reasonable range"""
        try:
            # Calculate reasonable bounds
            historical_min = patterns['statistical']['min']
            historical_max = patterns['statistical']['max']
            historical_range = patterns['statistical']['range']
            
            # Expand bounds slightly to allow for natural variation
            range_expansion = 0.2 * historical_range
            lower_bound = historical_min - range_expansion
            upper_bound = historical_max + range_expansion
            
            # Apply bounds
            predictions = np.clip(predictions, lower_bound, upper_bound)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in boundary preservation: {e}")
            return predictions
    
    def _preserve_statistical_characteristics(self, predictions: np.ndarray, data: np.ndarray, 
                                            patterns: Dict) -> np.ndarray:
        """Preserve statistical characteristics in predictions"""
        try:
            # Target statistics
            target_mean = patterns['statistical']['mean']
            target_std = patterns['statistical']['std']
            
            # Current statistics
            current_mean = np.mean(predictions)
            current_std = np.std(predictions)
            
            # Apply statistical preservation
            if abs(current_mean - target_mean) > 0.1 * target_std:
                # Adjust mean
                mean_adjustment = (target_mean - current_mean) * 0.5
                predictions += mean_adjustment
            
            # Adjust standard deviation
            if current_std > 0 and abs(current_std - target_std) > 0.2 * target_std:
                std_adjustment = target_std / current_std
                prediction_mean = np.mean(predictions)
                predictions = prediction_mean + (predictions - prediction_mean) * std_adjustment * 0.7
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in statistical characteristic preservation: {e}")
            return predictions
    
    def _preserve_trend_characteristics(self, predictions: np.ndarray, data: np.ndarray, 
                                      patterns: Dict) -> np.ndarray:
        """Preserve trend characteristics in predictions"""
        try:
            # Already handled in trend momentum preservation
            return predictions
            
        except Exception as e:
            logger.error(f"Error in trend characteristic preservation: {e}")
            return predictions
    
    def _preserve_cyclical_characteristics(self, predictions: np.ndarray, data: np.ndarray, 
                                         patterns: Dict) -> np.ndarray:
        """Preserve cyclical characteristics in predictions"""
        try:
            # Already handled in cyclical preservation
            return predictions
            
        except Exception as e:
            logger.error(f"Error in cyclical characteristic preservation: {e}")
            return predictions
    
    def _preserve_volatility_characteristics(self, predictions: np.ndarray, data: np.ndarray, 
                                           patterns: Dict) -> np.ndarray:
        """Preserve volatility characteristics in predictions"""
        try:
            target_volatility = patterns['volatility'].get('mean_volatility', np.std(data))
            
            # Calculate current volatility
            if len(predictions) >= 2:
                current_volatility = np.std(np.diff(predictions))
                
                if abs(current_volatility - target_volatility) > 0.3 * target_volatility:
                    # Apply volatility adjustment
                    volatility_factor = target_volatility / (current_volatility + 1e-10)
                    prediction_mean = np.mean(predictions)
                    predictions = prediction_mean + (predictions - prediction_mean) * volatility_factor * 0.5
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in volatility characteristic preservation: {e}")
            return predictions
    
    def _apply_accumulated_bias_prevention(self, predictions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply accumulated bias prevention based on historical bias tracking"""
        try:
            if len(self.bias_tracking) < 3:
                return predictions
            
            # Calculate accumulated bias
            recent_biases = self.bias_tracking[-3:]
            mean_bias_trend = np.mean([b['mean_bias'] for b in recent_biases])
            trend_bias_trend = np.mean([b['trend_bias'] for b in recent_biases])
            
            # Apply accumulated bias correction
            if abs(mean_bias_trend) > 0.05 * np.std(data):
                bias_correction = -mean_bias_trend * 0.3
                predictions += bias_correction
            
            if abs(trend_bias_trend) > 0.05 * np.std(data):
                trend_correction = -trend_bias_trend * 0.3
                correction_weights = np.linspace(0, 1, len(predictions))
                predictions += trend_correction * correction_weights
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in accumulated bias prevention: {e}")
            return predictions
    
    def _calculate_variability_preservation_score(self, predictions: np.ndarray, data: np.ndarray, 
                                                patterns: Dict) -> float:
        """Calculate variability preservation score"""
        try:
            target_std = patterns['statistical']['std']
            prediction_std = np.std(predictions)
            
            # Calculate similarity score
            if target_std > 0:
                similarity = 1.0 - min(1.0, abs(prediction_std - target_std) / target_std)
            else:
                similarity = 1.0 if prediction_std == 0 else 0.0
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error in variability preservation score: {e}")
            return 0.5
    
    def _calculate_trend_preservation_score(self, predictions: np.ndarray, data: np.ndarray, 
                                          patterns: Dict) -> float:
        """Calculate trend preservation score"""
        try:
            target_trend = patterns['trends'].get('recent_trend', 0)
            
            if len(predictions) >= 2:
                x = np.arange(len(predictions))
                prediction_trend = np.polyfit(x, predictions, 1)[0]
                
                # Calculate similarity
                if abs(target_trend) > 1e-6:
                    similarity = 1.0 - min(1.0, abs(prediction_trend - target_trend) / abs(target_trend))
                else:
                    similarity = 1.0 if abs(prediction_trend) < 1e-6 else 0.0
            else:
                similarity = 0.5
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error in trend preservation score: {e}")
            return 0.5
    
    def _calculate_bias_prevention_score(self, predictions: np.ndarray, data: np.ndarray, 
                                       patterns: Dict) -> float:
        """Calculate bias prevention score"""
        try:
            # Calculate different types of bias
            mean_bias = abs(np.mean(predictions) - patterns['statistical']['mean'])
            std_bias = abs(np.std(predictions) - patterns['statistical']['std'])
            
            # Normalize biases
            target_std = patterns['statistical']['std']
            if target_std > 0:
                mean_bias_norm = mean_bias / target_std
                std_bias_norm = std_bias / target_std
            else:
                mean_bias_norm = 0.0
                std_bias_norm = 0.0
            
            # Calculate bias prevention score
            bias_score = 1.0 - min(1.0, (mean_bias_norm + std_bias_norm) / 2)
            
            return max(0.0, min(1.0, bias_score))
            
        except Exception as e:
            logger.error(f"Error in bias prevention score: {e}")
            return 0.5

    def _generate_fallback_predictions(self, data: np.ndarray, steps: int) -> Dict[str, Any]:
        """Generate fallback predictions when main algorithm fails"""
        try:
            # Simple trend continuation
            if len(data) >= 2:
                trend = (data[-1] - data[-2])
                predictions = [data[-1] + trend * (i + 1) for i in range(steps)]
            else:
                predictions = [data[-1]] * steps if len(data) > 0 else [0.0] * steps
            
            return {
                'predictions': predictions,
                'quality_metrics': {'overall_quality_score': 0.3},
                'pattern_analysis': {},
                'pattern_following_score': 0.3,
                'variability_preservation': 0.3,
                'bias_prevention_score': 0.3
            }
            
        except Exception as e:
            logger.error(f"Error in fallback predictions: {e}")
            return {
                'predictions': [0.0] * steps,
                'quality_metrics': {'overall_quality_score': 0.1},
                'pattern_analysis': {},
                'pattern_following_score': 0.1,
                'variability_preservation': 0.1,
                'bias_prevention_score': 0.1
            }

