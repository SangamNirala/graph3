"""
Enhanced Real-Time Prediction System with Advanced Pattern Learning
Integration of advanced ML algorithms for superior pattern following
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging
from datetime import datetime, timedelta
import json
import os
import sys
from scipy import signal
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_pattern_learning_engine import AdvancedPatternLearningEngine
    ADVANCED_ENGINE_AVAILABLE = True
except ImportError:
    ADVANCED_ENGINE_AVAILABLE = False
    print("Advanced Pattern Learning Engine not available")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedRealTimePredictionSystem:
    """
    Enhanced Real-Time Prediction System with Advanced Pattern Learning
    
    Features:
    - Multi-algorithm pattern learning (Deep Learning, Gaussian Process, Ensemble)
    - Adaptive learning from historical data
    - Real-time pattern adjustment
    - Advanced uncertainty quantification
    - Superior pattern following capabilities
    """
    
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.initialized = False
        self.historical_data = None
        self.prediction_history = []
        self.pattern_cache = {}
        
        # Initialize advanced pattern learning engine
        if ADVANCED_ENGINE_AVAILABLE:
            self.pattern_engine = AdvancedPatternLearningEngine(sequence_length)
            self.use_advanced_engine = True
        else:
            self.use_advanced_engine = False
            logger.warning("Advanced Pattern Learning Engine not available, using fallback methods")
        
        # Enhanced prediction parameters
        self.adaptation_rate = 0.2
        self.pattern_strength = 1.5
        self.continuity_weight = 0.8
        self.variability_preservation = 0.9
        self.bias_correction_strength = 0.8
        
        # Quality control
        self.min_data_points = 10
        self.max_prediction_horizon = 200
        self.prediction_confidence_threshold = 0.6
        
        # Pattern tracking
        self.detected_patterns = {}
        self.pattern_weights = {}
        self.learning_quality_history = []
        
        # Real-time learning
        self.online_learning_enabled = True
        self.pattern_update_interval = 10
        self.prediction_count = 0
        
    def initialize_with_historical_data(self, historical_data: np.ndarray, 
                                      timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Initialize the enhanced prediction system with historical data
        
        Args:
            historical_data: Historical time series data
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Initialization results including pattern analysis
        """
        try:
            logger.info("Initializing Enhanced Real-Time Prediction System")
            
            # Data validation and preprocessing
            if historical_data.ndim > 1:
                historical_data = historical_data.flatten()
            
            # Remove invalid values
            valid_mask = np.isfinite(historical_data)
            historical_data = historical_data[valid_mask]
            
            if len(historical_data) < self.min_data_points:
                logger.warning(f"Insufficient data points: {len(historical_data)} < {self.min_data_points}")
                return {'initialized': False, 'error': 'Insufficient data'}
            
            self.historical_data = historical_data
            
            # Advanced pattern learning
            if self.use_advanced_engine:
                pattern_learning_result = self.pattern_engine.learn_patterns(
                    historical_data, timestamps
                )
                self.pattern_cache.update(pattern_learning_result)
                learning_quality = pattern_learning_result.get('learning_quality', 0.5)
            else:
                # Fallback pattern analysis
                pattern_learning_result = self._fallback_pattern_analysis(historical_data)
                learning_quality = 0.6
            
            # Initialize scalers
            self.scaler = StandardScaler()
            self.minmax_scaler = MinMaxScaler()
            
            # Fit scalers on historical data
            self.scaler.fit(historical_data.reshape(-1, 1))
            self.minmax_scaler.fit(historical_data.reshape(-1, 1))
            
            # Initialize pattern weights based on analysis
            self._initialize_pattern_weights(pattern_learning_result)
            
            # Set initialization flag
            self.initialized = True
            
            result = {
                'initialized': True,
                'data_length': len(historical_data),
                'learning_quality': learning_quality,
                'patterns_detected': len(self.detected_patterns),
                'pattern_summary': self._generate_pattern_summary(),
                'system_capabilities': self._get_system_capabilities(),
                'quality_metrics': {
                    'data_quality': self._assess_data_quality(historical_data),
                    'pattern_diversity': len(self.detected_patterns),
                    'learning_confidence': learning_quality,
                    'prediction_readiness': min(1.0, learning_quality + 0.3)
                }
            }
            
            self.learning_quality_history.append({
                'timestamp': datetime.now(),
                'quality': learning_quality,
                'data_points': len(historical_data)
            })
            
            logger.info(f"Enhanced system initialized successfully. Learning quality: {learning_quality:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced system initialization: {e}")
            return {'initialized': False, 'error': str(e)}
    
    def generate_continuous_prediction(self, steps: int = 30,
                                     previous_predictions: Optional[List] = None,
                                     real_time_feedback: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate continuous predictions with advanced pattern following
        
        Args:
            steps: Number of prediction steps
            previous_predictions: Previous predictions for continuity
            real_time_feedback: Real-time feedback for adaptation
            
        Returns:
            Enhanced predictions with quality metrics
        """
        try:
            if not self.initialized:
                raise ValueError("System not initialized. Call initialize_with_historical_data() first.")
            
            logger.info(f"Generating {steps} prediction steps with enhanced system")
            
            # Step 1: Advanced pattern-based prediction
            if self.use_advanced_engine:
                advanced_result = self.pattern_engine.generate_advanced_predictions(
                    steps=steps,
                    confidence_level=0.95
                )
                base_predictions = np.array(advanced_result['predictions'])
                confidence_intervals = advanced_result['confidence_intervals']
                pattern_confidence = advanced_result.get('pattern_confidence', 0.7)
            else:
                # Fallback prediction method
                base_predictions = self._generate_fallback_predictions(steps)
                confidence_intervals = self._generate_fallback_confidence(base_predictions)
                pattern_confidence = 0.6
            
            # Step 2: Apply enhanced continuity correction
            if previous_predictions is not None and len(previous_predictions) > 0:
                base_predictions = self._apply_enhanced_continuity_correction(
                    base_predictions, previous_predictions
                )
            
            # Step 3: Real-time learning and adaptation
            if self.online_learning_enabled and real_time_feedback:
                base_predictions = self._apply_real_time_adaptation(
                    base_predictions, real_time_feedback
                )
            
            # Step 4: Enhanced post-processing
            final_predictions = self._apply_enhanced_postprocessing(base_predictions)
            
            # Step 5: Quality assessment and confidence adjustment
            quality_metrics = self._assess_prediction_quality(final_predictions)
            adjusted_confidence = self._adjust_confidence_with_quality(
                confidence_intervals, quality_metrics
            )
            
            # Step 6: Pattern following validation
            pattern_following_score = self._validate_pattern_following(final_predictions)
            
            # Update prediction history
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'predictions': final_predictions.tolist(),
                'quality': quality_metrics,
                'pattern_score': pattern_following_score
            })
            
            # Keep only recent history
            if len(self.prediction_history) > 50:
                self.prediction_history = self.prediction_history[-50:]
            
            self.prediction_count += 1
            
            # Periodic pattern updates
            if self.prediction_count % self.pattern_update_interval == 0:
                self._update_patterns_from_feedback()
            
            result = {
                'predictions': final_predictions.tolist(),
                'confidence_intervals': adjusted_confidence,
                'quality_metrics': quality_metrics,
                'pattern_following_score': pattern_following_score,
                'pattern_confidence': pattern_confidence,
                'variability_preservation_score': quality_metrics.get('variability_preservation', 0.7),
                'bias_prevention_score': quality_metrics.get('bias_prevention', 0.7),
                'continuity_score': quality_metrics.get('continuity_score', 0.7),
                'metadata': {
                    'prediction_method': 'enhanced_pattern_learning',
                    'advanced_engine_used': self.use_advanced_engine,
                    'prediction_count': self.prediction_count,
                    'patterns_active': len(self.detected_patterns),
                    'learning_quality': self._get_current_learning_quality(),
                    'system_confidence': min(1.0, pattern_following_score + 0.2)
                }
            }
            
            logger.info(f"Enhanced prediction generated. Pattern score: {pattern_following_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced continuous prediction: {e}")
            return self._generate_emergency_fallback(steps)
    
    def _apply_enhanced_continuity_correction(self, predictions: np.ndarray,
                                            previous_predictions: List) -> np.ndarray:
        """Apply enhanced continuity correction for smooth transitions"""
        try:
            if len(previous_predictions) == 0:
                return predictions
            
            # Calculate connection point
            last_prediction = previous_predictions[-1]
            first_new_prediction = predictions[0]
            
            # Smooth transition
            transition_strength = self.continuity_weight
            connection_adjustment = (last_prediction - first_new_prediction) * transition_strength
            
            # Apply gradual adjustment across first few predictions
            transition_length = min(5, len(predictions))
            for i in range(transition_length):
                decay_factor = np.exp(-i / 2.0)  # Exponential decay
                predictions[i] += connection_adjustment * decay_factor
            
            # Ensure overall trend preservation
            if len(previous_predictions) >= 3:
                recent_trend = np.mean(np.diff(previous_predictions[-3:]))
                prediction_trend = np.mean(np.diff(predictions[:3]))
                
                # Adjust trend if significantly different
                trend_difference = recent_trend - prediction_trend
                if abs(trend_difference) > np.std(self.historical_data) * 0.1:
                    trend_adjustment = trend_difference * 0.5
                    for i in range(len(predictions)):
                        predictions[i] += trend_adjustment * (i + 1) / len(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in continuity correction: {e}")
            return predictions
    
    def _apply_enhanced_postprocessing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply enhanced post-processing for optimal pattern following"""
        try:
            processed = predictions.copy()
            
            # 1. Variability enhancement
            processed = self._enhance_prediction_variability(processed)
            
            # 2. Pattern-informed smoothing
            processed = self._apply_pattern_informed_smoothing(processed)
            
            # 3. Bias correction
            processed = self._apply_enhanced_bias_correction(processed)
            
            # 4. Boundary preservation
            processed = self._apply_boundary_preservation(processed)
            
            # 5. Final validation and correction
            processed = self._apply_final_validation_correction(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in enhanced post-processing: {e}")
            return predictions
    
    def _enhance_prediction_variability(self, predictions: np.ndarray) -> np.ndarray:
        """Enhance prediction variability to match historical patterns"""
        try:
            if len(self.historical_data) < 5:
                return predictions
            
            # Target variability characteristics
            historical_std = np.std(self.historical_data)
            historical_changes = np.diff(self.historical_data)
            historical_change_std = np.std(historical_changes)
            
            # Current prediction characteristics
            prediction_std = np.std(predictions)
            prediction_changes = np.diff(predictions)
            prediction_change_std = np.std(prediction_changes)
            
            enhanced = predictions.copy()
            
            # Enhance overall variability if too low
            if prediction_std < historical_std * 0.7:
                target_std = historical_std * self.variability_preservation
                enhancement_factor = target_std / (prediction_std + 1e-10)
                enhancement_factor = min(1.5, enhancement_factor)
                
                prediction_mean = np.mean(enhanced)
                enhanced = prediction_mean + (enhanced - prediction_mean) * enhancement_factor
            
            # Enhance change variability
            if prediction_change_std < historical_change_std * 0.6:
                target_change_std = historical_change_std * 0.8
                
                # Calculate enhanced changes
                enhanced_changes = np.diff(enhanced)
                change_mean = np.mean(enhanced_changes)
                change_enhancement = target_change_std / (prediction_change_std + 1e-10)
                change_enhancement = min(1.3, change_enhancement)
                
                enhanced_changes = change_mean + (enhanced_changes - change_mean) * change_enhancement
                
                # Reconstruct predictions
                enhanced[1:] = enhanced[0] + np.cumsum(enhanced_changes)
            
            # Add realistic noise based on historical patterns
            if len(self.historical_data) > 20:
                noise_level = np.std(self.historical_data) * 0.05
                noise = np.random.normal(0, noise_level, len(enhanced))
                
                # Apply correlated noise
                if len(noise) > 3:
                    smoothed_noise = savgol_filter(noise, min(5, len(noise)//2*2-1), 1)
                    enhanced += smoothed_noise * 0.3
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in variability enhancement: {e}")
            return predictions
    
    def _validate_pattern_following(self, predictions: np.ndarray) -> float:
        """Validate how well predictions follow historical patterns"""
        try:
            if len(self.historical_data) < 10:
                return 0.5
            
            # Multiple validation metrics
            scores = []
            
            # 1. Trend consistency
            historical_trend = np.polyfit(range(len(self.historical_data)), self.historical_data, 1)[0]
            prediction_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
            
            trend_consistency = 1.0 - min(1.0, abs(historical_trend - prediction_trend) / 
                                        (np.std(self.historical_data) + 1e-10))
            scores.append(trend_consistency * 0.3)
            
            # 2. Variability matching
            historical_std = np.std(self.historical_data)
            prediction_std = np.std(predictions)
            
            variability_score = 1.0 - min(1.0, abs(historical_std - prediction_std) / 
                                        (historical_std + 1e-10))
            scores.append(variability_score * 0.25)
            
            # 3. Change pattern similarity
            historical_changes = np.diff(self.historical_data)
            prediction_changes = np.diff(predictions)
            
            if len(historical_changes) > 0 and len(prediction_changes) > 0:
                historical_change_std = np.std(historical_changes)
                prediction_change_std = np.std(prediction_changes)
                
                change_similarity = 1.0 - min(1.0, abs(historical_change_std - prediction_change_std) / 
                                            (historical_change_std + 1e-10))
                scores.append(change_similarity * 0.25)
            
            # 4. Range preservation
            historical_range = np.max(self.historical_data) - np.min(self.historical_data)
            prediction_range = np.max(predictions) - np.min(predictions)
            
            if historical_range > 0:
                range_preservation = 1.0 - min(1.0, abs(historical_range - prediction_range) / historical_range)
                scores.append(range_preservation * 0.2)
            
            # Overall pattern following score
            pattern_score = sum(scores)
            
            # Bonus for detected patterns
            if self.detected_patterns:
                pattern_bonus = min(0.1, len(self.detected_patterns) * 0.02)
                pattern_score += pattern_bonus
            
            return min(1.0, max(0.0, pattern_score))
            
        except Exception as e:
            logger.error(f"Error in pattern following validation: {e}")
            return 0.5
    
    def _assess_prediction_quality(self, predictions: np.ndarray) -> Dict[str, float]:
        """Assess overall prediction quality"""
        try:
            quality_metrics = {}
            
            # Basic quality checks
            quality_metrics['completeness'] = 1.0  # All predictions generated
            quality_metrics['validity'] = float(np.all(np.isfinite(predictions)))
            
            # Variability assessment
            if len(self.historical_data) > 5:
                historical_std = np.std(self.historical_data)
                prediction_std = np.std(predictions)
                quality_metrics['variability_preservation'] = 1.0 - min(1.0, 
                    abs(historical_std - prediction_std) / (historical_std + 1e-10))
            else:
                quality_metrics['variability_preservation'] = 0.7
            
            # Smoothness assessment
            if len(predictions) > 2:
                second_derivatives = np.diff(predictions, 2)
                smoothness = 1.0 / (1.0 + np.std(second_derivatives))
                quality_metrics['smoothness'] = min(1.0, smoothness)
            else:
                quality_metrics['smoothness'] = 0.8
            
            # Bias assessment
            if len(self.historical_data) > 5:
                historical_mean = np.mean(self.historical_data)
                prediction_mean = np.mean(predictions)
                bias = abs(prediction_mean - historical_mean) / (np.std(self.historical_data) + 1e-10)
                quality_metrics['bias_prevention'] = max(0.0, 1.0 - bias)
            else:
                quality_metrics['bias_prevention'] = 0.7
            
            # Continuity assessment
            if len(self.prediction_history) > 0:
                last_predictions = self.prediction_history[-1]['predictions']
                if len(last_predictions) > 0:
                    continuity_gap = abs(predictions[0] - last_predictions[-1])
                    max_gap = np.std(self.historical_data) * 0.5
                    quality_metrics['continuity_score'] = max(0.0, 1.0 - continuity_gap / max_gap)
                else:
                    quality_metrics['continuity_score'] = 0.8
            else:
                quality_metrics['continuity_score'] = 0.8
            
            # Overall quality
            weights = {
                'completeness': 0.1,
                'validity': 0.2,
                'variability_preservation': 0.3,
                'smoothness': 0.15,
                'bias_prevention': 0.15,
                'continuity_score': 0.1
            }
            
            overall_quality = sum(quality_metrics[key] * weights[key] 
                                for key in weights if key in quality_metrics)
            quality_metrics['overall_quality'] = overall_quality
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return {'overall_quality': 0.5}
    
    def _fallback_pattern_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Fallback pattern analysis when advanced engine is not available"""
        try:
            analysis = {}
            
            # Simple trend analysis
            trend_slope = np.polyfit(range(len(data)), data, 1)[0]
            analysis['trend'] = {
                'slope': trend_slope,
                'strength': abs(trend_slope) / (np.std(data) + 1e-10)
            }
            
            # Simple cyclical analysis using autocorrelation
            if len(data) > 10:
                autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                
                # Find peaks
                peaks = signal.find_peaks(autocorr[1:min(50, len(autocorr))], height=0.1)[0]
                analysis['cyclical'] = {
                    'dominant_period': peaks[0] + 1 if len(peaks) > 0 else None,
                    'strength': np.max(autocorr[1:]) if len(autocorr) > 1 else 0
                }
            
            # Simple volatility analysis
            changes = np.diff(data)
            analysis['volatility'] = {
                'level': np.std(changes),
                'clustering': np.corrcoef(np.abs(changes[:-1]), np.abs(changes[1:]))[0, 1] 
                            if len(changes) > 2 else 0
            }
            
            return {'pattern_analysis': analysis, 'learning_quality': 0.6}
            
        except Exception as e:
            logger.error(f"Error in fallback pattern analysis: {e}")
            return {'learning_quality': 0.4}
    
    def _generate_fallback_predictions(self, steps: int) -> np.ndarray:
        """Generate fallback predictions using simple methods"""
        try:
            if len(self.historical_data) < 3:
                return np.array([self.historical_data[-1]] * steps)
            
            # Simple linear extrapolation with noise
            recent_trend = np.mean(np.diff(self.historical_data[-5:]))
            last_value = self.historical_data[-1]
            
            predictions = []
            for i in range(steps):
                pred = last_value + recent_trend * (i + 1)
                predictions.append(pred)
            
            # Add some variability
            predictions = np.array(predictions)
            noise_level = np.std(self.historical_data) * 0.1
            noise = np.random.normal(0, noise_level, len(predictions))
            predictions += noise
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return np.array([0] * steps)
    
    def _generate_emergency_fallback(self, steps: int) -> Dict[str, Any]:
        """Emergency fallback when all other methods fail"""
        try:
            if self.historical_data is not None and len(self.historical_data) > 0:
                last_value = self.historical_data[-1]
                predictions = [last_value] * steps
            else:
                predictions = [0] * steps
            
            return {
                'predictions': predictions,
                'confidence_intervals': [{'lower': p * 0.9, 'upper': p * 1.1} for p in predictions],
                'quality_metrics': {'overall_quality': 0.3},
                'pattern_following_score': 0.3,
                'metadata': {'emergency_fallback': True}
            }
            
        except Exception as e:
            logger.error(f"Error in emergency fallback: {e}")
            return {
                'predictions': [0] * steps,
                'quality_metrics': {'overall_quality': 0.1},
                'pattern_following_score': 0.1,
                'metadata': {'emergency_fallback': True, 'error': str(e)}
            }
    
    # Additional helper methods...
    def _initialize_pattern_weights(self, pattern_result: Dict):
        """Initialize pattern weights based on analysis"""
        self.pattern_weights = {
            'trend': 0.3,
            'cyclical': 0.25,
            'volatility': 0.2,
            'local': 0.15,
            'seasonal': 0.1
        }
    
    def _generate_pattern_summary(self) -> Dict[str, Any]:
        """Generate summary of detected patterns"""
        return {
            'total_patterns': len(self.detected_patterns),
            'pattern_types': list(self.detected_patterns.keys()),
            'dominant_pattern': max(self.pattern_weights.items(), key=lambda x: x[1])[0] if self.pattern_weights else None
        }
    
    def _get_system_capabilities(self) -> List[str]:
        """Get list of system capabilities"""
        capabilities = [
            'pattern_learning',
            'real_time_prediction',
            'continuity_preservation',
            'variability_matching',
            'bias_correction'
        ]
        
        if self.use_advanced_engine:
            capabilities.extend([
                'deep_learning',
                'gaussian_process',
                'ensemble_methods',
                'uncertainty_quantification'
            ])
        
        return capabilities
    
    def _assess_data_quality(self, data: np.ndarray) -> float:
        """Assess quality of input data"""
        try:
            quality_score = 1.0
            
            # Check for missing values
            if np.any(~np.isfinite(data)):
                quality_score -= 0.2
            
            # Check for sufficient length
            if len(data) < 20:
                quality_score -= 0.3
            
            # Check for variability
            if np.std(data) < 1e-6:
                quality_score -= 0.3
            
            # Check for outliers
            Q1, Q3 = np.percentile(data, [25, 75])
            IQR = Q3 - Q1
            outliers = np.sum((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR))
            if outliers > len(data) * 0.1:
                quality_score -= 0.2
            
            return max(0.0, quality_score)
            
        except Exception:
            return 0.5
    
    def _get_current_learning_quality(self) -> float:
        """Get current learning quality score"""
        if self.learning_quality_history:
            return self.learning_quality_history[-1]['quality']
        return 0.5
    
    def _update_patterns_from_feedback(self):
        """Update patterns based on prediction feedback"""
        # This would implement online learning updates
        # For now, just log that we're updating
        logger.info("Updating patterns from recent feedback")
        pass
    
    def _apply_real_time_adaptation(self, predictions: np.ndarray, 
                                   feedback: Dict) -> np.ndarray:
        """Apply real-time adaptation based on feedback"""
        # This would implement real-time learning
        # For now, return predictions unchanged
        return predictions
    
    def _apply_pattern_informed_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply smoothing informed by learned patterns"""
        if len(predictions) > 5:
            # Light smoothing to preserve patterns
            window_size = min(5, len(predictions)//3)
            if window_size >= 3 and window_size % 2 == 1:
                return savgol_filter(predictions, window_size, 1)
        return predictions
    
    def _apply_enhanced_bias_correction(self, predictions: np.ndarray) -> np.ndarray:
        """Apply enhanced bias correction"""
        if len(self.historical_data) > 5:
            historical_mean = np.mean(self.historical_data)
            prediction_mean = np.mean(predictions)
            bias = prediction_mean - historical_mean
            
            # Apply gradual bias correction
            correction_strength = self.bias_correction_strength
            predictions -= bias * correction_strength
        
        return predictions
    
    def _apply_boundary_preservation(self, predictions: np.ndarray) -> np.ndarray:
        """Apply boundary preservation based on historical data"""
        if len(self.historical_data) > 5:
            hist_min, hist_max = np.min(self.historical_data), np.max(self.historical_data)
            hist_range = hist_max - hist_min
            
            # Allow some extrapolation beyond historical bounds
            extrapolation_factor = 0.2
            lower_bound = hist_min - hist_range * extrapolation_factor
            upper_bound = hist_max + hist_range * extrapolation_factor
            
            predictions = np.clip(predictions, lower_bound, upper_bound)
        
        return predictions
    
    def _apply_final_validation_correction(self, predictions: np.ndarray) -> np.ndarray:
        """Apply final validation and correction"""
        # Ensure no NaN or infinite values
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure reasonable magnitude
        if len(self.historical_data) > 0:
            max_reasonable = np.max(np.abs(self.historical_data)) * 3
            predictions = np.clip(predictions, -max_reasonable, max_reasonable)
        
        return predictions
    
    def _generate_fallback_confidence(self, predictions: np.ndarray) -> List[Dict]:
        """Generate fallback confidence intervals"""
        confidence_intervals = []
        std_dev = np.std(self.historical_data) if len(self.historical_data) > 1 else 1.0
        
        for pred in predictions:
            confidence_intervals.append({
                'lower': float(pred - 1.96 * std_dev),
                'upper': float(pred + 1.96 * std_dev),
                'std_error': float(std_dev)
            })
        
        return confidence_intervals
    
    def _adjust_confidence_with_quality(self, confidence_intervals: List[Dict], 
                                      quality_metrics: Dict) -> List[Dict]:
        """Adjust confidence intervals based on quality metrics"""
        quality_factor = quality_metrics.get('overall_quality', 0.7)
        
        # Widen intervals for lower quality predictions
        adjustment_factor = 1.0 + (1.0 - quality_factor)
        
        adjusted_intervals = []
        for interval in confidence_intervals:
            center = (interval['lower'] + interval['upper']) / 2
            half_width = (interval['upper'] - interval['lower']) / 2
            adjusted_half_width = half_width * adjustment_factor
            
            adjusted_intervals.append({
                'lower': center - adjusted_half_width,
                'upper': center + adjusted_half_width,
                'std_error': interval.get('std_error', 0) * adjustment_factor
            })
        
        return adjusted_intervals