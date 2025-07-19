"""
Enhanced Real-Time Continuous Prediction System v2
Advanced real-time prediction system with superior pattern following
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging
from datetime import datetime, timedelta
import json
import os
import sys

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_pattern_memory_v2 import AdvancedPatternMemoryV2

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedRealtimeContinuousPredictionV2:
    """
    Enhanced Real-Time Continuous Prediction System v2
    
    Features:
    - Advanced pattern memory and learning
    - Superior historical pattern following
    - Dynamic variability preservation
    - Real-time bias correction
    - Multi-scale pattern adaptation
    """
    
    def __init__(self):
        # Initialize advanced pattern memory
        self.pattern_memory = AdvancedPatternMemoryV2(
            max_patterns=200,
            pattern_similarity_threshold=0.75
        )
        
        # Prediction state
        self.initialized = False
        self.historical_data = None
        self.prediction_history = []
        self.pattern_analysis_cache = {}
        
        # Enhanced prediction parameters
        self.pattern_strength_multiplier = 1.2  # Increase pattern influence
        self.variability_preservation_strength = 0.95  # Strong variability preservation
        self.bias_correction_strength = 0.7  # Strong bias correction
        self.continuity_preservation_strength = 0.8  # Strong continuity
        
        # Real-time learning parameters
        self.learning_rate = 0.15  # Increased learning rate
        self.adaptation_threshold = 0.05  # Lower threshold for faster adaptation
        self.pattern_update_frequency = 5  # Update patterns every 5 predictions
        self.max_prediction_history = 100  # Keep last 100 predictions
        
        # Quality control parameters
        self.min_quality_threshold = 0.7  # Higher quality threshold
        self.pattern_confidence_threshold = 0.6  # Higher confidence threshold
        self.variability_tolerance = 0.15  # Tighter variability control
        
        # Pattern following enhancement
        self.trend_following_strength = 0.9
        self.cyclical_following_strength = 0.85
        self.volatility_matching_strength = 0.8
        self.local_pattern_strength = 0.75
        
    def initialize_with_historical_data(self, historical_data: np.ndarray, 
                                      timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Initialize the prediction system with comprehensive historical data analysis
        
        Args:
            historical_data: Historical time series data
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Initialization results and pattern analysis
        """
        try:
            logger.info("Initializing Enhanced Real-Time Continuous Prediction System v2")
            
            # Store historical data
            if historical_data.ndim > 1:
                historical_data = historical_data.flatten()
            
            # Remove invalid values
            valid_mask = np.isfinite(historical_data)
            historical_data = historical_data[valid_mask]
            
            if timestamps is not None:
                timestamps = timestamps[valid_mask]
            
            self.historical_data = historical_data
            
            if len(historical_data) < 5:
                logger.warning("Insufficient historical data for proper initialization")
                return self._minimal_initialization(historical_data)
            
            # Comprehensive pattern learning
            logger.info(f"Learning patterns from {len(historical_data)} data points")
            pattern_analysis = self.pattern_memory.learn_patterns_from_data(
                historical_data, timestamps
            )
            
            # Cache pattern analysis
            self.pattern_analysis_cache = pattern_analysis
            
            # Initialize prediction parameters based on patterns
            self._initialize_prediction_parameters(pattern_analysis)
            
            # Calculate data quality metrics
            quality_metrics = self._calculate_data_quality(historical_data)
            
            # Set initialization flag
            self.initialized = True
            
            initialization_result = {
                'status': 'success',
                'data_points': len(historical_data),
                'pattern_analysis': pattern_analysis,
                'quality_metrics': quality_metrics,
                'initialization_parameters': {
                    'pattern_strength_multiplier': self.pattern_strength_multiplier,
                    'variability_preservation_strength': self.variability_preservation_strength,
                    'bias_correction_strength': self.bias_correction_strength
                }
            }
            
            logger.info("Initialization completed successfully")
            return initialization_result
            
        except Exception as e:
            logger.error(f"Error in initialization: {e}")
            return self._minimal_initialization(historical_data)
    
    def generate_continuous_prediction(self, steps: int = 30,
                                     previous_predictions: Optional[List] = None,
                                     real_time_feedback: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate enhanced continuous predictions with superior pattern following
        
        Args:
            steps: Number of prediction steps
            previous_predictions: Previous predictions for continuity
            real_time_feedback: Real-time feedback for adaptation
            
        Returns:
            Enhanced predictions with pattern analysis and quality metrics
        """
        try:
            if not self.initialized or self.historical_data is None:
                logger.warning("System not properly initialized")
                return self._generate_fallback_prediction(steps)
            
            logger.info(f"Generating enhanced continuous prediction for {steps} steps")
            
            # Update pattern analysis if needed
            if len(self.prediction_history) % self.pattern_update_frequency == 0:
                self._update_pattern_analysis()
            
            # Generate pattern-based predictions
            pattern_predictions = self._generate_pattern_based_predictions(
                steps, previous_predictions
            )
            
            # Apply enhanced continuous corrections
            corrected_predictions = self._apply_enhanced_continuous_corrections(
                pattern_predictions, previous_predictions
            )
            
            # Apply real-time learning adaptations
            adaptive_predictions = self._apply_real_time_adaptations(
                corrected_predictions, real_time_feedback
            )
            
            # Apply final quality enhancements
            final_predictions = self._apply_final_quality_enhancements(
                adaptive_predictions
            )
            
            # Calculate comprehensive quality metrics
            quality_metrics = self._calculate_prediction_quality_metrics(
                final_predictions
            )
            
            # Update prediction history
            self._update_prediction_history(final_predictions, quality_metrics)
            
            # Generate prediction result
            result = {
                'predictions': final_predictions.tolist(),
                'quality_metrics': quality_metrics,
                'pattern_analysis': self.pattern_analysis_cache.get('learned_patterns', {}),
                'prediction_confidence': quality_metrics.get('overall_confidence', 0.5),
                'pattern_following_score': quality_metrics.get('pattern_following_score', 0.5),
                'variability_preservation_score': quality_metrics.get('variability_preservation', 0.5),
                'bias_prevention_score': quality_metrics.get('bias_prevention', 0.5),
                'continuity_score': quality_metrics.get('continuity_score', 0.5),
                'metadata': {
                    'prediction_count': len(self.prediction_history),
                    'pattern_confidence': self.pattern_analysis_cache.get('pattern_confidence', {}),
                    'system_status': 'active',
                    'learning_active': True
                }
            }
            
            logger.info(f"Prediction generated successfully. Quality score: {quality_metrics.get('overall_quality', 0):.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in continuous prediction generation: {e}")
            return self._generate_fallback_prediction(steps)
    
    def _generate_pattern_based_predictions(self, steps: int, 
                                          previous_predictions: Optional[List] = None) -> np.ndarray:
        """Generate predictions based on learned patterns"""
        try:
            # Use pattern memory for prediction generation
            pattern_result = self.pattern_memory.generate_pattern_based_predictions(
                self.historical_data, 
                steps=steps,
                pattern_weights={
                    'cyclical': 0.35,     # Increased cyclical weight
                    'trending': 0.25,
                    'volatility': 0.2,    # Increased volatility weight
                    'local': 0.15,
                    'seasonal': 0.03,
                    'complex': 0.02
                }
            )
            
            base_predictions = np.array(pattern_result['predictions'])
            
            # Apply pattern strength multiplier
            if len(self.historical_data) > 0:
                historical_mean = np.mean(self.historical_data)
                enhanced_predictions = historical_mean + (base_predictions - historical_mean) * self.pattern_strength_multiplier
            else:
                enhanced_predictions = base_predictions
            
            # Ensure continuity with previous predictions
            if previous_predictions is not None and len(previous_predictions) > 0:
                enhanced_predictions = self._apply_prediction_continuity(
                    enhanced_predictions, previous_predictions
                )
            
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"Error in pattern-based prediction generation: {e}")
            return self._generate_simple_extrapolation(steps)
    
    def _apply_enhanced_continuous_corrections(self, predictions: np.ndarray,
                                             previous_predictions: Optional[List] = None) -> np.ndarray:
        """Apply enhanced corrections for continuous predictions"""
        try:
            corrected = predictions.copy()
            
            # 1. Enhanced variability preservation
            corrected = self._apply_enhanced_variability_preservation(corrected)
            
            # 2. Advanced bias correction
            corrected = self._apply_advanced_bias_correction(corrected)
            
            # 3. Pattern consistency enforcement
            corrected = self._enforce_pattern_consistency(corrected)
            
            # 4. Boundary preservation with adaptive limits
            corrected = self._apply_adaptive_boundary_preservation(corrected)
            
            # 5. Smoothness preservation
            corrected = self._apply_smoothness_preservation(corrected)
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error in enhanced continuous corrections: {e}")
            return predictions
    
    def _apply_enhanced_variability_preservation(self, predictions: np.ndarray) -> np.ndarray:
        """Apply enhanced variability preservation to match historical characteristics"""
        try:
            if len(self.historical_data) < 3:
                return predictions
            
            # Calculate target variability characteristics
            historical_std = np.std(self.historical_data)
            historical_changes = np.diff(self.historical_data)
            historical_change_std = np.std(historical_changes)
            historical_range = np.max(self.historical_data) - np.min(self.historical_data)
            
            # Current prediction characteristics
            prediction_std = np.std(predictions)
            prediction_changes = np.diff(predictions)
            prediction_change_std = np.std(prediction_changes)
            prediction_range = np.max(predictions) - np.min(predictions)
            
            enhanced_predictions = predictions.copy()
            
            # 1. Overall variability enhancement
            if prediction_std < historical_std * 0.6:
                # Predictions too flat - enhance variability
                target_std = historical_std * self.variability_preservation_strength
                enhancement_factor = target_std / (prediction_std + 1e-10)
                enhancement_factor = min(2.0, enhancement_factor)  # Limit enhancement
                
                prediction_mean = np.mean(enhanced_predictions)
                enhanced_predictions = prediction_mean + (enhanced_predictions - prediction_mean) * enhancement_factor
            
            # 2. Change variability enhancement
            enhanced_changes = np.diff(enhanced_predictions)
            enhanced_change_std = np.std(enhanced_changes)
            
            if enhanced_change_std < historical_change_std * 0.5:
                # Enhance change variability
                target_change_std = historical_change_std * 0.8
                change_enhancement = target_change_std / (enhanced_change_std + 1e-10)
                change_enhancement = min(1.5, change_enhancement)
                
                # Apply to changes
                change_mean = np.mean(enhanced_changes)
                enhanced_changes = change_mean + (enhanced_changes - change_mean) * change_enhancement
                
                # Reconstruct predictions from enhanced changes
                enhanced_predictions[1:] = enhanced_predictions[0] + np.cumsum(enhanced_changes)
            
            # 3. Inject realistic noise pattern
            if len(self.historical_data) > 10:
                # Learn noise characteristics from historical data
                detrended_historical = self._detrend_data(self.historical_data)
                noise_std = np.std(detrended_historical) * 0.3  # Moderate noise injection
                
                # Generate correlated noise
                noise = np.random.normal(0, noise_std, len(enhanced_predictions))
                # Apply smoothing to noise for realism
                if len(noise) > 5:
                    smoothed_noise = savgol_filter(noise, min(5, len(noise)//2*2-1), 1)
                    enhanced_predictions += smoothed_noise * 0.5
            
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"Error in enhanced variability preservation: {e}")
            return predictions
    
    def _apply_advanced_bias_correction(self, predictions: np.ndarray) -> np.ndarray:
        """Apply advanced bias correction to prevent accumulation"""
        try:
            if len(self.historical_data) < 5:
                return predictions
            
            corrected = predictions.copy()
            
            # 1. Detect potential bias
            historical_mean = np.mean(self.historical_data)
            prediction_mean = np.mean(corrected)
            bias = prediction_mean - historical_mean
            
            # 2. Trend bias correction
            if len(self.historical_data) > 10:
                # Historical trend
                hist_x = np.arange(len(self.historical_data))
                hist_trend = np.polyfit(hist_x, self.historical_data, 1)[0]
                
                # Prediction trend
                pred_x = np.arange(len(corrected))
                pred_trend = np.polyfit(pred_x, corrected, 1)[0]
                
                trend_bias = pred_trend - hist_trend
                
                # Apply trend bias correction
                if abs(trend_bias) > 0.01:  # Threshold for trend bias
                    correction_strength = min(1.0, abs(trend_bias) * 10)
                    trend_correction = -trend_bias * correction_strength * self.bias_correction_strength
                    
                    for i in range(len(corrected)):
                        corrected[i] += trend_correction * (i + 1) / len(corrected)
            
            # 3. Mean reversion correction
            if abs(bias) > np.std(self.historical_data) * 0.5:
                # Apply mean reversion
                reversion_strength = min(1.0, abs(bias) / np.std(self.historical_data))
                correction_factor = -bias * reversion_strength * self.bias_correction_strength
                corrected += correction_factor
            
            # 4. Progressive bias correction (stronger for later predictions)
            if len(self.prediction_history) > 3:
                # Check for accumulated bias in prediction history
                recent_predictions = [ph['predictions'] for ph in self.prediction_history[-3:]]
                if recent_predictions:
                    all_recent = np.concatenate(recent_predictions)
                    recent_bias = np.mean(all_recent) - historical_mean
                    
                    if abs(recent_bias) > np.std(self.historical_data) * 0.3:
                        # Apply progressive correction
                        progressive_correction = -recent_bias * 0.5
                        for i in range(len(corrected)):
                            weight = (i + 1) / len(corrected)  # Stronger correction for later predictions
                            corrected[i] += progressive_correction * weight
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error in advanced bias correction: {e}")
            return predictions
    
    def _enforce_pattern_consistency(self, predictions: np.ndarray) -> np.ndarray:
        """Enforce consistency with learned patterns"""
        try:
            if not self.pattern_analysis_cache:
                return predictions
            
            enhanced = predictions.copy()
            patterns = self.pattern_analysis_cache.get('learned_patterns', {})
            confidence = self.pattern_analysis_cache.get('pattern_confidence', {})
            
            # 1. Cyclical pattern enforcement
            cyclical_patterns = patterns.get('cyclical', {})
            cyclical_confidence = confidence.get('cyclical', 0)
            
            if cyclical_confidence > 0.5 and cyclical_patterns.get('components'):
                enhanced = self._apply_cyclical_pattern_consistency(
                    enhanced, cyclical_patterns, cyclical_confidence
                )
            
            # 2. Trending pattern enforcement
            trending_patterns = patterns.get('trending', {})
            trending_confidence = confidence.get('trending', 0)
            
            if trending_confidence > 0.4:
                enhanced = self._apply_trending_pattern_consistency(
                    enhanced, trending_patterns, trending_confidence
                )
            
            # 3. Volatility pattern enforcement
            volatility_patterns = patterns.get('volatility', {})
            volatility_confidence = confidence.get('volatility', 0)
            
            if volatility_confidence > 0.3:
                enhanced = self._apply_volatility_pattern_consistency(
                    enhanced, volatility_patterns, volatility_confidence
                )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in pattern consistency enforcement: {e}")
            return predictions
    
    def _apply_cyclical_pattern_consistency(self, predictions: np.ndarray, 
                                          cyclical_patterns: Dict, 
                                          confidence: float) -> np.ndarray:
        """Apply cyclical pattern consistency"""
        try:
            components = cyclical_patterns.get('components', [])
            if not components:
                return predictions
            
            enhanced = predictions.copy()
            
            # Use dominant cyclical component
            dominant_component = components[0]
            period = dominant_component['period']
            cycle_pattern = np.array(dominant_component['average_cycle'])
            
            if len(cycle_pattern) > 0 and period > 0:
                # Apply cyclical adjustment
                strength = confidence * self.cyclical_following_strength * 0.4  # Moderate strength
                
                for i in range(len(enhanced)):
                    cycle_position = i % period
                    if cycle_position < len(cycle_pattern):
                        cycle_value = cycle_pattern[cycle_position]
                        cycle_mean = np.mean(cycle_pattern)
                        cycle_contribution = (cycle_value - cycle_mean) * strength
                        enhanced[i] += cycle_contribution
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in cyclical pattern consistency: {e}")
            return predictions
    
    def _apply_trending_pattern_consistency(self, predictions: np.ndarray, 
                                          trending_patterns: Dict, 
                                          confidence: float) -> np.ndarray:
        """Apply trending pattern consistency"""
        try:
            linear_trend = trending_patterns.get('linear', {})
            historical_slope = linear_trend.get('slope', 0)
            trend_r2 = linear_trend.get('r2', 0)
            
            if trend_r2 < 0.3:  # Weak trend
                return predictions
            
            enhanced = predictions.copy()
            
            # Calculate current trend in predictions
            x = np.arange(len(enhanced))
            current_slope = np.polyfit(x, enhanced, 1)[0]
            
            # Adjust trend to be more consistent with historical trend
            trend_adjustment_strength = confidence * self.trend_following_strength * trend_r2
            slope_difference = historical_slope - current_slope
            
            # Apply trend adjustment
            for i in range(len(enhanced)):
                trend_adjustment = slope_difference * (i + 1) * trend_adjustment_strength * 0.3
                enhanced[i] += trend_adjustment
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in trending pattern consistency: {e}")
            return predictions
    
    def _apply_volatility_pattern_consistency(self, predictions: np.ndarray, 
                                            volatility_patterns: Dict, 
                                            confidence: float) -> np.ndarray:
        """Apply volatility pattern consistency"""
        try:
            historical_volatility = volatility_patterns.get('overall_volatility', 0)
            
            if historical_volatility <= 0:
                return predictions
            
            enhanced = predictions.copy()
            
            # Calculate current volatility
            current_changes = np.diff(enhanced)
            current_volatility = np.std(current_changes)
            
            # Adjust volatility to match historical patterns
            if current_volatility > 0:
                target_volatility = historical_volatility * 0.8  # Slightly reduced for stability
                volatility_ratio = target_volatility / current_volatility
                volatility_ratio = np.clip(volatility_ratio, 0.5, 2.0)  # Limit adjustment
                
                adjustment_strength = confidence * self.volatility_matching_strength * 0.5
                
                # Apply volatility adjustment to changes
                adjusted_changes = current_changes * (1 + (volatility_ratio - 1) * adjustment_strength)
                
                # Reconstruct predictions
                enhanced[1:] = enhanced[0] + np.cumsum(adjusted_changes)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in volatility pattern consistency: {e}")
            return predictions
    
    def _apply_adaptive_boundary_preservation(self, predictions: np.ndarray) -> np.ndarray:
        """Apply adaptive boundary preservation"""
        try:
            if len(self.historical_data) < 3:
                return predictions
            
            # Calculate adaptive boundaries
            historical_mean = np.mean(self.historical_data)
            historical_std = np.std(self.historical_data)
            historical_min = np.min(self.historical_data)
            historical_max = np.max(self.historical_data)
            
            # Adaptive boundary calculation
            boundary_expansion = historical_std * 1.5  # Allow reasonable expansion
            soft_lower = historical_min - boundary_expansion
            soft_upper = historical_max + boundary_expansion
            
            # Hard boundaries (extreme limits)
            hard_expansion = historical_std * 3
            hard_lower = historical_min - hard_expansion
            hard_upper = historical_max + hard_expansion
            
            bounded = predictions.copy()
            
            # Apply soft boundaries with gradual correction
            for i in range(len(bounded)):
                if bounded[i] < soft_lower:
                    excess = soft_lower - bounded[i]
                    correction = min(excess, boundary_expansion * 0.5)
                    bounded[i] += correction * 0.8
                elif bounded[i] > soft_upper:
                    excess = bounded[i] - soft_upper
                    correction = min(excess, boundary_expansion * 0.5)
                    bounded[i] -= correction * 0.8
            
            # Apply hard boundaries
            bounded = np.clip(bounded, hard_lower, hard_upper)
            
            return bounded
            
        except Exception as e:
            logger.error(f"Error in adaptive boundary preservation: {e}")
            return predictions
    
    def _apply_smoothness_preservation(self, predictions: np.ndarray) -> np.ndarray:
        """Apply smoothness preservation to avoid unrealistic jumps"""
        try:
            if len(predictions) < 3:
                return predictions
            
            smoothed = predictions.copy()
            
            # Calculate smoothness metrics from historical data
            if len(self.historical_data) > 3:
                historical_changes = np.diff(self.historical_data)
                typical_change = np.median(np.abs(historical_changes))
                max_reasonable_change = np.percentile(np.abs(historical_changes), 95)
            else:
                typical_change = np.std(predictions) * 0.1
                max_reasonable_change = typical_change * 5
            
            # Smooth extreme changes
            changes = np.diff(smoothed)
            for i in range(len(changes)):
                if abs(changes[i]) > max_reasonable_change:
                    # Reduce extreme change
                    sign = np.sign(changes[i])
                    changes[i] = sign * max_reasonable_change * 0.8
            
            # Reconstruct predictions from smoothed changes
            smoothed[1:] = smoothed[0] + np.cumsum(changes)
            
            # Apply light smoothing filter
            if len(smoothed) >= 5:
                window_size = min(5, len(smoothed)//2*2-1)
                if window_size >= 3:
                    filter_strength = 0.3  # Light smoothing
                    filtered = savgol_filter(smoothed, window_size, 2)
                    smoothed = smoothed * (1 - filter_strength) + filtered * filter_strength
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error in smoothness preservation: {e}")
            return predictions
    
    def _apply_real_time_adaptations(self, predictions: np.ndarray,
                                   real_time_feedback: Optional[Dict] = None) -> np.ndarray:
        """Apply real-time learning adaptations"""
        try:
            adaptive = predictions.copy()
            
            # 1. Historical prediction performance adaptation
            if len(self.prediction_history) > 2:
                recent_quality = [ph.get('quality_metrics', {}).get('overall_quality', 0.5) 
                                for ph in self.prediction_history[-3:]]
                avg_quality = np.mean(recent_quality)
                
                if avg_quality < self.min_quality_threshold:
                    # Increase pattern following strength
                    self.pattern_strength_multiplier = min(1.5, self.pattern_strength_multiplier * 1.1)
                    self.variability_preservation_strength = min(1.0, self.variability_preservation_strength * 1.05)
                elif avg_quality > 0.85:
                    # Slightly reduce strength to prevent overfitting
                    self.pattern_strength_multiplier = max(0.8, self.pattern_strength_multiplier * 0.98)
            
            # 2. Real-time feedback adaptation
            if real_time_feedback is not None:
                user_quality_score = real_time_feedback.get('quality_score')
                if user_quality_score is not None:
                    if user_quality_score < 0.5:
                        # Increase variability and pattern following
                        adaptive = self._enhance_prediction_responsiveness(adaptive)
                    elif user_quality_score > 0.8:
                        # Apply additional smoothing
                        adaptive = self._apply_prediction_smoothing(adaptive)
            
            # 3. Continuous learning from prediction history
            if len(self.prediction_history) >= 5:
                adaptive = self._apply_continuous_learning_adaptation(adaptive)
            
            return adaptive
            
        except Exception as e:
            logger.error(f"Error in real-time adaptations: {e}")
            return predictions
    
    def _apply_final_quality_enhancements(self, predictions: np.ndarray) -> np.ndarray:
        """Apply final quality enhancements before output"""
        try:
            enhanced = predictions.copy()
            
            # 1. Final variability check and enhancement
            if len(self.historical_data) > 5:
                target_variability = np.std(self.historical_data) * 0.9
                current_variability = np.std(enhanced)
                
                if current_variability < target_variability * 0.7:
                    # Final variability enhancement
                    enhancement_factor = target_variability / (current_variability + 1e-10)
                    enhancement_factor = min(1.3, enhancement_factor)
                    
                    mean_pred = np.mean(enhanced)
                    enhanced = mean_pred + (enhanced - mean_pred) * enhancement_factor
            
            # 2. Final continuity check
            if len(self.historical_data) > 0:
                last_historical = self.historical_data[-1]
                first_prediction = enhanced[0]
                
                # Ensure reasonable transition
                historical_std = np.std(self.historical_data) if len(self.historical_data) > 1 else 1.0
                max_jump = historical_std * 2
                
                if abs(first_prediction - last_historical) > max_jump:
                    # Apply transition smoothing
                    transition_adjustment = (last_historical - first_prediction) * 0.5
                    transition_decay = np.exp(-np.arange(len(enhanced)) * 0.2)
                    enhanced += transition_adjustment * transition_decay
            
            # 3. Final boundary check
            if len(self.historical_data) > 3:
                hist_min, hist_max = np.min(self.historical_data), np.max(self.historical_data)
                hist_range = hist_max - hist_min
                buffer = hist_range * 0.2
                
                final_lower = hist_min - buffer
                final_upper = hist_max + buffer
                
                enhanced = np.clip(enhanced, final_lower, final_upper)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in final quality enhancements: {e}")
            return predictions
    
    def _calculate_prediction_quality_metrics(self, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for predictions"""
        try:
            if len(self.historical_data) < 3:
                return {'overall_quality': 0.5, 'overall_confidence': 0.5}
            
            metrics = {}
            
            # 1. Variability preservation score
            hist_std = np.std(self.historical_data)
            pred_std = np.std(predictions)
            variability_score = 1 - abs(hist_std - pred_std) / (hist_std + 1e-10)
            metrics['variability_preservation'] = max(0, min(1, variability_score))
            
            # 2. Range consistency score
            hist_range = np.max(self.historical_data) - np.min(self.historical_data)
            pred_range = np.max(predictions) - np.min(predictions)
            range_score = 1 - abs(hist_range - pred_range) / (hist_range + 1e-10)
            metrics['range_consistency'] = max(0, min(1, range_score))
            
            # 3. Pattern following score
            pattern_confidence = self.pattern_analysis_cache.get('pattern_confidence', {})
            overall_pattern_confidence = pattern_confidence.get('overall', 0.5)
            
            # Enhanced pattern following calculation
            pattern_scores = []
            
            # Cyclical pattern following
            cyclical_confidence = pattern_confidence.get('cyclical', 0)
            if cyclical_confidence > 0.3:
                autocorr = np.correlate(predictions, predictions, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                if len(autocorr) > 1:
                    autocorr = autocorr / autocorr[0]
                    pred_cyclical_strength = np.max(autocorr[1:min(len(autocorr), 10)]) if len(autocorr) > 1 else 0
                    cyclical_score = min(1, pred_cyclical_strength * 2)
                    pattern_scores.append(cyclical_score * cyclical_confidence)
            
            # Trending pattern following
            trending_confidence = pattern_confidence.get('trending', 0)
            if trending_confidence > 0.3 and len(predictions) > 3:
                x = np.arange(len(predictions))
                pred_r2 = r2_score(predictions, np.polyval(np.polyfit(x, predictions, 1), x))
                trending_score = max(0, pred_r2)
                pattern_scores.append(trending_score * trending_confidence)
            
            if pattern_scores:
                metrics['pattern_following_score'] = np.mean(pattern_scores)
            else:
                metrics['pattern_following_score'] = overall_pattern_confidence
            
            # 4. Bias prevention score
            hist_mean = np.mean(self.historical_data)
            pred_mean = np.mean(predictions)
            bias = abs(pred_mean - hist_mean) / (np.std(self.historical_data) + 1e-10)
            bias_score = max(0, 1 - bias)
            metrics['bias_prevention'] = bias_score
            
            # 5. Continuity score
            if len(self.historical_data) > 0:
                transition_diff = abs(predictions[0] - self.historical_data[-1])
                typical_change = np.mean(np.abs(np.diff(self.historical_data[-10:]))) if len(self.historical_data) > 10 else np.std(self.historical_data)
                continuity_score = max(0, 1 - transition_diff / (typical_change * 3 + 1e-10))
                metrics['continuity_score'] = continuity_score
            else:
                metrics['continuity_score'] = 1.0
            
            # 6. Smoothness score
            pred_changes = np.diff(predictions)
            hist_changes = np.diff(self.historical_data)
            
            pred_change_std = np.std(pred_changes)
            hist_change_std = np.std(hist_changes)
            
            smoothness_score = 1 - abs(pred_change_std - hist_change_std) / (hist_change_std + 1e-10)
            metrics['smoothness_score'] = max(0, min(1, smoothness_score))
            
            # 7. Overall quality and confidence
            quality_weights = {
                'variability_preservation': 0.25,
                'range_consistency': 0.15,
                'pattern_following_score': 0.25,
                'bias_prevention': 0.2,
                'continuity_score': 0.1,
                'smoothness_score': 0.05
            }
            
            overall_quality = sum(metrics.get(k, 0.5) * weight for k, weight in quality_weights.items())
            metrics['overall_quality'] = overall_quality
            
            # Overall confidence combines quality with pattern confidence
            overall_confidence = (overall_quality * 0.7 + overall_pattern_confidence * 0.3)
            metrics['overall_confidence'] = overall_confidence
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in quality metrics calculation: {e}")
            return {'overall_quality': 0.5, 'overall_confidence': 0.5}
    
    # Helper methods
    def _detrend_data(self, data: np.ndarray) -> np.ndarray:
        """Remove trend from data"""
        try:
            x = np.arange(len(data))
            trend_line = np.polyval(np.polyfit(x, data, 1), x)
            return data - trend_line
        except:
            return data - np.mean(data)
    
    def _generate_simple_extrapolation(self, steps: int) -> np.ndarray:
        """Generate simple extrapolation when pattern analysis fails"""
        try:
            if len(self.historical_data) < 2:
                return np.full(steps, np.mean(self.historical_data) if len(self.historical_data) > 0 else 0)
            
            # Simple trend extrapolation with noise
            recent_data = self.historical_data[-min(10, len(self.historical_data)):]
            trend = np.polyfit(np.arange(len(recent_data)), recent_data, 1)[0]
            start_value = self.historical_data[-1]
            
            predictions = np.array([start_value + trend * (i + 1) * 0.5 for i in range(steps)])
            
            # Add realistic noise
            noise_std = np.std(self.historical_data) * 0.1
            noise = np.random.normal(0, noise_std, steps)
            predictions += noise
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in simple extrapolation: {e}")
            return np.full(steps, np.mean(self.historical_data) if len(self.historical_data) > 0 else 0)
    
    def _apply_prediction_continuity(self, predictions: np.ndarray, 
                                   previous_predictions: List) -> np.ndarray:
        """Apply continuity with previous predictions"""
        try:
            if not previous_predictions or len(previous_predictions) == 0:
                return predictions
            
            # Get last few values from previous predictions
            prev_values = previous_predictions[-min(5, len(previous_predictions)):]
            
            # Ensure smooth transition
            if prev_values:
                last_prev = prev_values[-1]
                first_current = predictions[0]
                
                # Calculate transition adjustment
                transition_diff = first_current - last_prev
                max_reasonable_diff = np.std(self.historical_data) * 1.5 if len(self.historical_data) > 1 else abs(first_current) * 0.5
                
                if abs(transition_diff) > max_reasonable_diff:
                    # Apply transition smoothing
                    adjustment = transition_diff * 0.5
                    decay = np.exp(-np.arange(len(predictions)) * 0.3)
                    predictions -= adjustment * decay
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction continuity: {e}")
            return predictions
    
    def _update_prediction_history(self, predictions: np.ndarray, quality_metrics: Dict[str, float]):
        """Update prediction history for learning"""
        try:
            prediction_entry = {
                'predictions': predictions.tolist(),
                'quality_metrics': quality_metrics,
                'timestamp': datetime.now().isoformat(),
                'pattern_confidence': self.pattern_analysis_cache.get('pattern_confidence', {})
            }
            
            self.prediction_history.append(prediction_entry)
            
            # Limit history size
            if len(self.prediction_history) > self.max_prediction_history:
                self.prediction_history = self.prediction_history[-self.max_prediction_history:]
            
        except Exception as e:
            logger.error(f"Error updating prediction history: {e}")
    
    def _update_pattern_analysis(self):
        """Update pattern analysis periodically"""
        try:
            if len(self.historical_data) > 5:
                # Re-analyze patterns
                updated_analysis = self.pattern_memory.learn_patterns_from_data(self.historical_data)
                self.pattern_analysis_cache = updated_analysis
                logger.info("Pattern analysis updated")
                
        except Exception as e:
            logger.error(f"Error updating pattern analysis: {e}")
    
    def _initialize_prediction_parameters(self, pattern_analysis: Dict[str, Any]):
        """Initialize prediction parameters based on pattern analysis"""
        try:
            confidence = pattern_analysis.get('pattern_confidence', {})
            overall_confidence = confidence.get('overall', 0.5)
            
            # Adjust parameters based on pattern confidence
            if overall_confidence > 0.7:
                self.pattern_strength_multiplier = 1.3
                self.variability_preservation_strength = 0.95
            elif overall_confidence > 0.5:
                self.pattern_strength_multiplier = 1.1
                self.variability_preservation_strength = 0.9
            else:
                self.pattern_strength_multiplier = 0.9
                self.variability_preservation_strength = 0.85
                
        except Exception as e:
            logger.error(f"Error initializing prediction parameters: {e}")
    
    def _calculate_data_quality(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate data quality metrics"""
        try:
            metrics = {}
            
            metrics['data_length'] = len(data)
            metrics['completeness'] = 1.0  # Already filtered for valid data
            metrics['variability'] = np.std(data) / (np.mean(data) + 1e-10)
            metrics['range_coverage'] = (np.max(data) - np.min(data)) / (np.std(data) + 1e-10)
            
            # Overall data quality
            quality_score = min(1.0, (
                min(1.0, len(data) / 50) * 0.3 +  # Length score
                min(1.0, metrics['variability'] * 2) * 0.4 +  # Variability score
                min(1.0, metrics['range_coverage'] / 3) * 0.3  # Range score
            ))
            metrics['overall_quality'] = quality_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            return {'overall_quality': 0.5}
    
    def _minimal_initialization(self, data: np.ndarray) -> Dict[str, Any]:
        """Minimal initialization when full initialization fails"""
        try:
            self.historical_data = data
            self.initialized = True
            
            return {
                'status': 'minimal',
                'data_points': len(data),
                'pattern_analysis': {'learned_patterns': {}, 'pattern_confidence': {'overall': 0.3}},
                'quality_metrics': {'overall_quality': 0.3}
            }
            
        except Exception as e:
            logger.error(f"Error in minimal initialization: {e}")
            return {
                'status': 'failed',
                'data_points': 0,
                'pattern_analysis': {'learned_patterns': {}, 'pattern_confidence': {'overall': 0.1}},
                'quality_metrics': {'overall_quality': 0.1}
            }
    
    def _generate_fallback_prediction(self, steps: int) -> Dict[str, Any]:
        """Generate fallback prediction when main prediction fails"""
        try:
            if self.historical_data is not None and len(self.historical_data) > 0:
                predictions = self._generate_simple_extrapolation(steps)
            else:
                predictions = np.zeros(steps)
            
            return {
                'predictions': predictions.tolist(),
                'quality_metrics': {'overall_quality': 0.3, 'overall_confidence': 0.3},
                'pattern_analysis': {},
                'prediction_confidence': 0.3,
                'metadata': {'system_status': 'fallback'}
            }
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return {
                'predictions': [0.0] * steps,
                'quality_metrics': {'overall_quality': 0.1, 'overall_confidence': 0.1},
                'pattern_analysis': {},
                'prediction_confidence': 0.1,
                'metadata': {'system_status': 'error'}
            }
    
    # Additional helper methods for real-time adaptations
    def _enhance_prediction_responsiveness(self, predictions: np.ndarray) -> np.ndarray:
        """Enhance prediction responsiveness when quality is low"""
        try:
            enhanced = predictions.copy()
            
            # Increase variability
            if len(self.historical_data) > 3:
                target_std = np.std(self.historical_data) * 1.1
                current_std = np.std(enhanced)
                if current_std < target_std * 0.8:
                    enhancement_factor = target_std / (current_std + 1e-10)
                    enhancement_factor = min(1.5, enhancement_factor)
                    mean_pred = np.mean(enhanced)
                    enhanced = mean_pred + (enhanced - mean_pred) * enhancement_factor
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in enhancing prediction responsiveness: {e}")
            return predictions
    
    def _apply_prediction_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply additional smoothing when quality is high"""
        try:
            if len(predictions) < 5:
                return predictions
            
            smoothed = predictions.copy()
            window_size = min(5, len(smoothed)//2*2-1)
            if window_size >= 3:
                filtered = savgol_filter(smoothed, window_size, 2)
                smoothed = smoothed * 0.8 + filtered * 0.2
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error in prediction smoothing: {e}")
            return predictions
    
    def _apply_continuous_learning_adaptation(self, predictions: np.ndarray) -> np.ndarray:
        """Apply continuous learning adaptation based on prediction history"""
        try:
            # Analyze recent prediction quality trends
            recent_qualities = [ph.get('quality_metrics', {}).get('overall_quality', 0.5) 
                             for ph in self.prediction_history[-5:]]
            
            if len(recent_qualities) >= 3:
                quality_trend = np.polyfit(range(len(recent_qualities)), recent_qualities, 1)[0]
                
                if quality_trend < -0.05:  # Quality declining
                    # Increase pattern following strength
                    adaptive_factor = 1.1
                elif quality_trend > 0.05:  # Quality improving
                    # Maintain current approach
                    adaptive_factor = 1.0
                else:
                    # Slight adjustment
                    adaptive_factor = 1.02
                
                # Apply adaptive factor
                if len(self.historical_data) > 0:
                    historical_mean = np.mean(self.historical_data)
                    predictions = historical_mean + (predictions - historical_mean) * adaptive_factor
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in continuous learning adaptation: {e}")
            return predictions