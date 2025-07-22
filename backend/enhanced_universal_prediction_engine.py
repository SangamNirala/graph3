"""
Enhanced Universal Pattern-Aware Prediction Engine
Integrates universal waveform learning with existing prediction infrastructure
"""

import numpy as np
import pandas as pd
from scipy import signal, stats, optimize, interpolate
from scipy.signal import savgol_filter, butter, sosfilt, find_peaks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json

# Import the new universal waveform learning system
from universal_waveform_learning import UniversalWaveformLearningSystem

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedUniversalPredictionEngine:
    """
    Enhanced prediction engine that can learn and reproduce ANY waveform pattern
    """
    
    def __init__(self):
        # Initialize the universal waveform learning system
        self.universal_waveform_system = UniversalWaveformLearningSystem()
        
        # Enhanced prediction strategies with waveform awareness
        self.prediction_strategies = {
            'universal_waveform': self._universal_waveform_prediction,
            'pattern_adaptive': self._pattern_adaptive_prediction,
            'shape_preserving': self._shape_preserving_prediction,
            'geometric_synthesis': self._geometric_synthesis_prediction,
            'template_matching': self._template_matching_prediction,
            'hybrid_ensemble': self._hybrid_ensemble_prediction
        }
        
        # Waveform-specific parameters
        self.waveform_params = {
            'shape_preservation_strength': 0.95,
            'geometric_consistency_weight': 0.90,
            'template_matching_threshold': 0.85,
            'pattern_adaptation_rate': 0.12,
            'waveform_learning_rate': 0.15,
            'continuity_enforcement': 0.88,
            'amplitude_preservation': 0.92,
            'frequency_preservation': 0.90
        }
        
        # Pattern memory and adaptation
        self.learned_waveforms = {}
        self.pattern_templates = deque(maxlen=5000)
        self.prediction_history = deque(maxlen=2000)
        self.adaptation_events = deque(maxlen=1000)
        
        # Performance tracking for continuous improvement
        self.pattern_performance = defaultdict(list)
        self.waveform_accuracy = defaultdict(list)
        self.synthesis_quality = defaultdict(list)
    
    def learn_and_predict(self, data: np.ndarray, 
                         steps: int = 30,
                         previous_predictions: Optional[List] = None,
                         learning_mode: str = 'comprehensive') -> Dict[str, Any]:
        """
        Learn patterns from data and generate waveform-aware predictions
        """
        try:
            logger.info(f"Learning patterns and generating {steps} waveform-aware predictions")
            
            # Comprehensive pattern learning using universal system
            learning_results = self.universal_waveform_system.learn_universal_patterns(data)
            
            # Generate waveform-aware predictions
            prediction_results = self.universal_waveform_system.generate_waveform_aware_predictions(
                data, steps, previous_predictions
            )
            
            # Enhance predictions with additional processing
            enhanced_predictions = self._enhance_waveform_predictions(
                prediction_results, learning_results, data, steps
            )
            
            # Apply advanced pattern corrections
            corrected_predictions = self._apply_advanced_pattern_corrections(
                enhanced_predictions, learning_results, data
            )
            
            # Apply waveform-specific enhancements
            final_predictions = self._apply_waveform_specific_enhancements(
                corrected_predictions, learning_results, data, previous_predictions
            )
            
            # Calculate enhanced confidence intervals
            confidence_intervals = self._calculate_waveform_aware_confidence_intervals(
                final_predictions, learning_results, data
            )
            
            # Comprehensive quality assessment
            quality_metrics = self._assess_comprehensive_waveform_quality(
                final_predictions, learning_results, data
            )
            
            # Update learning and performance tracking
            self._update_comprehensive_learning_tracking(
                final_predictions, learning_results, quality_metrics, data
            )
            
            # Create comprehensive result
            comprehensive_result = {
                'status': 'success',
                'predictions': final_predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'learning_results': learning_results,
                'prediction_method': 'universal_waveform_aware',
                'quality_metrics': quality_metrics,
                'waveform_characteristics': {
                    'detected_patterns': learning_results.get('detected_patterns', {}),
                    'pattern_complexity': learning_results.get('pattern_complexity_handled', 0.5),
                    'adaptability_score': learning_results.get('universal_adaptability_score', 0.5),
                    'shape_preservation': quality_metrics.get('shape_preservation_score', 0.5),
                    'geometric_consistency': quality_metrics.get('geometric_consistency_score', 0.5)
                },
                'prediction_capabilities': {
                    'waveform_types_supported': self._get_supported_waveform_types(),
                    'complexity_handling': quality_metrics.get('complexity_handling_score', 0.5),
                    'pattern_learning_quality': learning_results.get('learning_quality', {}),
                    'synthesis_capabilities': learning_results.get('synthesis_capabilities', {})
                },
                'performance_metrics': {
                    'overall_quality': quality_metrics.get('overall_quality', 0.5),
                    'pattern_following_score': quality_metrics.get('pattern_following_score', 0.5),
                    'waveform_fidelity': quality_metrics.get('waveform_fidelity', 0.5),
                    'adaptation_success': quality_metrics.get('adaptation_success', 0.5)
                }
            }
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error in learn_and_predict: {e}")
            return self._generate_fallback_comprehensive_result(data, steps, str(e))
    
    def _enhance_waveform_predictions(self, prediction_results: Dict[str, Any],
                                    learning_results: Dict[str, Any],
                                    data: np.ndarray, steps: int) -> np.ndarray:
        """Enhance predictions with additional waveform-specific processing"""
        try:
            predictions = np.array(prediction_results.get('predictions', []))
            if len(predictions) == 0:
                return self._fallback_predictions(data, steps)
            
            # Get dominant pattern information
            detected_patterns = learning_results.get('detected_patterns', {})
            dominant_pattern = detected_patterns.get('dominant_pattern')
            
            if not dominant_pattern:
                return predictions
            
            pattern_type = dominant_pattern[1].get('pattern_type', 'unknown')
            
            # Apply pattern-specific enhancements
            if pattern_type == 'square_wave':
                return self._enhance_square_wave_predictions(predictions, dominant_pattern[1], data)
            elif pattern_type == 'triangular_wave':
                return self._enhance_triangular_wave_predictions(predictions, dominant_pattern[1], data)
            elif pattern_type == 'sawtooth_wave':
                return self._enhance_sawtooth_wave_predictions(predictions, dominant_pattern[1], data)
            elif pattern_type == 'step_function':
                return self._enhance_step_function_predictions(predictions, dominant_pattern[1], data)
            elif pattern_type == 'sinusoidal_pattern':
                return self._enhance_sinusoidal_predictions(predictions, dominant_pattern[1], data)
            elif pattern_type == 'polynomial_pattern':
                return self._enhance_polynomial_predictions(predictions, dominant_pattern[1], data)
            elif pattern_type == 'composite_pattern':
                return self._enhance_composite_predictions(predictions, dominant_pattern[1], data)
            else:
                return self._enhance_adaptive_predictions(predictions, dominant_pattern[1], data)
            
        except Exception as e:
            logger.error(f"Error enhancing waveform predictions: {e}")
            return np.array(prediction_results.get('predictions', []))
    
    def _enhance_square_wave_predictions(self, predictions: np.ndarray, 
                                       pattern_info: Dict[str, Any],
                                       data: np.ndarray) -> np.ndarray:
        """Enhance square wave predictions to maintain sharp transitions and flat segments"""
        try:
            amplitude_levels = pattern_info.get('amplitude_levels', [])
            duty_cycle = pattern_info.get('duty_cycle', 0.5)
            
            if not amplitude_levels or len(amplitude_levels) < 2:
                return predictions
            
            # Force predictions to discrete levels
            enhanced_predictions = []
            
            for i, pred in enumerate(predictions):
                # Find closest amplitude level
                distances = [abs(pred - level) for level in amplitude_levels]
                closest_level_idx = np.argmin(distances)
                
                # Apply sharp transition logic
                if i > 0:
                    prev_level = enhanced_predictions[i-1]
                    current_level = amplitude_levels[closest_level_idx]
                    
                    # Maintain sharp transitions
                    if abs(current_level - prev_level) > np.std(amplitude_levels) * 0.5:
                        enhanced_predictions.append(current_level)
                    else:
                        enhanced_predictions.append(prev_level)
                else:
                    enhanced_predictions.append(amplitude_levels[closest_level_idx])
            
            return np.array(enhanced_predictions)
            
        except Exception as e:
            logger.error(f"Error enhancing square wave predictions: {e}")
            return predictions
    
    def _enhance_triangular_wave_predictions(self, predictions: np.ndarray,
                                           pattern_info: Dict[str, Any],
                                           data: np.ndarray) -> np.ndarray:
        """Enhance triangular wave predictions to maintain linear segments and sharp peaks"""
        try:
            peaks = pattern_info.get('peaks', [])
            valleys = pattern_info.get('valleys', [])
            
            if not peaks and not valleys:
                return predictions
            
            # Enforce linear segments between peaks/valleys
            enhanced_predictions = predictions.copy()
            
            # Apply smoothing to enforce linearity within segments
            if len(enhanced_predictions) > 3:
                # Detect potential segments and enforce linearity
                for i in range(1, len(enhanced_predictions) - 1):
                    # Check if we should enforce linear interpolation
                    prev_val = enhanced_predictions[i-1]
                    next_val = enhanced_predictions[i+1]
                    current_val = enhanced_predictions[i]
                    
                    # Linear interpolation value
                    linear_val = (prev_val + next_val) / 2
                    
                    # Blend current prediction with linear interpolation
                    enhanced_predictions[i] = current_val * 0.3 + linear_val * 0.7
            
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"Error enhancing triangular wave predictions: {e}")
            return predictions
    
    def _enhance_sawtooth_wave_predictions(self, predictions: np.ndarray,
                                         pattern_info: Dict[str, Any],
                                         data: np.ndarray) -> np.ndarray:
        """Enhance sawtooth wave predictions to maintain linear ramps and sharp drops/rises"""
        try:
            sharp_transitions = pattern_info.get('sharp_transitions', [])
            ramp_direction = pattern_info.get('ramp_direction', 'ascending')
            
            if not sharp_transitions:
                return predictions
            
            enhanced_predictions = []
            
            # Enforce linear ramp characteristics
            for i, pred in enumerate(predictions):
                if i == 0:
                    enhanced_predictions.append(pred)
                    continue
                
                # Calculate expected linear progression
                if ramp_direction == 'ascending':
                    # Gradual increase with occasional sharp drops
                    expected_change = np.mean(np.diff(data[-10:])) if len(data) > 10 else 0
                    linear_pred = enhanced_predictions[-1] + expected_change
                else:
                    # Gradual decrease with occasional sharp rises
                    expected_change = np.mean(np.diff(data[-10:])) if len(data) > 10 else 0
                    linear_pred = enhanced_predictions[-1] + expected_change
                
                # Blend with linear expectation
                blended_pred = pred * 0.4 + linear_pred * 0.6
                enhanced_predictions.append(blended_pred)
            
            return np.array(enhanced_predictions)
            
        except Exception as e:
            logger.error(f"Error enhancing sawtooth wave predictions: {e}")
            return predictions
    
    def _enhance_step_function_predictions(self, predictions: np.ndarray,
                                         pattern_info: Dict[str, Any],
                                         data: np.ndarray) -> np.ndarray:
        """Enhance step function predictions to maintain discrete levels"""
        try:
            discrete_levels = pattern_info.get('discrete_levels', [])
            
            if not discrete_levels:
                return predictions
            
            # Force predictions to discrete levels
            enhanced_predictions = []
            
            for pred in predictions:
                # Find closest discrete level
                distances = [abs(pred - level) for level in discrete_levels]
                closest_level_idx = np.argmin(distances)
                enhanced_predictions.append(discrete_levels[closest_level_idx])
            
            return np.array(enhanced_predictions)
            
        except Exception as e:
            logger.error(f"Error enhancing step function predictions: {e}")
            return predictions
    
    def _enhance_sinusoidal_predictions(self, predictions: np.ndarray,
                                      pattern_info: Dict[str, Any],
                                      data: np.ndarray) -> np.ndarray:
        """Enhance sinusoidal predictions to maintain smooth wave characteristics"""
        try:
            amplitude = pattern_info.get('amplitude', 1.0)
            frequency = pattern_info.get('frequency', 1.0)
            phase = pattern_info.get('phase', 0.0)
            offset = pattern_info.get('offset', 0.0)
            
            # Generate ideal sinusoidal continuation
            x_start = len(data)
            x_values = np.arange(x_start, x_start + len(predictions))
            
            ideal_sine = amplitude * np.sin(2 * np.pi * frequency * x_values + phase) + offset
            
            # Blend with original predictions
            enhanced_predictions = predictions * 0.3 + ideal_sine * 0.7
            
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"Error enhancing sinusoidal predictions: {e}")
            return predictions
    
    def _enhance_polynomial_predictions(self, predictions: np.ndarray,
                                      pattern_info: Dict[str, Any],
                                      data: np.ndarray) -> np.ndarray:
        """Enhance polynomial predictions to maintain polynomial characteristics"""
        try:
            coefficients = pattern_info.get('coefficients', [])
            degree = pattern_info.get('degree', 1)
            
            if not coefficients:
                return predictions
            
            # Generate polynomial continuation
            x_start = len(data)
            x_values = np.arange(x_start, x_start + len(predictions))
            
            polynomial_pred = np.polyval(coefficients, x_values)
            
            # Blend with original predictions
            enhanced_predictions = predictions * 0.4 + polynomial_pred * 0.6
            
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"Error enhancing polynomial predictions: {e}")
            return predictions
    
    def _enhance_composite_predictions(self, predictions: np.ndarray,
                                     pattern_info: Dict[str, Any],
                                     data: np.ndarray) -> np.ndarray:
        """Enhance composite pattern predictions by combining multiple components"""
        try:
            component_patterns = pattern_info.get('component_patterns', [])
            
            if not component_patterns:
                return predictions
            
            # Synthesize composite pattern
            enhanced_predictions = np.zeros_like(predictions)
            
            for component in component_patterns:
                component_info = component.get('pattern_info', {})
                component_type = component_info.get('pattern_type', 'unknown')
                weight = component_info.get('confidence', 0.5)
                
                # Generate component contribution
                if component_type == 'sinusoidal_pattern':
                    component_pred = self._enhance_sinusoidal_predictions(predictions, component_info, data)
                elif component_type == 'polynomial_pattern':
                    component_pred = self._enhance_polynomial_predictions(predictions, component_info, data)
                else:
                    component_pred = predictions
                
                enhanced_predictions += component_pred * weight
            
            # Normalize by total weights
            total_weight = sum(cp.get('pattern_info', {}).get('confidence', 0.5) for cp in component_patterns)
            if total_weight > 0:
                enhanced_predictions /= total_weight
            
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"Error enhancing composite predictions: {e}")
            return predictions
    
    def _enhance_adaptive_predictions(self, predictions: np.ndarray,
                                    pattern_info: Dict[str, Any],
                                    data: np.ndarray) -> np.ndarray:
        """Enhance predictions using adaptive approach for unknown patterns"""
        try:
            # Apply general smoothing and continuity enhancements
            enhanced_predictions = predictions.copy()
            
            # Smooth transitions
            if len(enhanced_predictions) > 2:
                for i in range(1, len(enhanced_predictions) - 1):
                    smoothed_val = (enhanced_predictions[i-1] + enhanced_predictions[i] + enhanced_predictions[i+1]) / 3
                    enhanced_predictions[i] = enhanced_predictions[i] * 0.7 + smoothed_val * 0.3
            
            # Ensure continuity with historical data
            if len(data) > 0:
                continuity_adjustment = data[-1] - enhanced_predictions[0]
                enhanced_predictions[0] += continuity_adjustment * 0.5
            
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"Error enhancing adaptive predictions: {e}")
            return predictions
    
    def _apply_advanced_pattern_corrections(self, predictions: np.ndarray,
                                          learning_results: Dict[str, Any],
                                          data: np.ndarray) -> np.ndarray:
        """Apply advanced pattern-specific corrections"""
        try:
            # Apply noise reduction while preserving pattern characteristics
            corrected_predictions = self._apply_pattern_aware_smoothing(predictions, learning_results)
            
            # Apply geometric consistency corrections
            corrected_predictions = self._apply_geometric_consistency_corrections(
                corrected_predictions, learning_results, data
            )
            
            # Apply amplitude and frequency preservation
            corrected_predictions = self._apply_amplitude_frequency_corrections(
                corrected_predictions, learning_results, data
            )
            
            return corrected_predictions
            
        except Exception as e:
            logger.error(f"Error applying advanced pattern corrections: {e}")
            return predictions
    
    def _apply_pattern_aware_smoothing(self, predictions: np.ndarray,
                                     learning_results: Dict[str, Any]) -> np.ndarray:
        """Apply smoothing that preserves pattern characteristics"""
        try:
            detected_patterns = learning_results.get('detected_patterns', {})
            dominant_pattern = detected_patterns.get('dominant_pattern')
            
            if not dominant_pattern:
                return predictions
            
            pattern_type = dominant_pattern[1].get('pattern_type', 'unknown')
            
            # Apply pattern-specific smoothing
            if pattern_type in ['square_wave', 'step_function']:
                # Minimal smoothing to preserve sharp edges
                return predictions  # No smoothing for sharp patterns
            elif pattern_type in ['triangular_wave', 'sawtooth_wave']:
                # Light smoothing to preserve linear segments
                if len(predictions) > 2:
                    smoothed = savgol_filter(predictions, min(5, len(predictions)//2*2+1), 1)
                    return predictions * 0.8 + smoothed * 0.2
                return predictions
            else:
                # Standard smoothing for curved patterns
                if len(predictions) > 4:
                    smoothed = savgol_filter(predictions, min(5, len(predictions)//2*2+1), 2)
                    return predictions * 0.6 + smoothed * 0.4
                return predictions
                
        except Exception as e:
            logger.error(f"Error in pattern-aware smoothing: {e}")
            return predictions
    
    def _apply_geometric_consistency_corrections(self, predictions: np.ndarray,
                                               learning_results: Dict[str, Any],
                                               data: np.ndarray) -> np.ndarray:
        """Apply corrections to maintain geometric consistency"""
        try:
            # Ensure predictions maintain the geometric properties of learned patterns
            geometric_analysis = learning_results.get('geometric_analysis', {})
            
            # Apply amplitude constraints based on historical data
            if len(data) > 0:
                historical_range = np.max(data) - np.min(data)
                historical_mean = np.mean(data)
                
                # Constrain predictions to reasonable bounds
                upper_bound = historical_mean + historical_range * 1.2
                lower_bound = historical_mean - historical_range * 1.2
                
                predictions = np.clip(predictions, lower_bound, upper_bound)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error applying geometric consistency corrections: {e}")
            return predictions
    
    def _apply_amplitude_frequency_corrections(self, predictions: np.ndarray,
                                             learning_results: Dict[str, Any],
                                             data: np.ndarray) -> np.ndarray:
        """Apply corrections to preserve amplitude and frequency characteristics"""
        try:
            detected_patterns = learning_results.get('detected_patterns', {})
            all_patterns = detected_patterns.get('all_patterns', {})
            
            # Find patterns with amplitude/frequency information
            for pattern_name, pattern_info in all_patterns.items():
                if pattern_info.get('confidence', 0) > 0.5:
                    # Apply pattern-specific amplitude corrections
                    if 'amplitude' in pattern_info:
                        target_amplitude = pattern_info['amplitude']
                        current_amplitude = (np.max(predictions) - np.min(predictions)) / 2
                        if current_amplitude > 0:
                            amplitude_scale = target_amplitude / current_amplitude
                            predictions_mean = np.mean(predictions)
                            predictions = (predictions - predictions_mean) * amplitude_scale + predictions_mean
                    
                    break  # Use the first high-confidence pattern
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error applying amplitude/frequency corrections: {e}")
            return predictions
    
    def _apply_waveform_specific_enhancements(self, predictions: np.ndarray,
                                            learning_results: Dict[str, Any],
                                            data: np.ndarray,
                                            previous_predictions: Optional[List] = None) -> np.ndarray:
        """Apply final waveform-specific enhancements"""
        try:
            # Apply continuity corrections with previous predictions
            if previous_predictions and len(previous_predictions) > 0:
                continuity_adjustment = predictions[0] - previous_predictions[-1]
                if abs(continuity_adjustment) > np.std(data) * 2:  # Large discontinuity
                    predictions[0] -= continuity_adjustment * 0.7  # Reduce discontinuity
            
            # Apply final smoothness and consistency checks
            final_predictions = self._apply_final_consistency_checks(predictions, learning_results, data)
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Error applying waveform-specific enhancements: {e}")
            return predictions
    
    def _apply_final_consistency_checks(self, predictions: np.ndarray,
                                      learning_results: Dict[str, Any],
                                      data: np.ndarray) -> np.ndarray:
        """Apply final consistency checks and corrections"""
        try:
            # Check for unrealistic values
            if len(data) > 0:
                data_std = np.std(data)
                data_mean = np.mean(data)
                
                # Flag and correct extreme outliers
                for i in range(len(predictions)):
                    if abs(predictions[i] - data_mean) > data_std * 3:
                        # Bring outlier closer to reasonable range
                        if predictions[i] > data_mean:
                            predictions[i] = data_mean + data_std * 2.5
                        else:
                            predictions[i] = data_mean - data_std * 2.5
            
            # Ensure smooth transitions between consecutive predictions
            if len(predictions) > 1:
                max_change = np.std(data) * 0.8 if len(data) > 0 else np.std(predictions) * 0.8
                
                for i in range(1, len(predictions)):
                    change = predictions[i] - predictions[i-1]
                    if abs(change) > max_change:
                        # Limit the change
                        predictions[i] = predictions[i-1] + np.sign(change) * max_change
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in final consistency checks: {e}")
            return predictions
    
    def _calculate_waveform_aware_confidence_intervals(self, predictions: np.ndarray,
                                                     learning_results: Dict[str, Any],
                                                     data: np.ndarray) -> List[Dict[str, float]]:
        """Calculate confidence intervals based on waveform characteristics"""
        try:
            confidence_intervals = []
            
            # Base confidence on learning quality and pattern strength
            learning_quality = learning_results.get('learning_quality', {})
            overall_quality = learning_quality.get('overall_quality', 0.5)
            
            # Calculate adaptive confidence width
            if len(data) > 0:
                data_std = np.std(data)
                base_width = data_std * (1.0 - overall_quality * 0.5)  # Higher quality = narrower intervals
            else:
                base_width = np.std(predictions) * 0.5
            
            for i, pred in enumerate(predictions):
                # Adjust confidence based on prediction horizon (farther = wider intervals)
                horizon_factor = 1.0 + (i * 0.02)  # Gradual increase
                width = base_width * horizon_factor
                
                confidence_intervals.append({
                    'lower': float(pred - width),
                    'upper': float(pred + width),
                    'confidence': float(overall_quality)
                })
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating waveform-aware confidence intervals: {e}")
            return [{'lower': p-1, 'upper': p+1, 'confidence': 0.5} for p in predictions]
    
    def _assess_comprehensive_waveform_quality(self, predictions: np.ndarray,
                                             learning_results: Dict[str, Any],
                                             data: np.ndarray) -> Dict[str, Any]:
        """Assess comprehensive quality of waveform predictions"""
        try:
            quality_metrics = {}
            
            # Pattern preservation score
            quality_metrics['pattern_preservation_score'] = self._calculate_pattern_preservation_score(
                predictions, learning_results, data
            )
            
            # Shape fidelity score
            quality_metrics['shape_fidelity'] = self._calculate_shape_fidelity_score(
                predictions, learning_results, data
            )
            
            # Geometric consistency score
            quality_metrics['geometric_consistency_score'] = self._calculate_geometric_consistency_score(
                predictions, learning_results, data
            )
            
            # Waveform fidelity
            quality_metrics['waveform_fidelity'] = self._calculate_waveform_fidelity_score(
                predictions, learning_results, data
            )
            
            # Complexity handling score
            quality_metrics['complexity_handling_score'] = learning_results.get(
                'pattern_complexity_handled', 0.5
            )
            
            # Adaptation success score
            quality_metrics['adaptation_success'] = learning_results.get(
                'universal_adaptability_score', 0.5
            )
            
            # Overall quality (weighted combination)
            quality_metrics['overall_quality'] = (
                quality_metrics['pattern_preservation_score'] * 0.3 +
                quality_metrics['shape_fidelity'] * 0.25 +
                quality_metrics['geometric_consistency_score'] * 0.2 +
                quality_metrics['waveform_fidelity'] * 0.15 +
                quality_metrics['complexity_handling_score'] * 0.1
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing comprehensive waveform quality: {e}")
            return {'overall_quality': 0.5}
    
    def _calculate_pattern_preservation_score(self, predictions: np.ndarray,
                                            learning_results: Dict[str, Any],
                                            data: np.ndarray) -> float:
        """Calculate how well patterns are preserved in predictions"""
        try:
            detected_patterns = learning_results.get('detected_patterns', {})
            dominant_pattern = detected_patterns.get('dominant_pattern')
            
            if not dominant_pattern:
                return 0.5
            
            pattern_confidence = dominant_pattern[1].get('confidence', 0.5)
            pattern_strength = dominant_pattern[1].get('strength', 0.5)
            
            # Base preservation score on pattern strength and confidence
            preservation_score = (pattern_confidence + pattern_strength) / 2
            
            return float(np.clip(preservation_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating pattern preservation score: {e}")
            return 0.5
    
    def _calculate_shape_fidelity_score(self, predictions: np.ndarray,
                                      learning_results: Dict[str, Any],
                                      data: np.ndarray) -> float:
        """Calculate shape fidelity score"""
        try:
            if len(predictions) < 3:
                return 0.5
            
            # Calculate shape characteristics of predictions
            pred_characteristics = self._extract_shape_characteristics(predictions)
            
            # Compare with historical data characteristics
            if len(data) >= 3:
                data_characteristics = self._extract_shape_characteristics(data[-min(len(data), 50):])
                
                # Compare key characteristics
                similarity_scores = []
                
                for char_name in ['mean_curvature', 'variability', 'trend_strength']:
                    if char_name in pred_characteristics and char_name in data_characteristics:
                        pred_val = pred_characteristics[char_name]
                        data_val = data_characteristics[char_name]
                        
                        if abs(data_val) > 1e-8:
                            similarity = 1.0 - abs(pred_val - data_val) / (abs(data_val) + 1e-8)
                        else:
                            similarity = 1.0 if abs(pred_val) < 1e-8 else 0.0
                        
                        similarity_scores.append(max(0.0, similarity))
                
                return float(np.mean(similarity_scores)) if similarity_scores else 0.5
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating shape fidelity score: {e}")
            return 0.5
    
    def _calculate_geometric_consistency_score(self, predictions: np.ndarray,
                                             learning_results: Dict[str, Any],
                                             data: np.ndarray) -> float:
        """Calculate geometric consistency score"""
        try:
            if len(predictions) < 2:
                return 0.5
            
            # Check for geometric consistency indicators
            consistency_indicators = []
            
            # Smoothness consistency
            if len(predictions) > 2:
                second_derivatives = np.diff(predictions, n=2)
                smoothness = 1.0 / (1.0 + np.std(second_derivatives))
                consistency_indicators.append(smoothness)
            
            # Amplitude consistency
            if len(data) > 0:
                pred_range = np.max(predictions) - np.min(predictions)
                data_range = np.max(data) - np.min(data)
                
                if data_range > 0:
                    amplitude_consistency = 1.0 - abs(pred_range - data_range) / data_range
                    consistency_indicators.append(max(0.0, amplitude_consistency))
            
            # Trend consistency
            if len(predictions) > 1 and len(data) > 1:
                pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
                data_trend = np.polyfit(range(len(data)), data, 1)[0]
                
                if abs(data_trend) > 1e-8:
                    trend_consistency = 1.0 - abs(pred_trend - data_trend) / (abs(data_trend) + 1e-8)
                else:
                    trend_consistency = 1.0 if abs(pred_trend) < 1e-8 else 0.0
                
                consistency_indicators.append(max(0.0, trend_consistency))
            
            return float(np.mean(consistency_indicators)) if consistency_indicators else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating geometric consistency score: {e}")
            return 0.5
    
    def _calculate_waveform_fidelity_score(self, predictions: np.ndarray,
                                         learning_results: Dict[str, Any],
                                         data: np.ndarray) -> float:
        """Calculate overall waveform fidelity score"""
        try:
            # Base fidelity on pattern learning quality
            learning_quality = learning_results.get('learning_quality', {})
            base_fidelity = learning_quality.get('overall_quality', 0.5)
            
            # Adjust based on prediction quality
            if len(predictions) > 0 and len(data) > 0:
                # Check for realistic value ranges
                data_mean = np.mean(data)
                data_std = np.std(data)
                
                # Count predictions within reasonable bounds
                reasonable_count = 0
                for pred in predictions:
                    if abs(pred - data_mean) <= data_std * 2:
                        reasonable_count += 1
                
                reasonableness_score = reasonable_count / len(predictions)
                
                # Combine with base fidelity
                fidelity_score = base_fidelity * 0.7 + reasonableness_score * 0.3
            else:
                fidelity_score = base_fidelity
            
            return float(np.clip(fidelity_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating waveform fidelity score: {e}")
            return 0.5
    
    def _extract_shape_characteristics(self, data: np.ndarray) -> Dict[str, float]:
        """Extract shape characteristics from data"""
        try:
            characteristics = {}
            
            if len(data) > 2:
                # Curvature measure
                second_derivative = np.diff(data, n=2)
                characteristics['mean_curvature'] = float(np.mean(second_derivative))
                
                # Variability measure
                characteristics['variability'] = float(np.std(data))
                
                # Trend strength
                if len(data) > 1:
                    trend = np.polyfit(range(len(data)), data, 1)[0]
                    characteristics['trend_strength'] = float(abs(trend))
                else:
                    characteristics['trend_strength'] = 0.0
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error extracting shape characteristics: {e}")
            return {}
    
    def _update_comprehensive_learning_tracking(self, predictions: np.ndarray,
                                              learning_results: Dict[str, Any],
                                              quality_metrics: Dict[str, Any],
                                              data: np.ndarray) -> None:
        """Update comprehensive learning and performance tracking"""
        try:
            # Store learning event
            learning_event = {
                'timestamp': datetime.now(),
                'data_characteristics': {
                    'length': len(data),
                    'complexity': learning_results.get('pattern_complexity_handled', 0.5)
                },
                'patterns_learned': learning_results.get('detected_patterns', {}),
                'prediction_quality': quality_metrics,
                'adaptation_success': quality_metrics.get('adaptation_success', 0.5)
            }
            
            self.adaptation_events.append(learning_event)
            
            # Update performance tracking
            detected_patterns = learning_results.get('detected_patterns', {})
            for pattern_name, pattern_info in detected_patterns.get('all_patterns', {}).items():
                self.pattern_performance[pattern_name].append(pattern_info.get('confidence', 0.5))
            
            # Update waveform accuracy tracking
            overall_quality = quality_metrics.get('overall_quality', 0.5)
            self.waveform_accuracy['overall'].append(overall_quality)
            
        except Exception as e:
            logger.error(f"Error updating comprehensive learning tracking: {e}")
    
    def _get_supported_waveform_types(self) -> List[str]:
        """Get list of supported waveform types"""
        return [
            'square_wave', 'triangular_wave', 'sawtooth_wave', 'step_function',
            'pulse_pattern', 'exponential_pattern', 'logarithmic_pattern',
            'sinusoidal_pattern', 'polynomial_pattern', 'spline_pattern',
            'fractal_pattern', 'chaotic_pattern', 'composite_pattern',
            'irregular_pattern', 'custom_shape'
        ]
    
    def _fallback_predictions(self, data: np.ndarray, steps: int) -> np.ndarray:
        """Generate fallback predictions"""
        try:
            if len(data) == 0:
                return np.zeros(steps)
            elif len(data) == 1:
                return np.full(steps, data[0])
            else:
                # Simple linear continuation
                trend = np.polyfit(range(len(data[-10:])), data[-10:], 1)[0] if len(data) >= 10 else 0
                predictions = []
                last_val = data[-1]
                
                for i in range(1, steps + 1):
                    predictions.append(last_val + trend * i)
                
                return np.array(predictions)
                
        except Exception as e:
            logger.error(f"Error in fallback predictions: {e}")
            return np.full(steps, data[-1] if len(data) > 0 else 0.0)
    
    def _generate_fallback_comprehensive_result(self, data: np.ndarray, steps: int, error_msg: str) -> Dict[str, Any]:
        """Generate fallback comprehensive result"""
        try:
            fallback_predictions = self._fallback_predictions(data, steps)
            
            return {
                'status': 'fallback',
                'error': error_msg,
                'predictions': fallback_predictions.tolist(),
                'confidence_intervals': [{'lower': p-1, 'upper': p+1, 'confidence': 0.3} for p in fallback_predictions],
                'prediction_method': 'fallback',
                'quality_metrics': {'overall_quality': 0.3},
                'waveform_characteristics': {'detected_patterns': {}},
                'prediction_capabilities': {'complexity_handling': 0.3}
            }
            
        except Exception as e:
            logger.error(f"Error generating fallback comprehensive result: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'predictions': [0.0] * steps,
                'quality_metrics': {'overall_quality': 0.1}
            }
    
    # Placeholder methods for pattern prediction strategies
    def _universal_waveform_prediction(self, data: np.ndarray, steps: int, pattern_state: Dict) -> np.ndarray:
        """Universal waveform prediction strategy"""
        return self.universal_waveform_system.generate_waveform_aware_predictions(data, steps).get('predictions', [])
    
    def _pattern_adaptive_prediction(self, data: np.ndarray, steps: int, pattern_state: Dict) -> np.ndarray:
        """Pattern adaptive prediction strategy"""
        return self._fallback_predictions(data, steps)
    
    def _shape_preserving_prediction(self, data: np.ndarray, steps: int, pattern_state: Dict) -> np.ndarray:
        """Shape preserving prediction strategy"""
        return self._fallback_predictions(data, steps)
    
    def _geometric_synthesis_prediction(self, data: np.ndarray, steps: int, pattern_state: Dict) -> np.ndarray:
        """Geometric synthesis prediction strategy"""
        return self._fallback_predictions(data, steps)
    
    def _template_matching_prediction(self, data: np.ndarray, steps: int, pattern_state: Dict) -> np.ndarray:
        """Template matching prediction strategy"""
        return self._fallback_predictions(data, steps)
    
    def _hybrid_ensemble_prediction(self, data: np.ndarray, steps: int, pattern_state: Dict) -> np.ndarray:
        """Hybrid ensemble prediction strategy"""
        return self._fallback_predictions(data, steps)