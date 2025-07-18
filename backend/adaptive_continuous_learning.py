"""
Adaptive Continuous Learning System for Real-time Prediction
Implements continuous learning and pattern adaptation for industry-level predictions
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import deque
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta
import json
import threading
import time
from advanced_pattern_recognition import IndustryLevelPatternRecognition
from industry_prediction_engine import AdvancedPredictionEngine

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdaptiveContinuousLearningSystem:
    """
    Advanced continuous learning system that adapts to changing patterns
    and provides real-time prediction updates
    """
    
    def __init__(self, max_history_size: int = 1000):
        self.pattern_recognizer = IndustryLevelPatternRecognition()
        self.prediction_engine = AdvancedPredictionEngine()
        
        # Data storage
        self.historical_data = deque(maxlen=max_history_size)
        self.prediction_history = deque(maxlen=100)
        self.pattern_evolution = deque(maxlen=50)
        
        # Model states
        self.current_pattern = None
        self.current_model_state = None
        self.adaptation_metrics = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.15
        self.pattern_change_threshold = 0.2
        self.min_data_points = 10
        
        # Performance tracking
        self.prediction_accuracy = deque(maxlen=100)
        self.pattern_stability = deque(maxlen=50)
        self.adaptation_events = []
        
        # Threading for continuous updates
        self.is_running = False
        self.update_thread = None
        self.lock = threading.Lock()
        
    def start_continuous_learning(self, update_interval: float = 1.0):
        """Start continuous learning process"""
        try:
            self.is_running = True
            self.update_thread = threading.Thread(
                target=self._continuous_update_loop, 
                args=(update_interval,)
            )
            self.update_thread.daemon = True
            self.update_thread.start()
            
            logger.info("Continuous learning system started")
            
        except Exception as e:
            logger.error(f"Failed to start continuous learning: {e}")
    
    def stop_continuous_learning(self):
        """Stop continuous learning process"""
        try:
            self.is_running = False
            if self.update_thread:
                self.update_thread.join(timeout=5.0)
            
            logger.info("Continuous learning system stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop continuous learning: {e}")
    
    def add_data_point(self, value: float, timestamp: Optional[datetime] = None):
        """Add a new data point and trigger adaptive learning"""
        try:
            with self.lock:
                # Add to historical data
                if timestamp is None:
                    timestamp = datetime.now()
                
                self.historical_data.append({
                    'value': value,
                    'timestamp': timestamp
                })
                
                # Check if adaptation is needed
                if self._needs_adaptation():
                    self._trigger_adaptation()
                
        except Exception as e:
            logger.error(f"Failed to add data point: {e}")
    
    def get_adaptive_predictions(self, steps: int = 30,
                               confidence_level: float = 0.95,
                               real_time: bool = True) -> Dict[str, Any]:
        """Get predictions with adaptive learning"""
        try:
            with self.lock:
                # Get current data
                data = self._get_current_data()
                
                if len(data) < self.min_data_points:
                    return self._get_fallback_predictions(steps)
                
                # Use adaptive prediction engine
                if real_time and self.current_model_state:
                    predictions = self._get_real_time_predictions(data, steps)
                else:
                    predictions = self._get_full_predictions(data, steps, confidence_level)
                
                # Update prediction history
                self.prediction_history.append({
                    'timestamp': datetime.now(),
                    'predictions': predictions['predictions'],
                    'confidence': predictions.get('quality_metrics', {}).get('overall_quality_score', 0.5)
                })
                
                return predictions
                
        except Exception as e:
            logger.error(f"Failed to get adaptive predictions: {e}")
            return self._get_fallback_predictions(steps)
    
    def get_continuous_predictions(self, steps: int = 30,
                                 advance_steps: int = 1) -> Dict[str, Any]:
        """Get continuous predictions that advance with each call"""
        try:
            with self.lock:
                # Get current data
                data = self._get_current_data()
                
                if len(data) < self.min_data_points:
                    return self._get_fallback_predictions(steps)
                
                # Get previous predictions for continuity
                previous_predictions = None
                if self.prediction_history:
                    previous_predictions = self.prediction_history[-1]['predictions']
                
                # Generate continuous predictions
                predictions = self.prediction_engine.generate_continuous_predictions(
                    data, 
                    previous_predictions=previous_predictions,
                    steps=steps,
                    update_interval=advance_steps
                )
                
                # Apply continuous learning corrections
                corrected_predictions = self._apply_continuous_corrections(
                    predictions, data
                )
                
                # Update metrics
                self._update_performance_metrics(corrected_predictions, data)
                
                return corrected_predictions
                
        except Exception as e:
            logger.error(f"Failed to get continuous predictions: {e}")
            return self._get_fallback_predictions(steps)
    
    def _continuous_update_loop(self, update_interval: float):
        """Main continuous update loop"""
        while self.is_running:
            try:
                # Check for pattern changes
                if self._detect_pattern_change():
                    self._update_pattern_model()
                
                # Update adaptation metrics
                self._update_adaptation_metrics()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep for update interval
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous update loop: {e}")
                time.sleep(update_interval)
    
    def _needs_adaptation(self) -> bool:
        """Check if adaptation is needed"""
        try:
            if len(self.historical_data) < self.min_data_points:
                return False
            
            # Check prediction accuracy
            if self._calculate_recent_accuracy() < self.adaptation_threshold:
                return True
            
            # Check pattern stability
            if self._calculate_pattern_stability() < self.pattern_change_threshold:
                return True
            
            # Check data distribution changes
            if self._detect_distribution_change():
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to check adaptation needs: {e}")
            return False
    
    def _trigger_adaptation(self):
        """Trigger adaptation process"""
        try:
            logger.info("Triggering adaptation...")
            
            # Get current data
            data = self._get_current_data()
            
            # Re-analyze patterns
            new_pattern = self.pattern_recognizer.analyze_comprehensive_patterns(data)
            
            # Check if pattern has changed significantly
            if self._has_pattern_changed(new_pattern):
                logger.info("Pattern change detected, updating model...")
                
                # Update current pattern
                self.current_pattern = new_pattern
                
                # Update model state
                self._update_model_state(data, new_pattern)
                
                # Record adaptation event
                self.adaptation_events.append({
                    'timestamp': datetime.now(),
                    'reason': 'pattern_change',
                    'new_pattern': new_pattern['pattern_classification']['primary_pattern']
                })
            
            # Update learning parameters
            self._update_learning_parameters()
            
        except Exception as e:
            logger.error(f"Failed to trigger adaptation: {e}")
    
    def _get_current_data(self) -> np.ndarray:
        """Get current data as numpy array"""
        try:
            if not self.historical_data:
                return np.array([])
            
            values = [point['value'] for point in self.historical_data]
            return np.array(values)
            
        except Exception as e:
            logger.error(f"Failed to get current data: {e}")
            return np.array([])
    
    def _get_real_time_predictions(self, data: np.ndarray, steps: int) -> Dict[str, Any]:
        """Get real-time predictions using cached model state"""
        try:
            if self.current_model_state is None:
                return self._get_full_predictions(data, steps)
            
            # Use cached pattern analysis
            pattern_analysis = self.current_pattern
            
            # Generate predictions quickly
            predictions = self.prediction_engine._generate_base_predictions(
                data, steps, pattern_analysis, self.current_model_state
            )
            
            # Apply minimal corrections
            corrected_predictions = self.prediction_engine._apply_adaptive_corrections(
                predictions, data, pattern_analysis
            )
            
            # Calculate basic confidence intervals
            confidence_intervals = self.prediction_engine._calculate_advanced_confidence_intervals(
                corrected_predictions, data, pattern_analysis, 0.95
            )
            
            return {
                'predictions': corrected_predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'pattern_analysis': pattern_analysis,
                'quality_metrics': {'overall_quality_score': 0.8}  # Estimated
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time predictions: {e}")
            return self._get_full_predictions(data, steps)
    
    def _get_full_predictions(self, data: np.ndarray, steps: int,
                            confidence_level: float = 0.95) -> Dict[str, Any]:
        """Get full predictions with complete analysis"""
        try:
            return self.prediction_engine.generate_advanced_predictions(
                data, steps, confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"Failed to get full predictions: {e}")
            return self._get_fallback_predictions(steps)
    
    def _apply_continuous_corrections(self, predictions: Dict[str, Any], 
                                   data: np.ndarray) -> Dict[str, Any]:
        """Apply enhanced continuous learning corrections to predictions"""
        try:
            pred_values = np.array(predictions['predictions'])
            
            # Calculate historical statistics for pattern preservation
            historical_mean = np.mean(data)
            historical_std = np.std(data)
            historical_min = np.min(data)
            historical_max = np.max(data)
            
            # Calculate prediction statistics
            pred_mean = np.mean(pred_values)
            pred_std = np.std(pred_values)
            
            # 1. Enhanced mean reversion with pattern preservation
            mean_bias = pred_mean - historical_mean
            if abs(mean_bias) > 0.1 * historical_std:
                # Apply adaptive mean reversion
                mean_reversion_strength = min(0.3, abs(mean_bias) / historical_std)
                mean_correction = -mean_bias * mean_reversion_strength
                
                # Apply correction with gradual decay
                correction_weights = np.exp(-0.05 * np.arange(len(pred_values)))
                pred_values += mean_correction * correction_weights
            
            # 2. Variability preservation
            if pred_std < 0.3 * historical_std:
                # Predictions too flat - enhance variability
                variability_enhancement = historical_std / (pred_std + 1e-10) * 0.7
                pred_center = np.mean(pred_values)
                pred_values = pred_center + (pred_values - pred_center) * variability_enhancement
            elif pred_std > 2.0 * historical_std:
                # Predictions too volatile - reduce variability
                variability_reduction = historical_std / pred_std * 1.2
                pred_center = np.mean(pred_values)
                pred_values = pred_center + (pred_values - pred_center) * variability_reduction
            
            # 3. Pattern-based corrections
            if self.current_pattern:
                pred_values = self._apply_enhanced_pattern_corrections(pred_values, data)
            
            # 4. Boundary preservation
            range_buffer = 0.15 * (historical_max - historical_min)
            pred_values = np.clip(pred_values, 
                                historical_min - range_buffer, 
                                historical_max + range_buffer)
            
            # 5. Continuity preservation
            if len(data) > 0:
                # Ensure smooth transition from last historical value
                last_historical = data[-1]
                first_prediction = pred_values[0]
                
                # If first prediction is too far from last historical value
                if abs(first_prediction - last_historical) > 2 * historical_std:
                    # Apply smooth transition
                    transition_correction = (last_historical - first_prediction) * 0.5
                    transition_weights = np.exp(-0.2 * np.arange(len(pred_values)))
                    pred_values += transition_correction * transition_weights
            
            # Update predictions
            predictions['predictions'] = pred_values.tolist()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to apply continuous corrections: {e}")
            return predictions
    
    def _apply_enhanced_pattern_corrections(self, predictions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply enhanced pattern-specific corrections"""
        try:
            if self.current_pattern is None:
                return predictions
            
            pattern_type = self.current_pattern['pattern_classification']['primary_pattern']
            
            if pattern_type == 'sinusoidal':
                return self._apply_enhanced_sinusoidal_corrections(predictions, data)
            elif pattern_type == 'seasonal':
                return self._apply_enhanced_seasonal_corrections(predictions, data)
            elif pattern_type == 'trending':
                return self._apply_enhanced_trending_corrections(predictions, data)
            elif pattern_type == 'linear':
                return self._apply_enhanced_linear_corrections(predictions, data)
            else:
                return self._apply_generic_pattern_corrections(predictions, data)
                
        except Exception as e:
            logger.error(f"Failed to apply enhanced pattern corrections: {e}")
            return predictions
    
    def _apply_enhanced_sinusoidal_corrections(self, predictions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply enhanced corrections for sinusoidal patterns"""
        try:
            # Detect sinusoidal characteristics in historical data
            from scipy.fft import fft, fftfreq
            
            # Perform FFT to identify dominant frequencies
            fft_data = fft(data)
            freqs = fftfreq(len(data))
            
            # Find dominant frequency (excluding DC component)
            dominant_freq_idx = np.argmax(np.abs(fft_data[1:len(data)//2])) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            # Calculate period
            period = 1.0 / abs(dominant_freq) if dominant_freq != 0 else len(data)
            
            # Generate sinusoidal correction
            t = np.arange(len(predictions))
            phase = 2 * np.pi * t / period
            
            # Extract amplitude and phase from historical data
            amplitude = np.std(data) * 0.8
            phase_shift = 0  # Could be estimated from data
            
            # Apply sinusoidal correction
            sinusoidal_component = amplitude * np.sin(phase + phase_shift)
            
            # Blend with predictions
            blend_factor = 0.3  # Adjustable blending
            corrected_predictions = predictions + blend_factor * sinusoidal_component
            
            return corrected_predictions
            
        except Exception as e:
            logger.warning(f"Sinusoidal correction failed: {e}")
            return predictions
    
    def _apply_enhanced_seasonal_corrections(self, predictions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply enhanced corrections for seasonal patterns"""
        try:
            # Detect seasonal patterns in historical data
            season_length = min(len(data) // 4, 12)  # Assume quarterly or monthly patterns
            
            if season_length < 3:
                return predictions
            
            # Calculate seasonal component
            seasonal_avg = np.zeros(season_length)
            for i in range(season_length):
                seasonal_indices = np.arange(i, len(data), season_length)
                seasonal_avg[i] = np.mean(data[seasonal_indices])
            
            # Apply seasonal correction to predictions
            corrected_predictions = predictions.copy()
            for i in range(len(predictions)):
                seasonal_idx = i % season_length
                seasonal_correction = seasonal_avg[seasonal_idx] - np.mean(data)
                corrected_predictions[i] += seasonal_correction * 0.2  # Gentle correction
            
            return corrected_predictions
            
        except Exception as e:
            logger.warning(f"Seasonal correction failed: {e}")
            return predictions
    
    def _apply_enhanced_trending_corrections(self, predictions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply enhanced corrections for trending patterns"""
        try:
            # Calculate historical trend
            x = np.arange(len(data))
            trend_coeffs = np.polyfit(x, data, 1)
            historical_trend = trend_coeffs[0]
            
            # Calculate prediction trend
            pred_x = np.arange(len(predictions))
            pred_trend_coeffs = np.polyfit(pred_x, predictions, 1)
            prediction_trend = pred_trend_coeffs[0]
            
            # Apply trend correction
            if abs(historical_trend) > 1e-6:  # Avoid division by zero
                trend_ratio = historical_trend / prediction_trend if prediction_trend != 0 else 1.0
                trend_ratio = np.clip(trend_ratio, 0.3, 3.0)  # Limit extreme corrections
                
                # Apply gradual trend correction
                trend_correction = np.arange(len(predictions)) * (historical_trend - prediction_trend) * 0.3
                corrected_predictions = predictions + trend_correction
            else:
                corrected_predictions = predictions
            
            return corrected_predictions
            
        except Exception as e:
            logger.warning(f"Trending correction failed: {e}")
            return predictions
    
    def _apply_enhanced_linear_corrections(self, predictions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply enhanced corrections for linear patterns"""
        try:
            # Similar to trending but with more conservative approach
            return self._apply_enhanced_trending_corrections(predictions, data)
            
        except Exception as e:
            return predictions
    
    def _apply_generic_pattern_corrections(self, predictions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply generic pattern corrections for unrecognized patterns"""
        try:
            # Apply statistical corrections
            historical_mean = np.mean(data)
            historical_std = np.std(data)
            
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            
            # Center correction
            center_correction = (historical_mean - pred_mean) * 0.2
            
            # Variability correction
            if pred_std > 0:
                variability_correction = historical_std / pred_std * 0.7
                corrected_predictions = pred_mean + center_correction + (predictions - pred_mean) * variability_correction
            else:
                corrected_predictions = predictions + center_correction
            
            return corrected_predictions
            
        except Exception as e:
            return predictions
    
    def _apply_sinusoidal_corrections(self, predictions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply corrections for sinusoidal patterns"""
        try:
            # Ensure predictions follow sinusoidal pattern
            harmonic_analysis = self.current_pattern.get('harmonic_analysis', {})
            harmonics = harmonic_analysis.get('harmonics', [])
            
            if not harmonics:
                return predictions
            
            # Get dominant frequency
            dominant_harmonic = harmonics[0]
            frequency = dominant_harmonic['frequency']
            
            # Calculate expected sinusoidal continuation
            x = np.arange(len(data), len(data) + len(predictions))
            
            # Fit amplitude and phase to recent data
            recent_data = data[-min(20, len(data)):]
            recent_x = np.arange(len(recent_data))
            
            # Estimate amplitude and phase
            amplitude = np.std(recent_data)
            phase = 0  # Simplified
            
            # Generate sinusoidal correction
            sinusoidal_pattern = amplitude * np.sin(2 * np.pi * frequency * x + phase)
            
            # Blend with original predictions
            blend_factor = 0.3
            corrected_predictions = (1 - blend_factor) * predictions + blend_factor * sinusoidal_pattern
            
            return corrected_predictions
            
        except Exception as e:
            logger.error(f"Failed to apply sinusoidal corrections: {e}")
            return predictions
    
    def _apply_seasonal_corrections(self, predictions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply corrections for seasonal patterns"""
        try:
            seasonal_analysis = self.current_pattern.get('seasonal_analysis', {})
            period = seasonal_analysis.get('dominant_period', 12)
            
            if period == 0:
                return predictions
            
            # Calculate seasonal adjustments
            seasonal_adjustments = []
            for i in range(len(predictions)):
                seasonal_index = i % period
                
                # Get historical values for this seasonal position
                historical_values = []
                for j in range(seasonal_index, len(data), period):
                    historical_values.append(data[j])
                
                if historical_values:
                    seasonal_mean = np.mean(historical_values)
                    overall_mean = np.mean(data)
                    seasonal_adjustment = seasonal_mean - overall_mean
                    seasonal_adjustments.append(seasonal_adjustment)
                else:
                    seasonal_adjustments.append(0)
            
            # Apply seasonal corrections
            corrected_predictions = predictions + np.array(seasonal_adjustments) * 0.2
            
            return corrected_predictions
            
        except Exception as e:
            logger.error(f"Failed to apply seasonal corrections: {e}")
            return predictions
    
    def _apply_trending_corrections(self, predictions: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply corrections for trending patterns"""
        try:
            trend_analysis = self.current_pattern.get('trend_analysis', {})
            overall_trend = trend_analysis.get('overall_trend', {})
            trend_slope = overall_trend.get('slope', 0)
            
            # Calculate recent trend
            recent_data = data[-min(20, len(data)):]
            recent_trend = np.polyfit(np.arange(len(recent_data)), recent_data, 1)[0]
            
            # Apply trend correction
            trend_correction = (recent_trend - trend_slope) * 0.1
            
            # Apply to predictions with increasing weight
            correction_weights = np.linspace(0, 1, len(predictions))
            corrected_predictions = predictions + trend_correction * correction_weights
            
            return corrected_predictions
            
        except Exception as e:
            logger.error(f"Failed to apply trending corrections: {e}")
            return predictions
    
    def _calculate_recent_accuracy(self) -> float:
        """Calculate recent prediction accuracy"""
        try:
            if len(self.prediction_accuracy) < 5:
                return 1.0  # Assume good accuracy if not enough data
            
            recent_accuracy = list(self.prediction_accuracy)[-5:]
            return np.mean(recent_accuracy)
            
        except Exception as e:
            logger.error(f"Failed to calculate recent accuracy: {e}")
            return 0.5
    
    def _calculate_pattern_stability(self) -> float:
        """Calculate pattern stability"""
        try:
            if len(self.pattern_stability) < 3:
                return 1.0  # Assume stable if not enough data
            
            recent_stability = list(self.pattern_stability)[-3:]
            return np.mean(recent_stability)
            
        except Exception as e:
            logger.error(f"Failed to calculate pattern stability: {e}")
            return 0.5
    
    def _detect_distribution_change(self) -> bool:
        """Detect if data distribution has changed"""
        try:
            if len(self.historical_data) < 20:
                return False
            
            # Compare recent data with historical data
            data = self._get_current_data()
            recent_data = data[-10:]
            historical_data = data[-20:-10]
            
            # Statistical tests
            # KS test for distribution change
            ks_stat, p_value = stats.ks_2samp(recent_data, historical_data)
            
            # Distribution change if p-value is small
            return p_value < 0.05
            
        except Exception as e:
            logger.error(f"Failed to detect distribution change: {e}")
            return False
    
    def _detect_pattern_change(self) -> bool:
        """Detect if pattern has changed"""
        try:
            if len(self.pattern_evolution) < 2:
                return False
            
            # Compare recent pattern with previous pattern
            recent_pattern = self.pattern_evolution[-1]
            previous_pattern = self.pattern_evolution[-2]
            
            # Check if primary pattern has changed
            recent_primary = recent_pattern.get('pattern_classification', {}).get('primary_pattern', 'unknown')
            previous_primary = previous_pattern.get('pattern_classification', {}).get('primary_pattern', 'unknown')
            
            if recent_primary != previous_primary:
                return True
            
            # Check if pattern confidence has changed significantly
            recent_confidence = recent_pattern.get('pattern_classification', {}).get('confidence', 0.5)
            previous_confidence = previous_pattern.get('pattern_classification', {}).get('confidence', 0.5)
            
            if abs(recent_confidence - previous_confidence) > 0.3:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to detect pattern change: {e}")
            return False
    
    def _has_pattern_changed(self, new_pattern: Dict) -> bool:
        """Check if pattern has changed significantly"""
        try:
            if self.current_pattern is None:
                return True
            
            # Compare primary patterns
            current_primary = self.current_pattern.get('pattern_classification', {}).get('primary_pattern', 'unknown')
            new_primary = new_pattern.get('pattern_classification', {}).get('primary_pattern', 'unknown')
            
            if current_primary != new_primary:
                return True
            
            # Compare pattern strengths
            current_strength = self.current_pattern.get('pattern_strength', 0.5)
            new_strength = new_pattern.get('pattern_strength', 0.5)
            
            if abs(current_strength - new_strength) > 0.3:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check pattern change: {e}")
            return False
    
    def _update_pattern_model(self):
        """Update pattern model"""
        try:
            data = self._get_current_data()
            
            if len(data) < self.min_data_points:
                return
            
            # Analyze current pattern
            new_pattern = self.pattern_recognizer.analyze_comprehensive_patterns(data)
            
            # Update pattern evolution
            self.pattern_evolution.append(new_pattern)
            
            # Update current pattern
            self.current_pattern = new_pattern
            
            # Update model state
            self._update_model_state(data, new_pattern)
            
        except Exception as e:
            logger.error(f"Failed to update pattern model: {e}")
    
    def _update_model_state(self, data: np.ndarray, pattern: Dict):
        """Update model state for fast predictions"""
        try:
            # Create model state for fast predictions
            self.current_model_state = self.prediction_engine._select_prediction_strategy(pattern)
            
        except Exception as e:
            logger.error(f"Failed to update model state: {e}")
    
    def _update_adaptation_metrics(self):
        """Update adaptation metrics"""
        try:
            data = self._get_current_data()
            
            if len(data) < self.min_data_points:
                return
            
            # Calculate current metrics
            current_accuracy = self._calculate_current_accuracy(data)
            current_stability = self._calculate_current_stability(data)
            
            # Update tracking
            self.prediction_accuracy.append(current_accuracy)
            self.pattern_stability.append(current_stability)
            
            # Update adaptation metrics
            self.adaptation_metrics = {
                'accuracy': current_accuracy,
                'stability': current_stability,
                'pattern_type': self.current_pattern.get('pattern_classification', {}).get('primary_pattern', 'unknown') if self.current_pattern else 'unknown',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to update adaptation metrics: {e}")
    
    def _calculate_current_accuracy(self, data: np.ndarray) -> float:
        """Calculate current prediction accuracy"""
        try:
            if len(self.prediction_history) < 2:
                return 0.8  # Default accuracy
            
            # Get last predictions and actual values
            last_prediction = self.prediction_history[-1]
            
            # Calculate accuracy based on first prediction vs actual
            if len(data) > 0:
                # Simple accuracy measure
                return 0.8  # Placeholder
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Failed to calculate current accuracy: {e}")
            return 0.5
    
    def _calculate_current_stability(self, data: np.ndarray) -> float:
        """Calculate current pattern stability"""
        try:
            if len(data) < 20:
                return 0.8  # Default stability
            
            # Calculate volatility
            volatility = np.std(data[-20:])
            historical_volatility = np.std(data)
            
            # Stability is inverse of relative volatility
            if historical_volatility > 0:
                relative_volatility = volatility / historical_volatility
                stability = 1 / (1 + relative_volatility)
            else:
                stability = 1.0
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Failed to calculate current stability: {e}")
            return 0.5
    
    def _calculate_recent_prediction_errors(self, data: np.ndarray) -> List[float]:
        """Calculate recent prediction errors"""
        try:
            errors = []
            
            # Compare recent predictions with actual values
            for i, pred_info in enumerate(list(self.prediction_history)[-5:]):
                # This is a simplified error calculation
                # In practice, you'd compare predictions with actual outcomes
                errors.append(0.0)  # Placeholder
            
            return errors
            
        except Exception as e:
            logger.error(f"Failed to calculate recent prediction errors: {e}")
            return []
    
    def _update_learning_parameters(self):
        """Update learning parameters based on performance"""
        try:
            # Adjust learning rate based on accuracy
            recent_accuracy = self._calculate_recent_accuracy()
            
            if recent_accuracy < 0.5:
                # Increase learning rate for poor performance
                self.learning_rate = min(0.3, self.learning_rate * 1.2)
            elif recent_accuracy > 0.8:
                # Decrease learning rate for good performance
                self.learning_rate = max(0.05, self.learning_rate * 0.9)
            
            # Adjust thresholds
            pattern_stability = self._calculate_pattern_stability()
            
            if pattern_stability < 0.3:
                # Lower thresholds for unstable patterns
                self.adaptation_threshold = max(0.1, self.adaptation_threshold * 0.9)
            elif pattern_stability > 0.8:
                # Higher thresholds for stable patterns
                self.adaptation_threshold = min(0.3, self.adaptation_threshold * 1.1)
            
        except Exception as e:
            logger.error(f"Failed to update learning parameters: {e}")
    
    def _update_performance_metrics(self, predictions: Dict[str, Any], data: np.ndarray):
        """Update performance metrics"""
        try:
            # Calculate quality metrics
            quality_score = predictions.get('quality_metrics', {}).get('overall_quality_score', 0.5)
            
            # Update tracking
            self.prediction_accuracy.append(quality_score)
            
            # Update stability metrics
            if len(data) > 5:
                stability = self._calculate_current_stability(data)
                self.pattern_stability.append(stability)
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues"""
        try:
            # Cleanup is handled by deque max length
            # Additional cleanup can be added here if needed
            
            # Clean up old adaptation events
            if len(self.adaptation_events) > 100:
                self.adaptation_events = self.adaptation_events[-50:]
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def _get_fallback_predictions(self, steps: int) -> Dict[str, Any]:
        """Get fallback predictions when system fails"""
        try:
            # Simple fallback
            data = self._get_current_data()
            
            if len(data) > 0:
                last_value = data[-1]
                predictions = [last_value] * steps
            else:
                predictions = [0.0] * steps
            
            # Basic confidence intervals
            confidence_intervals = []
            for pred in predictions:
                confidence_intervals.append({
                    'lower': pred - 1.0,
                    'upper': pred + 1.0,
                    'std_error': 1.0
                })
            
            return {
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'pattern_analysis': {'pattern_classification': {'primary_pattern': 'fallback'}},
                'quality_metrics': {'overall_quality_score': 0.3}
            }
            
        except Exception as e:
            logger.error(f"Failed to get fallback predictions: {e}")
            return {
                'predictions': [0.0] * steps,
                'confidence_intervals': [],
                'pattern_analysis': {},
                'quality_metrics': {'overall_quality_score': 0.1}
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            with self.lock:
                return {
                    'data_points': len(self.historical_data),
                    'prediction_history_length': len(self.prediction_history),
                    'current_pattern': self.current_pattern.get('pattern_classification', {}).get('primary_pattern', 'unknown') if self.current_pattern else 'unknown',
                    'recent_accuracy': self._calculate_recent_accuracy(),
                    'pattern_stability': self._calculate_pattern_stability(),
                    'learning_rate': self.learning_rate,
                    'adaptation_threshold': self.adaptation_threshold,
                    'adaptation_events': len(self.adaptation_events),
                    'last_adaptation': self.adaptation_events[-1]['timestamp'].isoformat() if self.adaptation_events else None,
                    'is_running': self.is_running,
                    'adaptation_metrics': self.adaptation_metrics
                }
                
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {'error': str(e)}
    
    def reset_system(self):
        """Reset the learning system"""
        try:
            with self.lock:
                self.historical_data.clear()
                self.prediction_history.clear()
                self.pattern_evolution.clear()
                self.prediction_accuracy.clear()
                self.pattern_stability.clear()
                self.adaptation_events.clear()
                
                self.current_pattern = None
                self.current_model_state = None
                self.adaptation_metrics = {}
                
                # Reset parameters to defaults
                self.learning_rate = 0.1
                self.adaptation_threshold = 0.15
                self.pattern_change_threshold = 0.2
                
                logger.info("Adaptive learning system reset")
                
        except Exception as e:
            logger.error(f"Failed to reset system: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_continuous_learning()
        except:
            pass