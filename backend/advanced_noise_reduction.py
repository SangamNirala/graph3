"""
Advanced Noise Reduction System for Real-Time Prediction Smoothing
Provides comprehensive noise reduction and smoothing for continuous predictions
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import savgol_filter, butter, filtfilt, medfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging
from collections import deque

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdvancedNoiseReductionSystem:
    """
    Advanced noise reduction system specifically designed for real-time prediction smoothing
    """
    
    def __init__(self):
        # Smoothing parameters
        self.smoothing_params = {
            'savgol_window': 5,          # Savitzky-Golay filter window
            'savgol_polyorder': 2,       # Polynomial order for Savgol
            'gaussian_sigma': 1.0,       # Gaussian filter sigma
            'median_kernel': 3,          # Median filter kernel size
            'moving_avg_window': 3,      # Moving average window
            'butterworth_order': 2,      # Butterworth filter order
            'butterworth_cutoff': 0.3,   # Butterworth cutoff frequency
            'alpha_smoothing': 0.7,      # Exponential smoothing alpha
            'adaptive_threshold': 0.2    # Threshold for adaptive smoothing
        }
        
        # Noise detection parameters
        self.noise_detection = {
            'spike_threshold': 3.0,      # Standard deviations for spike detection
            'jitter_threshold': 0.1,     # Threshold for jitter detection
            'oscillation_threshold': 0.15, # Threshold for oscillation detection
            'variation_coefficient_limit': 0.3  # Limit for variation coefficient
        }
        
        # Prediction history for continuity
        self.prediction_history = deque(maxlen=100)
        self.smoothed_history = deque(maxlen=100)
        
        # Adaptive parameters
        self.adaptive_params = {
            'noise_level': 0.0,
            'smoothing_strength': 0.5,
            'pattern_preservation': 0.8,
            'continuity_weight': 0.7
        }
        
    def apply_comprehensive_smoothing(self, predictions: List[float], 
                                   historical_data: Optional[List[float]] = None,
                                   previous_predictions: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Apply comprehensive smoothing to predictions for noise reduction
        """
        try:
            predictions_array = np.array(predictions, dtype=float)
            
            if len(predictions_array) < 3:
                return {
                    'smoothed_predictions': predictions,
                    'smoothing_applied': ['insufficient_data'],
                    'noise_reduction_score': 0.0
                }
            
            # Step 1: Detect noise characteristics
            noise_analysis = self._analyze_noise_characteristics(predictions_array, historical_data)
            
            # Step 2: Apply adaptive smoothing based on noise analysis
            smoothed_predictions = self._apply_adaptive_smoothing(
                predictions_array, noise_analysis, historical_data, previous_predictions
            )
            
            # Step 3: Ensure continuity with previous predictions
            if previous_predictions:
                smoothed_predictions = self._ensure_prediction_continuity(
                    smoothed_predictions, previous_predictions
                )
            
            # Step 4: Apply final polish smoothing
            final_smoothed = self._apply_final_polish_smoothing(
                smoothed_predictions, noise_analysis
            )
            
            # Step 5: Validate smoothing quality
            quality_metrics = self._validate_smoothing_quality(
                predictions_array, final_smoothed, historical_data
            )
            
            # Update history
            self.prediction_history.extend(predictions)
            self.smoothed_history.extend(final_smoothed)
            
            # Update adaptive parameters based on performance
            self._update_adaptive_parameters(quality_metrics)
            
            return {
                'smoothed_predictions': final_smoothed.tolist(),
                'smoothing_applied': noise_analysis['smoothing_methods_applied'],
                'noise_reduction_score': quality_metrics['noise_reduction_score'],
                'quality_metrics': quality_metrics,
                'noise_analysis': noise_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive smoothing: {e}")
            return {
                'smoothed_predictions': predictions,
                'smoothing_applied': ['error'],
                'noise_reduction_score': 0.0
            }
    
    def _analyze_noise_characteristics(self, predictions: np.ndarray, 
                                     historical_data: Optional[List[float]]) -> Dict[str, Any]:
        """
        Analyze noise characteristics in predictions
        """
        try:
            analysis = {
                'noise_level': 'low',
                'dominant_noise_type': 'none',
                'smoothing_methods_applied': [],
                'noise_metrics': {}
            }
            
            # Calculate noise metrics
            if len(predictions) >= 3:
                # Variation coefficient
                mean_val = np.mean(predictions)
                std_val = np.std(predictions)
                variation_coef = std_val / (abs(mean_val) + 1e-8)
                
                # Jitter detection (high-frequency changes)
                first_diff = np.diff(predictions)
                second_diff = np.diff(first_diff)
                jitter_level = np.std(second_diff) / (std_val + 1e-8)
                
                # Spike detection
                z_scores = np.abs((predictions - mean_val) / (std_val + 1e-8))
                spike_count = np.sum(z_scores > self.noise_detection['spike_threshold'])
                
                # Oscillation detection
                sign_changes = np.sum(np.diff(np.sign(first_diff)) != 0)
                oscillation_rate = sign_changes / len(predictions) if len(predictions) > 1 else 0
                
                analysis['noise_metrics'] = {
                    'variation_coefficient': float(variation_coef),
                    'jitter_level': float(jitter_level),
                    'spike_count': int(spike_count),
                    'oscillation_rate': float(oscillation_rate),
                    'std_dev': float(std_val)
                }
                
                # Determine noise level and type
                if (variation_coef > self.noise_detection['variation_coefficient_limit'] or
                    jitter_level > self.noise_detection['jitter_threshold'] or
                    spike_count > 0 or
                    oscillation_rate > self.noise_detection['oscillation_threshold']):
                    
                    if spike_count > 0:
                        analysis['noise_level'] = 'high'
                        analysis['dominant_noise_type'] = 'spikes'
                        analysis['smoothing_methods_applied'].extend(['spike_removal', 'median_filter'])
                    elif jitter_level > self.noise_detection['jitter_threshold']:
                        analysis['noise_level'] = 'medium'
                        analysis['dominant_noise_type'] = 'jitter'
                        analysis['smoothing_methods_applied'].extend(['savgol_filter', 'gaussian_smooth'])
                    elif oscillation_rate > self.noise_detection['oscillation_threshold']:
                        analysis['noise_level'] = 'medium'
                        analysis['dominant_noise_type'] = 'oscillation'
                        analysis['smoothing_methods_applied'].extend(['butterworth_filter', 'moving_average'])
                    else:
                        analysis['noise_level'] = 'low'
                        analysis['dominant_noise_type'] = 'general'
                        analysis['smoothing_methods_applied'].append('light_smoothing')
                else:
                    analysis['noise_level'] = 'low'
                    analysis['smoothing_methods_applied'].append('minimal_smoothing')
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in noise analysis: {e}")
            return {
                'noise_level': 'unknown',
                'dominant_noise_type': 'unknown',
                'smoothing_methods_applied': ['fallback_smoothing'],
                'noise_metrics': {}
            }
    
    def _apply_adaptive_smoothing(self, predictions: np.ndarray, noise_analysis: Dict,
                                historical_data: Optional[List[float]],
                                previous_predictions: Optional[List[float]]) -> np.ndarray:
        """
        Apply adaptive smoothing based on noise analysis
        """
        try:
            smoothed = predictions.copy()
            noise_level = noise_analysis['noise_level']
            noise_type = noise_analysis['dominant_noise_type']
            
            # Apply different smoothing strategies based on noise characteristics
            if noise_type == 'spikes':
                # Use median filter for spike removal
                smoothed = self._apply_spike_removal(smoothed)
                # Follow up with gentle smoothing
                smoothed = self._apply_savgol_smoothing(smoothed, window_size=3)
                
            elif noise_type == 'jitter':
                # Use Savitzky-Golay filter for jitter reduction
                smoothed = self._apply_savgol_smoothing(smoothed)
                # Add Gaussian smoothing for extra smoothness
                smoothed = self._apply_gaussian_smoothing(smoothed, sigma=0.8)
                
            elif noise_type == 'oscillation':
                # Use Butterworth low-pass filter for oscillation removal
                smoothed = self._apply_butterworth_smoothing(smoothed)
                # Add moving average for stability
                smoothed = self._apply_moving_average_smoothing(smoothed)
                
            elif noise_level == 'low':
                # Light smoothing to maintain pattern integrity
                smoothed = self._apply_light_smoothing(smoothed)
                
            else:
                # General purpose smoothing
                smoothed = self._apply_general_smoothing(smoothed)
            
            # Apply exponential smoothing for temporal consistency
            if len(self.smoothed_history) > 0:
                smoothed = self._apply_exponential_smoothing(smoothed)
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error in adaptive smoothing: {e}")
            return predictions
    
    def _apply_spike_removal(self, predictions: np.ndarray) -> np.ndarray:
        """Remove spikes using median filtering"""
        try:
            kernel_size = min(self.smoothing_params['median_kernel'], len(predictions))
            if kernel_size >= 3 and kernel_size % 2 == 1:
                return medfilt(predictions, kernel_size=kernel_size)
            else:
                return predictions
        except Exception as e:
            logger.error(f"Error in spike removal: {e}")
            return predictions
    
    def _apply_savgol_smoothing(self, predictions: np.ndarray, window_size: Optional[int] = None) -> np.ndarray:
        """Apply Savitzky-Golay smoothing"""
        try:
            window = window_size or self.smoothing_params['savgol_window']
            window = min(window, len(predictions))
            
            if window >= 3 and window % 2 == 1:
                polyorder = min(self.smoothing_params['savgol_polyorder'], window - 1)
                return savgol_filter(predictions, window, polyorder)
            else:
                return predictions
        except Exception as e:
            logger.error(f"Error in Savgol smoothing: {e}")
            return predictions
    
    def _apply_gaussian_smoothing(self, predictions: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """Apply Gaussian smoothing"""
        try:
            sigma_val = sigma or self.smoothing_params['gaussian_sigma']
            return gaussian_filter1d(predictions, sigma=sigma_val)
        except Exception as e:
            logger.error(f"Error in Gaussian smoothing: {e}")
            return predictions
    
    def _apply_butterworth_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply Butterworth low-pass filter"""
        try:
            if len(predictions) < 6:
                return predictions
                
            order = self.smoothing_params['butterworth_order']
            cutoff = self.smoothing_params['butterworth_cutoff']
            
            # Design Butterworth filter
            b, a = butter(order, cutoff, btype='low')
            
            # Apply zero-phase filtering
            return filtfilt(b, a, predictions)
            
        except Exception as e:
            logger.error(f"Error in Butterworth smoothing: {e}")
            return predictions
    
    def _apply_moving_average_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing"""
        try:
            window = min(self.smoothing_params['moving_avg_window'], len(predictions))
            if window < 2:
                return predictions
                
            smoothed = np.convolve(predictions, np.ones(window)/window, mode='same')
            
            # Fix edge effects
            for i in range(window//2):
                smoothed[i] = np.mean(predictions[:i+window//2+1])
                smoothed[-(i+1)] = np.mean(predictions[-(i+window//2+1):])
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error in moving average smoothing: {e}")
            return predictions
    
    def _apply_light_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply light smoothing for low-noise data"""
        try:
            # Very gentle Gaussian smoothing
            return gaussian_filter1d(predictions, sigma=0.5)
        except Exception as e:
            logger.error(f"Error in light smoothing: {e}")
            return predictions
    
    def _apply_general_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply general-purpose smoothing"""
        try:
            # Combination of light Savgol and Gaussian
            smoothed = self._apply_savgol_smoothing(predictions, window_size=3)
            smoothed = self._apply_gaussian_smoothing(smoothed, sigma=0.7)
            return smoothed
        except Exception as e:
            logger.error(f"Error in general smoothing: {e}")
            return predictions
    
    def _apply_exponential_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing for temporal consistency"""
        try:
            alpha = self.smoothing_params['alpha_smoothing']
            
            if len(self.smoothed_history) == 0:
                return predictions
            
            # Get the last smoothed value for continuity
            last_smoothed = self.smoothed_history[-1]
            
            # Apply exponential smoothing to the first prediction for smooth transition
            smoothed = predictions.copy()
            smoothed[0] = alpha * predictions[0] + (1 - alpha) * last_smoothed
            
            # Apply mild exponential smoothing throughout
            for i in range(1, len(smoothed)):
                smoothed[i] = alpha * predictions[i] + (1 - alpha) * smoothed[i-1]
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error in exponential smoothing: {e}")
            return predictions
    
    def _ensure_prediction_continuity(self, predictions: np.ndarray, 
                                    previous_predictions: List[float]) -> np.ndarray:
        """Ensure smooth continuity with previous predictions"""
        try:
            if not previous_predictions or len(previous_predictions) == 0:
                return predictions
            
            # Get the last value from previous predictions
            last_previous = previous_predictions[-1]
            
            # Calculate the gap between last previous and first current prediction
            continuity_gap = predictions[0] - last_previous
            
            # Apply continuity correction if gap is significant
            continuity_threshold = self.adaptive_params['continuity_weight']
            
            if abs(continuity_gap) > continuity_threshold * np.std(predictions):
                # Apply smooth transition by reducing the gap gradually
                correction_weights = np.exp(-0.3 * np.arange(len(predictions)))
                predictions = predictions - continuity_gap * correction_weights * 0.7
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in continuity ensuring: {e}")
            return predictions
    
    def _apply_final_polish_smoothing(self, predictions: np.ndarray, 
                                    noise_analysis: Dict) -> np.ndarray:
        """Apply final polish smoothing for extra smoothness"""
        try:
            # Very light final smoothing to ensure smoothness
            if noise_analysis['noise_level'] in ['medium', 'high']:
                # Apply light Gaussian smoothing as final polish
                return gaussian_filter1d(predictions, sigma=0.3)
            else:
                # Minimal smoothing for low-noise data
                return predictions
                
        except Exception as e:
            logger.error(f"Error in final polish smoothing: {e}")
            return predictions
    
    def _validate_smoothing_quality(self, original: np.ndarray, smoothed: np.ndarray,
                                   historical_data: Optional[List[float]]) -> Dict[str, float]:
        """Validate the quality of smoothing"""
        try:
            metrics = {}
            
            # Noise reduction score
            original_noise = np.std(np.diff(np.diff(original))) if len(original) >= 3 else 0
            smoothed_noise = np.std(np.diff(np.diff(smoothed))) if len(smoothed) >= 3 else 0
            
            if original_noise > 0:
                noise_reduction = max(0, 1 - smoothed_noise / original_noise)
            else:
                noise_reduction = 1.0
            
            metrics['noise_reduction_score'] = float(noise_reduction)
            
            # Smoothness score
            smoothness_score = self._calculate_smoothness_score(smoothed)
            metrics['smoothness_score'] = float(smoothness_score)
            
            # Pattern preservation score
            if historical_data and len(historical_data) > 0:
                preservation_score = self._calculate_pattern_preservation_score(
                    smoothed, historical_data
                )
                metrics['pattern_preservation_score'] = float(preservation_score)
            else:
                metrics['pattern_preservation_score'] = 0.8
            
            # Overall quality score
            metrics['overall_quality'] = np.mean([
                metrics['noise_reduction_score'],
                metrics['smoothness_score'],
                metrics['pattern_preservation_score']
            ])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in quality validation: {e}")
            return {'noise_reduction_score': 0.5, 'smoothness_score': 0.5, 'overall_quality': 0.5}
    
    def _calculate_smoothness_score(self, predictions: np.ndarray) -> float:
        """Calculate smoothness score for predictions"""
        try:
            if len(predictions) < 3:
                return 1.0
            
            # Calculate second derivative (acceleration)
            second_derivative = np.diff(np.diff(predictions))
            
            # Smoothness is inversely related to the magnitude of second derivative
            smoothness_metric = np.std(second_derivative)
            
            # Normalize to 0-1 scale (higher is smoother)
            smoothness_score = 1.0 / (1.0 + smoothness_metric * 10)
            
            return min(1.0, max(0.0, smoothness_score))
            
        except Exception as e:
            logger.error(f"Error calculating smoothness score: {e}")
            return 0.5
    
    def _calculate_pattern_preservation_score(self, smoothed: np.ndarray, 
                                            historical_data: List[float]) -> float:
        """Calculate how well patterns are preserved after smoothing"""
        try:
            historical_array = np.array(historical_data)
            
            if len(historical_array) < 2 or len(smoothed) < 2:
                return 0.8
            
            # Calculate trend similarity
            hist_trend = np.mean(np.diff(historical_array[-10:]))  # Recent trend
            smooth_trend = np.mean(np.diff(smoothed))
            
            trend_similarity = 1.0 - min(1.0, abs(hist_trend - smooth_trend) / (abs(hist_trend) + 1e-8))
            
            # Calculate variability similarity
            hist_std = np.std(historical_array)
            smooth_std = np.std(smoothed)
            
            variability_similarity = 1.0 - min(1.0, abs(hist_std - smooth_std) / (hist_std + 1e-8))
            
            # Combined pattern preservation score
            pattern_preservation = (trend_similarity + variability_similarity) / 2
            
            return min(1.0, max(0.0, pattern_preservation))
            
        except Exception as e:
            logger.error(f"Error calculating pattern preservation score: {e}")
            return 0.8
    
    def _update_adaptive_parameters(self, quality_metrics: Dict[str, float]):
        """Update adaptive parameters based on performance"""
        try:
            # Update smoothing strength based on quality
            overall_quality = quality_metrics.get('overall_quality', 0.5)
            
            if overall_quality < 0.6:
                # Increase smoothing strength
                self.adaptive_params['smoothing_strength'] = min(0.9, 
                    self.adaptive_params['smoothing_strength'] + 0.1)
            elif overall_quality > 0.8:
                # Decrease smoothing strength to preserve more detail
                self.adaptive_params['smoothing_strength'] = max(0.2, 
                    self.adaptive_params['smoothing_strength'] - 0.1)
            
            # Update pattern preservation weight
            pattern_score = quality_metrics.get('pattern_preservation_score', 0.8)
            if pattern_score < 0.7:
                self.adaptive_params['pattern_preservation'] = min(0.95,
                    self.adaptive_params['pattern_preservation'] + 0.05)
            
        except Exception as e:
            logger.error(f"Error updating adaptive parameters: {e}")
    
    def apply_real_time_smoothing(self, new_predictions: List[float],
                                 context_window: int = 10) -> List[float]:
        """
        Apply real-time smoothing optimized for continuous updates
        """
        try:
            if len(new_predictions) == 0:
                return new_predictions
            
            predictions_array = np.array(new_predictions, dtype=float)
            
            # Get context from history if available
            context_data = []
            if len(self.smoothed_history) >= context_window:
                context_data = list(self.smoothed_history)[-context_window:]
            
            # Apply lightweight smoothing optimized for real-time updates
            if len(context_data) > 0:
                # Combine context with new predictions for continuity
                combined = np.array(context_data + new_predictions.tolist())
                
                # Apply light Gaussian smoothing
                smoothed_combined = gaussian_filter1d(combined, sigma=0.6)
                
                # Extract the new predictions part
                smoothed_new = smoothed_combined[len(context_data):]
                
                # Ensure smooth transition
                if len(context_data) > 0:
                    transition_weight = 0.7
                    smoothed_new[0] = (transition_weight * smoothed_new[0] + 
                                     (1 - transition_weight) * context_data[-1])
            else:
                # No context available, apply basic smoothing
                if len(predictions_array) >= 3:
                    smoothed_new = gaussian_filter1d(predictions_array, sigma=0.5)
                else:
                    smoothed_new = predictions_array
            
            # Update history
            self.smoothed_history.extend(smoothed_new.tolist())
            
            return smoothed_new.tolist()
            
        except Exception as e:
            logger.error(f"Error in real-time smoothing: {e}")
            return new_predictions