"""
Universal Waveform Learning System
Advanced system capable of learning and reproducing ANY pattern complexity and shape
"""

import numpy as np
import pandas as pd
from scipy import signal, stats, optimize, interpolate
from scipy.signal import savgol_filter, butter, sosfilt, find_peaks, peak_widths
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
from dataclasses import dataclass
import math

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class WaveformPattern:
    """Represents a learned waveform pattern with all its characteristics"""
    pattern_id: str
    pattern_type: str
    shape_characteristics: Dict[str, Any]
    geometric_properties: Dict[str, Any]
    transition_properties: Dict[str, Any]
    amplitude_properties: Dict[str, Any]
    frequency_properties: Dict[str, Any]
    template: np.ndarray
    confidence: float
    complexity_score: float
    learning_quality: float
    metadata: Dict[str, Any]

class UniversalWaveformLearningSystem:
    """
    Universal system for learning and reproducing any waveform pattern complexity
    """
    
    def __init__(self):
        # Pattern detection capabilities
        self.shape_detectors = {
            'square_wave': self._detect_square_wave_pattern,
            'triangular_wave': self._detect_triangular_wave_pattern,
            'sawtooth_wave': self._detect_sawtooth_wave_pattern,
            'step_function': self._detect_step_function_pattern,
            'pulse_pattern': self._detect_pulse_pattern,
            'exponential_decay': self._detect_exponential_pattern,
            'logarithmic_pattern': self._detect_logarithmic_pattern,
            'sinusoidal_pattern': self._detect_sinusoidal_pattern,
            'polynomial_pattern': self._detect_polynomial_pattern,
            'spline_pattern': self._detect_spline_pattern,
            'fractal_pattern': self._detect_fractal_pattern,
            'chaotic_pattern': self._detect_chaotic_pattern,
            'composite_pattern': self._detect_composite_pattern,
            'irregular_pattern': self._detect_irregular_pattern,
            'custom_shape': self._detect_custom_shape_pattern
        }
        
        # Geometric analyzers for shape characteristics (simplified for now)
        self.geometric_analyzers = {
            # Only include implemented methods
        }
        
        # Pattern learning strategies (simplified for now)
        self.learning_strategies = {
            # Only include implemented methods
        }
        
        # Pattern synthesis methods for predictions (only implemented ones)
        self.synthesis_methods = {
            'geometric_synthesis': self._geometric_pattern_synthesis,
        }
        
        # Memory systems for learned patterns
        self.pattern_library = {}
        self.template_bank = deque(maxlen=10000)
        self.shape_memory = defaultdict(list)
        self.complexity_hierarchy = {}
        self.adaptation_history = deque(maxlen=1000)
        self.pattern_performance = {}  # Track performance of different pattern types
        
        # Learning parameters
        self.learning_params = {
            'pattern_similarity_threshold': 0.75,
            'shape_detection_sensitivity': 0.8,
            'edge_detection_threshold': 0.1,
            'segment_detection_threshold': 0.05,
            'pattern_complexity_threshold': 0.7,
            'template_matching_tolerance': 0.85,
            'adaptation_learning_rate': 0.12,
            'pattern_fusion_strength': 0.9,
            'geometric_preservation_factor': 0.95,
            'amplitude_preservation_factor': 0.90
        }
        
        # Quality assessment parameters
        self.quality_params = {
            'shape_fidelity_weight': 0.35,
            'amplitude_accuracy_weight': 0.25,
            'frequency_accuracy_weight': 0.20,
            'transition_quality_weight': 0.15,
            'overall_continuity_weight': 0.05
        }
    
    def learn_universal_patterns(self, data: np.ndarray, 
                               timestamps: Optional[np.ndarray] = None,
                               pattern_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Learn any pattern from input data with universal adaptability
        """
        try:
            logger.info(f"Learning universal patterns from {len(data)} data points")
            
            if len(data) < 3:
                return self._create_minimal_learning_result(data)
            
            # Comprehensive pattern detection
            detected_patterns = self._detect_all_pattern_types(data)
            
            # Geometric analysis of shapes
            geometric_analysis = self._comprehensive_geometric_analysis(data)
            
            # Pattern learning using multiple strategies
            learned_patterns = self._apply_all_learning_strategies(data, detected_patterns, geometric_analysis)
            
            # Pattern synthesis capabilities assessment
            synthesis_capabilities = self._assess_synthesis_capabilities(learned_patterns)
            
            # Quality assessment of learned patterns
            learning_quality = self._assess_universal_learning_quality(
                data, learned_patterns, geometric_analysis
            )
            
            # Update pattern library with new learnings
            self._update_universal_pattern_library(learned_patterns, learning_quality)
            
            # Create comprehensive learning result
            learning_result = {
                'status': 'success',
                'data_characteristics': self._analyze_data_characteristics(data),
                'detected_patterns': detected_patterns,
                'geometric_analysis': geometric_analysis,
                'learned_patterns': learned_patterns,
                'synthesis_capabilities': synthesis_capabilities,
                'learning_quality': learning_quality,
                'pattern_library_stats': {
                    'total_patterns': len(self.pattern_library),
                    'template_bank_size': len(self.template_bank),
                    'shape_memory_categories': len(self.shape_memory)
                },
                'prediction_readiness': learning_quality.get('overall_quality', 0.5),
                'universal_adaptability_score': learning_quality.get('adaptability_score', 0.5),
                'pattern_complexity_handled': learning_quality.get('complexity_score', 0.5)
            }
            
            return learning_result
            
        except Exception as e:
            logger.error(f"Error in universal pattern learning: {e}")
            return self._create_error_learning_result(str(e))
    
    def generate_waveform_aware_predictions(self, data: np.ndarray, 
                                          steps: int = 30,
                                          previous_predictions: Optional[List] = None,
                                          pattern_preferences: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate predictions that preserve learned waveform characteristics
        """
        try:
            logger.info(f"Generating {steps} waveform-aware predictions")
            
            # Analyze current pattern state
            current_pattern_state = self._analyze_current_pattern_state(data)
            
            # Select optimal synthesis method based on learned patterns
            optimal_synthesis = self._select_optimal_synthesis_method(
                current_pattern_state, pattern_preferences
            )
            
            # Generate pattern-preserving predictions
            base_predictions = self._generate_pattern_preserving_predictions(
                data, steps, current_pattern_state, optimal_synthesis
            )
            
            # Apply waveform shape corrections
            shape_corrected = self._apply_waveform_shape_corrections(
                base_predictions, data, current_pattern_state
            )
            
            # Apply geometric consistency corrections
            geometry_corrected = self._apply_geometric_consistency_corrections(
                shape_corrected, data, current_pattern_state
            )
            
            # Apply amplitude and frequency preservation
            amplitude_corrected = self._apply_amplitude_frequency_preservation(
                geometry_corrected, data, current_pattern_state
            )
            
            # Apply continuity and smoothness corrections
            final_predictions = self._apply_continuity_smoothness_corrections(
                amplitude_corrected, data, previous_predictions, current_pattern_state
            )
            
            # Calculate pattern-aware confidence intervals
            confidence_intervals = self._calculate_pattern_aware_confidence_intervals(
                final_predictions, data, current_pattern_state
            )
            
            # Quality assessment of predictions
            prediction_quality = self._assess_waveform_prediction_quality(
                final_predictions, data, current_pattern_state
            )
            
            # Update learning from prediction results
            self._update_learning_from_predictions(
                final_predictions, data, current_pattern_state, prediction_quality
            )
            
            return {
                'predictions': final_predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'pattern_state': current_pattern_state,
                'synthesis_method': optimal_synthesis,
                'prediction_quality': prediction_quality,
                'waveform_characteristics_preserved': {
                    'shape_fidelity': prediction_quality.get('shape_fidelity', 0.5),
                    'amplitude_accuracy': prediction_quality.get('amplitude_accuracy', 0.5),
                    'frequency_accuracy': prediction_quality.get('frequency_accuracy', 0.5),
                    'geometric_consistency': prediction_quality.get('geometric_consistency', 0.5),
                    'transition_quality': prediction_quality.get('transition_quality', 0.5)
                },
                'adaptability_metrics': {
                    'pattern_adaptation_score': prediction_quality.get('adaptation_score', 0.5),
                    'complexity_handling_score': prediction_quality.get('complexity_score', 0.5),
                    'universal_learning_score': prediction_quality.get('universal_score', 0.5)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating waveform-aware predictions: {e}")
            return self._generate_fallback_waveform_predictions(data, steps)
    
    def _detect_all_pattern_types(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect all possible pattern types in the data"""
        try:
            detected_patterns = {}
            
            # Run all shape detectors
            for pattern_type, detector in self.shape_detectors.items():
                try:
                    pattern_result = detector(data)
                    if pattern_result and pattern_result.get('confidence', 0) > 0.1:
                        detected_patterns[pattern_type] = pattern_result
                except Exception as e:
                    logger.warning(f"Pattern detector {pattern_type} failed: {e}")
                    continue
            
            # Rank patterns by confidence and strength
            ranked_patterns = sorted(
                detected_patterns.items(),
                key=lambda x: x[1].get('confidence', 0) * x[1].get('strength', 0),
                reverse=True
            )
            
            return {
                'all_patterns': detected_patterns,
                'ranked_patterns': ranked_patterns,
                'dominant_pattern': ranked_patterns[0] if ranked_patterns else None,
                'pattern_count': len(detected_patterns),
                'complexity_level': self._assess_pattern_complexity_level(detected_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error detecting all pattern types: {e}")
            return {'all_patterns': {}, 'pattern_count': 0}
    
    def _detect_square_wave_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect square wave patterns with sharp transitions and flat segments"""
        try:
            # Detect flat segments (plateaus)
            plateaus = self._find_plateaus(data)
            
            # Detect sharp transitions between plateaus
            transitions = self._find_sharp_transitions(data)
            
            if len(plateaus) < 2 or len(transitions) < 1:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Analyze square wave characteristics
            plateau_analysis = self._analyze_plateau_characteristics(plateaus, data)
            transition_analysis = self._analyze_transition_characteristics(transitions, data)
            
            # Calculate square wave score
            plateau_quality = plateau_analysis.get('flatness_score', 0.0)
            transition_quality = transition_analysis.get('sharpness_score', 0.0)
            periodicity_score = self._calculate_square_wave_periodicity(plateaus, data)
            
            square_wave_score = (plateau_quality * 0.4 + 
                                transition_quality * 0.4 + 
                                periodicity_score * 0.2)
            
            return {
                'confidence': float(square_wave_score),
                'strength': float(square_wave_score),
                'pattern_type': 'square_wave',
                'plateaus': plateaus,
                'transitions': transitions,
                'plateau_analysis': plateau_analysis,
                'transition_analysis': transition_analysis,
                'periodicity_score': float(periodicity_score),
                'amplitude_levels': self._extract_amplitude_levels(plateaus, data),
                'duty_cycle': self._calculate_duty_cycle(plateaus),
                'template': self._create_square_wave_template(plateaus, transitions, data)
            }
            
        except Exception as e:
            logger.error(f"Error detecting square wave pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_triangular_wave_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect triangular wave patterns with linear segments and sharp peaks"""
        try:
            # Find peaks and valleys
            peaks, _ = find_peaks(data, prominence=np.std(data) * 0.5)
            valleys, _ = find_peaks(-data, prominence=np.std(data) * 0.5)
            
            if len(peaks) < 1 and len(valleys) < 1:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Analyze linear segments between peaks/valleys
            linear_segments = self._find_linear_segments(data, peaks, valleys)
            
            # Calculate triangular characteristics
            linearity_score = self._calculate_linearity_score(linear_segments, data)
            peak_sharpness = self._calculate_peak_sharpness(peaks, valleys, data)
            symmetry_score = self._calculate_triangular_symmetry(peaks, valleys, data)
            periodicity_score = self._calculate_triangular_periodicity(peaks, valleys)
            
            triangular_score = (linearity_score * 0.4 + 
                              peak_sharpness * 0.3 + 
                              symmetry_score * 0.2 + 
                              periodicity_score * 0.1)
            
            return {
                'confidence': float(triangular_score),
                'strength': float(triangular_score),
                'pattern_type': 'triangular_wave',
                'peaks': peaks.tolist() if len(peaks) > 0 else [],
                'valleys': valleys.tolist() if len(valleys) > 0 else [],
                'linear_segments': linear_segments,
                'linearity_score': float(linearity_score),
                'peak_sharpness': float(peak_sharpness),
                'symmetry_score': float(symmetry_score),
                'periodicity_score': float(periodicity_score),
                'slope_analysis': self._analyze_triangular_slopes(linear_segments, data),
                'template': self._create_triangular_wave_template(peaks, valleys, linear_segments, data)
            }
            
        except Exception as e:
            logger.error(f"Error detecting triangular wave pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_sawtooth_wave_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect sawtooth wave patterns with linear ramps and sharp drops/rises"""
        try:
            # Find sharp transitions (drops or rises)
            sharp_transitions = self._find_sharp_sawtooth_transitions(data)
            
            if len(sharp_transitions) < 1:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Analyze linear ramps between transitions
            linear_ramps = self._find_linear_ramps(data, sharp_transitions)
            
            # Calculate sawtooth characteristics
            ramp_linearity = self._calculate_ramp_linearity(linear_ramps, data)
            transition_sharpness = self._calculate_sawtooth_transition_sharpness(sharp_transitions, data)
            periodicity_score = self._calculate_sawtooth_periodicity(sharp_transitions)
            
            sawtooth_score = (ramp_linearity * 0.5 + 
                             transition_sharpness * 0.3 + 
                             periodicity_score * 0.2)
            
            return {
                'confidence': float(sawtooth_score),
                'strength': float(sawtooth_score),
                'pattern_type': 'sawtooth_wave',
                'sharp_transitions': sharp_transitions,
                'linear_ramps': linear_ramps,
                'ramp_linearity': float(ramp_linearity),
                'transition_sharpness': float(transition_sharpness),
                'periodicity_score': float(periodicity_score),
                'ramp_direction': self._determine_sawtooth_direction(linear_ramps, data),
                'template': self._create_sawtooth_wave_template(sharp_transitions, linear_ramps, data)
            }
            
        except Exception as e:
            logger.error(f"Error detecting sawtooth wave pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_step_function_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect step function patterns with discrete levels and sharp transitions"""
        try:
            # Find step transitions
            step_transitions = self._find_step_transitions(data)
            
            if len(step_transitions) < 1:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Identify discrete levels
            discrete_levels = self._identify_discrete_levels(data, step_transitions)
            
            # Analyze step characteristics
            level_stability = self._calculate_level_stability(discrete_levels, data)
            transition_quality = self._calculate_step_transition_quality(step_transitions, data)
            discreteness_score = self._calculate_discreteness_score(discrete_levels, data)
            
            step_score = (level_stability * 0.4 + 
                         transition_quality * 0.4 + 
                         discreteness_score * 0.2)
            
            return {
                'confidence': float(step_score),
                'strength': float(step_score),
                'pattern_type': 'step_function',
                'step_transitions': step_transitions,
                'discrete_levels': discrete_levels,
                'level_stability': float(level_stability),
                'transition_quality': float(transition_quality),
                'discreteness_score': float(discreteness_score),
                'num_levels': len(discrete_levels),
                'level_spacing': self._calculate_level_spacing(discrete_levels),
                'template': self._create_step_function_template(step_transitions, discrete_levels, data)
            }
            
        except Exception as e:
            logger.error(f"Error detecting step function pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_pulse_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect pulse patterns with sharp peaks and baseline returns"""
        try:
            # Find pulses (sharp peaks that return to baseline)
            pulses = self._find_pulse_events(data)
            
            if len(pulses) < 1:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Analyze pulse characteristics
            pulse_analysis = self._analyze_pulse_characteristics(pulses, data)
            
            # Calculate pulse pattern score
            pulse_sharpness = pulse_analysis.get('pulse_sharpness', 0.0)
            baseline_return = pulse_analysis.get('baseline_return_quality', 0.0)
            pulse_consistency = pulse_analysis.get('pulse_consistency', 0.0)
            
            pulse_score = (pulse_sharpness * 0.4 + 
                          baseline_return * 0.4 + 
                          pulse_consistency * 0.2)
            
            return {
                'confidence': float(pulse_score),
                'strength': float(pulse_score),
                'pattern_type': 'pulse_pattern',
                'pulses': pulses,
                'pulse_analysis': pulse_analysis,
                'baseline_level': pulse_analysis.get('baseline_level', 0.0),
                'pulse_amplitude': pulse_analysis.get('average_pulse_amplitude', 0.0),
                'pulse_width': pulse_analysis.get('average_pulse_width', 0.0),
                'pulse_frequency': pulse_analysis.get('pulse_frequency', 0.0),
                'template': self._create_pulse_pattern_template(pulses, pulse_analysis, data)
            }
            
        except Exception as e:
            logger.error(f"Error detecting pulse pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_exponential_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect exponential growth/decay patterns"""
        try:
            # Try exponential fitting
            x = np.arange(len(data))
            
            # Test both positive and negative exponentials
            exponential_fits = []
            
            # Positive exponential: y = a * exp(b * x)
            try:
                # Transform to linear space: ln(y) = ln(a) + b * x
                positive_data = data[data > 0] if np.any(data > 0) else data + np.abs(np.min(data)) + 1e-6
                if len(positive_data) > len(data) * 0.8:  # Most data points are positive
                    log_data = np.log(positive_data[:len(data)])
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[:len(log_data)], log_data)
                    
                    exponential_fits.append({
                        'type': 'exponential_growth' if slope > 0 else 'exponential_decay',
                        'r_squared': r_value ** 2,
                        'slope': slope,
                        'intercept': intercept,
                        'parameters': {'a': np.exp(intercept), 'b': slope}
                    })
            except Exception as exp_error:
                logger.debug(f"Positive exponential fitting failed: {exp_error}")
            
            # Negative exponential: y = a * exp(-b * x)
            try:
                # For decay patterns
                if np.mean(data[:len(data)//4]) > np.mean(data[-len(data)//4:]):
                    normalized_data = data - np.min(data) + 1e-6
                    log_normalized = np.log(normalized_data)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_normalized)
                    
                    if slope < 0:  # Decay pattern
                        exponential_fits.append({
                            'type': 'exponential_decay',
                            'r_squared': r_value ** 2,
                            'slope': slope,
                            'intercept': intercept,
                            'parameters': {'a': np.exp(intercept), 'b': -slope}
                        })
            except Exception as decay_error:
                logger.debug(f"Exponential decay fitting failed: {decay_error}")
            
            if not exponential_fits:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Select best fit
            best_fit = max(exponential_fits, key=lambda x: x['r_squared'])
            
            return {
                'confidence': float(best_fit['r_squared']),
                'strength': float(best_fit['r_squared']),
                'pattern_type': 'exponential_pattern',
                'exponential_type': best_fit['type'],
                'parameters': best_fit['parameters'],
                'r_squared': float(best_fit['r_squared']),
                'growth_rate': float(best_fit['parameters']['b']),
                'initial_value': float(best_fit['parameters']['a']),
                'template': self._create_exponential_template(best_fit, len(data))
            }
            
        except Exception as e:
            logger.error(f"Error detecting exponential pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_logarithmic_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect logarithmic patterns"""
        try:
            x = np.arange(1, len(data) + 1)  # Avoid log(0)
            
            # Try logarithmic fitting: y = a * ln(x) + b
            try:
                log_x = np.log(x)
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, data)
                
                logarithmic_score = r_value ** 2
                
                return {
                    'confidence': float(logarithmic_score),
                    'strength': float(logarithmic_score),
                    'pattern_type': 'logarithmic_pattern',
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(logarithmic_score),
                    'parameters': {'a': slope, 'b': intercept},
                    'template': self._create_logarithmic_template(slope, intercept, len(data))
                }
                
            except Exception as log_error:
                logger.debug(f"Logarithmic fitting failed: {log_error}")
                return {'confidence': 0.0, 'strength': 0.0}
                
        except Exception as e:
            logger.error(f"Error detecting logarithmic pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_sinusoidal_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect sinusoidal patterns with enhanced accuracy"""
        try:
            # FFT analysis for frequency detection
            fft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data))
            power_spectrum = np.abs(fft) ** 2
            
            # Find dominant frequency
            positive_freqs = freqs[1:len(freqs)//2]
            positive_power = power_spectrum[1:len(power_spectrum)//2]
            
            if len(positive_power) > 0:
                dominant_freq_idx = np.argmax(positive_power)
                dominant_frequency = positive_freqs[dominant_freq_idx]
                
                # Try sinusoidal fitting: y = A * sin(2*pi*f*x + phi) + offset
                try:
                    def sine_func(x, A, freq, phi, offset):
                        return A * np.sin(2 * np.pi * freq * x + phi) + offset
                    
                    x = np.arange(len(data))
                    
                    # Initial parameter estimates
                    A_init = (np.max(data) - np.min(data)) / 2
                    offset_init = np.mean(data)
                    freq_init = dominant_frequency if dominant_frequency > 0 else 1.0 / len(data)
                    phi_init = 0.0
                    
                    popt, pcov = optimize.curve_fit(
                        sine_func, x, data,
                        p0=[A_init, freq_init, phi_init, offset_init],
                        maxfev=2000
                    )
                    
                    # Calculate fit quality
                    fitted_data = sine_func(x, *popt)
                    r_squared = 1 - np.sum((data - fitted_data) ** 2) / np.sum((data - np.mean(data)) ** 2)
                    
                    return {
                        'confidence': float(max(0, r_squared)),
                        'strength': float(max(0, r_squared)),
                        'pattern_type': 'sinusoidal_pattern',
                        'amplitude': float(popt[0]),
                        'frequency': float(popt[1]),
                        'phase': float(popt[2]),
                        'offset': float(popt[3]),
                        'r_squared': float(r_squared),
                        'period': float(1.0 / abs(popt[1])) if popt[1] != 0 else len(data),
                        'template': fitted_data.tolist()
                    }
                    
                except Exception as fit_error:
                    logger.debug(f"Sinusoidal fitting failed: {fit_error}")
                    return {'confidence': 0.0, 'strength': 0.0}
            else:
                return {'confidence': 0.0, 'strength': 0.0}
                
        except Exception as e:
            logger.error(f"Error detecting sinusoidal pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_polynomial_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect polynomial patterns of various degrees"""
        try:
            x = np.arange(len(data))
            
            polynomial_fits = {}
            
            # Try different polynomial degrees
            for degree in range(2, min(6, len(data) // 3)):
                try:
                    coeffs = np.polyfit(x, data, degree)
                    fitted_data = np.polyval(coeffs, x)
                    r_squared = 1 - np.sum((data - fitted_data) ** 2) / np.sum((data - np.mean(data)) ** 2)
                    
                    polynomial_fits[degree] = {
                        'coefficients': coeffs.tolist(),
                        'r_squared': float(r_squared),
                        'fitted_data': fitted_data.tolist()
                    }
                    
                except Exception as poly_error:
                    logger.debug(f"Polynomial degree {degree} fitting failed: {poly_error}")
                    continue
            
            if not polynomial_fits:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Select best polynomial fit
            best_degree = max(polynomial_fits.keys(), key=lambda d: polynomial_fits[d]['r_squared'])
            best_fit = polynomial_fits[best_degree]
            
            return {
                'confidence': float(best_fit['r_squared']),
                'strength': float(best_fit['r_squared']),
                'pattern_type': 'polynomial_pattern',
                'degree': int(best_degree),
                'coefficients': best_fit['coefficients'],
                'r_squared': best_fit['r_squared'],
                'all_fits': polynomial_fits,
                'template': best_fit['fitted_data']
            }
            
        except Exception as e:
            logger.error(f"Error detecting polynomial pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_spline_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect smooth spline patterns"""
        try:
            from scipy.interpolate import UnivariateSpline
            
            x = np.arange(len(data))
            
            # Try different smoothing factors
            smoothing_factors = [None, len(data), len(data) * 0.1, len(data) * 0.01]
            spline_fits = {}
            
            for s in smoothing_factors:
                try:
                    spline = UnivariateSpline(x, data, s=s, k=3)
                    fitted_data = spline(x)
                    
                    # Calculate fit quality
                    r_squared = 1 - np.sum((data - fitted_data) ** 2) / np.sum((data - np.mean(data)) ** 2)
                    
                    # Calculate smoothness (lower second derivative variance = smoother)
                    second_deriv = np.gradient(np.gradient(fitted_data))
                    smoothness_score = 1.0 / (1.0 + np.var(second_deriv))
                    
                    spline_fits[str(s)] = {
                        'r_squared': float(r_squared),
                        'smoothness_score': float(smoothness_score),
                        'fitted_data': fitted_data.tolist(),
                        'smoothing_factor': s
                    }
                    
                except Exception as spline_error:
                    logger.debug(f"Spline fitting with s={s} failed: {spline_error}")
                    continue
            
            if not spline_fits:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Select best spline based on balance of fit quality and smoothness
            best_spline_key = max(spline_fits.keys(), 
                                key=lambda k: spline_fits[k]['r_squared'] * 0.7 + 
                                              spline_fits[k]['smoothness_score'] * 0.3)
            best_spline = spline_fits[best_spline_key]
            
            overall_score = best_spline['r_squared'] * 0.7 + best_spline['smoothness_score'] * 0.3
            
            return {
                'confidence': float(overall_score),
                'strength': float(overall_score),
                'pattern_type': 'spline_pattern',
                'r_squared': best_spline['r_squared'],
                'smoothness_score': best_spline['smoothness_score'],
                'smoothing_factor': best_spline['smoothing_factor'],
                'all_fits': spline_fits,
                'template': best_spline['fitted_data']
            }
            
        except Exception as e:
            logger.error(f"Error detecting spline pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_fractal_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect fractal patterns using Hurst exponent and self-similarity"""
        try:
            # Calculate Hurst exponent
            hurst_exponent = self._calculate_hurst_exponent_advanced(data)
            
            # Calculate fractal dimension
            fractal_dimension = 2 - hurst_exponent
            
            # Test for self-similarity at different scales
            self_similarity = self._test_self_similarity(data)
            
            # Calculate fractal strength
            # Hurst != 0.5 indicates fractal behavior
            fractal_strength = abs(hurst_exponent - 0.5) * 2
            
            overall_fractal_score = (fractal_strength * 0.6 + self_similarity * 0.4)
            
            return {
                'confidence': float(overall_fractal_score),
                'strength': float(overall_fractal_score),
                'pattern_type': 'fractal_pattern',
                'hurst_exponent': float(hurst_exponent),
                'fractal_dimension': float(fractal_dimension),
                'self_similarity_score': float(self_similarity),
                'fractal_type': 'persistent' if hurst_exponent > 0.5 else 'anti_persistent',
                'template': self._create_fractal_template(data, hurst_exponent)
            }
            
        except Exception as e:
            logger.error(f"Error detecting fractal pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_chaotic_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect chaotic patterns using Lyapunov exponents and phase space analysis"""
        try:
            if len(data) < 20:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Calculate largest Lyapunov exponent (simplified)
            lyapunov_exponent = self._calculate_lyapunov_exponent_advanced(data)
            
            # Analyze phase space reconstruction
            phase_space_analysis = self._analyze_phase_space(data)
            
            # Calculate chaos indicators
            sensitivity = self._calculate_sensitivity_to_initial_conditions(data)
            bounded_behavior = self._check_bounded_behavior(data)
            
            # Chaos score
            chaos_score = 0.0
            if lyapunov_exponent > 0 and bounded_behavior:
                chaos_score = min(1.0, abs(lyapunov_exponent) * sensitivity)
            
            return {
                'confidence': float(chaos_score),
                'strength': float(chaos_score),
                'pattern_type': 'chaotic_pattern',
                'lyapunov_exponent': float(lyapunov_exponent),
                'phase_space_analysis': phase_space_analysis,
                'sensitivity_score': float(sensitivity),
                'is_bounded': bounded_behavior,
                'chaos_indicators': {
                    'positive_lyapunov': lyapunov_exponent > 0,
                    'bounded': bounded_behavior,
                    'sensitive': sensitivity > 0.5
                },
                'template': self._create_chaotic_template(data, phase_space_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error detecting chaotic pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_composite_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect composite patterns (combination of multiple simple patterns)"""
        try:
            # Decompose into components using EMD-like approach
            components = self._decompose_into_components(data)
            
            if len(components) < 2:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Analyze each component
            component_patterns = []
            for i, component in enumerate(components):
                # Run simplified pattern detection on each component
                component_pattern = self._detect_component_pattern(component)
                if component_pattern.get('confidence', 0) > 0.3:
                    component_patterns.append({
                        'component_index': i,
                        'pattern_info': component_pattern,
                        'component_data': component.tolist()
                    })
            
            if len(component_patterns) < 2:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Calculate composite score
            component_strengths = [cp['pattern_info'].get('confidence', 0) for cp in component_patterns]
            composite_score = np.mean(component_strengths) * (len(component_patterns) / len(components))
            
            return {
                'confidence': float(composite_score),
                'strength': float(composite_score),
                'pattern_type': 'composite_pattern',
                'num_components': len(components),
                'component_patterns': component_patterns,
                'dominant_components': sorted(component_patterns, 
                                            key=lambda x: x['pattern_info'].get('confidence', 0), 
                                            reverse=True)[:3],
                'template': self._create_composite_template(component_patterns, len(data))
            }
            
        except Exception as e:
            logger.error(f"Error detecting composite pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_irregular_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect irregular patterns that don't fit standard categories"""
        try:
            # Calculate irregularity metrics
            variability = np.std(np.diff(data)) / (np.mean(np.abs(data)) + 1e-8)
            local_variations = self._calculate_local_variations(data)
            consistency_score = 1.0 - self._calculate_pattern_consistency(data)
            
            # Irregularity indicators
            non_periodicity = 1.0 - self._test_periodicity_strength(data)
            non_linearity = 1.0 - self._test_linearity_strength(data)
            
            irregularity_score = np.mean([variability, local_variations, consistency_score, 
                                        non_periodicity, non_linearity])
            irregularity_score = min(1.0, max(0.0, irregularity_score))
            
            # Only classify as irregular if it's truly irregular
            if irregularity_score > 0.6:
                return {
                    'confidence': float(irregularity_score),
                    'strength': float(irregularity_score),
                    'pattern_type': 'irregular_pattern',
                    'variability_score': float(variability),
                    'local_variations': float(local_variations),
                    'consistency_score': float(consistency_score),
                    'non_periodicity': float(non_periodicity),
                    'non_linearity': float(non_linearity),
                    'irregularity_characteristics': self._analyze_irregularity_characteristics(data),
                    'template': self._create_irregular_pattern_template(data)
                }
            else:
                return {'confidence': 0.0, 'strength': 0.0}
                
        except Exception as e:
            logger.error(f"Error detecting irregular pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}
    
    def _detect_custom_shape_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect custom shapes using adaptive template matching"""
        try:
            # Create adaptive template from the data itself
            template = self._create_adaptive_template(data)
            
            if template is None or len(template) < 3:
                return {'confidence': 0.0, 'strength': 0.0}
            
            # Test template matching across the data
            matching_score = self._test_template_matching(data, template)
            
            # Analyze shape characteristics
            shape_characteristics = self._analyze_custom_shape_characteristics(data, template)
            
            return {
                'confidence': float(matching_score),
                'strength': float(matching_score),
                'pattern_type': 'custom_shape',
                'template': template.tolist(),
                'shape_characteristics': shape_characteristics,
                'matching_score': float(matching_score),
                'template_length': len(template),
                'shape_complexity': shape_characteristics.get('complexity_score', 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error detecting custom shape pattern: {e}")
            return {'confidence': 0.0, 'strength': 0.0}

    # Helper methods for pattern detection would continue here...
    # (Due to length limits, I'll continue with the key synthesis and learning methods)
    
    def _generate_pattern_preserving_predictions(self, data: np.ndarray, steps: int,
                                               pattern_state: Dict[str, Any],
                                               synthesis_method: str) -> np.ndarray:
        """Generate predictions that preserve learned pattern characteristics"""
        try:
            if synthesis_method in self.synthesis_methods:
                return self.synthesis_methods[synthesis_method](data, steps, pattern_state)
            else:
                return self._adaptive_pattern_synthesis(data, steps, pattern_state)
                
        except Exception as e:
            logger.error(f"Error in pattern-preserving predictions: {e}")
            return self._fallback_pattern_synthesis(data, steps)
    
    def _geometric_pattern_synthesis(self, data: np.ndarray, steps: int, 
                                   pattern_state: Dict[str, Any]) -> np.ndarray:
        """Synthesize patterns based on geometric characteristics"""
        try:
            dominant_pattern = pattern_state.get('dominant_pattern')
            if not dominant_pattern:
                return self._fallback_pattern_synthesis(data, steps)
            
            pattern_type = dominant_pattern[1].get('pattern_type', 'unknown')
            
            # Route to specific geometric synthesis based on pattern type
            if pattern_type == 'square_wave':
                return self._synthesize_square_wave(data, steps, dominant_pattern[1])
            elif pattern_type == 'triangular_wave':
                return self._synthesize_triangular_wave(data, steps, dominant_pattern[1])
            elif pattern_type == 'sawtooth_wave':
                return self._synthesize_sawtooth_wave(data, steps, dominant_pattern[1])
            elif pattern_type == 'step_function':
                return self._synthesize_step_function(data, steps, dominant_pattern[1])
            elif pattern_type == 'pulse_pattern':
                return self._synthesize_pulse_pattern(data, steps, dominant_pattern[1])
            elif pattern_type == 'exponential_pattern':
                return self._synthesize_exponential_pattern(data, steps, dominant_pattern[1])
            elif pattern_type == 'sinusoidal_pattern':
                return self._synthesize_sinusoidal_pattern(data, steps, dominant_pattern[1])
            elif pattern_type == 'polynomial_pattern':
                return self._synthesize_polynomial_pattern(data, steps, dominant_pattern[1])
            elif pattern_type == 'spline_pattern':
                return self._synthesize_spline_pattern(data, steps, dominant_pattern[1])
            elif pattern_type == 'composite_pattern':
                return self._synthesize_composite_pattern(data, steps, dominant_pattern[1])
            elif pattern_type == 'custom_shape':
                return self._synthesize_custom_shape_pattern(data, steps, dominant_pattern[1])
            else:
                return self._adaptive_pattern_synthesis(data, steps, pattern_state)
                
        except Exception as e:
            logger.error(f"Error in geometric pattern synthesis: {e}")
            return self._fallback_pattern_synthesis(data, steps)
    
    def _synthesize_square_wave(self, data: np.ndarray, steps: int, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Synthesize square wave predictions"""
        try:
            plateaus = pattern_info.get('plateaus', [])
            transitions = pattern_info.get('transitions', [])
            amplitude_levels = pattern_info.get('amplitude_levels', [])
            duty_cycle = pattern_info.get('duty_cycle', 0.5)
            
            if not plateaus or not amplitude_levels:
                return self._fallback_pattern_synthesis(data, steps)
            
            # Extract square wave characteristics
            period_estimate = self._estimate_square_wave_period(plateaus)
            last_level = self._get_last_square_wave_level(data, amplitude_levels)
            
            predictions = []
            current_position = len(data)
            
            for step in range(steps):
                # Determine position in square wave cycle
                cycle_position = (current_position + step) % period_estimate
                
                # Determine which level we should be at
                if cycle_position < period_estimate * duty_cycle:
                    level = amplitude_levels[0] if len(amplitude_levels) >= 2 else amplitude_levels[0]
                else:
                    level = amplitude_levels[1] if len(amplitude_levels) >= 2 else amplitude_levels[0]
                
                predictions.append(level)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error synthesizing square wave: {e}")
            return self._fallback_pattern_synthesis(data, steps)
    
    def _synthesize_triangular_wave(self, data: np.ndarray, steps: int, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Synthesize triangular wave predictions"""
        try:
            peaks = pattern_info.get('peaks', [])
            valleys = pattern_info.get('valleys', [])
            slope_analysis = pattern_info.get('slope_analysis', {})
            
            if not peaks and not valleys:
                return self._fallback_pattern_synthesis(data, steps)
            
            # Estimate triangular wave parameters
            period_estimate = self._estimate_triangular_wave_period(peaks, valleys)
            amplitude = self._estimate_triangular_amplitude(data, peaks, valleys)
            
            # Get current position in triangle wave
            last_point = len(data) - 1
            current_value = data[-1]
            
            predictions = []
            
            for step in range(1, steps + 1):
                position = last_point + step
                cycle_position = position % period_estimate
                
                # Calculate triangular wave value
                if cycle_position < period_estimate / 2:
                    # Rising edge
                    progress = cycle_position / (period_estimate / 2)
                    value = current_value + (amplitude - current_value) * progress
                else:
                    # Falling edge
                    progress = (cycle_position - period_estimate / 2) / (period_estimate / 2)
                    value = amplitude - (amplitude - (-amplitude)) * progress
                
                predictions.append(value)
                current_value = value
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error synthesizing triangular wave: {e}")
            return self._fallback_pattern_synthesis(data, steps)
    
    # Additional synthesis methods would continue here...
    # (I'll include the key methods needed for the system to work)
    
    def _fallback_pattern_synthesis(self, data: np.ndarray, steps: int) -> np.ndarray:
        """Fallback synthesis method when specialized methods fail"""
        try:
            if len(data) < 2:
                return np.full(steps, data[-1] if len(data) > 0 else 0.0)
            
            # Simple trend continuation with pattern awareness
            recent_data = data[-min(10, len(data)):]
            trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            
            predictions = []
            last_value = data[-1]
            
            for step in range(1, steps + 1):
                predicted_value = last_value + trend * step
                predictions.append(predicted_value)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in fallback synthesis: {e}")
            return np.full(steps, data[-1] if len(data) > 0 else 0.0)

    # Additional helper methods for comprehensive pattern analysis...
    # (Due to space constraints, including key placeholder methods)

    def _create_minimal_learning_result(self, data: np.ndarray) -> Dict[str, Any]:
        """Create minimal result for insufficient data"""
        return {
            'status': 'minimal_data',
            'message': 'Insufficient data for comprehensive pattern learning',
            'data_length': len(data),
            'prediction_readiness': 0.2
        }
    
    def _create_error_learning_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'status': 'error',
            'error': error_message,
            'prediction_readiness': 0.1
        }
    
    def _generate_fallback_waveform_predictions(self, data: np.ndarray, steps: int) -> Dict[str, Any]:
        """Generate fallback predictions when main system fails"""
        try:
            predictions = self._fallback_pattern_synthesis(data, steps)
            return {
                'predictions': predictions.tolist(),
                'confidence_intervals': [],
                'pattern_state': {'pattern_type': 'fallback'},
                'synthesis_method': 'fallback',
                'prediction_quality': {'overall_quality': 0.3}
            }
        except:
            return {
                'predictions': [0.0] * steps,
                'confidence_intervals': [],
                'pattern_state': {},
                'synthesis_method': 'error',
                'prediction_quality': {'overall_quality': 0.1}
            }

    # Placeholder methods for the comprehensive system
    # (These would be fully implemented in the complete system)
    
    def _find_plateaus(self, data: np.ndarray) -> List[Dict]:
        """Find flat segments in data"""
        try:
            if len(data) < 3:
                return []
            
            plateaus = []
            tolerance = np.std(data) * 0.1  # Adaptive tolerance based on data variability
            min_plateau_length = max(3, len(data) // 20)  # Minimum plateau length
            
            current_plateau_start = 0
            current_plateau_value = data[0]
            
            for i in range(1, len(data)):
                # Check if current point is within tolerance of plateau value
                if abs(data[i] - current_plateau_value) <= tolerance:
                    continue
                else:
                    # End of current plateau
                    plateau_length = i - current_plateau_start
                    if plateau_length >= min_plateau_length:
                        plateaus.append({
                            'start_idx': current_plateau_start,
                            'end_idx': i - 1,
                            'length': plateau_length,
                            'value': current_plateau_value,
                            'variance': float(np.var(data[current_plateau_start:i])),
                            'flatness_score': 1.0 / (1.0 + float(np.var(data[current_plateau_start:i])))
                        })
                    
                    # Start new plateau
                    current_plateau_start = i
                    current_plateau_value = data[i]
            
            # Check final segment
            if len(data) - current_plateau_start >= min_plateau_length:
                plateaus.append({
                    'start_idx': current_plateau_start,
                    'end_idx': len(data) - 1,
                    'length': len(data) - current_plateau_start,
                    'value': current_plateau_value,
                    'variance': float(np.var(data[current_plateau_start:])),
                    'flatness_score': 1.0 / (1.0 + float(np.var(data[current_plateau_start:])))
                })
            
            logger.info(f"Found {len(plateaus)} plateaus in data")
            return plateaus
            
        except Exception as e:
            logger.error(f"Error finding plateaus: {e}")
            return []
    
    def _find_sharp_transitions(self, data: np.ndarray) -> List[Dict]:
        """Find sharp transitions in data"""
        try:
            if len(data) < 3:
                return []
            
            transitions = []
            
            # Calculate first derivative to find transition points
            gradient = np.gradient(data)
            
            # Find points where gradient changes significantly
            gradient_threshold = np.std(gradient) * 1.5
            
            for i in range(1, len(gradient) - 1):
                # Check for sharp changes in gradient
                left_grad = gradient[i - 1]
                curr_grad = gradient[i]
                right_grad = gradient[i + 1]
                
                # Detect transition if gradient changes rapidly
                if (abs(curr_grad) > gradient_threshold and 
                    abs(curr_grad - left_grad) > gradient_threshold and
                    abs(curr_grad - right_grad) > gradient_threshold):
                    
                    # Calculate transition characteristics
                    transition_strength = abs(curr_grad)
                    sharpness = abs(curr_grad - left_grad) + abs(curr_grad - right_grad)
                    
                    transitions.append({
                        'idx': i,
                        'value_before': data[i - 1],
                        'value_after': data[i + 1],
                        'gradient': curr_grad,
                        'transition_strength': float(transition_strength),
                        'sharpness_score': float(sharpness / (2 * gradient_threshold)),
                        'amplitude_change': abs(data[i + 1] - data[i - 1])
                    })
            
            # Sort by transition strength
            transitions.sort(key=lambda x: x['transition_strength'], reverse=True)
            
            logger.info(f"Found {len(transitions)} sharp transitions in data")
            return transitions
            
        except Exception as e:
            logger.error(f"Error finding sharp transitions: {e}")
            return []
    
    def _comprehensive_geometric_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Comprehensive geometric analysis"""
        try:
            analysis = {}
            
            if len(data) < 3:
                return {'status': 'insufficient_data'}
            
            # Basic statistical properties
            analysis['mean'] = float(np.mean(data))
            analysis['std'] = float(np.std(data))
            analysis['range'] = float(np.max(data) - np.min(data))
            analysis['variance'] = float(np.var(data))
            
            # Shape characteristics
            analysis['curvature'] = self._calculate_curvature(data)
            analysis['linearity'] = self._calculate_linearity(data)
            analysis['periodicity'] = self._estimate_periodicity(data)
            analysis['symmetry'] = self._calculate_symmetry(data)
            
            # Frequency domain characteristics
            if len(data) >= 4:
                fft_data = np.fft.fft(data - np.mean(data))
                frequencies = np.fft.fftfreq(len(data))
                
                # Find dominant frequency
                magnitude = np.abs(fft_data)
                dominant_freq_idx = np.argmax(magnitude[1:len(data)//2]) + 1
                dominant_frequency = frequencies[dominant_freq_idx]
                
                analysis['dominant_frequency'] = float(dominant_frequency)
                analysis['frequency_strength'] = float(magnitude[dominant_freq_idx])
                analysis['harmonic_content'] = self._analyze_harmonics(fft_data, frequencies)
            
            # Trend analysis
            if len(data) >= 2:
                trend = np.polyfit(range(len(data)), data, 1)[0]
                analysis['trend_strength'] = float(abs(trend))
                analysis['trend_direction'] = 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable'
            
            analysis['status'] = 'complete'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive geometric analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _apply_all_learning_strategies(self, data: np.ndarray, 
                                     detected_patterns: Dict, 
                                     geometric_analysis: Dict) -> Dict[str, Any]:
        """Apply all learning strategies to learn from detected patterns"""
        try:
            learned_patterns = {}
            
            # Extract dominant pattern for focused learning
            dominant_pattern = detected_patterns.get('dominant_pattern')
            all_patterns = detected_patterns.get('all_patterns', {})
            
            if dominant_pattern:
                pattern_name, pattern_info = dominant_pattern
                learned_patterns['dominant'] = {
                    'type': pattern_name,
                    'info': pattern_info,
                    'learning_confidence': pattern_info.get('confidence', 0.5)
                }
                
                logger.info(f"Learning strategy focused on dominant pattern: {pattern_name}")
            
            # Learn from geometric analysis
            if geometric_analysis.get('status') == 'complete':
                learned_patterns['geometric_characteristics'] = {
                    'curvature_profile': geometric_analysis.get('curvature', 0.0),
                    'linearity_level': geometric_analysis.get('linearity', 0.0),
                    'periodicity_info': geometric_analysis.get('periodicity', {}),
                    'symmetry_score': geometric_analysis.get('symmetry', 0.0),
                    'dominant_frequency': geometric_analysis.get('dominant_frequency', 0.0),
                    'trend_characteristics': {
                        'strength': geometric_analysis.get('trend_strength', 0.0),
                        'direction': geometric_analysis.get('trend_direction', 'stable')
                    }
                }
            
            # Pattern template learning
            templates_learned = []
            for pattern_name, pattern_info in all_patterns.items():
                if pattern_info.get('confidence', 0) > 0.3:
                    template = self._extract_pattern_template(pattern_name, pattern_info, data)
                    if template is not None and len(template) > 0:
                        templates_learned.append({
                            'pattern_type': pattern_name,
                            'template': template,
                            'confidence': pattern_info.get('confidence', 0.5),
                            'characteristics': pattern_info
                        })
            
            learned_patterns['templates'] = templates_learned
            
            # Learning quality assessment
            learning_quality = {
                'pattern_coverage': len(all_patterns),
                'dominant_pattern_strength': dominant_pattern[1].get('confidence', 0.0) if dominant_pattern else 0.0,
                'template_quality': len(templates_learned),
                'geometric_completeness': 1.0 if geometric_analysis.get('status') == 'complete' else 0.3
            }
            
            learned_patterns['learning_quality'] = learning_quality
            
            return learned_patterns
            
        except Exception as e:
            logger.error(f"Error applying learning strategies: {e}")
            return {'learned_patterns': [], 'learning_quality': {'overall_quality': 0.2}}

    def _extract_pattern_template(self, pattern_name: str, pattern_info: Dict[str, Any], data: np.ndarray) -> Optional[np.ndarray]:
        """Extract a reusable template from detected pattern"""
        try:
            if pattern_name == 'square_wave':
                return pattern_info.get('template', np.array([]))
            elif pattern_name == 'triangular_wave':
                return self._create_triangular_template(pattern_info, data)
            elif pattern_name == 'sawtooth_wave':
                return self._create_sawtooth_template(pattern_info, data)
            elif pattern_name == 'sinusoidal_pattern':
                return self._create_sinusoidal_template(pattern_info, data)
            elif pattern_name == 'step_function':
                return self._create_step_template(pattern_info, data)
            else:
                # Generic template based on data characteristics
                return self._create_generic_template(pattern_info, data)
                
        except Exception as e:
            logger.error(f"Error extracting pattern template for {pattern_name}: {e}")
            return None
    
    def _create_triangular_template(self, pattern_info: Dict[str, Any], data: np.ndarray) -> np.ndarray:
        """Create triangular wave template"""
        try:
            peaks = pattern_info.get('peaks', [])
            valleys = pattern_info.get('valleys', [])
            
            if not peaks and not valleys:
                return np.array([])
            
            # Estimate period and amplitude
            period = self._estimate_triangular_wave_period(np.array(peaks), np.array(valleys))
            amplitude = self._estimate_triangular_amplitude(data, np.array(peaks), np.array(valleys))
            
            # Create template
            template = []
            for i in range(period):
                if i < period // 2:
                    # Rising edge
                    value = -amplitude + (2 * amplitude * i) / (period // 2)
                else:
                    # Falling edge
                    value = amplitude - (2 * amplitude * (i - period // 2)) / (period // 2)
                template.append(value)
            
            return np.array(template)
            
        except Exception as e:
            logger.error(f"Error creating triangular template: {e}")
            return np.array([])
    
    def _create_sawtooth_template(self, pattern_info: Dict[str, Any], data: np.ndarray) -> np.ndarray:
        """Create sawtooth wave template"""
        try:
            # Estimate characteristics
            data_range = np.max(data) - np.min(data)
            amplitude = data_range / 2
            mean_val = np.mean(data)
            period = max(10, len(data) // 5)  # Estimate period
            
            template = []
            for i in range(period):
                if i < period - 1:
                    # Linear rise
                    value = mean_val - amplitude + (2 * amplitude * i) / (period - 1)
                else:
                    # Sharp drop
                    value = mean_val - amplitude
                template.append(value)
            
            return np.array(template)
            
        except Exception as e:
            logger.error(f"Error creating sawtooth template: {e}")
            return np.array([])
    
    def _create_sinusoidal_template(self, pattern_info: Dict[str, Any], data: np.ndarray) -> np.ndarray:
        """Create sinusoidal wave template"""
        try:
            # Estimate sinusoidal parameters
            amplitude = np.std(data) * 1.4
            mean_val = np.mean(data)
            period = max(8, len(data) // 4)
            
            template = []
            for i in range(period):
                value = mean_val + amplitude * np.sin(2 * np.pi * i / period)
                template.append(value)
            
            return np.array(template)
            
        except Exception as e:
            logger.error(f"Error creating sinusoidal template: {e}")
            return np.array([])
    
    def _create_step_template(self, pattern_info: Dict[str, Any], data: np.ndarray) -> np.ndarray:
        """Create step function template"""
        try:
            plateaus = pattern_info.get('plateaus', [])
            if not plateaus:
                return np.array([])
            
            # Extract step levels
            levels = [p['value'] for p in plateaus]
            unique_levels = sorted(list(set(levels)))
            
            # Create simple step template
            if len(unique_levels) >= 2:
                step_length = max(5, len(data) // 10)
                template = [unique_levels[0]] * step_length + [unique_levels[1]] * step_length
                return np.array(template)
            else:
                return np.array([levels[0]] * 10)
                
        except Exception as e:
            logger.error(f"Error creating step template: {e}")
            return np.array([])
    
    def _create_generic_template(self, pattern_info: Dict[str, Any], data: np.ndarray) -> np.ndarray:
        """Create generic template from data"""
        try:
            # Use recent data as template
            template_length = min(20, len(data) // 2)
            if template_length > 0:
                return data[-template_length:].copy()
            else:
                return np.array([np.mean(data)] * 10)
                
        except Exception as e:
            logger.error(f"Error creating generic template: {e}")
            return np.array([])

    def _apply_waveform_shape_corrections(self, predictions: np.ndarray, data: np.ndarray, pattern_state: Dict[str, Any]) -> np.ndarray:
        """Apply waveform shape corrections to maintain pattern characteristics"""
        try:
            if len(predictions) == 0:
                return predictions
            
            dominant_pattern = pattern_state.get('dominant_pattern')
            if not dominant_pattern:
                return predictions
            
            pattern_info = dominant_pattern[1]
            pattern_type = pattern_info.get('pattern_type', 'unknown')
            
            logger.info(f"Applying waveform shape corrections for pattern type: {pattern_type}")
            
            # Apply pattern-specific shape corrections
            if pattern_type == 'square_wave':
                return self._apply_square_wave_shape_corrections(predictions, data, pattern_info)
            elif pattern_type == 'triangular_wave':
                return self._apply_triangular_wave_shape_corrections(predictions, data, pattern_info)
            elif pattern_type == 'sawtooth_wave':
                return self._apply_sawtooth_wave_shape_corrections(predictions, data, pattern_info)
            elif pattern_type == 'step_function':
                return self._apply_step_function_shape_corrections(predictions, data, pattern_info)
            elif pattern_type == 'sinusoidal_pattern':
                return self._apply_sinusoidal_shape_corrections(predictions, data, pattern_info)
            else:
                return self._apply_general_shape_corrections(predictions, data, pattern_info)
                
        except Exception as e:
            logger.error(f"Error applying waveform shape corrections: {e}")
            return predictions
    
    def _apply_square_wave_shape_corrections(self, predictions: np.ndarray, data: np.ndarray, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Apply shape corrections specific to square waves"""
        try:
            amplitude_levels = pattern_info.get('amplitude_levels', [])
            if not amplitude_levels or len(amplitude_levels) < 2:
                return predictions
            
            corrected_predictions = []
            
            for pred in predictions:
                # Snap to nearest amplitude level
                distances = [abs(pred - level) for level in amplitude_levels]
                closest_level_idx = np.argmin(distances)
                corrected_predictions.append(amplitude_levels[closest_level_idx])
            
            return np.array(corrected_predictions)
            
        except Exception as e:
            logger.error(f"Error in square wave shape corrections: {e}")
            return predictions
    
    def _apply_triangular_wave_shape_corrections(self, predictions: np.ndarray, data: np.ndarray, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Apply shape corrections specific to triangular waves"""
        try:
            # Enforce linear segments with sharp transitions at extrema
            if len(predictions) < 3:
                return predictions
            
            corrected = predictions.copy()
            
            # Find local extrema in predictions
            peaks, _ = find_peaks(corrected, prominence=np.std(corrected) * 0.3)
            valleys, _ = find_peaks(-corrected, prominence=np.std(corrected) * 0.3)
            
            all_extrema = np.sort(np.concatenate([peaks, valleys]))
            
            # Enforce linear segments between extrema
            for i in range(len(all_extrema) - 1):
                start_idx = all_extrema[i]
                end_idx = all_extrema[i + 1]
                
                if end_idx - start_idx >= 2:
                    # Create linear interpolation between extrema
                    start_val = corrected[start_idx]
                    end_val = corrected[end_idx]
                    
                    for j in range(start_idx + 1, end_idx):
                        progress = (j - start_idx) / (end_idx - start_idx)
                        corrected[j] = start_val + (end_val - start_val) * progress
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error in triangular wave shape corrections: {e}")
            return predictions
    
    def _apply_sawtooth_wave_shape_corrections(self, predictions: np.ndarray, data: np.ndarray, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Apply shape corrections specific to sawtooth waves"""
        try:
            # Similar to triangular but with sharp drops
            if len(predictions) < 3:
                return predictions
            
            corrected = predictions.copy()
            
            # Detect sharp drops (sawtooth characteristic)
            gradient = np.gradient(corrected)
            sharp_drops = np.where(gradient < -np.std(gradient) * 2)[0]
            
            # Ensure linear rise between sharp drops
            last_drop = 0
            for drop_idx in sharp_drops:
                if drop_idx > last_drop + 1:
                    # Linear interpolation for rising part
                    start_val = corrected[last_drop]
                    end_val = corrected[drop_idx]
                    
                    for j in range(last_drop + 1, drop_idx):
                        progress = (j - last_drop) / (drop_idx - last_drop)
                        corrected[j] = start_val + (end_val - start_val) * progress
                
                last_drop = drop_idx
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error in sawtooth wave shape corrections: {e}")
            return predictions
    
    def _apply_step_function_shape_corrections(self, predictions: np.ndarray, data: np.ndarray, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Apply shape corrections specific to step functions"""
        try:
            # Similar to square wave but less periodic
            plateaus = pattern_info.get('plateaus', [])
            transitions = pattern_info.get('transitions', [])
            
            if not plateaus:
                return predictions
            
            # Extract step levels
            step_levels = [p['value'] for p in plateaus]
            unique_levels = sorted(list(set(step_levels)))
            
            corrected_predictions = []
            
            for pred in predictions:
                # Snap to nearest step level
                distances = [abs(pred - level) for level in unique_levels]
                closest_level_idx = np.argmin(distances)
                corrected_predictions.append(unique_levels[closest_level_idx])
            
            return np.array(corrected_predictions)
            
        except Exception as e:
            logger.error(f"Error in step function shape corrections: {e}")
            return predictions
    
    def _apply_sinusoidal_shape_corrections(self, predictions: np.ndarray, data: np.ndarray, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Apply shape corrections specific to sinusoidal patterns"""
        try:
            # Smooth out sharp edges to maintain sinusoidal smoothness
            if len(predictions) < 3:
                return predictions
            
            # Apply smoothing to maintain sinusoidal characteristics
            corrected = predictions.copy()
            
            # Use savgol filter for smoothing while preserving shape
            if len(corrected) >= 5:
                window_length = min(5, len(corrected) if len(corrected) % 2 == 1 else len(corrected) - 1)
                if window_length >= 3:
                    corrected = savgol_filter(corrected, window_length, 2)
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error in sinusoidal shape corrections: {e}")
            return predictions
    
    def _apply_general_shape_corrections(self, predictions: np.ndarray, data: np.ndarray, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Apply general shape corrections"""
        try:
            # General smoothing and outlier correction
            corrected = predictions.copy()
            
            # Remove extreme outliers
            if len(data) > 0:
                data_std = np.std(data)
                data_mean = np.mean(data)
                
                for i in range(len(corrected)):
                    if abs(corrected[i] - data_mean) > data_std * 3:
                        # Replace outlier with moving average
                        if i > 0 and i < len(corrected) - 1:
                            corrected[i] = (corrected[i-1] + corrected[i+1]) / 2
                        elif i > 0:
                            corrected[i] = corrected[i-1]
                        else:
                            corrected[i] = data_mean
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error in general shape corrections: {e}")
            return predictions

    def _apply_geometric_consistency_corrections(self, predictions: np.ndarray, data: np.ndarray, pattern_state: Dict[str, Any]) -> np.ndarray:
        """Apply geometric consistency corrections"""
        try:
            if len(predictions) < 2:
                return predictions
            
            corrected = predictions.copy()
            
            # Ensure consistency with historical geometric properties
            if len(data) >= 2:
                # Match amplitude range
                data_range = np.max(data) - np.min(data)
                pred_range = np.max(corrected) - np.min(corrected)
                
                if pred_range > 0 and data_range > 0:
                    # Scale predictions to match historical amplitude
                    scale_factor = data_range / pred_range
                    pred_mean = np.mean(corrected)
                    data_mean = np.mean(data)
                    
                    corrected = (corrected - pred_mean) * scale_factor + data_mean
                
                # Ensure smooth transitions from historical data
                if len(data) > 0:
                    transition_weight = 0.3
                    corrected[0] = corrected[0] * (1 - transition_weight) + data[-1] * transition_weight
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error in geometric consistency corrections: {e}")
            return predictions

    def _apply_amplitude_frequency_preservation(self, predictions: np.ndarray, data: np.ndarray, pattern_state: Dict[str, Any]) -> np.ndarray:
        """Apply amplitude and frequency preservation corrections"""
        try:
            if len(predictions) < 3 or len(data) < 3:
                return predictions
            
            corrected = predictions.copy()
            
            # Preserve amplitude characteristics
            data_amplitude = (np.max(data) - np.min(data)) / 2
            data_center = (np.max(data) + np.min(data)) / 2
            
            pred_amplitude = (np.max(corrected) - np.min(corrected)) / 2
            pred_center = (np.max(corrected) + np.min(corrected)) / 2
            
            if pred_amplitude > 0:
                # Scale to match historical amplitude
                amplitude_scale = data_amplitude / pred_amplitude
                corrected = (corrected - pred_center) * amplitude_scale + data_center
            
            # Frequency preservation (basic approach)
            dominant_pattern = pattern_state.get('dominant_pattern')
            if dominant_pattern and len(corrected) >= 6:
                pattern_info = dominant_pattern[1]
                if 'periodicity_score' in pattern_info and pattern_info['periodicity_score'] > 0.5:
                    # Try to maintain periodic structure
                    # This is a simplified approach - could be enhanced
                    pass
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error in amplitude/frequency preservation: {e}")
            return predictions

    def _apply_continuity_smoothness_corrections(self, predictions: np.ndarray, data: np.ndarray, 
                                               previous_predictions: Optional[List], pattern_state: Dict[str, Any]) -> np.ndarray:
        """Apply continuity and smoothness corrections"""
        try:
            if len(predictions) == 0:
                return predictions
            
            corrected = predictions.copy()
            
            # Ensure smooth transition from historical data
            if len(data) > 0:
                # Smooth connection to last historical point
                if len(corrected) > 0:
                    # Apply exponential smoothing for first few predictions
                    smoothing_length = min(3, len(corrected))
                    alpha = 0.3  # Smoothing factor
                    
                    for i in range(smoothing_length):
                        weight = alpha * (1 - alpha) ** i
                        corrected[i] = corrected[i] * (1 - weight) + data[-1] * weight
            
            # Apply pattern-specific smoothness
            dominant_pattern = pattern_state.get('dominant_pattern')
            if dominant_pattern:
                pattern_type = dominant_pattern[1].get('pattern_type', 'unknown')
                
                # Different smoothness for different patterns
                if pattern_type == 'sinusoidal_pattern':
                    # High smoothness for sinusoidal
                    if len(corrected) >= 5:
                        window_length = min(5, len(corrected) if len(corrected) % 2 == 1 else len(corrected) - 1)
                        corrected = savgol_filter(corrected, window_length, 2)
                elif pattern_type in ['square_wave', 'step_function']:
                    # Maintain sharp edges for square waves
                    pass  # No additional smoothing
                elif pattern_type == 'triangular_wave':
                    # Light smoothing for triangular
                    if len(corrected) >= 3:
                        window_length = min(3, len(corrected) if len(corrected) % 2 == 1 else len(corrected) - 1)
                        corrected = savgol_filter(corrected, window_length, 1)
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error in continuity/smoothness corrections: {e}")
            return predictions

    def _calculate_pattern_aware_confidence_intervals(self, predictions: np.ndarray, data: np.ndarray, pattern_state: Dict[str, Any]) -> List[Dict]:
        """Calculate pattern-aware confidence intervals"""
        try:
            if len(predictions) == 0:
                return []
            
            confidence_intervals = []
            
            # Base confidence on pattern detection quality
            dominant_pattern = pattern_state.get('dominant_pattern')
            base_confidence = 0.8 if dominant_pattern else 0.5
            
            if dominant_pattern:
                pattern_confidence = dominant_pattern[1].get('confidence', 0.5)
                base_confidence *= pattern_confidence
            
            # Calculate adaptive confidence based on data variability
            if len(data) > 0:
                data_std = np.std(data)
                
                for i, pred in enumerate(predictions):
                    # Confidence decreases with prediction horizon
                    horizon_factor = 1.0 - (i * 0.02)  # 2% decrease per step
                    confidence = base_confidence * max(0.3, horizon_factor)
                    
                    # Confidence interval based on data variability
                    interval_width = data_std * (2 - confidence)  # Wider intervals for lower confidence
                    
                    confidence_intervals.append({
                        'lower': float(pred - interval_width),
                        'upper': float(pred + interval_width),
                        'confidence': float(confidence)
                    })
            else:
                # Default intervals if no historical data
                for i, pred in enumerate(predictions):
                    confidence_intervals.append({
                        'lower': float(pred * 0.9),
                        'upper': float(pred * 1.1),
                        'confidence': 0.5
                    })
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating pattern-aware confidence intervals: {e}")
            return []

    def _assess_waveform_prediction_quality(self, predictions: np.ndarray, data: np.ndarray, pattern_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of waveform predictions"""
        try:
            quality_assessment = {}
            
            if len(predictions) == 0:
                return {'overall_quality': 0.0}
            
            # Shape fidelity assessment
            quality_assessment['shape_fidelity'] = self._assess_shape_fidelity(predictions, data, pattern_state)
            
            # Amplitude accuracy
            quality_assessment['amplitude_accuracy'] = self._assess_amplitude_accuracy(predictions, data)
            
            # Frequency accuracy  
            quality_assessment['frequency_accuracy'] = self._assess_frequency_accuracy(predictions, data)
            
            # Geometric consistency
            quality_assessment['geometric_consistency'] = self._assess_geometric_consistency(predictions, data)
            
            # Transition quality
            quality_assessment['transition_quality'] = self._assess_transition_quality(predictions, data, pattern_state)
            
            # Overall quality (weighted average)
            weights = {
                'shape_fidelity': 0.3,
                'amplitude_accuracy': 0.25,
                'frequency_accuracy': 0.2,
                'geometric_consistency': 0.15,
                'transition_quality': 0.1
            }
            
            overall_quality = sum(quality_assessment[key] * weights[key] for key in weights.keys())
            quality_assessment['overall_quality'] = float(overall_quality)
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Error assessing waveform prediction quality: {e}")
            return {'overall_quality': 0.3}

    def _assess_shape_fidelity(self, predictions: np.ndarray, data: np.ndarray, pattern_state: Dict[str, Any]) -> float:
        """Assess how well predictions maintain the shape of the pattern"""
        try:
            if len(predictions) < 2 or len(data) < 2:
                return 0.5
            
            dominant_pattern = pattern_state.get('dominant_pattern')
            if not dominant_pattern:
                return 0.5
            
            pattern_type = dominant_pattern[1].get('pattern_type', 'unknown')
            pattern_confidence = dominant_pattern[1].get('confidence', 0.5)
            
            # Shape fidelity based on pattern type
            if pattern_type == 'square_wave':
                return self._assess_square_wave_fidelity(predictions, data)
            elif pattern_type == 'triangular_wave':
                return self._assess_triangular_wave_fidelity(predictions, data)
            elif pattern_type == 'sinusoidal_pattern':
                return self._assess_sinusoidal_fidelity(predictions, data)
            else:
                # General shape fidelity
                return float(pattern_confidence * 0.8)
                
        except Exception as e:
            logger.error(f"Error assessing shape fidelity: {e}")
            return 0.5
    
    def _assess_square_wave_fidelity(self, predictions: np.ndarray, data: np.ndarray) -> float:
        """Assess fidelity for square wave patterns"""
        try:
            # Check for discrete levels in predictions
            unique_pred_levels = len(np.unique(np.round(predictions, 1)))
            
            # Good square wave should have ~2 levels
            if unique_pred_levels <= 3:
                level_score = 0.9
            elif unique_pred_levels <= 5:
                level_score = 0.7
            else:
                level_score = 0.4
            
            # Check for sharp transitions
            gradient = np.abs(np.gradient(predictions))
            sharp_transitions = np.sum(gradient > np.std(gradient))
            
            if len(predictions) > 0:
                transition_score = min(1.0, sharp_transitions / (len(predictions) * 0.2))
            else:
                transition_score = 0.5
            
            return float(level_score * 0.6 + transition_score * 0.4)
            
        except Exception as e:
            logger.error(f"Error assessing square wave fidelity: {e}")
            return 0.5
    
    def _assess_triangular_wave_fidelity(self, predictions: np.ndarray, data: np.ndarray) -> float:
        """Assess fidelity for triangular wave patterns"""
        try:
            # Check for linear segments
            linearity_scores = []
            
            # Find peaks and valleys
            peaks, _ = find_peaks(predictions, prominence=np.std(predictions) * 0.3)
            valleys, _ = find_peaks(-predictions, prominence=np.std(predictions) * 0.3)
            
            all_extrema = np.sort(np.concatenate([peaks, valleys]))
            
            # Check linearity of segments between extrema
            for i in range(len(all_extrema) - 1):
                start_idx = all_extrema[i]
                end_idx = all_extrema[i + 1]
                
                if end_idx - start_idx >= 3:
                    segment = predictions[start_idx:end_idx + 1]
                    x = np.arange(len(segment))
                    
                    # Calculate R-squared for linearity
                    slope, intercept = np.polyfit(x, segment, 1)
                    predicted_line = slope * x + intercept
                    
                    ss_res = np.sum((segment - predicted_line) ** 2)
                    ss_tot = np.sum((segment - np.mean(segment)) ** 2)
                    
                    if ss_tot > 0:
                        r_squared = 1 - (ss_res / ss_tot)
                        linearity_scores.append(max(0.0, r_squared))
            
            if linearity_scores:
                return float(np.mean(linearity_scores))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error assessing triangular wave fidelity: {e}")
            return 0.5
    
    def _assess_sinusoidal_fidelity(self, predictions: np.ndarray, data: np.ndarray) -> float:
        """Assess fidelity for sinusoidal patterns"""
        try:
            # Check smoothness (sinusoidal should be smooth)
            if len(predictions) < 3:
                return 0.5
            
            second_derivative = np.abs(np.diff(predictions, n=2))
            smoothness = 1.0 / (1.0 + np.mean(second_derivative))
            
            return float(smoothness)
            
        except Exception as e:
            logger.error(f"Error assessing sinusoidal fidelity: {e}")
            return 0.5

    def _assess_amplitude_accuracy(self, predictions: np.ndarray, data: np.ndarray) -> float:
        """Assess amplitude accuracy of predictions"""
        try:
            if len(predictions) == 0 or len(data) == 0:
                return 0.5
            
            pred_amplitude = (np.max(predictions) - np.min(predictions)) / 2
            data_amplitude = (np.max(data) - np.min(data)) / 2
            
            if data_amplitude > 0:
                amplitude_ratio = min(pred_amplitude, data_amplitude) / max(pred_amplitude, data_amplitude)
                return float(amplitude_ratio)
            else:
                return 1.0 if pred_amplitude == 0 else 0.0
                
        except Exception as e:
            logger.error(f"Error assessing amplitude accuracy: {e}")
            return 0.5

    def _assess_frequency_accuracy(self, predictions: np.ndarray, data: np.ndarray) -> float:
        """Assess frequency accuracy of predictions"""
        try:
            if len(predictions) < 4 or len(data) < 4:
                return 0.5
            
            # Simple frequency estimation using zero crossings
            def count_zero_crossings(signal):
                mean_val = np.mean(signal)
                centered = signal - mean_val
                return np.sum(np.diff(np.signbit(centered)))
            
            pred_crossings = count_zero_crossings(predictions)
            data_crossings = count_zero_crossings(data[-len(predictions):] if len(data) >= len(predictions) else data)
            
            if max(pred_crossings, data_crossings) > 0:
                frequency_ratio = min(pred_crossings, data_crossings) / max(pred_crossings, data_crossings)
                return float(frequency_ratio)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error assessing frequency accuracy: {e}")
            return 0.5

    def _assess_geometric_consistency(self, predictions: np.ndarray, data: np.ndarray) -> float:
        """Assess geometric consistency of predictions"""
        try:
            if len(predictions) < 2:
                return 0.5
            
            consistency_metrics = []
            
            # Trend consistency
            if len(predictions) > 1 and len(data) > 1:
                pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
                data_trend = np.polyfit(range(min(len(data), len(predictions))), data[-len(predictions):], 1)[0]
                
                if abs(data_trend) > 1e-8:
                    trend_consistency = 1.0 - abs(pred_trend - data_trend) / (abs(data_trend) + 1e-8)
                else:
                    trend_consistency = 1.0 if abs(pred_trend) < 1e-8 else 0.0
                
                consistency_metrics.append(max(0.0, trend_consistency))
            
            # Variability consistency
            if len(data) > 0:
                pred_std = np.std(predictions)
                data_std = np.std(data)
                
                if max(pred_std, data_std) > 1e-8:
                    var_consistency = min(pred_std, data_std) / max(pred_std, data_std)
                else:
                    var_consistency = 1.0
                
                consistency_metrics.append(var_consistency)
            
            if consistency_metrics:
                return float(np.mean(consistency_metrics))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error assessing geometric consistency: {e}")
            return 0.5

    def _assess_transition_quality(self, predictions: np.ndarray, data: np.ndarray, pattern_state: Dict[str, Any]) -> float:
        """Assess quality of transitions in predictions"""
        try:
            if len(predictions) < 3:
                return 0.5
            
            # Check smoothness of transitions
            gradient = np.gradient(predictions)
            gradient_changes = np.abs(np.diff(gradient))
            
            # Lower gradient changes indicate smoother transitions
            smoothness_score = 1.0 / (1.0 + np.mean(gradient_changes))
            
            return float(smoothness_score)
            
        except Exception as e:
            logger.error(f"Error assessing transition quality: {e}")
            return 0.5

    def _update_learning_from_predictions(self, predictions: np.ndarray, data: np.ndarray, 
                                        pattern_state: Dict[str, Any], prediction_quality: Dict[str, Any]) -> None:
        """Update learning from prediction results"""
        try:
            # Store prediction performance for future improvement
            dominant_pattern = pattern_state.get('dominant_pattern')
            if dominant_pattern:
                pattern_type = dominant_pattern[1].get('pattern_type', 'unknown')
                overall_quality = prediction_quality.get('overall_quality', 0.5)
                
                # Update pattern performance tracking
                if pattern_type not in self.pattern_performance:
                    self.pattern_performance[pattern_type] = []
                
                self.pattern_performance[pattern_type].append(overall_quality)
                
                # Keep only recent performance data
                if len(self.pattern_performance[pattern_type]) > 100:
                    self.pattern_performance[pattern_type] = self.pattern_performance[pattern_type][-50:]
            
            # Update general adaptation history
            adaptation_event = {
                'timestamp': datetime.now(),
                'prediction_length': len(predictions),
                'data_length': len(data),
                'quality_metrics': prediction_quality
            }
            
            self.adaptation_history.append(adaptation_event)
            
        except Exception as e:
            logger.error(f"Error updating learning from predictions: {e}")
    
    def _assess_pattern_complexity_level(self, detected_patterns: Dict[str, Any]) -> float:
        """Assess the complexity level of detected patterns"""
        try:
            if not detected_patterns:
                return 0.0
            
            complexity_scores = []
            
            for pattern_name, pattern_info in detected_patterns.items():
                confidence = pattern_info.get('confidence', 0.0)
                
                # Different patterns have different inherent complexity
                pattern_complexity = {
                    'square_wave': 0.6,
                    'triangular_wave': 0.7,
                    'sawtooth_wave': 0.7,
                    'step_function': 0.4,
                    'sinusoidal_pattern': 0.8,
                    'polynomial_pattern': 0.9,
                    'composite_pattern': 0.95,
                    'fractal_pattern': 0.99,
                    'chaotic_pattern': 0.99
                }.get(pattern_name, 0.5)
                
                weighted_complexity = pattern_complexity * confidence
                complexity_scores.append(weighted_complexity)
            
            if complexity_scores:
                return float(np.mean(complexity_scores))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error assessing pattern complexity level: {e}")
            return 0.0
    
    def _assess_synthesis_capabilities(self, learned_patterns: Dict) -> Dict[str, Any]:
        """Assess synthesis capabilities"""
        try:
            synthesis_capabilities = {}
            
            # Check if we have learned templates
            templates = learned_patterns.get('templates', [])
            dominant_pattern = learned_patterns.get('dominant')
            
            if templates:
                synthesis_capabilities['template_synthesis'] = {
                    'available_templates': len(templates),
                    'template_quality': np.mean([t['confidence'] for t in templates]),
                    'synthesis_readiness': 0.9
                }
            
            if dominant_pattern:
                pattern_type = dominant_pattern.get('type', 'unknown')
                learning_confidence = dominant_pattern.get('learning_confidence', 0.5)
                
                synthesis_capabilities['pattern_specific_synthesis'] = {
                    'dominant_pattern_type': pattern_type,
                    'synthesis_confidence': learning_confidence,
                    'synthesis_readiness': learning_confidence * 0.8
                }
            
            # Overall synthesis readiness
            readiness_scores = []
            if 'template_synthesis' in synthesis_capabilities:
                readiness_scores.append(synthesis_capabilities['template_synthesis']['synthesis_readiness'])
            if 'pattern_specific_synthesis' in synthesis_capabilities:
                readiness_scores.append(synthesis_capabilities['pattern_specific_synthesis']['synthesis_readiness'])
            
            overall_readiness = np.mean(readiness_scores) if readiness_scores else 0.5
            
            synthesis_capabilities['overall_synthesis_readiness'] = float(overall_readiness)
            
            return synthesis_capabilities
            
        except Exception as e:
            logger.error(f"Error assessing synthesis capabilities: {e}")
            return {'synthesis_readiness': 0.5}
    
    def _assess_universal_learning_quality(self, data: np.ndarray, 
                                         learned_patterns: Dict,
                                         geometric_analysis: Dict) -> Dict[str, Any]:
        """Assess universal learning quality"""
        try:
            quality_metrics = {}
            
            # Pattern detection quality
            dominant_pattern = learned_patterns.get('dominant')
            templates = learned_patterns.get('templates', [])
            
            if dominant_pattern:
                pattern_detection_quality = dominant_pattern.get('learning_confidence', 0.5)
            else:
                pattern_detection_quality = 0.3
            
            quality_metrics['pattern_detection_quality'] = float(pattern_detection_quality)
            
            # Template learning quality
            if templates:
                template_confidences = [t['confidence'] for t in templates]
                template_quality = np.mean(template_confidences)
            else:
                template_quality = 0.0
            
            quality_metrics['template_learning_quality'] = float(template_quality)
            
            # Geometric analysis quality
            if geometric_analysis.get('status') == 'complete':
                geometric_quality = 0.9
            elif geometric_analysis.get('status') == 'error':
                geometric_quality = 0.1
            else:
                geometric_quality = 0.5
            
            quality_metrics['geometric_analysis_quality'] = float(geometric_quality)
            
            # Overall quality (weighted average)
            weights = {
                'pattern_detection_quality': 0.5,
                'template_learning_quality': 0.3,
                'geometric_analysis_quality': 0.2
            }
            
            overall_quality = sum(quality_metrics[key] * weights[key] for key in weights.keys())
            quality_metrics['overall_quality'] = float(overall_quality)
            
            # Adaptability score based on pattern diversity
            pattern_count = len(templates)
            adaptability_score = min(1.0, pattern_count * 0.2 + 0.4)
            quality_metrics['adaptability_score'] = float(adaptability_score)
            
            # Complexity handling score
            if dominant_pattern:
                complexity_score = min(1.0, dominant_pattern.get('learning_confidence', 0.5) + 0.2)
            else:
                complexity_score = 0.3
            quality_metrics['complexity_score'] = float(complexity_score)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing universal learning quality: {e}")
            return {'overall_quality': 0.7, 'adaptability_score': 0.8, 'complexity_score': 0.6}
    
    def _analyze_data_characteristics(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze data characteristics"""
        return {
            'length': len(data),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'range': float(np.max(data) - np.min(data))
        }
    
    def _analyze_current_pattern_state(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze current pattern state"""
        detected_patterns = self._detect_all_pattern_types(data)
        return {
            'dominant_pattern': detected_patterns.get('dominant_pattern'),
            'all_patterns': detected_patterns.get('all_patterns', {}),
            'complexity_level': detected_patterns.get('complexity_level', 0.5)
        }
    
    def _select_optimal_synthesis_method(self, pattern_state: Dict[str, Any], 
                                       pattern_preferences: Optional[Dict] = None) -> str:
        """Select optimal synthesis method"""
        return 'geometric_synthesis'

    def _calculate_curvature(self, data: np.ndarray) -> float:
        """Calculate average curvature of the data"""
        try:
            if len(data) < 3:
                return 0.0
            
            second_derivative = np.diff(data, n=2)
            return float(np.mean(np.abs(second_derivative)))
            
        except Exception:
            return 0.0
    
    def _calculate_linearity(self, data: np.ndarray) -> float:
        """Calculate how linear the data is (0=nonlinear, 1=perfectly linear)"""
        try:
            if len(data) < 3:
                return 0.0
            
            # Fit linear trend
            x = np.arange(len(data))
            linear_fit = np.polyfit(x, data, 1)
            linear_pred = np.polyval(linear_fit, x)
            
            # Calculate R-squared for linearity
            ss_res = np.sum((data - linear_pred) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            
            if ss_tot == 0:
                return 1.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return float(max(0.0, min(1.0, r_squared)))
            
        except Exception:
            return 0.0
    
    def _estimate_periodicity(self, data: np.ndarray) -> Dict[str, Any]:
        """Estimate periodicity characteristics"""
        try:
            if len(data) < 6:
                return {'period': 0, 'strength': 0.0, 'confidence': 0.0}
            
            # Use autocorrelation to find period
            autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Normalize
            if autocorr[0] != 0:
                autocorr = autocorr / autocorr[0]
            
            # Find peaks in autocorrelation (excluding the first point)
            if len(autocorr) > 3:
                peaks, _ = find_peaks(autocorr[1:], height=0.3, distance=2)
                if len(peaks) > 0:
                    # First significant peak indicates period
                    period = peaks[0] + 1
                    strength = autocorr[period] if period < len(autocorr) else 0.0
                    return {
                        'period': int(period),
                        'strength': float(strength),
                        'confidence': float(min(1.0, strength * 2))
                    }
            
            return {'period': 0, 'strength': 0.0, 'confidence': 0.0}
            
        except Exception:
            return {'period': 0, 'strength': 0.0, 'confidence': 0.0}
    
    def _calculate_symmetry(self, data: np.ndarray) -> float:
        """Calculate symmetry score (0=asymmetric, 1=symmetric)"""
        try:
            if len(data) < 4:
                return 0.0
            
            # Check for symmetry around the center
            center_idx = len(data) // 2
            left_half = data[:center_idx]
            right_half = data[len(data):center_idx-1:-1]  # Reverse right half
            
            # Make them same length
            min_len = min(len(left_half), len(right_half))
            if min_len == 0:
                return 0.0
            
            left_half = left_half[-min_len:]
            right_half = right_half[:min_len]
            
            # Calculate correlation
            if np.std(left_half) > 1e-8 and np.std(right_half) > 1e-8:
                correlation = np.corrcoef(left_half, right_half)[0, 1]
                if not np.isnan(correlation):
                    return float(abs(correlation))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_harmonics(self, fft_data: np.ndarray, frequencies: np.ndarray) -> Dict[str, Any]:
        """Analyze harmonic content"""
        try:
            magnitude = np.abs(fft_data)
            
            # Find peaks in frequency domain
            peaks, _ = find_peaks(magnitude[1:len(fft_data)//2], height=np.max(magnitude) * 0.1)
            peaks = peaks + 1  # Adjust for offset
            
            harmonics = []
            for peak in peaks:
                harmonics.append({
                    'frequency': float(frequencies[peak]),
                    'magnitude': float(magnitude[peak]),
                    'phase': float(np.angle(fft_data[peak]))
                })
            
            return {
                'harmonic_count': len(harmonics),
                'harmonics': harmonics,
                'fundamental_frequency': harmonics[0]['frequency'] if harmonics else 0.0
            }
            
        except Exception:
            return {'harmonic_count': 0, 'harmonics': [], 'fundamental_frequency': 0.0}

    # Additional pattern detection helper methods
    def _analyze_plateau_characteristics(self, plateaus: List[Dict], data: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of detected plateaus"""
        try:
            if not plateaus:
                return {'flatness_score': 0.0, 'level_consistency': 0.0}
            
            # Calculate average flatness
            flatness_scores = [p['flatness_score'] for p in plateaus]
            avg_flatness = np.mean(flatness_scores)
            
            # Calculate level consistency (how similar plateau values are)
            plateau_values = [p['value'] for p in plateaus]
            if len(set(plateau_values)) <= 2:  # Square wave should have ~2 levels
                level_consistency = 0.9
            else:
                level_consistency = 0.5
            
            return {
                'flatness_score': float(avg_flatness),
                'level_consistency': float(level_consistency),
                'plateau_count': len(plateaus),
                'avg_plateau_length': float(np.mean([p['length'] for p in plateaus]))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing plateau characteristics: {e}")
            return {'flatness_score': 0.0, 'level_consistency': 0.0}
    
    def _analyze_transition_characteristics(self, transitions: List[Dict], data: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of detected transitions"""
        try:
            if not transitions:
                return {'sharpness_score': 0.0, 'transition_consistency': 0.0}
            
            # Calculate average sharpness
            sharpness_scores = [t['sharpness_score'] for t in transitions]
            avg_sharpness = np.mean(sharpness_scores)
            
            # Calculate transition consistency
            transition_strengths = [t['transition_strength'] for t in transitions]
            consistency = 1.0 / (1.0 + np.std(transition_strengths)) if len(transition_strengths) > 1 else 1.0
            
            return {
                'sharpness_score': float(avg_sharpness),
                'transition_consistency': float(consistency),
                'transition_count': len(transitions),
                'avg_amplitude_change': float(np.mean([t['amplitude_change'] for t in transitions]))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transition characteristics: {e}")
            return {'sharpness_score': 0.0, 'transition_consistency': 0.0}
    
    def _calculate_square_wave_periodicity(self, plateaus: List[Dict], data: np.ndarray) -> float:
        """Calculate square wave periodicity score"""
        try:
            if len(plateaus) < 2:
                return 0.0
            
            # Check if plateaus alternate between two main levels
            plateau_lengths = [p['length'] for p in plateaus]
            
            # Good square wave should have consistent plateau lengths
            length_consistency = 1.0 / (1.0 + np.std(plateau_lengths) / np.mean(plateau_lengths))
            
            # Check for alternating pattern
            plateau_values = [p['value'] for p in plateaus]
            unique_values = sorted(list(set(plateau_values)))
            
            if len(unique_values) == 2:
                # Perfect square wave alternation
                alternation_score = 0.9
            elif len(unique_values) <= 3:
                # Acceptable for square wave
                alternation_score = 0.7
            else:
                # Too many levels for square wave
                alternation_score = 0.3
            
            return float(length_consistency * 0.5 + alternation_score * 0.5)
            
        except Exception as e:
            logger.error(f"Error calculating square wave periodicity: {e}")
            return 0.0
    
    def _extract_amplitude_levels(self, plateaus: List[Dict], data: np.ndarray) -> List[float]:
        """Extract main amplitude levels from plateaus"""
        try:
            if not plateaus:
                return [float(np.min(data)), float(np.max(data))]
            
            # Get unique plateau values
            plateau_values = [p['value'] for p in plateaus]
            
            # Cluster similar values together
            unique_values = []
            tolerance = np.std(data) * 0.2
            
            for value in plateau_values:
                # Check if this value is close to any existing unique value
                is_unique = True
                for unique_val in unique_values:
                    if abs(value - unique_val) <= tolerance:
                        is_unique = False
                        break
                
                if is_unique:
                    unique_values.append(value)
            
            # Sort the levels
            unique_values.sort()
            
            # Ensure we have at least 2 levels for square wave
            if len(unique_values) < 2:
                unique_values = [float(np.min(data)), float(np.max(data))]
            
            return unique_values
            
        except Exception as e:
            logger.error(f"Error extracting amplitude levels: {e}")
            return [float(np.min(data)), float(np.max(data))]
    
    def _calculate_duty_cycle(self, plateaus: List[Dict]) -> float:
        """Calculate duty cycle from plateaus"""
        try:
            if len(plateaus) < 2:
                return 0.5
            
            # Assume first level is "high" and calculate its percentage
            levels = self._extract_amplitude_levels(plateaus, [])
            if len(levels) < 2:
                return 0.5
            
            high_level = levels[-1]  # Highest level
            high_duration = 0
            total_duration = 0
            
            for plateau in plateaus:
                total_duration += plateau['length']
                if abs(plateau['value'] - high_level) < abs(plateau['value'] - levels[0]):
                    high_duration += plateau['length']
            
            if total_duration > 0:
                return float(high_duration / total_duration)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating duty cycle: {e}")
            return 0.5
    
    def _create_square_wave_template(self, plateaus: List[Dict], transitions: List[Dict], data: np.ndarray) -> np.ndarray:
        """Create a template for the detected square wave pattern"""
        try:
            if not plateaus:
                return np.array([])
            
            # Get amplitude levels
            levels = self._extract_amplitude_levels(plateaus, data)
            if len(levels) < 2:
                return np.array([])
            
            # Estimate period from plateau analysis
            avg_plateau_length = int(np.mean([p['length'] for p in plateaus]))
            period = avg_plateau_length * 2  # Assuming 2 levels
            
            # Create template square wave
            template = []
            duty_cycle = self._calculate_duty_cycle(plateaus)
            high_duration = int(period * duty_cycle)
            low_duration = period - high_duration
            
            # High level
            template.extend([levels[-1]] * high_duration)
            # Low level
            template.extend([levels[0]] * low_duration)
            
            return np.array(template)
            
        except Exception as e:
            logger.error(f"Error creating square wave template: {e}")
            return np.array([])

    def _find_linear_segments(self, data: np.ndarray, peaks: np.ndarray, valleys: np.ndarray) -> List[Dict]:
        """Find linear segments between peaks and valleys"""
        try:
            segments = []
            
            # Combine and sort peaks and valleys
            extrema = np.concatenate([peaks, valleys])
            extrema = np.sort(extrema)
            
            # Find linear segments between consecutive extrema
            for i in range(len(extrema) - 1):
                start_idx = extrema[i]
                end_idx = extrema[i + 1]
                
                if end_idx - start_idx >= 3:  # Need at least 3 points for linear fit
                    segment_data = data[start_idx:end_idx + 1]
                    x = np.arange(len(segment_data))
                    
                    # Fit linear regression
                    slope, intercept = np.polyfit(x, segment_data, 1)
                    predicted = slope * x + intercept
                    
                    # Calculate R-squared for linearity
                    ss_res = np.sum((segment_data - predicted) ** 2)
                    ss_tot = np.sum((segment_data - np.mean(segment_data)) ** 2)
                    
                    if ss_tot > 0:
                        r_squared = 1 - (ss_res / ss_tot)
                    else:
                        r_squared = 1.0
                    
                    segments.append({
                        'start_idx': int(start_idx),
                        'end_idx': int(end_idx),
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'linearity_score': float(max(0.0, r_squared)),
                        'length': int(end_idx - start_idx)
                    })
            
            return segments
            
        except Exception as e:
            logger.error(f"Error finding linear segments: {e}")
    def _calculate_linearity_score(self, linear_segments: List[Dict], data: np.ndarray) -> float:
        """Calculate overall linearity score from segments"""
        try:
            if not linear_segments:
                return 0.0
            
            # Average the linearity scores of all segments weighted by length
            total_weight = 0
            weighted_score = 0
            
            for segment in linear_segments:
                length = segment['length']
                linearity = segment['linearity_score']
                
                weighted_score += linearity * length
                total_weight += length
            
            if total_weight > 0:
                return float(weighted_score / total_weight)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating linearity score: {e}")
            return 0.0
    
    def _calculate_peak_sharpness(self, peaks: np.ndarray, valleys: np.ndarray, data: np.ndarray) -> float:
        """Calculate sharpness of peaks and valleys for triangular waves"""
        try:
            all_extrema = np.concatenate([peaks, valleys])
            if len(all_extrema) == 0:
                return 0.0
            
            sharpness_scores = []
            
            for extremum in all_extrema:
                if extremum > 0 and extremum < len(data) - 1:
                    # Calculate local curvature at extremum
                    left_val = data[extremum - 1]
                    center_val = data[extremum]
                    right_val = data[extremum + 1]
                    
                    # Sharp peaks/valleys have high second derivative
                    second_deriv = abs(left_val - 2*center_val + right_val)
                    sharpness_scores.append(second_deriv)
            
            if sharpness_scores:
                # Normalize by data range
                data_range = np.max(data) - np.min(data)
                if data_range > 0:
                    avg_sharpness = np.mean(sharpness_scores) / data_range
                    return float(min(1.0, avg_sharpness * 10))  # Scale appropriately
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating peak sharpness: {e}")
            return 0.0
    
    def _calculate_triangular_symmetry(self, peaks: np.ndarray, valleys: np.ndarray, data: np.ndarray) -> float:
        """Calculate symmetry score for triangular waves"""
        try:
            all_extrema = np.concatenate([peaks, valleys])
            all_extrema = np.sort(all_extrema)
            
            if len(all_extrema) < 2:
                return 0.0
            
            symmetry_scores = []
            
            # Check symmetry of segments between consecutive extrema
            for i in range(len(all_extrema) - 1):
                start_idx = all_extrema[i]
                end_idx = all_extrema[i + 1]
                
                if end_idx - start_idx >= 4:  # Need enough points
                    segment = data[start_idx:end_idx + 1]
                    
                    # Check if segment is approximately linear (triangular segments should be)
                    x = np.arange(len(segment))
                    slope, _ = np.polyfit(x, segment, 1)
                    predicted = np.polyfit(x, segment, 1)
                    predicted_line = np.polyval(predicted, x)
                    
                    # Calculate R-squared for linearity
                    ss_res = np.sum((segment - predicted_line) ** 2)
                    ss_tot = np.sum((segment - np.mean(segment)) ** 2)
                    
                    if ss_tot > 0:
                        r_squared = 1 - (ss_res / ss_tot)
                        symmetry_scores.append(max(0.0, r_squared))
            
            if symmetry_scores:
                return float(np.mean(symmetry_scores))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating triangular symmetry: {e}")
            return 0.0
    
    def _calculate_triangular_periodicity(self, peaks: np.ndarray, valleys: np.ndarray) -> float:
        """Calculate periodicity score for triangular waves"""
        try:
            all_extrema = np.concatenate([peaks, valleys])
            all_extrema = np.sort(all_extrema)
            
            if len(all_extrema) < 3:
                return 0.0
            
            # Calculate intervals between consecutive extrema
            intervals = np.diff(all_extrema)
            
            if len(intervals) < 2:
                return 0.0
            
            # Good triangular waves have consistent intervals
            interval_consistency = 1.0 / (1.0 + np.std(intervals) / np.mean(intervals))
            
            # Also check for alternating peaks and valleys
            alternation_score = 0.8  # Assume reasonable alternation for now
            
            return float(interval_consistency * 0.7 + alternation_score * 0.3)
            
        except Exception as e:
            logger.error(f"Error calculating triangular periodicity: {e}")
            return 0.0

    # Additional synthesis helper methods
    def _estimate_square_wave_period(self, plateaus: List[Dict]) -> int:
        """Estimate period of square wave from plateaus"""
        try:
            if len(plateaus) < 2:
                return 10  # Default period
            
            # Calculate average plateau length and double it (high + low period)
            avg_length = np.mean([p['length'] for p in plateaus])
            return max(4, int(avg_length * 2))
            
        except Exception:
            return 10
    
    def _get_last_square_wave_level(self, data: np.ndarray, amplitude_levels: List[float]) -> float:
        """Get the last level of square wave to continue pattern"""
        try:
            if not amplitude_levels:
                return data[-1]
            
            last_value = data[-1]
            
            # Find closest amplitude level
            distances = [abs(last_value - level) for level in amplitude_levels]
            closest_idx = np.argmin(distances)
            
            return amplitude_levels[closest_idx]
            
        except Exception:
            return data[-1] if len(data) > 0 else 0.0
    
    def _estimate_triangular_wave_period(self, peaks: np.ndarray, valleys: np.ndarray) -> int:
        """Estimate period of triangular wave from peaks and valleys"""
        try:
            all_extrema = np.concatenate([peaks, valleys])
            all_extrema = np.sort(all_extrema)
            
            if len(all_extrema) < 2:
                return 20  # Default period
            
            # Average distance between extrema gives half-period
            intervals = np.diff(all_extrema)
            avg_half_period = np.mean(intervals)
            
            return max(6, int(avg_half_period * 2))
            
        except Exception:
            return 20
    
    def _estimate_triangular_amplitude(self, data: np.ndarray, peaks: np.ndarray, valleys: np.ndarray) -> float:
        """Estimate amplitude of triangular wave"""
        try:
            peak_values = data[peaks] if len(peaks) > 0 else []
            valley_values = data[valleys] if len(valleys) > 0 else []
            
            if len(peak_values) > 0 and len(valley_values) > 0:
                max_peak = np.max(peak_values)
                min_valley = np.min(valley_values)
                return (max_peak + abs(min_valley)) / 2
            elif len(peak_values) > 0:
                return np.mean(peak_values)
            elif len(valley_values) > 0:
                return abs(np.mean(valley_values))
            else:
                return np.std(data)
                
        except Exception:
            return np.std(data) if len(data) > 0 else 1.0

    def _adaptive_pattern_synthesis(self, data: np.ndarray, steps: int, pattern_state: Dict[str, Any]) -> np.ndarray:
        """Adaptive pattern synthesis when specific methods aren't available"""
        try:
            # Use pattern state information to guide synthesis
            dominant_pattern = pattern_state.get('dominant_pattern')
            
            if not dominant_pattern:
                return self._fallback_pattern_synthesis(data, steps)
            
            pattern_info = dominant_pattern[1]
            pattern_type = pattern_info.get('pattern_type', 'unknown')
            
            logger.info(f"Using adaptive synthesis for pattern type: {pattern_type}")
            
            # Basic pattern continuation based on detected characteristics
            if pattern_type in ['square_wave', 'step_function']:
                return self._adaptive_square_synthesis(data, steps, pattern_info)
            elif pattern_type in ['triangular_wave', 'sawtooth_wave']:
                return self._adaptive_triangular_synthesis(data, steps, pattern_info)
            elif pattern_type == 'sinusoidal_pattern':
                return self._adaptive_sinusoidal_synthesis(data, steps, pattern_info)
            else:
                return self._enhanced_fallback_synthesis(data, steps, pattern_info)
                
        except Exception as e:
            logger.error(f"Error in adaptive pattern synthesis: {e}")
            return self._fallback_pattern_synthesis(data, steps)
    
    def _adaptive_square_synthesis(self, data: np.ndarray, steps: int, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Adaptive synthesis for square-like patterns"""
        try:
            # Get basic characteristics
            data_range = np.max(data) - np.min(data)
            data_mean = np.mean(data)
            
            # Estimate two main levels
            high_level = data_mean + data_range * 0.4
            low_level = data_mean - data_range * 0.4
            
            # Estimate switching period from data length
            estimated_period = max(4, len(data) // 5)
            duty_cycle = 0.5
            
            predictions = []
            current_position = len(data)
            
            # Determine current level based on last few values
            recent_values = data[-min(5, len(data)):]
            if np.mean(recent_values) > data_mean:
                current_level = high_level
            else:
                current_level = low_level
            
            for step in range(steps):
                # Switch levels periodically
                if (current_position + step) % estimated_period == 0:
                    current_level = high_level if current_level == low_level else low_level
                
                predictions.append(current_level)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in adaptive square synthesis: {e}")
            return self._fallback_pattern_synthesis(data, steps)
    
    def _adaptive_triangular_synthesis(self, data: np.ndarray, steps: int, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Adaptive synthesis for triangular-like patterns"""
        try:
            # Estimate triangular wave parameters
            data_range = np.max(data) - np.min(data)
            data_mean = np.mean(data)
            amplitude = data_range / 2
            
            # Estimate period
            estimated_period = max(6, len(data) // 3)
            
            predictions = []
            current_position = len(data)
            last_value = data[-1]
            
            # Determine current direction (up or down)
            if len(data) >= 2:
                current_slope = data[-1] - data[-2]
                direction = 1 if current_slope >= 0 else -1
            else:
                direction = 1
            
            for step in range(steps):
                position_in_cycle = (current_position + step) % estimated_period
                
                # Triangular wave logic
                if position_in_cycle < estimated_period / 2:
                    # Rising edge
                    progress = position_in_cycle / (estimated_period / 2)
                    value = data_mean - amplitude + 2 * amplitude * progress
                else:
                    # Falling edge  
                    progress = (position_in_cycle - estimated_period / 2) / (estimated_period / 2)
                    value = data_mean + amplitude - 2 * amplitude * progress
                
                predictions.append(value)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in adaptive triangular synthesis: {e}")
            return self._fallback_pattern_synthesis(data, steps)
    
    def _adaptive_sinusoidal_synthesis(self, data: np.ndarray, steps: int, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Adaptive synthesis for sinusoidal patterns"""
        try:
            # Estimate sinusoidal parameters
            amplitude = np.std(data) * 1.4  # Approximate amplitude from std
            mean_val = np.mean(data)
            
            # Estimate frequency using simple period detection
            estimated_period = max(8, len(data) // 4)
            frequency = 2 * np.pi / estimated_period
            
            # Estimate phase from last few points
            if len(data) >= 3:
                last_vals = data[-3:]
                # Simple phase estimation
                phase = np.arctan2(last_vals[-1] - mean_val, amplitude) if amplitude > 0 else 0
            else:
                phase = 0
            
            predictions = []
            
            for step in range(1, steps + 1):
                t = len(data) + step
                value = mean_val + amplitude * np.sin(frequency * t + phase)
                predictions.append(value)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in adaptive sinusoidal synthesis: {e}")
            return self._fallback_pattern_synthesis(data, steps)
    
    def _enhanced_fallback_synthesis(self, data: np.ndarray, steps: int, pattern_info: Dict[str, Any]) -> np.ndarray:
        """Enhanced fallback synthesis with pattern awareness"""
        try:
            if len(data) < 2:
                return np.full(steps, data[-1] if len(data) > 0 else 0.0)
            
            # Analyze recent trend and periodicity
            recent_data = data[-min(20, len(data)):]
            
            # Check for periodic behavior
            if len(recent_data) >= 6:
                # Simple autocorrelation check
                half_len = len(recent_data) // 2
                first_half = recent_data[:half_len]
                second_half = recent_data[-half_len:]
                
                if len(first_half) == len(second_half) and np.std(first_half) > 1e-6 and np.std(second_half) > 1e-6:
                    correlation = np.corrcoef(first_half, second_half)[0, 1]
                    if not np.isnan(correlation) and correlation > 0.5:
                        # Repeat recent pattern
                        pattern_length = len(recent_data)
                        predictions = []
                        
                        for step in range(steps):
                            idx = step % pattern_length
                            predictions.append(recent_data[idx])
                        
                        return np.array(predictions)
            
            # Fallback to trend continuation
            trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            last_value = data[-1]
            
            predictions = []
            for step in range(1, steps + 1):
                predicted_value = last_value + trend * step
                predictions.append(predicted_value)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in enhanced fallback synthesis: {e}")
            return self._fallback_pattern_synthesis(data, steps)