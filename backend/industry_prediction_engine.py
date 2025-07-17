"""
Advanced Prediction Engine for Industry-Level Continuous Prediction
Sophisticated prediction algorithms that adapt to different pattern types
"""

import numpy as np
import pandas as pd
from scipy import signal, optimize, interpolate
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import interp1d, UnivariateSpline, CubicSpline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta
import json
from advanced_pattern_recognition import IndustryLevelPatternRecognition

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdvancedPredictionEngine:
    """
    Industry-level prediction engine that adapts to different pattern types
    and provides sophisticated continuous prediction capabilities
    """
    
    def __init__(self):
        self.pattern_recognizer = IndustryLevelPatternRecognition()
        self.prediction_history = []
        self.adaptive_models = {}
        self.pattern_specific_predictors = {
            'linear': LinearPatternPredictor(),
            'exponential': ExponentialPatternPredictor(),
            'sinusoidal': SinusoidalPatternPredictor(),
            'seasonal': SeasonalPatternPredictor(),
            'periodic': PeriodicPatternPredictor(),
            'trending': TrendingPatternPredictor(),
            'random_walk': RandomWalkPredictor(),
            'white_noise': WhiteNoisePredictor(),
            # Advanced pattern predictors
            'quadratic': QuadraticPatternPredictor(),
            'cubic': CubicPatternPredictor(),
            'polynomial': PolynomialPatternPredictor(),
            'spline': SplinePatternPredictor(),
            'custom_shape': CustomShapePredictor(),
            'composite': CompositePatternPredictor()
        }
        self.ensemble_predictor = EnsemblePredictor(self.pattern_specific_predictors)
        self.pattern_learning_engine = PatternLearningEngine()
        
    def generate_advanced_predictions(self, data: np.ndarray, 
                                    steps: int = 30,
                                    timestamps: Optional[pd.DatetimeIndex] = None,
                                    sampling_rate: float = 1.0,
                                    confidence_level: float = 0.95,
                                    adaptive_learning: bool = True) -> Dict[str, Any]:
        """
        Generate sophisticated predictions using pattern-aware algorithms
        
        Args:
            data: Historical time series data
            steps: Number of prediction steps
            timestamps: Optional timestamps for the data
            sampling_rate: Data sampling rate
            confidence_level: Confidence level for intervals
            adaptive_learning: Whether to use adaptive learning
            
        Returns:
            Comprehensive prediction results
        """
        try:
            # Step 1: Comprehensive pattern analysis
            pattern_analysis = self.pattern_recognizer.analyze_comprehensive_patterns(
                data, timestamps, sampling_rate
            )
            
            # Step 2: Select optimal prediction strategy
            prediction_strategy = self._select_prediction_strategy(pattern_analysis)
            
            # Step 3: Generate base predictions using multiple methods
            base_predictions = self._generate_base_predictions(
                data, steps, pattern_analysis, prediction_strategy
            )
            
            # Step 4: Apply adaptive corrections
            if adaptive_learning:
                corrected_predictions = self._apply_adaptive_corrections(
                    base_predictions, data, pattern_analysis
                )
            else:
                corrected_predictions = base_predictions
            
            # Step 5: Generate ensemble predictions
            ensemble_predictions = self.ensemble_predictor.predict(
                data, steps, pattern_analysis
            )
            
            # Step 6: Combine predictions optimally
            final_predictions = self._combine_predictions(
                corrected_predictions, ensemble_predictions, pattern_analysis
            )
            
            # Step 7: Apply smoothing and continuity constraints
            smoothed_predictions = self._apply_advanced_smoothing(
                final_predictions, data, pattern_analysis
            )
            
            # Step 8: Calculate confidence intervals
            confidence_intervals = self._calculate_advanced_confidence_intervals(
                smoothed_predictions, data, pattern_analysis, confidence_level
            )
            
            # Step 9: Generate quality metrics
            quality_metrics = self._calculate_prediction_quality(
                smoothed_predictions, data, pattern_analysis
            )
            
            # Step 10: Prepare comprehensive results
            results = {
                'predictions': smoothed_predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'pattern_analysis': pattern_analysis,
                'prediction_strategy': prediction_strategy,
                'quality_metrics': quality_metrics,
                'prediction_metadata': {
                    'steps': steps,
                    'confidence_level': confidence_level,
                    'sampling_rate': sampling_rate,
                    'adaptive_learning': adaptive_learning,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Store for adaptive learning
            self.prediction_history.append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in advanced prediction generation: {e}")
            return self._generate_fallback_predictions(data, steps)
    
    def generate_continuous_predictions(self, data: np.ndarray,
                                      previous_predictions: Optional[List[float]] = None,
                                      steps: int = 30,
                                      update_interval: int = 1,
                                      adaptive_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Generate continuous predictions with real-time adaptation
        
        Args:
            data: Current historical data
            previous_predictions: Previous predictions for comparison
            steps: Number of prediction steps
            update_interval: How often to update the model
            adaptive_threshold: Threshold for triggering model updates
            
        Returns:
            Continuous prediction results
        """
        try:
            # Check if model needs updating
            needs_update = self._needs_model_update(
                data, previous_predictions, adaptive_threshold
            )
            
            if needs_update or len(self.prediction_history) == 0:
                # Full pattern analysis and prediction
                results = self.generate_advanced_predictions(
                    data, steps, adaptive_learning=True
                )
                
                # Update adaptive models
                self._update_adaptive_models(data, results)
                
            else:
                # Use cached pattern analysis with incremental updates
                last_analysis = self.prediction_history[-1]['pattern_analysis']
                
                # Incremental pattern update
                updated_analysis = self._incremental_pattern_update(
                    data, last_analysis
                )
                
                # Generate predictions with updated analysis
                results = self._generate_incremental_predictions(
                    data, steps, updated_analysis
                )
            
            # Apply continuity constraints
            if previous_predictions:
                results['predictions'] = self._ensure_continuity(
                    results['predictions'], previous_predictions, data
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in continuous prediction generation: {e}")
            return self.generate_advanced_predictions(data, steps)
    
    def _select_prediction_strategy(self, pattern_analysis: Dict) -> Dict[str, Any]:
        """Select optimal prediction strategy based on pattern analysis"""
        try:
            pattern_class = pattern_analysis['pattern_classification']
            primary_pattern = pattern_class['primary_pattern']
            confidence = pattern_class['confidence']
            
            # Pattern-specific strategies
            strategy_map = {
                'linear': 'direct_linear',
                'exponential': 'exponential_fitting',
                'sinusoidal': 'harmonic_analysis',
                'seasonal': 'seasonal_decomposition',
                'periodic': 'frequency_domain',
                'trending': 'trend_extrapolation',
                'random_walk': 'stochastic_modeling',
                'white_noise': 'statistical_modeling'
            }
            
            primary_strategy = strategy_map.get(primary_pattern, 'ensemble_hybrid')
            
            # Determine if ensemble is needed
            use_ensemble = (
                confidence < 0.7 or 
                pattern_analysis['complexity_score'] > 0.7 or
                len(pattern_analysis['frequency_analysis']['dominant_frequencies']) > 2
            )
            
            return {
                'primary_strategy': primary_strategy,
                'use_ensemble': use_ensemble,
                'confidence_threshold': 0.7,
                'pattern_confidence': confidence,
                'fallback_strategy': 'ensemble_hybrid'
            }
            
        except Exception as e:
            return {'primary_strategy': 'ensemble_hybrid', 'use_ensemble': True}
    
    def _generate_base_predictions(self, data: np.ndarray, steps: int,
                                 pattern_analysis: Dict, strategy: Dict) -> np.ndarray:
        """Generate base predictions using the selected strategy"""
        try:
            primary_strategy = strategy['primary_strategy']
            
            if primary_strategy == 'direct_linear':
                return self._predict_linear(data, steps, pattern_analysis)
            elif primary_strategy == 'exponential_fitting':
                return self._predict_exponential(data, steps, pattern_analysis)
            elif primary_strategy == 'harmonic_analysis':
                return self._predict_harmonic(data, steps, pattern_analysis)
            elif primary_strategy == 'seasonal_decomposition':
                return self._predict_seasonal(data, steps, pattern_analysis)
            elif primary_strategy == 'frequency_domain':
                return self._predict_frequency_domain(data, steps, pattern_analysis)
            elif primary_strategy == 'trend_extrapolation':
                return self._predict_trend_extrapolation(data, steps, pattern_analysis)
            elif primary_strategy == 'stochastic_modeling':
                return self._predict_stochastic(data, steps, pattern_analysis)
            elif primary_strategy == 'statistical_modeling':
                return self._predict_statistical(data, steps, pattern_analysis)
            else:
                return self._predict_ensemble_hybrid(data, steps, pattern_analysis)
                
        except Exception as e:
            logger.warning(f"Base prediction failed: {e}")
            return self._predict_ensemble_hybrid(data, steps, pattern_analysis)
    
    def _predict_linear(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Linear prediction with trend analysis"""
        try:
            # Get trend information
            trend_analysis = pattern_analysis['trend_analysis']
            overall_trend = trend_analysis['overall_trend']
            
            # Fit linear model
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 1)
            
            # Generate predictions
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.polyval(coeffs, future_x)
            
            # Apply trend stabilization
            trend_strength = overall_trend.get('strength', 0.5)
            if trend_strength > 0.8:
                # Strong trend - continue as is
                pass
            else:
                # Weak trend - add mean reversion
                mean_reversion = (np.mean(data) - predictions) * 0.1
                predictions += mean_reversion
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1])
    
    def _predict_exponential(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Exponential prediction with growth rate analysis"""
        try:
            # Fit exponential model
            x = np.arange(len(data))
            
            # Transform to log space
            positive_data = np.abs(data) + 1e-10
            log_data = np.log(positive_data)
            
            # Fit linear in log space
            coeffs = np.polyfit(x, log_data, 1)
            
            # Generate predictions
            future_x = np.arange(len(data), len(data) + steps)
            log_predictions = np.polyval(coeffs, future_x)
            predictions = np.exp(log_predictions)
            
            # Adjust sign if necessary
            if np.mean(data) < 0:
                predictions = -predictions
            
            # Apply growth rate stabilization
            growth_rate = coeffs[0]
            if abs(growth_rate) > 0.1:  # High growth rate
                # Apply dampening
                dampening = np.exp(-0.1 * np.arange(steps))
                predictions = predictions * dampening
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1])
    
    def _predict_harmonic(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Harmonic prediction using frequency analysis"""
        try:
            # Get harmonic analysis
            harmonic_analysis = pattern_analysis['harmonic_analysis']
            harmonics = harmonic_analysis.get('harmonics', [])
            
            if not harmonics:
                return self._predict_linear(data, steps, pattern_analysis)
            
            # Reconstruct signal from harmonics
            x = np.arange(len(data))
            reconstructed = np.zeros_like(x, dtype=float)
            
            for harmonic in harmonics:
                freq = harmonic['frequency']
                power = harmonic['power']
                phase = harmonic.get('phase', 0)
                
                # Add harmonic component
                component = np.sqrt(power) * np.sin(2 * np.pi * freq * x + phase)
                reconstructed += component
            
            # Add trend component
            trend_analysis = pattern_analysis['trend_analysis']
            trend_slope = trend_analysis['overall_trend']['slope']
            trend_component = trend_slope * x
            reconstructed += trend_component
            
            # Adjust to match actual data
            offset = np.mean(data) - np.mean(reconstructed)
            reconstructed += offset
            
            # Generate future predictions
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.zeros(steps)
            
            for harmonic in harmonics:
                freq = harmonic['frequency']
                power = harmonic['power']
                phase = harmonic.get('phase', 0)
                
                # Add harmonic component
                component = np.sqrt(power) * np.sin(2 * np.pi * freq * future_x + phase)
                predictions += component
            
            # Add trend component
            trend_component = trend_slope * future_x
            predictions += trend_component + offset
            
            return predictions
            
        except Exception as e:
            return self._predict_linear(data, steps, pattern_analysis)
    
    def _predict_seasonal(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Seasonal prediction using decomposition"""
        try:
            # Get seasonal analysis
            seasonal_analysis = pattern_analysis['seasonal_analysis']
            seasonal_decomp = seasonal_analysis.get('seasonal_decomposition', {})
            
            if not seasonal_decomp.get('seasonal_component'):
                return self._predict_trend_extrapolation(data, steps, pattern_analysis)
            
            # Extract components
            seasonal_component = np.array(seasonal_decomp['seasonal_component'])
            trend_component = np.array(seasonal_decomp['trend_component'])
            period = seasonal_decomp.get('period', 12)
            
            # Extrapolate trend
            trend_slope = np.polyfit(np.arange(len(trend_component)), trend_component, 1)[0]
            last_trend = trend_component[-1]
            
            # Generate predictions
            predictions = []
            for i in range(steps):
                # Trend component
                trend_value = last_trend + trend_slope * (i + 1)
                
                # Seasonal component
                seasonal_index = i % period
                seasonal_value = seasonal_component[seasonal_index]
                
                # Combine
                prediction = trend_value + seasonal_value
                predictions.append(prediction)
            
            return np.array(predictions)
            
        except Exception as e:
            return self._predict_trend_extrapolation(data, steps, pattern_analysis)
    
    def _predict_frequency_domain(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Frequency domain prediction"""
        try:
            # Get frequency analysis
            frequency_analysis = pattern_analysis['frequency_analysis']
            dominant_frequencies = frequency_analysis.get('dominant_frequencies', [])
            
            if not dominant_frequencies:
                return self._predict_linear(data, steps, pattern_analysis)
            
            # Generate predictions using dominant frequencies
            x = np.arange(len(data))
            future_x = np.arange(len(data), len(data) + steps)
            
            # Fit sinusoidal components
            predictions = np.zeros(steps)
            
            for freq_info in dominant_frequencies[:3]:  # Top 3 frequencies
                freq = freq_info['frequency']
                power = freq_info['power']
                
                # Fit amplitude and phase
                component = np.sqrt(power) * np.sin(2 * np.pi * freq * x)
                
                # Find best fit amplitude and phase
                def fit_func(params):
                    amp, phase = params
                    fitted = amp * np.sin(2 * np.pi * freq * x + phase)
                    return np.sum((data - fitted) ** 2)
                
                try:
                    result = optimize.minimize(fit_func, [1.0, 0.0], method='L-BFGS-B')
                    amp, phase = result.x
                except:
                    amp, phase = 1.0, 0.0
                
                # Add to predictions
                component_pred = amp * np.sin(2 * np.pi * freq * future_x + phase)
                predictions += component_pred
            
            # Add trend component
            trend_slope = pattern_analysis['trend_analysis']['overall_trend']['slope']
            trend_component = trend_slope * np.arange(1, steps + 1)
            predictions += trend_component
            
            # Add mean
            predictions += np.mean(data)
            
            return predictions
            
        except Exception as e:
            return self._predict_linear(data, steps, pattern_analysis)
    
    def _predict_trend_extrapolation(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Advanced trend extrapolation"""
        try:
            # Get trend analysis
            trend_analysis = pattern_analysis['trend_analysis']
            multi_scale = trend_analysis.get('multi_scale_analysis', {})
            
            # Use multiple scales for robust trend estimation
            trends = []
            for scale_name, scale_data in multi_scale.items():
                if scale_data.get('best_fit') == 'linear':
                    trends.append(scale_data['linear_trend'])
                elif scale_data.get('best_fit') == 'exponential':
                    trends.append(scale_data['exponential_trend'])
                else:
                    trends.append(scale_data.get('linear_trend', 0))
            
            # Weighted average of trends
            if trends:
                weights = [0.4, 0.3, 0.2, 0.1][:len(trends)]
                weighted_trend = np.average(trends, weights=weights)
            else:
                weighted_trend = 0
            
            # Generate predictions
            last_value = data[-1]
            predictions = []
            
            for i in range(steps):
                # Basic trend extrapolation
                trend_value = last_value + weighted_trend * (i + 1)
                
                # Add trend acceleration
                acceleration = trend_analysis.get('trend_acceleration', 0)
                acceleration_value = 0.5 * acceleration * (i + 1) ** 2
                
                # Mean reversion
                mean_reversion = (np.mean(data) - trend_value) * 0.05
                
                # Combine
                prediction = trend_value + acceleration_value + mean_reversion
                predictions.append(prediction)
            
            return np.array(predictions)
            
        except Exception as e:
            return self._predict_linear(data, steps, pattern_analysis)
    
    def _predict_stochastic(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Stochastic prediction for random walk patterns"""
        try:
            # Calculate drift and volatility
            returns = np.diff(data)
            drift = np.mean(returns)
            volatility = np.std(returns)
            
            # Generate predictions using random walk model
            predictions = []
            current_value = data[-1]
            
            for i in range(steps):
                # Random walk step
                step = drift + np.random.normal(0, volatility)
                current_value += step
                predictions.append(current_value)
            
            return np.array(predictions)
            
        except Exception as e:
            return np.full(steps, data[-1])
    
    def _predict_statistical(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Statistical prediction for white noise patterns"""
        try:
            # For white noise, best prediction is the mean
            mean_value = np.mean(data)
            std_value = np.std(data)
            
            # Generate predictions with uncertainty
            predictions = np.random.normal(mean_value, std_value * 0.1, steps)
            
            return predictions
            
        except Exception as e:
            return np.full(steps, np.mean(data))
    
    def _predict_ensemble_hybrid(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Ensemble hybrid prediction combining multiple methods"""
        try:
            # Generate predictions using multiple methods
            methods = [
                ('linear', self._predict_linear),
                ('exponential', self._predict_exponential),
                ('harmonic', self._predict_harmonic),
                ('seasonal', self._predict_seasonal),
                ('trend', self._predict_trend_extrapolation)
            ]
            
            predictions = []
            weights = []
            
            for method_name, method_func in methods:
                try:
                    pred = method_func(data, steps, pattern_analysis)
                    predictions.append(pred)
                    
                    # Weight based on pattern fit
                    weight = self._calculate_method_weight(method_name, pattern_analysis)
                    weights.append(weight)
                    
                except Exception as e:
                    logger.warning(f"Method {method_name} failed: {e}")
                    continue
            
            if not predictions:
                return np.full(steps, data[-1])
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Weighted combination
            ensemble_pred = np.zeros(steps)
            for pred, weight in zip(predictions, weights):
                ensemble_pred += weight * pred
            
            return ensemble_pred
            
        except Exception as e:
            return np.full(steps, data[-1])
    
    def _calculate_method_weight(self, method_name: str, pattern_analysis: Dict) -> float:
        """Calculate weight for a prediction method based on pattern analysis"""
        try:
            pattern_class = pattern_analysis['pattern_classification']
            primary_pattern = pattern_class['primary_pattern']
            confidence = pattern_class['confidence']
            
            # Method-pattern compatibility
            compatibility_map = {
                'linear': {'linear': 1.0, 'trending': 0.8, 'exponential': 0.3},
                'exponential': {'exponential': 1.0, 'trending': 0.6, 'linear': 0.4},
                'harmonic': {'sinusoidal': 1.0, 'periodic': 0.9, 'seasonal': 0.7},
                'seasonal': {'seasonal': 1.0, 'periodic': 0.8, 'sinusoidal': 0.6},
                'trend': {'trending': 1.0, 'linear': 0.8, 'exponential': 0.7}
            }
            
            base_weight = compatibility_map.get(method_name, {}).get(primary_pattern, 0.5)
            
            # Adjust by confidence
            adjusted_weight = base_weight * confidence + 0.2 * (1 - confidence)
            
            return float(adjusted_weight)
            
        except Exception as e:
            return 0.5
    
    def _apply_adaptive_corrections(self, predictions: np.ndarray, data: np.ndarray,
                                  pattern_analysis: Dict) -> np.ndarray:
        """Apply adaptive corrections to predictions"""
        try:
            corrected_predictions = predictions.copy()
            
            # 1. Bias correction
            corrected_predictions = self._apply_bias_correction(
                corrected_predictions, data, pattern_analysis
            )
            
            # 2. Volatility correction
            corrected_predictions = self._apply_volatility_correction(
                corrected_predictions, data, pattern_analysis
            )
            
            # 3. Boundary correction
            corrected_predictions = self._apply_boundary_correction(
                corrected_predictions, data, pattern_analysis
            )
            
            # 4. Continuity correction
            corrected_predictions = self._apply_continuity_correction(
                corrected_predictions, data, pattern_analysis
            )
            
            return corrected_predictions
            
        except Exception as e:
            logger.warning(f"Adaptive corrections failed: {e}")
            return predictions
    
    def _apply_bias_correction(self, predictions: np.ndarray, data: np.ndarray,
                             pattern_analysis: Dict) -> np.ndarray:
        """Apply bias correction to predictions"""
        try:
            # Calculate historical bias
            historical_mean = np.mean(data)
            prediction_mean = np.mean(predictions)
            
            # Mean reversion factor
            mean_reversion_strength = 0.1
            bias_correction = (historical_mean - prediction_mean) * mean_reversion_strength
            
            # Apply gradual bias correction
            correction_weights = np.exp(-0.1 * np.arange(len(predictions)))
            bias_corrections = bias_correction * correction_weights
            
            return predictions + bias_corrections
            
        except Exception as e:
            return predictions
    
    def _apply_volatility_correction(self, predictions: np.ndarray, data: np.ndarray,
                                   pattern_analysis: Dict) -> np.ndarray:
        """Apply volatility correction to predictions"""
        try:
            # Historical volatility
            historical_volatility = np.std(data)
            
            # Prediction volatility
            prediction_volatility = np.std(np.diff(predictions))
            
            # Correct if volatility is too high
            if prediction_volatility > 2 * historical_volatility:
                smoothing_factor = historical_volatility / prediction_volatility
                
                # Apply smoothing
                for i in range(1, len(predictions)):
                    change = predictions[i] - predictions[i-1]
                    predictions[i] = predictions[i-1] + smoothing_factor * change
            
            return predictions
            
        except Exception as e:
            return predictions
    
    def _apply_boundary_correction(self, predictions: np.ndarray, data: np.ndarray,
                                 pattern_analysis: Dict) -> np.ndarray:
        """Apply boundary correction to keep predictions within reasonable bounds"""
        try:
            # Calculate reasonable bounds
            data_min = np.min(data)
            data_max = np.max(data)
            data_range = data_max - data_min
            
            # Expand bounds slightly
            lower_bound = data_min - 0.2 * data_range
            upper_bound = data_max + 0.2 * data_range
            
            # Apply soft clipping
            predictions = np.clip(predictions, lower_bound, upper_bound)
            
            return predictions
            
        except Exception as e:
            return predictions
    
    def _apply_continuity_correction(self, predictions: np.ndarray, data: np.ndarray,
                                   pattern_analysis: Dict) -> np.ndarray:
        """Apply continuity correction to ensure smooth transition"""
        try:
            if len(data) == 0:
                return predictions
            
            # Ensure smooth transition from last data point
            last_value = data[-1]
            first_prediction = predictions[0]
            
            # Calculate expected change based on recent trend
            recent_changes = np.diff(data[-5:]) if len(data) >= 5 else np.diff(data)
            expected_change = np.mean(recent_changes) if len(recent_changes) > 0 else 0
            
            # Calculate actual change
            actual_change = first_prediction - last_value
            
            # Correction factor
            correction = expected_change - actual_change
            
            # Apply diminishing correction
            correction_weights = np.exp(-0.2 * np.arange(len(predictions)))
            corrections = correction * correction_weights
            
            return predictions + corrections
            
        except Exception as e:
            return predictions
    
    def _combine_predictions(self, base_predictions: np.ndarray, 
                           ensemble_predictions: np.ndarray,
                           pattern_analysis: Dict) -> np.ndarray:
        """Combine different prediction sources optimally"""
        try:
            # Weight based on pattern confidence
            pattern_confidence = pattern_analysis['pattern_classification']['confidence']
            
            # Higher confidence in pattern -> higher weight for base predictions
            base_weight = 0.3 + 0.4 * pattern_confidence
            ensemble_weight = 1.0 - base_weight
            
            # Combine predictions
            combined_predictions = base_weight * base_predictions + ensemble_weight * ensemble_predictions
            
            return combined_predictions
            
        except Exception as e:
            return base_predictions
    
    def _apply_advanced_smoothing(self, predictions: np.ndarray, data: np.ndarray,
                                pattern_analysis: Dict) -> np.ndarray:
        """Apply advanced smoothing while preserving patterns"""
        try:
            if len(predictions) < 3:
                return predictions
            
            # Determine smoothing strength based on pattern characteristics
            pattern_strength = pattern_analysis.get('pattern_strength', 0.5)
            smoothing_strength = 0.3 * (1 - pattern_strength)  # Less smoothing for strong patterns
            
            if smoothing_strength < 0.1:
                return predictions
            
            # Apply Savitzky-Golay filter for pattern-preserving smoothing
            window_length = min(len(predictions), 5)
            if window_length % 2 == 0:
                window_length -= 1
            
            if window_length >= 3:
                smoothed = savgol_filter(predictions, window_length, 2)
                
                # Blend with original
                blended = (1 - smoothing_strength) * predictions + smoothing_strength * smoothed
                
                return blended
            else:
                return predictions
                
        except Exception as e:
            return predictions
    
    def _calculate_advanced_confidence_intervals(self, predictions: np.ndarray,
                                               data: np.ndarray,
                                               pattern_analysis: Dict,
                                               confidence_level: float) -> List[Dict]:
        """Calculate advanced confidence intervals"""
        try:
            confidence_intervals = []
            
            # Base uncertainty from historical data
            historical_std = np.std(data)
            
            # Pattern-based uncertainty adjustment
            pattern_confidence = pattern_analysis['pattern_classification']['confidence']
            uncertainty_multiplier = 2.0 - pattern_confidence
            
            # Calculate confidence intervals
            z_score = 1.96 if confidence_level == 0.95 else 2.576
            
            for i, pred in enumerate(predictions):
                # Uncertainty grows with prediction horizon
                horizon_factor = 1.0 + 0.1 * i
                
                # Calculate standard error
                std_error = historical_std * uncertainty_multiplier * horizon_factor
                
                # Confidence interval
                lower_bound = pred - z_score * std_error
                upper_bound = pred + z_score * std_error
                
                confidence_intervals.append({
                    'lower': float(lower_bound),
                    'upper': float(upper_bound),
                    'std_error': float(std_error),
                    'confidence_level': confidence_level
                })
            
            return confidence_intervals
            
        except Exception as e:
            return [{'lower': float(p), 'upper': float(p), 'std_error': 0.0} for p in predictions]
    
    def _calculate_prediction_quality(self, predictions: np.ndarray,
                                    data: np.ndarray,
                                    pattern_analysis: Dict) -> Dict[str, Any]:
        """Calculate prediction quality metrics"""
        try:
            # Pattern preservation score
            pattern_preservation = self._calculate_pattern_preservation_score(
                predictions, data, pattern_analysis
            )
            
            # Continuity score
            continuity_score = self._calculate_continuity_score(predictions, data)
            
            # Stability score
            stability_score = self._calculate_stability_score(predictions, data)
            
            # Realism score
            realism_score = self._calculate_realism_score(predictions, data)
            
            # Overall quality score
            overall_quality = (
                pattern_preservation * 0.3 +
                continuity_score * 0.25 +
                stability_score * 0.25 +
                realism_score * 0.2
            )
            
            return {
                'pattern_preservation_score': float(pattern_preservation),
                'continuity_score': float(continuity_score),
                'stability_score': float(stability_score),
                'realism_score': float(realism_score),
                'overall_quality_score': float(overall_quality),
                'prediction_volatility': float(np.std(np.diff(predictions))),
                'historical_volatility': float(np.std(np.diff(data)))
            }
            
        except Exception as e:
            return {'overall_quality_score': 0.5}
    
    def _calculate_pattern_preservation_score(self, predictions: np.ndarray,
                                            data: np.ndarray,
                                            pattern_analysis: Dict) -> float:
        """Calculate how well predictions preserve historical patterns"""
        try:
            # Statistical similarity
            historical_mean = np.mean(data)
            historical_std = np.std(data)
            
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            
            mean_similarity = 1 - abs(pred_mean - historical_mean) / (historical_std + 1e-10)
            std_similarity = 1 - abs(pred_std - historical_std) / (historical_std + 1e-10)
            
            # Trend similarity
            historical_trend = np.polyfit(range(len(data)), data, 1)[0]
            pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
            
            trend_similarity = 1 - abs(pred_trend - historical_trend) / (abs(historical_trend) + 1e-10)
            
            # Frequency similarity
            try:
                # Simple frequency domain comparison
                data_fft = np.fft.fft(data)
                pred_fft = np.fft.fft(predictions)
                
                # Compare dominant frequencies
                data_freqs = np.fft.fftfreq(len(data))
                pred_freqs = np.fft.fftfreq(len(predictions))
                
                data_dominant = np.argmax(np.abs(data_fft[1:len(data)//2])) + 1
                pred_dominant = np.argmax(np.abs(pred_fft[1:len(predictions)//2])) + 1
                
                freq_similarity = 1 - abs(data_freqs[data_dominant] - pred_freqs[pred_dominant])
                
            except:
                freq_similarity = 0.5
            
            # Combine scores
            preservation_score = (
                mean_similarity * 0.3 +
                std_similarity * 0.3 +
                trend_similarity * 0.2 +
                freq_similarity * 0.2
            )
            
            return max(0, min(1, preservation_score))
            
        except Exception as e:
            return 0.5
    
    def _calculate_continuity_score(self, predictions: np.ndarray, data: np.ndarray) -> float:
        """Calculate continuity score"""
        try:
            if len(data) == 0:
                return 0.5
            
            # Check for smooth transition
            last_historical = data[-1]
            first_prediction = predictions[0]
            
            # Calculate expected change based on recent trend
            recent_changes = np.diff(data[-5:]) if len(data) >= 5 else np.diff(data)
            expected_change = np.mean(recent_changes) if len(recent_changes) > 0 else 0
            
            actual_change = first_prediction - last_historical
            
            # Continuity score
            continuity = 1 - abs(actual_change - expected_change) / (np.std(data) + 1e-10)
            
            return max(0, min(1, continuity))
            
        except Exception as e:
            return 0.5
    
    def _calculate_stability_score(self, predictions: np.ndarray, data: np.ndarray) -> float:
        """Calculate stability score"""
        try:
            # Compare volatility
            historical_volatility = np.std(np.diff(data))
            prediction_volatility = np.std(np.diff(predictions))
            
            if historical_volatility == 0:
                return 1.0 if prediction_volatility == 0 else 0.0
            
            volatility_ratio = prediction_volatility / historical_volatility
            
            # Stability score (penalize excessive volatility)
            stability = 1 / (1 + abs(volatility_ratio - 1))
            
            return float(stability)
            
        except Exception as e:
            return 0.5
    
    def _calculate_realism_score(self, predictions: np.ndarray, data: np.ndarray) -> float:
        """Calculate realism score"""
        try:
            # Check if predictions are within reasonable bounds
            data_min = np.min(data)
            data_max = np.max(data)
            data_range = data_max - data_min
            
            # Expanded bounds
            lower_bound = data_min - 0.5 * data_range
            upper_bound = data_max + 0.5 * data_range
            
            # Count predictions within bounds
            within_bounds = np.sum((predictions >= lower_bound) & (predictions <= upper_bound))
            realism = within_bounds / len(predictions)
            
            return float(realism)
            
        except Exception as e:
            return 0.5
    
    def _needs_model_update(self, data: np.ndarray, previous_predictions: Optional[List[float]],
                          threshold: float) -> bool:
        """Determine if model needs updating"""
        try:
            if previous_predictions is None or len(self.prediction_history) == 0:
                return True
            
            # Compare recent data with previous predictions
            if len(data) > 0 and len(previous_predictions) > 0:
                # Calculate prediction error
                actual_value = data[-1]
                predicted_value = previous_predictions[0]  # First prediction
                
                error = abs(actual_value - predicted_value)
                relative_error = error / (np.std(data) + 1e-10)
                
                return relative_error > threshold
            
            return False
            
        except Exception as e:
            return True
    
    def _update_adaptive_models(self, data: np.ndarray, results: Dict):
        """Update adaptive models with new data"""
        try:
            # Store model parameters for adaptive learning
            pattern_type = results['pattern_analysis']['pattern_classification']['primary_pattern']
            
            model_params = {
                'timestamp': datetime.now().isoformat(),
                'pattern_type': pattern_type,
                'data_length': len(data),
                'quality_score': results['quality_metrics']['overall_quality_score']
            }
            
            # Update adaptive model for this pattern type
            if pattern_type not in self.adaptive_models:
                self.adaptive_models[pattern_type] = []
            
            self.adaptive_models[pattern_type].append(model_params)
            
            # Keep only recent models
            if len(self.adaptive_models[pattern_type]) > 10:
                self.adaptive_models[pattern_type] = self.adaptive_models[pattern_type][-10:]
                
        except Exception as e:
            logger.warning(f"Failed to update adaptive models: {e}")
    
    def _incremental_pattern_update(self, data: np.ndarray, last_analysis: Dict) -> Dict:
        """Incrementally update pattern analysis"""
        try:
            # For now, return last analysis
            # In a full implementation, this would update only changed parts
            return last_analysis
            
        except Exception as e:
            # Fall back to full analysis
            return self.pattern_recognizer.analyze_comprehensive_patterns(data)
    
    def _generate_incremental_predictions(self, data: np.ndarray, steps: int,
                                        analysis: Dict) -> Dict[str, Any]:
        """Generate predictions using incremental analysis"""
        try:
            # Use simplified prediction generation
            strategy = self._select_prediction_strategy(analysis)
            predictions = self._generate_base_predictions(data, steps, analysis, strategy)
            
            # Calculate basic confidence intervals
            confidence_intervals = self._calculate_advanced_confidence_intervals(
                predictions, data, analysis, 0.95
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_prediction_quality(predictions, data, analysis)
            
            return {
                'predictions': predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'pattern_analysis': analysis,
                'prediction_strategy': strategy,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            return self.generate_advanced_predictions(data, steps)
    
    def _ensure_continuity(self, predictions: List[float], 
                         previous_predictions: List[float],
                         data: np.ndarray) -> List[float]:
        """Ensure continuity between prediction cycles"""
        try:
            if not previous_predictions:
                return predictions
            
            # Calculate expected transition
            if len(data) > 0:
                last_actual = data[-1]
                expected_next = previous_predictions[0]
                
                # Smooth transition
                transition_error = predictions[0] - expected_next
                
                # Apply diminishing correction
                corrected_predictions = []
                for i, pred in enumerate(predictions):
                    correction = transition_error * np.exp(-0.1 * i)
                    corrected_predictions.append(pred - correction)
                
                return corrected_predictions
            
            return predictions
            
        except Exception as e:
            return predictions
    
    def _generate_fallback_predictions(self, data: np.ndarray, steps: int) -> Dict[str, Any]:
        """Generate fallback predictions when main method fails"""
        try:
            # Simple linear extrapolation
            if len(data) >= 2:
                x = np.arange(len(data))
                coeffs = np.polyfit(x, data, 1)
                future_x = np.arange(len(data), len(data) + steps)
                predictions = np.polyval(coeffs, future_x)
            else:
                predictions = np.full(steps, data[-1] if len(data) > 0 else 0)
            
            # Basic confidence intervals
            std_error = np.std(data) if len(data) > 1 else 1.0
            confidence_intervals = []
            
            for i, pred in enumerate(predictions):
                error = std_error * (1 + 0.1 * i)
                confidence_intervals.append({
                    'lower': float(pred - 1.96 * error),
                    'upper': float(pred + 1.96 * error),
                    'std_error': float(error)
                })
            
            return {
                'predictions': predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'pattern_analysis': {'pattern_classification': {'primary_pattern': 'fallback'}},
                'prediction_strategy': {'primary_strategy': 'fallback'},
                'quality_metrics': {'overall_quality_score': 0.3}
            }
            
        except Exception as e:
            return {
                'predictions': [0.0] * steps,
                'confidence_intervals': [],
                'pattern_analysis': {},
                'prediction_strategy': {},
                'quality_metrics': {'overall_quality_score': 0.1}
            }


# Pattern-specific predictor classes
class LinearPatternPredictor:
    """Specialized predictor for linear patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate linear predictions"""
        try:
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 1)
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.polyval(coeffs, future_x)
            
            # Add mean reversion for long-term stability
            mean_reversion = (np.mean(data) - predictions) * 0.05
            predictions += mean_reversion
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class ExponentialPatternPredictor:
    """Specialized predictor for exponential patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate exponential predictions"""
        try:
            # Fit exponential model
            x = np.arange(len(data))
            positive_data = np.abs(data) + 1e-10
            log_data = np.log(positive_data)
            
            coeffs = np.polyfit(x, log_data, 1)
            future_x = np.arange(len(data), len(data) + steps)
            log_predictions = np.polyval(coeffs, future_x)
            predictions = np.exp(log_predictions)
            
            # Adjust sign
            if np.mean(data) < 0:
                predictions = -predictions
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class SinusoidalPatternPredictor:
    """Specialized predictor for sinusoidal patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate sinusoidal predictions"""
        try:
            # Extract harmonics
            harmonics = pattern_analysis.get('harmonic_analysis', {}).get('harmonics', [])
            
            if not harmonics:
                # Fallback to basic sine fitting
                return self._fit_basic_sine(data, steps)
            
            # Generate predictions from harmonics
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.zeros(steps)
            
            for harmonic in harmonics:
                freq = harmonic['frequency']
                power = harmonic['power']
                phase = harmonic.get('phase', 0)
                
                component = np.sqrt(power) * np.sin(2 * np.pi * freq * future_x + phase)
                predictions += component
            
            # Add trend
            trend_slope = pattern_analysis.get('trend_analysis', {}).get('overall_trend', {}).get('slope', 0)
            predictions += trend_slope * np.arange(1, steps + 1)
            
            # Add offset
            predictions += np.mean(data)
            
            return predictions
            
        except Exception as e:
            return self._fit_basic_sine(data, steps)
    
    def _fit_basic_sine(self, data: np.ndarray, steps: int) -> np.ndarray:
        """Fit basic sine function to data"""
        try:
            x = np.arange(len(data))
            
            # Estimate parameters
            mean_val = np.mean(data)
            amplitude = np.std(data)
            
            # Estimate frequency using FFT
            fft_data = np.fft.fft(data - mean_val)
            freqs = np.fft.fftfreq(len(data))
            dominant_freq_idx = np.argmax(np.abs(fft_data[1:len(data)//2])) + 1
            frequency = freqs[dominant_freq_idx]
            
            # Estimate phase
            phase = np.angle(fft_data[dominant_freq_idx])
            
            # Generate predictions
            future_x = np.arange(len(data), len(data) + steps)
            predictions = amplitude * np.sin(2 * np.pi * frequency * future_x + phase) + mean_val
            
            return predictions
            
        except Exception as e:
            return np.full(steps, np.mean(data))


class SeasonalPatternPredictor:
    """Specialized predictor for seasonal patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate seasonal predictions"""
        try:
            seasonal_analysis = pattern_analysis.get('seasonal_analysis', {})
            seasonal_decomp = seasonal_analysis.get('seasonal_decomposition', {})
            
            if not seasonal_decomp:
                return np.full(steps, np.mean(data))
            
            seasonal_component = np.array(seasonal_decomp.get('seasonal_component', []))
            trend_component = np.array(seasonal_decomp.get('trend_component', []))
            period = seasonal_decomp.get('period', 12)
            
            # Extrapolate trend
            trend_slope = np.polyfit(np.arange(len(trend_component)), trend_component, 1)[0]
            last_trend = trend_component[-1]
            
            # Generate predictions
            predictions = []
            for i in range(steps):
                # Trend component
                trend_value = last_trend + trend_slope * (i + 1)
                
                # Seasonal component
                seasonal_index = i % period
                seasonal_value = seasonal_component[seasonal_index]
                
                # Combine
                prediction = trend_value + seasonal_value
                predictions.append(prediction)
            
            return np.array(predictions)
            
        except Exception as e:
            return np.full(steps, np.mean(data))


class PeriodicPatternPredictor:
    """Specialized predictor for periodic patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate periodic predictions"""
        try:
            # Use frequency domain analysis
            frequency_analysis = pattern_analysis.get('frequency_analysis', {})
            dominant_freqs = frequency_analysis.get('dominant_frequencies', [])
            
            if not dominant_freqs:
                return np.full(steps, np.mean(data))
            
            # Generate predictions using top frequencies
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.zeros(steps)
            
            for freq_info in dominant_freqs[:3]:  # Top 3 frequencies
                freq = freq_info['frequency']
                power = freq_info['power']
                
                # Simple sine wave
                component = np.sqrt(power) * np.sin(2 * np.pi * freq * future_x)
                predictions += component
            
            # Add mean
            predictions += np.mean(data)
            
            return predictions
            
        except Exception as e:
            return np.full(steps, np.mean(data))


class TrendingPatternPredictor:
    """Specialized predictor for trending patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate trending predictions"""
        try:
            # Use multi-scale trend analysis
            trend_analysis = pattern_analysis.get('trend_analysis', {})
            overall_trend = trend_analysis.get('overall_trend', {})
            
            trend_slope = overall_trend.get('slope', 0)
            trend_type = overall_trend.get('type', 'linear')
            
            if trend_type == 'exponential':
                return self._predict_exponential_trend(data, steps, trend_slope)
            else:
                return self._predict_linear_trend(data, steps, trend_slope)
                
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)
    
    def _predict_linear_trend(self, data: np.ndarray, steps: int, slope: float) -> np.ndarray:
        """Predict linear trend"""
        last_value = data[-1]
        predictions = []
        
        for i in range(steps):
            prediction = last_value + slope * (i + 1)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _predict_exponential_trend(self, data: np.ndarray, steps: int, slope: float) -> np.ndarray:
        """Predict exponential trend"""
        last_value = data[-1]
        predictions = []
        
        for i in range(steps):
            prediction = last_value * np.exp(slope * (i + 1))
            predictions.append(prediction)
        
        return np.array(predictions)


class RandomWalkPredictor:
    """Specialized predictor for random walk patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate random walk predictions"""
        try:
            # Calculate drift and volatility
            returns = np.diff(data)
            drift = np.mean(returns)
            volatility = np.std(returns)
            
            # Generate predictions
            predictions = []
            current_value = data[-1]
            
            for i in range(steps):
                # Random walk step with drift
                step = drift
                current_value += step
                predictions.append(current_value)
            
            return np.array(predictions)
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class WhiteNoisePredictor:
    """Specialized predictor for white noise patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate white noise predictions"""
        try:
            # For white noise, best prediction is the mean
            mean_value = np.mean(data)
            predictions = np.full(steps, mean_value)
            
            return predictions
            
        except Exception as e:
            return np.full(steps, np.mean(data) if len(data) > 0 else 0)


class QuadraticPatternPredictor:
    """Specialized predictor for quadratic patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate quadratic predictions"""
        try:
            # Fit quadratic model
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 2)
            
            # Generate predictions
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.polyval(coeffs, future_x)
            
            # Apply stabilization for long-term predictions
            if steps > 10:
                # Reduce quadratic effect for distant predictions
                stabilization_factor = np.exp(-0.05 * np.arange(steps))
                linear_trend = coeffs[1] * future_x + coeffs[2]
                predictions = linear_trend + (predictions - linear_trend) * stabilization_factor
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class CubicPatternPredictor:
    """Specialized predictor for cubic patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate cubic predictions"""
        try:
            # Fit cubic model
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 3)
            
            # Generate predictions
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.polyval(coeffs, future_x)
            
            # Apply strong stabilization for cubic patterns
            if steps > 5:
                # Reduce cubic effect significantly for distant predictions
                stabilization_factor = np.exp(-0.1 * np.arange(steps))
                linear_trend = coeffs[2] * future_x + coeffs[3]
                predictions = linear_trend + (predictions - linear_trend) * stabilization_factor
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class PolynomialPatternPredictor:
    """Specialized predictor for general polynomial patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate polynomial predictions"""
        try:
            # Determine optimal polynomial degree
            max_degree = min(5, len(data) - 1)
            best_degree = 1
            best_score = float('inf')
            
            for degree in range(1, max_degree + 1):
                try:
                    coeffs = np.polyfit(np.arange(len(data)), data, degree)
                    fitted = np.polyval(coeffs, np.arange(len(data)))
                    score = np.mean((data - fitted) ** 2)
                    
                    if score < best_score:
                        best_score = score
                        best_degree = degree
                except:
                    continue
            
            # Fit with best degree
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, best_degree)
            
            # Generate predictions
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.polyval(coeffs, future_x)
            
            # Apply degree-dependent stabilization
            if best_degree > 2:
                stabilization_factor = np.exp(-0.05 * best_degree * np.arange(steps))
                linear_trend = coeffs[-2] * future_x + coeffs[-1]
                predictions = linear_trend + (predictions - linear_trend) * stabilization_factor
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class SplinePatternPredictor:
    """Specialized predictor for spline-based patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate spline-based predictions"""
        try:
            x = np.arange(len(data))
            
            # Use cubic spline for smooth interpolation
            spline = CubicSpline(x, data)
            
            # Extrapolate using spline derivative at the end
            last_derivative = spline.derivative()(x[-1])
            last_value = data[-1]
            
            # Generate predictions using linear extrapolation with spline derivative
            predictions = []
            for i in range(steps):
                prediction = last_value + last_derivative * (i + 1)
                predictions.append(prediction)
            
            # Apply smoothing to reduce derivative discontinuities
            predictions = np.array(predictions)
            if len(predictions) > 3:
                predictions = savgol_filter(predictions, min(5, len(predictions)), 2)
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class CustomShapePredictor:
    """Specialized predictor for custom shape patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate custom shape predictions"""
        try:
            # Analyze local patterns in the data
            window_size = min(10, len(data) // 3)
            if window_size < 3:
                return np.full(steps, data[-1])
            
            # Extract recent pattern
            recent_pattern = data[-window_size:]
            
            # Calculate pattern characteristics
            pattern_mean = np.mean(recent_pattern)
            pattern_trend = np.polyfit(np.arange(window_size), recent_pattern, 1)[0]
            pattern_volatility = np.std(recent_pattern)
            
            # Generate predictions based on pattern repetition
            predictions = []
            for i in range(steps):
                # Base prediction from trend
                base_pred = data[-1] + pattern_trend * (i + 1)
                
                # Add pattern-based variation
                pattern_index = i % window_size
                pattern_variation = recent_pattern[pattern_index] - pattern_mean
                
                # Combine with diminishing pattern influence
                pattern_weight = np.exp(-0.1 * i)
                prediction = base_pred + pattern_variation * pattern_weight
                
                predictions.append(prediction)
            
            return np.array(predictions)
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class CompositePatternPredictor:
    """Specialized predictor for composite patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate composite pattern predictions"""
        try:
            # Decompose into multiple components
            components = self._decompose_pattern(data)
            
            # Predict each component separately
            predictions = np.zeros(steps)
            
            for component_name, component_data in components.items():
                component_pred = self._predict_component(
                    component_data, steps, component_name, pattern_analysis
                )
                predictions += component_pred
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)
    
    def _decompose_pattern(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose data into multiple components"""
        try:
            components = {}
            
            # Trend component
            x = np.arange(len(data))
            trend_coeffs = np.polyfit(x, data, 1)
            trend_component = np.polyval(trend_coeffs, x)
            components['trend'] = trend_component
            
            # Residual after trend removal
            residual = data - trend_component
            
            # Periodic component (if detectable)
            if len(data) > 10:
                # Simple periodic detection using autocorrelation
                autocorr = np.correlate(residual, residual, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find peaks in autocorrelation
                peaks = []
                for i in range(1, min(len(autocorr), len(data)//2)):
                    if autocorr[i] > 0.3 * autocorr[0]:
                        peaks.append(i)
                
                if peaks:
                    # Use the first significant peak as period
                    period = peaks[0]
                    periodic_component = self._extract_periodic_component(residual, period)
                    components['periodic'] = periodic_component
                    residual = residual - periodic_component
            
            # Remaining residual (noise)
            components['noise'] = residual
            
            return components
            
        except Exception as e:
            return {'trend': data, 'noise': np.zeros_like(data)}
    
    def _extract_periodic_component(self, data: np.ndarray, period: int) -> np.ndarray:
        """Extract periodic component from data"""
        try:
            # Create periodic pattern by averaging over periods
            periodic_pattern = np.zeros(period)
            count = np.zeros(period)
            
            for i in range(len(data)):
                idx = i % period
                periodic_pattern[idx] += data[i]
                count[idx] += 1
            
            # Avoid division by zero
            count[count == 0] = 1
            periodic_pattern = periodic_pattern / count
            
            # Extend pattern to match data length
            periodic_component = np.tile(periodic_pattern, len(data) // period + 1)[:len(data)]
            
            return periodic_component
            
        except Exception as e:
            return np.zeros_like(data)
    
    def _predict_component(self, component_data: np.ndarray, steps: int, 
                          component_name: str, pattern_analysis: Dict) -> np.ndarray:
        """Predict individual component"""
        try:
            if component_name == 'trend':
                # Linear extrapolation for trend
                x = np.arange(len(component_data))
                trend_coeffs = np.polyfit(x, component_data, 1)
                future_x = np.arange(len(component_data), len(component_data) + steps)
                return np.polyval(trend_coeffs, future_x)
            
            elif component_name == 'periodic':
                # Repeat the periodic pattern
                if len(component_data) > 0:
                    period = self._estimate_period(component_data)
                    if period > 0:
                        pattern = component_data[-period:]
                        return np.tile(pattern, steps // period + 1)[:steps]
                return np.zeros(steps)
            
            elif component_name == 'noise':
                # For noise, use mean (best predictor)
                return np.full(steps, np.mean(component_data))
            
            else:
                # Default: use last value
                return np.full(steps, component_data[-1] if len(component_data) > 0 else 0)
                
        except Exception as e:
            return np.zeros(steps)
    
    def _estimate_period(self, data: np.ndarray) -> int:
        """Estimate period of periodic data"""
        try:
            # Use autocorrelation to find period
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first significant peak
            for i in range(1, min(len(autocorr), len(data)//2)):
                if autocorr[i] > 0.5 * autocorr[0]:
                    return i
            
            return len(data) // 4  # Default fallback
            
        except Exception as e:
            return 10  # Default fallback


class PatternLearningEngine:
    """Engine for learning and adapting to patterns dynamically"""
    
    def __init__(self):
        self.learned_patterns = {}
        self.pattern_performance = {}
        self.learning_history = []
        
    def learn_pattern(self, data: np.ndarray, pattern_type: str, 
                     performance_metrics: Dict) -> None:
        """Learn from a pattern instance"""
        try:
            pattern_signature = self._extract_pattern_signature(data)
            
            if pattern_type not in self.learned_patterns:
                self.learned_patterns[pattern_type] = []
                self.pattern_performance[pattern_type] = []
            
            self.learned_patterns[pattern_type].append(pattern_signature)
            self.pattern_performance[pattern_type].append(performance_metrics)
            
            # Keep only recent patterns (last 50)
            if len(self.learned_patterns[pattern_type]) > 50:
                self.learned_patterns[pattern_type] = self.learned_patterns[pattern_type][-50:]
                self.pattern_performance[pattern_type] = self.pattern_performance[pattern_type][-50:]
            
            self.learning_history.append({
                'timestamp': datetime.now(),
                'pattern_type': pattern_type,
                'performance': performance_metrics
            })
            
        except Exception as e:
            logger.error(f"Error learning pattern: {e}")
    
    def _extract_pattern_signature(self, data: np.ndarray) -> Dict[str, float]:
        """Extract key characteristics of a pattern"""
        try:
            signature = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'trend': float(np.polyfit(np.arange(len(data)), data, 1)[0]),
                'autocorr_1': float(np.corrcoef(data[:-1], data[1:])[0, 1]) if len(data) > 1 else 0.0,
                'skewness': float(np.mean(((data - np.mean(data)) / np.std(data)) ** 3)),
                'kurtosis': float(np.mean(((data - np.mean(data)) / np.std(data)) ** 4)) - 3
            }
            
            return signature
            
        except Exception as e:
            return {'mean': 0.0, 'std': 1.0, 'trend': 0.0, 
                   'autocorr_1': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}
    
    def get_pattern_adaptation_suggestions(self, data: np.ndarray, 
                                         pattern_type: str) -> Dict[str, Any]:
        """Get suggestions for adapting to a specific pattern"""
        try:
            if pattern_type not in self.learned_patterns:
                return {'smoothing_factor': 0.5, 'prediction_horizon': 30, 
                       'confidence_level': 0.8}
            
            # Analyze performance of similar patterns
            patterns = self.learned_patterns[pattern_type]
            performances = self.pattern_performance[pattern_type]
            
            # Find patterns most similar to current data
            current_signature = self._extract_pattern_signature(data)
            similarities = []
            
            for i, pattern in enumerate(patterns):
                similarity = self._calculate_signature_similarity(current_signature, pattern)
                similarities.append((similarity, performances[i]))
            
            # Sort by similarity and use top performers
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_similarities = similarities[:5]
            
            # Extract adaptation suggestions
            suggestions = {
                'smoothing_factor': np.mean([s[1].get('smoothing_factor', 0.5) 
                                           for s in top_similarities]),
                'prediction_horizon': int(np.mean([s[1].get('prediction_horizon', 30) 
                                                 for s in top_similarities])),
                'confidence_level': np.mean([s[1].get('confidence_level', 0.8) 
                                           for s in top_similarities])
            }
            
            return suggestions
            
        except Exception as e:
            return {'smoothing_factor': 0.5, 'prediction_horizon': 30, 
                   'confidence_level': 0.8}
    
    def _calculate_signature_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between two pattern signatures"""
        try:
            # Normalize and calculate weighted similarity
            weights = {
                'mean': 0.2,
                'std': 0.2,
                'trend': 0.3,
                'autocorr_1': 0.2,
                'skewness': 0.05,
                'kurtosis': 0.05
            }
            
            similarity = 0.0
            total_weight = 0.0
            
            for key in weights:
                if key in sig1 and key in sig2:
                    # Calculate normalized difference
                    diff = abs(sig1[key] - sig2[key])
                    max_val = max(abs(sig1[key]), abs(sig2[key]), 1.0)
                    norm_diff = diff / max_val
                    
                    # Convert to similarity (0-1)
                    key_similarity = 1.0 / (1.0 + norm_diff)
                    
                    similarity += weights[key] * key_similarity
                    total_weight += weights[key]
            
            return similarity / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            return 0.0


class QuadraticPatternPredictor:
    """Specialized predictor for quadratic patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate quadratic predictions"""
        try:
            # Fit quadratic model
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 2)
            
            # Generate predictions
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.polyval(coeffs, future_x)
            
            # Apply stabilization for long-term predictions
            if steps > 10:
                # Reduce quadratic effect for distant predictions
                stabilization_factor = np.exp(-0.05 * np.arange(steps))
                linear_trend = coeffs[1] * future_x + coeffs[2]
                predictions = linear_trend + (predictions - linear_trend) * stabilization_factor
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class CubicPatternPredictor:
    """Specialized predictor for cubic patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate cubic predictions"""
        try:
            # Fit cubic model
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 3)
            
            # Generate predictions
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.polyval(coeffs, future_x)
            
            # Apply strong stabilization for cubic patterns
            if steps > 5:
                # Reduce cubic effect significantly for distant predictions
                stabilization_factor = np.exp(-0.1 * np.arange(steps))
                linear_trend = coeffs[2] * future_x + coeffs[3]
                predictions = linear_trend + (predictions - linear_trend) * stabilization_factor
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class PolynomialPatternPredictor:
    """Specialized predictor for general polynomial patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate polynomial predictions"""
        try:
            # Determine optimal polynomial degree
            max_degree = min(5, len(data) - 1)
            best_degree = 1
            best_score = float('inf')
            
            for degree in range(1, max_degree + 1):
                try:
                    coeffs = np.polyfit(np.arange(len(data)), data, degree)
                    fitted = np.polyval(coeffs, np.arange(len(data)))
                    score = np.mean((data - fitted) ** 2)
                    
                    if score < best_score:
                        best_score = score
                        best_degree = degree
                except:
                    continue
            
            # Fit with best degree
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, best_degree)
            
            # Generate predictions
            future_x = np.arange(len(data), len(data) + steps)
            predictions = np.polyval(coeffs, future_x)
            
            # Apply degree-dependent stabilization
            if best_degree > 2:
                stabilization_factor = np.exp(-0.05 * best_degree * np.arange(steps))
                linear_trend = coeffs[-2] * future_x + coeffs[-1]
                predictions = linear_trend + (predictions - linear_trend) * stabilization_factor
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class SplinePatternPredictor:
    """Specialized predictor for spline-based patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate spline-based predictions"""
        try:
            x = np.arange(len(data))
            
            # Use cubic spline for smooth interpolation
            spline = CubicSpline(x, data)
            
            # Extrapolate using spline derivative at the end
            last_derivative = spline.derivative()(x[-1])
            last_value = data[-1]
            
            # Generate predictions using linear extrapolation with spline derivative
            predictions = []
            for i in range(steps):
                prediction = last_value + last_derivative * (i + 1)
                predictions.append(prediction)
            
            # Apply smoothing to reduce derivative discontinuities
            predictions = np.array(predictions)
            if len(predictions) > 3:
                predictions = savgol_filter(predictions, min(5, len(predictions)), 2)
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class CustomShapePredictor:
    """Specialized predictor for custom shape patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate custom shape predictions"""
        try:
            # Analyze local patterns in the data
            window_size = min(10, len(data) // 3)
            if window_size < 3:
                return np.full(steps, data[-1])
            
            # Extract recent pattern
            recent_pattern = data[-window_size:]
            
            # Calculate pattern characteristics
            pattern_mean = np.mean(recent_pattern)
            pattern_trend = np.polyfit(np.arange(window_size), recent_pattern, 1)[0]
            pattern_volatility = np.std(recent_pattern)
            
            # Generate predictions based on pattern repetition
            predictions = []
            for i in range(steps):
                # Base prediction from trend
                base_pred = data[-1] + pattern_trend * (i + 1)
                
                # Add pattern-based variation
                pattern_index = i % window_size
                pattern_variation = recent_pattern[pattern_index] - pattern_mean
                
                # Combine with diminishing pattern influence
                pattern_weight = np.exp(-0.1 * i)
                prediction = base_pred + pattern_variation * pattern_weight
                
                predictions.append(prediction)
            
            return np.array(predictions)
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)


class CompositePatternPredictor:
    """Specialized predictor for composite patterns"""
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate composite pattern predictions"""
        try:
            # Decompose into multiple components
            components = self._decompose_pattern(data)
            
            # Predict each component separately
            predictions = np.zeros(steps)
            
            for component_name, component_data in components.items():
                component_pred = self._predict_component(
                    component_data, steps, component_name, pattern_analysis
                )
                predictions += component_pred
            
            return predictions
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)
    
    def _decompose_pattern(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose data into multiple components"""
        try:
            components = {}
            
            # Trend component
            x = np.arange(len(data))
            trend_coeffs = np.polyfit(x, data, 1)
            trend_component = np.polyval(trend_coeffs, x)
            components['trend'] = trend_component
            
            # Residual after trend removal
            residual = data - trend_component
            
            # Periodic component (if detectable)
            if len(data) > 10:
                # Simple periodic detection using autocorrelation
                autocorr = np.correlate(residual, residual, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find peaks in autocorrelation
                if len(autocorr) > 3:
                    peaks = []
                    for i in range(1, len(autocorr)-1):
                        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                            peaks.append(i)
                    
                    if peaks:
                        period = peaks[0]
                        if period < len(data) // 2:
                            # Extract periodic component
                            periodic_component = np.zeros_like(residual)
                            for i in range(len(residual)):
                                periodic_component[i] = residual[i % period]
                            components['periodic'] = periodic_component
                            
                            # Update residual
                            residual = residual - periodic_component
            
            # Remaining noise component
            components['noise'] = residual
            
            return components
            
        except Exception as e:
            return {'trend': data}
    
    def _predict_component(self, component_data: np.ndarray, steps: int, 
                          component_name: str, pattern_analysis: Dict) -> np.ndarray:
        """Predict individual component"""
        try:
            if component_name == 'trend':
                # Linear extrapolation for trend
                x = np.arange(len(component_data))
                coeffs = np.polyfit(x, component_data, 1)
                future_x = np.arange(len(component_data), len(component_data) + steps)
                return np.polyval(coeffs, future_x)
                
            elif component_name == 'periodic':
                # Repeat periodic pattern
                period = len(component_data)
                predictions = []
                for i in range(steps):
                    predictions.append(component_data[i % period])
                return np.array(predictions)
                
            elif component_name == 'noise':
                # For noise, predict mean with small random variation
                mean_noise = np.mean(component_data)
                std_noise = np.std(component_data)
                return np.random.normal(mean_noise, std_noise * 0.1, steps)
                
            else:
                return np.zeros(steps)
                
        except Exception as e:
            return np.zeros(steps)


class PatternLearningEngine:
    """Engine for learning and adapting to new patterns"""
    
    def __init__(self):
        self.learned_patterns = {}
        self.pattern_history = []
        self.adaptation_threshold = 0.1
    
    def learn_pattern(self, data: np.ndarray, pattern_type: str, 
                     performance_metrics: Dict) -> None:
        """Learn from a new pattern instance"""
        try:
            pattern_signature = self._extract_pattern_signature(data)
            
            if pattern_type not in self.learned_patterns:
                self.learned_patterns[pattern_type] = []
            
            pattern_info = {
                'signature': pattern_signature,
                'data_length': len(data),
                'performance': performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            self.learned_patterns[pattern_type].append(pattern_info)
            
            # Keep only recent patterns
            if len(self.learned_patterns[pattern_type]) > 20:
                self.learned_patterns[pattern_type] = self.learned_patterns[pattern_type][-20:]
                
        except Exception as e:
            logger.warning(f"Failed to learn pattern: {e}")
    
    def _extract_pattern_signature(self, data: np.ndarray) -> Dict:
        """Extract signature features from data"""
        try:
            signature = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'trend': float(np.polyfit(np.arange(len(data)), data, 1)[0]),
                'autocorr_1': float(np.corrcoef(data[:-1], data[1:])[0, 1]) if len(data) > 1 else 0,
                'skewness': float(self._calculate_skewness(data)),
                'kurtosis': float(self._calculate_kurtosis(data))
            }
            return signature
            
        except Exception as e:
            return {}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        except:
            return 0
    
    def adapt_predictor(self, pattern_type: str, current_performance: Dict) -> Dict:
        """Adapt predictor based on learned patterns"""
        try:
            if pattern_type not in self.learned_patterns:
                return {}
            
            # Analyze historical performance
            historical_patterns = self.learned_patterns[pattern_type]
            
            # Calculate adaptation suggestions
            adaptations = {
                'suggested_smoothing': self._suggest_smoothing(historical_patterns),
                'suggested_horizon': self._suggest_horizon(historical_patterns),
                'confidence_adjustment': self._suggest_confidence_adjustment(historical_patterns)
            }
            
            return adaptations
            
        except Exception as e:
            return {}
    
    def _suggest_smoothing(self, patterns: List[Dict]) -> float:
        """Suggest optimal smoothing parameter"""
        try:
            # Analyze volatility patterns
            volatilities = []
            for pattern in patterns[-10:]:  # Recent patterns
                perf = pattern.get('performance', {})
                volatility = perf.get('prediction_volatility', 0.1)
                volatilities.append(volatility)
            
            avg_volatility = np.mean(volatilities)
            
            # Higher volatility -> more smoothing
            suggested_smoothing = min(0.5, avg_volatility * 2)
            return float(suggested_smoothing)
            
        except:
            return 0.2
    
    def _suggest_horizon(self, patterns: List[Dict]) -> int:
        """Suggest optimal prediction horizon"""
        try:
            # Analyze performance degradation over horizon
            quality_scores = []
            for pattern in patterns[-10:]:
                perf = pattern.get('performance', {})
                quality = perf.get('overall_quality_score', 0.5)
                quality_scores.append(quality)
            
            avg_quality = np.mean(quality_scores)
            
            # Higher quality -> longer horizon
            if avg_quality > 0.8:
                return 50
            elif avg_quality > 0.6:
                return 30
            else:
                return 20
                
        except:
            return 30
    
    def _suggest_confidence_adjustment(self, patterns: List[Dict]) -> float:
        """Suggest confidence interval adjustment"""
        try:
            # Analyze prediction accuracy
            accuracies = []
            for pattern in patterns[-10:]:
                perf = pattern.get('performance', {})
                continuity = perf.get('continuity_score', 0.5)
                realism = perf.get('realism_score', 0.5)
                accuracy = (continuity + realism) / 2
                accuracies.append(accuracy)
            
            avg_accuracy = np.mean(accuracies)
            
            # Lower accuracy -> wider confidence intervals
            adjustment = 2.0 - avg_accuracy
            return float(max(0.5, min(2.0, adjustment)))
            
        except:
            return 1.0


class EnsemblePredictor:
    """Ensemble predictor combining multiple specialized predictors"""
    
    def __init__(self, predictors: Dict):
        self.predictors = predictors
    
    def predict(self, data: np.ndarray, steps: int, pattern_analysis: Dict) -> np.ndarray:
        """Generate ensemble predictions"""
        try:
            # Get pattern classification
            pattern_class = pattern_analysis.get('pattern_classification', {})
            primary_pattern = pattern_class.get('primary_pattern', 'linear')
            pattern_scores = pattern_class.get('pattern_scores', {})
            
            # Generate predictions from all predictors
            predictions = {}
            weights = {}
            
            for pattern_name, predictor in self.predictors.items():
                try:
                    pred = predictor.predict(data, steps, pattern_analysis)
                    predictions[pattern_name] = pred
                    weights[pattern_name] = pattern_scores.get(pattern_name, 0.1)
                except Exception as e:
                    logger.warning(f"Predictor {pattern_name} failed: {e}")
                    continue
            
            if not predictions:
                return np.full(steps, data[-1] if len(data) > 0 else 0)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for key in weights:
                    weights[key] /= total_weight
            
            # Weighted ensemble
            ensemble_pred = np.zeros(steps)
            for pattern_name, pred in predictions.items():
                ensemble_pred += weights[pattern_name] * pred
            
            return ensemble_pred
            
        except Exception as e:
            return np.full(steps, data[-1] if len(data) > 0 else 0)