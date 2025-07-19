"""
Enhanced Pattern-Aware Prediction Engine
Advanced prediction system with improved historical pattern following
"""

import numpy as np
import pandas as pd
from scipy import signal, stats, optimize
from scipy.signal import savgol_filter, butter, sosfilt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedPatternAwarePredictionEngine:
    """
    Enhanced prediction engine with improved historical pattern following
    """
    
    def __init__(self):
        # Prediction strategies with enhanced pattern awareness
        self.strategies = {
            'trend_continuation': self._enhanced_trend_continuation,
            'seasonal_decomposition': self._enhanced_seasonal_decomposition, 
            'cyclical_extrapolation': self._enhanced_cyclical_extrapolation,
            'pattern_matching': self._enhanced_pattern_matching,
            'ensemble': self._enhanced_ensemble,
            'adaptive': self._enhanced_adaptive_strategy
        }
        
        # Enhanced parameters for better pattern following
        self.prediction_params = {
            'trend_preservation_strength': 0.85,
            'seasonal_preservation_strength': 0.90,
            'cyclical_preservation_strength': 0.80,
            'pattern_matching_strength': 0.95,
            'variability_preservation_factor': 0.85,
            'historical_influence_decay': 0.03,
            'pattern_adaptation_rate': 0.08,
            'continuity_enforcement_strength': 0.90
        }
        
        # Multi-horizon prediction parameters
        self.horizon_params = {
            'short_term': {'weight': 0.4, 'horizon': 10, 'accuracy_weight': 0.9},
            'medium_term': {'weight': 0.35, 'horizon': 20, 'accuracy_weight': 0.8},
            'long_term': {'weight': 0.25, 'horizon': 30, 'accuracy_weight': 0.7}
        }
        
        # Performance tracking for adaptive improvement
        self.strategy_performance = defaultdict(list)
        self.pattern_performance = defaultdict(list)
        self.prediction_history = deque(maxlen=1000)
        
        # Enhanced pattern memory
        self.pattern_templates = {}
        self.learned_relationships = {}
        self.adaptation_history = deque(maxlen=500)
        
    def generate_pattern_aware_predictions(self, data: np.ndarray, 
                                         steps: int = 30,
                                         patterns: Dict[str, Any] = None,
                                         previous_predictions: Optional[List] = None,
                                         confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate predictions with enhanced pattern awareness
        """
        try:
            logger.info(f"Generating {steps} enhanced pattern-aware predictions")
            
            if len(data) < 3:
                return self._generate_minimal_predictions(data, steps)
            
            # Enhanced pattern analysis
            if patterns is None:
                patterns = self._analyze_enhanced_patterns(data)
            
            # Select optimal prediction strategy
            optimal_strategy = self._select_optimal_strategy(data, patterns, previous_predictions)
            
            # Generate multi-horizon predictions
            multi_horizon_predictions = self._generate_multi_horizon_predictions(
                data, steps, patterns, optimal_strategy
            )
            
            # Apply enhanced pattern corrections
            pattern_corrected = self._apply_enhanced_pattern_corrections(
                multi_horizon_predictions, data, patterns
            )
            
            # Apply advanced continuity enforcement
            continuity_corrected = self._apply_advanced_continuity_enforcement(
                pattern_corrected, data, previous_predictions
            )
            
            # Apply intelligent variability preservation
            final_predictions = self._apply_intelligent_variability_preservation(
                continuity_corrected, data, patterns
            )
            
            # Calculate dynamic confidence intervals
            confidence_intervals = self._calculate_enhanced_confidence_intervals(
                final_predictions, data, patterns, confidence_level
            )
            
            # Comprehensive quality assessment
            quality_metrics = self._assess_comprehensive_prediction_quality(
                final_predictions, data, patterns
            )
            
            # Update performance tracking
            self._update_performance_tracking(final_predictions, data, patterns, quality_metrics)
            
            # Create prediction result
            prediction_result = {
                'predictions': final_predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'pattern_analysis': patterns,
                'prediction_strategy': optimal_strategy,
                'quality_metrics': quality_metrics,
                'prediction_method': 'enhanced_pattern_aware',
                'pattern_preservation_score': quality_metrics.get('pattern_preservation_score', 0.5),
                'pattern_characteristics': self._extract_pattern_characteristics(patterns),
                'multi_horizon_info': {
                    'short_term_quality': quality_metrics.get('short_term_quality', 0.5),
                    'medium_term_quality': quality_metrics.get('medium_term_quality', 0.5),
                    'long_term_quality': quality_metrics.get('long_term_quality', 0.5)
                }
            }
            
            # Store prediction for learning
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'data_snapshot': data[-min(50, len(data)):].tolist(),
                'predictions': final_predictions.tolist(),
                'patterns': patterns,
                'quality': quality_metrics
            })
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error generating pattern-aware predictions: {e}")
            return self._generate_fallback_predictions(data, steps)
    
    def _analyze_enhanced_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Enhanced pattern analysis for better prediction"""
        try:
            patterns = {}
            
            # Multi-scale trend analysis
            patterns['trend_analysis'] = self._multi_scale_trend_analysis(data)
            
            # Advanced seasonal decomposition
            patterns['seasonal_analysis'] = self._advanced_seasonal_analysis(data)
            
            # Enhanced cyclical detection
            patterns['cyclical_analysis'] = self._enhanced_cyclical_analysis(data)
            
            # Pattern stability assessment
            patterns['stability_analysis'] = self._assess_pattern_stability(data)
            
            # Volatility and variance analysis
            patterns['volatility_analysis'] = self._analyze_volatility_patterns(data)
            
            # Correlation and dependency analysis
            patterns['correlation_analysis'] = self._analyze_correlation_patterns(data)
            
            # Overall pattern classification
            patterns['pattern_classification'] = self._classify_overall_pattern(patterns)
            
            # Prediction readiness score
            patterns['prediction_readiness'] = self._calculate_prediction_readiness(patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in enhanced pattern analysis: {e}")
            return {'error': str(e), 'prediction_readiness': 0.3}
    
    def _multi_scale_trend_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Multi-scale trend analysis for better trend detection"""
        try:
            trend_analysis = {}
            
            # Multiple time scales for trend analysis
            scales = [5, 10, 20, 30] if len(data) > 30 else [3, 5, min(len(data)//2, 10)]
            scale_trends = {}
            
            for scale in scales:
                if len(data) >= scale:
                    # Trend at this scale
                    scale_data = data[-scale:] if len(data) > scale else data
                    x = np.arange(len(scale_data))
                    
                    # Linear trend
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, scale_data)
                    
                    # Non-linear trend (polynomial)
                    if len(scale_data) > 3:
                        poly_coeffs = np.polyfit(x, scale_data, min(2, len(scale_data) - 1))
                        poly_pred = np.polyval(poly_coeffs, x)
                        poly_r2 = 1 - np.sum((scale_data - poly_pred) ** 2) / np.sum((scale_data - np.mean(scale_data)) ** 2)
                    else:
                        poly_r2 = r_value ** 2
                    
                    scale_trends[f'scale_{scale}'] = {
                        'linear_slope': float(slope),
                        'linear_r2': float(r_value ** 2),
                        'linear_p_value': float(p_value),
                        'polynomial_r2': float(max(0, poly_r2)),
                        'trend_strength': float(max(abs(r_value), np.sqrt(max(0, poly_r2))))
                    }
            
            # Aggregate trend information
            if scale_trends:
                trend_strengths = [t['trend_strength'] for t in scale_trends.values()]
                slopes = [t['linear_slope'] for t in scale_trends.values()]
                
                trend_analysis['overall_trend'] = {
                    'strength': float(np.mean(trend_strengths)),
                    'consistency': float(1.0 - np.std(slopes) / (np.mean(np.abs(slopes)) + 1e-8)),
                    'direction': float(np.sign(np.mean(slopes))),
                    'slope': float(np.mean(slopes)),
                    'confidence': float(np.mean([t['linear_r2'] for t in scale_trends.values()]))
                }
                
                trend_analysis['scale_trends'] = scale_trends
            else:
                trend_analysis['overall_trend'] = {
                    'strength': 0.0, 'consistency': 0.0, 'direction': 0.0, 'slope': 0.0, 'confidence': 0.0
                }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error in multi-scale trend analysis: {e}")
            return {'overall_trend': {'strength': 0.0, 'consistency': 0.0, 'direction': 0.0}}
    
    def _advanced_seasonal_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Advanced seasonal pattern analysis"""
        try:
            seasonal_analysis = {}
            
            if len(data) < 6:
                return {'seasonal_strength': 0.0, 'dominant_period': None}
            
            # Test multiple seasonal periods
            max_period = min(len(data) // 3, 100)
            seasonal_candidates = []
            
            for period in range(3, max_period + 1):
                seasonal_strength = self._calculate_seasonal_strength(data, period)
                if seasonal_strength > 0.1:
                    seasonal_candidates.append({
                        'period': period,
                        'strength': seasonal_strength,
                        'components': self._extract_seasonal_components(data, period)
                    })
            
            # Sort by strength
            seasonal_candidates.sort(key=lambda x: x['strength'], reverse=True)
            
            if seasonal_candidates:
                dominant_seasonal = seasonal_candidates[0]
                seasonal_analysis['seasonal_strength'] = dominant_seasonal['strength']
                seasonal_analysis['dominant_period'] = dominant_seasonal['period']
                seasonal_analysis['seasonal_components'] = dominant_seasonal['components']
                seasonal_analysis['all_periods'] = seasonal_candidates[:5]
                
                # Seasonal consistency across the data
                seasonal_analysis['consistency'] = self._calculate_seasonal_consistency(
                    data, dominant_seasonal['period']
                )
            else:
                seasonal_analysis['seasonal_strength'] = 0.0
                seasonal_analysis['dominant_period'] = None
                seasonal_analysis['consistency'] = 0.0
            
            return seasonal_analysis
            
        except Exception as e:
            logger.error(f"Error in advanced seasonal analysis: {e}")
            return {'seasonal_strength': 0.0, 'dominant_period': None}
    
    def _enhanced_cyclical_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Enhanced cyclical pattern analysis"""
        try:
            cyclical_analysis = {}
            
            if len(data) < 8:
                return {'cyclical_strength': 0.0}
            
            # FFT-based cyclical analysis
            fft = np.fft.fft(data - np.mean(data))  # Remove DC component
            freqs = np.fft.fftfreq(len(data))
            power_spectrum = np.abs(fft) ** 2
            
            # Focus on positive frequencies
            positive_freqs = freqs[1:len(freqs)//2]
            positive_power = power_spectrum[1:len(power_spectrum)//2]
            
            if len(positive_power) > 0:
                # Find dominant cycles
                sorted_indices = np.argsort(positive_power)[::-1]
                dominant_cycles = []
                
                for i in range(min(5, len(sorted_indices))):
                    idx = sorted_indices[i]
                    freq = positive_freqs[idx]
                    power = positive_power[idx]
                    period = 1.0 / abs(freq) if freq != 0 else len(data)
                    
                    dominant_cycles.append({
                        'frequency': float(freq),
                        'period': float(period),
                        'power': float(power),
                        'relative_strength': float(power / np.sum(positive_power))
                    })
                
                # Overall cyclical strength
                total_power = np.sum(positive_power)
                top_3_power = np.sum([c['power'] for c in dominant_cycles[:3]])
                cyclical_strength = top_3_power / (total_power + 1e-8)
                
                cyclical_analysis['cyclical_strength'] = float(cyclical_strength)
                cyclical_analysis['dominant_cycles'] = dominant_cycles
                cyclical_analysis['spectral_entropy'] = self._calculate_spectral_entropy(positive_power)
            else:
                cyclical_analysis['cyclical_strength'] = 0.0
            
            # Autocorrelation-based cyclical analysis
            cyclical_analysis['autocorr_cycles'] = self._detect_autocorr_cycles(data)
            
            return cyclical_analysis
            
        except Exception as e:
            logger.error(f"Error in enhanced cyclical analysis: {e}")
            return {'cyclical_strength': 0.0}
    
    def _select_optimal_strategy(self, data: np.ndarray, patterns: Dict[str, Any], 
                               previous_predictions: Optional[List] = None) -> str:
        """Select optimal prediction strategy based on patterns"""
        try:
            strategy_scores = {}
            
            # Score each strategy based on pattern characteristics
            trend_strength = patterns.get('trend_analysis', {}).get('overall_trend', {}).get('strength', 0)
            seasonal_strength = patterns.get('seasonal_analysis', {}).get('seasonal_strength', 0)
            cyclical_strength = patterns.get('cyclical_analysis', {}).get('cyclical_strength', 0)
            stability = patterns.get('stability_analysis', {}).get('overall_stability', 0.5)
            
            # Trend continuation strategy
            strategy_scores['trend_continuation'] = trend_strength * 0.8 + stability * 0.2
            
            # Seasonal decomposition strategy
            strategy_scores['seasonal_decomposition'] = seasonal_strength * 0.9 + stability * 0.1
            
            # Cyclical extrapolation strategy
            strategy_scores['cyclical_extrapolation'] = cyclical_strength * 0.85 + stability * 0.15
            
            # Pattern matching strategy
            pattern_diversity = len(patterns.get('pattern_classification', {}).get('top_patterns', {}))
            pattern_matching_score = min(pattern_diversity / 3.0, 1.0) * 0.7 + stability * 0.3
            strategy_scores['pattern_matching'] = pattern_matching_score
            
            # Ensemble strategy (good when multiple patterns are strong)
            ensemble_score = np.mean([trend_strength, seasonal_strength, cyclical_strength]) * 0.6 + stability * 0.4
            strategy_scores['ensemble'] = ensemble_score
            
            # Adaptive strategy (good for complex/changing patterns)
            complexity = 1.0 - stability  # Higher complexity when less stable
            adaptive_score = complexity * 0.7 + np.mean([trend_strength, seasonal_strength, cyclical_strength]) * 0.3
            strategy_scores['adaptive'] = adaptive_score
            
            # Consider historical performance
            for strategy, score in strategy_scores.items():
                if strategy in self.strategy_performance and self.strategy_performance[strategy]:
                    historical_performance = np.mean(self.strategy_performance[strategy][-10:])  # Last 10 predictions
                    strategy_scores[strategy] = score * 0.7 + historical_performance * 0.3
            
            # Select best strategy
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            
            logger.info(f"Selected strategy: {best_strategy} with score: {strategy_scores[best_strategy]:.3f}")
            
            return best_strategy
            
        except Exception as e:
            logger.error(f"Error selecting optimal strategy: {e}")
            return 'adaptive'  # Safe fallback
    
    def _generate_multi_horizon_predictions(self, data: np.ndarray, steps: int,
                                          patterns: Dict[str, Any], strategy: str) -> np.ndarray:
        """Generate predictions using multi-horizon approach"""
        try:
            # Generate predictions for different horizons
            short_horizon = min(steps, self.horizon_params['short_term']['horizon'])
            medium_horizon = min(steps, self.horizon_params['medium_term']['horizon'])
            long_horizon = steps
            
            # Generate predictions for each horizon
            short_pred = self._generate_horizon_predictions(data, short_horizon, patterns, strategy, 'short')
            medium_pred = self._generate_horizon_predictions(data, medium_horizon, patterns, strategy, 'medium')
            long_pred = self._generate_horizon_predictions(data, long_horizon, patterns, strategy, 'long')
            
            # Combine predictions with weighted ensemble
            final_predictions = np.zeros(steps)
            
            for i in range(steps):
                weights = []
                values = []
                
                # Short-term prediction
                if i < len(short_pred):
                    weight = self.horizon_params['short_term']['weight'] * np.exp(-0.1 * i)
                    weights.append(weight)
                    values.append(short_pred[i])
                
                # Medium-term prediction
                if i < len(medium_pred):
                    weight = self.horizon_params['medium_term']['weight'] * np.exp(-0.05 * i)
                    weights.append(weight)
                    values.append(medium_pred[i])
                
                # Long-term prediction
                if i < len(long_pred):
                    weight = self.horizon_params['long_term']['weight']
                    weights.append(weight)
                    values.append(long_pred[i])
                
                # Weighted average
                if weights and values:
                    final_predictions[i] = np.average(values, weights=weights)
                else:
                    # Fallback
                    final_predictions[i] = data[-1] if len(data) > 0 else 0.0
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Error generating multi-horizon predictions: {e}")
            return self._simple_prediction_fallback(data, steps)
    
    def _generate_horizon_predictions(self, data: np.ndarray, steps: int,
                                    patterns: Dict[str, Any], strategy: str, 
                                    horizon_type: str) -> np.ndarray:
        """Generate predictions for specific horizon"""
        try:
            # Adjust patterns based on horizon
            horizon_patterns = self._adapt_patterns_for_horizon(patterns, horizon_type)
            
            # Use the selected strategy
            if strategy in self.strategies:
                predictions = self.strategies[strategy](data, steps, horizon_patterns)
            else:
                predictions = self._enhanced_adaptive_strategy(data, steps, horizon_patterns)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating horizon predictions: {e}")
            return self._simple_prediction_fallback(data, steps)
    
    def _enhanced_trend_continuation(self, data: np.ndarray, steps: int, patterns: Dict[str, Any]) -> np.ndarray:
        """Enhanced trend continuation with better pattern preservation"""
        try:
            trend_info = patterns.get('trend_analysis', {}).get('overall_trend', {})
            slope = trend_info.get('slope', 0)
            confidence = trend_info.get('confidence', 0.5)
            
            # Enhanced trend calculation
            if len(data) >= 5:
                # Use multiple regression approaches
                x = np.arange(len(data))
                
                # Linear trend
                linear_slope = np.polyfit(x, data, 1)[0]
                
                # Weighted trend (more weight to recent data)
                weights = np.exp(0.1 * x)
                weighted_slope = np.polyfit(x, data, 1, w=weights)[0]
                
                # Robust trend (less sensitive to outliers)
                median_slope = np.median(np.diff(data))
                
                # Combine trends based on data characteristics
                if confidence > 0.7:
                    final_slope = linear_slope
                else:
                    final_slope = 0.4 * linear_slope + 0.3 * weighted_slope + 0.3 * median_slope
            else:
                final_slope = np.mean(np.diff(data)) if len(data) > 1 else 0
            
            # Generate predictions
            last_value = data[-1]
            predictions = []
            
            for i in range(1, steps + 1):
                # Trend continuation with decay for long-term predictions
                decay_factor = np.exp(-self.prediction_params['historical_influence_decay'] * i)
                trend_contribution = final_slope * i * decay_factor
                pred_value = last_value + trend_contribution
                predictions.append(pred_value)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in enhanced trend continuation: {e}")
            return self._simple_prediction_fallback(data, steps)
    
    def _enhanced_seasonal_decomposition(self, data: np.ndarray, steps: int, patterns: Dict[str, Any]) -> np.ndarray:
        """Enhanced seasonal decomposition with improved pattern preservation"""
        try:
            seasonal_info = patterns.get('seasonal_analysis', {})
            period = seasonal_info.get('dominant_period')
            seasonal_strength = seasonal_info.get('seasonal_strength', 0)
            
            if not period or seasonal_strength < 0.2:
                return self._enhanced_trend_continuation(data, steps, patterns)
            
            # Decompose into trend + seasonal + residual
            trend_component = self._extract_trend_component(data)
            seasonal_component = self._extract_seasonal_component(data, period)
            
            # Predict each component separately
            # Trend prediction
            trend_predictions = self._predict_trend_component(trend_component, steps)
            
            # Seasonal prediction (repeat seasonal pattern)
            seasonal_predictions = self._predict_seasonal_component(seasonal_component, steps, period)
            
            # Combine components
            predictions = trend_predictions + seasonal_predictions
            
            # Apply seasonal strength factor
            base_predictions = self._enhanced_trend_continuation(data, steps, patterns)
            final_predictions = (1 - seasonal_strength) * base_predictions + seasonal_strength * predictions
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Error in enhanced seasonal decomposition: {e}")
            return self._enhanced_trend_continuation(data, steps, patterns)
    
    def _enhanced_cyclical_extrapolation(self, data: np.ndarray, steps: int, patterns: Dict[str, Any]) -> np.ndarray:
        """Enhanced cyclical extrapolation with improved accuracy"""
        try:
            cyclical_info = patterns.get('cyclical_analysis', {})
            cyclical_strength = cyclical_info.get('cyclical_strength', 0)
            
            if cyclical_strength < 0.3:
                return self._enhanced_trend_continuation(data, steps, patterns)
            
            dominant_cycles = cyclical_info.get('dominant_cycles', [])
            if not dominant_cycles:
                return self._enhanced_trend_continuation(data, steps, patterns)
            
            # Extract cyclical components
            base_signal = data - np.mean(data)
            predictions = np.zeros(steps)
            
            # Reconstruct signal using dominant cycles
            for cycle in dominant_cycles[:3]:  # Use top 3 cycles
                frequency = cycle['frequency']
                strength = cycle['relative_strength']
                period = cycle['period']
                
                if period > 2:  # Valid period
                    # Estimate amplitude and phase from historical data
                    amplitude = np.std(base_signal) * strength
                    
                    # Phase estimation using correlation
                    phase = self._estimate_cycle_phase(base_signal, period)
                    
                    # Generate cyclical predictions
                    for i in range(steps):
                        cycle_index = (len(data) + i) % period
                        cycle_value = amplitude * np.sin(2 * np.pi * cycle_index / period + phase)
                        predictions[i] += cycle_value * strength
            
            # Add trend component
            trend_predictions = self._enhanced_trend_continuation(data, steps, patterns)
            trend_only = trend_predictions - trend_predictions[0] + data[-1]  # Center around last data point
            
            # Combine cyclical and trend components
            final_predictions = trend_only + predictions
            
            # Blend with base predictions based on cyclical strength
            base_predictions = self._enhanced_trend_continuation(data, steps, patterns)
            final_predictions = (1 - cyclical_strength) * base_predictions + cyclical_strength * final_predictions
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Error in enhanced cyclical extrapolation: {e}")
            return self._enhanced_trend_continuation(data, steps, patterns)
    
    def _enhanced_pattern_matching(self, data: np.ndarray, steps: int, patterns: Dict[str, Any]) -> np.ndarray:
        """Enhanced pattern matching with historical templates"""
        try:
            if len(data) < 10:
                return self._enhanced_trend_continuation(data, steps, patterns)
            
            # Find similar historical patterns
            pattern_length = min(20, len(data) // 2)
            current_pattern = data[-pattern_length:]
            
            best_matches = self._find_pattern_matches(current_pattern, data, min_match_length=pattern_length//2)
            
            if not best_matches:
                return self._enhanced_adaptive_strategy(data, steps, patterns)
            
            # Generate predictions based on pattern matches
            match_predictions = []
            
            for match in best_matches[:5]:  # Use top 5 matches
                match_start = match['start_index']
                match_end = match['end_index']
                match_similarity = match['similarity']
                
                # Get continuation after this pattern
                continuation_start = match_end + 1
                continuation_end = min(continuation_start + steps, len(data))
                
                if continuation_end > continuation_start:
                    continuation = data[continuation_start:continuation_end]
                    
                    # Pad if necessary
                    if len(continuation) < steps:
                        # Extend using trend from the continuation
                        if len(continuation) > 1:
                            trend = np.mean(np.diff(continuation))
                            last_value = continuation[-1]
                            extension = [last_value + trend * (i + 1) for i in range(steps - len(continuation))]
                            continuation = np.concatenate([continuation, extension])
                        else:
                            continuation = np.pad(continuation, (0, steps - len(continuation)), 'edge')
                    
                    match_predictions.append({
                        'predictions': continuation[:steps],
                        'weight': match_similarity
                    })
            
            if match_predictions:
                # Weighted average of pattern matches
                weights = np.array([mp['weight'] for mp in match_predictions])
                weights = weights / np.sum(weights)  # Normalize
                
                final_predictions = np.zeros(steps)
                for i, mp in enumerate(match_predictions):
                    final_predictions += weights[i] * mp['predictions']
                
                # Blend with trend-based predictions for robustness
                trend_predictions = self._enhanced_trend_continuation(data, steps, patterns)
                pattern_confidence = np.mean(weights)
                
                final_predictions = pattern_confidence * final_predictions + (1 - pattern_confidence) * trend_predictions
                
                return final_predictions
            else:
                return self._enhanced_adaptive_strategy(data, steps, patterns)
                
        except Exception as e:
            logger.error(f"Error in enhanced pattern matching: {e}")
            return self._enhanced_adaptive_strategy(data, steps, patterns)
    
    def _enhanced_ensemble(self, data: np.ndarray, steps: int, patterns: Dict[str, Any]) -> np.ndarray:
        """Enhanced ensemble method combining multiple strategies"""
        try:
            # Generate predictions using multiple strategies
            strategies_to_use = ['trend_continuation', 'seasonal_decomposition', 'cyclical_extrapolation']
            strategy_predictions = {}
            strategy_weights = {}
            
            # Get predictions from each strategy
            for strategy in strategies_to_use:
                try:
                    predictions = self.strategies[strategy](data, steps, patterns)
                    strategy_predictions[strategy] = predictions
                    
                    # Calculate strategy weight based on pattern fit
                    weight = self._calculate_strategy_weight(data, patterns, strategy)
                    strategy_weights[strategy] = weight
                    
                except Exception as e:
                    logger.warning(f"Strategy {strategy} failed: {e}")
                    continue
            
            if not strategy_predictions:
                return self._enhanced_adaptive_strategy(data, steps, patterns)
            
            # Normalize weights
            total_weight = sum(strategy_weights.values())
            if total_weight > 0:
                for strategy in strategy_weights:
                    strategy_weights[strategy] /= total_weight
            else:
                # Equal weights
                equal_weight = 1.0 / len(strategy_predictions)
                strategy_weights = {s: equal_weight for s in strategy_predictions.keys()}
            
            # Combine predictions
            ensemble_predictions = np.zeros(steps)
            for strategy, predictions in strategy_predictions.items():
                weight = strategy_weights[strategy]
                ensemble_predictions += weight * predictions
            
            return ensemble_predictions
            
        except Exception as e:
            logger.error(f"Error in enhanced ensemble: {e}")
            return self._enhanced_adaptive_strategy(data, steps, patterns)
    
    def _enhanced_adaptive_strategy(self, data: np.ndarray, steps: int, patterns: Dict[str, Any]) -> np.ndarray:
        """Enhanced adaptive strategy that adjusts to data characteristics"""
        try:
            # Analyze data characteristics for adaptation
            data_characteristics = self._analyze_data_characteristics(data)
            
            # Select approach based on characteristics
            if data_characteristics['is_trending']:
                base_predictions = self._enhanced_trend_continuation(data, steps, patterns)
            elif data_characteristics['is_seasonal']:
                base_predictions = self._enhanced_seasonal_decomposition(data, steps, patterns)
            elif data_characteristics['is_cyclical']:
                base_predictions = self._enhanced_cyclical_extrapolation(data, steps, patterns)
            else:
                # Mixed approach
                base_predictions = self._mixed_approach_predictions(data, steps, patterns)
            
            # Apply adaptive corrections
            adaptive_corrections = self._calculate_adaptive_corrections(data, base_predictions, patterns)
            final_predictions = base_predictions + adaptive_corrections
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Error in enhanced adaptive strategy: {e}")
            return self._simple_prediction_fallback(data, steps)
    
    def _apply_enhanced_pattern_corrections(self, predictions: np.ndarray, data: np.ndarray, 
                                          patterns: Dict[str, Any]) -> np.ndarray:
        """Apply enhanced pattern corrections for better historical pattern following"""
        try:
            corrected_predictions = predictions.copy()
            
            # 1. Statistical property preservation
            corrected_predictions = self._preserve_statistical_properties(corrected_predictions, data)
            
            # 2. Trend consistency enforcement
            corrected_predictions = self._enforce_trend_consistency(corrected_predictions, data, patterns)
            
            # 3. Seasonal pattern enforcement
            corrected_predictions = self._enforce_seasonal_patterns(corrected_predictions, data, patterns)
            
            # 4. Volatility pattern matching
            corrected_predictions = self._match_volatility_patterns(corrected_predictions, data)
            
            # 5. Range boundary enforcement
            corrected_predictions = self._enforce_range_boundaries(corrected_predictions, data)
            
            return corrected_predictions
            
        except Exception as e:
            logger.error(f"Error applying enhanced pattern corrections: {e}")
            return predictions
    
    def _apply_advanced_continuity_enforcement(self, predictions: np.ndarray, data: np.ndarray,
                                             previous_predictions: Optional[List] = None) -> np.ndarray:
        """Apply advanced continuity enforcement"""
        try:
            continuity_corrected = predictions.copy()
            
            # 1. Smooth transition from historical data
            if len(data) > 0:
                transition_strength = self.prediction_params['continuity_enforcement_strength']
                last_historical = data[-1]
                first_prediction = continuity_corrected[0]
                
                # Calculate transition adjustment
                transition_gap = first_prediction - last_historical
                
                # Apply exponential decay correction
                for i in range(len(continuity_corrected)):
                    decay_factor = np.exp(-0.2 * i)  # Exponential decay
                    correction = transition_gap * decay_factor * transition_strength
                    continuity_corrected[i] -= correction
            
            # 2. Smooth transitions within predictions
            if len(continuity_corrected) > 3:
                # Apply light smoothing to remove abrupt changes
                smoothed = savgol_filter(continuity_corrected, min(7, len(continuity_corrected)), 2)
                
                # Blend original and smoothed
                smoothing_strength = 0.3
                continuity_corrected = (1 - smoothing_strength) * continuity_corrected + smoothing_strength * smoothed
            
            # 3. Consistency with previous predictions (if available)
            if previous_predictions and len(previous_predictions) > 0:
                continuity_corrected = self._enforce_prediction_consistency(
                    continuity_corrected, previous_predictions, data
                )
            
            return continuity_corrected
            
        except Exception as e:
            logger.error(f"Error applying advanced continuity enforcement: {e}")
            return predictions
    
    def _apply_intelligent_variability_preservation(self, predictions: np.ndarray, data: np.ndarray,
                                                   patterns: Dict[str, Any]) -> np.ndarray:
        """Apply intelligent variability preservation"""
        try:
            variability_corrected = predictions.copy()
            
            # Historical variability characteristics
            historical_std = np.std(data) if len(data) > 1 else 1.0
            historical_range = np.max(data) - np.min(data) if len(data) > 1 else 2.0
            
            # Current prediction variability
            prediction_std = np.std(variability_corrected) if len(variability_corrected) > 1 else historical_std
            
            # Variability adjustment
            target_variability = historical_std * self.prediction_params['variability_preservation_factor']
            
            if prediction_std > 0:
                # Scale predictions to match target variability
                prediction_mean = np.mean(variability_corrected)
                scaled_predictions = prediction_mean + (variability_corrected - prediction_mean) * (target_variability / prediction_std)
                
                # Blend with original predictions
                blend_factor = 0.7
                variability_corrected = blend_factor * scaled_predictions + (1 - blend_factor) * variability_corrected
            
            # Add realistic noise based on historical patterns
            noise_level = historical_std * 0.05  # Small amount of realistic noise
            realistic_noise = np.random.normal(0, noise_level, len(variability_corrected))
            variability_corrected += realistic_noise
            
            return variability_corrected
            
        except Exception as e:
            logger.error(f"Error applying intelligent variability preservation: {e}")
            return predictions
    
    def _calculate_enhanced_confidence_intervals(self, predictions: np.ndarray, data: np.ndarray,
                                               patterns: Dict[str, Any], confidence_level: float) -> List[Dict]:
        """Calculate enhanced confidence intervals"""
        try:
            confidence_intervals = []
            
            # Base uncertainty from historical data
            historical_std = np.std(data) if len(data) > 1 else 1.0
            
            # Pattern-based uncertainty adjustment
            pattern_uncertainty = self._calculate_pattern_uncertainty(patterns)
            
            # Time-varying uncertainty (increases with prediction horizon)
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            
            for i, pred in enumerate(predictions):
                # Uncertainty increases with time
                time_factor = 1.0 + 0.05 * i  # 5% increase per step
                
                # Combine uncertainties
                total_uncertainty = historical_std * time_factor * (1 + pattern_uncertainty)
                
                # Confidence interval
                std_error = total_uncertainty / np.sqrt(len(data)) if len(data) > 0 else total_uncertainty
                margin_of_error = z_score * std_error
                
                confidence_intervals.append({
                    'lower': float(pred - margin_of_error),
                    'upper': float(pred + margin_of_error),
                    'std_error': float(std_error),
                    'confidence_level': float(confidence_level)
                })
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating enhanced confidence intervals: {e}")
            # Fallback simple confidence intervals
            return [{'lower': p - 1.0, 'upper': p + 1.0, 'std_error': 1.0} for p in predictions]
    
    def _assess_comprehensive_prediction_quality(self, predictions: np.ndarray, data: np.ndarray,
                                               patterns: Dict[str, Any]) -> Dict[str, float]:
        """Assess comprehensive prediction quality"""
        try:
            quality_metrics = {}
            
            # Pattern preservation score
            quality_metrics['pattern_preservation_score'] = self._calculate_pattern_preservation_score(
                predictions, data, patterns
            )
            
            # Continuity score
            quality_metrics['continuity_score'] = self._calculate_continuity_score(predictions, data)
            
            # Variability preservation score
            quality_metrics['variability_preservation_score'] = self._calculate_variability_preservation_score(
                predictions, data
            )
            
            # Trend consistency score
            quality_metrics['trend_consistency_score'] = self._calculate_trend_consistency_score(
                predictions, data, patterns
            )
            
            # Statistical similarity score
            quality_metrics['statistical_similarity_score'] = self._calculate_statistical_similarity_score(
                predictions, data
            )
            
            # Multi-horizon quality scores
            short_term_preds = predictions[:10] if len(predictions) >= 10 else predictions
            quality_metrics['short_term_quality'] = np.mean([
                quality_metrics['pattern_preservation_score'],
                quality_metrics['continuity_score']
            ])
            
            quality_metrics['medium_term_quality'] = quality_metrics['pattern_preservation_score'] * 0.8
            quality_metrics['long_term_quality'] = quality_metrics['pattern_preservation_score'] * 0.6
            
            # Overall quality score
            quality_metrics['overall_quality_score'] = np.mean([
                quality_metrics['pattern_preservation_score'],
                quality_metrics['continuity_score'],
                quality_metrics['variability_preservation_score'],
                quality_metrics['trend_consistency_score'],
                quality_metrics['statistical_similarity_score']
            ])
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing comprehensive prediction quality: {e}")
            return {'overall_quality_score': 0.5}
    
    def _simple_prediction_fallback(self, data: np.ndarray, steps: int) -> np.ndarray:
        """Simple prediction fallback"""
        try:
            if len(data) == 0:
                return np.zeros(steps)
            elif len(data) == 1:
                return np.full(steps, data[0])
            else:
                # Simple linear continuation
                slope = np.mean(np.diff(data[-5:])) if len(data) >= 5 else np.mean(np.diff(data))
                last_value = data[-1]
                predictions = [last_value + slope * (i + 1) for i in range(steps)]
                return np.array(predictions)
        except:
            return np.zeros(steps)
    
    def _generate_minimal_predictions(self, data: np.ndarray, steps: int) -> Dict[str, Any]:
        """Generate minimal predictions for insufficient data"""
        predictions = self._simple_prediction_fallback(data, steps)
        return {
            'predictions': predictions.tolist(),
            'confidence_intervals': [{'lower': p-1, 'upper': p+1} for p in predictions],
            'pattern_analysis': {'insufficient_data': True},
            'quality_metrics': {'overall_quality_score': 0.3}
        }
    
    def _generate_fallback_predictions(self, data: np.ndarray, steps: int) -> Dict[str, Any]:
        """Generate fallback predictions when main system fails"""
        predictions = self._simple_prediction_fallback(data, steps)
        return {
            'predictions': predictions.tolist(),
            'confidence_intervals': [{'lower': p-2, 'upper': p+2} for p in predictions],
            'pattern_analysis': {'error': 'fallback_mode'},
            'quality_metrics': {'overall_quality_score': 0.2}
        }