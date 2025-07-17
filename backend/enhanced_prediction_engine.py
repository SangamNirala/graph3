"""
Enhanced Prediction Engine for Real-time Graph Prediction System
Focus on pattern accuracy and historical trend preservation
"""

import numpy as np
import pandas as pd
from scipy import signal, optimize
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedPredictionEngine:
    """
    Enhanced prediction engine that focuses on preserving historical patterns
    and maintaining accurate trend continuation
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pattern_analyzer = None
        self.prediction_cache = {}
        self.historical_patterns = None
        
    def set_pattern_analyzer(self, analyzer):
        """Set the pattern analyzer"""
        self.pattern_analyzer = analyzer
        
    def generate_pattern_aware_predictions(self, data: np.ndarray, 
                                         steps: int = 30,
                                         patterns: Optional[Dict] = None,
                                         confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate predictions that are aware of and preserve historical patterns
        """
        try:
            if patterns is None and self.pattern_analyzer:
                patterns = self.pattern_analyzer.analyze_comprehensive_patterns(data)
            elif patterns is None:
                patterns = self._get_basic_patterns(data)
            
            # Store historical patterns for reference
            self.historical_patterns = patterns
            
            # Choose prediction method based on pattern characteristics
            prediction_method = self._select_optimal_prediction_method(patterns)
            
            # Generate base predictions
            base_predictions = self._generate_base_predictions(data, steps, patterns, prediction_method)
            
            # Apply pattern-aware corrections
            corrected_predictions = self._apply_pattern_corrections(base_predictions, data, patterns)
            
            # Apply smoothing while preserving patterns
            smoothed_predictions = self._apply_pattern_preserving_smoothing(corrected_predictions, data, patterns)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                smoothed_predictions, data, patterns, confidence_level
            )
            
            # Generate quality metrics
            quality_metrics = self._calculate_prediction_quality_metrics(
                smoothed_predictions, data, patterns
            )
            
            return {
                'predictions': smoothed_predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'quality_metrics': quality_metrics,
                'pattern_preservation_score': quality_metrics['pattern_preservation_score'],
                'prediction_method': prediction_method,
                'pattern_characteristics': {
                    'trend_strength': patterns['trend_analysis']['trend_strength'],
                    'seasonal_strength': patterns['seasonal_analysis']['seasonal_strength'],
                    'predictability_score': patterns['predictability']['predictability_score'],
                    'pattern_quality': patterns['quality_score']
                }
            }
            
        except Exception as e:
            logger.error(f"Error in pattern-aware prediction generation: {e}")
            return self._generate_fallback_predictions(data, steps)
    
    def _select_optimal_prediction_method(self, patterns: Dict) -> str:
        """Select the best prediction method based on pattern characteristics"""
        
        trend_strength = patterns['trend_analysis']['trend_strength']
        seasonal_strength = patterns['seasonal_analysis']['seasonal_strength']
        predictability = patterns['predictability']['predictability_score']
        volatility = patterns['volatility_analysis']['overall_volatility']
        
        # Decision logic based on pattern characteristics
        if seasonal_strength > 0.3:
            return 'seasonal_aware'
        elif trend_strength > 0.2 and predictability > 0.6:
            return 'trend_following'
        elif volatility < 0.1 and predictability > 0.7:
            return 'pattern_matching'
        elif len(patterns['cyclical_analysis']['detected_cycles']) > 0:
            return 'cyclical_aware'
        else:
            return 'adaptive_hybrid'
    
    def _generate_base_predictions(self, data: np.ndarray, steps: int, 
                                 patterns: Dict, method: str) -> np.ndarray:
        """Generate base predictions using the selected method"""
        
        if method == 'seasonal_aware':
            return self._seasonal_aware_prediction(data, steps, patterns)
        elif method == 'trend_following':
            return self._trend_following_prediction(data, steps, patterns)
        elif method == 'pattern_matching':
            return self._pattern_matching_prediction(data, steps, patterns)
        elif method == 'cyclical_aware':
            return self._cyclical_aware_prediction(data, steps, patterns)
        else:
            return self._adaptive_hybrid_prediction(data, steps, patterns)
    
    def _seasonal_aware_prediction(self, data: np.ndarray, steps: int, patterns: Dict) -> np.ndarray:
        """Generate predictions aware of seasonal patterns"""
        try:
            seasonal_analysis = patterns['seasonal_analysis']
            
            if 'seasonal_component' in seasonal_analysis:
                seasonal_component = np.array(seasonal_analysis['seasonal_component'])
                trend_component = np.array(seasonal_analysis['trend_component'])
                
                # Extend trend
                trend_slope = patterns['trend_analysis']['recent_trend']
                last_trend = trend_component[-1]
                
                extended_trend = []
                for i in range(steps):
                    extended_trend.append(last_trend + trend_slope * (i + 1))
                
                # Extend seasonal pattern
                seasonal_period = len(seasonal_component)
                extended_seasonal = []
                for i in range(steps):
                    seasonal_index = i % seasonal_period
                    extended_seasonal.append(seasonal_component[seasonal_index])
                
                # Combine trend and seasonal
                predictions = np.array(extended_trend) + np.array(extended_seasonal)
                
                # Add mean reversion
                historical_mean = patterns['statistical_properties']['mean']
                for i in range(len(predictions)):
                    reversion_factor = 0.05 * (i + 1) / steps
                    predictions[i] = (1 - reversion_factor) * predictions[i] + reversion_factor * historical_mean
                
                return predictions
            else:
                # Fallback to trend following
                return self._trend_following_prediction(data, steps, patterns)
                
        except Exception as e:
            logger.warning(f"Seasonal prediction failed: {e}")
            return self._trend_following_prediction(data, steps, patterns)
    
    def _trend_following_prediction(self, data: np.ndarray, steps: int, patterns: Dict) -> np.ndarray:
        """Generate predictions that follow the trend accurately"""
        try:
            trend_analysis = patterns['trend_analysis']
            
            # Use multiple trend components
            linear_trend = trend_analysis['recent_trend']
            acceleration = trend_analysis['trend_acceleration']
            
            last_value = data[-1]
            predictions = []
            
            for i in range(steps):
                # Linear trend component
                linear_component = linear_trend * (i + 1)
                
                # Acceleration component with decay
                acceleration_component = acceleration * (i + 1) * np.exp(-0.1 * i)
                
                # Mean reversion component
                historical_mean = patterns['statistical_properties']['mean']
                reversion_strength = 0.02 * (i + 1) / steps
                reversion_component = reversion_strength * (historical_mean - last_value)
                
                # Combine components
                predicted_value = last_value + linear_component + acceleration_component + reversion_component
                predictions.append(predicted_value)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.warning(f"Trend following prediction failed: {e}")
            return self._simple_linear_prediction(data, steps)
    
    def _pattern_matching_prediction(self, data: np.ndarray, steps: int, patterns: Dict) -> np.ndarray:
        """Generate predictions by matching similar historical patterns"""
        try:
            # Find similar patterns in historical data
            pattern_length = min(10, len(data) // 3)
            if pattern_length < 3:
                return self._trend_following_prediction(data, steps, patterns)
            
            current_pattern = data[-pattern_length:]
            
            # Find similar patterns in historical data
            similarities = []
            for i in range(len(data) - pattern_length - steps):
                historical_pattern = data[i:i + pattern_length]
                similarity = np.corrcoef(current_pattern, historical_pattern)[0, 1]
                
                if not np.isnan(similarity) and similarity > 0.7:  # High similarity threshold
                    future_pattern = data[i + pattern_length:i + pattern_length + steps]
                    if len(future_pattern) == steps:
                        similarities.append({
                            'similarity': similarity,
                            'future_pattern': future_pattern,
                            'weight': similarity ** 2
                        })
            
            if similarities:
                # Weighted average of similar patterns
                total_weight = sum(s['weight'] for s in similarities)
                predictions = np.zeros(steps)
                
                for sim in similarities:
                    weight = sim['weight'] / total_weight
                    predictions += weight * sim['future_pattern']
                
                # Adjust to start from last value
                adjustment = data[-1] - predictions[0]
                predictions += adjustment
                
                return predictions
            else:
                # No similar patterns found, use trend following
                return self._trend_following_prediction(data, steps, patterns)
                
        except Exception as e:
            logger.warning(f"Pattern matching prediction failed: {e}")
            return self._trend_following_prediction(data, steps, patterns)
    
    def _cyclical_aware_prediction(self, data: np.ndarray, steps: int, patterns: Dict) -> np.ndarray:
        """Generate predictions aware of cyclical patterns"""
        try:
            cyclical_analysis = patterns['cyclical_analysis']
            dominant_cycle = cyclical_analysis.get('dominant_cycle')
            
            if dominant_cycle and dominant_cycle['strength'] > 0.3:
                cycle_length = dominant_cycle['length']
                cycle_strength = dominant_cycle['strength']
                
                # Generate cyclical component
                cyclical_component = []
                for i in range(steps):
                    phase = 2 * np.pi * i / cycle_length
                    cycle_value = cycle_strength * np.sin(phase)
                    cyclical_component.append(cycle_value)
                
                # Generate trend component
                trend_component = self._trend_following_prediction(data, steps, patterns)
                
                # Combine cyclical and trend
                predictions = trend_component + np.array(cyclical_component) * patterns['statistical_properties']['std'] * 0.3
                
                return predictions
            else:
                return self._trend_following_prediction(data, steps, patterns)
                
        except Exception as e:
            logger.warning(f"Cyclical prediction failed: {e}")
            return self._trend_following_prediction(data, steps, patterns)
    
    def _adaptive_hybrid_prediction(self, data: np.ndarray, steps: int, patterns: Dict) -> np.ndarray:
        """Generate predictions using adaptive hybrid approach"""
        try:
            # Generate predictions using multiple methods
            trend_pred = self._trend_following_prediction(data, steps, patterns)
            pattern_pred = self._pattern_matching_prediction(data, steps, patterns)
            
            # Adaptive weighting based on pattern characteristics
            trend_weight = patterns['trend_analysis']['trend_strength']
            pattern_weight = patterns['pattern_similarity']['pattern_repetition']
            
            # Normalize weights
            total_weight = trend_weight + pattern_weight
            if total_weight > 0:
                trend_weight /= total_weight
                pattern_weight /= total_weight
            else:
                trend_weight = 0.7
                pattern_weight = 0.3
            
            # Combine predictions
            combined_predictions = trend_weight * trend_pred + pattern_weight * pattern_pred
            
            return combined_predictions
            
        except Exception as e:
            logger.warning(f"Adaptive hybrid prediction failed: {e}")
            return self._simple_linear_prediction(data, steps)
    
    def _apply_pattern_corrections(self, predictions: np.ndarray, 
                                 data: np.ndarray, patterns: Dict) -> np.ndarray:
        """Apply pattern-aware corrections to predictions"""
        try:
            corrected_predictions = predictions.copy()
            
            # Statistical bounds correction
            historical_mean = patterns['statistical_properties']['mean']
            historical_std = patterns['statistical_properties']['std']
            
            # Apply soft bounds based on historical statistics
            for i in range(len(corrected_predictions)):
                deviation = abs(corrected_predictions[i] - historical_mean)
                max_deviation = 3 * historical_std
                
                if deviation > max_deviation:
                    # Soft correction towards acceptable range
                    correction_factor = max_deviation / deviation
                    corrected_predictions[i] = historical_mean + correction_factor * (corrected_predictions[i] - historical_mean)
            
            # Trend consistency correction
            trend_consistency = patterns['trend_analysis']['trend_consistency']
            if trend_consistency > 0.7:
                # Apply trend consistency correction
                recent_trend = patterns['trend_analysis']['recent_trend']
                for i in range(1, len(corrected_predictions)):
                    predicted_trend = corrected_predictions[i] - corrected_predictions[i-1]
                    trend_correction = 0.3 * (recent_trend - predicted_trend)
                    corrected_predictions[i] += trend_correction
            
            # Volatility correction
            historical_volatility = patterns['volatility_analysis']['overall_volatility']
            prediction_volatility = np.std(np.diff(corrected_predictions))
            
            if prediction_volatility > 2 * historical_volatility:
                # Reduce excessive volatility
                smoothing_factor = historical_volatility / prediction_volatility
                for i in range(1, len(corrected_predictions)):
                    change = corrected_predictions[i] - corrected_predictions[i-1]
                    corrected_predictions[i] = corrected_predictions[i-1] + smoothing_factor * change
            
            return corrected_predictions
            
        except Exception as e:
            logger.warning(f"Pattern correction failed: {e}")
            return predictions
    
    def _apply_pattern_preserving_smoothing(self, predictions: np.ndarray, 
                                          data: np.ndarray, patterns: Dict) -> np.ndarray:
        """Apply smoothing while preserving important patterns"""
        try:
            if len(predictions) < 3:
                return predictions
            
            # Determine smoothing strength based on pattern characteristics
            predictability = patterns['predictability']['predictability_score']
            volatility = patterns['volatility_analysis']['overall_volatility']
            
            # Higher predictability and lower volatility allow for more smoothing
            smoothing_strength = min(0.7, predictability * (1 - volatility))
            
            if smoothing_strength < 0.1:
                return predictions  # No smoothing needed
            
            # Apply adaptive smoothing
            smoothed_predictions = predictions.copy()
            
            # Use a combination of moving average and spline smoothing
            window_size = max(2, int(len(predictions) * 0.1))
            
            for i in range(window_size, len(predictions)):
                # Moving average component
                ma_value = np.mean(predictions[i-window_size:i+1])
                
                # Weighted combination
                weight = smoothing_strength * (1 - i / len(predictions))  # Decreasing weight over time
                smoothed_predictions[i] = weight * ma_value + (1 - weight) * predictions[i]
            
            # Ensure smooth transition from historical data
            transition_length = min(3, len(predictions))
            if len(data) > 0:
                for i in range(transition_length):
                    transition_weight = (i + 1) / transition_length
                    direct_continuation = data[-1] + (predictions[i] - data[-1]) * transition_weight
                    smoothed_predictions[i] = (1 - transition_weight) * direct_continuation + transition_weight * smoothed_predictions[i]
            
            return smoothed_predictions
            
        except Exception as e:
            logger.warning(f"Pattern preserving smoothing failed: {e}")
            return predictions
    
    def _calculate_confidence_intervals(self, predictions: np.ndarray, 
                                      data: np.ndarray, patterns: Dict, 
                                      confidence_level: float) -> List[Dict]:
        """Calculate confidence intervals for predictions"""
        try:
            confidence_intervals = []
            
            # Base uncertainty from historical data
            historical_std = patterns['statistical_properties']['std']
            
            # Increasing uncertainty with prediction horizon
            for i, pred in enumerate(predictions):
                # Uncertainty grows with prediction horizon
                horizon_factor = 1 + 0.1 * i
                
                # Pattern-based uncertainty adjustment
                predictability = patterns['predictability']['predictability_score']
                uncertainty_factor = (2 - predictability) * horizon_factor
                
                # Calculate confidence interval
                std_error = historical_std * uncertainty_factor
                z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
                
                lower_bound = pred - z_score * std_error
                upper_bound = pred + z_score * std_error
                
                confidence_intervals.append({
                    'lower': float(lower_bound),
                    'upper': float(upper_bound),
                    'std_error': float(std_error)
                })
            
            return confidence_intervals
            
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            return [{'lower': float(p), 'upper': float(p), 'std_error': 0.0} for p in predictions]
    
    def _calculate_prediction_quality_metrics(self, predictions: np.ndarray, 
                                            data: np.ndarray, patterns: Dict) -> Dict:
        """Calculate quality metrics for predictions"""
        try:
            # Pattern preservation score
            pattern_preservation = self._calculate_pattern_preservation_score(predictions, data, patterns)
            
            # Continuity score
            continuity_score = self._calculate_continuity_score(predictions, data)
            
            # Trend consistency score
            trend_consistency = self._calculate_trend_consistency_score(predictions, data, patterns)
            
            # Overall quality score
            overall_score = (pattern_preservation + continuity_score + trend_consistency) / 3
            
            return {
                'pattern_preservation_score': float(pattern_preservation),
                'continuity_score': float(continuity_score),
                'trend_consistency_score': float(trend_consistency),
                'overall_quality_score': float(overall_score),
                'prediction_volatility': float(np.std(np.diff(predictions))),
                'historical_volatility': float(patterns['volatility_analysis']['overall_volatility'])
            }
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            return {'overall_quality_score': 0.5}
    
    def _calculate_pattern_preservation_score(self, predictions: np.ndarray, 
                                            data: np.ndarray, patterns: Dict) -> float:
        """Calculate how well predictions preserve historical patterns"""
        try:
            # Statistical similarity
            historical_mean = patterns['statistical_properties']['mean']
            historical_std = patterns['statistical_properties']['std']
            
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            
            mean_similarity = 1 - abs(pred_mean - historical_mean) / (historical_std + 1e-10)
            std_similarity = 1 - abs(pred_std - historical_std) / (historical_std + 1e-10)
            
            # Trend similarity
            historical_trend = patterns['trend_analysis']['recent_trend']
            pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
            
            trend_similarity = 1 - abs(pred_trend - historical_trend) / (abs(historical_trend) + 1e-10)
            
            # Combine scores
            preservation_score = (mean_similarity + std_similarity + trend_similarity) / 3
            
            return max(0, min(1, preservation_score))
            
        except Exception as e:
            return 0.5
    
    def _calculate_continuity_score(self, predictions: np.ndarray, data: np.ndarray) -> float:
        """Calculate continuity score between historical data and predictions"""
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
    
    def _calculate_trend_consistency_score(self, predictions: np.ndarray, 
                                         data: np.ndarray, patterns: Dict) -> float:
        """Calculate trend consistency score"""
        try:
            historical_trend = patterns['trend_analysis']['recent_trend']
            
            # Calculate prediction trend
            pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
            
            # Trend consistency
            trend_consistency = 1 - abs(pred_trend - historical_trend) / (abs(historical_trend) + 1e-10)
            
            return max(0, min(1, trend_consistency))
            
        except Exception as e:
            return 0.5
    
    def _simple_linear_prediction(self, data: np.ndarray, steps: int) -> np.ndarray:
        """Simple linear prediction as fallback"""
        if len(data) < 2:
            return np.full(steps, data[-1] if len(data) > 0 else 0)
        
        # Linear trend
        x = np.arange(len(data))
        trend = np.polyfit(x, data, 1)[0]
        
        predictions = []
        for i in range(steps):
            predictions.append(data[-1] + trend * (i + 1))
        
        return np.array(predictions)
    
    def _get_basic_patterns(self, data: np.ndarray) -> Dict:
        """Get basic patterns when full analysis is not available"""
        return {
            'statistical_properties': {
                'mean': float(np.mean(data)),
                'std': float(np.std(data))
            },
            'trend_analysis': {
                'recent_trend': float(np.polyfit(range(len(data)), data, 1)[0]),
                'trend_strength': 0.5,
                'trend_consistency': 0.5
            },
            'seasonal_analysis': {
                'seasonal_strength': 0.0
            },
            'volatility_analysis': {
                'overall_volatility': float(np.std(data))
            },
            'predictability': {
                'predictability_score': 0.5
            },
            'pattern_similarity': {
                'pattern_repetition': 0.0
            },
            'cyclical_analysis': {
                'detected_cycles': []
            },
            'quality_score': 0.5
        }
    
    def _generate_fallback_predictions(self, data: np.ndarray, steps: int) -> Dict:
        """Generate fallback predictions when main method fails"""
        predictions = self._simple_linear_prediction(data, steps)
        
        return {
            'predictions': predictions.tolist(),
            'confidence_intervals': [{'lower': float(p), 'upper': float(p), 'std_error': 0.0} for p in predictions],
            'quality_metrics': {'overall_quality_score': 0.3},
            'prediction_method': 'fallback',
            'pattern_preservation_score': 0.3
        }