"""
Advanced pH Prediction Engine with Historical Pattern Learning
Real-time continuous predictions that follow historical data patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from enhanced_pattern_learning import PatternAnalyzer, EnhancedPatternPredictor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class AdvancedPhPredictionEngine:
    """Advanced pH prediction engine with pattern learning capabilities"""
    
    def __init__(self):
        self.pattern_predictor = None
        self.historical_data = None
        self.pattern_analysis = None
        self.scaler = MinMaxScaler()
        self.fitted = False
        
        # Prediction parameters - smaller values for stability
        self.prediction_params = {
            'seq_len': 10,  # Reduced from 20
            'hidden_size': 32,  # Reduced from 64
            'num_layers': 1,  # Reduced from 2
            'batch_size': 16,  # Reduced from 32
            'epochs': 50,  # Reduced from 100
            'learning_rate': 0.01  # Increased for faster convergence
        }
        
        # Pattern preservation settings
        self.pattern_preservation = {
            'maintain_variance': True,
            'preserve_trends': True,
            'follow_cycles': True,
            'keep_bounds': True,
            'smooth_transitions': True
        }
    
    def fit(self, data: np.ndarray, timestamps: Optional[List] = None) -> Dict[str, Any]:
        """
        Fit the advanced prediction model with comprehensive pattern learning
        """
        if len(data) < self.prediction_params['seq_len']:
            # Adjust sequence length if data is too small
            self.prediction_params['seq_len'] = max(3, len(data) // 2)
        
        # Store historical data
        self.historical_data = data.copy()
        
        # Initialize pattern predictor
        self.pattern_predictor = EnhancedPatternPredictor(
            seq_len=self.prediction_params['seq_len'],
            hidden_size=self.prediction_params['hidden_size'],
            num_layers=self.prediction_params['num_layers']
        )
        
        logger.info(f"Training pattern-aware model with {len(data)} data points")
        
        try:
            # Fit the model
            training_results = self.pattern_predictor.fit(
                data,
                epochs=self.prediction_params['epochs'],
                batch_size=self.prediction_params['batch_size'],
                learning_rate=self.prediction_params['learning_rate']
            )
            
            # Store pattern analysis
            self.pattern_analysis = training_results['patterns']
            self.fitted = True
            
            # Evaluate model performance
            evaluation = self._evaluate_model(data)
            
            return {
                'training_results': training_results,
                'evaluation': evaluation,
                'pattern_analysis': self.pattern_analysis,
                'model_info': {
                    'data_points': len(data),
                    'seq_len': self.prediction_params['seq_len'],
                    'model_type': 'pattern_aware_lstm'
                }
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            # Return simplified result on error
            return {
                'training_results': {'error': str(e)},
                'evaluation': {'error': 'Training failed'},
                'pattern_analysis': {},
                'model_info': {
                    'data_points': len(data),
                    'seq_len': self.prediction_params['seq_len'],
                    'model_type': 'pattern_aware_lstm',
                    'error': str(e)
                }
            }
    
    def predict_continuous(self, steps: int = 30, 
                          maintain_patterns: bool = True) -> Dict[str, Any]:
        """
        Generate continuous predictions that follow historical patterns
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if self.historical_data is None:
            raise ValueError("No historical data available")
        
        # Generate predictions
        predictions = self.pattern_predictor.predict_next_steps(
            self.historical_data, steps=steps
        )
        
        # Apply additional pattern preservation if requested
        if maintain_patterns:
            predictions = self._apply_pattern_preservation(predictions)
        
        # Create timestamps
        timestamps = self._generate_timestamps(steps)
        
        # Calculate prediction metrics
        metrics = self._calculate_prediction_metrics(predictions)
        
        return {
            'predictions': predictions.tolist(),
            'timestamps': timestamps,
            'metrics': metrics,
            'pattern_analysis': self.pattern_analysis,
            'historical_continuity': self._check_historical_continuity(predictions)
        }
    
    def extend_predictions(self, additional_steps: int = 5) -> Dict[str, Any]:
        """
        Extend existing predictions with additional steps
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Generate additional predictions
        extended_predictions = self.pattern_predictor.predict_next_steps(
            self.historical_data, steps=additional_steps
        )
        
        # Apply pattern preservation
        extended_predictions = self._apply_pattern_preservation(extended_predictions)
        
        # Generate timestamps
        timestamps = self._generate_timestamps(additional_steps)
        
        return {
            'predictions': extended_predictions.tolist(),
            'timestamps': timestamps,
            'extension_info': {
                'steps': additional_steps,
                'continuity_score': self._calculate_continuity_score(extended_predictions)
            }
        }
    
    def analyze_prediction_quality(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the quality of predictions in terms of pattern preservation
        """
        if self.historical_data is None:
            return {}
        
        analysis = {}
        
        # Historical pattern compliance
        analysis['pattern_compliance'] = self._check_pattern_compliance(predictions)
        
        # Variance preservation
        historical_variance = np.var(self.historical_data)
        prediction_variance = np.var(predictions)
        analysis['variance_ratio'] = prediction_variance / historical_variance
        
        # Trend consistency
        analysis['trend_consistency'] = self._check_trend_consistency(predictions)
        
        # Smoothness assessment
        analysis['smoothness'] = self._assess_smoothness(predictions)
        
        # Realistic bounds
        analysis['bounds_compliance'] = self._check_bounds_compliance(predictions)
        
        return analysis
    
    def _evaluate_model(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on historical data
        """
        if len(data) < self.prediction_params['seq_len'] + 10:
            return {'error': 'Not enough data for evaluation'}
        
        # Split data for evaluation
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Create temporary predictor for evaluation
        temp_predictor = EnhancedPatternPredictor(
            seq_len=self.prediction_params['seq_len'],
            hidden_size=self.prediction_params['hidden_size'],
            num_layers=self.prediction_params['num_layers']
        )
        
        # Fit on training data
        temp_predictor.fit(train_data, epochs=50)  # Fewer epochs for evaluation
        
        # Predict on test data
        predictions = temp_predictor.predict_next_steps(train_data, steps=len(test_data))
        
        # Calculate metrics
        mse = mean_squared_error(test_data, predictions)
        r2 = r2_score(test_data, predictions)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2,
            'mean_absolute_error': np.mean(np.abs(test_data - predictions)),
            'prediction_variance': np.var(predictions),
            'historical_variance': np.var(test_data)
        }
    
    def _apply_pattern_preservation(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply additional pattern preservation techniques
        """
        if self.historical_data is None:
            return predictions
        
        preserved_predictions = predictions.copy()
        
        # Maintain variance
        if self.pattern_preservation['maintain_variance']:
            preserved_predictions = self._preserve_variance(preserved_predictions)
        
        # Preserve trends
        if self.pattern_preservation['preserve_trends']:
            preserved_predictions = self._preserve_trends(preserved_predictions)
        
        # Follow cycles
        if self.pattern_preservation['follow_cycles']:
            preserved_predictions = self._follow_cycles(preserved_predictions)
        
        # Keep bounds
        if self.pattern_preservation['keep_bounds']:
            preserved_predictions = self._enforce_bounds(preserved_predictions)
        
        # Smooth transitions
        if self.pattern_preservation['smooth_transitions']:
            preserved_predictions = self._smooth_transitions(preserved_predictions)
        
        return preserved_predictions
    
    def _preserve_variance(self, predictions: np.ndarray) -> np.ndarray:
        """
        Preserve historical variance in predictions
        """
        historical_std = np.std(self.historical_data)
        prediction_std = np.std(predictions)
        
        if prediction_std > 0:
            # Scale predictions to match historical variance
            variance_ratio = historical_std / prediction_std
            
            # Apply scaling with some smoothing
            if variance_ratio > 1.5 or variance_ratio < 0.5:
                prediction_mean = np.mean(predictions)
                scaled_predictions = (predictions - prediction_mean) * variance_ratio * 0.8 + prediction_mean
                return scaled_predictions
        
        return predictions
    
    def _preserve_trends(self, predictions: np.ndarray) -> np.ndarray:
        """
        Preserve trend characteristics from historical data
        """
        if self.pattern_analysis is None:
            return predictions
        
        trends = self.pattern_analysis.get('trends', {})
        recent_trend = trends.get('trend_5', 0)
        
        if abs(recent_trend) > 0.01:  # Significant trend
            # Calculate prediction trend
            pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
            
            # Adjust if trend is too different
            trend_diff = recent_trend - pred_trend
            if abs(trend_diff) > abs(recent_trend) * 0.5:
                # Apply trend correction
                correction = np.linspace(0, trend_diff * len(predictions), len(predictions))
                predictions = predictions + correction * 0.3
        
        return predictions
    
    def _follow_cycles(self, predictions: np.ndarray) -> np.ndarray:
        """
        Make predictions follow cyclical patterns
        """
        if self.pattern_analysis is None:
            return predictions
        
        cycles = self.pattern_analysis.get('cycles', {})
        avg_cycle_length = cycles.get('average_cycle_length', 0)
        
        if avg_cycle_length > 0:
            # Apply subtle cyclical adjustment
            cycle_adjustment = []
            for i in range(len(predictions)):
                cycle_pos = i % avg_cycle_length
                cycle_progress = cycle_pos / avg_cycle_length
                
                # Simple sinusoidal pattern
                cycle_value = 0.1 * np.sin(2 * np.pi * cycle_progress)
                cycle_adjustment.append(cycle_value)
            
            predictions = predictions + np.array(cycle_adjustment)
        
        return predictions
    
    def _enforce_bounds(self, predictions: np.ndarray) -> np.ndarray:
        """
        Keep predictions within realistic bounds
        """
        # Historical bounds
        hist_min = np.min(self.historical_data)
        hist_max = np.max(self.historical_data)
        hist_range = hist_max - hist_min
        
        # Allow some extension beyond historical bounds
        lower_bound = hist_min - 0.2 * hist_range
        upper_bound = hist_max + 0.2 * hist_range
        
        # pH specific bounds (pH typically ranges from 0-14, but for most systems 5-9)
        ph_min = max(5.0, lower_bound)
        ph_max = min(9.0, upper_bound)
        
        # Soft clipping to avoid abrupt changes
        predictions = np.clip(predictions, ph_min, ph_max)
        
        return predictions
    
    def _smooth_transitions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Smooth the transition from historical data to predictions
        """
        if len(self.historical_data) == 0:
            return predictions
        
        # Get last few historical points
        last_historical = self.historical_data[-min(5, len(self.historical_data)):]
        
        # Calculate transition smoothing
        transition_length = min(5, len(predictions))
        
        for i in range(transition_length):
            # Weight decreases from historical to prediction
            weight = (transition_length - i) / transition_length
            
            # Expected value based on historical trend
            if len(last_historical) > 1:
                trend = np.mean(np.diff(last_historical))
                expected = last_historical[-1] + trend * (i + 1)
                
                # Blend with prediction
                predictions[i] = weight * expected + (1 - weight) * predictions[i]
        
        return predictions
    
    def _generate_timestamps(self, steps: int) -> List[str]:
        """
        Generate timestamps for predictions
        """
        base_time = datetime.now()
        timestamps = []
        
        for i in range(steps):
            # Assuming 1-minute intervals for pH monitoring
            timestamp = base_time + timedelta(minutes=i)
            timestamps.append(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        
        return timestamps
    
    def _calculate_prediction_metrics(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Calculate various metrics for predictions
        """
        metrics = {}
        
        # Basic statistics
        metrics['mean'] = float(np.mean(predictions))
        metrics['std'] = float(np.std(predictions))
        metrics['min'] = float(np.min(predictions))
        metrics['max'] = float(np.max(predictions))
        metrics['range'] = float(metrics['max'] - metrics['min'])
        
        # Variability
        metrics['coefficient_of_variation'] = float(metrics['std'] / metrics['mean'] if metrics['mean'] != 0 else 0)
        
        # Trend analysis
        if len(predictions) > 1:
            trend = float(np.polyfit(range(len(predictions)), predictions, 1)[0])
            metrics['trend_slope'] = trend
            metrics['trend_direction'] = 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable'
        
        # Change characteristics
        if len(predictions) > 1:
            changes = np.diff(predictions)
            metrics['avg_change'] = float(np.mean(changes))
            metrics['change_volatility'] = float(np.std(changes))
            metrics['max_change'] = float(np.max(np.abs(changes)))
        
        return metrics
    
    def _check_historical_continuity(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Check how well predictions continue from historical data
        """
        if self.historical_data is None or len(self.historical_data) == 0:
            return {}
        
        last_historical = float(self.historical_data[-1])
        first_prediction = float(predictions[0])
        
        # Continuity gap
        continuity_gap = float(abs(first_prediction - last_historical))
        
        # Expected reasonable gap (based on historical changes)
        if len(self.historical_data) > 1:
            historical_changes = np.diff(self.historical_data)
            typical_change = float(np.mean(np.abs(historical_changes)))
            continuity_score = float(max(0, 1 - (continuity_gap / (typical_change * 2))))
        else:
            continuity_score = 1.0
        
        return {
            'continuity_gap': continuity_gap,
            'continuity_score': continuity_score,
            'last_historical': last_historical,
            'first_prediction': first_prediction
        }
    
    def _calculate_continuity_score(self, predictions: np.ndarray) -> float:
        """
        Calculate continuity score for predictions
        """
        if len(predictions) < 2:
            return 1.0
        
        # Check for smooth transitions
        changes = np.diff(predictions)
        avg_change = np.mean(np.abs(changes))
        max_change = np.max(np.abs(changes))
        
        # Continuity score based on change consistency
        if avg_change > 0:
            continuity_score = max(0, 1 - (max_change / (avg_change * 3)))
        else:
            continuity_score = 1.0
        
        return continuity_score
    
    def _check_pattern_compliance(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Check how well predictions comply with historical patterns
        """
        if self.pattern_analysis is None:
            return {}
        
        compliance = {}
        
        # Trend compliance
        historical_trends = self.pattern_analysis.get('trends', {})
        if 'trend_5' in historical_trends:
            pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
            hist_trend = historical_trends['trend_5']
            
            if abs(hist_trend) > 0.01:
                trend_compliance = 1 - abs(pred_trend - hist_trend) / abs(hist_trend)
                compliance['trend_compliance'] = max(0, trend_compliance)
        
        # Variance compliance
        historical_stats = self.pattern_analysis.get('statistical_props', {})
        if 'std' in historical_stats:
            pred_std = np.std(predictions)
            hist_std = historical_stats['std']
            
            if hist_std > 0:
                variance_compliance = 1 - abs(pred_std - hist_std) / hist_std
                compliance['variance_compliance'] = max(0, variance_compliance)
        
        return compliance
    
    def _check_trend_consistency(self, predictions: np.ndarray) -> float:
        """
        Check trend consistency in predictions
        """
        if len(predictions) < 3:
            return 1.0
        
        # Calculate local trends
        window_size = min(5, len(predictions) // 3)
        trends = []
        
        for i in range(0, len(predictions) - window_size + 1, window_size):
            segment = predictions[i:i + window_size]
            trend = np.polyfit(range(len(segment)), segment, 1)[0]
            trends.append(trend)
        
        if len(trends) < 2:
            return 1.0
        
        # Measure consistency
        trend_std = np.std(trends)
        trend_mean = np.mean(np.abs(trends))
        
        if trend_mean > 0:
            consistency = max(0, 1 - (trend_std / trend_mean))
        else:
            consistency = 1.0
        
        return consistency
    
    def _assess_smoothness(self, predictions: np.ndarray) -> float:
        """
        Assess smoothness of predictions
        """
        if len(predictions) < 3:
            return 1.0
        
        # Calculate second differences (acceleration)
        first_diff = np.diff(predictions)
        second_diff = np.diff(first_diff)
        
        # Smoothness based on second differences
        if len(second_diff) > 0:
            smoothness = 1 / (1 + np.mean(np.abs(second_diff)))
        else:
            smoothness = 1.0
        
        return smoothness
    
    def _check_bounds_compliance(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Check if predictions stay within reasonable bounds
        """
        if self.historical_data is None:
            return {}
        
        hist_min = float(np.min(self.historical_data))
        hist_max = float(np.max(self.historical_data))
        hist_range = hist_max - hist_min
        
        # Extended bounds
        lower_bound = hist_min - 0.5 * hist_range
        upper_bound = hist_max + 0.5 * hist_range
        
        # Check compliance
        within_bounds = bool(np.all((predictions >= lower_bound) & (predictions <= upper_bound)))
        outlier_count = int(np.sum((predictions < lower_bound) | (predictions > upper_bound)))
        
        return {
            'within_bounds': within_bounds,
            'outlier_count': outlier_count,
            'outlier_percentage': float(outlier_count / len(predictions) * 100),
            'bounds': {
                'lower': float(lower_bound),
                'upper': float(upper_bound)
            }
        }