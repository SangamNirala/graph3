"""
Enhanced Model Evaluation and Performance Metrics
Comprehensive evaluation framework for time series forecasting models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')


class TimeSeriesEvaluator:
    """
    Comprehensive evaluation framework for time series models
    Includes multiple metrics, backtesting, and visualization
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.backtesting_results = {}
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic forecasting metrics"""
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2': -float('inf'),
                'mape': float('inf'),
                'mse': float('inf')
            }
        
        # Basic metrics
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'mse': float(mse)
        }
    
    def calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate advanced forecasting metrics"""
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {}
        
        advanced_metrics = {}
        
        # Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = 200 * np.mean(np.abs(y_pred_clean - y_true_clean) / 
                             (np.abs(y_true_clean) + np.abs(y_pred_clean) + 1e-8))
        advanced_metrics['smape'] = float(smape)
        
        # Mean Absolute Scaled Error (MASE)
        # Use naive forecast as baseline
        if len(y_true_clean) > 1:
            naive_forecast = y_true_clean[:-1]
            naive_mae = np.mean(np.abs(y_true_clean[1:] - naive_forecast))
            if naive_mae > 0:
                current_mae = mean_absolute_error(y_true_clean, y_pred_clean)
                mase = current_mae / naive_mae
                advanced_metrics['mase'] = float(mase)
        
        # Directional Accuracy
        if len(y_true_clean) > 1:
            true_direction = np.sign(np.diff(y_true_clean))
            pred_direction = np.sign(np.diff(y_pred_clean))
            directional_accuracy = np.mean(true_direction == pred_direction)
            advanced_metrics['directional_accuracy'] = float(directional_accuracy)
        
        # Theil's U statistic
        numerator = np.sqrt(np.mean((y_pred_clean - y_true_clean) ** 2))
        denominator = np.sqrt(np.mean(y_true_clean ** 2)) + np.sqrt(np.mean(y_pred_clean ** 2))
        if denominator > 0:
            theil_u = numerator / denominator
            advanced_metrics['theil_u'] = float(theil_u)
        
        # Forecast Bias
        forecast_bias = np.mean(y_pred_clean - y_true_clean)
        advanced_metrics['forecast_bias'] = float(forecast_bias)
        
        # Forecast Accuracy
        forecast_accuracy = 1 - np.mean(np.abs(y_pred_clean - y_true_clean) / 
                                      (np.abs(y_true_clean) + 1e-8))
        advanced_metrics['forecast_accuracy'] = float(forecast_accuracy)
        
        return advanced_metrics
    
    def calculate_confidence_intervals(self, y_pred: np.ndarray, 
                                     prediction_errors: np.ndarray, 
                                     confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        alpha = 1 - confidence_level
        
        # Calculate standard error
        std_error = np.std(prediction_errors)
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
        
        lower_bound = y_pred - z_score * std_error
        upper_bound = y_pred + z_score * std_error
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std_error': std_error,
            'confidence_level': confidence_level
        }
    
    def generate_performance_report(self, model_name: str, 
                                  evaluation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report"""
        report = f"""
# Time Series Forecasting Performance Report
## Model: {model_name}
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Basic Metrics
"""
        
        if 'basic_metrics' in evaluation_results:
            basic_metrics = evaluation_results['basic_metrics']
            report += f"""
- **RMSE**: {basic_metrics.get('rmse', 'N/A'):.4f}
- **MAE**: {basic_metrics.get('mae', 'N/A'):.4f}
- **R²**: {basic_metrics.get('r2', 'N/A'):.4f}
- **MAPE**: {basic_metrics.get('mape', 'N/A'):.2f}%
- **MSE**: {basic_metrics.get('mse', 'N/A'):.4f}
"""
        
        if 'advanced_metrics' in evaluation_results:
            advanced_metrics = evaluation_results['advanced_metrics']
            report += f"""
### Advanced Metrics
- **SMAPE**: {advanced_metrics.get('smape', 'N/A'):.2f}%
- **MASE**: {advanced_metrics.get('mase', 'N/A'):.4f}
- **Directional Accuracy**: {advanced_metrics.get('directional_accuracy', 'N/A'):.2f}%
- **Theil's U**: {advanced_metrics.get('theil_u', 'N/A'):.4f}
- **Forecast Bias**: {advanced_metrics.get('forecast_bias', 'N/A'):.4f}
- **Forecast Accuracy**: {advanced_metrics.get('forecast_accuracy', 'N/A'):.2f}%
"""
        
        return report
    
    def create_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of evaluation results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_status': 'completed',
            'key_metrics': {},
            'performance_grade': 'N/A',
            'recommendations': []
        }
        
        # Extract key metrics
        if 'basic_metrics' in evaluation_results:
            basic = evaluation_results['basic_metrics']
            summary['key_metrics'].update({
                'rmse': basic.get('rmse', None),
                'mae': basic.get('mae', None),
                'r2': basic.get('r2', None),
                'mape': basic.get('mape', None)
            })
        
        # Performance grading
        r2 = summary['key_metrics'].get('r2', 0)
        mape = summary['key_metrics'].get('mape', 100)
        
        if r2 > 0.8 and mape < 10:
            summary['performance_grade'] = 'A'
        elif r2 > 0.6 and mape < 20:
            summary['performance_grade'] = 'B'
        elif r2 > 0.4 and mape < 30:
            summary['performance_grade'] = 'C'
        elif r2 > 0.2 and mape < 50:
            summary['performance_grade'] = 'D'
        else:
            summary['performance_grade'] = 'F'
        
        return summary


def evaluate_model_performance(model, data: pd.DataFrame, time_col: str, 
                             target_col: str) -> Dict[str, Any]:
    """
    Main function to evaluate model performance comprehensively
    
    Args:
        model: Trained model instance
        data: Input data
        time_col: Time column name
        target_col: Target column name
        
    Returns:
        Comprehensive evaluation results
    """
    evaluator = TimeSeriesEvaluator()
    evaluation_results = {
        'model_type': getattr(model, 'model_type', 'unknown'),
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    try:
        # Prepare data
        data_dict = model.prepare_data(data, time_col, target_col)
        
        # Make predictions on test set
        predictions = model.predict(data_dict['X_test'])
        
        # Calculate basic metrics
        basic_metrics = evaluator.calculate_basic_metrics(
            data_dict['y_test'].flatten(), predictions.flatten()
        )
        evaluation_results['basic_metrics'] = basic_metrics
        
        # Calculate advanced metrics
        advanced_metrics = evaluator.calculate_advanced_metrics(
            data_dict['y_test'].flatten(), predictions.flatten()
        )
        evaluation_results['advanced_metrics'] = advanced_metrics
        
        # Calculate confidence intervals
        prediction_errors = predictions.flatten() - data_dict['y_test'].flatten()
        confidence_intervals = evaluator.calculate_confidence_intervals(
            predictions.flatten(), prediction_errors
        )
        evaluation_results['confidence_intervals'] = {
            'lower_bound': confidence_intervals['lower_bound'].tolist(),
            'upper_bound': confidence_intervals['upper_bound'].tolist(),
            'std_error': float(confidence_intervals['std_error'])
        }
        
        # Generate performance report
        performance_report = evaluator.generate_performance_report(
            evaluation_results['model_type'], evaluation_results
        )
        evaluation_results['performance_report'] = performance_report
        
        # Create summary
        evaluation_summary = evaluator.create_evaluation_summary(evaluation_results)
        evaluation_results['evaluation_summary'] = evaluation_summary
        
    except Exception as e:
        evaluation_results['error'] = str(e)
        evaluation_results['evaluation_status'] = 'failed'
    
    return evaluation_results