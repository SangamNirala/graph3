"""
Enhanced Data Preprocessing for Time Series Forecasting
Advanced preprocessing techniques for better model performance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import signal
from scipy.stats import zscore
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class AdvancedTimeSeriesPreprocessor:
    """
    Advanced preprocessing pipeline for time series data
    Includes noise reduction, feature engineering, and data augmentation
    """
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scaler = None
        self.fitted = False
        self.feature_columns = []
        self.preprocessing_stats = {}
        
    def detect_outliers(self, data: np.ndarray, method: str = 'zscore', 
                       threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using various methods"""
        if method == 'zscore':
            z_scores = np.abs(zscore(data))
            return z_scores > threshold
        elif method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        elif method == 'modified_zscore':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores) > threshold
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def remove_noise(self, data: np.ndarray, method: str = 'savgol', 
                    **kwargs) -> np.ndarray:
        """Remove noise from time series data"""
        if method == 'savgol':
            # Savitzky-Golay filter
            window_length = kwargs.get('window_length', min(51, len(data) // 4))
            if window_length % 2 == 0:
                window_length += 1
            polyorder = kwargs.get('polyorder', 3)
            
            # Ensure window_length is valid
            if window_length > len(data):
                window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
            if window_length < polyorder + 1:
                window_length = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
            
            return signal.savgol_filter(data, window_length, polyorder)
        
        elif method == 'gaussian':
            # Gaussian filter
            sigma = kwargs.get('sigma', 1.0)
            return signal.gaussian_filter1d(data, sigma)
        
        elif method == 'median':
            # Median filter
            kernel_size = kwargs.get('kernel_size', 5)
            return signal.medfilt(data, kernel_size)
        
        elif method == 'moving_average':
            # Moving average
            window = kwargs.get('window', 5)
            return pd.Series(data).rolling(window=window, center=True).mean().fillna(data)
        
        elif method == 'exponential_smoothing':
            # Exponential smoothing
            alpha = kwargs.get('alpha', 0.3)
            smoothed = np.zeros_like(data)
            smoothed[0] = data[0]
            for i in range(1, len(data)):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            return smoothed
        
        else:
            raise ValueError(f"Unknown noise removal method: {method}")
    
    def create_features(self, data: pd.DataFrame, time_col: str, 
                       target_col: str) -> pd.DataFrame:
        """Create advanced features for time series"""
        # Ensure data is sorted by time
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Convert time column to datetime
        data[time_col] = pd.to_datetime(data[time_col])
        
        # Create a copy for feature engineering
        features_df = data.copy()
        
        # Time-based features
        features_df['hour'] = features_df[time_col].dt.hour
        features_df['day_of_week'] = features_df[time_col].dt.dayofweek
        features_df['day_of_month'] = features_df[time_col].dt.day
        features_df['month'] = features_df[time_col].dt.month
        features_df['quarter'] = features_df[time_col].dt.quarter
        features_df['year'] = features_df[time_col].dt.year
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for time features
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        # Lag features
        target_values = features_df[target_col].values
        
        # Create multiple lag features
        lag_periods = [1, 2, 3, 5, 7, 14, 30]
        for lag in lag_periods:
            if lag < len(target_values):
                features_df[f'{target_col}_lag_{lag}'] = features_df[target_col].shift(lag)
        
        # Rolling statistics
        rolling_windows = [3, 7, 14, 30]
        for window in rolling_windows:
            if window < len(target_values):
                features_df[f'{target_col}_rolling_mean_{window}'] = features_df[target_col].rolling(window=window).mean()
                features_df[f'{target_col}_rolling_std_{window}'] = features_df[target_col].rolling(window=window).std()
                features_df[f'{target_col}_rolling_min_{window}'] = features_df[target_col].rolling(window=window).min()
                features_df[f'{target_col}_rolling_max_{window}'] = features_df[target_col].rolling(window=window).max()
        
        # Differencing features
        features_df[f'{target_col}_diff_1'] = features_df[target_col].diff(1)
        features_df[f'{target_col}_diff_2'] = features_df[target_col].diff(2)
        features_df[f'{target_col}_pct_change'] = features_df[target_col].pct_change()
        
        # Statistical features
        features_df[f'{target_col}_zscore'] = zscore(features_df[target_col])
        
        # Trend features
        features_df['trend'] = np.arange(len(features_df))
        
        # Seasonal decomposition features (if enough data)
        if len(features_df) >= 24:  # At least 24 periods
            # Simple seasonal decomposition
            seasonal_period = min(24, len(features_df) // 3)
            seasonal_mean = features_df[target_col].rolling(window=seasonal_period).mean()
            features_df[f'{target_col}_seasonal'] = features_df[target_col] - seasonal_mean
            features_df[f'{target_col}_trend'] = seasonal_mean
        
        # Volatility features
        features_df[f'{target_col}_volatility'] = features_df[target_col].rolling(window=7).std()
        
        # Interaction features
        features_df[f'{target_col}_trend_interaction'] = features_df[target_col] * features_df['trend']
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Store feature column names
        self.feature_columns = [col for col in features_df.columns if col not in [time_col, target_col]]
        
        return features_df
    
    def scale_data(self, data: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Scale data using various methods"""
        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            else:
                scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
                
        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            else:
                scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
                
        elif method == 'robust':
            if self.scaler is None:
                self.scaler = RobustScaler()
                scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            else:
                scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
                
        elif method == 'log':
            # Log transformation (add small constant to avoid log(0))
            scaled_data = np.log(data + 1e-8)
            
        elif method == 'box_cox':
            # Box-Cox transformation
            from scipy.stats import boxcox
            if np.all(data > 0):
                scaled_data, _ = boxcox(data + 1e-8)
            else:
                # Fall back to log transformation
                scaled_data = np.log(data - np.min(data) + 1)
                
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        return scaled_data
    
    def inverse_scale_data(self, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse scale data back to original scale"""
        if self.scaler is None:
            return scaled_data
        
        if self.method in ['standard', 'minmax', 'robust']:
            return self.scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()
        elif self.method == 'log':
            return np.exp(scaled_data) - 1e-8
        else:
            # For box-cox and other methods, return as-is (would need to store lambda)
            return scaled_data
    
    def augment_data(self, data: np.ndarray, augmentation_factor: int = 2) -> np.ndarray:
        """Augment time series data"""
        augmented_data = [data]
        
        for _ in range(augmentation_factor - 1):
            # Add noise
            noise_level = np.std(data) * 0.05  # 5% noise
            noisy_data = data + np.random.normal(0, noise_level, size=data.shape)
            augmented_data.append(noisy_data)
            
            # Time warping (simple stretching/compressing)
            stretch_factor = np.random.uniform(0.9, 1.1)
            warped_indices = np.linspace(0, len(data) - 1, int(len(data) * stretch_factor))
            warped_data = np.interp(np.arange(len(data)), warped_indices, data)
            augmented_data.append(warped_data)
        
        return np.concatenate(augmented_data)
    
    def preprocess_pipeline(self, data: pd.DataFrame, time_col: str, 
                          target_col: str, **kwargs) -> Dict[str, Any]:
        """Complete preprocessing pipeline"""
        results = {}
        
        # Step 1: Basic cleaning
        data_clean = data.dropna().copy()
        results['initial_shape'] = data.shape
        results['cleaned_shape'] = data_clean.shape
        
        # Step 2: Outlier detection and handling
        target_values = data_clean[target_col].values
        outliers = self.detect_outliers(target_values, 
                                       method=kwargs.get('outlier_method', 'zscore'),
                                       threshold=kwargs.get('outlier_threshold', 3.0))
        
        # Replace outliers with median
        if np.any(outliers):
            median_value = np.median(target_values[~outliers])
            data_clean.loc[outliers, target_col] = median_value
            results['outliers_replaced'] = np.sum(outliers)
        else:
            results['outliers_replaced'] = 0
        
        # Step 3: Noise reduction
        if kwargs.get('denoise', True):
            denoised_values = self.remove_noise(
                data_clean[target_col].values,
                method=kwargs.get('denoise_method', 'savgol')
            )
            data_clean[target_col] = denoised_values
            results['denoised'] = True
        else:
            results['denoised'] = False
        
        # Step 4: Feature engineering
        if kwargs.get('create_features', True):
            data_features = self.create_features(data_clean, time_col, target_col)
            results['features_created'] = len(self.feature_columns)
        else:
            data_features = data_clean
            results['features_created'] = 0
        
        # Step 5: Scaling
        target_values_final = data_features[target_col].values
        scaled_values = self.scale_data(target_values_final, method=self.method)
        results['scaling_method'] = self.method
        results['scaling_stats'] = {
            'mean': np.mean(scaled_values),
            'std': np.std(scaled_values),
            'min': np.min(scaled_values),
            'max': np.max(scaled_values)
        }
        
        # Step 6: Data augmentation (if requested)
        if kwargs.get('augment', False):
            augmented_values = self.augment_data(
                scaled_values,
                augmentation_factor=kwargs.get('augmentation_factor', 2)
            )
            results['augmented'] = True
            results['augmentation_factor'] = kwargs.get('augmentation_factor', 2)
        else:
            augmented_values = scaled_values
            results['augmented'] = False
        
        # Store preprocessing statistics
        self.preprocessing_stats = results
        self.fitted = True
        
        return {
            'processed_data': data_features,
            'scaled_target': scaled_values,
            'augmented_target': augmented_values,
            'preprocessing_stats': results,
            'time_col': time_col,
            'target_col': target_col
        }
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps"""
        return {
            'method': self.method,
            'fitted': self.fitted,
            'feature_columns': self.feature_columns,
            'preprocessing_stats': self.preprocessing_stats
        }


class TimeSeriesValidator:
    """Validate time series data quality and provide recommendations"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_data_quality(self, data: pd.DataFrame, time_col: str, 
                            target_col: str) -> Dict[str, Any]:
        """Comprehensive data quality validation"""
        validation = {}
        
        # Basic statistics
        validation['total_rows'] = len(data)
        validation['missing_values'] = data.isnull().sum().to_dict()
        validation['duplicate_rows'] = data.duplicated().sum()
        
        # Time series specific checks
        time_series = data[time_col]
        target_series = data[target_col]
        
        # Time column validation
        validation['time_column'] = {
            'dtype': str(time_series.dtype),
            'unique_values': time_series.nunique(),
            'is_sorted': time_series.is_monotonic_increasing,
            'time_range': {
                'start': str(time_series.min()),
                'end': str(time_series.max())
            }
        }
        
        # Target column validation
        validation['target_column'] = {
            'dtype': str(target_series.dtype),
            'mean': float(target_series.mean()),
            'std': float(target_series.std()),
            'min': float(target_series.min()),
            'max': float(target_series.max()),
            'zeros': int((target_series == 0).sum()),
            'negative_values': int((target_series < 0).sum())
        }
        
        # Stationarity check (simple)
        diff_series = target_series.diff().dropna()
        validation['stationarity'] = {
            'mean_diff': float(diff_series.mean()),
            'std_diff': float(diff_series.std()),
            'likely_stationary': abs(diff_series.mean()) < 0.1 * target_series.std()
        }
        
        # Seasonality detection (simple)
        if len(data) >= 24:
            autocorr_1 = target_series.autocorr(lag=1)
            autocorr_7 = target_series.autocorr(lag=7) if len(data) >= 7 else 0
            autocorr_24 = target_series.autocorr(lag=24) if len(data) >= 24 else 0
            
            validation['seasonality'] = {
                'autocorr_lag_1': float(autocorr_1) if not np.isnan(autocorr_1) else 0,
                'autocorr_lag_7': float(autocorr_7) if not np.isnan(autocorr_7) else 0,
                'autocorr_lag_24': float(autocorr_24) if not np.isnan(autocorr_24) else 0,
                'has_seasonality': abs(autocorr_7) > 0.3 or abs(autocorr_24) > 0.3
            }
        
        # Data quality score
        quality_score = self.calculate_quality_score(validation)
        validation['quality_score'] = quality_score
        validation['recommendations'] = self.generate_recommendations(validation)
        
        self.validation_results = validation
        return validation
    
    def calculate_quality_score(self, validation: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100
        
        # Deduct points for missing values
        missing_ratio = sum(validation['missing_values'].values()) / validation['total_rows']
        score -= missing_ratio * 30
        
        # Deduct points for duplicates
        duplicate_ratio = validation['duplicate_rows'] / validation['total_rows']
        score -= duplicate_ratio * 20
        
        # Deduct points if not sorted
        if not validation['time_column']['is_sorted']:
            score -= 10
        
        # Deduct points for poor target distribution
        target_stats = validation['target_column']
        if target_stats['std'] == 0:
            score -= 30  # No variation
        
        # Bonus points for stationarity
        if 'stationarity' in validation and validation['stationarity']['likely_stationary']:
            score += 10
        
        # Bonus points for seasonality
        if 'seasonality' in validation and validation['seasonality']['has_seasonality']:
            score += 10
        
        return max(0, min(100, score))
    
    def generate_recommendations(self, validation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Missing values
        if sum(validation['missing_values'].values()) > 0:
            recommendations.append("Handle missing values using interpolation or forward fill")
        
        # Duplicates
        if validation['duplicate_rows'] > 0:
            recommendations.append("Remove duplicate rows")
        
        # Sorting
        if not validation['time_column']['is_sorted']:
            recommendations.append("Sort data by time column")
        
        # Stationarity
        if 'stationarity' in validation and not validation['stationarity']['likely_stationary']:
            recommendations.append("Consider differencing to make data stationary")
        
        # Outliers
        target_stats = validation['target_column']
        if target_stats['std'] > 3 * abs(target_stats['mean']):
            recommendations.append("Check for outliers and consider removal or transformation")
        
        # Data volume
        if validation['total_rows'] < 50:
            recommendations.append("Consider collecting more data for better model performance")
        
        return recommendations


def preprocess_time_series_data(data: pd.DataFrame, time_col: str, target_col: str,
                               preprocessing_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main function to preprocess time series data with all enhancements
    
    Args:
        data: Input DataFrame
        time_col: Name of time column
        target_col: Name of target column
        preprocessing_config: Configuration for preprocessing steps
        
    Returns:
        Dictionary with preprocessed data and statistics
    """
    # Default configuration
    default_config = {
        'scaling_method': 'standard',
        'outlier_method': 'zscore',
        'outlier_threshold': 3.0,
        'denoise': True,
        'denoise_method': 'savgol',
        'create_features': True,
        'augment': False,
        'augmentation_factor': 2
    }
    
    if preprocessing_config:
        default_config.update(preprocessing_config)
    
    # Validate data quality
    validator = TimeSeriesValidator()
    validation_results = validator.validate_data_quality(data, time_col, target_col)
    
    # Preprocess data
    preprocessor = AdvancedTimeSeriesPreprocessor(method=default_config['scaling_method'])
    preprocessing_results = preprocessor.preprocess_pipeline(
        data, time_col, target_col, **default_config
    )
    
    # Combine results
    final_results = {
        'validation_results': validation_results,
        'preprocessing_results': preprocessing_results,
        'preprocessor': preprocessor,
        'recommendations': validation_results['recommendations']
    }
    
    return final_results