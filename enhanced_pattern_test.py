#!/usr/bin/env python3
"""
Enhanced Pattern-Learning Prediction System Testing
Focus on testing the comprehensive pattern-learning improvements for continuous prediction downward bias resolution
"""

import requests
import json
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://909a9d1c-9da6-4ed6-bd0a-ff6c4fb747bb.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Enhanced Pattern-Learning System at: {API_BASE_URL}")

class EnhancedPatternTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        self.data_id = None
        self.model_id = None
        
    def create_pattern_data(self, pattern_type="linear", size=100):
        """Create different types of sensor data patterns for testing"""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='h')
        
        if pattern_type == "linear":
            # Linear upward trend with noise
            values = np.linspace(6.5, 7.5, size) + np.random.normal(0, 0.1, size)
            
        elif pattern_type == "sinusoidal":
            # Sinusoidal pattern (cyclical)
            t = np.linspace(0, 4*np.pi, size)
            values = 7.0 + 0.5 * np.sin(t) + np.random.normal(0, 0.05, size)
            
        elif pattern_type == "seasonal":
            # Seasonal pattern with daily cycles
            t = np.arange(size)
            daily_cycle = 0.3 * np.sin(2 * np.pi * t / 24)  # 24-hour cycle
            weekly_trend = 0.1 * np.sin(2 * np.pi * t / (24*7))  # Weekly trend
            values = 7.2 + daily_cycle + weekly_trend + np.random.normal(0, 0.08, size)
            
        elif pattern_type == "trending":
            # Non-linear trending pattern (quadratic)
            t = np.linspace(0, 1, size)
            values = 6.8 + 0.8 * t + 0.3 * t**2 + np.random.normal(0, 0.1, size)
            
        elif pattern_type == "stable":
            # Stable pattern around mean with small variations
            values = np.full(size, 7.3) + np.random.normal(0, 0.05, size)
            
        elif pattern_type == "volatile":
            # High volatility pattern
            values = 7.0 + np.cumsum(np.random.normal(0, 0.15, size))
            # Keep within reasonable pH bounds
            values = np.clip(values, 6.0, 8.0)
            
        # Ensure values are within realistic pH range
        values = np.clip(values, 5.5, 8.5)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph_value': values
        })
        
        return df
    
    def upload_pattern_data(self, pattern_type="linear"):
        """Upload pattern data and get analysis"""
        print(f"\n=== Uploading {pattern_type} pattern data ===")
        
        try:
            df = self.create_pattern_data(pattern_type, size=100)
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': (f'{pattern_type}_pattern.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                
                print(f"✅ {pattern_type} data uploaded successfully")
                print(f"   Data ID: {self.data_id}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                return True, df
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return False, None
                
        except Exception as e:
            print(f"❌ Upload error: {str(e)}")
            return False, None
    
    def train_arima_model(self):
        """Train ARIMA model for pattern learning"""
        print("\n=== Training ARIMA Model ===")
        
        if not self.data_id:
            print("❌ No data uploaded")
            return False
            
        try:
            training_params = {
                "time_column": "timestamp",
                "target_column": "ph_value",
                "order": [1, 1, 1]
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": "arima"},
                json=training_params
            )
            
            if response.status_code == 200:
                data = response.json()
                self.model_id = data.get('model_id')
                print(f"✅ ARIMA model trained successfully")
                print(f"   Model ID: {self.model_id}")
                return True
            else:
                print(f"❌ Training failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return False
    
    def test_continuous_prediction_bias(self, num_calls=10):
        """Test continuous prediction for downward bias - MAIN FOCUS"""
        print(f"\n=== Testing Continuous Prediction Bias ({num_calls} calls) ===")
        
        if not self.model_id:
            print("❌ No trained model")
            return False
            
        try:
            # Reset continuous predictions first
            reset_response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            if reset_response.status_code != 200:
                print(f"⚠️ Reset warning: {reset_response.status_code}")
            
            all_predictions = []
            timestamps = []
            
            for i in range(num_calls):
                print(f"   Call {i+1}/{num_calls}...")
                
                response = self.session.post(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    json={
                        "model_id": self.model_id,
                        "steps": 10,
                        "time_window": 50
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    
                    if predictions:
                        pred_values = [p['value'] for p in predictions]
                        pred_timestamps = [p['timestamp'] for p in predictions]
                        
                        all_predictions.extend(pred_values)
                        timestamps.extend(pred_timestamps)
                        
                        print(f"      Predictions: {len(pred_values)} values")
                        print(f"      Range: {min(pred_values):.3f} - {max(pred_values):.3f}")
                        print(f"      Mean: {np.mean(pred_values):.3f}")
                        
                        # Small delay between calls
                        time.sleep(0.5)
                    else:
                        print(f"      ❌ No predictions returned")
                        
                else:
                    print(f"      ❌ Call failed: {response.status_code}")
            
            # Analyze bias in continuous predictions
            if len(all_predictions) > 10:
                return self.analyze_prediction_bias(all_predictions, "Continuous Prediction")
            else:
                print("❌ Insufficient predictions for bias analysis")
                return False
                
        except Exception as e:
            print(f"❌ Continuous prediction test error: {str(e)}")
            return False
    
    def test_pattern_following(self, original_data):
        """Test if predictions follow historical patterns"""
        print("\n=== Testing Pattern Following ===")
        
        if not self.model_id:
            print("❌ No trained model")
            return False
            
        try:
            # Generate predictions
            response = self.session.post(
                f"{API_BASE_URL}/generate-prediction",
                json={
                    "model_id": self.model_id,
                    "steps": 30
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if predictions:
                    pred_values = [p['value'] for p in predictions]
                    historical_values = original_data['ph_value'].values
                    
                    # Analyze pattern following
                    return self.analyze_pattern_following(historical_values, pred_values)
                else:
                    print("❌ No predictions returned")
                    return False
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Pattern following test error: {str(e)}")
            return False
    
    def test_variability_preservation(self, original_data):
        """Test if predictions maintain realistic variability"""
        print("\n=== Testing Variability Preservation ===")
        
        if not self.model_id:
            print("❌ No trained model")
            return False
            
        try:
            # Generate multiple prediction sets
            all_predictions = []
            
            for i in range(5):
                response = self.session.post(
                    f"{API_BASE_URL}/generate-prediction",
                    json={
                        "model_id": self.model_id,
                        "steps": 20
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    
                    if predictions:
                        pred_values = [p['value'] for p in predictions]
                        all_predictions.extend(pred_values)
                
                time.sleep(0.3)  # Small delay
            
            if len(all_predictions) > 10:
                historical_values = original_data['ph_value'].values
                return self.analyze_variability_preservation(historical_values, all_predictions)
            else:
                print("❌ Insufficient predictions for variability analysis")
                return False
                
        except Exception as e:
            print(f"❌ Variability test error: {str(e)}")
            return False
    
    def analyze_prediction_bias(self, predictions, test_name):
        """Analyze predictions for downward bias"""
        print(f"\n--- {test_name} Bias Analysis ---")
        
        try:
            predictions = np.array(predictions)
            
            # Calculate trend slope
            x = np.arange(len(predictions))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, predictions)
            
            # Calculate statistics
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            min_pred = np.min(predictions)
            max_pred = np.max(predictions)
            
            print(f"   Predictions count: {len(predictions)}")
            print(f"   Mean: {mean_pred:.6f}")
            print(f"   Std: {std_pred:.6f}")
            print(f"   Range: {min_pred:.6f} - {max_pred:.6f}")
            print(f"   Trend slope: {slope:.6f}")
            print(f"   R-squared: {r_value**2:.6f}")
            print(f"   P-value: {p_value:.6f}")
            
            # Bias assessment
            bias_threshold = 0.01  # Acceptable slope threshold
            
            if abs(slope) <= bias_threshold:
                print(f"✅ NO DOWNWARD BIAS DETECTED (slope: {slope:.6f})")
                bias_result = True
            else:
                if slope < -bias_threshold:
                    print(f"❌ DOWNWARD BIAS DETECTED (slope: {slope:.6f})")
                else:
                    print(f"⚠️ UPWARD BIAS DETECTED (slope: {slope:.6f})")
                bias_result = False
            
            # Variability check
            if std_pred > 0.001:
                print(f"✅ Good variability (std: {std_pred:.6f})")
                variability_result = True
            else:
                print(f"❌ Low variability (std: {std_pred:.6f})")
                variability_result = False
            
            # pH range check
            if 5.5 <= min_pred <= max_pred <= 8.5:
                print(f"✅ Realistic pH range")
                range_result = True
            else:
                print(f"❌ Unrealistic pH range")
                range_result = False
            
            overall_result = bias_result and variability_result and range_result
            
            return {
                'overall': overall_result,
                'bias': bias_result,
                'variability': variability_result,
                'range': range_result,
                'slope': slope,
                'mean': mean_pred,
                'std': std_pred,
                'count': len(predictions)
            }
            
        except Exception as e:
            print(f"❌ Bias analysis error: {str(e)}")
            return {'overall': False, 'error': str(e)}
    
    def analyze_pattern_following(self, historical, predictions):
        """Analyze how well predictions follow historical patterns"""
        print("\n--- Pattern Following Analysis ---")
        
        try:
            hist_mean = np.mean(historical)
            hist_std = np.std(historical)
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            
            # Calculate pattern similarity metrics
            mean_deviation = abs(pred_mean - hist_mean) / hist_std
            std_ratio = pred_std / hist_std if hist_std > 0 else 0
            
            print(f"   Historical mean: {hist_mean:.6f}, std: {hist_std:.6f}")
            print(f"   Prediction mean: {pred_mean:.6f}, std: {pred_std:.6f}")
            print(f"   Mean deviation: {mean_deviation:.6f}")
            print(f"   Std ratio: {std_ratio:.6f}")
            
            # Pattern following assessment
            mean_threshold = 0.5  # Acceptable mean deviation
            std_threshold_low = 0.3  # Minimum std ratio
            std_threshold_high = 3.0  # Maximum std ratio
            
            mean_ok = mean_deviation <= mean_threshold
            std_ok = std_threshold_low <= std_ratio <= std_threshold_high
            
            if mean_ok and std_ok:
                print("✅ Good pattern following")
                result = True
            else:
                print("❌ Poor pattern following")
                result = False
            
            return {
                'result': result,
                'mean_deviation': mean_deviation,
                'std_ratio': std_ratio,
                'historical_stats': {'mean': hist_mean, 'std': hist_std},
                'prediction_stats': {'mean': pred_mean, 'std': pred_std}
            }
            
        except Exception as e:
            print(f"❌ Pattern following analysis error: {str(e)}")
            return {'result': False, 'error': str(e)}
    
    def analyze_variability_preservation(self, historical, predictions):
        """Analyze if predictions preserve realistic variability"""
        print("\n--- Variability Preservation Analysis ---")
        
        try:
            hist_std = np.std(historical)
            pred_std = np.std(predictions)
            
            # Calculate change variability
            hist_changes = np.diff(historical)
            pred_changes = np.diff(predictions)
            
            hist_change_std = np.std(hist_changes) if len(hist_changes) > 0 else 0
            pred_change_std = np.std(pred_changes) if len(pred_changes) > 0 else 0
            
            # Calculate variability ratios
            value_variability_ratio = pred_std / hist_std if hist_std > 0 else 0
            change_variability_ratio = pred_change_std / hist_change_std if hist_change_std > 0 else 0
            
            print(f"   Historical value std: {hist_std:.6f}")
            print(f"   Prediction value std: {pred_std:.6f}")
            print(f"   Value variability ratio: {value_variability_ratio:.6f}")
            print(f"   Historical change std: {hist_change_std:.6f}")
            print(f"   Prediction change std: {pred_change_std:.6f}")
            print(f"   Change variability ratio: {change_variability_ratio:.6f}")
            
            # Variability assessment
            value_threshold_low = 0.2
            value_threshold_high = 5.0
            change_threshold_low = 0.1
            change_threshold_high = 10.0
            
            value_ok = value_threshold_low <= value_variability_ratio <= value_threshold_high
            change_ok = change_threshold_low <= change_variability_ratio <= change_threshold_high
            
            if value_ok and change_ok:
                print("✅ Good variability preservation")
                result = True
            else:
                print("❌ Poor variability preservation")
                result = False
            
            return {
                'result': result,
                'value_variability_ratio': value_variability_ratio,
                'change_variability_ratio': change_variability_ratio,
                'historical_std': hist_std,
                'prediction_std': pred_std
            }
            
        except Exception as e:
            print(f"❌ Variability analysis error: {str(e)}")
            return {'result': False, 'error': str(e)}
    
    def test_sensor_data_adaptability(self):
        """Test system adaptability to different sensor data patterns"""
        print("\n=== Testing Sensor Data Adaptability ===")
        
        pattern_types = ["linear", "sinusoidal", "seasonal", "trending", "stable"]
        adaptability_results = {}
        
        for pattern_type in pattern_types:
            print(f"\n--- Testing {pattern_type} pattern ---")
            
            # Upload pattern data
            upload_success, original_data = self.upload_pattern_data(pattern_type)
            if not upload_success:
                adaptability_results[pattern_type] = {'success': False, 'error': 'Upload failed'}
                continue
            
            # Train model
            train_success = self.train_arima_model()
            if not train_success:
                adaptability_results[pattern_type] = {'success': False, 'error': 'Training failed'}
                continue
            
            # Test continuous prediction bias
            bias_result = self.test_continuous_prediction_bias(num_calls=5)
            
            # Test pattern following
            pattern_result = self.test_pattern_following(original_data)
            
            # Test variability preservation
            variability_result = self.test_variability_preservation(original_data)
            
            adaptability_results[pattern_type] = {
                'success': True,
                'bias_test': bias_result,
                'pattern_test': pattern_result,
                'variability_test': variability_result
            }
        
        return adaptability_results
    
    def run_comprehensive_test(self):
        """Run comprehensive enhanced pattern-learning test"""
        print("🎯 ENHANCED PATTERN-LEARNING PREDICTION SYSTEM TESTING")
        print("=" * 60)
        
        # Test 1: Continuous Prediction Downward Bias (Main Focus)
        print("\n🔬 TEST 1: CONTINUOUS PREDICTION DOWNWARD BIAS")
        upload_success, original_data = self.upload_pattern_data("linear")
        if upload_success and self.train_arima_model():
            bias_test_result = self.test_continuous_prediction_bias(num_calls=15)
            self.test_results['continuous_bias'] = bias_test_result
        else:
            self.test_results['continuous_bias'] = {'overall': False, 'error': 'Setup failed'}
        
        # Test 2: Pattern Following with Different Data Types
        print("\n🔬 TEST 2: PATTERN FOLLOWING")
        if original_data is not None:
            pattern_result = self.test_pattern_following(original_data)
            self.test_results['pattern_following'] = pattern_result
        else:
            self.test_results['pattern_following'] = {'result': False, 'error': 'No data'}
        
        # Test 3: Variability Preservation
        print("\n🔬 TEST 3: VARIABILITY PRESERVATION")
        if original_data is not None:
            variability_result = self.test_variability_preservation(original_data)
            self.test_results['variability_preservation'] = variability_result
        else:
            self.test_results['variability_preservation'] = {'result': False, 'error': 'No data'}
        
        # Test 4: Sensor Data Adaptability
        print("\n🔬 TEST 4: SENSOR DATA ADAPTABILITY")
        adaptability_results = self.test_sensor_data_adaptability()
        self.test_results['sensor_adaptability'] = adaptability_results
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("🎯 ENHANCED PATTERN-LEARNING SYSTEM TEST REPORT")
        print("=" * 60)
        
        # Test 1: Continuous Bias
        bias_result = self.test_results.get('continuous_bias', {})
        if isinstance(bias_result, dict) and bias_result.get('overall'):
            print("✅ CONTINUOUS PREDICTION BIAS: RESOLVED")
            print(f"   Slope: {bias_result.get('slope', 0):.6f}")
            print(f"   Variability: {bias_result.get('std', 0):.6f}")
        else:
            print("❌ CONTINUOUS PREDICTION BIAS: ISSUES DETECTED")
            if isinstance(bias_result, dict) and 'error' in bias_result:
                print(f"   Error: {bias_result['error']}")
        
        # Test 2: Pattern Following
        pattern_result = self.test_results.get('pattern_following', {})
        if isinstance(pattern_result, dict) and pattern_result.get('result'):
            print("✅ PATTERN FOLLOWING: WORKING")
        else:
            print("❌ PATTERN FOLLOWING: NEEDS IMPROVEMENT")
        
        # Test 3: Variability Preservation
        variability_result = self.test_results.get('variability_preservation', {})
        if isinstance(variability_result, dict) and variability_result.get('result'):
            print("✅ VARIABILITY PRESERVATION: WORKING")
        else:
            print("❌ VARIABILITY PRESERVATION: NEEDS IMPROVEMENT")
        
        # Test 4: Sensor Adaptability
        adaptability_results = self.test_results.get('sensor_adaptability', {})
        successful_patterns = sum(1 for result in adaptability_results.values() 
                                if result.get('success', False))
        total_patterns = len(adaptability_results)
        
        if successful_patterns >= total_patterns * 0.8:  # 80% success rate
            print(f"✅ SENSOR DATA ADAPTABILITY: WORKING ({successful_patterns}/{total_patterns})")
        else:
            print(f"❌ SENSOR DATA ADAPTABILITY: NEEDS IMPROVEMENT ({successful_patterns}/{total_patterns})")
        
        # Overall Assessment
        print("\n" + "-" * 40)
        print("OVERALL ASSESSMENT:")
        
        success_count = 0
        total_tests = 4
        
        if isinstance(bias_result, dict) and bias_result.get('overall'): success_count += 1
        if isinstance(pattern_result, dict) and pattern_result.get('result'): success_count += 1
        if isinstance(variability_result, dict) and variability_result.get('result'): success_count += 1
        if successful_patterns >= total_patterns * 0.8: success_count += 1
        
        success_rate = (success_count / total_tests) * 100
        
        print(f"Success Rate: {success_rate:.1f}% ({success_count}/{total_tests})")
        
        if success_rate >= 75:
            print("🎉 ENHANCED PATTERN-LEARNING SYSTEM: WORKING EXCELLENTLY!")
            print("   Downward bias issue has been resolved.")
            print("   System maintains historical patterns and variability.")
        elif success_rate >= 50:
            print("⚠️ ENHANCED PATTERN-LEARNING SYSTEM: PARTIALLY WORKING")
            print("   Some improvements needed for optimal performance.")
        else:
            print("❌ ENHANCED PATTERN-LEARNING SYSTEM: NEEDS SIGNIFICANT WORK")
            print("   Major issues detected that require attention.")
        
        return success_rate

if __name__ == "__main__":
    tester = EnhancedPatternTester()
    tester.run_comprehensive_test()
"""
Enhanced Pattern-Following Algorithm Testing
Tests the comprehensive bias correction, pattern-based prediction, trend stabilization, 
and error correction techniques to verify proper pattern following.
"""

import requests
import json
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://909a9d1c-9da6-4ed6-bd0a-ff6c4fb747bb.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing enhanced pattern-following algorithms at: {API_BASE_URL}")

class EnhancedPatternTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_ph_pattern_data(self, pattern_type="stable_with_trend"):
        """Create pH data with specific patterns for testing"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='H')
        
        if pattern_type == "stable_with_trend":
            # Stable pH with slight upward trend
            base_ph = 7.2
            trend = np.linspace(0, 0.3, 50)  # Slight upward trend
            noise = np.random.normal(0, 0.05, 50)  # Small noise
            ph_values = base_ph + trend + noise
            
        elif pattern_type == "cyclical":
            # Cyclical pH pattern (daily cycle)
            base_ph = 7.0
            cycle = 0.3 * np.sin(2 * np.pi * np.arange(50) / 24)  # Daily cycle
            trend = 0.1 * np.arange(50) / 50  # Small trend
            noise = np.random.normal(0, 0.03, 50)
            ph_values = base_ph + cycle + trend + noise
            
        elif pattern_type == "u_shaped":
            # U-shaped pattern (pH drops then recovers)
            x = np.linspace(-2, 2, 50)
            base_ph = 7.5
            u_shape = 0.2 * x**2  # U-shaped curve
            noise = np.random.normal(0, 0.04, 50)
            ph_values = base_ph - u_shape + noise
            
        elif pattern_type == "step_change":
            # Step change pattern
            ph_values = np.ones(50) * 7.0
            ph_values[25:] = 7.4  # Step up at midpoint
            noise = np.random.normal(0, 0.03, 50)
            ph_values += noise
            
        elif pattern_type == "complex_pattern":
            # Complex pattern with multiple components
            base_ph = 7.3
            trend = 0.15 * np.arange(50) / 50
            cycle1 = 0.2 * np.sin(2 * np.pi * np.arange(50) / 12)  # Short cycle
            cycle2 = 0.1 * np.sin(2 * np.pi * np.arange(50) / 30)  # Long cycle
            noise = np.random.normal(0, 0.04, 50)
            ph_values = base_ph + trend + cycle1 + cycle2 + noise
            
        # Ensure pH values are in realistic range
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'pH': ph_values
        })
        
        return df
    
    def test_multi_scale_pattern_analysis(self):
        """Test 1: Multi-scale pattern analysis functions"""
        print("\n=== Testing Multi-Scale Pattern Analysis ===")
        
        try:
            # Create complex pattern data
            df = self.create_ph_pattern_data("complex_pattern")
            csv_content = df.to_csv(index=False)
            
            # Upload data
            files = {'file': ('complex_pattern.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Data upload failed: {response.status_code}")
                self.test_results['multi_scale_analysis'] = False
                return
                
            data_id = response.json().get('data_id')
            print(f"✅ Complex pattern data uploaded (ID: {data_id})")
            
            # Train ARIMA model for pattern analysis
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
            )
            
            if response.status_code != 200:
                print(f"❌ Model training failed: {response.status_code}")
                self.test_results['multi_scale_analysis'] = False
                return
                
            model_id = response.json().get('model_id')
            print(f"✅ ARIMA model trained (ID: {model_id})")
            
            # Test continuous prediction with pattern analysis
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 20, "time_window": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                pattern_analysis = data.get('pattern_analysis', {})
                
                print("✅ Multi-scale pattern analysis successful")
                print(f"   Predictions generated: {len(predictions)}")
                
                # Check for pattern analysis components
                analysis_tests = []
                
                # Test for trend analysis
                if 'trend_slope' in pattern_analysis:
                    trend_slope = pattern_analysis['trend_slope']
                    print(f"   ✅ Trend slope detected: {trend_slope:.6f}")
                    analysis_tests.append(True)
                else:
                    print("   ❌ Trend slope missing")
                    analysis_tests.append(False)
                
                # Test for velocity analysis
                if 'velocity' in pattern_analysis:
                    velocity = pattern_analysis['velocity']
                    print(f"   ✅ Velocity analysis: {velocity:.6f}")
                    analysis_tests.append(True)
                else:
                    print("   ❌ Velocity analysis missing")
                    analysis_tests.append(False)
                
                # Test for pattern type detection
                if 'pattern_type' in pattern_analysis:
                    pattern_type = pattern_analysis['pattern_type']
                    print(f"   ✅ Pattern type detected: {pattern_type}")
                    analysis_tests.append(True)
                else:
                    print("   ❌ Pattern type detection missing")
                    analysis_tests.append(False)
                
                # Test for multi-scale components
                multi_scale_components = ['recent_trend', 'short_trend', 'overall_trend']
                multi_scale_detected = sum(1 for comp in multi_scale_components if comp in pattern_analysis)
                
                if multi_scale_detected >= 2:
                    print(f"   ✅ Multi-scale analysis: {multi_scale_detected}/3 components detected")
                    analysis_tests.append(True)
                else:
                    print(f"   ❌ Multi-scale analysis incomplete: {multi_scale_detected}/3 components")
                    analysis_tests.append(False)
                
                # Overall test result
                self.test_results['multi_scale_analysis'] = sum(analysis_tests) >= 3
                
            else:
                print(f"❌ Pattern analysis failed: {response.status_code}")
                self.test_results['multi_scale_analysis'] = False
                
        except Exception as e:
            print(f"❌ Multi-scale pattern analysis error: {str(e)}")
            self.test_results['multi_scale_analysis'] = False
    
    def test_bias_correction_historical_ranges(self):
        """Test 2: Enhanced bias correction maintains historical value ranges"""
        print("\n=== Testing Bias Correction & Historical Range Maintenance ===")
        
        try:
            # Create stable pH data with known range
            df = self.create_ph_pattern_data("stable_with_trend")
            historical_min = df['pH'].min()
            historical_max = df['pH'].max()
            historical_mean = df['pH'].mean()
            historical_std = df['pH'].std()
            
            print(f"   Historical pH range: {historical_min:.3f} - {historical_max:.3f}")
            print(f"   Historical mean: {historical_mean:.3f} ± {historical_std:.3f}")
            
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('stable_trend.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [1, 1, 1]}
            )
            model_id = response.json().get('model_id')
            
            # Test multiple prediction calls to check for bias accumulation
            bias_tests = []
            all_predictions = []
            
            for i in range(5):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 10, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    all_predictions.extend(predictions)
                    time.sleep(0.5)  # Small delay between calls
            
            if all_predictions:
                pred_min = min(all_predictions)
                pred_max = max(all_predictions)
                pred_mean = np.mean(all_predictions)
                pred_std = np.std(all_predictions)
                
                print(f"   Predicted pH range: {pred_min:.3f} - {pred_max:.3f}")
                print(f"   Predicted mean: {pred_mean:.3f} ± {pred_std:.3f}")
                
                # Test 1: Range maintenance (predictions within reasonable bounds)
                reasonable_lower = historical_min - 2 * historical_std
                reasonable_upper = historical_max + 2 * historical_std
                
                range_maintained = (pred_min >= reasonable_lower and pred_max <= reasonable_upper)
                print(f"   ✅ Range maintenance: {range_maintained}")
                bias_tests.append(range_maintained)
                
                # Test 2: Mean preservation (no significant bias)
                mean_deviation = abs(pred_mean - historical_mean)
                mean_preserved = mean_deviation <= historical_std
                print(f"   ✅ Mean preservation: {mean_preserved} (deviation: {mean_deviation:.3f})")
                bias_tests.append(mean_preserved)
                
                # Test 3: No downward bias (predictions don't consistently trend down)
                prediction_trend = np.polyfit(range(len(all_predictions)), all_predictions, 1)[0]
                no_downward_bias = prediction_trend > -0.01  # Allow small negative trend
                print(f"   ✅ No downward bias: {no_downward_bias} (trend: {prediction_trend:.6f})")
                bias_tests.append(no_downward_bias)
                
                # Test 4: Variability preservation
                variability_ratio = pred_std / historical_std
                variability_preserved = 0.5 <= variability_ratio <= 2.0  # Within reasonable range
                print(f"   ✅ Variability preservation: {variability_preserved} (ratio: {variability_ratio:.3f})")
                bias_tests.append(variability_preserved)
                
                self.test_results['bias_correction'] = sum(bias_tests) >= 3
                
            else:
                print("❌ No predictions generated")
                self.test_results['bias_correction'] = False
                
        except Exception as e:
            print(f"❌ Bias correction test error: {str(e)}")
            self.test_results['bias_correction'] = False
    
    def test_cyclical_pattern_detection(self):
        """Test 3: Improved cyclical pattern detection"""
        print("\n=== Testing Cyclical Pattern Detection ===")
        
        try:
            # Create cyclical pH data
            df = self.create_ph_pattern_data("cyclical")
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('cyclical_pattern.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
            )
            model_id = response.json().get('model_id')
            
            # Generate predictions and analyze for cyclical patterns
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 30, "time_window": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                pattern_analysis = data.get('pattern_analysis', {})
                
                cyclical_tests = []
                
                # Test 1: Cyclical pattern detection in analysis
                if 'pattern_type' in pattern_analysis:
                    pattern_type = pattern_analysis['pattern_type']
                    cyclical_detected = 'cycl' in pattern_type.lower() or 'complex' in pattern_type.lower()
                    print(f"   ✅ Pattern type detection: {pattern_type} (cyclical: {cyclical_detected})")
                    cyclical_tests.append(cyclical_detected)
                else:
                    print("   ❌ Pattern type not detected")
                    cyclical_tests.append(False)
                
                # Test 2: Predictions show cyclical behavior
                if len(predictions) >= 24:  # Need enough points to detect cycles
                    # Simple cycle detection: check for oscillations
                    diffs = np.diff(predictions)
                    sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i-1] < 0)
                    
                    # Expect some oscillations in cyclical data
                    cyclical_behavior = sign_changes >= 3
                    print(f"   ✅ Cyclical behavior in predictions: {cyclical_behavior} ({sign_changes} sign changes)")
                    cyclical_tests.append(cyclical_behavior)
                else:
                    print("   ❌ Insufficient predictions for cycle analysis")
                    cyclical_tests.append(False)
                
                # Test 3: Pattern continuation parameters
                if 'pattern_continuation' in pattern_analysis:
                    continuation = pattern_analysis['pattern_continuation']
                    has_continuation = isinstance(continuation, dict) and len(continuation) > 0
                    print(f"   ✅ Pattern continuation parameters: {has_continuation}")
                    cyclical_tests.append(has_continuation)
                else:
                    print("   ❌ Pattern continuation parameters missing")
                    cyclical_tests.append(False)
                
                self.test_results['cyclical_detection'] = sum(cyclical_tests) >= 2
                
            else:
                print(f"❌ Cyclical pattern prediction failed: {response.status_code}")
                self.test_results['cyclical_detection'] = False
                
        except Exception as e:
            print(f"❌ Cyclical pattern detection error: {str(e)}")
            self.test_results['cyclical_detection'] = False
    
    def test_adaptive_trend_decay(self):
        """Test 4: Adaptive trend decay follows historical trends better"""
        print("\n=== Testing Adaptive Trend Decay ===")
        
        try:
            # Create data with clear trend
            df = self.create_ph_pattern_data("stable_with_trend")
            historical_trend = np.polyfit(range(len(df)), df['pH'].values, 1)[0]
            print(f"   Historical trend slope: {historical_trend:.6f}")
            
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('trend_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [1, 1, 1]}
            )
            model_id = response.json().get('model_id')
            
            # Test trend decay over multiple prediction horizons
            trend_tests = []
            
            for steps in [10, 20, 30]:
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": steps, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    pattern_analysis = data.get('pattern_analysis', {})
                    
                    if predictions and len(predictions) >= 5:
                        # Calculate prediction trend
                        pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
                        
                        # Test trend consistency
                        trend_consistency = pattern_analysis.get('trend_consistency', 0)
                        
                        print(f"   Steps {steps}: pred_trend={pred_trend:.6f}, consistency={trend_consistency:.3f}")
                        
                        # Adaptive decay should maintain trend direction but with appropriate decay
                        trend_direction_maintained = (historical_trend * pred_trend >= 0)  # Same sign
                        reasonable_decay = abs(pred_trend) <= abs(historical_trend) * 2  # Not amplified too much
                        
                        trend_test = trend_direction_maintained and reasonable_decay
                        trend_tests.append(trend_test)
                        
                        print(f"   ✅ Trend decay test (steps {steps}): {trend_test}")
                    else:
                        trend_tests.append(False)
                else:
                    trend_tests.append(False)
            
            # Test adaptive decay parameters
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 20, "time_window": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                pattern_analysis = data.get('pattern_analysis', {})
                
                # Check for adaptive parameters
                adaptive_params = ['trend_consistency', 'stability_factor', 'bias_correction_factor']
                adaptive_detected = sum(1 for param in adaptive_params if param in pattern_analysis)
                
                print(f"   ✅ Adaptive parameters detected: {adaptive_detected}/3")
                trend_tests.append(adaptive_detected >= 2)
            
            self.test_results['adaptive_trend_decay'] = sum(trend_tests) >= len(trend_tests) * 0.7
            
        except Exception as e:
            print(f"❌ Adaptive trend decay test error: {str(e)}")
            self.test_results['adaptive_trend_decay'] = False
    
    def test_volatility_aware_adjustments(self):
        """Test 5: Volatility-aware adjustments maintain realistic variation"""
        print("\n=== Testing Volatility-Aware Adjustments ===")
        
        try:
            # Create data with specific volatility characteristics
            df = self.create_ph_pattern_data("complex_pattern")
            historical_volatility = df['pH'].std()
            historical_changes = np.diff(df['pH'].values)
            historical_change_std = np.std(historical_changes)
            
            print(f"   Historical volatility (std): {historical_volatility:.4f}")
            print(f"   Historical change volatility: {historical_change_std:.4f}")
            
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('volatility_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
            )
            model_id = response.json().get('model_id')
            
            # Generate multiple prediction sets to analyze volatility
            all_predictions = []
            volatility_tests = []
            
            for i in range(3):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 25, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    all_predictions.extend(predictions)
                    time.sleep(0.3)
            
            if all_predictions:
                pred_volatility = np.std(all_predictions)
                pred_changes = np.diff(all_predictions)
                pred_change_std = np.std(pred_changes)
                
                print(f"   Predicted volatility (std): {pred_volatility:.4f}")
                print(f"   Predicted change volatility: {pred_change_std:.4f}")
                
                # Test 1: Volatility preservation
                volatility_ratio = pred_volatility / historical_volatility
                volatility_preserved = 0.3 <= volatility_ratio <= 3.0  # Reasonable range
                print(f"   ✅ Volatility preservation: {volatility_preserved} (ratio: {volatility_ratio:.3f})")
                volatility_tests.append(volatility_preserved)
                
                # Test 2: Change volatility preservation
                change_ratio = pred_change_std / historical_change_std
                change_volatility_preserved = 0.2 <= change_ratio <= 4.0  # Reasonable range
                print(f"   ✅ Change volatility preservation: {change_volatility_preserved} (ratio: {change_ratio:.3f})")
                volatility_tests.append(change_volatility_preserved)
                
                # Test 3: Realistic variation (not too smooth, not too erratic)
                # Check for reasonable number of direction changes
                sign_changes = sum(1 for i in range(1, len(pred_changes)) if pred_changes[i] * pred_changes[i-1] < 0)
                expected_changes = len(pred_changes) * 0.2  # Expect some variation
                realistic_variation = sign_changes >= expected_changes
                print(f"   ✅ Realistic variation: {realistic_variation} ({sign_changes} changes, expected >= {expected_changes:.1f})")
                volatility_tests.append(realistic_variation)
                
                # Test 4: Check for volatility-aware parameters in analysis
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 15, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    pattern_analysis = data.get('pattern_analysis', {})
                    
                    volatility_params = ['volatility', 'stability_factor', 'recent_std']
                    volatility_param_count = sum(1 for param in volatility_params if param in pattern_analysis)
                    
                    volatility_aware = volatility_param_count >= 2
                    print(f"   ✅ Volatility-aware parameters: {volatility_aware} ({volatility_param_count}/3)")
                    volatility_tests.append(volatility_aware)
                
                self.test_results['volatility_adjustments'] = sum(volatility_tests) >= 3
                
            else:
                print("❌ No predictions generated for volatility analysis")
                self.test_results['volatility_adjustments'] = False
                
        except Exception as e:
            print(f"❌ Volatility-aware adjustments test error: {str(e)}")
            self.test_results['volatility_adjustments'] = False
    
    def test_enhanced_bounds_checking(self):
        """Test 6: Enhanced bounds checking keeps predictions within reasonable ranges"""
        print("\n=== Testing Enhanced Bounds Checking ===")
        
        try:
            # Create data with known bounds
            df = self.create_ph_pattern_data("u_shaped")
            data_min = df['pH'].min()
            data_max = df['pH'].max()
            data_mean = df['pH'].mean()
            data_std = df['pH'].std()
            
            # Define reasonable bounds (should be wider than historical but not unlimited)
            reasonable_lower = data_min - 2 * data_std
            reasonable_upper = data_max + 2 * data_std
            
            print(f"   Historical range: {data_min:.3f} - {data_max:.3f}")
            print(f"   Reasonable bounds: {reasonable_lower:.3f} - {reasonable_upper:.3f}")
            
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('bounds_test.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
            )
            model_id = response.json().get('model_id')
            
            # Test bounds checking with various prediction horizons
            bounds_tests = []
            all_predictions = []
            
            for steps in [10, 20, 30, 50]:  # Test different horizons
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": steps, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    all_predictions.extend(predictions)
                    
                    if predictions:
                        pred_min = min(predictions)
                        pred_max = max(predictions)
                        
                        # Test bounds for this horizon
                        within_reasonable_bounds = (pred_min >= reasonable_lower and pred_max <= reasonable_upper)
                        
                        print(f"   Steps {steps}: range {pred_min:.3f} - {pred_max:.3f}, within bounds: {within_reasonable_bounds}")
                        bounds_tests.append(within_reasonable_bounds)
                    else:
                        bounds_tests.append(False)
                else:
                    bounds_tests.append(False)
            
            # Test extreme case: very long prediction horizon
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 100, "time_window": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if predictions:
                    pred_min = min(predictions)
                    pred_max = max(predictions)
                    
                    # Even for very long horizons, should stay within expanded reasonable bounds
                    expanded_lower = data_min - 4 * data_std
                    expanded_upper = data_max + 4 * data_std
                    
                    extreme_bounds_test = (pred_min >= expanded_lower and pred_max <= expanded_upper)
                    print(f"   Extreme horizon (100 steps): range {pred_min:.3f} - {pred_max:.3f}, within expanded bounds: {extreme_bounds_test}")
                    bounds_tests.append(extreme_bounds_test)
                else:
                    bounds_tests.append(False)
            else:
                bounds_tests.append(False)
            
            # Test for bounds checking parameters in analysis
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 20, "time_window": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                pattern_analysis = data.get('pattern_analysis', {})
                
                # Check for bounds-related parameters
                bounds_params = ['mean', 'std', 'recent_mean', 'recent_std']
                bounds_param_count = sum(1 for param in bounds_params if param in pattern_analysis)
                
                bounds_aware = bounds_param_count >= 3
                print(f"   ✅ Bounds-aware parameters: {bounds_aware} ({bounds_param_count}/4)")
                bounds_tests.append(bounds_aware)
            
            # Overall bounds checking test
            self.test_results['bounds_checking'] = sum(bounds_tests) >= len(bounds_tests) * 0.8
            
        except Exception as e:
            print(f"❌ Enhanced bounds checking test error: {str(e)}")
            self.test_results['bounds_checking'] = False
    
    def test_pattern_preservation_score(self):
        """Test 7: Pattern preservation score improvements"""
        print("\n=== Testing Pattern Preservation Score ===")
        
        try:
            # Create data with clear patterns
            df = self.create_ph_pattern_data("cyclical")
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('pattern_preservation.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
            )
            model_id = response.json().get('model_id')
            
            # Generate predictions and analyze pattern preservation
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 30, "time_window": 50}
            )
            
            preservation_tests = []
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                pattern_analysis = data.get('pattern_analysis', {})
                
                # Test 1: Pattern quality score
                if 'pattern_quality_score' in pattern_analysis:
                    quality_score = pattern_analysis['pattern_quality_score']
                    high_quality = quality_score >= 0.7  # Expect good quality
                    print(f"   ✅ Pattern quality score: {quality_score:.3f} (high quality: {high_quality})")
                    preservation_tests.append(high_quality)
                else:
                    print("   ❌ Pattern quality score missing")
                    preservation_tests.append(False)
                
                # Test 2: Pattern preservation metrics
                preservation_metrics = ['trend_consistency', 'pattern_continuation', 'stability_factor']
                metrics_present = sum(1 for metric in preservation_metrics if metric in pattern_analysis)
                
                metrics_adequate = metrics_present >= 2
                print(f"   ✅ Preservation metrics: {metrics_adequate} ({metrics_present}/3 present)")
                preservation_tests.append(metrics_adequate)
                
                # Test 3: Predictions maintain pattern characteristics
                if predictions and len(predictions) >= 20:
                    # Check if predictions show similar characteristics to historical data
                    historical_mean = df['pH'].mean()
                    historical_std = df['pH'].std()
                    
                    pred_mean = np.mean(predictions)
                    pred_std = np.std(predictions)
                    
                    mean_similarity = abs(pred_mean - historical_mean) <= historical_std
                    std_similarity = 0.5 <= (pred_std / historical_std) <= 2.0
                    
                    characteristics_preserved = mean_similarity and std_similarity
                    print(f"   ✅ Pattern characteristics preserved: {characteristics_preserved}")
                    print(f"      Mean similarity: {mean_similarity} ({pred_mean:.3f} vs {historical_mean:.3f})")
                    print(f"      Std similarity: {std_similarity} (ratio: {pred_std/historical_std:.3f})")
                    preservation_tests.append(characteristics_preserved)
                else:
                    preservation_tests.append(False)
                
                # Test 4: Pattern continuation quality
                if 'pattern_continuation' in pattern_analysis:
                    continuation = pattern_analysis['pattern_continuation']
                    if isinstance(continuation, dict):
                        continuation_quality = len(continuation) >= 2  # Has meaningful continuation data
                        print(f"   ✅ Pattern continuation quality: {continuation_quality}")
                        preservation_tests.append(continuation_quality)
                    else:
                        preservation_tests.append(False)
                else:
                    preservation_tests.append(False)
                
                self.test_results['pattern_preservation'] = sum(preservation_tests) >= 3
                
            else:
                print(f"❌ Pattern preservation test failed: {response.status_code}")
                self.test_results['pattern_preservation'] = False
                
        except Exception as e:
            print(f"❌ Pattern preservation score test error: {str(e)}")
            self.test_results['pattern_preservation'] = False
    
    def test_comprehensive_pattern_following(self):
        """Test 8: Comprehensive pattern following with complex scenarios"""
        print("\n=== Testing Comprehensive Pattern Following ===")
        
        try:
            comprehensive_tests = []
            
            # Test different pattern types
            pattern_types = ["stable_with_trend", "cyclical", "u_shaped", "step_change", "complex_pattern"]
            
            for pattern_type in pattern_types:
                print(f"\n   Testing pattern type: {pattern_type}")
                
                # Create and upload data
                df = self.create_ph_pattern_data(pattern_type)
                csv_content = df.to_csv(index=False)
                
                files = {'file': (f'{pattern_type}.csv', csv_content, 'text/csv')}
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code != 200:
                    comprehensive_tests.append(False)
                    continue
                    
                data_id = response.json().get('data_id')
                
                # Train model
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": "arima"},
                    json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
                )
                
                if response.status_code != 200:
                    comprehensive_tests.append(False)
                    continue
                    
                model_id = response.json().get('model_id')
                
                # Test pattern following
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 25, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    pattern_analysis = data.get('pattern_analysis', {})
                    
                    # Evaluate pattern following quality
                    pattern_detected = 'pattern_type' in pattern_analysis
                    reasonable_predictions = (
                        predictions and 
                        len(predictions) == 25 and 
                        all(6.0 <= p <= 8.0 for p in predictions)  # pH range check
                    )
                    
                    pattern_following_quality = pattern_detected and reasonable_predictions
                    print(f"      Pattern following quality: {pattern_following_quality}")
                    comprehensive_tests.append(pattern_following_quality)
                else:
                    comprehensive_tests.append(False)
            
            # Overall comprehensive test result
            passed_patterns = sum(comprehensive_tests)
            total_patterns = len(pattern_types)
            
            print(f"\n   ✅ Comprehensive pattern following: {passed_patterns}/{total_patterns} patterns handled correctly")
            self.test_results['comprehensive_pattern_following'] = passed_patterns >= total_patterns * 0.8
            
        except Exception as e:
            print(f"❌ Comprehensive pattern following test error: {str(e)}")
            self.test_results['comprehensive_pattern_following'] = False
    
    def run_all_tests(self):
        """Run all enhanced pattern-following algorithm tests"""
        print("🎯 ENHANCED PATTERN-FOLLOWING ALGORITHM TESTING")
        print("=" * 60)
        
        # Run all tests
        self.test_multi_scale_pattern_analysis()
        self.test_bias_correction_historical_ranges()
        self.test_cyclical_pattern_detection()
        self.test_adaptive_trend_decay()
        self.test_volatility_aware_adjustments()
        self.test_enhanced_bounds_checking()
        self.test_pattern_preservation_score()
        self.test_comprehensive_pattern_following()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 ENHANCED PATTERN-FOLLOWING TEST RESULTS")
        print("=" * 60)
        
        test_names = [
            ("Multi-Scale Pattern Analysis", "multi_scale_analysis"),
            ("Bias Correction & Historical Ranges", "bias_correction"),
            ("Cyclical Pattern Detection", "cyclical_detection"),
            ("Adaptive Trend Decay", "adaptive_trend_decay"),
            ("Volatility-Aware Adjustments", "volatility_adjustments"),
            ("Enhanced Bounds Checking", "bounds_checking"),
            ("Pattern Preservation Score", "pattern_preservation"),
            ("Comprehensive Pattern Following", "comprehensive_pattern_following")
        ]
        
        passed_tests = 0
        total_tests = len(test_names)
        
        for test_name, test_key in test_names:
            result = self.test_results.get(test_key, False)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            if result:
                passed_tests += 1
        
        print("=" * 60)
        success_rate = (passed_tests / total_tests) * 100
        print(f"🎯 OVERALL SUCCESS RATE: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 EXCELLENT: Enhanced pattern-following algorithms are working well!")
        elif success_rate >= 60:
            print("✅ GOOD: Most enhanced pattern-following features are functional")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Several pattern-following features need attention")
        
        return self.test_results

if __name__ == "__main__":
    tester = EnhancedPatternTester()
    results = tester.run_all_tests()