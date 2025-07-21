#!/usr/bin/env python3
"""
Simplified Enhanced Pattern-Learning Test - Bypassing Problematic Continuous System
Focus on testing basic prediction improvements and bias analysis
"""

import requests
import json
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from scipy import stats

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://064f3bb3-c010-4892-8a8e-8e29d9900fe8.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Enhanced Pattern-Learning System at: {API_BASE_URL}")

class SimplifiedPatternTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_pattern_data(self, pattern_type="linear", size=50):
        """Create different sensor data patterns"""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='h')
        
        if pattern_type == "linear":
            # Linear upward trend
            values = np.linspace(6.8, 7.4, size) + np.random.normal(0, 0.08, size)
        elif pattern_type == "sinusoidal":
            # Sinusoidal pattern
            t = np.linspace(0, 4*np.pi, size)
            values = 7.0 + 0.3 * np.sin(t) + np.random.normal(0, 0.05, size)
        elif pattern_type == "trending":
            # Quadratic trend
            t = np.linspace(0, 1, size)
            values = 6.9 + 0.5 * t + 0.2 * t**2 + np.random.normal(0, 0.06, size)
        else:
            # Stable pattern
            values = np.full(size, 7.2) + np.random.normal(0, 0.04, size)
        
        # Ensure realistic pH range
        values = np.clip(values, 6.0, 8.0)
        
        return pd.DataFrame({
            'timestamp': dates,
            'ph_value': values
        })
    
    def test_basic_prediction_bias(self):
        """Test basic prediction for bias using multiple calls"""
        print("\n🎯 TESTING BASIC PREDICTION BIAS (Multiple Calls)")
        print("=" * 60)
        
        pattern_types = ["linear", "sinusoidal", "trending", "stable"]
        bias_results = {}
        
        for pattern_type in pattern_types:
            print(f"\n--- Testing {pattern_type} pattern ---")
            
            # Upload data
            df = self.create_pattern_data(pattern_type)
            csv_content = df.to_csv(index=False)
            files = {'file': (f'{pattern_type}_test.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            if response.status_code != 200:
                print(f"❌ Upload failed for {pattern_type}")
                bias_results[pattern_type] = {'success': False, 'error': 'Upload failed'}
                continue
            
            data_id = response.json().get('data_id')
            
            # Train model
            training_params = {
                "time_column": "timestamp",
                "target_column": "ph_value",
                "order": [1, 1, 1]
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json=training_params
            )
            
            if response.status_code != 200:
                print(f"❌ Training failed for {pattern_type}")
                bias_results[pattern_type] = {'success': False, 'error': 'Training failed'}
                continue
            
            model_id = response.json().get('model_id')
            print(f"✅ Model trained: {model_id}")
            
            # Generate multiple prediction sets to test for bias
            all_predictions = []
            successful_calls = 0
            
            for i in range(10):  # Multiple prediction calls
                try:
                    print(f"   Call {i+1}/10...")
                    
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-prediction",
                        params={"model_id": model_id, "steps": 10},
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        predictions = data.get('predictions', [])
                        
                        if predictions and len(predictions) > 0:
                            # Handle different response formats
                            if isinstance(predictions[0], dict):
                                pred_values = [p['value'] for p in predictions]
                            else:
                                pred_values = predictions
                            
                            all_predictions.extend(pred_values)
                            successful_calls += 1
                            
                            print(f"      ✅ Got {len(pred_values)} predictions")
                            print(f"      Range: {min(pred_values):.3f} - {max(pred_values):.3f}")
                        else:
                            print(f"      ⚠️ No predictions in response")
                    else:
                        print(f"      ❌ Call failed: {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    print(f"      ⚠️ Call timed out")
                except Exception as e:
                    print(f"      ❌ Call error: {str(e)}")
                
                time.sleep(0.5)  # Small delay between calls
            
            # Analyze bias
            if len(all_predictions) >= 20:
                bias_analysis = self.analyze_bias(all_predictions, pattern_type)
                bias_results[pattern_type] = {
                    'success': True,
                    'successful_calls': successful_calls,
                    'total_predictions': len(all_predictions),
                    'bias_analysis': bias_analysis,
                    'historical_stats': {
                        'mean': np.mean(df['ph_value']),
                        'std': np.std(df['ph_value'])
                    }
                }
            else:
                bias_results[pattern_type] = {
                    'success': False,
                    'error': f'Insufficient predictions ({len(all_predictions)})',
                    'successful_calls': successful_calls
                }
        
        return bias_results
    
    def analyze_bias(self, predictions, pattern_type):
        """Analyze predictions for bias"""
        predictions = np.array(predictions)
        
        # Calculate trend slope
        x = np.arange(len(predictions))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        
        # Bias assessment
        bias_threshold = 0.02  # Acceptable slope threshold
        
        has_downward_bias = slope < -bias_threshold
        has_upward_bias = slope > bias_threshold
        no_bias = abs(slope) <= bias_threshold
        
        # Variability check
        good_variability = std_pred > 0.001
        
        # pH range check
        realistic_range = 6.0 <= min_pred <= max_pred <= 8.0
        
        return {
            'slope': slope,
            'mean': mean_pred,
            'std': std_pred,
            'min': min_pred,
            'max': max_pred,
            'r_squared': r_value**2,
            'p_value': p_value,
            'has_downward_bias': has_downward_bias,
            'has_upward_bias': has_upward_bias,
            'no_bias': no_bias,
            'good_variability': good_variability,
            'realistic_range': realistic_range,
            'overall_good': no_bias and good_variability and realistic_range
        }
    
    def test_pattern_following(self):
        """Test pattern following with different data types"""
        print("\n🔬 TESTING PATTERN FOLLOWING")
        print("=" * 60)
        
        pattern_results = {}
        pattern_types = ["linear", "sinusoidal", "trending"]
        
        for pattern_type in pattern_types:
            print(f"\n--- Testing {pattern_type} pattern following ---")
            
            # Create pattern data
            df = self.create_pattern_data(pattern_type, size=60)
            csv_content = df.to_csv(index=False)
            files = {'file': (f'{pattern_type}_pattern.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            if response.status_code != 200:
                pattern_results[pattern_type] = {'success': False, 'error': 'Upload failed'}
                continue
            
            data_id = response.json().get('data_id')
            
            # Train model with appropriate order for pattern type
            if pattern_type == "sinusoidal":
                order = [2, 1, 2]  # Higher order for cyclical patterns
            elif pattern_type == "trending":
                order = [2, 1, 1]  # Higher AR order for trending
            else:
                order = [1, 1, 1]  # Standard order for linear
            
            training_params = {
                "time_column": "timestamp",
                "target_column": "ph_value",
                "order": order
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json=training_params
            )
            
            if response.status_code != 200:
                pattern_results[pattern_type] = {'success': False, 'error': 'Training failed'}
                continue
            
            model_id = response.json().get('model_id')
            
            # Generate predictions
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 20}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if predictions:
                    if isinstance(predictions[0], dict):
                        pred_values = [p['value'] for p in predictions]
                    else:
                        pred_values = predictions
                    
                    # Analyze pattern following
                    historical_values = df['ph_value'].values
                    
                    hist_mean = np.mean(historical_values)
                    hist_std = np.std(historical_values)
                    pred_mean = np.mean(pred_values)
                    pred_std = np.std(pred_values)
                    
                    mean_deviation = abs(pred_mean - hist_mean) / hist_std if hist_std > 0 else 0
                    std_ratio = pred_std / hist_std if hist_std > 0 else 0
                    
                    # Pattern following assessment
                    good_mean_following = mean_deviation <= 0.5
                    good_variability_following = 0.3 <= std_ratio <= 3.0
                    
                    pattern_results[pattern_type] = {
                        'success': True,
                        'historical_stats': {'mean': hist_mean, 'std': hist_std},
                        'prediction_stats': {'mean': pred_mean, 'std': pred_std},
                        'mean_deviation': mean_deviation,
                        'std_ratio': std_ratio,
                        'good_mean_following': good_mean_following,
                        'good_variability_following': good_variability_following,
                        'overall_good': good_mean_following and good_variability_following
                    }
                    
                    print(f"   Historical: mean={hist_mean:.3f}, std={hist_std:.3f}")
                    print(f"   Prediction: mean={pred_mean:.3f}, std={pred_std:.3f}")
                    print(f"   Mean deviation: {mean_deviation:.3f}")
                    print(f"   Std ratio: {std_ratio:.3f}")
                    print(f"   Pattern following: {'✅' if good_mean_following and good_variability_following else '❌'}")
                else:
                    pattern_results[pattern_type] = {'success': False, 'error': 'No predictions returned'}
            else:
                pattern_results[pattern_type] = {'success': False, 'error': f'Prediction failed: {response.status_code}'}
        
        return pattern_results
    
    def test_variability_preservation(self):
        """Test variability preservation"""
        print("\n🔬 TESTING VARIABILITY PRESERVATION")
        print("=" * 60)
        
        # Test with trending pattern (has natural variability)
        df = self.create_pattern_data("trending", size=50)
        csv_content = df.to_csv(index=False)
        files = {'file': ('variability_test.csv', csv_content, 'text/csv')}
        
        response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
        if response.status_code != 200:
            return {'success': False, 'error': 'Upload failed'}
        
        data_id = response.json().get('data_id')
        
        # Train model
        training_params = {
            "time_column": "timestamp",
            "target_column": "ph_value",
            "order": [1, 1, 1]
        }
        
        response = self.session.post(
            f"{API_BASE_URL}/train-model",
            params={"data_id": data_id, "model_type": "arima"},
            json=training_params
        )
        
        if response.status_code != 200:
            return {'success': False, 'error': 'Training failed'}
        
        model_id = response.json().get('model_id')
        
        # Generate multiple prediction sets to test variability
        all_predictions = []
        
        for i in range(5):
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 15}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if predictions:
                    if isinstance(predictions[0], dict):
                        pred_values = [p['value'] for p in predictions]
                    else:
                        pred_values = predictions
                    
                    all_predictions.extend(pred_values)
        
        if len(all_predictions) >= 30:
            historical_values = df['ph_value'].values
            
            hist_std = np.std(historical_values)
            pred_std = np.std(all_predictions)
            
            # Calculate change variability
            hist_changes = np.diff(historical_values)
            pred_changes = np.diff(all_predictions)
            
            hist_change_std = np.std(hist_changes) if len(hist_changes) > 0 else 0
            pred_change_std = np.std(pred_changes) if len(pred_changes) > 0 else 0
            
            value_variability_ratio = pred_std / hist_std if hist_std > 0 else 0
            change_variability_ratio = pred_change_std / hist_change_std if hist_change_std > 0 else 0
            
            # Variability assessment
            good_value_variability = 0.2 <= value_variability_ratio <= 5.0
            good_change_variability = 0.1 <= change_variability_ratio <= 10.0
            
            print(f"   Historical std: {hist_std:.6f}")
            print(f"   Prediction std: {pred_std:.6f}")
            print(f"   Value variability ratio: {value_variability_ratio:.3f}")
            print(f"   Change variability ratio: {change_variability_ratio:.3f}")
            print(f"   Variability preservation: {'✅' if good_value_variability and good_change_variability else '❌'}")
            
            return {
                'success': True,
                'historical_std': hist_std,
                'prediction_std': pred_std,
                'value_variability_ratio': value_variability_ratio,
                'change_variability_ratio': change_variability_ratio,
                'good_value_variability': good_value_variability,
                'good_change_variability': good_change_variability,
                'overall_good': good_value_variability and good_change_variability
            }
        else:
            return {'success': False, 'error': 'Insufficient predictions for variability analysis'}
    
    def run_comprehensive_test(self):
        """Run comprehensive simplified pattern-learning test"""
        print("🎯 ENHANCED PATTERN-LEARNING PREDICTION SYSTEM TESTING")
        print("Focus: Basic Prediction Bias Resolution & Pattern Following")
        print("=" * 70)
        
        # Test 1: Basic Prediction Bias Resolution
        bias_results = self.test_basic_prediction_bias()
        self.test_results['bias_resolution'] = bias_results
        
        # Test 2: Pattern Following
        pattern_results = self.test_pattern_following()
        self.test_results['pattern_following'] = pattern_results
        
        # Test 3: Variability Preservation
        variability_results = self.test_variability_preservation()
        self.test_results['variability_preservation'] = variability_results
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("🎯 ENHANCED PATTERN-LEARNING SYSTEM TEST REPORT")
        print("=" * 70)
        
        # Test 1: Bias Resolution Results
        bias_results = self.test_results.get('bias_resolution', {})
        successful_patterns = 0
        total_patterns = len(bias_results)
        no_bias_patterns = 0
        
        print("\n📊 BASIC PREDICTION BIAS RESOLUTION:")
        for pattern_type, result in bias_results.items():
            if result.get('success'):
                successful_patterns += 1
                bias_analysis = result.get('bias_analysis', {})
                
                if bias_analysis.get('no_bias'):
                    no_bias_patterns += 1
                    print(f"   ✅ {pattern_type}: NO BIAS (slope: {bias_analysis.get('slope', 0):.6f})")
                elif bias_analysis.get('has_downward_bias'):
                    print(f"   ❌ {pattern_type}: DOWNWARD BIAS (slope: {bias_analysis.get('slope', 0):.6f})")
                else:
                    print(f"   ⚠️ {pattern_type}: UPWARD BIAS (slope: {bias_analysis.get('slope', 0):.6f})")
                
                print(f"      Predictions: {result.get('total_predictions', 0)}, Calls: {result.get('successful_calls', 0)}")
            else:
                print(f"   ❌ {pattern_type}: FAILED - {result.get('error', 'Unknown error')}")
        
        # Test 2: Pattern Following Results
        pattern_results = self.test_results.get('pattern_following', {})
        print(f"\n📊 PATTERN FOLLOWING:")
        successful_pattern_following = 0
        total_pattern_tests = len(pattern_results)
        
        for pattern_type, result in pattern_results.items():
            if result.get('success') and result.get('overall_good'):
                successful_pattern_following += 1
                print(f"   ✅ {pattern_type}: GOOD pattern following")
            elif result.get('success'):
                print(f"   ⚠️ {pattern_type}: PARTIAL pattern following")
            else:
                print(f"   ❌ {pattern_type}: FAILED pattern following")
        
        # Test 3: Variability Preservation Results
        variability_results = self.test_results.get('variability_preservation', {})
        print(f"\n📊 VARIABILITY PRESERVATION:")
        if variability_results.get('success') and variability_results.get('overall_good'):
            print("   ✅ WORKING - Realistic variability maintained")
            variability_success = True
        else:
            print("   ❌ NEEDS IMPROVEMENT - Poor variability preservation")
            variability_success = False
        
        # Overall Assessment
        print("\n" + "-" * 50)
        print("OVERALL ASSESSMENT:")
        
        # Calculate success metrics
        bias_resolution_success = no_bias_patterns >= total_patterns * 0.75  # 75% of patterns should have no bias
        pattern_following_success = successful_pattern_following >= total_pattern_tests * 0.67  # 67% success rate
        
        success_count = sum([bias_resolution_success, pattern_following_success, variability_success])
        total_tests = 3
        
        print(f"Bias Resolution: {'✅' if bias_resolution_success else '❌'} ({no_bias_patterns}/{total_patterns} patterns)")
        print(f"Pattern Following: {'✅' if pattern_following_success else '❌'} ({successful_pattern_following}/{total_pattern_tests} patterns)")
        print(f"Variability Preservation: {'✅' if variability_success else '❌'}")
        print(f"Overall Success Rate: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
        
        if success_count == total_tests:
            print("\n🎉 ENHANCED PATTERN-LEARNING SYSTEM: WORKING EXCELLENTLY!")
            print("   ✅ Downward bias issue has been RESOLVED")
            print("   ✅ System maintains historical patterns and variability")
            print("   ✅ All key improvements are functioning correctly")
        elif success_count >= 2:
            print("\n⚠️ ENHANCED PATTERN-LEARNING SYSTEM: MOSTLY WORKING")
            print("   ✅ Major improvements are functional")
            print("   ⚠️ Some minor issues need attention")
        else:
            print("\n❌ ENHANCED PATTERN-LEARNING SYSTEM: NEEDS SIGNIFICANT WORK")
            print("   ❌ Major issues detected that require attention")
        
        # Note about continuous prediction system
        print("\n📝 NOTE: Continuous prediction system has dependency issues")
        print("   (adaptive_continuous_learning module missing)")
        print("   Basic prediction system tested instead")
        
        return success_count / total_tests

if __name__ == "__main__":
    tester = SimplifiedPatternTester()
    tester.run_comprehensive_test()