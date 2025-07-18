#!/usr/bin/env python3
"""
Focused Advanced ML Models Testing
Tests specifically for SymPy/mpmath dependency resolution and advanced model functionality
"""

import requests
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://c865ad75-dc9b-46cb-8e0f-591f829ae762.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing backend at: {API_BASE_URL}")

class FocusedAdvancedTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_test_data(self, pattern_type="quadratic", size=50):
        """Create test data with specific patterns"""
        dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
        
        if pattern_type == "quadratic":
            x = np.linspace(-3, 3, size)
            values = x**2 + np.random.normal(0, 0.3, size) + 10
        elif pattern_type == "stable":
            values = 25 + np.random.normal(0, 1, size) + 2 * np.sin(np.linspace(0, 2*np.pi, size))
        else:
            values = np.linspace(10, 30, size) + np.random.normal(0, 1, size)
        
        return pd.DataFrame({
            'timestamp': dates,
            'value': values
        })
    
    def test_dependency_resolution(self):
        """Test that SymPy/mpmath dependency errors are resolved"""
        print("\n=== Testing SymPy/mpmath Dependency Resolution ===")
        
        # Create test data
        df = self.create_test_data("quadratic", 60)
        csv_content = df.to_csv(index=False)
        
        # Upload data
        files = {'file': ('dependency_test.csv', csv_content, 'text/csv')}
        response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
        
        if response.status_code != 200:
            print(f"❌ Data upload failed: {response.status_code}")
            self.test_results['dependency_resolution'] = False
            return
            
        data_id = response.json().get('data_id')
        print("✅ Test data uploaded successfully")
        
        # Test each advanced model for dependency errors
        models_to_test = ['lstm', 'dlinear', 'nbeats']
        dependency_results = {}
        
        for model_type in models_to_test:
            print(f"\n--- Testing {model_type.upper()} for dependency errors ---")
            
            try:
                # Train model
                training_params = {
                    "time_column": "timestamp",
                    "target_column": "value",
                    "seq_len": 15,
                    "pred_len": 8,
                    "epochs": 15,
                    "batch_size": 4
                }
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": model_type},
                    json=training_params
                )
                
                if response.status_code == 200:
                    print(f"✅ {model_type.upper()} training successful - no dependency errors")
                    dependency_results[model_type] = {
                        'training_success': True,
                        'dependency_error': False,
                        'error_message': None
                    }
                    
                    # Test prediction to ensure full functionality
                    model_id = response.json().get('model_id')
                    pred_response = self.session.get(
                        f"{API_BASE_URL}/generate-prediction",
                        params={"model_id": model_id, "steps": 10}
                    )
                    
                    if pred_response.status_code == 200:
                        print(f"✅ {model_type.upper()} prediction successful")
                        dependency_results[model_type]['prediction_success'] = True
                    else:
                        print(f"⚠️ {model_type.upper()} prediction failed: {pred_response.status_code}")
                        error_text = pred_response.text
                        if 'SymPy' in error_text or 'mpmath' in error_text:
                            print(f"❌ SymPy/mpmath dependency error in prediction!")
                            dependency_results[model_type]['dependency_error'] = True
                            dependency_results[model_type]['error_message'] = error_text
                        dependency_results[model_type]['prediction_success'] = False
                        
                else:
                    print(f"❌ {model_type.upper()} training failed: {response.status_code}")
                    error_text = response.text
                    
                    if 'SymPy' in error_text or 'mpmath' in error_text:
                        print(f"❌ SymPy/mpmath dependency error detected!")
                        dependency_results[model_type] = {
                            'training_success': False,
                            'dependency_error': True,
                            'error_message': error_text,
                            'prediction_success': False
                        }
                    else:
                        print(f"❌ Other training error: {error_text[:200]}...")
                        dependency_results[model_type] = {
                            'training_success': False,
                            'dependency_error': False,
                            'error_message': error_text,
                            'prediction_success': False
                        }
                        
            except Exception as e:
                print(f"❌ {model_type.upper()} test exception: {str(e)}")
                if 'SymPy' in str(e) or 'mpmath' in str(e):
                    print(f"❌ SymPy/mpmath dependency error in exception!")
                    dependency_results[model_type] = {
                        'training_success': False,
                        'dependency_error': True,
                        'error_message': str(e),
                        'prediction_success': False
                    }
                else:
                    dependency_results[model_type] = {
                        'training_success': False,
                        'dependency_error': False,
                        'error_message': str(e),
                        'prediction_success': False
                    }
        
        # Analyze dependency resolution results
        print(f"\n📊 Dependency Resolution Results:")
        dependency_resolved_count = 0
        working_models_count = 0
        
        for model_type, results in dependency_results.items():
            dependency_status = "❌" if results['dependency_error'] else "✅"
            training_status = "✅" if results['training_success'] else "❌"
            prediction_status = "✅" if results.get('prediction_success', False) else "❌"
            
            print(f"   {model_type.upper()}:")
            print(f"     No dependency errors: {dependency_status}")
            print(f"     Training success: {training_status}")
            print(f"     Prediction success: {prediction_status}")
            
            if not results['dependency_error']:
                dependency_resolved_count += 1
            if results['training_success'] and results.get('prediction_success', False):
                working_models_count += 1
                
            if results['dependency_error']:
                print(f"     Error: {results['error_message'][:100]}...")
        
        print(f"\n🎯 Overall Assessment:")
        print(f"   Models without dependency errors: {dependency_resolved_count}/{len(models_to_test)}")
        print(f"   Fully working models: {working_models_count}/{len(models_to_test)}")
        
        # Success if no dependency errors detected
        self.test_results['dependency_resolution'] = dependency_resolved_count == len(models_to_test)
        
        return dependency_results
    
    def test_downward_bias_fix(self):
        """Test that downward bias in predictions has been fixed"""
        print("\n=== Testing Downward Bias Fix ===")
        
        # Create stable data that should not trend downward
        df = self.create_test_data("stable", 70)
        csv_content = df.to_csv(index=False)
        
        # Upload data
        files = {'file': ('bias_test.csv', csv_content, 'text/csv')}
        response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
        
        if response.status_code != 200:
            print(f"❌ Bias test data upload failed: {response.status_code}")
            self.test_results['downward_bias_fix'] = False
            return
            
        data_id = response.json().get('data_id')
        print("✅ Stable test data uploaded")
        
        # Train LSTM model (most reliable)
        training_params = {
            "time_column": "timestamp",
            "target_column": "value",
            "seq_len": 20,
            "pred_len": 10,
            "epochs": 20
        }
        
        response = self.session.post(
            f"{API_BASE_URL}/train-model",
            params={"data_id": data_id, "model_type": "lstm"},
            json=training_params
        )
        
        if response.status_code != 200:
            print(f"❌ LSTM training failed: {response.status_code}")
            self.test_results['downward_bias_fix'] = False
            return
            
        model_id = response.json().get('model_id')
        print("✅ LSTM model trained on stable data")
        
        # Test multiple prediction calls for bias
        bias_results = []
        historical_mean = df['value'].mean()
        
        for i in range(3):
            pred_response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 25}
            )
            
            if pred_response.status_code == 200:
                pred_data = pred_response.json()
                predictions = pred_data.get('predictions', [])
                
                if len(predictions) >= 20:
                    # Calculate trend slope
                    trend_slope = np.polyfit(range(len(predictions)), predictions, 1)[0]
                    prediction_mean = np.mean(predictions)
                    prediction_std = np.std(predictions)
                    
                    # Check for downward bias
                    has_downward_bias = trend_slope < -0.05  # Significant downward trend
                    mean_deviation = abs(prediction_mean - historical_mean)
                    
                    bias_results.append({
                        'call': i+1,
                        'trend_slope': trend_slope,
                        'has_downward_bias': has_downward_bias,
                        'prediction_mean': prediction_mean,
                        'mean_deviation': mean_deviation,
                        'prediction_std': prediction_std,
                        'predictions_count': len(predictions)
                    })
                    
                    print(f"   Call {i+1}: Slope={trend_slope:.6f}, Mean={prediction_mean:.2f}, Bias={'YES' if has_downward_bias else 'NO'}")
                else:
                    print(f"❌ Call {i+1}: Insufficient predictions: {len(predictions)}")
                    bias_results.append({
                        'call': i+1,
                        'has_downward_bias': True,
                        'predictions_count': len(predictions)
                    })
            else:
                print(f"❌ Call {i+1}: Prediction failed: {pred_response.status_code}")
                bias_results.append({
                    'call': i+1,
                    'has_downward_bias': True,
                    'predictions_count': 0
                })
        
        # Analyze bias results
        successful_calls = [r for r in bias_results if not r.get('has_downward_bias', True)]
        bias_free_calls = len(successful_calls)
        
        print(f"\n📊 Downward Bias Test Results:")
        print(f"   Total prediction calls: {len(bias_results)}")
        print(f"   Calls without downward bias: {bias_free_calls}")
        print(f"   Historical data mean: {historical_mean:.2f}")
        
        if successful_calls:
            avg_slope = np.mean([r['trend_slope'] for r in successful_calls])
            avg_pred_mean = np.mean([r['prediction_mean'] for r in successful_calls])
            print(f"   Average trend slope (successful): {avg_slope:.6f}")
            print(f"   Average prediction mean (successful): {avg_pred_mean:.2f}")
        
        # Success if majority of calls don't have downward bias
        bias_fix_success = bias_free_calls >= len(bias_results) * 0.67  # 67% success rate
        
        print(f"\n🎯 Downward Bias Fix Assessment:")
        print(f"   Success rate: {(bias_free_calls/len(bias_results))*100:.1f}%")
        print(f"   Downward bias fixed: {'✅ YES' if bias_fix_success else '❌ NO'}")
        
        self.test_results['downward_bias_fix'] = bias_fix_success
        
        return bias_results
    
    def test_pattern_following(self):
        """Test that predictions follow historical patterns"""
        print("\n=== Testing Pattern Following ===")
        
        # Create quadratic pattern data
        df = self.create_test_data("quadratic", 80)
        csv_content = df.to_csv(index=False)
        
        # Upload data
        files = {'file': ('pattern_test.csv', csv_content, 'text/csv')}
        response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
        
        if response.status_code != 200:
            print(f"❌ Pattern test data upload failed: {response.status_code}")
            self.test_results['pattern_following'] = False
            return
            
        data_id = response.json().get('data_id')
        print("✅ Quadratic pattern data uploaded")
        
        # Train LSTM model
        training_params = {
            "time_column": "timestamp",
            "target_column": "value",
            "seq_len": 25,
            "pred_len": 12,
            "epochs": 25
        }
        
        response = self.session.post(
            f"{API_BASE_URL}/train-model",
            params={"data_id": data_id, "model_type": "lstm"},
            json=training_params
        )
        
        if response.status_code != 200:
            print(f"❌ LSTM training failed: {response.status_code}")
            self.test_results['pattern_following'] = False
            return
            
        model_id = response.json().get('model_id')
        print("✅ LSTM model trained on quadratic pattern")
        
        # Test pattern following
        pred_response = self.session.get(
            f"{API_BASE_URL}/generate-prediction",
            params={"model_id": model_id, "steps": 20}
        )
        
        if pred_response.status_code != 200:
            print(f"❌ Pattern prediction failed: {pred_response.status_code}")
            self.test_results['pattern_following'] = False
            return
            
        pred_data = pred_response.json()
        predictions = pred_data.get('predictions', [])
        
        if len(predictions) < 15:
            print(f"❌ Insufficient predictions for pattern analysis: {len(predictions)}")
            self.test_results['pattern_following'] = False
            return
        
        # Analyze pattern following
        historical_mean = df['value'].mean()
        historical_std = df['value'].std()
        prediction_mean = np.mean(predictions)
        prediction_std = np.std(predictions)
        
        # Check if predictions are within reasonable range
        within_range = abs(prediction_mean - historical_mean) < 2 * historical_std
        
        # Check for variability (not flat predictions)
        has_variability = prediction_std > 0.5
        
        # Check trend consistency with historical pattern
        historical_trend = np.polyfit(range(len(df)), df['value'], 1)[0]
        prediction_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
        
        # For quadratic data, we expect some curvature, not just linear trend
        prediction_curvature = abs(np.polyfit(range(len(predictions)), predictions, 2)[0])
        has_curvature = prediction_curvature > 0.001
        
        print(f"\n📊 Pattern Following Analysis:")
        print(f"   Historical mean: {historical_mean:.2f}, Prediction mean: {prediction_mean:.2f}")
        print(f"   Historical std: {historical_std:.2f}, Prediction std: {prediction_std:.2f}")
        print(f"   Within reasonable range: {'✅ YES' if within_range else '❌ NO'}")
        print(f"   Has variability: {'✅ YES' if has_variability else '❌ NO'}")
        print(f"   Historical trend: {historical_trend:.4f}, Prediction trend: {prediction_trend:.4f}")
        print(f"   Has curvature (quadratic): {'✅ YES' if has_curvature else '❌ NO'}")
        print(f"   Prediction curvature: {prediction_curvature:.6f}")
        
        # Success criteria
        pattern_criteria = [
            within_range,
            has_variability,
            has_curvature
        ]
        
        pattern_success = sum(pattern_criteria) >= 2  # At least 2 out of 3 criteria
        
        print(f"\n🎯 Pattern Following Assessment:")
        print(f"   Criteria met: {sum(pattern_criteria)}/3")
        print(f"   Pattern following: {'✅ YES' if pattern_success else '❌ NO'}")
        
        self.test_results['pattern_following'] = pattern_success
        
        return {
            'historical_mean': historical_mean,
            'prediction_mean': prediction_mean,
            'within_range': within_range,
            'has_variability': has_variability,
            'has_curvature': has_curvature,
            'predictions': predictions[:10]  # First 10 predictions
        }
    
    def run_focused_tests(self):
        """Run focused advanced ML tests"""
        print("🎯 FOCUSED ADVANCED ML MODELS TESTING")
        print("=" * 60)
        print("Focus: SymPy/mpmath dependency fix, downward bias resolution, pattern following")
        print("=" * 60)
        
        # Run focused tests
        dependency_results = self.test_dependency_resolution()
        bias_results = self.test_downward_bias_fix()
        pattern_results = self.test_pattern_following()
        
        # Overall assessment
        print(f"\n" + "="*60)
        print("🎯 FOCUSED TESTING SUMMARY")
        print("="*60)
        
        tests_passed = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        
        print(f"Tests passed: {tests_passed}/{total_tests}")
        
        for test_name, result in self.test_results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {test_name.replace('_', ' ').title()}")
        
        success_rate = (tests_passed / total_tests) * 100 if total_tests > 0 else 0
        print(f"\nOverall success rate: {success_rate:.1f}%")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS:")
        if self.test_results.get('dependency_resolution', False):
            print("   ✅ SymPy/mpmath dependency errors have been RESOLVED")
        else:
            print("   ❌ SymPy/mpmath dependency errors still present")
            
        if self.test_results.get('downward_bias_fix', False):
            print("   ✅ Downward bias in predictions has been FIXED")
        else:
            print("   ❌ Downward bias in predictions still present")
            
        if self.test_results.get('pattern_following', False):
            print("   ✅ Predictions follow historical patterns correctly")
        else:
            print("   ❌ Predictions do not follow historical patterns well")
        
        return self.test_results

if __name__ == "__main__":
    tester = FocusedAdvancedTester()
    results = tester.run_focused_tests()