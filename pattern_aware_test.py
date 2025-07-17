#!/usr/bin/env python3
"""
Enhanced Pattern-Aware Prediction System Testing
Tests the improved pattern detection and prediction algorithms
"""

import requests
import json
import pandas as pd
import numpy as np
import io
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://6bfa3f5d-c0d4-49d4-ad50-9f02160bc053.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing enhanced pattern-aware prediction system at: {API_BASE_URL}")

class PatternAwareTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        self.data_ids = {}
        self.model_ids = {}
        
    def create_u_shaped_data(self, points=50):
        """Create U-shaped pattern data for testing"""
        x = np.linspace(-2, 2, points)
        y = x**2 + 3 + np.random.normal(0, 0.1, points)  # U-shape with noise
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph': y
        })
        
        return df
    
    def create_linear_data(self, points=50):
        """Create linear pattern data for testing"""
        x = np.linspace(0, points-1, points)
        y = 2 * x + 10 + np.random.normal(0, 0.5, points)  # Linear trend with noise
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph': y
        })
        
        return df
    
    def create_cubic_data(self, points=50):
        """Create cubic pattern data for testing"""
        x = np.linspace(-1, 1, points)
        y = x**3 - 0.5*x + 5 + np.random.normal(0, 0.1, points)  # Cubic pattern with noise
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph': y
        })
        
        return df
    
    def create_inverted_u_data(self, points=50):
        """Create inverted U-shaped pattern data for testing"""
        x = np.linspace(-2, 2, points)
        y = -x**2 + 8 + np.random.normal(0, 0.1, points)  # Inverted U-shape with noise
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph': y
        })
        
        return df
    
    def upload_and_train_model(self, data, pattern_name, model_type='arima'):
        """Upload data and train model"""
        try:
            # Upload data
            csv_content = data.to_csv(index=False)
            files = {'file': (f'{pattern_name}_data.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Failed to upload {pattern_name} data: {response.status_code}")
                return None, None
            
            data_id = response.json().get('data_id')
            self.data_ids[pattern_name] = data_id
            
            # Train model
            training_params = {
                "time_column": "timestamp",
                "target_column": "ph",
                "order": [1, 1, 1] if model_type == 'arima' else {}
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": model_type},
                json=training_params
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to train {pattern_name} model: {response.status_code}")
                return data_id, None
            
            model_id = response.json().get('model_id')
            self.model_ids[pattern_name] = model_id
            
            print(f"✅ Successfully uploaded and trained {pattern_name} model")
            return data_id, model_id
            
        except Exception as e:
            print(f"❌ Error in upload_and_train_model for {pattern_name}: {str(e)}")
            return None, None
    
    def test_pattern_detection(self):
        """Test 1: Pattern Detection Capabilities"""
        print("\n=== Testing Pattern Detection Capabilities ===")
        
        pattern_tests = []
        
        # Test different pattern types
        patterns = {
            'u_shape': self.create_u_shaped_data(),
            'linear': self.create_linear_data(),
            'cubic': self.create_cubic_data(),
            'inverted_u': self.create_inverted_u_data()
        }
        
        for pattern_name, data in patterns.items():
            print(f"\nTesting {pattern_name} pattern detection...")
            
            data_id, model_id = self.upload_and_train_model(data, pattern_name)
            
            if data_id and model_id:
                # Generate predictions to test pattern analysis
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 30, "time_window": 100}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    pattern_analysis = result.get('pattern_analysis', {})
                    
                    if pattern_analysis:
                        detected_pattern = pattern_analysis.get('pattern_type', 'unknown')
                        print(f"   Detected pattern type: {detected_pattern}")
                        print(f"   Trend slope: {pattern_analysis.get('trend_slope', 'N/A')}")
                        print(f"   Velocity: {pattern_analysis.get('velocity', 'N/A')}")
                        print(f"   Recent mean: {pattern_analysis.get('recent_mean', 'N/A')}")
                        
                        # Check if pattern detection is reasonable
                        pattern_detected = detected_pattern != 'unknown'
                        pattern_tests.append((f"{pattern_name} pattern detection", pattern_detected))
                        
                        if pattern_detected:
                            print(f"   ✅ {pattern_name} pattern detected successfully")
                        else:
                            print(f"   ❌ {pattern_name} pattern not detected")
                    else:
                        print(f"   ❌ No pattern analysis data for {pattern_name}")
                        pattern_tests.append((f"{pattern_name} pattern detection", False))
                else:
                    print(f"   ❌ Failed to generate predictions for {pattern_name}: {response.status_code}")
                    pattern_tests.append((f"{pattern_name} pattern detection", False))
            else:
                pattern_tests.append((f"{pattern_name} pattern detection", False))
        
        # Evaluate pattern detection results
        passed_tests = sum(1 for _, passed in pattern_tests if passed)
        total_tests = len(pattern_tests)
        
        print(f"\n📊 Pattern detection test results: {passed_tests}/{total_tests}")
        for test_name, passed in pattern_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['pattern_detection'] = passed_tests >= total_tests * 0.75
    
    def test_pattern_aware_predictions(self):
        """Test 2: Pattern-Aware Prediction Quality"""
        print("\n=== Testing Pattern-Aware Prediction Quality ===")
        
        prediction_tests = []
        
        # Test U-shaped pattern predictions
        if 'u_shape' in self.model_ids:
            model_id = self.model_ids['u_shape']
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 20, "time_window": 100}
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                pattern_analysis = result.get('pattern_analysis', {})
                
                if len(predictions) >= 10:
                    # Check if predictions maintain realistic pH range
                    ph_range_valid = all(6.0 <= p <= 8.0 for p in predictions)
                    
                    # Check if predictions show variability (not monotonic)
                    prediction_variability = len(set([round(p, 1) for p in predictions])) > 3
                    
                    # Check if predictions follow pattern characteristics
                    pattern_type = pattern_analysis.get('pattern_type', 'unknown')
                    pattern_following = pattern_type in ['u_shape', 'quadratic', 'complex']
                    
                    print(f"   U-shape predictions: {len(predictions)} points")
                    print(f"   pH range valid (6.0-8.0): {ph_range_valid}")
                    print(f"   Prediction variability: {prediction_variability}")
                    print(f"   Pattern following: {pattern_following} (detected: {pattern_type})")
                    
                    prediction_tests.append(("U-shape pH range", ph_range_valid))
                    prediction_tests.append(("U-shape variability", prediction_variability))
                    prediction_tests.append(("U-shape pattern following", pattern_following))
                else:
                    print("   ❌ Insufficient U-shape predictions generated")
                    prediction_tests.extend([
                        ("U-shape pH range", False),
                        ("U-shape variability", False),
                        ("U-shape pattern following", False)
                    ])
            else:
                print(f"   ❌ Failed to generate U-shape predictions: {response.status_code}")
                prediction_tests.extend([
                    ("U-shape pH range", False),
                    ("U-shape variability", False),
                    ("U-shape pattern following", False)
                ])
        
        # Test linear pattern predictions
        if 'linear' in self.model_ids:
            model_id = self.model_ids['linear']
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 20, "time_window": 100}
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                pattern_analysis = result.get('pattern_analysis', {})
                
                if len(predictions) >= 10:
                    # Check trend consistency for linear pattern
                    trend_slope = pattern_analysis.get('trend_slope', 0)
                    
                    # Calculate actual trend in predictions
                    if len(predictions) >= 5:
                        actual_trend = (predictions[-1] - predictions[0]) / len(predictions)
                        trend_consistency = abs(actual_trend) > 0.01  # Some trend should be present
                    else:
                        trend_consistency = False
                    
                    # Check if predictions maintain reasonable bounds
                    reasonable_bounds = all(-10 <= p <= 50 for p in predictions)  # Reasonable for linear growth
                    
                    print(f"   Linear predictions: {len(predictions)} points")
                    print(f"   Detected trend slope: {trend_slope}")
                    print(f"   Trend consistency: {trend_consistency}")
                    print(f"   Reasonable bounds: {reasonable_bounds}")
                    
                    prediction_tests.append(("Linear trend consistency", trend_consistency))
                    prediction_tests.append(("Linear reasonable bounds", reasonable_bounds))
                else:
                    print("   ❌ Insufficient linear predictions generated")
                    prediction_tests.extend([
                        ("Linear trend consistency", False),
                        ("Linear reasonable bounds", False)
                    ])
            else:
                print(f"   ❌ Failed to generate linear predictions: {response.status_code}")
                prediction_tests.extend([
                    ("Linear trend consistency", False),
                    ("Linear reasonable bounds", False)
                ])
        
        # Evaluate prediction quality results
        passed_tests = sum(1 for _, passed in prediction_tests if passed)
        total_tests = len(prediction_tests)
        
        print(f"\n📊 Pattern-aware prediction test results: {passed_tests}/{total_tests}")
        for test_name, passed in prediction_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['pattern_aware_predictions'] = passed_tests >= total_tests * 0.7
    
    def test_continuous_prediction_continuity(self):
        """Test 3: Continuous Prediction Continuity"""
        print("\n=== Testing Continuous Prediction Continuity ===")
        
        continuity_tests = []
        
        # Test with U-shaped pattern for continuity
        if 'u_shape' in self.model_ids:
            model_id = self.model_ids['u_shape']
            
            # Make multiple continuous prediction calls
            prediction_sets = []
            timestamp_sets = []
            
            for i in range(3):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 10, "time_window": 50}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    predictions = result.get('predictions', [])
                    timestamps = result.get('timestamps', [])
                    
                    prediction_sets.append(predictions)
                    timestamp_sets.append(timestamps)
                    
                    time.sleep(1)  # Wait between calls
                else:
                    print(f"   ❌ Continuous prediction call {i+1} failed: {response.status_code}")
            
            if len(prediction_sets) >= 2:
                # Test timestamp progression
                timestamp_progression = True
                if len(timestamp_sets) >= 2:
                    # Check if timestamps are progressing forward
                    try:
                        first_timestamps = timestamp_sets[0]
                        second_timestamps = timestamp_sets[1]
                        
                        if first_timestamps and second_timestamps:
                            # Parse first timestamp from each set
                            first_time = datetime.fromisoformat(first_timestamps[0].replace('Z', '+00:00'))
                            second_time = datetime.fromisoformat(second_timestamps[0].replace('Z', '+00:00'))
                            
                            timestamp_progression = second_time > first_time
                    except Exception as e:
                        print(f"   ⚠️ Timestamp parsing error: {e}")
                        timestamp_progression = False
                
                # Test prediction continuity (no extreme jumps)
                prediction_continuity = True
                for i in range(1, len(prediction_sets)):
                    if prediction_sets[i] and prediction_sets[i-1]:
                        # Check for extreme jumps between prediction sets
                        last_prev = prediction_sets[i-1][-1] if prediction_sets[i-1] else 0
                        first_curr = prediction_sets[i][0] if prediction_sets[i] else 0
                        
                        jump = abs(first_curr - last_prev)
                        if jump > 5.0:  # Threshold for "extreme jump" in pH context
                            prediction_continuity = False
                            break
                
                # Test pattern maintenance across calls
                pattern_maintenance = True
                if len(prediction_sets) >= 2:
                    # Check if predictions maintain similar characteristics
                    for pred_set in prediction_sets:
                        if pred_set:
                            # Check if predictions stay within reasonable pH bounds
                            if not all(4.0 <= p <= 10.0 for p in pred_set):
                                pattern_maintenance = False
                                break
                
                print(f"   Continuous prediction calls made: {len(prediction_sets)}")
                print(f"   Timestamp progression: {timestamp_progression}")
                print(f"   Prediction continuity: {prediction_continuity}")
                print(f"   Pattern maintenance: {pattern_maintenance}")
                
                continuity_tests.append(("Timestamp progression", timestamp_progression))
                continuity_tests.append(("Prediction continuity", prediction_continuity))
                continuity_tests.append(("Pattern maintenance", pattern_maintenance))
            else:
                print("   ❌ Insufficient continuous prediction calls")
                continuity_tests.extend([
                    ("Timestamp progression", False),
                    ("Prediction continuity", False),
                    ("Pattern maintenance", False)
                ])
        else:
            print("   ❌ No U-shape model available for continuity testing")
            continuity_tests.extend([
                ("Timestamp progression", False),
                ("Prediction continuity", False),
                ("Pattern maintenance", False)
            ])
        
        # Evaluate continuity results
        passed_tests = sum(1 for _, passed in continuity_tests if passed)
        total_tests = len(continuity_tests)
        
        print(f"\n📊 Continuous prediction continuity test results: {passed_tests}/{total_tests}")
        for test_name, passed in continuity_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['continuous_prediction_continuity'] = passed_tests >= total_tests * 0.7
    
    def test_downward_bias_elimination(self):
        """Test 4: Downward Bias Elimination"""
        print("\n=== Testing Downward Bias Elimination ===")
        
        bias_tests = []
        
        # Test with different patterns to ensure no downward bias
        for pattern_name in ['u_shape', 'linear']:
            if pattern_name in self.model_ids:
                model_id = self.model_ids[pattern_name]
                
                print(f"\nTesting downward bias elimination for {pattern_name}...")
                
                # Generate multiple prediction sets
                all_predictions = []
                
                for i in range(5):  # Make 5 calls to test for accumulated bias
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": model_id, "steps": 10, "time_window": 50}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        predictions = result.get('predictions', [])
                        all_predictions.extend(predictions)
                        time.sleep(0.5)
                    else:
                        print(f"   ❌ Prediction call {i+1} failed for {pattern_name}")
                
                if len(all_predictions) >= 20:
                    # Calculate trend slope across all predictions
                    x = np.arange(len(all_predictions))
                    slope = np.polyfit(x, all_predictions, 1)[0]
                    
                    # Check for downward bias (significant negative slope)
                    no_downward_bias = slope > -0.1  # Allow slight negative slope but not strong downward bias
                    
                    # Check prediction range stability
                    pred_std = np.std(all_predictions)
                    pred_mean = np.mean(all_predictions)
                    range_stability = pred_std < abs(pred_mean) * 0.5  # Standard deviation should be reasonable
                    
                    # Check for realistic values
                    realistic_values = all(0 <= p <= 20 for p in all_predictions)  # Reasonable range for pH-like data
                    
                    print(f"   {pattern_name} predictions analyzed: {len(all_predictions)} points")
                    print(f"   Overall slope: {slope:.6f}")
                    print(f"   No downward bias: {no_downward_bias}")
                    print(f"   Range stability: {range_stability} (std: {pred_std:.3f}, mean: {pred_mean:.3f})")
                    print(f"   Realistic values: {realistic_values}")
                    
                    bias_tests.append((f"{pattern_name} no downward bias", no_downward_bias))
                    bias_tests.append((f"{pattern_name} range stability", range_stability))
                    bias_tests.append((f"{pattern_name} realistic values", realistic_values))
                else:
                    print(f"   ❌ Insufficient predictions for {pattern_name} bias testing")
                    bias_tests.extend([
                        (f"{pattern_name} no downward bias", False),
                        (f"{pattern_name} range stability", False),
                        (f"{pattern_name} realistic values", False)
                    ])
            else:
                print(f"   ❌ No {pattern_name} model available for bias testing")
                bias_tests.extend([
                    (f"{pattern_name} no downward bias", False),
                    (f"{pattern_name} range stability", False),
                    (f"{pattern_name} realistic values", False)
                ])
        
        # Evaluate bias elimination results
        passed_tests = sum(1 for _, passed in bias_tests if passed)
        total_tests = len(bias_tests)
        
        print(f"\n📊 Downward bias elimination test results: {passed_tests}/{total_tests}")
        for test_name, passed in bias_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['downward_bias_elimination'] = passed_tests >= total_tests * 0.8
    
    def test_pattern_specific_algorithms(self):
        """Test 5: Pattern-Specific Algorithm Performance"""
        print("\n=== Testing Pattern-Specific Algorithm Performance ===")
        
        algorithm_tests = []
        
        # Test U-shaped pattern algorithm
        if 'u_shape' in self.model_ids:
            model_id = self.model_ids['u_shape']
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 30, "time_window": 100}
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                pattern_analysis = result.get('pattern_analysis', {})
                
                if len(predictions) >= 20:
                    # For U-shaped data, check if predictions follow quadratic-like behavior
                    x = np.arange(len(predictions))
                    
                    # Fit quadratic to predictions
                    try:
                        quadratic_coeffs = np.polyfit(x, predictions, 2)
                        quadratic_fit = np.polyval(quadratic_coeffs, x)
                        
                        # Calculate R-squared for quadratic fit
                        ss_res = np.sum((predictions - quadratic_fit) ** 2)
                        ss_tot = np.sum((predictions - np.mean(predictions)) ** 2)
                        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                        
                        quadratic_fit_quality = r_squared > 0.3  # Reasonable quadratic fit
                        
                        print(f"   U-shape quadratic fit R²: {r_squared:.3f}")
                        print(f"   Quadratic fit quality: {quadratic_fit_quality}")
                        
                        algorithm_tests.append(("U-shape quadratic fit", quadratic_fit_quality))
                    except Exception as e:
                        print(f"   ❌ U-shape quadratic fit error: {e}")
                        algorithm_tests.append(("U-shape quadratic fit", False))
                else:
                    print("   ❌ Insufficient U-shape predictions for algorithm testing")
                    algorithm_tests.append(("U-shape quadratic fit", False))
            else:
                print(f"   ❌ Failed to generate U-shape predictions: {response.status_code}")
                algorithm_tests.append(("U-shape quadratic fit", False))
        
        # Test linear pattern algorithm
        if 'linear' in self.model_ids:
            model_id = self.model_ids['linear']
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 30, "time_window": 100}
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                pattern_analysis = result.get('pattern_analysis', {})
                
                if len(predictions) >= 20:
                    # For linear data, check if predictions maintain linear trend
                    x = np.arange(len(predictions))
                    
                    # Fit linear to predictions
                    try:
                        linear_coeffs = np.polyfit(x, predictions, 1)
                        linear_fit = np.polyval(linear_coeffs, x)
                        
                        # Calculate R-squared for linear fit
                        ss_res = np.sum((predictions - linear_fit) ** 2)
                        ss_tot = np.sum((predictions - np.mean(predictions)) ** 2)
                        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                        
                        linear_fit_quality = r_squared > 0.4  # Good linear fit
                        
                        # Check if trend is maintained (slope should be reasonable)
                        slope = linear_coeffs[0]
                        trend_maintained = abs(slope) > 0.01  # Some trend should be present
                        
                        print(f"   Linear fit R²: {r_squared:.3f}")
                        print(f"   Linear fit quality: {linear_fit_quality}")
                        print(f"   Slope: {slope:.3f}")
                        print(f"   Trend maintained: {trend_maintained}")
                        
                        algorithm_tests.append(("Linear fit quality", linear_fit_quality))
                        algorithm_tests.append(("Linear trend maintained", trend_maintained))
                    except Exception as e:
                        print(f"   ❌ Linear fit error: {e}")
                        algorithm_tests.extend([
                            ("Linear fit quality", False),
                            ("Linear trend maintained", False)
                        ])
                else:
                    print("   ❌ Insufficient linear predictions for algorithm testing")
                    algorithm_tests.extend([
                        ("Linear fit quality", False),
                        ("Linear trend maintained", False)
                    ])
            else:
                print(f"   ❌ Failed to generate linear predictions: {response.status_code}")
                algorithm_tests.extend([
                    ("Linear fit quality", False),
                    ("Linear trend maintained", False)
                ])
        
        # Evaluate algorithm performance results
        passed_tests = sum(1 for _, passed in algorithm_tests if passed)
        total_tests = len(algorithm_tests)
        
        print(f"\n📊 Pattern-specific algorithm test results: {passed_tests}/{total_tests}")
        for test_name, passed in algorithm_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['pattern_specific_algorithms'] = passed_tests >= total_tests * 0.7
    
    def run_all_tests(self):
        """Run all pattern-aware prediction tests"""
        print("🎯 Starting Enhanced Pattern-Aware Prediction System Testing")
        print("=" * 70)
        
        # Run all tests
        self.test_pattern_detection()
        self.test_pattern_aware_predictions()
        self.test_continuous_prediction_continuity()
        self.test_downward_bias_elimination()
        self.test_pattern_specific_algorithms()
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("🎯 ENHANCED PATTERN-AWARE PREDICTION SYSTEM TEST REPORT")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"\n📊 OVERALL RESULTS: {passed_tests}/{total_tests} test categories passed")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 DETAILED RESULTS:")
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {test_name.replace('_', ' ').title()}")
        
        # Determine overall system status
        if passed_tests >= total_tests * 0.8:
            print("\n🎉 SYSTEM STATUS: EXCELLENT - Pattern-aware prediction system is working well!")
        elif passed_tests >= total_tests * 0.6:
            print("\n✅ SYSTEM STATUS: GOOD - Pattern-aware prediction system is functional with minor issues")
        elif passed_tests >= total_tests * 0.4:
            print("\n⚠️  SYSTEM STATUS: NEEDS IMPROVEMENT - Some pattern-aware features need attention")
        else:
            print("\n❌ SYSTEM STATUS: CRITICAL ISSUES - Pattern-aware prediction system needs significant fixes")
        
        print("\n🔍 KEY FINDINGS:")
        
        if self.test_results.get('pattern_detection', False):
            print("   ✅ Pattern detection is working correctly")
        else:
            print("   ❌ Pattern detection needs improvement")
        
        if self.test_results.get('downward_bias_elimination', False):
            print("   ✅ Downward bias has been successfully eliminated")
        else:
            print("   ❌ Downward bias issues still present")
        
        if self.test_results.get('continuous_prediction_continuity', False):
            print("   ✅ Continuous predictions maintain proper continuity")
        else:
            print("   ❌ Continuous prediction continuity needs work")
        
        if self.test_results.get('pattern_specific_algorithms', False):
            print("   ✅ Pattern-specific algorithms are performing well")
        else:
            print("   ❌ Pattern-specific algorithms need optimization")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    tester = PatternAwareTester()
    tester.run_all_tests()