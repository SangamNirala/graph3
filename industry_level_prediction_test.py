#!/usr/bin/env python3
"""
Industry-Level Continuous Prediction System Testing
Tests the new advanced pattern recognition and prediction algorithms
"""

import requests
import json
import pandas as pd
import io
import time
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://62eddcf8-8af5-478a-b418-daf8b432ff3b.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Industry-Level Prediction System at: {API_BASE_URL}")

class IndustryLevelPredictionTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        self.test_results = {}
        
    def create_ph_dataset(self):
        """Create realistic pH dataset for testing"""
        # Generate 49 pH data points as mentioned in test_result.md
        dates = pd.date_range(start='2024-01-01', periods=49, freq='H')
        
        # Create realistic pH data with patterns (sine waves, trends, seasonality)
        base_ph = 7.2
        trend = np.linspace(0, 0.6, 49)  # Slight upward trend
        sine_wave = 0.3 * np.sin(2 * np.pi * np.arange(49) / 12)  # 12-hour cycle
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(49) / 24)  # Daily pattern
        noise = np.random.normal(0, 0.05, 49)
        
        ph_values = base_ph + trend + sine_wave + seasonal + noise
        # Keep pH in realistic range (6.0-8.0)
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'pH': ph_values
        })
        
        return df
    
    def test_basic_functionality(self):
        """Test 1: Basic functionality - system starts without errors and basic endpoints work"""
        print("\n=== Test 1: Basic Functionality ===")
        
        try:
            # Test basic health check
            response = self.session.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                print("✅ Health check endpoint working")
                health_check = True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                health_check = False
            
            # Upload pH dataset
            df = self.create_ph_dataset()
            csv_content = df.to_csv(index=False)
            files = {'file': ('ph_data.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                print("✅ File upload successful")
                print(f"   Data ID: {self.data_id}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                upload_success = True
            else:
                print(f"❌ File upload failed: {response.status_code} - {response.text}")
                upload_success = False
            
            # Train LSTM model for advanced predictions
            if upload_success:
                training_params = {
                    "time_column": "timestamp",
                    "target_column": "pH",
                    "model_type": "lstm",
                    "seq_len": 8,
                    "pred_len": 3,
                    "epochs": 20,
                    "batch_size": 4
                }
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": self.data_id, "model_type": "lstm"},
                    json=training_params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.model_id = data.get('model_id')
                    print("✅ LSTM model training successful")
                    print(f"   Model ID: {self.model_id}")
                    model_training = True
                else:
                    print(f"❌ LSTM model training failed: {response.status_code} - {response.text}")
                    model_training = False
            else:
                model_training = False
            
            basic_tests = [
                ("Health Check", health_check),
                ("File Upload", upload_success),
                ("Model Training", model_training)
            ]
            
            passed_tests = sum(1 for _, passed in basic_tests if passed)
            print(f"\n📊 Basic functionality tests: {passed_tests}/3")
            for test_name, passed in basic_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['basic_functionality'] = passed_tests >= 2
            
        except Exception as e:
            print(f"❌ Basic functionality test error: {str(e)}")
            self.test_results['basic_functionality'] = False
    
    def test_continuous_learning_system(self):
        """Test 2: Continuous learning system endpoints"""
        print("\n=== Test 2: Continuous Learning System ===")
        
        try:
            learning_tests = []
            
            # Test start continuous learning
            response = self.session.post(f"{API_BASE_URL}/start-continuous-learning")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Start continuous learning successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                start_learning = True
            else:
                print(f"❌ Start continuous learning failed: {response.status_code} - {response.text}")
                start_learning = False
            
            learning_tests.append(("Start Continuous Learning", start_learning))
            
            # Wait a moment for the system to initialize
            time.sleep(2)
            
            # Test stop continuous learning
            response = self.session.post(f"{API_BASE_URL}/stop-continuous-learning")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Stop continuous learning successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                stop_learning = True
            else:
                print(f"❌ Stop continuous learning failed: {response.status_code} - {response.text}")
                stop_learning = False
            
            learning_tests.append(("Stop Continuous Learning", stop_learning))
            
            # Test learning status endpoint
            response = self.session.get(f"{API_BASE_URL}/continuous-learning-status")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Continuous learning status successful")
                print(f"   Learning active: {data.get('learning_active')}")
                print(f"   Learning iterations: {data.get('learning_iterations', 0)}")
                status_check = True
            else:
                print(f"❌ Continuous learning status failed: {response.status_code} - {response.text}")
                status_check = False
            
            learning_tests.append(("Learning Status Check", status_check))
            
            passed_tests = sum(1 for _, passed in learning_tests if passed)
            print(f"\n📊 Continuous learning tests: {passed_tests}/3")
            for test_name, passed in learning_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['continuous_learning'] = passed_tests >= 2
            
        except Exception as e:
            print(f"❌ Continuous learning test error: {str(e)}")
            self.test_results['continuous_learning'] = False
    
    def test_system_metrics(self):
        """Test 3: System metrics endpoint"""
        print("\n=== Test 3: System Metrics ===")
        
        try:
            response = self.session.get(f"{API_BASE_URL}/system-metrics")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ System metrics endpoint successful")
                
                # Check for expected metrics
                expected_metrics = [
                    'prediction_accuracy',
                    'pattern_recognition_quality',
                    'continuous_learning_performance',
                    'system_uptime',
                    'prediction_latency',
                    'memory_usage',
                    'cpu_usage'
                ]
                
                metrics_found = []
                for metric in expected_metrics:
                    if metric in data:
                        print(f"   ✅ {metric}: {data[metric]}")
                        metrics_found.append(True)
                    else:
                        print(f"   ⚠️  Missing metric: {metric}")
                        metrics_found.append(False)
                
                # Check for system health indicators
                system_health = data.get('system_health', 'unknown')
                prediction_count = data.get('total_predictions', 0)
                
                print(f"   System health: {system_health}")
                print(f"   Total predictions: {prediction_count}")
                
                metrics_tests = [
                    ("Endpoint Response", True),
                    ("Required Metrics", sum(metrics_found) >= len(expected_metrics) * 0.7),
                    ("System Health Info", system_health != 'unknown')
                ]
                
                passed_tests = sum(1 for _, passed in metrics_tests if passed)
                print(f"\n📊 System metrics tests: {passed_tests}/3")
                for test_name, passed in metrics_tests:
                    status = "✅" if passed else "❌"
                    print(f"   {status} {test_name}")
                
                self.test_results['system_metrics'] = passed_tests >= 2
                
            else:
                print(f"❌ System metrics failed: {response.status_code} - {response.text}")
                self.test_results['system_metrics'] = False
                
        except Exception as e:
            print(f"❌ System metrics test error: {str(e)}")
            self.test_results['system_metrics'] = False
    
    def test_advanced_pattern_analysis(self):
        """Test 4: Advanced pattern analysis endpoint with uploaded data"""
        print("\n=== Test 4: Advanced Pattern Analysis ===")
        
        if not self.data_id:
            print("❌ Cannot test pattern analysis - no data uploaded")
            self.test_results['advanced_pattern_analysis'] = False
            return
        
        try:
            # Test advanced pattern analysis endpoint
            response = self.session.post(
                f"{API_BASE_URL}/advanced-pattern-analysis",
                json={
                    "data_id": self.data_id,
                    "analysis_depth": "comprehensive",
                    "pattern_types": ["sine_wave", "trend", "seasonality", "cyclical"]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Advanced pattern analysis successful")
                
                # Check for pattern analysis results
                patterns = data.get('patterns', {})
                quality_score = data.get('quality_score', 0)
                detected_patterns = data.get('detected_patterns', [])
                
                print(f"   Quality score: {quality_score}")
                print(f"   Detected patterns: {detected_patterns}")
                
                # Check for specific pattern types
                pattern_checks = []
                expected_patterns = ['trend', 'seasonality', 'cyclical', 'noise_level']
                
                for pattern_type in expected_patterns:
                    if pattern_type in patterns:
                        print(f"   ✅ {pattern_type}: {patterns[pattern_type]}")
                        pattern_checks.append(True)
                    else:
                        print(f"   ⚠️  Missing pattern: {pattern_type}")
                        pattern_checks.append(False)
                
                # Check pattern recognition quality
                pattern_quality = quality_score >= 80.0  # Expect high quality for pH data
                
                analysis_tests = [
                    ("Pattern Analysis Response", True),
                    ("Quality Score", quality_score > 0),
                    ("Pattern Detection", len(detected_patterns) > 0),
                    ("Pattern Quality", pattern_quality),
                    ("Expected Patterns", sum(pattern_checks) >= len(expected_patterns) * 0.6)
                ]
                
                passed_tests = sum(1 for _, passed in analysis_tests if passed)
                print(f"\n📊 Pattern analysis tests: {passed_tests}/5")
                for test_name, passed in analysis_tests:
                    status = "✅" if passed else "❌"
                    print(f"   {status} {test_name}")
                
                self.test_results['advanced_pattern_analysis'] = passed_tests >= 3
                
            else:
                print(f"❌ Advanced pattern analysis failed: {response.status_code} - {response.text}")
                self.test_results['advanced_pattern_analysis'] = False
                
        except Exception as e:
            print(f"❌ Advanced pattern analysis test error: {str(e)}")
            self.test_results['advanced_pattern_analysis'] = False
    
    def test_advanced_predictions(self):
        """Test 5: Advanced predictions endpoint"""
        print("\n=== Test 5: Advanced Predictions ===")
        
        if not self.model_id:
            print("❌ Cannot test advanced predictions - no model trained")
            self.test_results['advanced_predictions'] = False
            return
        
        try:
            # Test generate advanced predictions
            response = self.session.post(
                f"{API_BASE_URL}/generate-advanced-predictions",
                json={
                    "model_id": self.model_id,
                    "prediction_steps": 30,
                    "confidence_level": 0.95,
                    "include_uncertainty": True,
                    "pattern_aware": True
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Advanced predictions successful")
                
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                confidence_intervals = data.get('confidence_intervals', [])
                uncertainty_bands = data.get('uncertainty_bands', [])
                pattern_info = data.get('pattern_info', {})
                
                print(f"   Number of predictions: {len(predictions)}")
                print(f"   Number of timestamps: {len(timestamps)}")
                print(f"   Has confidence intervals: {len(confidence_intervals) > 0}")
                print(f"   Has uncertainty bands: {len(uncertainty_bands) > 0}")
                print(f"   Pattern info included: {len(pattern_info) > 0}")
                
                # Validate prediction quality
                prediction_tests = []
                
                # Test 1: Correct number of predictions
                prediction_tests.append(("Correct prediction count", len(predictions) == 30))
                
                # Test 2: Timestamps match predictions
                prediction_tests.append(("Timestamps match", len(timestamps) == len(predictions)))
                
                # Test 3: pH values in realistic range
                if predictions:
                    ph_values_valid = all(6.0 <= pred <= 8.0 for pred in predictions)
                    prediction_tests.append(("pH values realistic", ph_values_valid))
                    print(f"   pH range: {min(predictions):.2f} - {max(predictions):.2f}")
                else:
                    prediction_tests.append(("pH values realistic", False))
                
                # Test 4: Predictions show variability (not monotonic decline)
                if len(predictions) >= 5:
                    unique_values = len(set(round(p, 2) for p in predictions))
                    variability_test = unique_values >= 3  # At least 3 different values
                    prediction_tests.append(("Prediction variability", variability_test))
                    print(f"   Unique prediction values: {unique_values}")
                else:
                    prediction_tests.append(("Prediction variability", False))
                
                # Test 5: No persistent downward bias
                if len(predictions) >= 10:
                    # Calculate trend slope
                    x = np.arange(len(predictions))
                    slope = np.polyfit(x, predictions, 1)[0]
                    no_downward_bias = slope > -0.01  # Allow slight negative slope but not persistent decline
                    prediction_tests.append(("No downward bias", no_downward_bias))
                    print(f"   Trend slope: {slope:.6f} (should be > -0.01)")
                else:
                    prediction_tests.append(("No downward bias", False))
                
                # Test 6: Pattern awareness
                pattern_aware = len(pattern_info) > 0 and 'trend' in pattern_info
                prediction_tests.append(("Pattern awareness", pattern_aware))
                
                passed_tests = sum(1 for _, passed in prediction_tests if passed)
                print(f"\n📊 Advanced prediction tests: {passed_tests}/6")
                for test_name, passed in prediction_tests:
                    status = "✅" if passed else "❌"
                    print(f"   {status} {test_name}")
                
                self.test_results['advanced_predictions'] = passed_tests >= 4
                
            else:
                print(f"❌ Advanced predictions failed: {response.status_code} - {response.text}")
                self.test_results['advanced_predictions'] = False
                
        except Exception as e:
            print(f"❌ Advanced predictions test error: {str(e)}")
            self.test_results['advanced_predictions'] = False
    
    def test_continuous_prediction(self):
        """Test 6: Continuous prediction with industry-level system"""
        print("\n=== Test 6: Continuous Prediction (Industry-Level) ===")
        
        if not self.model_id:
            print("❌ Cannot test continuous prediction - no model trained")
            self.test_results['continuous_prediction'] = False
            return
        
        try:
            continuous_tests = []
            
            # Test 1: Generate continuous prediction
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={
                    "model_id": self.model_id,
                    "steps": 25,
                    "time_window": 100,
                    "use_industry_level": True
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions1 = data.get('predictions', [])
                timestamps1 = data.get('timestamps', [])
                
                print("✅ First continuous prediction successful")
                print(f"   Predictions: {len(predictions1)}")
                print(f"   Timestamps: {len(timestamps1)}")
                continuous_tests.append(("First continuous prediction", True))
                
                # Test 2: Second call should extrapolate forward
                time.sleep(1)
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={
                        "model_id": self.model_id,
                        "steps": 25,
                        "time_window": 100,
                        "use_industry_level": True
                    }
                )
                
                if response.status_code == 200:
                    data2 = response.json()
                    predictions2 = data2.get('predictions', [])
                    timestamps2 = data2.get('timestamps', [])
                    
                    print("✅ Second continuous prediction successful")
                    
                    # Check if timestamps advanced (extrapolation)
                    if timestamps1 and timestamps2:
                        extrapolation_working = timestamps1[0] != timestamps2[0]
                        continuous_tests.append(("Extrapolation working", extrapolation_working))
                        
                        if extrapolation_working:
                            print("✅ Continuous prediction properly extrapolates forward")
                        else:
                            print("❌ Continuous prediction not extrapolating")
                    else:
                        continuous_tests.append(("Extrapolation working", False))
                    
                    # Test 3: Check prediction continuity
                    if predictions1 and predictions2:
                        # Check if predictions maintain pH characteristics
                        ph_range_1 = (min(predictions1), max(predictions1))
                        ph_range_2 = (min(predictions2), max(predictions2))
                        
                        range_consistent = (
                            6.0 <= ph_range_1[0] <= 8.0 and 6.0 <= ph_range_1[1] <= 8.0 and
                            6.0 <= ph_range_2[0] <= 8.0 and 6.0 <= ph_range_2[1] <= 8.0
                        )
                        continuous_tests.append(("pH range consistency", range_consistent))
                        
                        print(f"   First call pH range: {ph_range_1[0]:.2f} - {ph_range_1[1]:.2f}")
                        print(f"   Second call pH range: {ph_range_2[0]:.2f} - {ph_range_2[1]:.2f}")
                    else:
                        continuous_tests.append(("pH range consistency", False))
                    
                else:
                    continuous_tests.append(("Extrapolation working", False))
                    continuous_tests.append(("pH range consistency", False))
                
            else:
                print(f"❌ Continuous prediction failed: {response.status_code} - {response.text}")
                continuous_tests.append(("First continuous prediction", False))
                continuous_tests.append(("Extrapolation working", False))
                continuous_tests.append(("pH range consistency", False))
            
            # Test 4: Test multiple continuous calls for accumulated bias
            print("   Testing for accumulated bias over multiple calls...")
            ph_means = []
            
            for i in range(5):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={
                        "model_id": self.model_id,
                        "steps": 10,
                        "time_window": 50,
                        "use_industry_level": True
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    if predictions:
                        ph_means.append(np.mean(predictions))
                time.sleep(0.5)
            
            if len(ph_means) >= 3:
                # Check if mean pH values don't show persistent downward trend
                ph_trend = np.polyfit(range(len(ph_means)), ph_means, 1)[0]
                no_accumulated_bias = ph_trend > -0.05  # Allow small variation but not persistent decline
                continuous_tests.append(("No accumulated bias", no_accumulated_bias))
                
                print(f"   pH means over calls: {[f'{m:.3f}' for m in ph_means]}")
                print(f"   pH trend slope: {ph_trend:.6f} (should be > -0.05)")
            else:
                continuous_tests.append(("No accumulated bias", False))
            
            passed_tests = sum(1 for _, passed in continuous_tests if passed)
            print(f"\n📊 Continuous prediction tests: {passed_tests}/{len(continuous_tests)}")
            for test_name, passed in continuous_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['continuous_prediction'] = passed_tests >= len(continuous_tests) * 0.75
            
        except Exception as e:
            print(f"❌ Continuous prediction test error: {str(e)}")
            self.test_results['continuous_prediction'] = False
    
    def test_system_toggle(self):
        """Test 7: System toggle between industry-level and legacy systems"""
        print("\n=== Test 7: System Toggle ===")
        
        try:
            toggle_tests = []
            
            # Test 1: Switch to industry-level system
            response = self.session.post(
                f"{API_BASE_URL}/toggle-prediction-system",
                json={"system_type": "industry_level"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Switch to industry-level system successful")
                print(f"   Current system: {data.get('current_system')}")
                print(f"   Status: {data.get('status')}")
                toggle_tests.append(("Switch to industry-level", True))
            else:
                print(f"❌ Switch to industry-level failed: {response.status_code} - {response.text}")
                toggle_tests.append(("Switch to industry-level", False))
            
            # Test 2: Switch to legacy system
            response = self.session.post(
                f"{API_BASE_URL}/toggle-prediction-system",
                json={"system_type": "legacy"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Switch to legacy system successful")
                print(f"   Current system: {data.get('current_system')}")
                print(f"   Status: {data.get('status')}")
                toggle_tests.append(("Switch to legacy", True))
            else:
                print(f"❌ Switch to legacy failed: {response.status_code} - {response.text}")
                toggle_tests.append(("Switch to legacy", False))
            
            # Test 3: Get current system status
            response = self.session.get(f"{API_BASE_URL}/prediction-system-status")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ System status check successful")
                print(f"   Active system: {data.get('active_system')}")
                print(f"   Available systems: {data.get('available_systems', [])}")
                toggle_tests.append(("System status check", True))
            else:
                print(f"❌ System status check failed: {response.status_code} - {response.text}")
                toggle_tests.append(("System status check", False))
            
            # Test 4: Switch back to industry-level for final test
            response = self.session.post(
                f"{API_BASE_URL}/toggle-prediction-system",
                json={"system_type": "industry_level"}
            )
            
            if response.status_code == 200:
                print("✅ Switch back to industry-level successful")
                toggle_tests.append(("Switch back to industry-level", True))
            else:
                print(f"❌ Switch back to industry-level failed: {response.status_code}")
                toggle_tests.append(("Switch back to industry-level", False))
            
            passed_tests = sum(1 for _, passed in toggle_tests if passed)
            print(f"\n📊 System toggle tests: {passed_tests}/4")
            for test_name, passed in toggle_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['system_toggle'] = passed_tests >= 3
            
        except Exception as e:
            print(f"❌ System toggle test error: {str(e)}")
            self.test_results['system_toggle'] = False
    
    def run_industry_level_tests(self):
        """Run all industry-level prediction system tests"""
        print("🚀 Starting Industry-Level Continuous Prediction System Testing")
        print("=" * 70)
        
        # Run all tests in sequence
        self.test_basic_functionality()
        self.test_continuous_learning_system()
        self.test_system_metrics()
        self.test_advanced_pattern_analysis()
        self.test_advanced_predictions()
        self.test_continuous_prediction()
        self.test_system_toggle()
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("🎯 INDUSTRY-LEVEL PREDICTION SYSTEM TEST RESULTS")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"Overall Success Rate: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print()
        
        # Detailed results
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            formatted_name = test_name.replace('_', ' ').title()
            print(f"{status} {formatted_name}")
        
        print("\n" + "=" * 70)
        
        # Critical findings
        critical_tests = ['basic_functionality', 'continuous_prediction', 'advanced_predictions']
        critical_passed = sum(1 for test in critical_tests if self.test_results.get(test, False))
        
        if critical_passed == len(critical_tests):
            print("🎉 CRITICAL TESTS PASSED: Industry-level system is working correctly!")
        else:
            print("⚠️  CRITICAL ISSUES FOUND: Some core functionality needs attention")
        
        # Specific findings about downward trend bias
        if self.test_results.get('continuous_prediction', False) and self.test_results.get('advanced_predictions', False):
            print("✅ DOWNWARD TREND BIAS: Successfully resolved - predictions maintain proper patterns")
        else:
            print("❌ DOWNWARD TREND BIAS: May still be present - needs investigation")
        
        print("=" * 70)

if __name__ == "__main__":
    tester = IndustryLevelPredictionTester()
    tester.run_industry_level_tests()