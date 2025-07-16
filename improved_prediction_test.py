#!/usr/bin/env python3
"""
Comprehensive Testing for Improved Prediction System
Tests the enhanced prediction algorithms with pattern recognition and better extrapolation
"""

import requests
import json
import pandas as pd
import numpy as np
import io
import time
from datetime import datetime, timedelta
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://8ecdc457-5a6d-405e-b237-5d7187d1504c.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing improved prediction system at: {API_BASE_URL}")

class ImprovedPredictionTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_trend_data(self):
        """Create sample data with clear upward trend for testing pattern recognition"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        # Strong upward trend with some noise
        trend = np.linspace(100, 200, 50)  # Clear upward trend
        noise = np.random.normal(0, 5, 50)  # Small noise
        values = trend + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'category': ['A'] * 25 + ['B'] * 25
        })
        
        return df
    
    def create_seasonal_data(self):
        """Create sample data with clear seasonal patterns for testing seasonality recognition"""
        dates = pd.date_range(start='2023-01-01', periods=84, freq='D')  # 12 weeks
        
        # Weekly seasonal pattern + slight trend
        base_trend = np.linspace(50, 60, 84)
        weekly_pattern = 20 * np.sin(2 * np.pi * np.arange(84) / 7)  # Weekly cycle
        noise = np.random.normal(0, 3, 84)
        values = base_trend + weekly_pattern + noise
        
        df = pd.DataFrame({
            'date': dates,
            'sales': values,
            'region': ['North'] * 42 + ['South'] * 42
        })
        
        return df
    
    def create_volatile_data(self):
        """Create sample data with high volatility for testing smoothing algorithms"""
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        
        # Base trend with high volatility
        base = np.linspace(1000, 1100, 60)
        volatility = np.random.normal(0, 50, 60)  # High volatility
        spikes = np.zeros(60)
        spikes[10] = 200  # Add some spikes
        spikes[30] = -150
        spikes[45] = 180
        
        values = base + volatility + spikes
        
        df = pd.DataFrame({
            'date': dates,
            'price': values,
            'market': ['crypto'] * 60
        })
        
        return df
    
    def test_core_functionality_with_improved_algorithms(self):
        """Test 1: Core Functionality with Improved Algorithms"""
        print("\n=== Testing Core Functionality with Improved Algorithms ===")
        
        try:
            # Test with trend data
            df = self.create_trend_data()
            csv_content = df.to_csv(index=False)
            
            # Upload data
            files = {'file': ('trend_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ File upload failed: {response.status_code}")
                self.test_results['core_functionality'] = False
                return
            
            data = response.json()
            data_id = data.get('data_id')
            print(f"✅ Data uploaded successfully. Data ID: {data_id}")
            
            # Train improved ARIMA model
            training_params = {
                "time_column": "date",
                "target_column": "value"
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json=training_params
            )
            
            if response.status_code != 200:
                print(f"❌ ARIMA model training failed: {response.status_code} - {response.text}")
                self.test_results['core_functionality'] = False
                return
            
            model_data = response.json()
            model_id = model_data.get('model_id')
            print(f"✅ Improved ARIMA model trained successfully. Model ID: {model_id}")
            
            # Generate predictions with improved algorithm
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 20}
            )
            
            if response.status_code != 200:
                print(f"❌ Prediction generation failed: {response.status_code}")
                self.test_results['core_functionality'] = False
                return
            
            pred_data = response.json()
            predictions = pred_data.get('predictions', [])
            timestamps = pred_data.get('timestamps', [])
            
            print(f"✅ Improved predictions generated: {len(predictions)} points")
            print(f"   Sample predictions: {predictions[:5]}")
            print(f"   Sample timestamps: {timestamps[:3]}")
            
            # Validate prediction quality
            if len(predictions) == 20 and len(timestamps) == 20:
                # Check if predictions follow the upward trend
                trend_following = predictions[-1] > predictions[0]  # Should be increasing
                print(f"✅ Predictions follow trend pattern: {trend_following}")
                
                self.test_results['core_functionality'] = trend_following
            else:
                print("❌ Incorrect prediction structure")
                self.test_results['core_functionality'] = False
                
        except Exception as e:
            print(f"❌ Core functionality test error: {str(e)}")
            self.test_results['core_functionality'] = False
    
    def test_pattern_recognition_with_seasonal_data(self):
        """Test 2: Pattern Recognition with Seasonal Data"""
        print("\n=== Testing Pattern Recognition with Seasonal Data ===")
        
        try:
            # Test with seasonal data
            df = self.create_seasonal_data()
            csv_content = df.to_csv(index=False)
            
            # Upload seasonal data
            files = {'file': ('seasonal_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Seasonal data upload failed: {response.status_code}")
                self.test_results['pattern_recognition'] = False
                return
            
            data = response.json()
            data_id = data.get('data_id')
            print(f"✅ Seasonal data uploaded successfully")
            
            # Train improved Prophet model (better for seasonality)
            training_params = {
                "time_column": "date",
                "target_column": "sales",
                "seasonality_mode": "additive"
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "prophet"},
                json=training_params
            )
            
            if response.status_code != 200:
                print(f"❌ Prophet model training failed: {response.status_code} - {response.text}")
                self.test_results['pattern_recognition'] = False
                return
            
            model_data = response.json()
            model_id = model_data.get('model_id')
            print(f"✅ Improved Prophet model trained for seasonal data")
            
            # Generate predictions to test pattern recognition
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 14}  # 2 weeks to capture pattern
            )
            
            if response.status_code != 200:
                print(f"❌ Pattern recognition prediction failed: {response.status_code}")
                self.test_results['pattern_recognition'] = False
                return
            
            pred_data = response.json()
            predictions = pred_data.get('predictions', [])
            confidence_intervals = pred_data.get('confidence_intervals', [])
            
            print(f"✅ Pattern-aware predictions generated: {len(predictions)} points")
            print(f"   Has confidence intervals: {confidence_intervals is not None}")
            
            # Test pattern continuity - check if weekly pattern is maintained
            if len(predictions) >= 14:
                # Compare predictions 7 days apart (should show weekly pattern)
                weekly_diff_1 = abs(predictions[0] - predictions[7])
                weekly_diff_2 = abs(predictions[1] - predictions[8])
                avg_weekly_diff = (weekly_diff_1 + weekly_diff_2) / 2
                
                # Check if pattern is maintained (weekly differences should be small)
                pattern_maintained = avg_weekly_diff < np.std(predictions) * 0.5
                print(f"✅ Weekly pattern maintained in predictions: {pattern_maintained}")
                print(f"   Average weekly difference: {avg_weekly_diff:.2f}")
                
                self.test_results['pattern_recognition'] = pattern_maintained
            else:
                print("❌ Insufficient predictions for pattern analysis")
                self.test_results['pattern_recognition'] = False
                
        except Exception as e:
            print(f"❌ Pattern recognition test error: {str(e)}")
            self.test_results['pattern_recognition'] = False
    
    def test_continuous_prediction_extrapolation(self):
        """Test 3: Enhanced Continuous Prediction Extrapolation"""
        print("\n=== Testing Enhanced Continuous Prediction Extrapolation ===")
        
        try:
            # Use trend data for extrapolation testing
            df = self.create_trend_data()
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('extrapolation_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Data upload for extrapolation test failed")
                self.test_results['continuous_extrapolation'] = False
                return
            
            data_id = response.json().get('data_id')
            
            # Train ARIMA model
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "date", "target_column": "value"}
            )
            
            if response.status_code != 200:
                print(f"❌ Model training for extrapolation test failed")
                self.test_results['continuous_extrapolation'] = False
                return
            
            model_id = response.json().get('model_id')
            print(f"✅ Model trained for extrapolation testing")
            
            # Reset continuous prediction state
            response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            if response.status_code != 200:
                print(f"❌ Reset continuous prediction failed")
                self.test_results['continuous_extrapolation'] = False
                return
            
            print("✅ Continuous prediction state reset")
            
            # Test multiple continuous prediction calls for extrapolation
            timestamps_sequence = []
            predictions_sequence = []
            
            for i in range(4):  # Make 4 calls to test extrapolation
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 10, "time_window": 50}
                )
                
                if response.status_code != 200:
                    print(f"❌ Continuous prediction call {i+1} failed")
                    self.test_results['continuous_extrapolation'] = False
                    return
                
                data = response.json()
                timestamps = data.get('timestamps', [])
                predictions = data.get('predictions', [])
                
                timestamps_sequence.append(timestamps)
                predictions_sequence.append(predictions)
                
                print(f"   Call {i+1}: Generated {len(predictions)} predictions")
                if timestamps:
                    print(f"   First timestamp: {timestamps[0]}")
                
                time.sleep(0.5)  # Small delay between calls
            
            # Analyze extrapolation behavior
            if len(timestamps_sequence) >= 2:
                # Check if timestamps are advancing (extrapolating forward)
                first_call_start = timestamps_sequence[0][0] if timestamps_sequence[0] else ""
                second_call_start = timestamps_sequence[1][0] if timestamps_sequence[1] else ""
                
                extrapolation_working = first_call_start != second_call_start
                print(f"✅ Continuous extrapolation working: {extrapolation_working}")
                
                if extrapolation_working:
                    # Check if predictions maintain trend continuity
                    trend_continuity = True
                    for i in range(1, len(predictions_sequence)):
                        if predictions_sequence[i] and predictions_sequence[i-1]:
                            # Check if general trend is maintained
                            current_avg = np.mean(predictions_sequence[i])
                            previous_avg = np.mean(predictions_sequence[i-1])
                            
                            # For upward trend data, predictions should generally increase
                            if current_avg < previous_avg * 0.8:  # Allow some variation
                                trend_continuity = False
                                break
                    
                    print(f"✅ Trend continuity maintained: {trend_continuity}")
                    self.test_results['continuous_extrapolation'] = extrapolation_working and trend_continuity
                else:
                    self.test_results['continuous_extrapolation'] = False
            else:
                print("❌ Insufficient data for extrapolation analysis")
                self.test_results['continuous_extrapolation'] = False
                
        except Exception as e:
            print(f"❌ Continuous extrapolation test error: {str(e)}")
            self.test_results['continuous_extrapolation'] = False
    
    def test_prediction_accuracy_and_realism(self):
        """Test 4: Prediction Accuracy and Realism"""
        print("\n=== Testing Prediction Accuracy and Realism ===")
        
        try:
            # Test with volatile data to check smoothing and bounds
            df = self.create_volatile_data()
            csv_content = df.to_csv(index=False)
            
            # Upload volatile data
            files = {'file': ('volatile_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Volatile data upload failed")
                self.test_results['prediction_accuracy'] = False
                return
            
            data_id = response.json().get('data_id')
            
            # Train ARIMA model
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "date", "target_column": "price"}
            )
            
            if response.status_code != 200:
                print(f"❌ Model training for accuracy test failed")
                self.test_results['prediction_accuracy'] = False
                return
            
            model_id = response.json().get('model_id')
            print(f"✅ Model trained for accuracy testing")
            
            # Generate predictions
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 15}
            )
            
            if response.status_code != 200:
                print(f"❌ Prediction generation for accuracy test failed")
                self.test_results['prediction_accuracy'] = False
                return
            
            pred_data = response.json()
            predictions = pred_data.get('predictions', [])
            
            print(f"✅ Predictions generated for accuracy testing: {len(predictions)} points")
            
            # Analyze prediction quality
            if predictions:
                # Calculate historical statistics for comparison
                historical_values = df['price'].values
                hist_mean = np.mean(historical_values)
                hist_std = np.std(historical_values)
                hist_min = np.min(historical_values)
                hist_max = np.max(historical_values)
                
                pred_mean = np.mean(predictions)
                pred_std = np.std(predictions)
                pred_min = np.min(predictions)
                pred_max = np.max(predictions)
                
                print(f"   Historical: mean={hist_mean:.2f}, std={hist_std:.2f}, range=[{hist_min:.2f}, {hist_max:.2f}]")
                print(f"   Predictions: mean={pred_mean:.2f}, std={pred_std:.2f}, range=[{pred_min:.2f}, {pred_max:.2f}]")
                
                # Test 1: Predictions should be within reasonable bounds
                reasonable_bounds = (pred_min >= hist_mean - 3*hist_std) and (pred_max <= hist_mean + 3*hist_std)
                print(f"✅ Predictions within reasonable bounds: {reasonable_bounds}")
                
                # Test 2: Predictions should be smoother than historical data (less volatile)
                smoothness_improved = pred_std <= hist_std * 1.2  # Allow some increase but not too much
                print(f"✅ Predictions are appropriately smoothed: {smoothness_improved}")
                
                # Test 3: Check for unrealistic jumps between consecutive predictions
                max_jump = 0
                for i in range(1, len(predictions)):
                    jump = abs(predictions[i] - predictions[i-1])
                    max_jump = max(max_jump, jump)
                
                # Max jump should be reasonable compared to historical volatility
                realistic_jumps = max_jump <= hist_std * 2
                print(f"✅ No unrealistic jumps in predictions: {realistic_jumps}")
                print(f"   Max prediction jump: {max_jump:.2f} vs historical std: {hist_std:.2f}")
                
                # Test 4: Predictions should maintain some continuity with last historical value
                last_historical = historical_values[-1]
                first_prediction = predictions[0]
                continuity_maintained = abs(first_prediction - last_historical) <= hist_std * 1.5
                print(f"✅ Continuity with historical data maintained: {continuity_maintained}")
                print(f"   Last historical: {last_historical:.2f}, First prediction: {first_prediction:.2f}")
                
                # Overall accuracy score
                accuracy_tests = [reasonable_bounds, smoothness_improved, realistic_jumps, continuity_maintained]
                accuracy_score = sum(accuracy_tests) / len(accuracy_tests)
                
                print(f"✅ Overall accuracy score: {accuracy_score:.2f} ({sum(accuracy_tests)}/{len(accuracy_tests)} tests passed)")
                
                self.test_results['prediction_accuracy'] = accuracy_score >= 0.75  # 75% of tests should pass
            else:
                print("❌ No predictions generated")
                self.test_results['prediction_accuracy'] = False
                
        except Exception as e:
            print(f"❌ Prediction accuracy test error: {str(e)}")
            self.test_results['prediction_accuracy'] = False
    
    def test_api_response_structures(self):
        """Test 5: API Response Structures for Improved Predictions"""
        print("\n=== Testing API Response Structures ===")
        
        try:
            # Use simple data for structure testing
            df = self.create_trend_data()
            csv_content = df.to_csv(index=False)
            
            # Upload and train
            files = {'file': ('structure_test.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Data upload for structure test failed")
                self.test_results['api_response_structure'] = False
                return
            
            data_id = response.json().get('data_id')
            
            # Train both models to test different response structures
            models_to_test = [
                ("prophet", "Prophet model response structure"),
                ("arima", "ARIMA model response structure")
            ]
            
            structure_tests = []
            
            for model_type, test_name in models_to_test:
                print(f"\n   Testing {test_name}...")
                
                # Train model
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": model_type},
                    json={"time_column": "date", "target_column": "value"}
                )
                
                if response.status_code != 200:
                    print(f"   ❌ {model_type} training failed")
                    structure_tests.append(False)
                    continue
                
                model_id = response.json().get('model_id')
                
                # Test regular prediction structure
                response = self.session.get(
                    f"{API_BASE_URL}/generate-prediction",
                    params={"model_id": model_id, "steps": 10}
                )
                
                if response.status_code != 200:
                    print(f"   ❌ {model_type} prediction failed")
                    structure_tests.append(False)
                    continue
                
                pred_data = response.json()
                
                # Validate response structure
                required_fields = ['timestamps', 'predictions']
                structure_valid = all(field in pred_data for field in required_fields)
                
                if structure_valid:
                    timestamps = pred_data.get('timestamps', [])
                    predictions = pred_data.get('predictions', [])
                    confidence_intervals = pred_data.get('confidence_intervals')
                    
                    # Validate data types and lengths
                    timestamps_valid = isinstance(timestamps, list) and len(timestamps) == 10
                    predictions_valid = isinstance(predictions, list) and len(predictions) == 10
                    
                    # Check timestamp format
                    timestamp_format_valid = True
                    if timestamps:
                        try:
                            datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
                        except:
                            timestamp_format_valid = False
                    
                    # Check prediction values are numeric
                    predictions_numeric = all(isinstance(p, (int, float)) for p in predictions)
                    
                    # For Prophet, confidence intervals should exist
                    confidence_valid = True
                    if model_type == 'prophet':
                        confidence_valid = confidence_intervals is not None
                        if confidence_intervals:
                            confidence_valid = all('lower' in ci and 'upper' in ci for ci in confidence_intervals[:3])
                    
                    test_results = [
                        timestamps_valid,
                        predictions_valid,
                        timestamp_format_valid,
                        predictions_numeric,
                        confidence_valid
                    ]
                    
                    test_passed = all(test_results)
                    structure_tests.append(test_passed)
                    
                    print(f"   ✅ {test_name}: {test_passed}")
                    print(f"      Timestamps valid: {timestamps_valid}")
                    print(f"      Predictions valid: {predictions_valid}")
                    print(f"      Timestamp format valid: {timestamp_format_valid}")
                    print(f"      Predictions numeric: {predictions_numeric}")
                    print(f"      Confidence intervals valid: {confidence_valid}")
                    
                else:
                    print(f"   ❌ {test_name}: Missing required fields")
                    structure_tests.append(False)
            
            # Test continuous prediction structure
            print(f"\n   Testing continuous prediction response structure...")
            if models_to_test:  # Use last trained model
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 5, "time_window": 20}
                )
                
                if response.status_code == 200:
                    cont_data = response.json()
                    cont_structure_valid = all(field in cont_data for field in ['timestamps', 'predictions'])
                    structure_tests.append(cont_structure_valid)
                    print(f"   ✅ Continuous prediction structure: {cont_structure_valid}")
                else:
                    structure_tests.append(False)
                    print(f"   ❌ Continuous prediction structure test failed")
            
            # Overall structure test result
            passed_structure_tests = sum(structure_tests)
            total_structure_tests = len(structure_tests)
            
            print(f"\n✅ API structure tests passed: {passed_structure_tests}/{total_structure_tests}")
            
            self.test_results['api_response_structure'] = passed_structure_tests >= total_structure_tests * 0.8
            
        except Exception as e:
            print(f"❌ API response structure test error: {str(e)}")
            self.test_results['api_response_structure'] = False
    
    def run_improved_prediction_tests(self):
        """Run all improved prediction system tests"""
        print("🚀 Starting Improved Prediction System Testing")
        print("=" * 70)
        
        # Run all improved prediction tests
        self.test_core_functionality_with_improved_algorithms()
        self.test_pattern_recognition_with_seasonal_data()
        self.test_continuous_prediction_extrapolation()
        self.test_prediction_accuracy_and_realism()
        self.test_api_response_structures()
        
        # Print final results
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary for improved predictions"""
        print("\n" + "=" * 70)
        print("📊 IMPROVED PREDICTION SYSTEM TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        test_descriptions = {
            'core_functionality': 'Core Functionality with Improved Algorithms',
            'pattern_recognition': 'Pattern Recognition with Historical Data',
            'continuous_extrapolation': 'Enhanced Continuous Prediction Extrapolation',
            'prediction_accuracy': 'Prediction Accuracy and Realism',
            'api_response_structure': 'API Response Data Structures'
        }
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            description = test_descriptions.get(test_name, test_name.replace('_', ' ').title())
            print(f"  {status} - {description}")
        
        print("\n" + "=" * 70)
        
        # Return overall success
        return passed_tests >= total_tests * 0.8  # 80% pass rate for overall success

if __name__ == "__main__":
    tester = ImprovedPredictionTester()
    overall_success = tester.run_improved_prediction_tests()
    
    if overall_success:
        print("🎉 Improved prediction system testing completed successfully!")
        exit(0)
    else:
        print("⚠️  Improved prediction system testing completed with some failures.")
        exit(1)