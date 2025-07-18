#!/usr/bin/env python3
"""
Focused Backend Testing for New Features
Tests the specific new features mentioned in the review request
"""

import requests
import json
import pandas as pd
import io
import time
import numpy as np
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://4f76f575-d5bb-4a63-b0d6-32438c43963e.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing backend at: {API_BASE_URL}")

class FocusedBackendTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        self.test_results = {}
        
    def create_sample_data(self):
        """Create realistic time-series sample data for testing"""
        # Generate 100 days of daily sales data with trend and seasonality
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create realistic sales data with trend and weekly seasonality
        trend = np.linspace(1000, 1500, 100)
        seasonal = 200 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly pattern
        noise = np.random.normal(0, 50, 100)
        sales = trend + seasonal + noise
        
        df = pd.DataFrame({
            'date': dates,
            'sales': sales,
            'region': ['North'] * 50 + ['South'] * 50
        })
        
        return df
    
    def setup_model(self):
        """Setup data and train ARIMA model for testing"""
        print("\n=== Setting up ARIMA Model for Testing ===")
        
        try:
            # Upload data
            df = self.create_sample_data()
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('sales_data.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                print(f"✅ Data uploaded successfully: {self.data_id}")
                
                # Train ARIMA model
                training_data = {
                    "data_id": self.data_id,
                    "model_type": "arima",
                    "parameters": {
                        "time_column": "date",
                        "target_column": "sales",
                        "order": [1, 1, 1]
                    }
                }
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": self.data_id, "model_type": "arima"},
                    json=training_data["parameters"]
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.model_id = data.get('model_id')
                    print(f"✅ ARIMA model trained successfully: {self.model_id}")
                    return True
                else:
                    print(f"❌ ARIMA model training failed: {response.status_code} - {response.text}")
                    return False
            else:
                print(f"❌ Data upload failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Setup error: {str(e)}")
            return False
    
    def test_ph_simulation_comprehensive(self):
        """Test 1: Comprehensive pH Simulation Testing"""
        print("\n=== Testing pH Simulation Endpoints (Comprehensive) ===")
        
        try:
            # Test real-time pH simulation multiple times
            ph_values = []
            for i in range(5):
                response = self.session.get(f"{API_BASE_URL}/ph-simulation")
                
                if response.status_code == 200:
                    data = response.json()
                    ph_value = data.get('ph_value')
                    confidence = data.get('confidence')
                    timestamp = data.get('timestamp')
                    
                    ph_values.append(ph_value)
                    print(f"   Reading {i+1}: pH={ph_value}, Confidence={confidence:.1f}%")
                    
                    # Validate structure
                    if not all([ph_value, confidence, timestamp]):
                        print(f"❌ Missing data in pH reading {i+1}")
                        self.test_results['ph_simulation_comprehensive'] = False
                        return
                        
                    # Validate pH range
                    if not (6.0 <= ph_value <= 8.0):
                        print(f"❌ pH value {ph_value} outside realistic range (6.0-8.0)")
                        self.test_results['ph_simulation_comprehensive'] = False
                        return
                        
                    # Validate confidence range
                    if not (80 <= confidence <= 100):
                        print(f"❌ Confidence {confidence} outside expected range (80-100)")
                        self.test_results['ph_simulation_comprehensive'] = False
                        return
                        
                else:
                    print(f"❌ pH simulation failed on reading {i+1}: {response.status_code}")
                    self.test_results['ph_simulation_comprehensive'] = False
                    return
                    
                time.sleep(0.5)  # Small delay between readings
            
            # Test pH simulation history with different time windows
            for hours in [1, 12, 24]:
                response = self.session.get(f"{API_BASE_URL}/ph-simulation-history", params={"hours": hours})
                
                if response.status_code == 200:
                    data = response.json()
                    history_data = data.get('data', [])
                    current_ph = data.get('current_ph')
                    target_ph = data.get('target_ph')
                    status = data.get('status')
                    
                    expected_points = hours * 60  # One reading per minute
                    actual_points = len(history_data)
                    
                    print(f"   History ({hours}h): {actual_points} points, Current pH: {current_ph}")
                    
                    # Validate data structure
                    if not history_data:
                        print(f"❌ No history data for {hours} hours")
                        self.test_results['ph_simulation_comprehensive'] = False
                        return
                    
                    # Check sample data points
                    sample_points = history_data[:10]  # Check first 10 points
                    for point in sample_points:
                        if not all(key in point for key in ['timestamp', 'ph_value', 'confidence']):
                            print(f"❌ Invalid data structure in history point")
                            self.test_results['ph_simulation_comprehensive'] = False
                            return
                        
                        if not (6.0 <= point['ph_value'] <= 8.0):
                            print(f"❌ Historical pH value {point['ph_value']} outside range")
                            self.test_results['ph_simulation_comprehensive'] = False
                            return
                else:
                    print(f"❌ pH history failed for {hours} hours: {response.status_code}")
                    self.test_results['ph_simulation_comprehensive'] = False
                    return
            
            # Validate pH values show realistic variation
            ph_range = max(ph_values) - min(ph_values)
            if ph_range > 0.01:  # Should have some variation
                print(f"✅ pH values show realistic variation (range: {ph_range:.3f})")
            else:
                print(f"⚠️  pH values show minimal variation (range: {ph_range:.3f})")
            
            print("✅ pH simulation comprehensive testing passed")
            self.test_results['ph_simulation_comprehensive'] = True
            
        except Exception as e:
            print(f"❌ pH simulation comprehensive test error: {str(e)}")
            self.test_results['ph_simulation_comprehensive'] = False
    
    def test_enhanced_continuous_prediction_detailed(self):
        """Test 2: Enhanced Continuous Prediction with Detailed Extrapolation Testing"""
        print("\n=== Testing Enhanced Continuous Prediction (Detailed) ===")
        
        if not self.model_id:
            print("❌ Cannot test enhanced continuous prediction - no model trained")
            self.test_results['enhanced_continuous_prediction_detailed'] = False
            return
            
        try:
            # Reset continuous prediction state first
            reset_response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            if reset_response.status_code != 200:
                print("⚠️  Reset failed, continuing anyway")
            
            # Test multiple calls to verify extrapolation
            prediction_calls = []
            
            for i in range(5):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 10, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    timestamps = data.get('timestamps', [])
                    
                    prediction_calls.append({
                        'call': i + 1,
                        'predictions': predictions,
                        'timestamps': timestamps,
                        'first_timestamp': timestamps[0] if timestamps else None,
                        'last_timestamp': timestamps[-1] if timestamps else None
                    })
                    
                    print(f"   Call {i+1}: {len(predictions)} predictions, First: {timestamps[0] if timestamps else 'None'}")
                    
                    # Validate structure
                    if len(predictions) != 10 or len(timestamps) != 10:
                        print(f"❌ Call {i+1}: Incorrect data structure")
                        self.test_results['enhanced_continuous_prediction_detailed'] = False
                        return
                        
                else:
                    print(f"❌ Call {i+1} failed: {response.status_code} - {response.text}")
                    self.test_results['enhanced_continuous_prediction_detailed'] = False
                    return
                
                time.sleep(0.5)  # Small delay between calls
            
            # Analyze extrapolation behavior
            print("\n   Analyzing extrapolation behavior:")
            
            # Check if timestamps are advancing (extrapolating forward)
            extrapolation_working = True
            for i in range(1, len(prediction_calls)):
                prev_call = prediction_calls[i-1]
                curr_call = prediction_calls[i]
                
                if prev_call['first_timestamp'] == curr_call['first_timestamp']:
                    print(f"   ⚠️  Calls {i} and {i+1} have same starting timestamp")
                    extrapolation_working = False
                else:
                    print(f"   ✅ Call {i+1} advanced from previous call")
            
            # Test different time windows
            print("\n   Testing different time windows:")
            for window in [20, 100, 200]:
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 15, "time_window": window}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    print(f"   Window {window}: {len(predictions)} predictions generated")
                else:
                    print(f"   ❌ Window {window} failed: {response.status_code}")
                    extrapolation_working = False
            
            if extrapolation_working:
                print("✅ Enhanced continuous prediction extrapolation working correctly")
                self.test_results['enhanced_continuous_prediction_detailed'] = True
            else:
                print("❌ Enhanced continuous prediction extrapolation issues detected")
                self.test_results['enhanced_continuous_prediction_detailed'] = False
                
        except Exception as e:
            print(f"❌ Enhanced continuous prediction detailed test error: {str(e)}")
            self.test_results['enhanced_continuous_prediction_detailed'] = False
    
    def test_complete_continuous_flow_integration(self):
        """Test 3: Complete Continuous Prediction Flow Integration"""
        print("\n=== Testing Complete Continuous Prediction Flow Integration ===")
        
        if not self.model_id:
            print("❌ Cannot test continuous prediction flow - no model trained")
            self.test_results['complete_continuous_flow'] = False
            return
            
        try:
            flow_steps = []
            
            # Step 1: Reset continuous prediction state
            print("   Step 1: Resetting continuous prediction state...")
            response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            flow_steps.append(("Reset", response.status_code == 200))
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Reset successful: {data.get('message')}")
            else:
                print(f"   ❌ Reset failed: {response.status_code}")
            
            # Step 2: Start continuous prediction
            print("   Step 2: Starting continuous prediction...")
            response = self.session.post(f"{API_BASE_URL}/start-continuous-prediction")
            flow_steps.append(("Start", response.status_code == 200))
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Start successful: {data.get('message')}")
            else:
                print(f"   ❌ Start failed: {response.status_code}")
                self.test_results['complete_continuous_flow'] = False
                return
            
            # Step 3: Test continuous prediction generation over time
            print("   Step 3: Testing continuous prediction generation...")
            time.sleep(2)  # Let the background task start
            
            prediction_results = []
            for i in range(3):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 10, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    prediction_results.append({
                        'iteration': i + 1,
                        'timestamps': data.get('timestamps', []),
                        'predictions': data.get('predictions', [])
                    })
                    print(f"   ✅ Prediction {i+1}: Generated {len(data.get('predictions', []))} points")
                else:
                    print(f"   ❌ Prediction {i+1} failed: {response.status_code}")
                    flow_steps.append((f"Prediction_{i+1}", False))
                    break
                
                time.sleep(1)  # Wait between predictions
            
            # Verify predictions are extrapolating
            if len(prediction_results) >= 2:
                first_timestamps = prediction_results[0]['timestamps']
                second_timestamps = prediction_results[1]['timestamps']
                
                if first_timestamps != second_timestamps:
                    print("   ✅ Predictions are properly extrapolating forward")
                    flow_steps.append(("Extrapolation", True))
                else:
                    print("   ❌ Predictions are not extrapolating (same timestamps)")
                    flow_steps.append(("Extrapolation", False))
            
            # Step 4: Test pH simulation integration
            print("   Step 4: Testing pH simulation integration...")
            ph_response = self.session.get(f"{API_BASE_URL}/ph-simulation")
            
            if ph_response.status_code == 200:
                ph_data = ph_response.json()
                print(f"   ✅ pH simulation: {ph_data.get('ph_value')} (confidence: {ph_data.get('confidence'):.1f}%)")
                flow_steps.append(("pH_Integration", True))
            else:
                print(f"   ❌ pH simulation failed: {ph_response.status_code}")
                flow_steps.append(("pH_Integration", False))
            
            # Step 5: Stop continuous prediction
            print("   Step 5: Stopping continuous prediction...")
            response = self.session.post(f"{API_BASE_URL}/stop-continuous-prediction")
            flow_steps.append(("Stop", response.status_code == 200))
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Stop successful: {data.get('message')}")
            else:
                print(f"   ❌ Stop failed: {response.status_code}")
            
            # Evaluate overall flow
            passed_steps = sum(1 for _, passed in flow_steps if passed)
            total_steps = len(flow_steps)
            
            print(f"\n   Flow test results: {passed_steps}/{total_steps} steps passed")
            for step_name, passed in flow_steps:
                status = "✅" if passed else "❌"
                print(f"   {status} {step_name}")
            
            # Require 80% success rate
            if passed_steps >= total_steps * 0.8:
                print("✅ Complete continuous prediction flow integration passed")
                self.test_results['complete_continuous_flow'] = True
            else:
                print("❌ Complete continuous prediction flow integration failed")
                self.test_results['complete_continuous_flow'] = False
                
        except Exception as e:
            print(f"❌ Complete continuous flow integration test error: {str(e)}")
            self.test_results['complete_continuous_flow'] = False
    
    def run_focused_tests(self):
        """Run focused tests on new features"""
        print("🚀 Starting Focused Backend Testing for New Features")
        print("=" * 70)
        
        # Setup model first
        if not self.setup_model():
            print("❌ Failed to setup model, some tests will be skipped")
        
        # Run focused tests
        self.test_ph_simulation_comprehensive()
        self.test_enhanced_continuous_prediction_detailed()
        self.test_complete_continuous_flow_integration()
        
        # Print results
        self.print_focused_summary()
    
    def print_focused_summary(self):
        """Print focused test summary"""
        print("\n" + "=" * 70)
        print("📊 FOCUSED BACKEND TEST SUMMARY - NEW FEATURES")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"Total New Feature Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} - {test_name.replace('_', ' ').title()}")
        
        print("\n" + "=" * 70)
        
        return passed_tests >= total_tests * 0.8 if total_tests > 0 else False

if __name__ == "__main__":
    tester = FocusedBackendTester()
    overall_success = tester.run_focused_tests()
    
    if overall_success:
        print("🎉 Focused backend testing completed successfully!")
        exit(0)
    else:
        print("⚠️  Focused backend testing completed with some failures.")
        exit(1)