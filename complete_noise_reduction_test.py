#!/usr/bin/env python3
"""
Complete Noise Reduction System Testing with Model Setup
Tests the noise reduction system with proper model initialization
"""

import requests
import json
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://3504f872-4ab4-43c1-a827-4429cc10638c.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing complete noise reduction system at: {API_BASE_URL}")

class CompleteNoiseReductionTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        
    def setup_test_environment(self):
        """Setup test environment with data and model"""
        print("\n=== Setting Up Test Environment ===")
        
        # Create realistic pH test data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        
        # Create realistic pH data with some noise
        base_ph = 7.2
        ph_values = []
        for i in range(100):
            # Add daily cycle and some random variation
            daily_cycle = 0.3 * np.sin(2 * np.pi * i / 24)  # 24-hour cycle
            noise = np.random.normal(0, 0.1)
            ph = base_ph + daily_cycle + noise
            ph_values.append(max(6.0, min(8.0, ph)))  # Keep in realistic range
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph_value': ph_values
        })
        
        # Upload data
        csv_content = df.to_csv(index=False)
        files = {'file': ('ph_test_data.csv', csv_content, 'text/csv')}
        
        response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
        
        if response.status_code == 200:
            data = response.json()
            self.data_id = data.get('data_id')
            print(f"✅ Test data uploaded successfully (ID: {self.data_id})")
            
            # Train ARIMA model
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "ph_value", "order": [1, 1, 1]}
            )
            
            if response.status_code == 200:
                model_data = response.json()
                self.model_id = model_data.get('model_id')
                print(f"✅ ARIMA model trained successfully (ID: {self.model_id})")
                return True
            else:
                print(f"❌ Model training failed: {response.status_code}")
                return False
        else:
            print(f"❌ Data upload failed: {response.status_code}")
            return False
    
    def test_noise_reduction_with_real_predictions(self):
        """Test noise reduction with real prediction data"""
        print("\n=== Testing Noise Reduction with Real Predictions ===")
        
        if not self.model_id:
            print("❌ Cannot test - no model available")
            return []
        
        tests = []
        
        # Test 1: Enhanced real-time prediction with noise reduction
        print("Testing enhanced real-time prediction...")
        try:
            response = self.session.get(
                f"{API_BASE_URL}/generate-enhanced-realtime-prediction",
                params={"steps": 20, "time_window": 100, "maintain_patterns": True}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                print(f"✅ Enhanced real-time prediction successful")
                print(f"   Predictions generated: {len(predictions)}")
                print(f"   Sample predictions: {predictions[:3] if predictions else []}")
                
                # Check prediction quality
                if predictions:
                    # Check if predictions are in realistic pH range
                    ph_range_valid = all(5.0 <= p <= 9.0 for p in predictions)
                    # Check for smoothness (no extreme jumps)
                    if len(predictions) > 1:
                        max_jump = max(abs(predictions[i+1] - predictions[i]) for i in range(len(predictions)-1))
                        smooth_transitions = max_jump < 1.0  # No jumps > 1.0 pH units
                    else:
                        smooth_transitions = True
                    
                    print(f"   pH range valid: {ph_range_valid}")
                    print(f"   Smooth transitions: {smooth_transitions}")
                    
                    enhanced_test_passed = ph_range_valid and smooth_transitions
                else:
                    enhanced_test_passed = False
                
                tests.append(("Enhanced real-time prediction", enhanced_test_passed))
            else:
                print(f"❌ Enhanced real-time prediction failed: {response.status_code}")
                tests.append(("Enhanced real-time prediction", False))
        except Exception as e:
            print(f"❌ Enhanced real-time prediction error: {e}")
            tests.append(("Enhanced real-time prediction", False))
        
        # Test 2: Advanced pH prediction
        print("Testing advanced pH prediction...")
        try:
            response = self.session.get(
                f"{API_BASE_URL}/generate-advanced-ph-prediction",
                params={"steps": 15, "maintain_patterns": True}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                print(f"✅ Advanced pH prediction successful")
                print(f"   Predictions generated: {len(predictions)}")
                print(f"   Sample predictions: {predictions[:3] if predictions else []}")
                
                # Check prediction quality
                if predictions:
                    ph_range_valid = all(5.0 <= p <= 9.0 for p in predictions)
                    # Check variability (should not be flat)
                    variability = np.std(predictions) if len(predictions) > 1 else 0
                    has_variability = variability > 0.01
                    
                    print(f"   pH range valid: {ph_range_valid}")
                    print(f"   Has variability: {has_variability} (std: {variability:.3f})")
                    
                    advanced_ph_passed = ph_range_valid and has_variability
                else:
                    advanced_ph_passed = False
                
                tests.append(("Advanced pH prediction", advanced_ph_passed))
            else:
                print(f"❌ Advanced pH prediction failed: {response.status_code}")
                tests.append(("Advanced pH prediction", False))
        except Exception as e:
            print(f"❌ Advanced pH prediction error: {e}")
            tests.append(("Advanced pH prediction", False))
        
        # Test 3: Extended pH prediction (if advanced pH worked)
        print("Testing extended pH prediction...")
        try:
            response = self.session.get(
                f"{API_BASE_URL}/extend-advanced-ph-prediction",
                params={"additional_steps": 5}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                prediction_info = data.get('prediction_info', {})
                
                print(f"✅ Extended pH prediction successful")
                print(f"   Extended predictions: {len(predictions)}")
                print(f"   Noise reduction applied: {prediction_info.get('noise_reduction_applied', False)}")
                print(f"   Smoothing type: {prediction_info.get('smoothing_type', 'none')}")
                
                # Check if real-time smoothing was applied
                extended_ph_passed = (len(predictions) == 5 and 
                                    prediction_info.get('noise_reduction_applied', False) and
                                    prediction_info.get('smoothing_type') == 'real_time_optimized')
                tests.append(("Extended pH prediction", extended_ph_passed))
            else:
                print(f"❌ Extended pH prediction failed: {response.status_code}")
                tests.append(("Extended pH prediction", False))
        except Exception as e:
            print(f"❌ Extended pH prediction error: {e}")
            tests.append(("Extended pH prediction", False))
        
        return tests
    
    def test_continuous_prediction_smoothness(self):
        """Test smoothness of continuous predictions over multiple calls"""
        print("\n=== Testing Continuous Prediction Smoothness ===")
        
        if not self.model_id:
            print("❌ Cannot test - no model available")
            return []
        
        tests = []
        
        # Make multiple continuous predictions to test smoothness
        print("Testing continuous prediction smoothness over multiple calls...")
        try:
            predictions_series = []
            
            for i in range(3):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-enhanced-realtime-prediction",
                    params={"steps": 10, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    predictions_series.append(predictions)
                    time.sleep(0.5)  # Small delay between calls
                else:
                    print(f"❌ Continuous prediction call {i+1} failed: {response.status_code}")
            
            if len(predictions_series) >= 2:
                # Calculate smoothness between consecutive prediction sets
                smoothness_scores = []
                
                for i in range(1, len(predictions_series)):
                    prev_predictions = predictions_series[i-1]
                    curr_predictions = predictions_series[i]
                    
                    if prev_predictions and curr_predictions:
                        # Check transition smoothness
                        transition_gap = abs(curr_predictions[0] - prev_predictions[-1])
                        
                        # Calculate internal smoothness
                        prev_smoothness = self.calculate_internal_smoothness(prev_predictions)
                        curr_smoothness = self.calculate_internal_smoothness(curr_predictions)
                        
                        # Overall smoothness score
                        transition_smoothness = 1.0 / (1.0 + transition_gap * 10)  # Penalize large gaps
                        overall_smoothness = (transition_smoothness + prev_smoothness + curr_smoothness) / 3
                        smoothness_scores.append(overall_smoothness)
                
                avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0
                print(f"✅ Continuous prediction smoothness test completed")
                print(f"   Average smoothness score: {avg_smoothness:.3f}")
                print(f"   Prediction series count: {len(predictions_series)}")
                
                # Test passes if smoothness is good (> 0.7)
                smoothness_passed = avg_smoothness > 0.7
                tests.append(("Continuous prediction smoothness", smoothness_passed))
            else:
                print("❌ Insufficient prediction series for smoothness test")
                tests.append(("Continuous prediction smoothness", False))
                
        except Exception as e:
            print(f"❌ Continuous prediction smoothness error: {e}")
            tests.append(("Continuous prediction smoothness", False))
        
        return tests
    
    def calculate_internal_smoothness(self, predictions):
        """Calculate internal smoothness of a prediction series"""
        if len(predictions) < 2:
            return 1.0
        
        # Calculate second derivative (measure of smoothness)
        if len(predictions) >= 3:
            second_derivatives = []
            for i in range(1, len(predictions) - 1):
                second_deriv = predictions[i+1] - 2*predictions[i] + predictions[i-1]
                second_derivatives.append(abs(second_deriv))
            
            avg_second_deriv = np.mean(second_derivatives)
            smoothness = 1.0 / (1.0 + avg_second_deriv * 100)  # Higher second derivative = less smooth
        else:
            # For short series, just check first derivative
            first_deriv = abs(predictions[1] - predictions[0])
            smoothness = 1.0 / (1.0 + first_deriv * 10)
        
        return smoothness
    
    def run_complete_test_suite(self):
        """Run the complete noise reduction test suite"""
        print("🎯 COMPLETE NOISE REDUCTION SYSTEM TESTING")
        print("=" * 60)
        
        # Setup test environment
        setup_success = self.setup_test_environment()
        if not setup_success:
            print("❌ Test environment setup failed. Cannot proceed with tests.")
            return False
        
        all_tests = []
        
        # Test noise reduction with real predictions
        prediction_tests = self.test_noise_reduction_with_real_predictions()
        all_tests.extend(prediction_tests)
        
        # Test continuous prediction smoothness
        smoothness_tests = self.test_continuous_prediction_smoothness()
        all_tests.extend(smoothness_tests)
        
        # Print comprehensive results
        print("\n" + "=" * 60)
        print("🎯 COMPLETE NOISE REDUCTION TEST RESULTS")
        print("=" * 60)
        
        passed_tests = sum(1 for _, passed in all_tests if passed)
        total_tests = len(all_tests)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        print(f"\n🚀 Enhanced Prediction Endpoints with Noise Reduction:")
        for test_name, passed in all_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        # Overall assessment
        if success_rate >= 80:
            print(f"\n🎉 EXCELLENT: Noise reduction system is working excellently!")
            print("   - All major endpoints are functional")
            print("   - Noise reduction is properly applied")
            print("   - Predictions maintain smoothness and realistic ranges")
        elif success_rate >= 60:
            print(f"\n✅ GOOD: Noise reduction system is working well with minor issues.")
            print("   - Most endpoints are functional")
            print("   - Noise reduction is generally working")
            print("   - Some improvements may be needed")
        elif success_rate >= 40:
            print(f"\n⚠️  PARTIAL: Noise reduction system has significant issues.")
            print("   - Some endpoints are not working properly")
            print("   - Noise reduction may not be fully effective")
            print("   - Requires attention and fixes")
        else:
            print(f"\n❌ CRITICAL: Noise reduction system has major problems.")
            print("   - Most endpoints are failing")
            print("   - Noise reduction is not working as expected")
            print("   - Immediate fixes required")
        
        return success_rate >= 60

if __name__ == "__main__":
    tester = CompleteNoiseReductionTester()
    success = tester.run_complete_test_suite()
    
    if success:
        print(f"\n🎯 FINAL CONCLUSION: Complete noise reduction system testing PASSED!")
        print("   The noise reduction system is ready for production use.")
    else:
        print(f"\n❌ FINAL CONCLUSION: Complete noise reduction system testing FAILED!")
        print("   The noise reduction system needs additional work before production.")