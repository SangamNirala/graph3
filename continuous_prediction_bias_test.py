#!/usr/bin/env python3
"""
Focused Testing for Continuous Prediction Bias Correction
Tests specifically for the downward bias issue that was identified and fixed
"""

import requests
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://ee04ac22-cb45-4b61-832c-93de71320985.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing continuous prediction bias correction at: {API_BASE_URL}")

class ContinuousPredictionBiasTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        self.test_results = {}
        
    def create_realistic_ph_data(self, num_points=50):
        """Create realistic pH data for testing bias correction"""
        # Generate realistic pH data with natural variations
        dates = pd.date_range(start='2024-01-01', periods=num_points, freq='H')
        
        # Create realistic pH values (6.0-8.0 range) with natural patterns
        base_ph = 7.2
        trend = np.linspace(0, 0.3, num_points)  # Slight upward trend
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(num_points) / 24)  # Daily cycle
        noise = np.random.normal(0, 0.1, num_points)
        ph_values = base_ph + trend + seasonal + noise
        
        # Ensure pH values stay within realistic bounds
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'pH': ph_values
        })
        
        return df
    
    def upload_ph_data(self):
        """Upload pH data for testing"""
        print("\n=== Uploading pH Test Data ===")
        
        try:
            # Create realistic pH data
            df = self.create_realistic_ph_data()
            csv_content = df.to_csv(index=False)
            
            # Prepare file for upload
            files = {
                'file': ('ph_data.csv', csv_content, 'text/csv')
            }
            
            # Upload data
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                
                print("✅ pH data upload successful")
                print(f"   Data ID: {self.data_id}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   pH range: {df['pH'].min():.3f} - {df['pH'].max():.3f}")
                print(f"   pH mean: {df['pH'].mean():.3f}")
                print(f"   pH std: {df['pH'].std():.3f}")
                
                self.test_results['ph_data_upload'] = True
                return True
                
            else:
                print(f"❌ pH data upload failed: {response.status_code} - {response.text}")
                self.test_results['ph_data_upload'] = False
                return False
                
        except Exception as e:
            print(f"❌ pH data upload error: {str(e)}")
            self.test_results['ph_data_upload'] = False
            return False
    
    def train_model_for_bias_testing(self):
        """Train a model specifically for bias testing"""
        print("\n=== Training Model for Bias Testing ===")
        
        if not self.data_id:
            print("❌ Cannot train model - no data uploaded")
            self.test_results['model_training'] = False
            return False
            
        try:
            # Train ARIMA model (known to work well)
            training_params = {
                "time_column": "timestamp",
                "target_column": "pH",
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
                
                print("✅ Model training successful")
                print(f"   Model ID: {self.model_id}")
                print(f"   Model type: ARIMA")
                
                self.test_results['model_training'] = True
                return True
                
            else:
                print(f"❌ Model training failed: {response.status_code} - {response.text}")
                self.test_results['model_training'] = False
                return False
                
        except Exception as e:
            print(f"❌ Model training error: {str(e)}")
            self.test_results['model_training'] = False
            return False
    
    def test_single_prediction_bias(self):
        """Test 1: Single prediction to check for immediate bias"""
        print("\n=== Test 1: Single Prediction Bias Check ===")
        
        if not self.model_id:
            print("❌ Cannot test prediction - no model trained")
            self.test_results['single_prediction_bias'] = False
            return False
            
        try:
            # Generate single prediction
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": self.model_id, "steps": 10}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if len(predictions) >= 10:
                    # Analyze bias in single prediction
                    pred_values = [p['value'] for p in predictions]
                    
                    # Calculate trend slope
                    x = np.arange(len(pred_values))
                    slope = np.polyfit(x, pred_values, 1)[0]
                    
                    # Check pH range
                    min_ph = min(pred_values)
                    max_ph = max(pred_values)
                    mean_ph = np.mean(pred_values)
                    std_ph = np.std(pred_values)
                    
                    print(f"✅ Single prediction generated successfully")
                    print(f"   Predictions: {len(predictions)}")
                    print(f"   pH range: {min_ph:.3f} - {max_ph:.3f}")
                    print(f"   pH mean: {mean_ph:.3f}")
                    print(f"   pH std: {std_ph:.3f}")
                    print(f"   Trend slope: {slope:.6f}")
                    
                    # Check for downward bias (slope should not be strongly negative)
                    if slope > -0.01:  # Acceptable small negative slope
                        print("✅ No significant downward bias detected")
                        bias_result = True
                    else:
                        print(f"❌ Downward bias detected (slope: {slope:.6f})")
                        bias_result = False
                    
                    # Check pH range is realistic
                    if 6.0 <= min_ph and max_ph <= 8.0:
                        print("✅ pH predictions within realistic range")
                        range_result = True
                    else:
                        print(f"❌ pH predictions outside realistic range")
                        range_result = False
                    
                    self.test_results['single_prediction_bias'] = bias_result and range_result
                    return bias_result and range_result
                    
                else:
                    print(f"❌ Insufficient predictions generated: {len(predictions)}")
                    self.test_results['single_prediction_bias'] = False
                    return False
                    
            else:
                print(f"❌ Single prediction failed: {response.status_code} - {response.text}")
                self.test_results['single_prediction_bias'] = False
                return False
                
        except Exception as e:
            print(f"❌ Single prediction test error: {str(e)}")
            self.test_results['single_prediction_bias'] = False
            return False
    
    def test_continuous_prediction_bias(self):
        """Test 2: Multiple sequential continuous predictions to check for accumulated bias"""
        print("\n=== Test 2: Continuous Prediction Bias Check ===")
        
        if not self.model_id:
            print("❌ Cannot test continuous prediction - no model trained")
            self.test_results['continuous_prediction_bias'] = False
            return False
            
        try:
            # Reset continuous predictions first
            reset_response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            if reset_response.status_code != 200:
                print(f"⚠️ Reset warning: {reset_response.status_code}")
            
            # Start continuous prediction
            start_response = self.session.post(f"{API_BASE_URL}/start-continuous-prediction")
            
            if start_response.status_code != 200:
                print(f"❌ Failed to start continuous prediction: {start_response.status_code}")
                self.test_results['continuous_prediction_bias'] = False
                return False
            
            print("✅ Continuous prediction started")
            
            # Generate multiple sequential predictions
            all_predictions = []
            num_calls = 5  # Test with 5 sequential calls
            
            for i in range(num_calls):
                print(f"   Generating prediction batch {i+1}/{num_calls}...")
                
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"steps": 5}  # 5 steps per call
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    
                    if predictions:
                        pred_values = [p['value'] for p in predictions]
                        all_predictions.extend(pred_values)
                        print(f"     Generated {len(predictions)} predictions, pH range: {min(pred_values):.3f} - {max(pred_values):.3f}")
                    else:
                        print(f"     No predictions in response")
                        
                    time.sleep(0.5)  # Small delay between calls
                    
                else:
                    print(f"❌ Continuous prediction call {i+1} failed: {response.status_code}")
                    break
            
            # Stop continuous prediction
            stop_response = self.session.post(f"{API_BASE_URL}/stop-continuous-prediction")
            
            # Analyze accumulated bias
            if len(all_predictions) >= 15:  # Should have 5*5 = 25 predictions
                # Calculate overall trend
                x = np.arange(len(all_predictions))
                slope = np.polyfit(x, all_predictions, 1)[0]
                
                # Calculate statistics
                min_ph = min(all_predictions)
                max_ph = max(all_predictions)
                mean_ph = np.mean(all_predictions)
                std_ph = np.std(all_predictions)
                
                print(f"✅ Continuous prediction analysis:")
                print(f"   Total predictions: {len(all_predictions)}")
                print(f"   pH range: {min_ph:.3f} - {max_ph:.3f}")
                print(f"   pH mean: {mean_ph:.3f}")
                print(f"   pH std: {std_ph:.3f}")
                print(f"   Overall trend slope: {slope:.6f}")
                
                # Check for accumulated downward bias
                # The key issue was slope=-0.230727, so we want to see significant improvement
                if slope > -0.05:  # Much more lenient than the previous -0.230727
                    print("✅ No significant accumulated downward bias detected")
                    bias_result = True
                else:
                    print(f"❌ Accumulated downward bias detected (slope: {slope:.6f})")
                    bias_result = False
                
                # Check pH range is realistic
                if 6.0 <= min_ph and max_ph <= 8.0:
                    print("✅ All pH predictions within realistic range")
                    range_result = True
                else:
                    print(f"❌ Some pH predictions outside realistic range")
                    range_result = False
                
                # Check for variability (not flat predictions)
                if std_ph > 0.01:  # Should have some variation
                    print("✅ Predictions show realistic variability")
                    variability_result = True
                else:
                    print(f"❌ Predictions too flat (std: {std_ph:.6f})")
                    variability_result = False
                
                overall_result = bias_result and range_result and variability_result
                self.test_results['continuous_prediction_bias'] = overall_result
                
                # Store detailed results for comparison
                self.test_results['continuous_bias_details'] = {
                    'slope': slope,
                    'min_ph': min_ph,
                    'max_ph': max_ph,
                    'mean_ph': mean_ph,
                    'std_ph': std_ph,
                    'num_predictions': len(all_predictions)
                }
                
                return overall_result
                
            else:
                print(f"❌ Insufficient continuous predictions: {len(all_predictions)}")
                self.test_results['continuous_prediction_bias'] = False
                return False
                
        except Exception as e:
            print(f"❌ Continuous prediction test error: {str(e)}")
            self.test_results['continuous_prediction_bias'] = False
            return False
    
    def test_extend_prediction_endpoint(self):
        """Test 3: Test the /extend-prediction endpoint specifically"""
        print("\n=== Test 3: Extend Prediction Endpoint ===")
        
        if not self.model_id:
            print("❌ Cannot test extend prediction - no model trained")
            self.test_results['extend_prediction'] = False
            return False
            
        try:
            # Test the extend-prediction endpoint multiple times
            all_extensions = []
            num_extensions = 3
            
            for i in range(num_extensions):
                print(f"   Extension call {i+1}/{num_extensions}...")
                
                response = self.session.get(
                    f"{API_BASE_URL}/extend-prediction",
                    params={"steps": 10}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    
                    if predictions:
                        pred_values = [p['value'] for p in predictions]
                        all_extensions.extend(pred_values)
                        
                        # Calculate slope for this extension
                        x = np.arange(len(pred_values))
                        slope = np.polyfit(x, pred_values, 1)[0]
                        
                        print(f"     Extension {i+1}: {len(predictions)} predictions, slope: {slope:.6f}")
                        print(f"     pH range: {min(pred_values):.3f} - {max(pred_values):.3f}")
                    else:
                        print(f"     No predictions in extension {i+1}")
                        
                    time.sleep(0.5)
                    
                else:
                    print(f"❌ Extend prediction call {i+1} failed: {response.status_code}")
                    break
            
            # Analyze all extensions
            if len(all_extensions) >= 20:  # Should have 3*10 = 30 predictions
                # Calculate overall trend across all extensions
                x = np.arange(len(all_extensions))
                overall_slope = np.polyfit(x, all_extensions, 1)[0]
                
                # Calculate statistics
                min_ph = min(all_extensions)
                max_ph = max(all_extensions)
                mean_ph = np.mean(all_extensions)
                std_ph = np.std(all_extensions)
                
                print(f"✅ Extend prediction analysis:")
                print(f"   Total extensions: {len(all_extensions)}")
                print(f"   pH range: {min_ph:.3f} - {max_ph:.3f}")
                print(f"   pH mean: {mean_ph:.3f}")
                print(f"   pH std: {std_ph:.3f}")
                print(f"   Overall slope: {overall_slope:.6f}")
                
                # Check for bias in extensions
                if overall_slope > -0.02:  # Should not have strong downward bias
                    print("✅ No significant bias in extend predictions")
                    bias_result = True
                else:
                    print(f"❌ Downward bias in extend predictions (slope: {overall_slope:.6f})")
                    bias_result = False
                
                # Check pH range
                if 6.0 <= min_ph and max_ph <= 8.0:
                    print("✅ Extended predictions within realistic pH range")
                    range_result = True
                else:
                    print(f"❌ Extended predictions outside realistic pH range")
                    range_result = False
                
                overall_result = bias_result and range_result
                self.test_results['extend_prediction'] = overall_result
                
                return overall_result
                
            else:
                print(f"❌ Insufficient extend predictions: {len(all_extensions)}")
                self.test_results['extend_prediction'] = False
                return False
                
        except Exception as e:
            print(f"❌ Extend prediction test error: {str(e)}")
            self.test_results['extend_prediction'] = False
            return False
    
    def test_bias_correction_algorithms(self):
        """Test 4: Test specific bias correction features"""
        print("\n=== Test 4: Bias Correction Algorithms ===")
        
        try:
            # Test pH simulation endpoint (should have bias correction)
            response = self.session.get(f"{API_BASE_URL}/ph-simulation")
            
            if response.status_code == 200:
                data = response.json()
                ph_value = data.get('ph_value')
                confidence = data.get('confidence')
                
                if ph_value and 6.0 <= ph_value <= 8.0:
                    print(f"✅ pH simulation working: pH={ph_value:.3f}, confidence={confidence}")
                    simulation_result = True
                else:
                    print(f"❌ pH simulation out of range: pH={ph_value}")
                    simulation_result = False
            else:
                print(f"❌ pH simulation failed: {response.status_code}")
                simulation_result = False
            
            # Test pH simulation history (should show realistic patterns)
            history_response = self.session.get(f"{API_BASE_URL}/ph-simulation-history")
            
            if history_response.status_code == 200:
                history_data = history_response.json()
                history_points = history_data.get('data', [])
                
                if len(history_points) > 100:  # Should have substantial history
                    ph_values = [point['ph_value'] for point in history_points]
                    
                    # Analyze historical pH data
                    min_ph = min(ph_values)
                    max_ph = max(ph_values)
                    mean_ph = np.mean(ph_values)
                    std_ph = np.std(ph_values)
                    
                    print(f"✅ pH history analysis:")
                    print(f"   History points: {len(history_points)}")
                    print(f"   pH range: {min_ph:.3f} - {max_ph:.3f}")
                    print(f"   pH mean: {mean_ph:.3f}")
                    print(f"   pH std: {std_ph:.3f}")
                    
                    # Check if history shows realistic pH characteristics
                    if 6.0 <= min_ph and max_ph <= 8.0 and 0.1 <= std_ph <= 0.5:
                        print("✅ pH history shows realistic characteristics")
                        history_result = True
                    else:
                        print("❌ pH history characteristics unrealistic")
                        history_result = False
                else:
                    print(f"❌ Insufficient pH history: {len(history_points)}")
                    history_result = False
            else:
                print(f"❌ pH history failed: {history_response.status_code}")
                history_result = False
            
            overall_result = simulation_result and history_result
            self.test_results['bias_correction_algorithms'] = overall_result
            
            return overall_result
            
        except Exception as e:
            print(f"❌ Bias correction algorithms test error: {str(e)}")
            self.test_results['bias_correction_algorithms'] = False
            return False
    
    def test_pattern_following(self):
        """Test 5: Test that predictions follow historical patterns"""
        print("\n=== Test 5: Pattern Following Verification ===")
        
        if not self.model_id:
            print("❌ Cannot test pattern following - no model trained")
            self.test_results['pattern_following'] = False
            return False
            
        try:
            # Generate a longer prediction to test pattern following
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": self.model_id, "steps": 25}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if len(predictions) >= 20:
                    pred_values = [p['value'] for p in predictions]
                    
                    # Analyze pattern characteristics
                    # 1. Check for variability (not flat)
                    std_pred = np.std(pred_values)
                    
                    # 2. Check for sign changes (oscillations)
                    changes = np.diff(pred_values)
                    sign_changes = np.sum(np.diff(np.sign(changes)) != 0)
                    
                    # 3. Check mean reversion tendency
                    mean_pred = np.mean(pred_values)
                    deviations = np.abs(np.array(pred_values) - mean_pred)
                    max_deviation = np.max(deviations)
                    
                    print(f"✅ Pattern following analysis:")
                    print(f"   Prediction std: {std_pred:.6f}")
                    print(f"   Sign changes: {sign_changes}")
                    print(f"   Mean: {mean_pred:.3f}")
                    print(f"   Max deviation: {max_deviation:.3f}")
                    
                    # Evaluate pattern following
                    variability_ok = std_pred > 0.005  # Should have some variation
                    oscillation_ok = sign_changes >= 2  # Should have some oscillations
                    bounded_ok = max_deviation < 1.0  # Should not deviate too much
                    
                    if variability_ok and oscillation_ok and bounded_ok:
                        print("✅ Predictions show good pattern following")
                        pattern_result = True
                    else:
                        print("❌ Predictions show poor pattern following")
                        print(f"   Variability: {'✅' if variability_ok else '❌'}")
                        print(f"   Oscillations: {'✅' if oscillation_ok else '❌'}")
                        print(f"   Bounded: {'✅' if bounded_ok else '❌'}")
                        pattern_result = False
                    
                    self.test_results['pattern_following'] = pattern_result
                    return pattern_result
                    
                else:
                    print(f"❌ Insufficient predictions for pattern analysis: {len(predictions)}")
                    self.test_results['pattern_following'] = False
                    return False
                    
            else:
                print(f"❌ Pattern following test failed: {response.status_code}")
                self.test_results['pattern_following'] = False
                return False
                
        except Exception as e:
            print(f"❌ Pattern following test error: {str(e)}")
            self.test_results['pattern_following'] = False
            return False
    
    def run_all_tests(self):
        """Run all continuous prediction bias tests"""
        print("🔬 CONTINUOUS PREDICTION BIAS CORRECTION TESTING")
        print("=" * 60)
        
        # Test sequence
        tests = [
            ("Upload pH Data", self.upload_ph_data),
            ("Train Model", self.train_model_for_bias_testing),
            ("Single Prediction Bias", self.test_single_prediction_bias),
            ("Continuous Prediction Bias", self.test_continuous_prediction_bias),
            ("Extend Prediction Endpoint", self.test_extend_prediction_endpoint),
            ("Bias Correction Algorithms", self.test_bias_correction_algorithms),
            ("Pattern Following", self.test_pattern_following)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func()
                if result:
                    passed_tests += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {str(e)}")
        
        # Final summary
        print("\n" + "="*60)
        print("🎯 CONTINUOUS PREDICTION BIAS CORRECTION TEST SUMMARY")
        print("="*60)
        
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 EXCELLENT: Continuous prediction bias correction is working well!")
        elif success_rate >= 60:
            print("✅ GOOD: Most bias correction features are working")
        else:
            print("❌ NEEDS WORK: Significant bias correction issues remain")
        
        # Detailed results
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        # Specific bias analysis
        if 'continuous_bias_details' in self.test_results:
            details = self.test_results['continuous_bias_details']
            print(f"\n🔍 Bias Analysis Details:")
            print(f"  Previous issue: slope=-0.230727 (strong downward bias)")
            print(f"  Current result: slope={details['slope']:.6f}")
            
            if details['slope'] > -0.05:
                improvement = abs(-0.230727 - details['slope'])
                print(f"  🎉 MAJOR IMPROVEMENT: Bias reduced by {improvement:.6f}")
                print(f"  ✅ Downward bias issue RESOLVED!")
            else:
                print(f"  ⚠️ Some downward bias still present")
        
        return self.test_results

if __name__ == "__main__":
    tester = ContinuousPredictionBiasTester()
    results = tester.run_all_tests()