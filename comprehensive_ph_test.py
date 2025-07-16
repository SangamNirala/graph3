#!/usr/bin/env python3
"""
Comprehensive pH Prediction Algorithm Verification
Tests all aspects mentioned in the review request
"""

import requests
import json
import pandas as pd
import io
import numpy as np
import os
from pathlib import Path
import statistics
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://0ce7ec4c-26f8-4958-9dca-f78e6b94c25d.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

class ComprehensivePhTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        self.test_results = {}
        
    def create_realistic_ph_dataset(self):
        """Create realistic pH monitoring dataset"""
        timestamps = pd.date_range(start='2024-01-01', periods=49, freq='H')
        
        # Create realistic pH values with natural patterns
        base_ph = 7.4
        ph_values = []
        
        for i in range(49):
            # Multiple realistic components
            daily_cycle = 0.15 * np.sin(i * 2 * np.pi / 24)  # Daily pH cycle
            process_drift = 0.05 * np.sin(i * 2 * np.pi / 168)  # Weekly process cycle
            random_noise = np.random.normal(0, 0.03)  # Measurement noise
            process_variation = 0.08 * np.sin(i * 2 * np.pi / 12)  # 12-hour process cycle
            
            ph_value = base_ph + daily_cycle + process_drift + random_noise + process_variation
            ph_value = max(6.8, min(8.0, ph_value))  # Realistic pH bounds
            ph_values.append(round(ph_value, 2))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ph_value': ph_values
        })
        
        return df
    
    def test_file_upload_and_model_training(self):
        """Test 1: File Upload & Model Training"""
        print("=== Test 1: File Upload & LSTM Model Training ===")
        
        try:
            # Create and upload pH dataset
            df = self.create_realistic_ph_dataset()
            csv_content = df.to_csv(index=False)
            
            print(f"Created pH dataset: {len(df)} samples")
            print(f"pH characteristics: mean={df['ph_value'].mean():.2f}, std={df['ph_value'].std():.3f}")
            print(f"pH range: {df['ph_value'].min():.2f} - {df['ph_value'].max():.2f}")
            
            # Upload file
            files = {'file': ('comprehensive_ph_test.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                print(f"✅ File upload successful: {self.data_id}")
                
                # Train LSTM model
                training_params = {
                    "time_column": "timestamp",
                    "target_column": "ph_value",
                    "model_type": "lstm",
                    "seq_len": 8,
                    "pred_len": 3,
                    "epochs": 25,
                    "batch_size": 4,
                    "learning_rate": 0.001
                }
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": self.data_id, "model_type": "lstm"},
                    json=training_params
                )
                
                if response.status_code == 200:
                    model_data = response.json()
                    self.model_id = model_data.get('model_id')
                    print(f"✅ LSTM model training successful: {self.model_id}")
                    
                    if 'performance_metrics' in model_data:
                        metrics = model_data['performance_metrics']
                        print(f"   Performance: RMSE={metrics.get('rmse', 'N/A'):.3f}, MAE={metrics.get('mae', 'N/A'):.3f}")
                    
                    self.test_results['file_upload_training'] = True
                    return True
                else:
                    print(f"❌ LSTM training failed: {response.status_code}")
                    self.test_results['file_upload_training'] = False
                    return False
            else:
                print(f"❌ File upload failed: {response.status_code}")
                self.test_results['file_upload_training'] = False
                return False
                
        except Exception as e:
            print(f"❌ Test 1 error: {str(e)}")
            self.test_results['file_upload_training'] = False
            return False
    
    def test_prediction_quality(self):
        """Test 2: Prediction Quality - Historical Pattern Following"""
        print("\n=== Test 2: Prediction Quality Analysis ===")
        
        if not self.model_id:
            print("❌ Cannot test - no model trained")
            self.test_results['prediction_quality'] = False
            return False
        
        try:
            # Generate predictions
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": self.model_id, "steps": 30}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if predictions:
                    print(f"Generated {len(predictions)} predictions")
                    
                    # Test 1: Realistic pH values (6.0-8.0 range)
                    realistic_range = all(6.0 <= p <= 8.0 for p in predictions)
                    print(f"✅ Realistic pH range (6.0-8.0): {'PASS' if realistic_range else 'FAIL'}")
                    
                    # Test 2: Predictions show variability (not monotonic decline)
                    unique_values = len(set([round(p, 2) for p in predictions]))
                    has_variability = unique_values > 5
                    print(f"✅ Has variability ({unique_values} unique values): {'PASS' if has_variability else 'FAIL'}")
                    
                    # Test 3: No persistent downward bias
                    x = np.arange(len(predictions))
                    slope = np.polyfit(x, predictions, 1)[0]
                    no_downward_bias = slope > -0.01  # Not strongly downward
                    print(f"✅ No downward bias (slope={slope:.6f}): {'PASS' if no_downward_bias else 'FAIL'}")
                    
                    # Test 4: Predictions maintain historical characteristics
                    pred_mean = statistics.mean(predictions)
                    pred_std = statistics.stdev(predictions) if len(predictions) > 1 else 0
                    historical_mean = 7.4  # Expected from our dataset
                    maintains_characteristics = abs(pred_mean - historical_mean) < 0.5
                    print(f"✅ Maintains historical characteristics (mean={pred_mean:.3f}): {'PASS' if maintains_characteristics else 'FAIL'}")
                    
                    # Overall prediction quality assessment
                    quality_score = sum([realistic_range, has_variability, no_downward_bias, maintains_characteristics])
                    print(f"📊 Prediction Quality Score: {quality_score}/4 ({quality_score/4*100:.1f}%)")
                    
                    self.test_results['prediction_quality'] = quality_score >= 3
                    return quality_score >= 3
                else:
                    print("❌ No predictions generated")
                    self.test_results['prediction_quality'] = False
                    return False
            else:
                print(f"❌ Prediction generation failed: {response.status_code}")
                self.test_results['prediction_quality'] = False
                return False
                
        except Exception as e:
            print(f"❌ Test 2 error: {str(e)}")
            self.test_results['prediction_quality'] = False
            return False
    
    def test_pattern_analysis_functions(self):
        """Test 3: Enhanced Pattern Analysis Functions"""
        print("\n=== Test 3: Pattern Analysis Functions ===")
        
        try:
            # Test data quality report
            response = self.session.get(f"{API_BASE_URL}/data-quality-report")
            
            if response.status_code == 200:
                data = response.json()
                quality_score = data.get('quality_score', 0)
                recommendations = data.get('recommendations', [])
                
                print(f"✅ Data quality analysis successful")
                print(f"   Quality Score: {quality_score}")
                print(f"   Recommendations: {len(recommendations)}")
                
                # Test if analysis provides meaningful insights
                meaningful_analysis = quality_score > 50 and isinstance(recommendations, list)
                print(f"✅ Meaningful analysis: {'PASS' if meaningful_analysis else 'FAIL'}")
                
                self.test_results['pattern_analysis'] = meaningful_analysis
                return meaningful_analysis
            else:
                print(f"❌ Data quality report failed: {response.status_code}")
                self.test_results['pattern_analysis'] = False
                return False
                
        except Exception as e:
            print(f"❌ Test 3 error: {str(e)}")
            self.test_results['pattern_analysis'] = False
            return False
    
    def test_advanced_models_predict_next_steps(self):
        """Test 4: Advanced Models - predict_next_steps Method"""
        print("\n=== Test 4: Advanced Models predict_next_steps ===")
        
        if not self.model_id:
            print("❌ Cannot test - no model trained")
            self.test_results['advanced_models'] = False
            return False
        
        try:
            # Test advanced prediction endpoint
            payload = {"model_id": self.model_id, "steps": 20}
            response = self.session.post(f"{API_BASE_URL}/advanced-prediction", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if predictions:
                    print(f"✅ Advanced prediction successful: {len(predictions)} predictions")
                    
                    # Test 1: LSTM produces varied predictions (not monotonic)
                    x = np.arange(len(predictions))
                    slope = np.polyfit(x, predictions, 1)[0]
                    correlation = abs(np.corrcoef(x, predictions)[0, 1])
                    
                    not_monotonic = correlation < 0.8  # Not strongly linear
                    print(f"✅ Predictions vary (correlation={correlation:.3f}): {'PASS' if not_monotonic else 'FAIL'}")
                    
                    # Test 2: Bias correction prevents drift
                    pred_range = max(predictions) - min(predictions)
                    reasonable_range = 0.01 < pred_range < 0.5  # Reasonable variation
                    print(f"✅ Bias correction effective (range={pred_range:.3f}): {'PASS' if reasonable_range else 'FAIL'}")
                    
                    # Test 3: Maintains pH monitoring characteristics
                    pred_mean = statistics.mean(predictions)
                    in_ph_range = 6.5 <= pred_mean <= 8.0
                    print(f"✅ pH characteristics maintained (mean={pred_mean:.3f}): {'PASS' if in_ph_range else 'FAIL'}")
                    
                    advanced_score = sum([not_monotonic, reasonable_range, in_ph_range])
                    print(f"📊 Advanced Models Score: {advanced_score}/3 ({advanced_score/3*100:.1f}%)")
                    
                    self.test_results['advanced_models'] = advanced_score >= 2
                    return advanced_score >= 2
                else:
                    print("❌ No advanced predictions generated")
                    self.test_results['advanced_models'] = False
                    return False
            else:
                print(f"❌ Advanced prediction failed: {response.status_code}")
                self.test_results['advanced_models'] = False
                return False
                
        except Exception as e:
            print(f"❌ Test 4 error: {str(e)}")
            self.test_results['advanced_models'] = False
            return False
    
    def test_continuous_prediction_flow(self):
        """Test 5: Continuous Prediction Flow"""
        print("\n=== Test 5: Continuous Prediction Flow ===")
        
        if not self.model_id:
            print("❌ Cannot test - no model trained")
            self.test_results['continuous_prediction'] = False
            return False
        
        try:
            # Reset continuous predictions
            reset_response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            print(f"Reset continuous predictions: {reset_response.status_code}")
            
            # Test multiple prediction calls to check for accumulated bias
            all_predictions = []
            call_means = []
            successful_calls = 0
            
            for i in range(3):  # Test 3 calls
                response = self.session.get(
                    f"{API_BASE_URL}/generate-prediction",
                    params={"model_id": self.model_id, "steps": 10, "offset": i*3}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    
                    if predictions:
                        all_predictions.extend(predictions)
                        call_mean = statistics.mean(predictions)
                        call_means.append(call_mean)
                        successful_calls += 1
                        print(f"   Call {i+1}: {len(predictions)} predictions, mean={call_mean:.3f}")
                        time.sleep(0.3)
            
            if successful_calls >= 2:
                # Test 1: Multiple calls show realistic progression
                overall_range = max(all_predictions) - min(all_predictions)
                realistic_progression = 0.01 < overall_range < 0.3
                print(f"✅ Realistic progression (range={overall_range:.3f}): {'PASS' if realistic_progression else 'FAIL'}")
                
                # Test 2: No accumulated downward bias over time
                if len(call_means) >= 2:
                    means_slope = np.polyfit(range(len(call_means)), call_means, 1)[0]
                    no_accumulated_bias = means_slope > -0.02
                    print(f"✅ No accumulated bias (slope={means_slope:.6f}): {'PASS' if no_accumulated_bias else 'FAIL'}")
                else:
                    no_accumulated_bias = True
                
                # Test 3: Time series maintains pH characteristics
                overall_mean = statistics.mean(all_predictions)
                maintains_ph_characteristics = 6.5 <= overall_mean <= 8.0
                print(f"✅ Maintains pH characteristics (mean={overall_mean:.3f}): {'PASS' if maintains_ph_characteristics else 'FAIL'}")
                
                continuous_score = sum([realistic_progression, no_accumulated_bias, maintains_ph_characteristics])
                print(f"📊 Continuous Prediction Score: {continuous_score}/3 ({continuous_score/3*100:.1f}%)")
                
                self.test_results['continuous_prediction'] = continuous_score >= 2
                return continuous_score >= 2
            else:
                print("❌ Insufficient successful calls for continuous prediction test")
                self.test_results['continuous_prediction'] = False
                return False
                
        except Exception as e:
            print(f"❌ Test 5 error: {str(e)}")
            self.test_results['continuous_prediction'] = False
            return False
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*70)
        print("📋 COMPREHENSIVE pH PREDICTION ALGORITHM VERIFICATION REPORT")
        print("="*70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        # Individual test results
        test_descriptions = {
            'file_upload_training': 'File Upload & LSTM Model Training',
            'prediction_quality': 'Prediction Quality (Historical Pattern Following)',
            'pattern_analysis': 'Enhanced Pattern Analysis Functions',
            'advanced_models': 'Advanced Models predict_next_steps Method',
            'continuous_prediction': 'Continuous Prediction Flow'
        }
        
        for test_key, result in self.test_results.items():
            test_name = test_descriptions.get(test_key, test_key)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:45} {status}")
        
        print("-" * 70)
        print(f"OVERALL TEST RESULTS: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        # Specific downward trend assessment
        critical_tests = ['prediction_quality', 'advanced_models', 'continuous_prediction']
        critical_passed = sum(1 for test in critical_tests if self.test_results.get(test, False))
        
        print(f"DOWNWARD TREND CRITICAL TESTS: {critical_passed}/{len(critical_tests)} passed ({critical_passed/len(critical_tests)*100:.1f}%)")
        
        # Final conclusion
        print("\n🔍 FINAL ASSESSMENT:")
        
        if passed_tests == total_tests:
            print("🎉 EXCELLENT: All tests passed!")
            print("   ✅ pH prediction algorithm improvements are fully working")
            print("   ✅ Downward trend issue has been completely resolved")
            print("   ✅ All requirements from review request have been met")
        elif passed_tests >= total_tests * 0.8:
            print("✅ GOOD: Most tests passed")
            print("   ✅ Significant improvements in pH prediction algorithm")
            print("   ✅ Downward trend issue largely resolved")
            print("   ⚠️  Minor issues may need attention")
        elif critical_passed >= len(critical_tests) * 0.7:
            print("⚠️  PARTIAL: Core functionality improved")
            print("   ✅ Key downward trend issues addressed")
            print("   ⚠️  Some implementation aspects need refinement")
        else:
            print("❌ NEEDS WORK: Significant issues remain")
            print("   ❌ Downward trend issue not fully resolved")
            print("   ❌ Algorithm improvements need more work")
        
        return passed_tests >= total_tests * 0.8
    
    def run_comprehensive_test(self):
        """Run all comprehensive tests"""
        print("🧪 COMPREHENSIVE pH PREDICTION ALGORITHM VERIFICATION")
        print("="*70)
        print("Testing improved pH prediction algorithm for downward trend resolution")
        print("="*70)
        
        # Run all tests in sequence
        success = True
        success &= self.test_file_upload_and_model_training()
        success &= self.test_prediction_quality()
        success &= self.test_pattern_analysis_functions()
        success &= self.test_advanced_models_predict_next_steps()
        success &= self.test_continuous_prediction_flow()
        
        # Generate final report
        overall_success = self.generate_final_report()
        
        return overall_success

if __name__ == "__main__":
    tester = ComprehensivePhTester()
    success = tester.run_comprehensive_test()