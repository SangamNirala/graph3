#!/usr/bin/env python3
"""
pH Prediction Algorithm Testing - Focus on Downward Trend Issue Resolution
Tests the improved pH prediction algorithm to verify it no longer shows consistent downward trend
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
import statistics

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://064f3bb3-c010-4892-8a8e-8e29d9900fe8.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing pH prediction algorithm at: {API_BASE_URL}")

class PhPredictionTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        self.test_results = {}
        
    def create_ph_dataset(self):
        """Create realistic pH monitoring dataset for testing"""
        # Generate 49 samples of pH data (matching the test_result.md dataset size)
        timestamps = pd.date_range(start='2024-01-01', periods=49, freq='H')
        
        # Create realistic pH values around 7.0-7.8 range with natural variations
        base_ph = 7.4
        ph_values = []
        
        for i in range(49):
            # Add realistic pH fluctuations
            trend = 0.1 * np.sin(i * 2 * np.pi / 24)  # Daily cycle
            noise = np.random.normal(0, 0.05)  # Small random variations
            periodic = 0.05 * np.sin(i * 2 * np.pi / 8)  # 8-hour cycle
            
            ph_value = base_ph + trend + noise + periodic
            # Keep pH in realistic range (6.5-8.0)
            ph_value = max(6.5, min(8.0, ph_value))
            ph_values.append(round(ph_value, 2))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ph_value': ph_values
        })
        
        return df
    
    def test_file_upload_and_analysis(self):
        """Test 1: File Upload & Data Analysis"""
        print("\n=== Testing pH Dataset Upload and Analysis ===")
        
        try:
            # Create pH dataset
            df = self.create_ph_dataset()
            csv_content = df.to_csv(index=False)
            
            print(f"Created pH dataset with {len(df)} samples")
            print(f"pH range: {df['ph_value'].min():.2f} - {df['ph_value'].max():.2f}")
            print(f"pH mean: {df['ph_value'].mean():.2f}")
            
            # Prepare file for upload
            files = {
                'file': ('ph_data.csv', csv_content, 'text/csv')
            }
            
            # Test file upload
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                
                print("✅ pH dataset upload successful")
                print(f"   Data ID: {self.data_id}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Validate analysis results
                analysis = data['analysis']
                if 'timestamp' in analysis['time_columns'] and 'ph_value' in analysis['numeric_columns']:
                    print("✅ Data analysis correctly identified pH columns")
                    self.test_results['file_upload'] = True
                else:
                    print("❌ Data analysis failed to identify pH columns correctly")
                    self.test_results['file_upload'] = False
                    
            else:
                print(f"❌ pH dataset upload failed: {response.status_code} - {response.text}")
                self.test_results['file_upload'] = False
                
        except Exception as e:
            print(f"❌ pH dataset upload test error: {str(e)}")
            self.test_results['file_upload'] = False
    
    def test_lstm_model_training(self):
        """Test 2: LSTM Model Training for pH Prediction"""
        print("\n=== Testing LSTM Model Training ===")
        
        if not self.data_id:
            print("❌ Cannot test LSTM training - no data uploaded")
            self.test_results['lstm_training'] = False
            return
            
        try:
            # Prepare LSTM training parameters
            training_params = {
                "time_column": "timestamp",
                "target_column": "ph_value",
                "model_type": "lstm",
                "seq_len": 8,  # Adjusted for small dataset
                "pred_len": 3,
                "epochs": 20,
                "batch_size": 4,
                "learning_rate": 0.001
            }
            
            # Test LSTM model training
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
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                
                # Check for performance metrics
                if 'performance_metrics' in data:
                    metrics = data['performance_metrics']
                    print(f"   Performance Metrics: {metrics}")
                
                if 'evaluation_grade' in data:
                    print(f"   Evaluation Grade: {data['evaluation_grade']}")
                
                self.test_results['lstm_training'] = True
                
            else:
                print(f"❌ LSTM model training failed: {response.status_code} - {response.text}")
                self.test_results['lstm_training'] = False
                
        except Exception as e:
            print(f"❌ LSTM model training test error: {str(e)}")
            self.test_results['lstm_training'] = False
    
    def test_single_prediction_quality(self):
        """Test 3: Single Prediction Generation - Check for Downward Trend Issues"""
        print("\n=== Testing Single Prediction Quality (Downward Trend Check) ===")
        
        if not self.model_id:
            print("❌ Cannot test predictions - no model trained")
            self.test_results['single_prediction'] = False
            return
            
        try:
            # Generate single prediction
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": self.model_id, "steps": 30}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                
                print("✅ Single prediction generation successful")
                print(f"   Generated {len(predictions)} predictions")
                print(f"   Prediction range: {min(predictions):.2f} - {max(predictions):.2f}")
                print(f"   Prediction mean: {statistics.mean(predictions):.2f}")
                
                # CRITICAL TEST: Check for downward trend bias
                trend_analysis = self.analyze_prediction_trend(predictions)
                print(f"   Trend Analysis: {trend_analysis}")
                
                # Test for realistic pH values
                realistic_range = all(6.0 <= p <= 8.5 for p in predictions)
                print(f"   Realistic pH range (6.0-8.5): {'✅' if realistic_range else '❌'}")
                
                # Test for variability (not monotonic)
                has_variability = len(set([round(p, 1) for p in predictions])) > 5
                print(f"   Has variability (not monotonic): {'✅' if has_variability else '❌'}")
                
                # Test for no persistent downward bias
                no_downward_bias = trend_analysis['trend_direction'] != 'strongly_downward'
                print(f"   No persistent downward bias: {'✅' if no_downward_bias else '❌'}")
                
                self.test_results['single_prediction'] = realistic_range and has_variability and no_downward_bias
                
            else:
                print(f"❌ Single prediction failed: {response.status_code} - {response.text}")
                self.test_results['single_prediction'] = False
                
        except Exception as e:
            print(f"❌ Single prediction test error: {str(e)}")
            self.test_results['single_prediction'] = False
    
    def test_continuous_prediction_flow(self):
        """Test 4: Continuous Prediction Flow - Multiple Calls for Trend Analysis"""
        print("\n=== Testing Continuous Prediction Flow ===")
        
        if not self.model_id:
            print("❌ Cannot test continuous predictions - no model trained")
            self.test_results['continuous_prediction'] = False
            return
            
        try:
            # Reset continuous predictions
            reset_response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            print(f"Reset response: {reset_response.status_code}")
            
            # Generate multiple continuous predictions to test for accumulated bias
            all_predictions = []
            prediction_calls = 5  # Test 5 continuous calls
            
            for i in range(prediction_calls):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 10, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data is not None:
                        predictions = data.get('predictions', [])
                        all_predictions.extend(predictions)
                        
                        if predictions:
                            print(f"   Call {i+1}: Generated {len(predictions)} predictions, mean: {statistics.mean(predictions):.2f}")
                            
                            # Check pattern analysis if available
                            if 'pattern_analysis' in data:
                                pattern = data['pattern_analysis']
                                print(f"   Pattern Analysis - Trend: {pattern.get('trend_slope', 'N/A'):.4f}, "
                                      f"Velocity: {pattern.get('velocity', 'N/A'):.4f}")
                        else:
                            print(f"   Call {i+1}: No predictions returned")
                    else:
                        print(f"   Call {i+1}: Null response data")
                else:
                    print(f"❌ Continuous prediction call {i+1} failed: {response.status_code} - {response.text}")
                    
                time.sleep(0.5)  # Small delay between calls
            
            if all_predictions:
                print("✅ Continuous prediction flow successful")
                print(f"   Total predictions generated: {len(all_predictions)}")
                print(f"   Overall range: {min(all_predictions):.2f} - {max(all_predictions):.2f}")
                print(f"   Overall mean: {statistics.mean(all_predictions):.2f}")
                
                # CRITICAL TEST: Check for accumulated downward bias over time
                bias_analysis = self.analyze_accumulated_bias(all_predictions, prediction_calls)
                print(f"   Accumulated Bias Analysis: {bias_analysis}")
                
                # Test for realistic pH maintenance
                realistic_maintenance = all(6.0 <= p <= 8.5 for p in all_predictions)
                print(f"   Maintains realistic pH range: {'✅' if realistic_maintenance else '❌'}")
                
                # Test for no accumulated downward drift
                no_accumulated_drift = bias_analysis['has_accumulated_drift'] == False
                print(f"   No accumulated downward drift: {'✅' if no_accumulated_drift else '❌'}")
                
                self.test_results['continuous_prediction'] = realistic_maintenance and no_accumulated_drift
                
            else:
                print("❌ No continuous predictions generated")
                self.test_results['continuous_prediction'] = False
                
        except Exception as e:
            print(f"❌ Continuous prediction test error: {str(e)}")
            self.test_results['continuous_prediction'] = False
    
    def test_pattern_analysis_functions(self):
        """Test 5: Enhanced Pattern Analysis Functions"""
        print("\n=== Testing Enhanced Pattern Analysis Functions ===")
        
        try:
            # Test data quality report (includes pattern analysis)
            response = self.session.get(f"{API_BASE_URL}/data-quality-report")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Data quality report successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Quality Score: {data.get('quality_score')}")
                print(f"   Recommendations: {len(data.get('recommendations', []))}")
                
                # Check if quality analysis provides proper insights
                quality_score = data.get('quality_score', 0)
                has_recommendations = 'recommendations' in data
                
                print(f"   Quality score > 50: {'✅' if quality_score > 50 else '❌'}")
                print(f"   Has recommendations structure: {'✅' if has_recommendations else '❌'}")
                
                self.test_results['pattern_analysis'] = quality_score > 50 and has_recommendations
                
            else:
                print(f"❌ Data quality report failed: {response.status_code} - {response.text}")
                self.test_results['pattern_analysis'] = False
                
        except Exception as e:
            print(f"❌ Pattern analysis test error: {str(e)}")
            self.test_results['pattern_analysis'] = False
    
    def test_advanced_prediction_endpoint(self):
        """Test 6: Advanced Prediction Endpoint with Bias Correction"""
        print("\n=== Testing Advanced Prediction Endpoint ===")
        
        if not self.model_id:
            print("❌ Cannot test advanced predictions - no model trained")
            self.test_results['advanced_prediction'] = False
            return
            
        try:
            # Test advanced prediction endpoint (POST method)
            payload = {
                "model_id": self.model_id,
                "steps": 20
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/advanced-prediction",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                print("✅ Advanced prediction endpoint successful")
                print(f"   Generated {len(predictions)} advanced predictions")
                
                if predictions:
                    print(f"   Prediction range: {min(predictions):.2f} - {max(predictions):.2f}")
                    print(f"   Prediction mean: {statistics.mean(predictions):.2f}")
                    
                    # Test for bias correction effectiveness
                    bias_corrected = self.test_bias_correction(predictions)
                    print(f"   Bias correction effective: {'✅' if bias_corrected else '❌'}")
                    
                    self.test_results['advanced_prediction'] = bias_corrected
                else:
                    print("❌ No predictions returned from advanced endpoint")
                    self.test_results['advanced_prediction'] = False
                
            else:
                print(f"❌ Advanced prediction failed: {response.status_code} - {response.text}")
                self.test_results['advanced_prediction'] = False
                
        except Exception as e:
            print(f"❌ Advanced prediction test error: {str(e)}")
            self.test_results['advanced_prediction'] = False
    
    def analyze_prediction_trend(self, predictions):
        """Analyze prediction trend to detect downward bias"""
        if len(predictions) < 3:
            return {"trend_direction": "insufficient_data"}
        
        # Calculate linear trend
        x = np.arange(len(predictions))
        slope = np.polyfit(x, predictions, 1)[0]
        
        # Calculate trend strength
        correlation = np.corrcoef(x, predictions)[0, 1]
        
        # Determine trend direction
        if slope < -0.01 and correlation < -0.5:
            trend_direction = "strongly_downward"
        elif slope < -0.005:
            trend_direction = "moderately_downward"
        elif slope > 0.005:
            trend_direction = "upward"
        else:
            trend_direction = "stable"
        
        return {
            "slope": slope,
            "correlation": correlation,
            "trend_direction": trend_direction,
            "trend_strength": abs(correlation)
        }
    
    def analyze_accumulated_bias(self, all_predictions, num_calls):
        """Analyze accumulated bias across multiple prediction calls"""
        if len(all_predictions) < num_calls:
            return {"has_accumulated_drift": True, "reason": "insufficient_data"}
        
        # Split predictions by call and analyze means
        predictions_per_call = len(all_predictions) // num_calls
        call_means = []
        
        for i in range(num_calls):
            start_idx = i * predictions_per_call
            end_idx = (i + 1) * predictions_per_call
            call_predictions = all_predictions[start_idx:end_idx]
            if call_predictions:
                call_means.append(statistics.mean(call_predictions))
        
        if len(call_means) < 2:
            return {"has_accumulated_drift": True, "reason": "insufficient_calls"}
        
        # Check for consistent downward trend in means
        mean_trend = self.analyze_prediction_trend(call_means)
        
        # Consider accumulated drift if there's a strong downward trend in call means
        has_drift = mean_trend["trend_direction"] == "strongly_downward"
        
        return {
            "has_accumulated_drift": has_drift,
            "call_means": call_means,
            "mean_trend": mean_trend,
            "overall_slope": mean_trend.get("slope", 0)
        }
    
    def test_bias_correction(self, predictions):
        """Test if bias correction is working effectively"""
        if len(predictions) < 5:
            return False
        
        # Check for reasonable pH values
        realistic_range = all(6.0 <= p <= 8.5 for p in predictions)
        
        # Check for variability (not all same value)
        has_variation = len(set([round(p, 1) for p in predictions])) > 2
        
        # Check that predictions don't show extreme downward trend
        trend_analysis = self.analyze_prediction_trend(predictions)
        no_extreme_downward = trend_analysis["trend_direction"] != "strongly_downward"
        
        return realistic_range and has_variation and no_extreme_downward
    
    def run_all_tests(self):
        """Run all pH prediction algorithm tests"""
        print("🧪 Starting pH Prediction Algorithm Testing")
        print("=" * 60)
        
        # Run tests in sequence
        self.test_file_upload_and_analysis()
        self.test_lstm_model_training()
        self.test_single_prediction_quality()
        self.test_continuous_prediction_flow()
        self.test_pattern_analysis_functions()
        self.test_advanced_prediction_endpoint()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🧪 pH PREDICTION ALGORITHM TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:30} {status}")
        
        print("-" * 60)
        print(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        # Specific analysis for downward trend issue
        critical_tests = ['single_prediction', 'continuous_prediction', 'advanced_prediction']
        critical_passed = sum(1 for test in critical_tests if self.test_results.get(test, False))
        
        print(f"CRITICAL DOWNWARD TREND TESTS: {critical_passed}/{len(critical_tests)} passed")
        
        if critical_passed == len(critical_tests):
            print("🎉 DOWNWARD TREND ISSUE APPEARS TO BE RESOLVED!")
        else:
            print("⚠️  DOWNWARD TREND ISSUE MAY STILL EXIST")
        
        return self.test_results

if __name__ == "__main__":
    tester = PhPredictionTester()
    results = tester.run_all_tests()