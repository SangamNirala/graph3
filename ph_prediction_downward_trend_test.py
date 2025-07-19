#!/usr/bin/env python3
"""
Focused pH Prediction Downward Trend Testing
Tests specifically for the downward trend issue that the user is still experiencing
"""

import requests
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
from pathlib import Path
import statistics
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://909a9d1c-9da6-4ed6-bd0a-ff6c4fb747bb.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing pH prediction system at: {API_BASE_URL}")

class PhPredictionTester:
    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification for testing
        self.data_id = None
        self.model_id = None
        self.test_results = {}
        
    def create_realistic_ph_data(self, num_points=50):
        """Create realistic pH data for testing - should stay in 6.0-8.0 range"""
        # Generate timestamps
        start_time = datetime.now() - timedelta(hours=num_points)
        timestamps = [start_time + timedelta(hours=i) for i in range(num_points)]
        
        # Generate realistic pH values (6.0-8.0 range with natural variation)
        base_ph = 7.2  # Slightly alkaline
        ph_values = []
        
        for i in range(num_points):
            # Add some natural variation and patterns
            time_factor = i / num_points
            
            # Slight daily cycle
            daily_cycle = 0.2 * np.sin(2 * np.pi * time_factor * 2)  # 2 cycles over the period
            
            # Random noise
            noise = np.random.normal(0, 0.1)
            
            # Slight trend (should be minimal)
            trend = 0.05 * np.sin(time_factor * np.pi)  # Gentle wave pattern
            
            ph_value = base_ph + daily_cycle + noise + trend
            
            # Ensure realistic bounds
            ph_value = max(6.0, min(8.0, ph_value))
            ph_values.append(ph_value)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ph_value': ph_values
        })
        
        return df
    
    def test_ph_simulation_endpoints(self):
        """Test 1: pH simulation endpoints for realistic data generation"""
        print("\n=== Testing pH Simulation Endpoints ===")
        
        try:
            # Test current pH simulation
            response = self.session.get(f"{API_BASE_URL}/ph-simulation")
            
            if response.status_code == 200:
                data = response.json()
                ph_value = data.get('ph_value')
                confidence = data.get('confidence')
                
                print(f"✅ pH simulation endpoint working")
                print(f"   Current pH: {ph_value}")
                print(f"   Confidence: {confidence}%")
                
                # Validate pH is in realistic range
                if 6.0 <= ph_value <= 8.0:
                    print("✅ pH value in realistic range (6.0-8.0)")
                    self.test_results['ph_simulation_range'] = True
                else:
                    print(f"❌ pH value {ph_value} outside realistic range (6.0-8.0)")
                    self.test_results['ph_simulation_range'] = False
                
                # Test historical pH data
                hist_response = self.session.get(f"{API_BASE_URL}/ph-simulation-history")
                if hist_response.status_code == 200:
                    hist_data = hist_response.json()
                    ph_history = [point['ph_value'] for point in hist_data['data']]
                    
                    print(f"✅ pH history endpoint working ({len(ph_history)} points)")
                    
                    # Check if all historical values are in range
                    in_range = all(6.0 <= ph <= 8.0 for ph in ph_history)
                    if in_range:
                        print("✅ All historical pH values in realistic range")
                        self.test_results['ph_history_range'] = True
                    else:
                        out_of_range = [ph for ph in ph_history if not (6.0 <= ph <= 8.0)]
                        print(f"❌ {len(out_of_range)} pH values outside range: {out_of_range[:5]}...")
                        self.test_results['ph_history_range'] = False
                        
                    # Check for natural variability
                    ph_std = np.std(ph_history)
                    ph_mean = np.mean(ph_history)
                    print(f"   pH statistics: mean={ph_mean:.3f}, std={ph_std:.3f}")
                    
                    if 0.05 <= ph_std <= 0.5:  # Reasonable variability
                        print("✅ pH shows natural variability")
                        self.test_results['ph_variability'] = True
                    else:
                        print(f"❌ pH variability unusual: std={ph_std:.3f}")
                        self.test_results['ph_variability'] = False
                        
                else:
                    print(f"❌ pH history endpoint failed: {hist_response.status_code}")
                    self.test_results['ph_history_range'] = False
                    
            else:
                print(f"❌ pH simulation endpoint failed: {response.status_code}")
                self.test_results['ph_simulation_range'] = False
                
        except Exception as e:
            print(f"❌ pH simulation test error: {str(e)}")
            self.test_results['ph_simulation_range'] = False
            self.test_results['ph_history_range'] = False
    
    def test_upload_ph_data_and_train(self):
        """Test 2: Upload pH data and train LSTM model"""
        print("\n=== Testing pH Data Upload and LSTM Training ===")
        
        try:
            # Create realistic pH dataset
            df = self.create_realistic_ph_data(50)
            csv_content = df.to_csv(index=False)
            
            print(f"Created pH dataset: {len(df)} points, range {df['ph_value'].min():.2f}-{df['ph_value'].max():.2f}")
            
            # Upload data
            files = {
                'file': ('ph_data.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                
                print("✅ pH data upload successful")
                print(f"   Data ID: {self.data_id}")
                
                # Check if pH column was detected
                analysis = data['analysis']
                if 'ph_value' in analysis['numeric_columns']:
                    print("✅ pH column correctly identified as numeric")
                    self.test_results['ph_data_upload'] = True
                else:
                    print("❌ pH column not identified correctly")
                    self.test_results['ph_data_upload'] = False
                    return
                
                # Train LSTM model
                training_params = {
                    "time_column": "timestamp",
                    "target_column": "ph_value",
                    "seq_len": 10,  # Smaller for small dataset
                    "pred_len": 5,
                    "epochs": 20,
                    "batch_size": 4
                }
                
                train_response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": self.data_id, "model_type": "lstm"},
                    json=training_params
                )
                
                if train_response.status_code == 200:
                    train_data = train_response.json()
                    self.model_id = train_data.get('model_id')
                    
                    print("✅ LSTM model training successful")
                    print(f"   Model ID: {self.model_id}")
                    self.test_results['lstm_training'] = True
                else:
                    print(f"❌ LSTM training failed: {train_response.status_code} - {train_response.text}")
                    self.test_results['lstm_training'] = False
                    
            else:
                print(f"❌ pH data upload failed: {response.status_code} - {response.text}")
                self.test_results['ph_data_upload'] = False
                
        except Exception as e:
            print(f"❌ pH data upload/training error: {str(e)}")
            self.test_results['ph_data_upload'] = False
            self.test_results['lstm_training'] = False
    
    def test_single_prediction_bias(self):
        """Test 3: Single prediction to check for downward bias"""
        print("\n=== Testing Single Prediction for Downward Bias ===")
        
        if not self.model_id:
            print("❌ Cannot test predictions - no trained model")
            self.test_results['single_prediction_bias'] = False
            return
            
        try:
            # Generate single prediction
            pred_response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": self.model_id, "steps": 10}
            )
            
            if pred_response.status_code == 200:
                pred_data = pred_response.json()
                predictions = pred_data.get('predictions', [])
                
                if predictions:
                    print(f"✅ Single prediction generated: {len(predictions)} steps")
                    print(f"   Predictions: {[f'{p:.3f}' for p in predictions[:5]]}...")
                    
                    # Check for realistic range
                    in_range = all(5.5 <= p <= 8.5 for p in predictions)  # Slightly wider tolerance
                    if in_range:
                        print("✅ All predictions in realistic pH range")
                        self.test_results['prediction_range'] = True
                    else:
                        out_of_range = [p for p in predictions if not (5.5 <= p <= 8.5)]
                        print(f"❌ {len(out_of_range)} predictions outside range: {out_of_range}")
                        self.test_results['prediction_range'] = False
                    
                    # Check for downward trend bias
                    if len(predictions) >= 5:
                        # Calculate trend slope
                        x = np.arange(len(predictions))
                        slope = np.polyfit(x, predictions, 1)[0]
                        
                        print(f"   Prediction trend slope: {slope:.6f}")
                        
                        # Check if slope is significantly negative (downward bias)
                        if slope < -0.01:  # Significant downward trend
                            print(f"❌ DOWNWARD BIAS DETECTED: slope = {slope:.6f}")
                            self.test_results['single_prediction_bias'] = False
                        elif slope > 0.01:  # Significant upward trend
                            print(f"⚠️  Upward bias detected: slope = {slope:.6f}")
                            self.test_results['single_prediction_bias'] = True  # Not downward, but still biased
                        else:
                            print(f"✅ No significant trend bias: slope = {slope:.6f}")
                            self.test_results['single_prediction_bias'] = True
                    
                    # Check for natural variability
                    pred_std = np.std(predictions)
                    if pred_std < 0.01:  # Too little variation
                        print(f"❌ Predictions lack natural variability: std = {pred_std:.6f}")
                        self.test_results['prediction_variability'] = False
                    else:
                        print(f"✅ Predictions show natural variability: std = {pred_std:.6f}")
                        self.test_results['prediction_variability'] = True
                        
                else:
                    print("❌ No predictions returned")
                    self.test_results['single_prediction_bias'] = False
                    
            else:
                print(f"❌ Prediction generation failed: {pred_response.status_code} - {pred_response.text}")
                self.test_results['single_prediction_bias'] = False
                
        except Exception as e:
            print(f"❌ Single prediction test error: {str(e)}")
            self.test_results['single_prediction_bias'] = False
    
    def test_multiple_sequential_predictions(self):
        """Test 4: Multiple sequential predictions to detect accumulated bias"""
        print("\n=== Testing Multiple Sequential Predictions for Accumulated Bias ===")
        
        if not self.model_id:
            print("❌ Cannot test sequential predictions - no trained model")
            self.test_results['sequential_bias'] = False
            return
            
        try:
            all_predictions = []
            prediction_means = []
            
            # Generate 5 sequential predictions
            for i in range(5):
                pred_response = self.session.get(
                    f"{API_BASE_URL}/generate-prediction",
                    params={"model_id": self.model_id, "steps": 10}
                )
                
                if pred_response.status_code == 200:
                    pred_data = pred_response.json()
                    predictions = pred_data.get('predictions', [])
                    
                    if predictions:
                        all_predictions.extend(predictions)
                        prediction_means.append(np.mean(predictions))
                        print(f"   Call {i+1}: mean = {np.mean(predictions):.3f}, range = {min(predictions):.3f}-{max(predictions):.3f}")
                    else:
                        print(f"❌ No predictions in call {i+1}")
                        
                    # Small delay between calls
                    time.sleep(0.5)
                else:
                    print(f"❌ Prediction call {i+1} failed: {pred_response.status_code}")
            
            if len(prediction_means) >= 3:
                print(f"✅ Generated {len(prediction_means)} sequential prediction sets")
                
                # Check if means are consistently decreasing (accumulated downward bias)
                mean_trend = np.polyfit(range(len(prediction_means)), prediction_means, 1)[0]
                print(f"   Sequential means trend slope: {mean_trend:.6f}")
                
                if mean_trend < -0.01:
                    print(f"❌ ACCUMULATED DOWNWARD BIAS DETECTED: {mean_trend:.6f}")
                    self.test_results['sequential_bias'] = False
                else:
                    print(f"✅ No significant accumulated bias: {mean_trend:.6f}")
                    self.test_results['sequential_bias'] = True
                
                # Check overall range of all predictions
                if all_predictions:
                    overall_min = min(all_predictions)
                    overall_max = max(all_predictions)
                    overall_mean = np.mean(all_predictions)
                    overall_std = np.std(all_predictions)
                    
                    print(f"   Overall statistics: mean={overall_mean:.3f}, std={overall_std:.3f}")
                    print(f"   Overall range: {overall_min:.3f} to {overall_max:.3f}")
                    
                    # Check if predictions maintain realistic characteristics
                    if 6.0 <= overall_mean <= 8.0 and overall_std > 0.02:
                        print("✅ Sequential predictions maintain realistic characteristics")
                        self.test_results['sequential_characteristics'] = True
                    else:
                        print("❌ Sequential predictions lose realistic characteristics")
                        self.test_results['sequential_characteristics'] = False
                        
            else:
                print("❌ Insufficient sequential predictions generated")
                self.test_results['sequential_bias'] = False
                
        except Exception as e:
            print(f"❌ Sequential prediction test error: {str(e)}")
            self.test_results['sequential_bias'] = False
    
    def test_continuous_prediction_system(self):
        """Test 5: Continuous prediction system for bias accumulation"""
        print("\n=== Testing Continuous Prediction System ===")
        
        try:
            # Reset continuous predictions
            reset_response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            if reset_response.status_code == 200:
                print("✅ Continuous prediction system reset")
            else:
                print(f"⚠️  Reset failed: {reset_response.status_code}")
            
            # Start continuous prediction
            start_response = self.session.post(f"{API_BASE_URL}/start-continuous-prediction")
            if start_response.status_code == 200:
                print("✅ Continuous prediction started")
                time.sleep(2)  # Let it run briefly
                
                # Generate several continuous predictions
                continuous_means = []
                for i in range(3):
                    cont_response = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": self.model_id, "steps": 10}
                    )
                    
                    if cont_response.status_code == 200:
                        cont_data = cont_response.json()
                        predictions = cont_data.get('predictions', [])
                        
                        if predictions:
                            mean_pred = np.mean(predictions)
                            continuous_means.append(mean_pred)
                            print(f"   Continuous call {i+1}: mean = {mean_pred:.3f}")
                        
                        time.sleep(1)
                    else:
                        print(f"❌ Continuous prediction call {i+1} failed: {cont_response.status_code}")
                
                # Stop continuous prediction
                stop_response = self.session.post(f"{API_BASE_URL}/stop-continuous-prediction")
                if stop_response.status_code == 200:
                    print("✅ Continuous prediction stopped")
                
                # Analyze continuous prediction bias
                if len(continuous_means) >= 2:
                    cont_trend = np.polyfit(range(len(continuous_means)), continuous_means, 1)[0]
                    print(f"   Continuous prediction trend: {cont_trend:.6f}")
                    
                    if cont_trend < -0.01:
                        print(f"❌ CONTINUOUS DOWNWARD BIAS DETECTED: {cont_trend:.6f}")
                        self.test_results['continuous_bias'] = False
                    else:
                        print(f"✅ No significant continuous bias: {cont_trend:.6f}")
                        self.test_results['continuous_bias'] = True
                else:
                    print("❌ Insufficient continuous predictions")
                    self.test_results['continuous_bias'] = False
                    
            else:
                print(f"❌ Failed to start continuous prediction: {start_response.status_code}")
                self.test_results['continuous_bias'] = False
                
        except Exception as e:
            print(f"❌ Continuous prediction test error: {str(e)}")
            self.test_results['continuous_bias'] = False
    
    def test_pattern_following_capabilities(self):
        """Test 7: Pattern following capabilities with different data patterns"""
        print("\n=== Testing Pattern Following Capabilities ===")
        
        try:
            # Test with different pattern types
            pattern_tests = []
            
            # 1. Test with U-shaped pattern
            u_shaped_data = self.create_u_shaped_ph_data()
            pattern_tests.append(("U-shaped", u_shaped_data))
            
            # 2. Test with trending pattern
            trending_data = self.create_trending_ph_data()
            pattern_tests.append(("Trending", trending_data))
            
            # 3. Test with cyclical pattern
            cyclical_data = self.create_cyclical_ph_data()
            pattern_tests.append(("Cyclical", cyclical_data))
            
            pattern_results = []
            
            for pattern_name, test_data in pattern_tests:
                print(f"\n   Testing {pattern_name} pattern:")
                
                # Upload pattern data
                csv_content = test_data.to_csv(index=False)
                files = {'file': (f'{pattern_name.lower()}_ph_data.csv', csv_content, 'text/csv')}
                
                upload_response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if upload_response.status_code == 200:
                    upload_data = upload_response.json()
                    pattern_data_id = upload_data.get('data_id')
                    
                    # Test advanced prediction with this pattern
                    adv_response = self.session.post(
                        f"{API_BASE_URL}/advanced-prediction",
                        params={"data_id": pattern_data_id, "steps": 20}
                    )
                    
                    if adv_response.status_code == 200:
                        adv_data = adv_response.json()
                        predictions = adv_data.get('predictions', [])
                        
                        if predictions:
                            # Analyze how well predictions follow the pattern
                            historical_values = test_data['ph_value'].values
                            
                            # Calculate pattern characteristics
                            hist_mean = np.mean(historical_values)
                            hist_std = np.std(historical_values)
                            hist_trend = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
                            
                            pred_mean = np.mean(predictions)
                            pred_std = np.std(predictions)
                            pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
                            
                            # Check pattern preservation
                            mean_preservation = abs(pred_mean - hist_mean) / hist_mean
                            std_preservation = abs(pred_std - hist_std) / hist_std if hist_std > 0 else 0
                            trend_preservation = abs(pred_trend - hist_trend) / abs(hist_trend) if abs(hist_trend) > 0.001 else 0
                            
                            pattern_score = 1.0 - (mean_preservation + std_preservation + trend_preservation) / 3
                            pattern_score = max(0, min(1, pattern_score))
                            
                            print(f"     Pattern preservation score: {pattern_score:.3f}")
                            print(f"     Historical: mean={hist_mean:.3f}, std={hist_std:.3f}, trend={hist_trend:.6f}")
                            print(f"     Predicted:  mean={pred_mean:.3f}, std={pred_std:.3f}, trend={pred_trend:.6f}")
                            
                            pattern_results.append({
                                'pattern': pattern_name,
                                'score': pattern_score,
                                'predictions_count': len(predictions),
                                'in_range': all(5.5 <= p <= 8.5 for p in predictions)
                            })
                            
                        else:
                            print(f"     ❌ No predictions generated for {pattern_name}")
                    else:
                        print(f"     ❌ Advanced prediction failed for {pattern_name}: {adv_response.status_code}")
                else:
                    print(f"     ❌ Upload failed for {pattern_name}: {upload_response.status_code}")
            
            # Evaluate overall pattern following capability
            if pattern_results:
                avg_score = np.mean([r['score'] for r in pattern_results])
                all_in_range = all(r['in_range'] for r in pattern_results)
                
                print(f"\n   Overall pattern following score: {avg_score:.3f}")
                
                if avg_score >= 0.7 and all_in_range:
                    print("✅ Excellent pattern following capabilities")
                    self.test_results['pattern_following'] = True
                elif avg_score >= 0.5:
                    print("⚠️  Moderate pattern following capabilities")
                    self.test_results['pattern_following'] = True
                else:
                    print("❌ Poor pattern following capabilities")
                    self.test_results['pattern_following'] = False
            else:
                print("❌ No pattern tests completed")
                self.test_results['pattern_following'] = False
                
        except Exception as e:
            print(f"❌ Pattern following test error: {str(e)}")
            self.test_results['pattern_following'] = False
    
    def create_u_shaped_ph_data(self, num_points=40):
        """Create U-shaped pH data pattern"""
        timestamps = [datetime.now() - timedelta(hours=num_points-i) for i in range(num_points)]
        
        # Create U-shaped pattern
        x = np.linspace(-2, 2, num_points)
        ph_values = 7.0 + 0.3 * x**2 + np.random.normal(0, 0.05, num_points)
        
        # Ensure realistic bounds
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        return pd.DataFrame({'timestamp': timestamps, 'ph_value': ph_values})
    
    def create_trending_ph_data(self, num_points=40):
        """Create trending pH data pattern"""
        timestamps = [datetime.now() - timedelta(hours=num_points-i) for i in range(num_points)]
        
        # Create trending pattern
        trend = np.linspace(6.8, 7.4, num_points)
        noise = np.random.normal(0, 0.08, num_points)
        ph_values = trend + noise
        
        # Ensure realistic bounds
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        return pd.DataFrame({'timestamp': timestamps, 'ph_value': ph_values})
    
    def create_cyclical_ph_data(self, num_points=40):
        """Create cyclical pH data pattern"""
        timestamps = [datetime.now() - timedelta(hours=num_points-i) for i in range(num_points)]
        
        # Create cyclical pattern
        x = np.linspace(0, 4*np.pi, num_points)
        ph_values = 7.2 + 0.2 * np.sin(x) + np.random.normal(0, 0.05, num_points)
        
        # Ensure realistic bounds
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        return pd.DataFrame({'timestamp': timestamps, 'ph_value': ph_values})
    def test_advanced_prediction_endpoints(self):
        """Test 6: Advanced prediction endpoints for pattern following"""
        print("\n=== Testing Advanced Prediction Endpoints ===")
        
        if not self.data_id:
            print("❌ Cannot test advanced predictions - no data uploaded")
            self.test_results['advanced_predictions'] = False
            return
            
        try:
            # Test advanced prediction endpoint
            adv_response = self.session.post(
                f"{API_BASE_URL}/advanced-prediction",
                params={"data_id": self.data_id, "steps": 15}
            )
            
            if adv_response.status_code == 200:
                adv_data = adv_response.json()
                predictions = adv_data.get('predictions', [])
                
                if predictions:
                    print(f"✅ Advanced predictions generated: {len(predictions)} steps")
                    
                    # Check for realistic range
                    in_range = all(5.5 <= p <= 8.5 for p in predictions)
                    print(f"   Range check: {'✅' if in_range else '❌'} All in realistic range")
                    
                    # Check for downward bias
                    if len(predictions) >= 5:
                        slope = np.polyfit(range(len(predictions)), predictions, 1)[0]
                        print(f"   Trend slope: {slope:.6f}")
                        
                        if slope < -0.01:
                            print(f"❌ ADVANCED DOWNWARD BIAS: {slope:.6f}")
                            self.test_results['advanced_predictions'] = False
                        else:
                            print(f"✅ No significant downward bias in advanced predictions")
                            self.test_results['advanced_predictions'] = True
                    
                    # Check variability
                    pred_std = np.std(predictions)
                    if pred_std > 0.01:
                        print(f"✅ Good variability: std = {pred_std:.6f}")
                    else:
                        print(f"⚠️  Low variability: std = {pred_std:.6f}")
                        
                else:
                    print("❌ No advanced predictions returned")
                    self.test_results['advanced_predictions'] = False
                    
            else:
                print(f"❌ Advanced prediction failed: {adv_response.status_code} - {adv_response.text}")
                self.test_results['advanced_predictions'] = False
                
        except Exception as e:
            print(f"❌ Advanced prediction test error: {str(e)}")
            self.test_results['advanced_predictions'] = False
    
    def run_all_tests(self):
        """Run all pH prediction downward trend tests"""
        print("🧪 Starting Comprehensive pH Prediction Downward Trend Testing")
        print("=" * 70)
        
        # Run all tests in sequence
        self.test_ph_simulation_endpoints()
        self.test_upload_ph_data_and_train()
        self.test_single_prediction_bias()
        self.test_multiple_sequential_predictions()
        self.test_continuous_prediction_system()
        self.test_advanced_prediction_endpoints()
        self.test_pattern_following_capabilities()
        
        # Summary
        print("\n" + "=" * 70)
        print("🎯 pH PREDICTION DOWNWARD TREND TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        # Critical findings
        print("\n🔍 CRITICAL FINDINGS:")
        
        downward_bias_tests = [
            'single_prediction_bias',
            'sequential_bias', 
            'continuous_bias',
            'advanced_predictions'
        ]
        
        bias_issues = [test for test in downward_bias_tests if test in self.test_results and not self.test_results[test]]
        
        if bias_issues:
            print(f"❌ DOWNWARD BIAS DETECTED in: {', '.join(bias_issues)}")
            print("   The user's reported issue is CONFIRMED - predictions are trending downward")
        else:
            print("✅ NO DOWNWARD BIAS DETECTED in prediction algorithms")
            print("   The downward trend issue appears to be resolved")
        
        # Range and variability issues
        range_tests = ['ph_simulation_range', 'ph_history_range', 'prediction_range']
        range_issues = [test for test in range_tests if test in self.test_results and not self.test_results[test]]
        
        if range_issues:
            print(f"⚠️  RANGE ISSUES detected in: {', '.join(range_issues)}")
        
        variability_tests = ['ph_variability', 'prediction_variability']
        variability_issues = [test for test in variability_tests if test in self.test_results and not self.test_results[test]]
        
        if variability_issues:
            print(f"⚠️  VARIABILITY ISSUES detected in: {', '.join(variability_issues)}")
        
        return self.test_results

if __name__ == "__main__":
    tester = PhPredictionTester()
    results = tester.run_all_tests()