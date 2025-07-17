#!/usr/bin/env python3
"""
Enhanced Prediction System Testing - Focused on Existing Implementation
Tests the current prediction system improvements and identifies what needs to be enhanced
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://4f535dbd-21ac-4151-8dfe-215665939abd.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Enhanced Prediction System at: {API_BASE_URL}")

class FocusedPredictionTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_ph_data(self, samples=49):
        """Create realistic pH data for testing (matching review request)"""
        # Generate pH data in realistic range (6.0-8.0)
        base_ph = 7.0
        trend = np.linspace(0, 0.3, samples)  # Slight upward trend
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(samples) / 12)  # 12-sample cycle
        noise = np.random.normal(0, 0.05, samples)
        
        ph_values = base_ph + trend + seasonal + noise
        ph_values = np.clip(ph_values, 6.0, 8.0)  # Keep in realistic range
        
        # Create timestamps
        start_time = datetime.now() - timedelta(hours=samples)
        timestamps = [start_time + timedelta(hours=i) for i in range(samples)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ph_value': ph_values
        })
        
        return df
    
    def create_sine_wave_data(self, samples=100):
        """Create sine wave data for pattern testing"""
        x = np.linspace(0, 4 * np.pi, samples)
        y = np.sin(x) + 0.1 * np.random.normal(0, 1, samples)
        
        start_date = datetime.now() - timedelta(days=samples)
        timestamps = [start_date + timedelta(days=i) for i in range(samples)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': y
        })
        
        return df
    
    def upload_and_train(self, df, filename, time_col, target_col, model_type='lstm'):
        """Upload data and train model"""
        try:
            # Upload data
            csv_content = df.to_csv(index=False)
            files = {'file': (filename, csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                return None, None, f"Upload failed: {response.status_code}"
            
            data = response.json()
            data_id = data.get('data_id')
            
            # Train model
            training_params = {
                "time_column": time_col,
                "target_column": target_col,
                "seq_len": 20,
                "pred_len": 10,
                "epochs": 30,
                "batch_size": 8
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": model_type},
                json=training_params
            )
            
            if response.status_code != 200:
                return data_id, None, f"Training failed: {response.status_code} - {response.text}"
            
            model_data = response.json()
            model_id = model_data.get('model_id')
            
            return data_id, model_id, "Success"
            
        except Exception as e:
            return None, None, f"Exception: {str(e)}"
    
    def test_basic_prediction_functionality(self):
        """Test 1: Basic Prediction Functionality"""
        print("\n=== Testing Basic Prediction Functionality ===")
        
        # Test with pH data
        ph_data = self.create_ph_data(samples=49)
        data_id, model_id, status = self.upload_and_train(
            ph_data, "ph_test.csv", "timestamp", "ph_value", "lstm"
        )
        
        if model_id:
            print("✅ Data upload and model training successful")
            
            # Test basic prediction
            try:
                response = self.session.get(
                    f"{API_BASE_URL}/generate-prediction",
                    params={"model_id": model_id, "steps": 30}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    
                    if len(predictions) == 30:
                        # Check if predictions are in realistic pH range
                        valid_range = all(6.0 <= p <= 8.0 for p in predictions)
                        
                        if valid_range:
                            print("✅ Basic predictions generated successfully")
                            print(f"   Predictions in valid pH range: {valid_range}")
                            print(f"   Sample predictions: {predictions[:5]}")
                            self.test_results['basic_prediction'] = True
                        else:
                            print("❌ Predictions outside valid pH range")
                            self.test_results['basic_prediction'] = False
                    else:
                        print(f"❌ Wrong number of predictions: {len(predictions)}")
                        self.test_results['basic_prediction'] = False
                else:
                    print(f"❌ Prediction API failed: {response.status_code}")
                    self.test_results['basic_prediction'] = False
                    
            except Exception as e:
                print(f"❌ Prediction test error: {e}")
                self.test_results['basic_prediction'] = False
        else:
            print(f"❌ Setup failed: {status}")
            self.test_results['basic_prediction'] = False
    
    def test_advanced_model_training(self):
        """Test 2: Advanced Model Training"""
        print("\n=== Testing Advanced Model Training ===")
        
        ph_data = self.create_ph_data(samples=49)
        
        model_types = ['lstm', 'arima', 'prophet']
        training_results = []
        
        for model_type in model_types:
            print(f"\n--- Testing {model_type.upper()} Training ---")
            
            data_id, model_id, status = self.upload_and_train(
                ph_data, f"{model_type}_test.csv", "timestamp", "ph_value", model_type
            )
            
            if model_id:
                training_results.append((model_type, True, f"Model ID: {model_id}"))
                print(f"✅ {model_type.upper()} training successful")
            else:
                training_results.append((model_type, False, status))
                print(f"❌ {model_type.upper()} training failed: {status}")
        
        # Evaluate results
        passed_models = sum(1 for _, success, _ in training_results if success)
        total_models = len(training_results)
        
        print(f"\n📊 Advanced Model Training Results: {passed_models}/{total_models}")
        for model_type, success, details in training_results:
            status = "✅" if success else "❌"
            print(f"   {status} {model_type.upper()}: {details if not success else 'SUCCESS'}")
        
        self.test_results['advanced_model_training'] = passed_models >= 2
    
    def test_continuous_prediction_system(self):
        """Test 3: Continuous Prediction System"""
        print("\n=== Testing Continuous Prediction System ===")
        
        # Use sine wave data for better pattern testing
        sine_data = self.create_sine_wave_data(samples=80)
        data_id, model_id, status = self.upload_and_train(
            sine_data, "sine_continuous_test.csv", "timestamp", "value", "lstm"
        )
        
        if model_id:
            continuous_tests = []
            
            # Test 1: Generate continuous prediction
            try:
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 20, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    timestamps = data.get('timestamps', [])
                    
                    if len(predictions) == 20 and len(timestamps) == 20:
                        continuous_tests.append(("Generate Continuous", True, f"{len(predictions)} predictions"))
                        print("✅ Continuous prediction generation working")
                    else:
                        continuous_tests.append(("Generate Continuous", False, f"Wrong counts: {len(predictions)}, {len(timestamps)}"))
                else:
                    continuous_tests.append(("Generate Continuous", False, f"API error: {response.status_code}"))
                    
            except Exception as e:
                continuous_tests.append(("Generate Continuous", False, f"Exception: {str(e)}"))
            
            # Test 2: Reset functionality
            try:
                response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
                
                if response.status_code == 200:
                    continuous_tests.append(("Reset Continuous", True, "Reset successful"))
                    print("✅ Continuous prediction reset working")
                else:
                    continuous_tests.append(("Reset Continuous", False, f"Reset failed: {response.status_code}"))
                    
            except Exception as e:
                continuous_tests.append(("Reset Continuous", False, f"Exception: {str(e)}"))
            
            # Test 3: Start/Stop continuous prediction
            try:
                # Start
                response = self.session.post(f"{API_BASE_URL}/start-continuous-prediction")
                start_success = response.status_code == 200
                
                # Stop
                response = self.session.post(f"{API_BASE_URL}/stop-continuous-prediction")
                stop_success = response.status_code == 200
                
                if start_success and stop_success:
                    continuous_tests.append(("Start/Stop Control", True, "Control working"))
                    print("✅ Continuous prediction start/stop working")
                else:
                    continuous_tests.append(("Start/Stop Control", False, f"Start: {start_success}, Stop: {stop_success}"))
                    
            except Exception as e:
                continuous_tests.append(("Start/Stop Control", False, f"Exception: {str(e)}"))
            
            # Evaluate continuous prediction results
            passed_tests = sum(1 for _, success, _ in continuous_tests if success)
            total_tests = len(continuous_tests)
            
            print(f"\n📊 Continuous Prediction Results: {passed_tests}/{total_tests}")
            for test_name, success, details in continuous_tests:
                status = "✅" if success else "❌"
                print(f"   {status} {test_name}: {details}")
            
            self.test_results['continuous_prediction'] = passed_tests >= 2
        else:
            print(f"❌ Setup failed: {status}")
            self.test_results['continuous_prediction'] = False
    
    def test_ph_simulation_integration(self):
        """Test 4: pH Simulation Integration"""
        print("\n=== Testing pH Simulation Integration ===")
        
        ph_tests = []
        
        # Test 1: Real-time pH simulation
        try:
            response = self.session.get(f"{API_BASE_URL}/ph-simulation")
            
            if response.status_code == 200:
                data = response.json()
                ph_value = data.get('ph_value')
                confidence = data.get('confidence')
                
                if ph_value and 6.0 <= ph_value <= 8.0:
                    ph_tests.append(("Real-time pH", True, f"pH: {ph_value}, Confidence: {confidence}"))
                    print(f"✅ Real-time pH simulation working: {ph_value}")
                else:
                    ph_tests.append(("Real-time pH", False, f"Invalid pH: {ph_value}"))
            else:
                ph_tests.append(("Real-time pH", False, f"API error: {response.status_code}"))
                
        except Exception as e:
            ph_tests.append(("Real-time pH", False, f"Exception: {str(e)}"))
        
        # Test 2: pH simulation history
        try:
            response = self.session.get(f"{API_BASE_URL}/ph-simulation-history", params={"hours": 24})
            
            if response.status_code == 200:
                data = response.json()
                history_data = data.get('data', [])
                
                if len(history_data) > 0:
                    # Check first few pH values
                    valid_ph_count = sum(1 for point in history_data[:10] 
                                       if 6.0 <= point.get('ph_value', 0) <= 8.0)
                    
                    if valid_ph_count >= 8:  # At least 80% valid
                        ph_tests.append(("pH History", True, f"{len(history_data)} data points"))
                        print(f"✅ pH simulation history working: {len(history_data)} points")
                    else:
                        ph_tests.append(("pH History", False, f"Invalid pH values: {valid_ph_count}/10"))
                else:
                    ph_tests.append(("pH History", False, "No history data"))
            else:
                ph_tests.append(("pH History", False, f"API error: {response.status_code}"))
                
        except Exception as e:
            ph_tests.append(("pH History", False, f"Exception: {str(e)}"))
        
        # Test 3: pH target management
        try:
            # Set pH target
            response = self.session.post(
                f"{API_BASE_URL}/set-ph-target",
                json={"target_ph": 7.5}
            )
            
            if response.status_code == 200:
                ph_tests.append(("pH Target", True, "Target setting working"))
                print("✅ pH target management working")
            else:
                ph_tests.append(("pH Target", False, f"API error: {response.status_code}"))
                
        except Exception as e:
            ph_tests.append(("pH Target", False, f"Exception: {str(e)}"))
        
        # Evaluate pH simulation results
        passed_tests = sum(1 for _, success, _ in ph_tests if success)
        total_tests = len(ph_tests)
        
        print(f"\n📊 pH Simulation Results: {passed_tests}/{total_tests}")
        for test_name, success, details in ph_tests:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}: {details}")
        
        self.test_results['ph_simulation'] = passed_tests >= 2
    
    def test_data_preprocessing_quality(self):
        """Test 5: Data Preprocessing and Quality Validation"""
        print("\n=== Testing Data Preprocessing and Quality Validation ===")
        
        preprocessing_tests = []
        
        # Test with different data types
        test_datasets = [
            ("pH Data", self.create_ph_data(samples=50), "timestamp", "ph_value"),
            ("Sine Wave", self.create_sine_wave_data(samples=60), "timestamp", "value")
        ]
        
        for dataset_name, test_data, time_col, target_col in test_datasets:
            print(f"\n--- Testing {dataset_name} Preprocessing ---")
            
            try:
                # Upload data and check analysis
                csv_content = test_data.to_csv(index=False)
                files = {'file': (f"{dataset_name.lower().replace(' ', '_')}_preprocessing.csv", csv_content, 'text/csv')}
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    analysis = data.get('analysis', {})
                    
                    # Check if analysis detected columns correctly
                    time_columns = analysis.get('time_columns', [])
                    numeric_columns = analysis.get('numeric_columns', [])
                    
                    time_detected = time_col in time_columns
                    target_detected = target_col in numeric_columns
                    
                    if time_detected and target_detected:
                        preprocessing_tests.append((f"{dataset_name} Analysis", True, "Columns detected correctly"))
                        print(f"✅ {dataset_name}: Column detection working")
                        
                        # Test data quality report if available
                        try:
                            response = self.session.get(f"{API_BASE_URL}/data-quality-report")
                            
                            if response.status_code == 200:
                                quality_data = response.json()
                                quality_score = quality_data.get('quality_score', 0)
                                
                                if quality_score > 0:
                                    preprocessing_tests.append((f"{dataset_name} Quality", True, f"Quality score: {quality_score}"))
                                    print(f"✅ {dataset_name}: Quality validation working (score: {quality_score})")
                                else:
                                    preprocessing_tests.append((f"{dataset_name} Quality", False, "Zero quality score"))
                            else:
                                preprocessing_tests.append((f"{dataset_name} Quality", False, f"Quality API error: {response.status_code}"))
                                
                        except Exception as e:
                            preprocessing_tests.append((f"{dataset_name} Quality", False, f"Quality exception: {str(e)}"))
                    else:
                        preprocessing_tests.append((f"{dataset_name} Analysis", False, f"Time: {time_detected}, Target: {target_detected}"))
                else:
                    preprocessing_tests.append((f"{dataset_name} Analysis", False, f"Upload failed: {response.status_code}"))
                    
            except Exception as e:
                preprocessing_tests.append((f"{dataset_name} Analysis", False, f"Exception: {str(e)}"))
        
        # Evaluate preprocessing results
        passed_tests = sum(1 for _, success, _ in preprocessing_tests if success)
        total_tests = len(preprocessing_tests)
        
        print(f"\n📊 Data Preprocessing Results: {passed_tests}/{total_tests}")
        for test_name, success, details in preprocessing_tests:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}: {details}")
        
        self.test_results['data_preprocessing'] = passed_tests >= total_tests * 0.6
    
    def analyze_prediction_patterns(self):
        """Test 6: Analyze Prediction Patterns for Downward Trend Issues"""
        print("\n=== Analyzing Prediction Patterns ===")
        
        # Test with pH data to check for downward trend bias
        ph_data = self.create_ph_data(samples=49)
        data_id, model_id, status = self.upload_and_train(
            ph_data, "pattern_analysis.csv", "timestamp", "ph_value", "lstm"
        )
        
        if model_id:
            pattern_tests = []
            
            # Generate multiple prediction sets to check for bias
            prediction_sets = []
            
            for i in range(5):
                try:
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-prediction",
                        params={"model_id": model_id, "steps": 20}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        predictions = data.get('predictions', [])
                        if len(predictions) == 20:
                            prediction_sets.append(predictions)
                    
                    time.sleep(0.5)  # Small delay between calls
                    
                except Exception as e:
                    print(f"Error in prediction set {i}: {e}")
            
            if len(prediction_sets) >= 3:
                # Analyze for downward trend bias
                trend_analysis = []
                
                for i, predictions in enumerate(prediction_sets):
                    # Calculate trend slope
                    x = np.arange(len(predictions))
                    slope, _ = np.polyfit(x, predictions, 1)
                    trend_analysis.append(slope)
                    
                    print(f"   Prediction set {i+1}: slope = {slope:.6f}, range = {min(predictions):.3f} - {max(predictions):.3f}")
                
                # Check if there's consistent downward bias
                avg_slope = np.mean(trend_analysis)
                downward_bias = avg_slope < -0.01  # Significant downward trend
                
                if not downward_bias:
                    pattern_tests.append(("Downward Trend Bias", True, f"Average slope: {avg_slope:.6f}"))
                    print(f"✅ No significant downward trend bias detected (avg slope: {avg_slope:.6f})")
                else:
                    pattern_tests.append(("Downward Trend Bias", False, f"Downward bias detected: {avg_slope:.6f}"))
                    print(f"❌ Downward trend bias detected (avg slope: {avg_slope:.6f})")
                
                # Check prediction variability
                all_predictions = np.concatenate(prediction_sets)
                prediction_std = np.std(all_predictions)
                
                if prediction_std > 0.01:  # Some variability
                    pattern_tests.append(("Prediction Variability", True, f"Std: {prediction_std:.6f}"))
                    print(f"✅ Predictions show good variability (std: {prediction_std:.6f})")
                else:
                    pattern_tests.append(("Prediction Variability", False, f"Low variability: {prediction_std:.6f}"))
                
                # Check pH range maintenance
                valid_ph_ratio = np.mean([6.0 <= p <= 8.0 for p in all_predictions])
                
                if valid_ph_ratio >= 0.95:
                    pattern_tests.append(("pH Range Maintenance", True, f"Valid ratio: {valid_ph_ratio:.3f}"))
                    print(f"✅ pH range well maintained ({valid_ph_ratio:.1%} valid)")
                else:
                    pattern_tests.append(("pH Range Maintenance", False, f"Poor range: {valid_ph_ratio:.3f}"))
            else:
                pattern_tests.append(("Pattern Analysis", False, f"Insufficient prediction sets: {len(prediction_sets)}"))
            
            # Evaluate pattern analysis results
            passed_tests = sum(1 for _, success, _ in pattern_tests if success)
            total_tests = len(pattern_tests)
            
            print(f"\n📊 Pattern Analysis Results: {passed_tests}/{total_tests}")
            for test_name, success, details in pattern_tests:
                status = "✅" if success else "❌"
                print(f"   {status} {test_name}: {details}")
            
            self.test_results['pattern_analysis'] = passed_tests >= total_tests * 0.67
        else:
            print(f"❌ Setup failed: {status}")
            self.test_results['pattern_analysis'] = False
    
    def run_all_tests(self):
        """Run all focused prediction tests"""
        print("🚀 Starting Focused Enhanced Prediction System Testing")
        print("=" * 60)
        
        # Run all test categories
        self.test_basic_prediction_functionality()
        self.test_advanced_model_training()
        self.test_continuous_prediction_system()
        self.test_ph_simulation_integration()
        self.test_data_preprocessing_quality()
        self.analyze_prediction_patterns()
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("🎯 ENHANCED PREDICTION SYSTEM TEST REPORT")
        print("=" * 60)
        
        test_categories = [
            ("Basic Prediction Functionality", self.test_results.get('basic_prediction', False)),
            ("Advanced Model Training", self.test_results.get('advanced_model_training', False)),
            ("Continuous Prediction System", self.test_results.get('continuous_prediction', False)),
            ("pH Simulation Integration", self.test_results.get('ph_simulation', False)),
            ("Data Preprocessing Quality", self.test_results.get('data_preprocessing', False)),
            ("Pattern Analysis (Downward Trend)", self.test_results.get('pattern_analysis', False))
        ]
        
        passed_categories = sum(1 for _, passed in test_categories if passed)
        total_categories = len(test_categories)
        
        print(f"\n📊 OVERALL RESULTS: {passed_categories}/{total_categories} categories passed")
        print(f"Success Rate: {(passed_categories/total_categories)*100:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        for category, passed in test_categories:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {status} {category}")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS:")
        
        if self.test_results.get('basic_prediction', False):
            print("   ✅ Basic prediction functionality is working")
        else:
            print("   ❌ Basic prediction functionality has issues")
        
        if self.test_results.get('pattern_analysis', False):
            print("   ✅ No significant downward trend bias detected")
        else:
            print("   ⚠️  Potential downward trend bias or pattern issues")
        
        if self.test_results.get('continuous_prediction', False):
            print("   ✅ Continuous prediction system is functional")
        else:
            print("   ❌ Continuous prediction system needs attention")
        
        # Overall assessment
        if passed_categories >= total_categories * 0.8:
            print(f"\n🎉 OVERALL ASSESSMENT: EXCELLENT")
            print("   Current prediction system is working well")
        elif passed_categories >= total_categories * 0.6:
            print(f"\n⚠️  OVERALL ASSESSMENT: GOOD")
            print("   Current prediction system is mostly working but could be enhanced")
        else:
            print(f"\n❌ OVERALL ASSESSMENT: NEEDS IMPROVEMENT")
            print("   Current prediction system requires significant enhancements")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if not self.test_results.get('pattern_analysis', False):
            print("   • Implement enhanced prediction system to address pattern issues")
            print("   • Add bias correction mechanisms to prevent downward trends")
        
        if not self.test_results.get('continuous_prediction', False):
            print("   • Fix continuous prediction extrapolation issues")
            print("   • Improve smooth transition between predictions")
        
        if passed_categories < total_categories * 0.8:
            print("   • Consider implementing the EnhancedTimeSeriesPredictor class")
            print("   • Add comprehensive pattern analysis for better predictions")
            print("   • Implement statistical property preservation mechanisms")
        
        print("\n" + "=" * 60)


def main():
    """Main test execution"""
    tester = FocusedPredictionTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()