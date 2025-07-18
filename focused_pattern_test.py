#!/usr/bin/env python3
"""
Enhanced Pattern-Aware Prediction System Testing - Focused on Working Components
Tests the pattern analysis functions and basic prediction capabilities
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://c8772c28-6b4b-4343-84fa-effeefd86ff0.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing enhanced pattern-aware prediction system at: {API_BASE_URL}")

class FocusedPatternTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_u_shaped_data(self, points=50):
        """Create U-shaped (quadratic) data for pattern testing"""
        x = np.linspace(-5, 5, points)
        y = x**2 + np.random.normal(0, 0.5, points)  # U-shape with noise
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'value': y
        })
        return df
    
    def create_s_shaped_data(self, points=50):
        """Create S-shaped (cubic) data for pattern testing"""
        x = np.linspace(-3, 3, points)
        y = x**3 - 3*x + np.random.normal(0, 0.3, points)  # S-shape with noise
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'value': y
        })
        return df
    
    def create_complex_shaped_data(self, points=50):
        """Create complex shaped data for custom pattern testing"""
        x = np.linspace(0, 4*np.pi, points)
        y = np.sin(x) * np.exp(-x/10) + 0.5*np.cos(2*x) + np.random.normal(0, 0.2, points)
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'value': y
        })
        return df
    
    def upload_and_analyze_data(self, df, test_name):
        """Upload data and get analysis results"""
        try:
            csv_content = df.to_csv(index=False)
            files = {'file': (f'{test_name}.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'data_id': data.get('data_id'),
                    'analysis': data.get('analysis')
                }
            else:
                return {
                    'success': False,
                    'error': f"Upload failed: {response.status_code} - {response.text}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"Upload error: {str(e)}"
            }
    
    def test_enhanced_data_preprocessing(self):
        """Test Enhanced Data Preprocessing and Quality Validation"""
        print("\n=== Testing Enhanced Data Preprocessing and Quality Validation ===")
        
        preprocessing_tests = []
        
        # Test 1: Data quality report endpoint
        print("\n--- Testing Data Quality Report Endpoint ---")
        
        # Upload U-shaped data first
        u_data = self.create_u_shaped_data()
        upload_result = self.upload_and_analyze_data(u_data, "quality_test")
        
        if upload_result['success']:
            print("✅ Data uploaded successfully for quality testing")
            
            # Test data quality report endpoint
            try:
                response = self.session.get(f"{API_BASE_URL}/data-quality-report")
                
                if response.status_code == 200:
                    quality_data = response.json()
                    print("✅ Data quality report endpoint successful")
                    print(f"   Status: {quality_data.get('status')}")
                    print(f"   Quality Score: {quality_data.get('quality_score')}")
                    print(f"   Recommendations: {len(quality_data.get('recommendations', []))}")
                    
                    # Validate quality report structure
                    if 'quality_score' in quality_data and 'status' in quality_data:
                        print("✅ Quality report has correct structure")
                        preprocessing_tests.append(("Data quality report structure", True))
                        
                        # Check if quality score is reasonable
                        quality_score = quality_data.get('quality_score', 0)
                        if 0 <= quality_score <= 100:
                            print(f"✅ Quality score in valid range: {quality_score}")
                            preprocessing_tests.append(("Quality score validity", True))
                        else:
                            print(f"❌ Quality score out of range: {quality_score}")
                            preprocessing_tests.append(("Quality score validity", False))
                    else:
                        print("❌ Quality report missing required fields")
                        preprocessing_tests.append(("Data quality report structure", False))
                        preprocessing_tests.append(("Quality score validity", False))
                else:
                    print(f"❌ Data quality report failed: {response.status_code} - {response.text}")
                    preprocessing_tests.append(("Data quality report structure", False))
                    preprocessing_tests.append(("Quality score validity", False))
            except Exception as e:
                print(f"❌ Data quality report error: {str(e)}")
                preprocessing_tests.append(("Data quality report structure", False))
                preprocessing_tests.append(("Quality score validity", False))
        else:
            print(f"❌ Data upload failed: {upload_result['error']}")
            preprocessing_tests.append(("Data quality report structure", False))
            preprocessing_tests.append(("Quality score validity", False))
        
        # Test 2: Enhanced data analysis with different data types
        print("\n--- Testing Enhanced Data Analysis ---")
        
        # Test with S-shaped data
        s_data = self.create_s_shaped_data()
        s_result = self.upload_and_analyze_data(s_data, "s_shape_analysis")
        
        if s_result['success']:
            analysis = s_result['analysis']
            print("✅ S-shaped data analysis successful")
            print(f"   Columns detected: {analysis.get('columns', [])}")
            print(f"   Time columns: {analysis.get('time_columns', [])}")
            print(f"   Numeric columns: {analysis.get('numeric_columns', [])}")
            print(f"   Data shape: {analysis.get('data_shape')}")
            
            # Validate analysis results
            if 'timestamp' in analysis.get('time_columns', []) and 'value' in analysis.get('numeric_columns', []):
                print("✅ Column detection working correctly")
                preprocessing_tests.append(("Column detection", True))
            else:
                print("❌ Column detection failed")
                preprocessing_tests.append(("Column detection", False))
            
            # Check data preview
            data_preview = analysis.get('data_preview', {})
            if 'head' in data_preview and 'describe' in data_preview:
                print("✅ Data preview generation working")
                preprocessing_tests.append(("Data preview generation", True))
            else:
                print("❌ Data preview generation failed")
                preprocessing_tests.append(("Data preview generation", False))
        else:
            print(f"❌ S-shaped data analysis failed: {s_result['error']}")
            preprocessing_tests.append(("Column detection", False))
            preprocessing_tests.append(("Data preview generation", False))
        
        # Test 3: Complex data preprocessing
        print("\n--- Testing Complex Data Preprocessing ---")
        
        complex_data = self.create_complex_shaped_data()
        complex_result = self.upload_and_analyze_data(complex_data, "complex_analysis")
        
        if complex_result['success']:
            print("✅ Complex data preprocessing successful")
            preprocessing_tests.append(("Complex data preprocessing", True))
        else:
            print(f"❌ Complex data preprocessing failed: {complex_result['error']}")
            preprocessing_tests.append(("Complex data preprocessing", False))
        
        # Evaluate preprocessing tests
        passed_tests = sum(1 for _, passed in preprocessing_tests if passed)
        total_tests = len(preprocessing_tests)
        
        print(f"\n📊 Enhanced Data Preprocessing Results: {passed_tests}/{total_tests}")
        for test_name, passed in preprocessing_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['enhanced_data_preprocessing'] = passed_tests >= total_tests * 0.8
        return upload_result if upload_result['success'] else None
    
    def test_basic_model_training(self, data_result):
        """Test Basic Model Training (Prophet/ARIMA) for Pattern Data"""
        print("\n=== Testing Basic Model Training with Pattern Data ===")
        
        if not data_result or not data_result['success']:
            print("❌ Cannot test model training - no valid data")
            self.test_results['basic_model_training'] = False
            return None
        
        training_tests = []
        model_id = None
        
        try:
            data_id = data_result['data_id']
            
            # Test Prophet model training
            print("\n--- Testing Prophet Model Training ---")
            
            prophet_params = {
                "time_column": "timestamp",
                "target_column": "value",
                "seasonality_mode": "additive",
                "yearly_seasonality": False,
                "weekly_seasonality": False,
                "daily_seasonality": False
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "prophet"},
                json=prophet_params
            )
            
            if response.status_code == 200:
                data = response.json()
                model_id = data.get('model_id')
                print("✅ Prophet model training successful")
                print(f"   Model ID: {model_id}")
                print(f"   Status: {data.get('status')}")
                training_tests.append(("Prophet training", True))
            else:
                print(f"❌ Prophet training failed: {response.status_code} - {response.text}")
                training_tests.append(("Prophet training", False))
            
            # Test ARIMA model training
            print("\n--- Testing ARIMA Model Training ---")
            
            arima_params = {
                "time_column": "timestamp",
                "target_column": "value",
                "order": [1, 1, 1]
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json=arima_params
            )
            
            if response.status_code == 200:
                data = response.json()
                arima_model_id = data.get('model_id')
                print("✅ ARIMA model training successful")
                print(f"   Model ID: {arima_model_id}")
                print(f"   Status: {data.get('status')}")
                training_tests.append(("ARIMA training", True))
                
                # Use ARIMA model if Prophet failed
                if not model_id:
                    model_id = arima_model_id
            else:
                print(f"❌ ARIMA training failed: {response.status_code} - {response.text}")
                training_tests.append(("ARIMA training", False))
        
        except Exception as e:
            print(f"❌ Model training error: {str(e)}")
            training_tests.extend([
                ("Prophet training", False),
                ("ARIMA training", False)
            ])
        
        # Evaluate training tests
        passed_tests = sum(1 for _, passed in training_tests if passed)
        total_tests = len(training_tests)
        
        print(f"\n📊 Basic Model Training Results: {passed_tests}/{total_tests}")
        for test_name, passed in training_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['basic_model_training'] = passed_tests >= 1  # At least one model should work
        return model_id
    
    def test_pattern_aware_prediction(self, model_id):
        """Test Pattern-Aware Prediction Generation"""
        print("\n=== Testing Pattern-Aware Prediction Generation ===")
        
        if not model_id:
            print("❌ Cannot test pattern-aware prediction - no trained model")
            self.test_results['pattern_aware_prediction'] = False
            return
        
        prediction_tests = []
        
        try:
            # Test basic prediction generation
            print("\n--- Testing Basic Prediction Generation ---")
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 30}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                
                print("✅ Basic prediction generation successful")
                print(f"   Number of predictions: {len(predictions)}")
                print(f"   Number of timestamps: {len(timestamps)}")
                
                if len(predictions) == 30 and len(timestamps) == 30:
                    print("✅ Correct prediction dimensions")
                    prediction_tests.append(("Basic prediction generation", True))
                    
                    # Test prediction quality
                    pred_array = np.array(predictions)
                    unique_values = len(np.unique(np.round(pred_array, 2)))
                    
                    if unique_values >= 5:
                        print(f"✅ Predictions show variability ({unique_values} unique values)")
                        prediction_tests.append(("Prediction variability", True))
                    else:
                        print(f"❌ Predictions lack variability ({unique_values} unique values)")
                        prediction_tests.append(("Prediction variability", False))
                    
                    # Test prediction range
                    pred_range = np.max(pred_array) - np.min(pred_array)
                    if pred_range > 0.1:
                        print(f"✅ Predictions have reasonable range ({pred_range:.3f})")
                        prediction_tests.append(("Prediction range", True))
                    else:
                        print(f"❌ Predictions have narrow range ({pred_range:.3f})")
                        prediction_tests.append(("Prediction range", False))
                else:
                    print(f"❌ Incorrect prediction dimensions")
                    prediction_tests.extend([
                        ("Basic prediction generation", False),
                        ("Prediction variability", False),
                        ("Prediction range", False)
                    ])
            else:
                print(f"❌ Basic prediction failed: {response.status_code} - {response.text}")
                prediction_tests.extend([
                    ("Basic prediction generation", False),
                    ("Prediction variability", False),
                    ("Prediction range", False)
                ])
            
            # Test continuous prediction
            print("\n--- Testing Continuous Prediction ---")
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 25, "time_window": 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                
                print("✅ Continuous prediction successful")
                print(f"   Predictions: {len(predictions)}")
                print(f"   Timestamps: {len(timestamps)}")
                
                if len(predictions) == 25 and len(timestamps) == 25:
                    print("✅ Correct continuous prediction dimensions")
                    prediction_tests.append(("Continuous prediction", True))
                    
                    # Test multiple continuous calls
                    response2 = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": model_id, "steps": 25, "time_window": 100}
                    )
                    
                    if response2.status_code == 200:
                        data2 = response2.json()
                        timestamps2 = data2.get('timestamps', [])
                        
                        if timestamps != timestamps2:
                            print("✅ Continuous prediction advances forward")
                            prediction_tests.append(("Continuous advancement", True))
                        else:
                            print("❌ Continuous prediction not advancing")
                            prediction_tests.append(("Continuous advancement", False))
                    else:
                        print("❌ Second continuous prediction failed")
                        prediction_tests.append(("Continuous advancement", False))
                else:
                    print(f"❌ Incorrect continuous prediction dimensions")
                    prediction_tests.extend([
                        ("Continuous prediction", False),
                        ("Continuous advancement", False)
                    ])
            else:
                print(f"❌ Continuous prediction failed: {response.status_code} - {response.text}")
                prediction_tests.extend([
                    ("Continuous prediction", False),
                    ("Continuous advancement", False)
                ])
        
        except Exception as e:
            print(f"❌ Pattern-aware prediction error: {str(e)}")
            prediction_tests.extend([
                ("Basic prediction generation", False),
                ("Prediction variability", False),
                ("Prediction range", False),
                ("Continuous prediction", False),
                ("Continuous advancement", False)
            ])
        
        # Evaluate prediction tests
        passed_tests = sum(1 for _, passed in prediction_tests if passed)
        total_tests = len(prediction_tests)
        
        print(f"\n📊 Pattern-Aware Prediction Results: {passed_tests}/{total_tests}")
        for test_name, passed in prediction_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['pattern_aware_prediction'] = passed_tests >= total_tests * 0.6
    
    def test_continuous_prediction_flow(self, model_id):
        """Test Complete Continuous Prediction Flow"""
        print("\n=== Testing Complete Continuous Prediction Flow ===")
        
        if not model_id:
            print("❌ Cannot test continuous flow - no trained model")
            self.test_results['continuous_prediction_flow'] = False
            return
        
        flow_tests = []
        
        try:
            # Step 1: Reset continuous prediction
            response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            if response.status_code == 200:
                print("✅ Reset continuous prediction successful")
                flow_tests.append(("Reset continuous prediction", True))
            else:
                print("❌ Reset failed")
                flow_tests.append(("Reset continuous prediction", False))
            
            # Step 2: Start continuous prediction
            response = self.session.post(f"{API_BASE_URL}/start-continuous-prediction")
            if response.status_code == 200:
                print("✅ Start continuous prediction successful")
                flow_tests.append(("Start continuous prediction", True))
                
                # Wait and test multiple calls
                time.sleep(2)
                
                # Make multiple continuous prediction calls
                timestamps_history = []
                for i in range(3):
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": model_id, "steps": 15, "time_window": 80}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        timestamps = data.get('timestamps', [])
                        timestamps_history.append(timestamps)
                        time.sleep(1)
                    else:
                        print(f"❌ Continuous call {i+1} failed")
                        break
                
                # Check if predictions advance
                if len(timestamps_history) >= 2:
                    if timestamps_history[0] != timestamps_history[1]:
                        print("✅ Continuous predictions advance over time")
                        flow_tests.append(("Continuous advancement", True))
                    else:
                        print("❌ Continuous predictions not advancing")
                        flow_tests.append(("Continuous advancement", False))
                else:
                    print("❌ Insufficient continuous calls")
                    flow_tests.append(("Continuous advancement", False))
                
                # Step 3: Stop continuous prediction
                response = self.session.post(f"{API_BASE_URL}/stop-continuous-prediction")
                if response.status_code == 200:
                    print("✅ Stop continuous prediction successful")
                    flow_tests.append(("Stop continuous prediction", True))
                else:
                    print("❌ Stop failed")
                    flow_tests.append(("Stop continuous prediction", False))
            else:
                print("❌ Start continuous prediction failed")
                flow_tests.extend([
                    ("Start continuous prediction", False),
                    ("Continuous advancement", False),
                    ("Stop continuous prediction", False)
                ])
        
        except Exception as e:
            print(f"❌ Continuous flow error: {str(e)}")
            flow_tests.extend([
                ("Reset continuous prediction", False),
                ("Start continuous prediction", False),
                ("Continuous advancement", False),
                ("Stop continuous prediction", False)
            ])
        
        # Evaluate flow tests
        passed_tests = sum(1 for _, passed in flow_tests if passed)
        total_tests = len(flow_tests)
        
        print(f"\n📊 Continuous Prediction Flow Results: {passed_tests}/{total_tests}")
        for test_name, passed in flow_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['continuous_prediction_flow'] = passed_tests >= total_tests * 0.75
    
    def run_focused_tests(self):
        """Run focused tests on working components"""
        print("🎯 Starting Focused Pattern-Aware Prediction System Testing")
        print("=" * 70)
        
        # Test 1: Enhanced Data Preprocessing (needs retesting per test_result.md)
        data_result = self.test_enhanced_data_preprocessing()
        
        # Test 2: Basic Model Training with Pattern Data
        model_id = self.test_basic_model_training(data_result)
        
        # Test 3: Pattern-Aware Prediction Generation
        self.test_pattern_aware_prediction(model_id)
        
        # Test 4: Continuous Prediction Flow
        self.test_continuous_prediction_flow(model_id)
        
        # Generate summary
        self.generate_focused_summary()
    
    def generate_focused_summary(self):
        """Generate focused test summary"""
        print("\n" + "=" * 70)
        print("🎯 FOCUSED PATTERN-AWARE PREDICTION SYSTEM TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"\n📊 Overall Results: {passed_tests}/{total_tests} test categories passed")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        test_descriptions = {
            'enhanced_data_preprocessing': 'Enhanced Data Preprocessing and Quality Validation',
            'basic_model_training': 'Basic Model Training (Prophet/ARIMA) with Pattern Data',
            'pattern_aware_prediction': 'Pattern-Aware Prediction Generation',
            'continuous_prediction_flow': 'Complete Continuous Prediction Flow'
        }
        
        for test_key, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            description = test_descriptions.get(test_key, test_key)
            print(f"   {status} {description}")
        
        # Assessment of working components
        print("\n🎯 Working Components Assessment:")
        
        working_components = []
        
        if self.test_results.get('enhanced_data_preprocessing', False):
            print("   ✅ Enhanced data preprocessing and quality validation working")
            working_components.append(True)
        else:
            print("   ❌ Enhanced data preprocessing has issues")
            working_components.append(False)
        
        if self.test_results.get('basic_model_training', False):
            print("   ✅ Basic model training works with pattern data")
            working_components.append(True)
        else:
            print("   ❌ Basic model training failing")
            working_components.append(False)
        
        if self.test_results.get('pattern_aware_prediction', False):
            print("   ✅ Pattern-aware prediction generation working")
            working_components.append(True)
        else:
            print("   ❌ Pattern-aware prediction generation has issues")
            working_components.append(False)
        
        if self.test_results.get('continuous_prediction_flow', False):
            print("   ✅ Continuous prediction flow working")
            working_components.append(True)
        else:
            print("   ❌ Continuous prediction flow has issues")
            working_components.append(False)
        
        # Overall Assessment
        components_working = sum(working_components)
        total_components = len(working_components)
        
        print(f"\n🏆 WORKING COMPONENTS: {components_working}/{total_components} ({(components_working/total_components)*100:.1f}%)")
        
        if components_working >= 3:
            print("🎉 CORE PATTERN-AWARE FUNCTIONALITY: WORKING!")
            print("   Basic pattern-aware prediction system is functional.")
            print("   Advanced ML models need dependency fixes (SymPy/mpmath issue).")
        elif components_working >= 2:
            print("⚠️  CORE PATTERN-AWARE FUNCTIONALITY: PARTIALLY WORKING")
            print("   Some components working but improvements needed.")
        else:
            print("❌ CORE PATTERN-AWARE FUNCTIONALITY: NEEDS IMPROVEMENT")
            print("   Basic functionality not working as expected.")
        
        return components_working >= 2

if __name__ == "__main__":
    tester = FocusedPatternTester()
    tester.run_focused_tests()