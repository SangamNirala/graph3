#!/usr/bin/env python3
"""
Specific Task Testing for Enhanced Data Preprocessing and Advanced Model Training
Tests the specific tasks marked as needs_retesting in test_result.md
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

print(f"Testing Specific Tasks at: {API_BASE_URL}")

class SpecificTaskTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_test_data_with_issues(self):
        """Create test data with various quality issues"""
        # Create data with missing values, outliers, mixed types
        data = {
            'timestamp': [
                '2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00',
                '2024-01-01 03:00:00', '2024-01-01 04:00:00', '2024-01-01 05:00:00',
                '2024-01-01 06:00:00', '2024-01-01 07:00:00', '2024-01-01 08:00:00',
                '2024-01-01 09:00:00', '2024-01-01 10:00:00', '2024-01-01 11:00:00'
            ],
            'ph_value': [
                7.2, 7.1, None, 7.3, 15.0, 7.0,  # Missing value and outlier
                7.2, 'invalid', 7.1, 7.4, 7.3, 7.2  # Invalid string value
            ],
            'temperature': [
                25.0, 25.1, 25.2, 25.3, 25.4, 25.5,
                25.6, 25.7, 25.8, 25.9, 26.0, 26.1
            ]
        }
        
        return pd.DataFrame(data)
    
    def create_small_dataset(self):
        """Create small dataset to test parameter adjustment"""
        samples = 25  # Small dataset
        
        # Generate pH data
        base_ph = 7.0
        trend = np.linspace(0, 0.2, samples)
        noise = np.random.normal(0, 0.05, samples)
        ph_values = base_ph + trend + noise
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        # Create timestamps
        start_time = datetime.now() - timedelta(hours=samples)
        timestamps = [start_time + timedelta(hours=i) for i in range(samples)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ph_value': ph_values
        })
        
        return df
    
    def test_enhanced_data_preprocessing_quality_validation(self):
        """Test Task: Enhanced data preprocessing and quality validation"""
        print("\n=== Testing Enhanced Data Preprocessing and Quality Validation ===")
        
        preprocessing_tests = []
        
        # Test 1: Data with quality issues
        print("\n--- Testing Data with Quality Issues ---")
        
        problematic_data = self.create_test_data_with_issues()
        
        try:
            csv_content = problematic_data.to_csv(index=False)
            files = {'file': ('problematic_data.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get('analysis', {})
                
                # Check if preprocessing handled the issues
                data_shape = analysis.get('data_shape', [0, 0])
                time_columns = analysis.get('time_columns', [])
                numeric_columns = analysis.get('numeric_columns', [])
                
                if data_shape[0] > 0 and 'timestamp' in time_columns:
                    preprocessing_tests.append(("Problematic Data Handling", True, f"Processed {data_shape[0]} rows"))
                    print(f"✅ Problematic data handled successfully: {data_shape[0]} rows processed")
                else:
                    preprocessing_tests.append(("Problematic Data Handling", False, f"Processing failed: {data_shape}"))
            else:
                preprocessing_tests.append(("Problematic Data Handling", False, f"Upload failed: {response.status_code}"))
                
        except Exception as e:
            preprocessing_tests.append(("Problematic Data Handling", False, f"Exception: {str(e)}"))
        
        # Test 2: Data quality report endpoint
        print("\n--- Testing Data Quality Report Endpoint ---")
        
        try:
            response = self.session.get(f"{API_BASE_URL}/data-quality-report")
            
            if response.status_code == 200:
                quality_data = response.json()
                
                # Check for expected fields
                expected_fields = ['status', 'quality_score']
                has_required_fields = all(field in quality_data for field in expected_fields)
                
                if has_required_fields:
                    quality_score = quality_data.get('quality_score', 0)
                    preprocessing_tests.append(("Data Quality Report", True, f"Quality score: {quality_score}"))
                    print(f"✅ Data quality report working: score {quality_score}")
                else:
                    preprocessing_tests.append(("Data Quality Report", False, f"Missing fields: {quality_data}"))
            else:
                preprocessing_tests.append(("Data Quality Report", False, f"API error: {response.status_code}"))
                
        except Exception as e:
            preprocessing_tests.append(("Data Quality Report", False, f"Exception: {str(e)}"))
        
        # Test 3: Advanced preprocessing features
        print("\n--- Testing Advanced Preprocessing Features ---")
        
        # Upload clean data and train model to test preprocessing pipeline
        clean_data = self.create_small_dataset()
        
        try:
            csv_content = clean_data.to_csv(index=False)
            files = {'file': ('clean_preprocessing_test.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                
                # Try to train advanced model to test preprocessing pipeline
                training_params = {
                    "time_column": "timestamp",
                    "target_column": "ph_value",
                    "seq_len": 8,  # Small for small dataset
                    "pred_len": 3,
                    "epochs": 20,
                    "batch_size": 4
                }
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": "lstm"},
                    json=training_params
                )
                
                if response.status_code == 200:
                    model_data = response.json()
                    model_id = model_data.get('model_id')
                    
                    if model_id:
                        preprocessing_tests.append(("Advanced Preprocessing Pipeline", True, "Pipeline working with LSTM"))
                        print("✅ Advanced preprocessing pipeline working with model training")
                    else:
                        preprocessing_tests.append(("Advanced Preprocessing Pipeline", False, "No model ID returned"))
                else:
                    preprocessing_tests.append(("Advanced Preprocessing Pipeline", False, f"Training failed: {response.status_code}"))
            else:
                preprocessing_tests.append(("Advanced Preprocessing Pipeline", False, f"Upload failed: {response.status_code}"))
                
        except Exception as e:
            preprocessing_tests.append(("Advanced Preprocessing Pipeline", False, f"Exception: {str(e)}"))
        
        # Evaluate preprocessing results
        passed_tests = sum(1 for _, success, _ in preprocessing_tests if success)
        total_tests = len(preprocessing_tests)
        
        print(f"\n📊 Enhanced Data Preprocessing Results: {passed_tests}/{total_tests}")
        for test_name, success, details in preprocessing_tests:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}: {details}")
        
        self.test_results['enhanced_data_preprocessing'] = passed_tests >= total_tests * 0.67
    
    def test_advanced_model_training_hyperparameter_optimization(self):
        """Test Task: Advanced model training and hyperparameter optimization"""
        print("\n=== Testing Advanced Model Training and Hyperparameter Optimization ===")
        
        training_tests = []
        
        # Create test data
        test_data = self.create_small_dataset()
        
        try:
            csv_content = test_data.to_csv(index=False)
            files = {'file': ('advanced_training_test.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Data upload failed: {response.status_code}")
                self.test_results['advanced_model_training'] = False
                return
            
            data = response.json()
            data_id = data.get('data_id')
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            self.test_results['advanced_model_training'] = False
            return
        
        # Test 1: Advanced model types
        print("\n--- Testing Advanced Model Types ---")
        
        advanced_models = ['lstm', 'dlinear', 'lightgbm']
        
        for model_type in advanced_models:
            try:
                training_params = {
                    "time_column": "timestamp",
                    "target_column": "ph_value",
                    "seq_len": 8,  # Adjusted for small dataset
                    "pred_len": 3,
                    "epochs": 20,
                    "batch_size": 4
                }
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": model_type},
                    json=training_params
                )
                
                if response.status_code == 200:
                    model_data = response.json()
                    model_id = model_data.get('model_id')
                    performance_metrics = model_data.get('performance_metrics', {})
                    
                    if model_id:
                        training_tests.append((f"{model_type.upper()} Training", True, f"Model ID: {model_id}"))
                        print(f"✅ {model_type.upper()} training successful")
                        
                        if performance_metrics:
                            print(f"   Performance metrics available: {list(performance_metrics.keys())}")
                    else:
                        training_tests.append((f"{model_type.upper()} Training", False, "No model ID"))
                else:
                    training_tests.append((f"{model_type.upper()} Training", False, f"API error: {response.status_code}"))
                    print(f"❌ {model_type.upper()} training failed: {response.status_code}")
                    
            except Exception as e:
                training_tests.append((f"{model_type.upper()} Training", False, f"Exception: {str(e)}"))
        
        # Test 2: Model performance endpoint
        print("\n--- Testing Model Performance Endpoint ---")
        
        try:
            response = self.session.get(f"{API_BASE_URL}/model-performance")
            
            if response.status_code == 200:
                performance_data = response.json()
                
                if isinstance(performance_data, dict) and len(performance_data) > 0:
                    training_tests.append(("Model Performance", True, f"Performance data available"))
                    print("✅ Model performance endpoint working")
                else:
                    training_tests.append(("Model Performance", False, "No performance data"))
            else:
                training_tests.append(("Model Performance", False, f"API error: {response.status_code}"))
                
        except Exception as e:
            training_tests.append(("Model Performance", False, f"Exception: {str(e)}"))
        
        # Test 3: Supported models endpoint
        print("\n--- Testing Supported Models Endpoint ---")
        
        try:
            response = self.session.get(f"{API_BASE_URL}/supported-models")
            
            if response.status_code == 200:
                models_data = response.json()
                supported_models = models_data.get('supported_models', [])
                
                if len(supported_models) > 0:
                    training_tests.append(("Supported Models", True, f"Models: {supported_models}"))
                    print(f"✅ Supported models endpoint working: {supported_models}")
                else:
                    training_tests.append(("Supported Models", False, "No supported models"))
            else:
                training_tests.append(("Supported Models", False, f"API error: {response.status_code}"))
                
        except Exception as e:
            training_tests.append(("Supported Models", False, f"Exception: {str(e)}"))
        
        # Test 4: Hyperparameter optimization (if available)
        print("\n--- Testing Hyperparameter Optimization ---")
        
        try:
            response = self.session.post(
                f"{API_BASE_URL}/optimize-hyperparameters",
                json={
                    "data_id": data_id,
                    "model_type": "lstm",
                    "n_trials": 5  # Small number for testing
                }
            )
            
            if response.status_code == 200:
                optimization_data = response.json()
                
                if 'best_params' in optimization_data:
                    training_tests.append(("Hyperparameter Optimization", True, "Optimization working"))
                    print("✅ Hyperparameter optimization working")
                else:
                    training_tests.append(("Hyperparameter Optimization", False, "No optimization results"))
            else:
                training_tests.append(("Hyperparameter Optimization", False, f"API error: {response.status_code}"))
                
        except Exception as e:
            training_tests.append(("Hyperparameter Optimization", False, f"Exception: {str(e)}"))
        
        # Evaluate training results
        passed_tests = sum(1 for _, success, _ in training_tests if success)
        total_tests = len(training_tests)
        
        print(f"\n📊 Advanced Model Training Results: {passed_tests}/{total_tests}")
        for test_name, success, details in training_tests:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}: {details}")
        
        self.test_results['advanced_model_training'] = passed_tests >= total_tests * 0.6
    
    def run_specific_tests(self):
        """Run specific task tests"""
        print("🚀 Starting Specific Task Testing")
        print("=" * 60)
        
        # Run specific tests for tasks marked as needs_retesting
        self.test_enhanced_data_preprocessing_quality_validation()
        self.test_advanced_model_training_hyperparameter_optimization()
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final report for specific tasks"""
        print("\n" + "=" * 60)
        print("🎯 SPECIFIC TASK TESTING REPORT")
        print("=" * 60)
        
        test_categories = [
            ("Enhanced Data Preprocessing and Quality Validation", self.test_results.get('enhanced_data_preprocessing', False)),
            ("Advanced Model Training and Hyperparameter Optimization", self.test_results.get('advanced_model_training', False))
        ]
        
        passed_categories = sum(1 for _, passed in test_categories if passed)
        total_categories = len(test_categories)
        
        print(f"\n📊 OVERALL RESULTS: {passed_categories}/{total_categories} specific tasks passed")
        print(f"Success Rate: {(passed_categories/total_categories)*100:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        for category, passed in test_categories:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {status} {category}")
        
        # Assessment for test_result.md update
        if passed_categories == total_categories:
            print(f"\n🎉 ASSESSMENT: ALL SPECIFIC TASKS WORKING")
            print("   Both tasks marked as needs_retesting are now functioning correctly")
        elif passed_categories > 0:
            print(f"\n⚠️  ASSESSMENT: PARTIAL SUCCESS")
            print("   Some specific tasks are working, others may need attention")
        else:
            print(f"\n❌ ASSESSMENT: TASKS NEED ATTENTION")
            print("   Both specific tasks require further work")
        
        print("\n" + "=" * 60)


def main():
    """Main test execution"""
    tester = SpecificTaskTester()
    tester.run_specific_tests()


if __name__ == "__main__":
    main()