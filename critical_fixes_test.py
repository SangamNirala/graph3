#!/usr/bin/env python3
"""
Critical Fixes Testing for Advanced ML Models Application
Tests the specific fixes applied for datetime arithmetic, N-BEATS state dict loading, 
duplicate keys errors, and data preparation for small datasets.
"""

import requests
import json
import pandas as pd
import io
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://064f3bb3-c010-4892-8a8e-8e29d9900fe8.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing critical fixes at: {API_BASE_URL}")

class CriticalFixesTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_ids = {}
        self.test_results = {}
        
    def create_ph_dataset(self):
        """Create pH dataset with 49 samples for testing small dataset handling"""
        # Generate 49 days of pH data (small dataset to test parameter adjustment)
        dates = pd.date_range(start='2023-01-01', periods=49, freq='D')
        
        # Create realistic pH data with trend and variations
        base_ph = 7.2
        trend = np.linspace(0, 0.3, 49)  # Slight upward trend
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(49) / 7)  # Weekly pattern
        noise = np.random.normal(0, 0.1, 49)
        ph_values = base_ph + trend + seasonal + noise
        
        # Keep pH in realistic range
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph_value': ph_values
        })
        
        return df
    
    def test_file_upload_ph_dataset(self):
        """Test 1: Upload pH dataset (49 samples)"""
        print("\n=== Testing pH Dataset Upload (49 samples) ===")
        
        try:
            # Create pH dataset
            df = self.create_ph_dataset()
            csv_content = df.to_csv(index=False)
            
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
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                
                # Validate small dataset handling
                if data['analysis']['data_shape'][0] == 49:
                    print("✅ Small dataset (49 samples) correctly processed")
                    self.test_results['ph_dataset_upload'] = True
                else:
                    print(f"❌ Dataset size mismatch: expected 49, got {data['analysis']['data_shape'][0]}")
                    self.test_results['ph_dataset_upload'] = False
                    
            else:
                print(f"❌ pH dataset upload failed: {response.status_code} - {response.text}")
                self.test_results['ph_dataset_upload'] = False
                
        except Exception as e:
            print(f"❌ pH dataset upload error: {str(e)}")
            self.test_results['ph_dataset_upload'] = False
    
    def test_advanced_model_training(self, model_type):
        """Test advanced model training with small dataset parameter adjustment"""
        print(f"\n=== Testing {model_type.upper()} Model Training (Small Dataset) ===")
        
        if not self.data_id:
            print(f"❌ Cannot test {model_type} training - no data uploaded")
            self.test_results[f'{model_type}_training'] = False
            return
            
        try:
            # Prepare training parameters adjusted for small dataset
            training_params = {
                "time_column": "timestamp",
                "target_column": "ph_value",
                "seq_len": 8,  # Reduced for small dataset
                "pred_len": 3,  # Reduced for small dataset
                "epochs": 50,   # Reduced for faster testing
                "batch_size": 8,  # Reduced for small dataset
                "learning_rate": 0.001
            }
            
            # Test model training
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": model_type},
                json=training_params
            )
            
            if response.status_code == 200:
                data = response.json()
                model_id = data.get('model_id')
                self.model_ids[model_type] = model_id
                
                print(f"✅ {model_type.upper()} model training successful")
                print(f"   Model ID: {model_id}")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                
                # Check for performance metrics
                if 'performance_metrics' in data:
                    metrics = data['performance_metrics']
                    print(f"   Performance Metrics: {metrics}")
                
                if 'evaluation_grade' in data:
                    grade = data['evaluation_grade']
                    print(f"   Evaluation Grade: {grade}")
                
                self.test_results[f'{model_type}_training'] = True
                
            else:
                print(f"❌ {model_type.upper()} model training failed: {response.status_code}")
                print(f"   Error details: {response.text}")
                self.test_results[f'{model_type}_training'] = False
                
        except Exception as e:
            print(f"❌ {model_type.upper()} model training error: {str(e)}")
            self.test_results[f'{model_type}_training'] = False
    
    def test_advanced_prediction_endpoint(self):
        """Test 2: Advanced prediction endpoint (datetime arithmetic fix)"""
        print("\n=== Testing Advanced Prediction Endpoint (DateTime Fix) ===")
        
        # Find a trained advanced model
        trained_model = None
        for model_type, model_id in self.model_ids.items():
            if model_id and model_type in ['dlinear', 'nbeats', 'lstm', 'lightgbm']:
                trained_model = model_type
                break
        
        if not trained_model:
            print("❌ No advanced model trained for testing advanced prediction")
            self.test_results['advanced_prediction_endpoint'] = False
            return
            
        try:
            # Test advanced prediction endpoint
            response = self.session.post(
                f"{API_BASE_URL}/advanced-prediction",
                json={"steps": 10, "confidence_level": 0.95}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Advanced prediction endpoint successful")
                print(f"   Model type: {data.get('model_type')}")
                print(f"   Number of predictions: {len(data.get('predictions', []))}")
                print(f"   Number of timestamps: {len(data.get('timestamps', []))}")
                print(f"   Has confidence intervals: {'confidence_intervals' in data}")
                
                # Validate datetime format in timestamps
                timestamps = data.get('timestamps', [])
                if timestamps:
                    try:
                        # Try to parse first timestamp
                        datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
                        print("✅ Timestamp format is correct (datetime arithmetic fix working)")
                        self.test_results['advanced_prediction_endpoint'] = True
                    except ValueError as ve:
                        print(f"❌ Timestamp format error: {ve}")
                        self.test_results['advanced_prediction_endpoint'] = False
                else:
                    print("❌ No timestamps returned")
                    self.test_results['advanced_prediction_endpoint'] = False
                    
            else:
                print(f"❌ Advanced prediction endpoint failed: {response.status_code}")
                print(f"   Error details: {response.text}")
                
                # Check if it's the specific datetime arithmetic error
                if "unsupported operand type(s) for +: 'int' and 'datetime.timedelta'" in response.text:
                    print("❌ CRITICAL: DateTime arithmetic error still present!")
                
                self.test_results['advanced_prediction_endpoint'] = False
                
        except Exception as e:
            print(f"❌ Advanced prediction endpoint error: {str(e)}")
            self.test_results['advanced_prediction_endpoint'] = False
    
    def test_generate_prediction_endpoint(self):
        """Test 3: Generate prediction endpoint (datetime fix)"""
        print("\n=== Testing Generate Prediction Endpoint (DateTime Fix) ===")
        
        # Find any trained model
        model_id = None
        for mid in self.model_ids.values():
            if mid:
                model_id = mid
                break
        
        if not model_id:
            print("❌ No model trained for testing generate prediction")
            self.test_results['generate_prediction_endpoint'] = False
            return
            
        try:
            # Test generate prediction endpoint
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 10}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Generate prediction endpoint successful")
                print(f"   Number of predictions: {len(data.get('predictions', []))}")
                print(f"   Number of timestamps: {len(data.get('timestamps', []))}")
                
                # Validate datetime format in timestamps
                timestamps = data.get('timestamps', [])
                if timestamps:
                    try:
                        # Try to parse first timestamp
                        datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
                        print("✅ Timestamp format is correct (datetime arithmetic fix working)")
                        self.test_results['generate_prediction_endpoint'] = True
                    except ValueError as ve:
                        print(f"❌ Timestamp format error: {ve}")
                        self.test_results['generate_prediction_endpoint'] = False
                else:
                    print("❌ No timestamps returned")
                    self.test_results['generate_prediction_endpoint'] = False
                    
            else:
                print(f"❌ Generate prediction endpoint failed: {response.status_code}")
                print(f"   Error details: {response.text}")
                
                # Check if it's the specific datetime arithmetic error
                if "unsupported operand type(s) for +: 'int' and 'datetime.timedelta'" in response.text:
                    print("❌ CRITICAL: DateTime arithmetic error still present!")
                
                self.test_results['generate_prediction_endpoint'] = False
                
        except Exception as e:
            print(f"❌ Generate prediction endpoint error: {str(e)}")
            self.test_results['generate_prediction_endpoint'] = False
    
    def test_model_comparison_endpoint(self):
        """Test 4: Model comparison endpoint (duplicate keys fix)"""
        print("\n=== Testing Model Comparison Endpoint (Duplicate Keys Fix) ===")
        
        if not self.data_id:
            print("❌ Cannot test model comparison - no data uploaded")
            self.test_results['model_comparison_endpoint'] = False
            return
            
        try:
            # Test model comparison endpoint
            response = self.session.get(f"{API_BASE_URL}/model-comparison")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Model comparison endpoint successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Models compared: {data.get('models_compared', [])}")
                print(f"   Best model: {data.get('best_model')}")
                print(f"   Best score: {data.get('best_score')}")
                
                # Check comparison results structure
                comparison_results = data.get('comparison_results', {})
                if comparison_results:
                    print("✅ Comparison results structure is correct")
                    for model_type, results in comparison_results.items():
                        if 'error' in results:
                            print(f"   ⚠️  {model_type}: {results['error']}")
                        else:
                            print(f"   ✅ {model_type}: Grade {results.get('performance_grade', 'N/A')}")
                    
                    self.test_results['model_comparison_endpoint'] = True
                else:
                    print("❌ No comparison results returned")
                    self.test_results['model_comparison_endpoint'] = False
                    
            else:
                print(f"❌ Model comparison endpoint failed: {response.status_code}")
                print(f"   Error details: {response.text}")
                
                # Check if it's the specific duplicate keys error
                if "duplicate" in response.text.lower() and "keys" in response.text.lower():
                    print("❌ CRITICAL: Duplicate keys pandas DataFrame error still present!")
                
                self.test_results['model_comparison_endpoint'] = False
                
        except Exception as e:
            print(f"❌ Model comparison endpoint error: {str(e)}")
            self.test_results['model_comparison_endpoint'] = False
    
    def test_hyperparameter_optimization_endpoint(self):
        """Test 5: Hyperparameter optimization endpoint (duplicate keys fix)"""
        print("\n=== Testing Hyperparameter Optimization Endpoint (Duplicate Keys Fix) ===")
        
        if not self.data_id:
            print("❌ Cannot test hyperparameter optimization - no data uploaded")
            self.test_results['hyperparameter_optimization_endpoint'] = False
            return
            
        try:
            # Test hyperparameter optimization endpoint with reduced trials for faster testing
            response = self.session.post(
                f"{API_BASE_URL}/optimize-hyperparameters",
                params={"model_type": "lstm", "n_trials": 5}  # Reduced trials for testing
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Hyperparameter optimization endpoint successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Model type: {data.get('model_type')}")
                print(f"   Best score: {data.get('best_score')}")
                
                # Check optimization results structure
                optimization_results = data.get('optimization_results', {})
                best_params = data.get('best_parameters', {})
                
                if optimization_results and best_params:
                    print("✅ Optimization results structure is correct")
                    print(f"   Best parameters: {best_params}")
                    self.test_results['hyperparameter_optimization_endpoint'] = True
                else:
                    print("❌ Incomplete optimization results")
                    self.test_results['hyperparameter_optimization_endpoint'] = False
                    
            else:
                print(f"❌ Hyperparameter optimization endpoint failed: {response.status_code}")
                print(f"   Error details: {response.text}")
                
                # Check if it's the specific duplicate keys error
                if "duplicate" in response.text.lower() and "keys" in response.text.lower():
                    print("❌ CRITICAL: Duplicate keys pandas DataFrame error still present!")
                
                self.test_results['hyperparameter_optimization_endpoint'] = False
                
        except Exception as e:
            print(f"❌ Hyperparameter optimization endpoint error: {str(e)}")
            self.test_results['hyperparameter_optimization_endpoint'] = False
    
    def test_data_quality_report_endpoint(self):
        """Test 6: Data quality report endpoint"""
        print("\n=== Testing Data Quality Report Endpoint ===")
        
        if not self.data_id:
            print("❌ Cannot test data quality report - no data uploaded")
            self.test_results['data_quality_report_endpoint'] = False
            return
            
        try:
            # Test data quality report endpoint
            response = self.session.get(f"{API_BASE_URL}/data-quality-report")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Data quality report endpoint successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Quality score: {data.get('quality_score')}")
                
                # Check validation results structure
                validation_results = data.get('validation_results', {})
                recommendations = data.get('recommendations', [])
                
                if validation_results and 'quality_score' in data:
                    print("✅ Data quality report structure is correct")
                    print(f"   Recommendations: {len(recommendations)} items")
                    self.test_results['data_quality_report_endpoint'] = True
                else:
                    print("❌ Incomplete data quality report")
                    self.test_results['data_quality_report_endpoint'] = False
                    
            else:
                print(f"❌ Data quality report endpoint failed: {response.status_code}")
                print(f"   Error details: {response.text}")
                self.test_results['data_quality_report_endpoint'] = False
                
        except Exception as e:
            print(f"❌ Data quality report endpoint error: {str(e)}")
            self.test_results['data_quality_report_endpoint'] = False
    
    def test_supported_models_endpoint(self):
        """Test 7: Supported models endpoint"""
        print("\n=== Testing Supported Models Endpoint ===")
        
        try:
            # Test supported models endpoint
            response = self.session.get(f"{API_BASE_URL}/supported-models")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Supported models endpoint successful")
                print(f"   Status: {data.get('status')}")
                
                traditional_models = data.get('traditional_models', [])
                advanced_models = data.get('advanced_models', [])
                all_models = data.get('all_models', [])
                
                print(f"   Traditional models: {traditional_models}")
                print(f"   Advanced models: {advanced_models}")
                print(f"   All models: {len(all_models)} total")
                
                # Validate expected models are present
                expected_advanced = ['dlinear', 'nbeats', 'lstm', 'lightgbm', 'ensemble']
                missing_models = [model for model in expected_advanced if model not in advanced_models]
                
                if not missing_models:
                    print("✅ All expected advanced models are supported")
                    self.test_results['supported_models_endpoint'] = True
                else:
                    print(f"❌ Missing advanced models: {missing_models}")
                    self.test_results['supported_models_endpoint'] = False
                    
            else:
                print(f"❌ Supported models endpoint failed: {response.status_code}")
                print(f"   Error details: {response.text}")
                self.test_results['supported_models_endpoint'] = False
                
        except Exception as e:
            print(f"❌ Supported models endpoint error: {str(e)}")
            self.test_results['supported_models_endpoint'] = False
    
    def test_model_performance_endpoint(self):
        """Test 8: Model performance endpoint"""
        print("\n=== Testing Model Performance Endpoint ===")
        
        # Find a trained model
        if not self.model_ids:
            print("❌ Cannot test model performance - no models trained")
            self.test_results['model_performance_endpoint'] = False
            return
            
        try:
            # Test model performance endpoint
            response = self.session.get(f"{API_BASE_URL}/model-performance")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Model performance endpoint successful")
                print(f"   Status: {data.get('status')}")
                
                performance_data = data.get('performance_data', {})
                if performance_data:
                    print(f"   Model type: {performance_data.get('model_type')}")
                    print(f"   Is advanced: {performance_data.get('is_advanced')}")
                    
                    # Check for evaluation results
                    if 'evaluation_results' in performance_data:
                        eval_results = performance_data['evaluation_results']
                        print("✅ Evaluation results available")
                        if 'basic_metrics' in eval_results:
                            metrics = eval_results['basic_metrics']
                            print(f"   Basic metrics: RMSE={metrics.get('rmse', 'N/A')}, MAE={metrics.get('mae', 'N/A')}")
                    
                    self.test_results['model_performance_endpoint'] = True
                else:
                    print("❌ No performance data returned")
                    self.test_results['model_performance_endpoint'] = False
                    
            else:
                print(f"❌ Model performance endpoint failed: {response.status_code}")
                print(f"   Error details: {response.text}")
                self.test_results['model_performance_endpoint'] = False
                
        except Exception as e:
            print(f"❌ Model performance endpoint error: {str(e)}")
            self.test_results['model_performance_endpoint'] = False
    
    def run_critical_fixes_tests(self):
        """Run all critical fixes tests"""
        print("🔍 Starting Critical Fixes Testing for Advanced ML Models")
        print("=" * 70)
        
        # Test 1: Upload pH dataset (49 samples)
        self.test_file_upload_ph_dataset()
        
        # Test 2: Train all supported advanced models with small dataset
        advanced_models = ['dlinear', 'nbeats', 'lstm', 'lightgbm', 'ensemble']
        for model_type in advanced_models:
            self.test_advanced_model_training(model_type)
        
        # Test 3: Test critical endpoints
        self.test_supported_models_endpoint()
        self.test_data_quality_report_endpoint()
        self.test_model_performance_endpoint()
        
        # Test 4: Test critical fixes
        self.test_advanced_prediction_endpoint()
        self.test_generate_prediction_endpoint()
        self.test_model_comparison_endpoint()
        self.test_hyperparameter_optimization_endpoint()
        
        # Print final results
        self.print_critical_fixes_summary()
    
    def print_critical_fixes_summary(self):
        """Print comprehensive test summary focused on critical fixes"""
        print("\n" + "=" * 70)
        print("📊 CRITICAL FIXES TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n🔧 CRITICAL FIXES STATUS:")
        
        # Group results by fix category
        fix_categories = {
            "DateTime Arithmetic Fixes": [
                'advanced_prediction_endpoint',
                'generate_prediction_endpoint'
            ],
            "N-BEATS Model State Dict": [
                'nbeats_training'
            ],
            "Duplicate Keys DataFrame Fixes": [
                'model_comparison_endpoint',
                'hyperparameter_optimization_endpoint'
            ],
            "Small Dataset Parameter Adjustment": [
                'ph_dataset_upload',
                'dlinear_training',
                'lstm_training',
                'lightgbm_training',
                'ensemble_training'
            ],
            "Core Endpoint Functionality": [
                'supported_models_endpoint',
                'data_quality_report_endpoint',
                'model_performance_endpoint'
            ]
        }
        
        for category, test_keys in fix_categories.items():
            print(f"\n{category}:")
            category_passed = 0
            category_total = 0
            
            for test_key in test_keys:
                if test_key in self.test_results:
                    result = self.test_results[test_key]
                    status = "✅ FIXED" if result else "❌ STILL BROKEN"
                    test_name = test_key.replace('_', ' ').title()
                    print(f"  {status} - {test_name}")
                    
                    if result:
                        category_passed += 1
                    category_total += 1
            
            if category_total > 0:
                category_rate = (category_passed / category_total) * 100
                print(f"  📈 Category Success Rate: {category_rate:.1f}% ({category_passed}/{category_total})")
        
        print("\n" + "=" * 70)
        
        # Determine overall critical fixes status
        critical_endpoints = [
            'advanced_prediction_endpoint',
            'generate_prediction_endpoint',
            'model_comparison_endpoint',
            'hyperparameter_optimization_endpoint'
        ]
        
        critical_passed = sum(1 for key in critical_endpoints if self.test_results.get(key, False))
        critical_total = len(critical_endpoints)
        
        print(f"🎯 CRITICAL ENDPOINTS STATUS: {critical_passed}/{critical_total} working")
        
        if critical_passed == critical_total:
            print("🎉 ALL CRITICAL FIXES VERIFIED - READY FOR PRODUCTION!")
        elif critical_passed >= critical_total * 0.75:
            print("⚠️  MOST CRITICAL FIXES WORKING - MINOR ISSUES REMAIN")
        else:
            print("❌ CRITICAL FIXES INCOMPLETE - MAJOR ISSUES NEED ATTENTION")
        
        return passed_tests >= total_tests * 0.7  # 70% pass rate for critical fixes

if __name__ == "__main__":
    tester = CriticalFixesTester()
    overall_success = tester.run_critical_fixes_tests()
    
    if overall_success:
        print("🎉 Critical fixes testing completed successfully!")
        exit(0)
    else:
        print("⚠️  Critical fixes testing completed with failures.")
        exit(1)