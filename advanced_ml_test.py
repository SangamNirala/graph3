#!/usr/bin/env python3
"""
Advanced ML Features Testing for Real-time Graph Prediction Application
Tests all new advanced ML capabilities including DLinear, N-BEATS, LSTM, ensemble methods,
data preprocessing, model evaluation, and advanced prediction endpoints.
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

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://3953ed3b-a104-45f4-961d-3b0701b9a50c.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing advanced ML features at: {API_BASE_URL}")

class AdvancedMLTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        self.test_results = {}
        
    def create_realistic_time_series_data(self):
        """Create realistic time-series data for advanced ML testing"""
        # Generate 200 days of hourly data with complex patterns
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Create complex time series with trend, seasonality, and noise
        trend = np.linspace(100, 200, 200)
        seasonal = 30 * np.sin(2 * np.pi * np.arange(200) / 7)  # Weekly pattern
        monthly_seasonal = 15 * np.sin(2 * np.pi * np.arange(200) / 30)  # Monthly pattern
        noise = np.random.normal(0, 10, 200)
        
        # Add some non-linear patterns
        non_linear = 5 * np.sin(2 * np.pi * np.arange(200) / 50) * np.exp(-np.arange(200) / 100)
        
        values = trend + seasonal + monthly_seasonal + noise + non_linear
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'category': ['A'] * 100 + ['B'] * 100
        })
        
        return df
    
    def test_supported_models_endpoint(self):
        """Test 1: Test /api/supported-models endpoint"""
        print("\n=== Testing Supported Models Endpoint ===")
        
        try:
            response = self.session.get(f"{API_BASE_URL}/supported-models")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Supported models endpoint successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Traditional models: {data.get('traditional_models', [])}")
                print(f"   Advanced models: {data.get('advanced_models', [])}")
                print(f"   All models: {data.get('all_models', [])}")
                
                # Validate that new advanced models are included
                advanced_models = data.get('advanced_models', [])
                expected_models = ['dlinear', 'nbeats', 'lstm', 'lightgbm', 'xgboost', 'ensemble']
                
                missing_models = [model for model in expected_models if model not in advanced_models]
                
                if not missing_models:
                    print("✅ All expected advanced models are supported")
                    self.test_results['supported_models'] = True
                else:
                    print(f"❌ Missing advanced models: {missing_models}")
                    self.test_results['supported_models'] = False
                    
            else:
                print(f"❌ Supported models endpoint failed: {response.status_code} - {response.text}")
                self.test_results['supported_models'] = False
                
        except Exception as e:
            print(f"❌ Supported models test error: {str(e)}")
            self.test_results['supported_models'] = False
    
    def test_file_upload_for_advanced_models(self):
        """Test 2: Upload data for advanced model testing"""
        print("\n=== Testing File Upload for Advanced Models ===")
        
        try:
            # Create realistic time series data
            df = self.create_realistic_time_series_data()
            csv_content = df.to_csv(index=False)
            
            # Prepare file for upload
            files = {
                'file': ('advanced_time_series.csv', csv_content, 'text/csv')
            }
            
            # Test file upload
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                
                print("✅ File upload for advanced models successful")
                print(f"   Data ID: {self.data_id}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                
                # Validate analysis results
                analysis = data['analysis']
                if 'date' in analysis['time_columns'] and 'value' in analysis['numeric_columns']:
                    print("✅ Data analysis correctly identified columns for advanced models")
                    self.test_results['advanced_file_upload'] = True
                else:
                    print("❌ Data analysis failed for advanced models")
                    self.test_results['advanced_file_upload'] = False
                    
            else:
                print(f"❌ File upload for advanced models failed: {response.status_code} - {response.text}")
                self.test_results['advanced_file_upload'] = False
                
        except Exception as e:
            print(f"❌ Advanced file upload test error: {str(e)}")
            self.test_results['advanced_file_upload'] = False
    
    def test_advanced_model_training(self, model_type='dlinear'):
        """Test 3: Train advanced ML models"""
        print(f"\n=== Testing Advanced Model Training ({model_type.upper()}) ===")
        
        if not self.data_id:
            print("❌ Cannot test advanced model training - no data uploaded")
            self.test_results[f'{model_type}_training'] = False
            return
            
        try:
            # Prepare training parameters for advanced model
            training_data = {
                "data_id": self.data_id,
                "model_type": model_type,
                "parameters": {
                    "time_column": "date",
                    "target_column": "value",
                    "seq_len": 50,
                    "pred_len": 30,
                    "epochs": 50,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "scaling_method": "standard",
                    "denoise": True,
                    "denoise_method": "savgol"
                }
            }
            
            # Test advanced model training
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": model_type},
                json=training_data["parameters"]
            )
            
            if response.status_code == 200:
                data = response.json()
                self.model_id = data.get('model_id')
                
                print(f"✅ {model_type.upper()} model training successful")
                print(f"   Model ID: {self.model_id}")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                
                # Check for performance metrics
                performance_metrics = data.get('performance_metrics', {})
                if performance_metrics:
                    print("✅ Performance metrics included:")
                    for metric, value in performance_metrics.items():
                        print(f"     {metric}: {value}")
                
                # Check for evaluation grade
                evaluation_grade = data.get('evaluation_grade', 'N/A')
                print(f"   Evaluation Grade: {evaluation_grade}")
                
                self.test_results[f'{model_type}_training'] = True
                
            else:
                print(f"❌ {model_type.upper()} model training failed: {response.status_code} - {response.text}")
                self.test_results[f'{model_type}_training'] = False
                
        except Exception as e:
            print(f"❌ {model_type.upper()} model training error: {str(e)}")
            self.test_results[f'{model_type}_training'] = False
    
    def test_data_quality_report(self):
        """Test 4: Data quality reporting endpoint"""
        print("\n=== Testing Data Quality Report ===")
        
        if not self.data_id:
            print("❌ Cannot test data quality report - no data uploaded")
            self.test_results['data_quality_report'] = False
            return
            
        try:
            response = self.session.get(f"{API_BASE_URL}/data-quality-report")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Data quality report successful")
                print(f"   Status: {data.get('status')}")
                
                validation_results = data.get('validation_results', {})
                quality_score = data.get('quality_score', 0)
                recommendations = data.get('recommendations', [])
                
                print(f"   Quality Score: {quality_score:.2f}/100")
                print(f"   Total Rows: {validation_results.get('total_rows', 'N/A')}")
                print(f"   Missing Values: {validation_results.get('missing_values', {})}")
                print(f"   Recommendations: {len(recommendations)} items")
                
                # Validate structure
                if quality_score > 0 and isinstance(recommendations, list):
                    print("✅ Data quality report structure is correct")
                    self.test_results['data_quality_report'] = True
                else:
                    print("❌ Data quality report structure is incorrect")
                    self.test_results['data_quality_report'] = False
                    
            else:
                print(f"❌ Data quality report failed: {response.status_code} - {response.text}")
                self.test_results['data_quality_report'] = False
                
        except Exception as e:
            print(f"❌ Data quality report error: {str(e)}")
            self.test_results['data_quality_report'] = False
    
    def test_advanced_prediction_endpoint(self):
        """Test 5: Advanced prediction capabilities"""
        print("\n=== Testing Advanced Prediction Endpoint ===")
        
        if not self.model_id:
            print("❌ Cannot test advanced prediction - no advanced model trained")
            self.test_results['advanced_prediction'] = False
            return
            
        try:
            response = self.session.post(
                f"{API_BASE_URL}/advanced-prediction",
                json={"steps": 30, "confidence_level": 0.95}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Advanced prediction successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Model Type: {data.get('model_type')}")
                print(f"   Prediction Horizon: {data.get('prediction_horizon')}")
                
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                confidence = data.get('confidence', [])
                confidence_intervals = data.get('confidence_intervals', [])
                individual_predictions = data.get('individual_predictions', {})
                
                print(f"   Number of predictions: {len(predictions)}")
                print(f"   Number of timestamps: {len(timestamps)}")
                print(f"   Has confidence scores: {len(confidence) > 0}")
                print(f"   Has confidence intervals: {len(confidence_intervals) > 0}")
                print(f"   Individual predictions: {len(individual_predictions)} models")
                
                # Validate prediction structure
                if (len(predictions) == 30 and len(timestamps) == 30 and 
                    len(confidence_intervals) == 30):
                    print("✅ Advanced prediction structure is correct")
                    print(f"   Sample predictions: {predictions[:3]}")
                    print(f"   Sample confidence: {confidence[:3] if confidence else 'N/A'}")
                    self.test_results['advanced_prediction'] = True
                else:
                    print("❌ Advanced prediction structure is incorrect")
                    self.test_results['advanced_prediction'] = False
                    
            else:
                print(f"❌ Advanced prediction failed: {response.status_code} - {response.text}")
                self.test_results['advanced_prediction'] = False
                
        except Exception as e:
            print(f"❌ Advanced prediction error: {str(e)}")
            self.test_results['advanced_prediction'] = False
    
    def test_model_performance_endpoint(self):
        """Test 6: Model performance tracking"""
        print("\n=== Testing Model Performance Endpoint ===")
        
        if not self.model_id:
            print("❌ Cannot test model performance - no model trained")
            self.test_results['model_performance'] = False
            return
            
        try:
            response = self.session.get(f"{API_BASE_URL}/model-performance")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Model performance endpoint successful")
                print(f"   Status: {data.get('status')}")
                
                performance_data = data.get('performance_data', {})
                model_type = performance_data.get('model_type', 'N/A')
                is_advanced = performance_data.get('is_advanced', False)
                parameters = performance_data.get('parameters', {})
                evaluation_results = performance_data.get('evaluation_results', {})
                
                print(f"   Model Type: {model_type}")
                print(f"   Is Advanced: {is_advanced}")
                print(f"   Parameters: {len(parameters)} items")
                
                if evaluation_results:
                    print("✅ Evaluation results included:")
                    basic_metrics = evaluation_results.get('basic_metrics', {})
                    for metric, value in basic_metrics.items():
                        print(f"     {metric}: {value}")
                    
                    evaluation_summary = evaluation_results.get('evaluation_summary', {})
                    performance_grade = evaluation_summary.get('performance_grade', 'N/A')
                    print(f"   Performance Grade: {performance_grade}")
                
                self.test_results['model_performance'] = True
                
            else:
                print(f"❌ Model performance endpoint failed: {response.status_code} - {response.text}")
                self.test_results['model_performance'] = False
                
        except Exception as e:
            print(f"❌ Model performance error: {str(e)}")
            self.test_results['model_performance'] = False
    
    def test_enhanced_prediction_endpoint(self):
        """Test 7: Enhanced prediction with advanced models"""
        print("\n=== Testing Enhanced Prediction Endpoint ===")
        
        if not self.model_id:
            print("❌ Cannot test enhanced prediction - no model trained")
            self.test_results['enhanced_prediction'] = False
            return
            
        try:
            # Test traditional generate-prediction endpoint with advanced model
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": self.model_id, "steps": 20}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Enhanced prediction endpoint successful")
                print(f"   Model Type: {data.get('model_type')}")
                print(f"   Is Advanced: {data.get('is_advanced', False)}")
                
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                confidence = data.get('confidence', [])
                confidence_intervals = data.get('confidence_intervals')
                
                print(f"   Number of predictions: {len(predictions)}")
                print(f"   Number of timestamps: {len(timestamps)}")
                print(f"   Has confidence scores: {len(confidence) > 0}")
                print(f"   Has confidence intervals: {confidence_intervals is not None}")
                
                # Validate that advanced models work with traditional endpoint
                if len(predictions) == 20 and len(timestamps) == 20:
                    print("✅ Enhanced prediction works with traditional endpoint")
                    print(f"   Sample predictions: {predictions[:3]}")
                    self.test_results['enhanced_prediction'] = True
                else:
                    print("❌ Enhanced prediction structure is incorrect")
                    self.test_results['enhanced_prediction'] = False
                    
            else:
                print(f"❌ Enhanced prediction failed: {response.status_code} - {response.text}")
                self.test_results['enhanced_prediction'] = False
                
        except Exception as e:
            print(f"❌ Enhanced prediction error: {str(e)}")
            self.test_results['enhanced_prediction'] = False
    
    def test_hyperparameter_optimization(self):
        """Test 8: Hyperparameter optimization"""
        print("\n=== Testing Hyperparameter Optimization ===")
        
        if not self.data_id:
            print("❌ Cannot test hyperparameter optimization - no data uploaded")
            self.test_results['hyperparameter_optimization'] = False
            return
            
        try:
            # Test hyperparameter optimization for DLinear model
            response = self.session.post(
                f"{API_BASE_URL}/optimize-hyperparameters",
                params={"model_type": "dlinear", "n_trials": 10}  # Reduced trials for testing
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Hyperparameter optimization successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Model Type: {data.get('model_type')}")
                
                optimization_results = data.get('optimization_results', {})
                best_parameters = data.get('best_parameters', {})
                best_score = data.get('best_score', 'N/A')
                
                print(f"   Best Score: {best_score}")
                print(f"   Best Parameters: {best_parameters}")
                print(f"   Trials Completed: {optimization_results.get('n_trials', 'N/A')}")
                
                # Validate optimization results
                if best_parameters and isinstance(best_score, (int, float)):
                    print("✅ Hyperparameter optimization results are valid")
                    self.test_results['hyperparameter_optimization'] = True
                else:
                    print("❌ Hyperparameter optimization results are invalid")
                    self.test_results['hyperparameter_optimization'] = False
                    
            else:
                print(f"❌ Hyperparameter optimization failed: {response.status_code} - {response.text}")
                self.test_results['hyperparameter_optimization'] = False
                
        except Exception as e:
            print(f"❌ Hyperparameter optimization error: {str(e)}")
            self.test_results['hyperparameter_optimization'] = False
    
    def test_model_comparison(self):
        """Test 9: Model comparison capabilities"""
        print("\n=== Testing Model Comparison ===")
        
        if not self.data_id:
            print("❌ Cannot test model comparison - no data uploaded")
            self.test_results['model_comparison'] = False
            return
            
        try:
            response = self.session.get(f"{API_BASE_URL}/model-comparison")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Model comparison successful")
                print(f"   Status: {data.get('status')}")
                
                comparison_results = data.get('comparison_results', {})
                best_model = data.get('best_model', 'N/A')
                best_score = data.get('best_score', 'N/A')
                models_compared = data.get('models_compared', [])
                
                print(f"   Best Model: {best_model}")
                print(f"   Best Score: {best_score}")
                print(f"   Models Compared: {models_compared}")
                
                # Show comparison results
                for model_name, results in comparison_results.items():
                    if 'metrics' in results:
                        metrics = results['metrics']
                        grade = results.get('performance_grade', 'N/A')
                        print(f"   {model_name}: RMSE={metrics.get('rmse', 'N/A'):.4f}, Grade={grade}")
                    elif 'error' in results:
                        print(f"   {model_name}: Error - {results['error']}")
                
                # Validate comparison results
                if comparison_results and best_model != 'N/A':
                    print("✅ Model comparison results are valid")
                    self.test_results['model_comparison'] = True
                else:
                    print("❌ Model comparison results are invalid")
                    self.test_results['model_comparison'] = False
                    
            else:
                print(f"❌ Model comparison failed: {response.status_code} - {response.text}")
                self.test_results['model_comparison'] = False
                
        except Exception as e:
            print(f"❌ Model comparison error: {str(e)}")
            self.test_results['model_comparison'] = False
    
    def test_ensemble_model_training(self):
        """Test 10: Ensemble model training and prediction"""
        print("\n=== Testing Ensemble Model Training ===")
        
        if not self.data_id:
            print("❌ Cannot test ensemble model - no data uploaded")
            self.test_results['ensemble_model'] = False
            return
            
        try:
            # Train ensemble model
            training_data = {
                "data_id": self.data_id,
                "model_type": "ensemble",
                "parameters": {
                    "time_column": "date",
                    "target_column": "value",
                    "seq_len": 40,
                    "pred_len": 20,
                    "epochs": 30,  # Reduced for testing
                    "batch_size": 32,
                    "learning_rate": 0.001
                }
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": "ensemble"},
                json=training_data["parameters"]
            )
            
            if response.status_code == 200:
                data = response.json()
                ensemble_model_id = data.get('model_id')
                
                print("✅ Ensemble model training successful")
                print(f"   Model ID: {ensemble_model_id}")
                print(f"   Status: {data.get('status')}")
                
                # Test ensemble prediction
                pred_response = self.session.post(
                    f"{API_BASE_URL}/advanced-prediction",
                    json={"steps": 15, "confidence_level": 0.95}
                )
                
                if pred_response.status_code == 200:
                    pred_data = pred_response.json()
                    individual_predictions = pred_data.get('individual_predictions', {})
                    
                    print("✅ Ensemble prediction successful")
                    print(f"   Individual models in ensemble: {len(individual_predictions)}")
                    
                    if len(individual_predictions) > 1:
                        print("✅ Ensemble includes multiple individual model predictions")
                        self.test_results['ensemble_model'] = True
                    else:
                        print("❌ Ensemble does not include multiple models")
                        self.test_results['ensemble_model'] = False
                else:
                    print("❌ Ensemble prediction failed")
                    self.test_results['ensemble_model'] = False
                    
            else:
                print(f"❌ Ensemble model training failed: {response.status_code} - {response.text}")
                self.test_results['ensemble_model'] = False
                
        except Exception as e:
            print(f"❌ Ensemble model error: {str(e)}")
            self.test_results['ensemble_model'] = False
    
    def test_dependencies_and_imports(self):
        """Test 11: Test that all dependencies are properly imported"""
        print("\n=== Testing Dependencies and Imports ===")
        
        try:
            # Test that advanced model endpoints are accessible (indicates imports work)
            endpoints_to_test = [
                "/supported-models",
                "/data-quality-report" if self.data_id else None,
                "/model-performance" if self.model_id else None
            ]
            
            working_endpoints = 0
            total_endpoints = 0
            
            for endpoint in endpoints_to_test:
                if endpoint is None:
                    continue
                    
                total_endpoints += 1
                try:
                    response = self.session.get(f"{API_BASE_URL}{endpoint}")
                    if response.status_code == 200:
                        working_endpoints += 1
                        print(f"✅ {endpoint} - Working")
                    else:
                        print(f"❌ {endpoint} - Failed ({response.status_code})")
                except Exception as e:
                    print(f"❌ {endpoint} - Error: {str(e)}")
            
            # Test that we can import the modules (indirectly through API responses)
            if working_endpoints >= total_endpoints * 0.8:  # 80% success rate
                print("✅ Dependencies and imports appear to be working")
                self.test_results['dependencies_imports'] = True
            else:
                print("❌ Dependencies and imports have issues")
                self.test_results['dependencies_imports'] = False
                
        except Exception as e:
            print(f"❌ Dependencies test error: {str(e)}")
            self.test_results['dependencies_imports'] = False
    
    def run_all_advanced_tests(self):
        """Run all advanced ML feature tests"""
        print("🚀 Starting Advanced ML Features Testing")
        print("=" * 60)
        
        # Core advanced ML tests
        self.test_supported_models_endpoint()
        self.test_file_upload_for_advanced_models()
        
        # Test multiple advanced models
        advanced_models_to_test = ['dlinear', 'lstm', 'lightgbm']
        for model_type in advanced_models_to_test:
            self.test_advanced_model_training(model_type)
        
        # Advanced feature tests
        self.test_data_quality_report()
        self.test_advanced_prediction_endpoint()
        self.test_model_performance_endpoint()
        self.test_enhanced_prediction_endpoint()
        
        # Optimization and comparison tests
        self.test_hyperparameter_optimization()
        self.test_model_comparison()
        
        # Ensemble and dependencies tests
        self.test_ensemble_model_training()
        self.test_dependencies_and_imports()
        
        # Print final results
        self.print_advanced_test_summary()
    
    def print_advanced_test_summary(self):
        """Print comprehensive advanced ML test summary"""
        print("\n" + "=" * 60)
        print("📊 ADVANCED ML FEATURES TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} - {test_name.replace('_', ' ').title()}")
        
        print("\n" + "=" * 60)
        
        # Return overall success
        return passed_tests >= total_tests * 0.75  # 75% pass rate for advanced features

if __name__ == "__main__":
    tester = AdvancedMLTester()
    overall_success = tester.run_all_advanced_tests()
    
    if overall_success:
        print("🎉 Advanced ML features testing completed successfully!")
        exit(0)
    else:
        print("⚠️  Advanced ML features testing completed with some failures.")
        exit(1)