#!/usr/bin/env python3
"""
Enhanced Prediction System Testing
Tests the enhanced prediction system for real-time graph prediction with focus on:
1. Enhanced Prediction System Core
2. Integration with Existing Models  
3. Server-Side Integration
4. Key Prediction Quality Metrics
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
import matplotlib.pyplot as plt
from scipy import stats

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://4f535dbd-21ac-4151-8dfe-215665939abd.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Enhanced Prediction System at: {API_BASE_URL}")

class EnhancedPredictionTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        self.data_id = None
        self.model_id = None
        
    def create_sine_wave_data(self, periods=3, samples_per_period=50, noise_level=0.1):
        """Create sine wave dataset for pattern testing"""
        total_samples = periods * samples_per_period
        x = np.linspace(0, periods * 2 * np.pi, total_samples)
        
        # Pure sine wave with some noise
        y = np.sin(x) + np.random.normal(0, noise_level, total_samples)
        
        # Create timestamps
        start_date = datetime.now() - timedelta(days=total_samples)
        timestamps = [start_date + timedelta(days=i) for i in range(total_samples)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': y
        })
        
        return df
    
    def create_ph_data(self, samples=50):
        """Create realistic pH data for testing"""
        # Generate pH data in realistic range (6.0-8.0)
        base_ph = 7.0
        trend = np.linspace(0, 0.5, samples)  # Slight upward trend
        seasonal = 0.3 * np.sin(2 * np.pi * np.arange(samples) / 12)  # 12-sample cycle
        noise = np.random.normal(0, 0.1, samples)
        
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
    
    def create_random_walk_data(self, samples=100, drift=0.01):
        """Create random walk data for trend testing"""
        values = np.cumsum(np.random.normal(drift, 1, samples))
        
        start_date = datetime.now() - timedelta(days=samples)
        timestamps = [start_date + timedelta(days=i) for i in range(samples)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        
        return df
    
    def create_seasonal_data(self, samples=200):
        """Create data with seasonal patterns"""
        x = np.arange(samples)
        
        # Trend + seasonal + noise
        trend = 0.02 * x
        seasonal = 5 * np.sin(2 * np.pi * x / 24) + 2 * np.sin(2 * np.pi * x / 7)
        noise = np.random.normal(0, 1, samples)
        
        values = 100 + trend + seasonal + noise
        
        start_date = datetime.now() - timedelta(days=samples)
        timestamps = [start_date + timedelta(days=i) for i in range(samples)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        
        return df
    
    def upload_test_data(self, df, filename):
        """Upload test data to backend"""
        try:
            csv_content = df.to_csv(index=False)
            files = {'file': (filename, csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data_id'), data.get('analysis')
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None, None
                
        except Exception as e:
            print(f"❌ Upload error: {str(e)}")
            return None, None
    
    def train_model(self, data_id, time_col, target_col, model_type='lstm'):
        """Train model with uploaded data"""
        try:
            training_params = {
                "time_column": time_col,
                "target_column": target_col,
                "seq_len": 20,  # Smaller for test data
                "pred_len": 10,
                "epochs": 50,
                "batch_size": 16
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": model_type},
                json=training_params
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('model_id')
            else:
                print(f"❌ Training failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return None
    
    def test_enhanced_prediction_core(self):
        """Test 1: Enhanced Prediction System Core"""
        print("\n=== Testing Enhanced Prediction System Core ===")
        
        test_scenarios = [
            ("Sine Wave", self.create_sine_wave_data(), 'timestamp', 'value'),
            ("pH Data", self.create_ph_data(), 'timestamp', 'ph_value'),
            ("Random Walk", self.create_random_walk_data(), 'timestamp', 'value'),
            ("Seasonal Data", self.create_seasonal_data(), 'timestamp', 'value')
        ]
        
        core_test_results = []
        
        for scenario_name, test_data, time_col, target_col in test_scenarios:
            print(f"\n--- Testing {scenario_name} ---")
            
            # Upload data
            data_id, analysis = self.upload_test_data(test_data, f"{scenario_name.lower().replace(' ', '_')}_test.csv")
            
            if not data_id:
                core_test_results.append((scenario_name, False, "Upload failed"))
                continue
            
            # Train model
            model_id = self.train_model(data_id, time_col, target_col)
            
            if not model_id:
                core_test_results.append((scenario_name, False, "Training failed"))
                continue
            
            # Test enhanced predictions
            try:
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 30, "time_window": 100}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    
                    if len(predictions) == 30:
                        # Test prediction quality
                        quality_metrics = self.analyze_prediction_quality(
                            test_data[target_col].values, predictions, scenario_name
                        )
                        
                        core_test_results.append((scenario_name, True, quality_metrics))
                        print(f"✅ {scenario_name}: Enhanced predictions generated successfully")
                        print(f"   Quality Score: {quality_metrics.get('overall_score', 'N/A')}")
                    else:
                        core_test_results.append((scenario_name, False, f"Wrong prediction count: {len(predictions)}"))
                else:
                    core_test_results.append((scenario_name, False, f"API error: {response.status_code}"))
                    
            except Exception as e:
                core_test_results.append((scenario_name, False, f"Exception: {str(e)}"))
        
        # Evaluate core test results
        passed_tests = sum(1 for _, success, _ in core_test_results if success)
        total_tests = len(core_test_results)
        
        print(f"\n📊 Enhanced Prediction Core Results: {passed_tests}/{total_tests}")
        for scenario, success, details in core_test_results:
            status = "✅" if success else "❌"
            print(f"   {status} {scenario}: {details if not success else 'PASSED'}")
        
        self.test_results['enhanced_prediction_core'] = passed_tests >= total_tests * 0.75
        return core_test_results
    
    def analyze_prediction_quality(self, historical_data, predictions, scenario_name):
        """Analyze prediction quality metrics"""
        try:
            historical_data = np.array(historical_data)
            predictions = np.array(predictions)
            
            # 1. Smoothness - check for abrupt changes
            pred_diffs = np.diff(predictions)
            smoothness_score = 100 - min(100, np.std(pred_diffs) * 10)
            
            # 2. Historical Pattern Following - correlation with recent trend
            if len(historical_data) >= 10:
                recent_trend = np.polyfit(np.arange(10), historical_data[-10:], 1)[0]
                pred_trend = np.polyfit(np.arange(len(predictions)), predictions, 1)[0]
                trend_similarity = max(0, 100 - abs(recent_trend - pred_trend) * 100)
            else:
                trend_similarity = 50
            
            # 3. Statistical Property Preservation
            hist_mean = np.mean(historical_data)
            hist_std = np.std(historical_data)
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            
            mean_preservation = max(0, 100 - abs(hist_mean - pred_mean) / (hist_std + 1e-8) * 20)
            std_preservation = max(0, 100 - abs(hist_std - pred_std) / (hist_std + 1e-8) * 50)
            
            # 4. Generalization - check if predictions are not constant
            prediction_variance = np.var(predictions)
            generalization_score = min(100, prediction_variance * 100) if prediction_variance > 0 else 0
            
            # Overall score
            overall_score = (smoothness_score + trend_similarity + mean_preservation + 
                           std_preservation + generalization_score) / 5
            
            return {
                'smoothness_score': round(smoothness_score, 2),
                'trend_similarity': round(trend_similarity, 2),
                'mean_preservation': round(mean_preservation, 2),
                'std_preservation': round(std_preservation, 2),
                'generalization_score': round(generalization_score, 2),
                'overall_score': round(overall_score, 2),
                'scenario': scenario_name
            }
            
        except Exception as e:
            print(f"Error analyzing prediction quality: {e}")
            return {'overall_score': 0, 'error': str(e)}
    
    def test_integration_with_existing_models(self):
        """Test 2: Integration with Existing Models"""
        print("\n=== Testing Integration with Existing Models ===")
        
        # Use pH data for this test
        test_data = self.create_ph_data(samples=60)
        data_id, analysis = self.upload_test_data(test_data, "integration_test.csv")
        
        if not data_id:
            self.test_results['model_integration'] = False
            return
        
        integration_tests = []
        
        # Test different model types with enhanced predictions
        model_types = ['lstm', 'arima', 'prophet']
        
        for model_type in model_types:
            print(f"\n--- Testing {model_type.upper()} Integration ---")
            
            model_id = self.train_model(data_id, 'timestamp', 'ph_value', model_type)
            
            if model_id:
                # Test enhanced prediction integration
                try:
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": model_id, "steps": 20, "time_window": 50}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        predictions = data.get('predictions', [])
                        
                        # Test fallback mechanism
                        if len(predictions) == 20:
                            # Check if predictions are reasonable (pH range 6.0-8.0)
                            valid_ph_range = all(6.0 <= p <= 8.0 for p in predictions)
                            
                            if valid_ph_range:
                                integration_tests.append((model_type, True, "Enhanced integration working"))
                                print(f"✅ {model_type.upper()}: Enhanced integration successful")
                            else:
                                integration_tests.append((model_type, False, "Predictions outside valid range"))
                        else:
                            integration_tests.append((model_type, False, f"Wrong prediction count: {len(predictions)}"))
                    else:
                        integration_tests.append((model_type, False, f"API error: {response.status_code}"))
                        
                except Exception as e:
                    integration_tests.append((model_type, False, f"Exception: {str(e)}"))
            else:
                integration_tests.append((model_type, False, "Model training failed"))
        
        # Evaluate integration results
        passed_tests = sum(1 for _, success, _ in integration_tests if success)
        total_tests = len(integration_tests)
        
        print(f"\n📊 Model Integration Results: {passed_tests}/{total_tests}")
        for model_type, success, details in integration_tests:
            status = "✅" if success else "❌"
            print(f"   {status} {model_type.upper()}: {details}")
        
        self.test_results['model_integration'] = passed_tests >= total_tests * 0.67
    
    def test_server_side_integration(self):
        """Test 3: Server-Side Integration"""
        print("\n=== Testing Server-Side Integration ===")
        
        server_tests = []
        
        # Test 1: analyze_historical_patterns function
        print("\n--- Testing Historical Pattern Analysis ---")
        test_data = self.create_sine_wave_data(periods=2, samples_per_period=30)
        data_id, analysis = self.upload_test_data(test_data, "pattern_analysis_test.csv")
        
        if data_id:
            model_id = self.train_model(data_id, 'timestamp', 'value', 'lstm')
            
            if model_id:
                try:
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": model_id, "steps": 15, "time_window": 50}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        pattern_analysis = data.get('pattern_analysis')
                        
                        if pattern_analysis:
                            server_tests.append(("Pattern Analysis", True, "Pattern analysis included"))
                            print("✅ Historical pattern analysis working")
                        else:
                            server_tests.append(("Pattern Analysis", False, "No pattern analysis in response"))
                    else:
                        server_tests.append(("Pattern Analysis", False, f"API error: {response.status_code}"))
                        
                except Exception as e:
                    server_tests.append(("Pattern Analysis", False, f"Exception: {str(e)}"))
            else:
                server_tests.append(("Pattern Analysis", False, "Model training failed"))
        else:
            server_tests.append(("Pattern Analysis", False, "Data upload failed"))
        
        # Test 2: generate_advanced_extrapolation function
        print("\n--- Testing Advanced Extrapolation ---")
        try:
            if hasattr(self, 'model_id') and self.model_id:
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 25, "time_window": 75}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    
                    if len(predictions) == 25:
                        # Test multiple calls for extrapolation
                        response2 = self.session.get(
                            f"{API_BASE_URL}/generate-continuous-prediction",
                            params={"model_id": self.model_id, "steps": 25, "time_window": 75}
                        )
                        
                        if response2.status_code == 200:
                            data2 = response2.json()
                            predictions2 = data2.get('predictions', [])
                            
                            # Check if extrapolation is working (different predictions)
                            if predictions != predictions2:
                                server_tests.append(("Advanced Extrapolation", True, "Extrapolation working"))
                                print("✅ Advanced extrapolation working")
                            else:
                                server_tests.append(("Advanced Extrapolation", False, "No extrapolation detected"))
                        else:
                            server_tests.append(("Advanced Extrapolation", False, "Second call failed"))
                    else:
                        server_tests.append(("Advanced Extrapolation", False, f"Wrong prediction count: {len(predictions)}"))
                else:
                    server_tests.append(("Advanced Extrapolation", False, f"API error: {response.status_code}"))
            else:
                server_tests.append(("Advanced Extrapolation", False, "No model available"))
                
        except Exception as e:
            server_tests.append(("Advanced Extrapolation", False, f"Exception: {str(e)}"))
        
        # Test 3: create_smooth_transition function
        print("\n--- Testing Smooth Transition ---")
        try:
            # Test with pH data for smooth transitions
            ph_data = self.create_ph_data(samples=40)
            data_id, analysis = self.upload_test_data(ph_data, "smooth_transition_test.csv")
            
            if data_id:
                model_id = self.train_model(data_id, 'timestamp', 'ph_value', 'lstm')
                
                if model_id:
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": model_id, "steps": 20, "time_window": 40}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        predictions = data.get('predictions', [])
                        
                        if len(predictions) >= 5:
                            # Check smoothness of transitions
                            transitions = np.diff(predictions[:5])  # First 5 transitions
                            max_transition = np.max(np.abs(transitions))
                            
                            # For pH data, transitions should be smooth (< 0.5 pH units)
                            if max_transition < 0.5:
                                server_tests.append(("Smooth Transition", True, f"Max transition: {max_transition:.3f}"))
                                print(f"✅ Smooth transition working (max change: {max_transition:.3f})")
                            else:
                                server_tests.append(("Smooth Transition", False, f"Abrupt transition: {max_transition:.3f}"))
                        else:
                            server_tests.append(("Smooth Transition", False, "Insufficient predictions"))
                    else:
                        server_tests.append(("Smooth Transition", False, f"API error: {response.status_code}"))
                else:
                    server_tests.append(("Smooth Transition", False, "Model training failed"))
            else:
                server_tests.append(("Smooth Transition", False, "Data upload failed"))
                
        except Exception as e:
            server_tests.append(("Smooth Transition", False, f"Exception: {str(e)}"))
        
        # Evaluate server-side integration results
        passed_tests = sum(1 for _, success, _ in server_tests if success)
        total_tests = len(server_tests)
        
        print(f"\n📊 Server-Side Integration Results: {passed_tests}/{total_tests}")
        for test_name, success, details in server_tests:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}: {details}")
        
        self.test_results['server_side_integration'] = passed_tests >= total_tests * 0.67
    
    def test_prediction_quality_metrics(self):
        """Test 4: Key Prediction Quality Metrics"""
        print("\n=== Testing Prediction Quality Metrics ===")
        
        quality_tests = []
        
        # Test with different data patterns
        test_scenarios = [
            ("Sine Wave Pattern", self.create_sine_wave_data(periods=2, samples_per_period=40)),
            ("pH Stability Pattern", self.create_ph_data(samples=50)),
            ("Trending Pattern", self.create_random_walk_data(samples=60, drift=0.05))
        ]
        
        for scenario_name, test_data in test_scenarios:
            print(f"\n--- Testing {scenario_name} ---")
            
            # Determine columns based on data
            if 'ph_value' in test_data.columns:
                time_col, target_col = 'timestamp', 'ph_value'
            else:
                time_col, target_col = 'timestamp', 'value'
            
            data_id, analysis = self.upload_test_data(test_data, f"quality_{scenario_name.lower().replace(' ', '_')}.csv")
            
            if data_id:
                model_id = self.train_model(data_id, time_col, target_col, 'lstm')
                
                if model_id:
                    try:
                        response = self.session.get(
                            f"{API_BASE_URL}/generate-continuous-prediction",
                            params={"model_id": model_id, "steps": 30, "time_window": 50}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            predictions = data.get('predictions', [])
                            
                            if len(predictions) == 30:
                                # Analyze quality metrics
                                historical_data = test_data[target_col].values
                                quality_metrics = self.analyze_prediction_quality(
                                    historical_data, predictions, scenario_name
                                )
                                
                                # Check if quality meets thresholds
                                quality_passed = (
                                    quality_metrics.get('smoothness_score', 0) >= 70 and
                                    quality_metrics.get('overall_score', 0) >= 60
                                )
                                
                                quality_tests.append((scenario_name, quality_passed, quality_metrics))
                                
                                if quality_passed:
                                    print(f"✅ {scenario_name}: Quality metrics passed")
                                    print(f"   Overall Score: {quality_metrics.get('overall_score', 'N/A')}")
                                    print(f"   Smoothness: {quality_metrics.get('smoothness_score', 'N/A')}")
                                else:
                                    print(f"❌ {scenario_name}: Quality metrics below threshold")
                                    print(f"   Overall Score: {quality_metrics.get('overall_score', 'N/A')}")
                            else:
                                quality_tests.append((scenario_name, False, f"Wrong prediction count: {len(predictions)}"))
                        else:
                            quality_tests.append((scenario_name, False, f"API error: {response.status_code}"))
                            
                    except Exception as e:
                        quality_tests.append((scenario_name, False, f"Exception: {str(e)}"))
                else:
                    quality_tests.append((scenario_name, False, "Model training failed"))
            else:
                quality_tests.append((scenario_name, False, "Data upload failed"))
        
        # Evaluate quality test results
        passed_tests = sum(1 for _, success, _ in quality_tests if success)
        total_tests = len(quality_tests)
        
        print(f"\n📊 Prediction Quality Metrics Results: {passed_tests}/{total_tests}")
        for scenario, success, details in quality_tests:
            status = "✅" if success else "❌"
            if success and isinstance(details, dict):
                print(f"   {status} {scenario}: Overall Score {details.get('overall_score', 'N/A')}")
            else:
                print(f"   {status} {scenario}: {details if not success else 'PASSED'}")
        
        self.test_results['prediction_quality_metrics'] = passed_tests >= total_tests * 0.67
    
    def test_api_integration(self):
        """Test 5: Full API Integration"""
        print("\n=== Testing Full API Integration ===")
        
        api_tests = []
        
        # Test complete workflow
        print("\n--- Testing Complete Workflow ---")
        
        try:
            # Step 1: Upload data
            test_data = self.create_ph_data(samples=45)
            data_id, analysis = self.upload_test_data(test_data, "api_integration_test.csv")
            
            if data_id:
                api_tests.append(("Data Upload", True, f"Data ID: {data_id}"))
                
                # Step 2: Train model
                model_id = self.train_model(data_id, 'timestamp', 'ph_value', 'lstm')
                
                if model_id:
                    api_tests.append(("Model Training", True, f"Model ID: {model_id}"))
                    self.model_id = model_id  # Store for other tests
                    
                    # Step 3: Generate predictions
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-prediction",
                        params={"model_id": model_id, "steps": 20}
                    )
                    
                    if response.status_code == 200:
                        api_tests.append(("Basic Prediction", True, "Standard prediction working"))
                        
                        # Step 4: Generate continuous predictions
                        response = self.session.get(
                            f"{API_BASE_URL}/generate-continuous-prediction",
                            params={"model_id": model_id, "steps": 25, "time_window": 50}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            predictions = data.get('predictions', [])
                            
                            if len(predictions) == 25:
                                api_tests.append(("Enhanced Continuous Prediction", True, "Enhanced prediction working"))
                                
                                # Step 5: Test prediction reset
                                response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
                                
                                if response.status_code == 200:
                                    api_tests.append(("Prediction Reset", True, "Reset functionality working"))
                                else:
                                    api_tests.append(("Prediction Reset", False, f"Reset failed: {response.status_code}"))
                            else:
                                api_tests.append(("Enhanced Continuous Prediction", False, f"Wrong count: {len(predictions)}"))
                        else:
                            api_tests.append(("Enhanced Continuous Prediction", False, f"API error: {response.status_code}"))
                    else:
                        api_tests.append(("Basic Prediction", False, f"API error: {response.status_code}"))
                else:
                    api_tests.append(("Model Training", False, "Training failed"))
            else:
                api_tests.append(("Data Upload", False, "Upload failed"))
                
        except Exception as e:
            api_tests.append(("API Integration", False, f"Exception: {str(e)}"))
        
        # Evaluate API integration results
        passed_tests = sum(1 for _, success, _ in api_tests if success)
        total_tests = len(api_tests)
        
        print(f"\n📊 API Integration Results: {passed_tests}/{total_tests}")
        for test_name, success, details in api_tests:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}: {details}")
        
        self.test_results['api_integration'] = passed_tests >= total_tests * 0.8
    
    def run_all_tests(self):
        """Run all enhanced prediction system tests"""
        print("🚀 Starting Enhanced Prediction System Testing")
        print("=" * 60)
        
        # Run all test categories
        self.test_enhanced_prediction_core()
        self.test_integration_with_existing_models()
        self.test_server_side_integration()
        self.test_prediction_quality_metrics()
        self.test_api_integration()
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("🎯 ENHANCED PREDICTION SYSTEM TEST REPORT")
        print("=" * 60)
        
        test_categories = [
            ("Enhanced Prediction Core", self.test_results.get('enhanced_prediction_core', False)),
            ("Model Integration", self.test_results.get('model_integration', False)),
            ("Server-Side Integration", self.test_results.get('server_side_integration', False)),
            ("Prediction Quality Metrics", self.test_results.get('prediction_quality_metrics', False)),
            ("API Integration", self.test_results.get('api_integration', False))
        ]
        
        passed_categories = sum(1 for _, passed in test_categories if passed)
        total_categories = len(test_categories)
        
        print(f"\n📊 OVERALL RESULTS: {passed_categories}/{total_categories} categories passed")
        print(f"Success Rate: {(passed_categories/total_categories)*100:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        for category, passed in test_categories:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {status} {category}")
        
        # Overall assessment
        if passed_categories >= total_categories * 0.8:
            print(f"\n🎉 OVERALL ASSESSMENT: EXCELLENT")
            print("   Enhanced prediction system is working well with high quality predictions")
        elif passed_categories >= total_categories * 0.6:
            print(f"\n⚠️  OVERALL ASSESSMENT: GOOD")
            print("   Enhanced prediction system is mostly working but needs some improvements")
        else:
            print(f"\n❌ OVERALL ASSESSMENT: NEEDS IMPROVEMENT")
            print("   Enhanced prediction system requires significant fixes")
        
        print("\n" + "=" * 60)


def main():
    """Main test execution"""
    tester = EnhancedPredictionTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()