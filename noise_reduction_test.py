#!/usr/bin/env python3
"""
Enhanced Noise Reduction System Testing for Real-Time Continuous Prediction Graph
Tests the comprehensive noise reduction system as requested in the review
"""

import requests
import json
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, List, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://909a9d1c-9da6-4ed6-bd0a-ff6c4fb747bb.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"🎯 Testing Enhanced Noise Reduction System at: {API_BASE_URL}")

class NoiseReductionTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        self.data_uploaded = False
        self.data_id = None
        self.model_id = None
        
    def create_realistic_ph_data(self, num_points=50):
        """Create realistic pH data with various noise patterns for testing"""
        # Generate realistic pH data with different noise characteristics
        time_points = np.arange(num_points)
        
        # Base pH trend (realistic range 6.0-8.0)
        base_ph = 7.0 + 0.5 * np.sin(time_points * 0.2) + 0.2 * np.cos(time_points * 0.1)
        
        # Add different types of noise for comprehensive testing
        spike_noise = np.zeros(num_points)
        spike_indices = np.random.choice(num_points, size=5, replace=False)
        spike_noise[spike_indices] = np.random.choice([-1, 1], size=5) * np.random.uniform(0.5, 1.0, size=5)
        
        jitter_noise = np.random.normal(0, 0.1, num_points)
        oscillation_noise = 0.15 * np.sin(time_points * 2.0)
        
        # Combine different noise types
        noisy_ph = base_ph + spike_noise + jitter_noise + oscillation_noise
        
        # Ensure realistic pH range
        noisy_ph = np.clip(noisy_ph, 5.0, 9.0)
        
        # Create timestamps
        timestamps = pd.date_range(start='2024-01-01', periods=num_points, freq='H')
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'pH': noisy_ph
        })
        
        return df
    
    def upload_test_data(self):
        """Upload test pH data for noise reduction testing"""
        print("\n=== 📤 Uploading Test pH Data ===")
        
        try:
            # Create realistic pH data with noise
            df = self.create_realistic_ph_data(50)
            csv_content = df.to_csv(index=False)
            
            # Prepare file for upload
            files = {
                'file': ('ph_test_data.csv', csv_content, 'text/csv')
            }
            
            # Upload data
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                print("✅ Test data uploaded successfully")
                print(f"   Data ID: {self.data_id}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   pH column detected: {'pH' in data['analysis']['numeric_columns']}")
                self.data_uploaded = True
                return True
            else:
                print(f"❌ Data upload failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading data: {e}")
            return False
    
    def train_basic_model(self):
        """Train a basic model for testing predictions"""
        print("\n=== 🤖 Training Basic Model ===")
        
        try:
            if not self.data_id:
                print("❌ No data ID available for training")
                return False
            
            # Train ARIMA model (simpler and more reliable)
            training_params = {
                'data_id': self.data_id,
                'model_type': 'arima',
                'parameters': {
                    'time_column': 'timestamp',
                    'target_column': 'pH',
                    'order': [1, 1, 1]
                }
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={
                    'data_id': self.data_id,
                    'model_type': 'arima'
                },
                json={
                    'time_column': 'timestamp',
                    'target_column': 'pH',
                    'order': [1, 1, 1]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.model_id = data.get('model_id')
                print("✅ Basic model trained successfully")
                print(f"   Model ID: {self.model_id}")
                return True
            else:
                print(f"❌ Model training failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    def test_enhanced_realtime_prediction_endpoint(self):
        """Test 1: Enhanced Real-Time Prediction Endpoint with Noise Reduction"""
        print("\n=== 🎯 Testing Enhanced Real-Time Prediction Endpoint ===")
        
        try:
            # Test with different smoothing parameters
            test_params = [
                {'steps': 25, 'time_window': 100, 'maintain_patterns': True},
                {'steps': 15, 'time_window': 50, 'maintain_patterns': False},
                {'steps': 30, 'time_window': 150, 'maintain_patterns': True}
            ]
            
            success_count = 0
            total_tests = len(test_params)
            
            for i, params in enumerate(test_params):
                print(f"\n   Test {i+1}/{total_tests}: steps={params['steps']}, window={params['time_window']}")
                
                response = self.session.get(
                    f"{API_BASE_URL}/generate-enhanced-realtime-prediction",
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    
                    # Validate predictions
                    if len(predictions) == params['steps']:
                        # Check if predictions are in realistic pH range
                        ph_values = np.array(predictions)
                        in_range = np.all((ph_values >= 5.0) & (ph_values <= 9.0))
                        
                        # Check for smoothness (noise reduction effectiveness)
                        if len(predictions) > 2:
                            differences = np.diff(predictions)
                            smoothness_score = 1.0 / (1.0 + np.std(differences))
                        else:
                            smoothness_score = 1.0
                        
                        print(f"   ✅ Generated {len(predictions)} predictions")
                        print(f"   ✅ pH range valid: {in_range} (range: {ph_values.min():.2f}-{ph_values.max():.2f})")
                        print(f"   ✅ Smoothness score: {smoothness_score:.3f}")
                        
                        # Check for noise reduction info
                        if 'noise_reduction_info' in data:
                            noise_info = data['noise_reduction_info']
                            print(f"   ✅ Noise reduction applied: {noise_info.get('smoothing_applied', [])}")
                            print(f"   ✅ Noise reduction score: {noise_info.get('noise_reduction_score', 0):.3f}")
                        
                        success_count += 1
                    else:
                        print(f"   ❌ Expected {params['steps']} predictions, got {len(predictions)}")
                else:
                    print(f"   ❌ Request failed: {response.status_code} - {response.text}")
            
            success_rate = success_count / total_tests
            self.test_results['enhanced_realtime_prediction'] = success_rate >= 0.8
            print(f"\n🎯 Enhanced Real-Time Prediction Test: {success_count}/{total_tests} passed ({success_rate*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error testing enhanced real-time prediction: {e}")
            self.test_results['enhanced_realtime_prediction'] = False
    
    def test_advanced_ph_prediction_endpoints(self):
        """Test 2: Advanced pH Prediction and Extension Endpoints"""
        print("\n=== 🧪 Testing Advanced pH Prediction Endpoints ===")
        
        try:
            # Test 2a: Generate Advanced pH Prediction
            print("\n   Testing generate-advanced-ph-prediction...")
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-advanced-ph-prediction",
                params={'steps': 20, 'maintain_patterns': True}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if len(predictions) == 20:
                    ph_values = np.array(predictions)
                    in_range = np.all((ph_values >= 5.0) & (ph_values <= 9.0))
                    
                    print(f"   ✅ Generated {len(predictions)} advanced pH predictions")
                    print(f"   ✅ pH range valid: {in_range} (range: {ph_values.min():.2f}-{ph_values.max():.2f})")
                    
                    # Test 2b: Extend Advanced pH Prediction
                    print("\n   Testing extend-advanced-ph-prediction...")
                    
                    extend_response = self.session.get(
                        f"{API_BASE_URL}/extend-advanced-ph-prediction",
                        params={'additional_steps': 10}
                    )
                    
                    if extend_response.status_code == 200:
                        extend_data = extend_response.json()
                        extended_predictions = extend_data.get('predictions', [])
                        
                        if len(extended_predictions) == 10:
                            extended_ph = np.array(extended_predictions)
                            extended_in_range = np.all((extended_ph >= 5.0) & (extended_ph <= 9.0))
                            
                            print(f"   ✅ Extended with {len(extended_predictions)} additional predictions")
                            print(f"   ✅ Extended pH range valid: {extended_in_range}")
                            
                            # Check for smooth transitions
                            if len(predictions) > 0 and len(extended_predictions) > 0:
                                transition_diff = abs(extended_predictions[0] - predictions[-1])
                                smooth_transition = transition_diff < 0.5  # Reasonable transition
                                print(f"   ✅ Smooth transition: {smooth_transition} (diff: {transition_diff:.3f})")
                            
                            self.test_results['advanced_ph_prediction'] = True
                        else:
                            print(f"   ❌ Expected 10 extended predictions, got {len(extended_predictions)}")
                            self.test_results['advanced_ph_prediction'] = False
                    else:
                        print(f"   ❌ Extension failed: {extend_response.status_code} - {extend_response.text}")
                        self.test_results['advanced_ph_prediction'] = False
                else:
                    print(f"   ❌ Expected 20 predictions, got {len(predictions)}")
                    self.test_results['advanced_ph_prediction'] = False
            else:
                print(f"   ❌ Advanced pH prediction failed: {response.status_code} - {response.text}")
                self.test_results['advanced_ph_prediction'] = False
                
        except Exception as e:
            print(f"❌ Error testing advanced pH prediction: {e}")
            self.test_results['advanced_ph_prediction'] = False
    
    def test_noise_reduction_integration(self):
        """Test 3: Verify Advanced Noise Reduction System Integration"""
        print("\n=== 🔧 Testing Noise Reduction System Integration ===")
        
        try:
            # Create test data with different noise patterns
            test_cases = [
                {'name': 'Smooth Data', 'noise_type': 'minimal'},
                {'name': 'Spike Noise', 'noise_type': 'spikes'},
                {'name': 'Jitter Noise', 'noise_type': 'jitter'},
                {'name': 'Oscillation Noise', 'noise_type': 'oscillation'}
            ]
            
            success_count = 0
            
            for test_case in test_cases:
                print(f"\n   Testing {test_case['name']}...")
                
                # Generate specific noise pattern data
                df = self._create_noise_pattern_data(test_case['noise_type'])
                csv_content = df.to_csv(index=False)
                
                # Upload test data
                files = {'file': (f'noise_test_{test_case["noise_type"]}.csv', csv_content, 'text/csv')}
                upload_response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if upload_response.status_code == 200:
                    upload_data = upload_response.json()
                    temp_data_id = upload_data.get('data_id')
                    
                    # Train a model with this data
                    training_params = {
                        'data_id': temp_data_id,
                        'model_type': 'arima',
                        'time_column': 'timestamp',
                        'target_column': 'pH',
                        'parameters': {'order': [1, 1, 1]}
                    }
                    
                    train_response = self.session.post(f"{API_BASE_URL}/train-model", json=training_params)
                    
                    if train_response.status_code == 200:
                        # Test enhanced prediction with this data
                        pred_response = self.session.get(
                            f"{API_BASE_URL}/generate-enhanced-realtime-prediction",
                            params={'steps': 15, 'maintain_patterns': True}
                        )
                        
                        if pred_response.status_code == 200:
                            data = pred_response.json()
                            predictions = data.get('predictions', [])
                            
                            if len(predictions) > 0:
                                # Analyze noise reduction effectiveness
                                noise_reduction_score = self._analyze_noise_reduction_effectiveness(
                                    predictions, test_case['noise_type']
                                )
                                
                                print(f"   ✅ {test_case['name']}: Noise reduction score {noise_reduction_score:.3f}")
                                
                                if noise_reduction_score > 0.3:  # Threshold for effective noise reduction
                                    success_count += 1
                                else:
                                    print(f"   ⚠️ Low noise reduction effectiveness for {test_case['name']}")
                            else:
                                print(f"   ❌ No predictions generated for {test_case['name']}")
                        else:
                            print(f"   ❌ Prediction failed for {test_case['name']}: {pred_response.status_code}")
                    else:
                        print(f"   ❌ Model training failed for {test_case['name']}")
                else:
                    print(f"   ❌ Data upload failed for {test_case['name']}")
            
            success_rate = success_count / len(test_cases)
            self.test_results['noise_reduction_integration'] = success_rate >= 0.75
            print(f"\n🔧 Noise Reduction Integration Test: {success_count}/{len(test_cases)} passed ({success_rate*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error testing noise reduction integration: {e}")
            self.test_results['noise_reduction_integration'] = False
    
    def test_continuous_prediction_flow(self):
        """Test 4: Complete Continuous Prediction Flow"""
        print("\n=== 🔄 Testing Continuous Prediction Flow ===")
        
        try:
            # Step 1: Reset system
            print("   Step 1: Resetting advanced pH engine...")
            reset_response = self.session.post(f"{API_BASE_URL}/reset-advanced-ph-engine")
            
            if reset_response.status_code == 200:
                print("   ✅ System reset successful")
            else:
                print(f"   ⚠️ Reset failed: {reset_response.status_code}")
            
            # Step 2: Start with initial prediction
            print("   Step 2: Generating initial predictions...")
            initial_response = self.session.get(
                f"{API_BASE_URL}/generate-advanced-ph-prediction",
                params={'steps': 15, 'maintain_patterns': True}
            )
            
            initial_predictions = []
            if initial_response.status_code == 200:
                initial_data = initial_response.json()
                initial_predictions = initial_data.get('predictions', [])
                print(f"   ✅ Generated {len(initial_predictions)} initial predictions")
            else:
                print(f"   ❌ Initial prediction failed: {initial_response.status_code}")
                self.test_results['continuous_prediction_flow'] = False
                return
            
            # Step 3: Extend predictions multiple times
            print("   Step 3: Extending predictions...")
            all_predictions = initial_predictions.copy()
            
            for i in range(3):  # Extend 3 times
                extend_response = self.session.get(
                    f"{API_BASE_URL}/extend-advanced-ph-prediction",
                    params={'additional_steps': 5}
                )
                
                if extend_response.status_code == 200:
                    extend_data = extend_response.json()
                    extended_predictions = extend_data.get('predictions', [])
                    all_predictions.extend(extended_predictions)
                    print(f"   ✅ Extension {i+1}: Added {len(extended_predictions)} predictions")
                else:
                    print(f"   ❌ Extension {i+1} failed: {extend_response.status_code}")
            
            # Step 4: Analyze complete flow
            if len(all_predictions) >= 30:  # 15 initial + 3*5 extensions
                # Check for smooth transitions throughout the flow
                smoothness_score = self._calculate_smoothness_score(all_predictions)
                ph_range_valid = all(5.0 <= p <= 9.0 for p in all_predictions)
                
                print(f"   ✅ Total predictions generated: {len(all_predictions)}")
                print(f"   ✅ pH range valid: {ph_range_valid}")
                print(f"   ✅ Overall smoothness score: {smoothness_score:.3f}")
                
                self.test_results['continuous_prediction_flow'] = (
                    smoothness_score > 0.5 and ph_range_valid
                )
            else:
                print(f"   ❌ Insufficient predictions generated: {len(all_predictions)}")
                self.test_results['continuous_prediction_flow'] = False
            
        except Exception as e:
            print(f"❌ Error testing continuous prediction flow: {e}")
            self.test_results['continuous_prediction_flow'] = False
    
    def test_pattern_preservation(self):
        """Test 5: Verify Pattern Preservation During Noise Reduction"""
        print("\n=== 📊 Testing Pattern Preservation ===")
        
        try:
            # Create data with known patterns
            pattern_tests = [
                {'name': 'Sine Wave Pattern', 'pattern_type': 'sine'},
                {'name': 'Linear Trend Pattern', 'pattern_type': 'linear'}
            ]
            
            success_count = 0
            
            for test in pattern_tests:
                print(f"\n   Testing {test['name']}...")
                
                # Create pattern data
                df = self._create_pattern_data(test['pattern_type'])
                csv_content = df.to_csv(index=False)
                
                # Upload pattern data
                files = {'file': (f'pattern_test_{test["pattern_type"]}.csv', csv_content, 'text/csv')}
                upload_response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if upload_response.status_code == 200:
                    # Generate predictions
                    pred_response = self.session.get(
                        f"{API_BASE_URL}/generate-enhanced-realtime-prediction",
                        params={'steps': 20, 'maintain_patterns': True}
                    )
                    
                    if pred_response.status_code == 200:
                        data = pred_response.json()
                        predictions = data.get('predictions', [])
                        
                        if len(predictions) > 0:
                            # Analyze pattern preservation
                            preservation_score = self._analyze_pattern_preservation(
                                predictions, test['pattern_type']
                            )
                            
                            print(f"   ✅ {test['name']}: Pattern preservation score {preservation_score:.3f}")
                            
                            if preservation_score > 0.6:  # Threshold for good pattern preservation
                                success_count += 1
                            else:
                                print(f"   ⚠️ Low pattern preservation for {test['name']}")
                        else:
                            print(f"   ❌ No predictions generated for {test['name']}")
                    else:
                        print(f"   ❌ Prediction failed for {test['name']}")
                else:
                    print(f"   ❌ Data upload failed for {test['name']}")
            
            success_rate = success_count / len(pattern_tests)
            self.test_results['pattern_preservation'] = success_rate >= 0.5
            print(f"\n📊 Pattern Preservation Test: {success_count}/{len(pattern_tests)} passed ({success_rate*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error testing pattern preservation: {e}")
            self.test_results['pattern_preservation'] = False
    
    def _create_noise_pattern_data(self, noise_type: str) -> pd.DataFrame:
        """Create data with specific noise patterns for testing"""
        num_points = 40
        time_points = np.arange(num_points)
        base_ph = 7.0 + 0.3 * np.sin(time_points * 0.3)
        
        if noise_type == 'minimal':
            noise = np.random.normal(0, 0.02, num_points)
        elif noise_type == 'spikes':
            noise = np.zeros(num_points)
            spike_indices = np.random.choice(num_points, size=8, replace=False)
            noise[spike_indices] = np.random.choice([-1, 1], size=8) * np.random.uniform(0.8, 1.5, size=8)
        elif noise_type == 'jitter':
            noise = np.random.normal(0, 0.2, num_points)
        elif noise_type == 'oscillation':
            noise = 0.3 * np.sin(time_points * 3.0)
        else:
            noise = np.zeros(num_points)
        
        noisy_ph = np.clip(base_ph + noise, 5.0, 9.0)
        timestamps = pd.date_range(start='2024-01-01', periods=num_points, freq='H')
        
        return pd.DataFrame({'timestamp': timestamps, 'pH': noisy_ph})
    
    def _create_pattern_data(self, pattern_type: str) -> pd.DataFrame:
        """Create data with specific patterns for preservation testing"""
        num_points = 50
        time_points = np.arange(num_points)
        
        if pattern_type == 'sine':
            ph_values = 7.0 + 0.8 * np.sin(time_points * 0.4)
        elif pattern_type == 'linear':
            ph_values = 6.5 + 0.03 * time_points
        else:
            ph_values = np.full(num_points, 7.0)
        
        # Add minimal noise
        ph_values += np.random.normal(0, 0.05, num_points)
        ph_values = np.clip(ph_values, 5.0, 9.0)
        
        timestamps = pd.date_range(start='2024-01-01', periods=num_points, freq='H')
        return pd.DataFrame({'timestamp': timestamps, 'pH': ph_values})
    
    def _analyze_noise_reduction_effectiveness(self, predictions: List[float], noise_type: str) -> float:
        """Analyze how effectively noise was reduced"""
        if len(predictions) < 3:
            return 0.0
        
        # Calculate smoothness metrics
        differences = np.diff(predictions)
        smoothness = 1.0 / (1.0 + np.std(differences))
        
        # Calculate variability reduction
        variability = np.std(predictions)
        variability_score = 1.0 / (1.0 + variability)
        
        # Combine metrics based on noise type
        if noise_type == 'spikes':
            # For spike noise, focus on smoothness
            return smoothness * 0.8 + variability_score * 0.2
        elif noise_type == 'jitter':
            # For jitter, balance smoothness and variability
            return smoothness * 0.6 + variability_score * 0.4
        else:
            # General case
            return smoothness * 0.7 + variability_score * 0.3
    
    def _analyze_pattern_preservation(self, predictions: List[float], pattern_type: str) -> float:
        """Analyze how well patterns were preserved during noise reduction"""
        if len(predictions) < 5:
            return 0.0
        
        predictions_array = np.array(predictions)
        
        if pattern_type == 'sine':
            # For sine patterns, check for oscillatory behavior
            # Calculate autocorrelation to detect periodicity
            autocorr = np.correlate(predictions_array, predictions_array, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for peaks in autocorrelation (indicating periodicity)
            if len(autocorr) > 5:
                peak_strength = np.max(autocorr[1:5]) / autocorr[0] if autocorr[0] > 0 else 0
                return min(1.0, peak_strength)
            else:
                return 0.5
                
        elif pattern_type == 'linear':
            # For linear patterns, check trend consistency
            time_points = np.arange(len(predictions))
            correlation = np.corrcoef(time_points, predictions_array)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.5  # Default score
    
    def _calculate_smoothness_score(self, predictions: List[float]) -> float:
        """Calculate overall smoothness score for predictions"""
        if len(predictions) < 2:
            return 0.0
        
        differences = np.diff(predictions)
        smoothness = 1.0 / (1.0 + np.std(differences))
        return smoothness
    
    def run_comprehensive_test(self):
        """Run all noise reduction tests"""
        print("🎯 COMPREHENSIVE NOISE REDUCTION SYSTEM TESTING STARTED")
        print("=" * 70)
        
        # Upload test data first
        if not self.upload_test_data():
            print("❌ Failed to upload test data. Cannot proceed with testing.")
            return
        
        # Train basic model for predictions
        if not self.train_basic_model():
            print("❌ Failed to train basic model. Cannot proceed with prediction testing.")
            return
        
        # Run all tests
        self.test_enhanced_realtime_prediction_endpoint()
        self.test_advanced_ph_prediction_endpoints()
        self.test_noise_reduction_integration()
        self.test_continuous_prediction_flow()
        self.test_pattern_preservation()
        
        # Generate final report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("🎯 COMPREHENSIVE NOISE REDUCTION SYSTEM TEST RESULTS")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        print("\n📋 DETAILED RESULTS:")
        test_descriptions = {
            'enhanced_realtime_prediction': 'Enhanced Real-Time Prediction Endpoint',
            'advanced_ph_prediction': 'Advanced pH Prediction Endpoints',
            'noise_reduction_integration': 'Noise Reduction System Integration',
            'continuous_prediction_flow': 'Continuous Prediction Flow',
            'pattern_preservation': 'Pattern Preservation During Noise Reduction'
        }
        
        for test_key, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            description = test_descriptions.get(test_key, test_key)
            print(f"   {status} - {description}")
        
        print("\n🎯 KEY FINDINGS:")
        if self.test_results.get('enhanced_realtime_prediction', False):
            print("   ✅ Enhanced real-time prediction endpoint working with noise reduction")
        else:
            print("   ❌ Enhanced real-time prediction endpoint has issues")
            
        if self.test_results.get('advanced_ph_prediction', False):
            print("   ✅ Advanced pH prediction and extension endpoints working")
        else:
            print("   ❌ Advanced pH prediction endpoints have issues")
            
        if self.test_results.get('noise_reduction_integration', False):
            print("   ✅ Advanced noise reduction system properly integrated")
        else:
            print("   ❌ Noise reduction system integration has issues")
            
        if self.test_results.get('continuous_prediction_flow', False):
            print("   ✅ Complete continuous prediction flow working smoothly")
        else:
            print("   ❌ Continuous prediction flow has issues")
            
        if self.test_results.get('pattern_preservation', False):
            print("   ✅ Pattern preservation during noise reduction verified")
        else:
            print("   ❌ Pattern preservation needs improvement")
        
        print("\n🎉 CONCLUSION:")
        if success_rate >= 80:
            print("   🎯 EXCELLENT: Enhanced noise reduction system is working correctly!")
            print("   🎯 The system successfully reduces noise while preserving patterns.")
            print("   🎯 Real-time continuous predictions are smooth and realistic.")
        elif success_rate >= 60:
            print("   ⚠️ GOOD: Most noise reduction features are working with minor issues.")
            print("   ⚠️ Some improvements needed for optimal performance.")
        else:
            print("   ❌ NEEDS WORK: Significant issues found in noise reduction system.")
            print("   ❌ Multiple components require fixes for proper functionality.")
        
        print("=" * 70)

if __name__ == "__main__":
    tester = NoiseReductionTester()
    tester.run_comprehensive_test()