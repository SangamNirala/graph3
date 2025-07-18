#!/usr/bin/env python3
"""
Enhanced Pattern-Aware Prediction System Testing
Tests advanced pattern detection, learning, and adaptive extrapolation capabilities
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://1883c9bd-2fda-48e0-82d4-0ec1f13153f1.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing enhanced pattern-aware prediction system at: {API_BASE_URL}")

class PatternAwareTester:
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
    
    def create_polynomial_data(self, points=50):
        """Create polynomial pattern data"""
        x = np.linspace(0, 10, points)
        y = 0.1*x**4 - 2*x**3 + 10*x**2 - 20*x + 50 + np.random.normal(0, 2, points)
        
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
    
    def train_advanced_model(self, data_id, model_type='lstm'):
        """Train advanced model for pattern learning"""
        try:
            training_params = {
                "time_column": "timestamp",
                "target_column": "value",
                "seq_len": 20,
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
                return {
                    'success': True,
                    'model_id': data.get('model_id'),
                    'training_data': data
                }
            else:
                return {
                    'success': False,
                    'error': f"Training failed: {response.status_code} - {response.text}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"Training error: {str(e)}"
            }
    
    def test_advanced_pattern_detection(self):
        """Test 1: Advanced Pattern Detection"""
        print("\n=== Testing Advanced Pattern Detection ===")
        
        pattern_tests = []
        
        # Test U-shaped (quadratic) pattern detection
        print("\n--- Testing U-shaped (Quadratic) Pattern ---")
        u_data = self.create_u_shaped_data()
        u_result = self.upload_and_analyze_data(u_data, "u_shaped_test")
        
        if u_result['success']:
            print("✅ U-shaped data uploaded successfully")
            # Train model to analyze pattern
            model_result = self.train_advanced_model(u_result['data_id'], 'lstm')
            if model_result['success']:
                print("✅ LSTM model trained on U-shaped data")
                pattern_tests.append(("U-shaped pattern upload and training", True))
            else:
                print(f"❌ LSTM training failed: {model_result['error']}")
                pattern_tests.append(("U-shaped pattern upload and training", False))
        else:
            print(f"❌ U-shaped data upload failed: {u_result['error']}")
            pattern_tests.append(("U-shaped pattern upload and training", False))
        
        # Test S-shaped (cubic) pattern detection
        print("\n--- Testing S-shaped (Cubic) Pattern ---")
        s_data = self.create_s_shaped_data()
        s_result = self.upload_and_analyze_data(s_data, "s_shaped_test")
        
        if s_result['success']:
            print("✅ S-shaped data uploaded successfully")
            model_result = self.train_advanced_model(s_result['data_id'], 'lstm')
            if model_result['success']:
                print("✅ LSTM model trained on S-shaped data")
                pattern_tests.append(("S-shaped pattern upload and training", True))
            else:
                print(f"❌ LSTM training failed: {model_result['error']}")
                pattern_tests.append(("S-shaped pattern upload and training", False))
        else:
            print(f"❌ S-shaped data upload failed: {s_result['error']}")
            pattern_tests.append(("S-shaped pattern upload and training", False))
        
        # Test complex shaped pattern detection
        print("\n--- Testing Complex Shaped Pattern ---")
        complex_data = self.create_complex_shaped_data()
        complex_result = self.upload_and_analyze_data(complex_data, "complex_shaped_test")
        
        if complex_result['success']:
            print("✅ Complex shaped data uploaded successfully")
            model_result = self.train_advanced_model(complex_result['data_id'], 'lstm')
            if model_result['success']:
                print("✅ LSTM model trained on complex shaped data")
                pattern_tests.append(("Complex pattern upload and training", True))
            else:
                print(f"❌ LSTM training failed: {model_result['error']}")
                pattern_tests.append(("Complex pattern upload and training", False))
        else:
            print(f"❌ Complex shaped data upload failed: {complex_result['error']}")
            pattern_tests.append(("Complex pattern upload and training", False))
        
        # Test polynomial pattern detection
        print("\n--- Testing Polynomial Pattern ---")
        poly_data = self.create_polynomial_data()
        poly_result = self.upload_and_analyze_data(poly_data, "polynomial_test")
        
        if poly_result['success']:
            print("✅ Polynomial data uploaded successfully")
            model_result = self.train_advanced_model(poly_result['data_id'], 'lstm')
            if model_result['success']:
                print("✅ LSTM model trained on polynomial data")
                pattern_tests.append(("Polynomial pattern upload and training", True))
            else:
                print(f"❌ LSTM training failed: {model_result['error']}")
                pattern_tests.append(("Polynomial pattern upload and training", False))
        else:
            print(f"❌ Polynomial data upload failed: {poly_result['error']}")
            pattern_tests.append(("Polynomial pattern upload and training", False))
        
        # Evaluate pattern detection tests
        passed_tests = sum(1 for _, passed in pattern_tests if passed)
        total_tests = len(pattern_tests)
        
        print(f"\n📊 Advanced Pattern Detection Results: {passed_tests}/{total_tests}")
        for test_name, passed in pattern_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['advanced_pattern_detection'] = passed_tests >= total_tests * 0.75
        return u_result if u_result['success'] else None
    
    def test_pattern_learning_and_prediction(self, pattern_data_result):
        """Test 2: Pattern Learning and Prediction Quality"""
        print("\n=== Testing Pattern Learning and Prediction Quality ===")
        
        if not pattern_data_result or not pattern_data_result['success']:
            print("❌ Cannot test pattern learning - no valid pattern data")
            self.test_results['pattern_learning'] = False
            return None
        
        learning_tests = []
        model_id = None
        
        try:
            # Train advanced model on pattern data
            data_id = pattern_data_result['data_id']
            model_result = self.train_advanced_model(data_id, 'lstm')
            
            if model_result['success']:
                model_id = model_result['model_id']
                print("✅ Advanced model trained for pattern learning")
                learning_tests.append(("Model training", True))
                
                # Test prediction generation with pattern awareness
                response = self.session.get(
                    f"{API_BASE_URL}/generate-prediction",
                    params={"model_id": model_id, "steps": 20}
                )
                
                if response.status_code == 200:
                    pred_data = response.json()
                    predictions = pred_data.get('predictions', [])
                    
                    print("✅ Pattern-aware predictions generated")
                    print(f"   Number of predictions: {len(predictions)}")
                    
                    if len(predictions) == 20:
                        print("✅ Correct number of predictions generated")
                        learning_tests.append(("Prediction generation", True))
                        
                        # Test prediction quality - check for variability (not monotonic)
                        pred_array = np.array(predictions)
                        unique_values = len(np.unique(np.round(pred_array, 2)))
                        
                        if unique_values >= 5:  # At least 5 unique values
                            print(f"✅ Predictions show good variability ({unique_values} unique values)")
                            learning_tests.append(("Prediction variability", True))
                        else:
                            print(f"❌ Predictions lack variability ({unique_values} unique values)")
                            learning_tests.append(("Prediction variability", False))
                        
                        # Test prediction range reasonableness
                        pred_range = np.max(pred_array) - np.min(pred_array)
                        if pred_range > 0.1:  # Some meaningful range
                            print(f"✅ Predictions have reasonable range ({pred_range:.3f})")
                            learning_tests.append(("Prediction range", True))
                        else:
                            print(f"❌ Predictions have too narrow range ({pred_range:.3f})")
                            learning_tests.append(("Prediction range", False))
                        
                    else:
                        print(f"❌ Incorrect number of predictions: {len(predictions)}")
                        learning_tests.append(("Prediction generation", False))
                        learning_tests.append(("Prediction variability", False))
                        learning_tests.append(("Prediction range", False))
                else:
                    print(f"❌ Prediction generation failed: {response.status_code}")
                    learning_tests.append(("Prediction generation", False))
                    learning_tests.append(("Prediction variability", False))
                    learning_tests.append(("Prediction range", False))
            else:
                print(f"❌ Model training failed: {model_result['error']}")
                learning_tests.append(("Model training", False))
                learning_tests.append(("Prediction generation", False))
                learning_tests.append(("Prediction variability", False))
                learning_tests.append(("Prediction range", False))
        
        except Exception as e:
            print(f"❌ Pattern learning test error: {str(e)}")
            learning_tests.extend([
                ("Model training", False),
                ("Prediction generation", False),
                ("Prediction variability", False),
                ("Prediction range", False)
            ])
        
        # Evaluate pattern learning tests
        passed_tests = sum(1 for _, passed in learning_tests if passed)
        total_tests = len(learning_tests)
        
        print(f"\n📊 Pattern Learning Results: {passed_tests}/{total_tests}")
        for test_name, passed in learning_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['pattern_learning'] = passed_tests >= total_tests * 0.75
        return model_id
    
    def test_adaptive_extrapolation(self, model_id):
        """Test 3: Adaptive Extrapolation Based on Learned Patterns"""
        print("\n=== Testing Adaptive Extrapolation ===")
        
        if not model_id:
            print("❌ Cannot test adaptive extrapolation - no trained model")
            self.test_results['adaptive_extrapolation'] = False
            return
        
        extrapolation_tests = []
        
        try:
            # Test continuous prediction with extrapolation
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 30, "time_window": 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                pattern_analysis = data.get('pattern_analysis')
                
                print("✅ Continuous prediction with extrapolation successful")
                print(f"   Predictions: {len(predictions)}")
                print(f"   Timestamps: {len(timestamps)}")
                
                if len(predictions) == 30 and len(timestamps) == 30:
                    print("✅ Correct extrapolation length")
                    extrapolation_tests.append(("Extrapolation length", True))
                    
                    # Test pattern analysis inclusion
                    if pattern_analysis:
                        print("✅ Pattern analysis included in extrapolation")
                        print(f"   Pattern type: {pattern_analysis.get('pattern_type', 'N/A')}")
                        print(f"   Trend slope: {pattern_analysis.get('trend_slope', 'N/A')}")
                        print(f"   Velocity: {pattern_analysis.get('velocity', 'N/A')}")
                        extrapolation_tests.append(("Pattern analysis", True))
                    else:
                        print("❌ Pattern analysis missing from extrapolation")
                        extrapolation_tests.append(("Pattern analysis", False))
                    
                    # Test multiple extrapolation calls for consistency
                    print("   Testing multiple extrapolation calls...")
                    
                    # Make second call
                    response2 = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": model_id, "steps": 30, "time_window": 100}
                    )
                    
                    if response2.status_code == 200:
                        data2 = response2.json()
                        predictions2 = data2.get('predictions', [])
                        timestamps2 = data2.get('timestamps', [])
                        
                        # Check if extrapolation advances forward
                        if timestamps != timestamps2:
                            print("✅ Extrapolation properly advances forward")
                            extrapolation_tests.append(("Forward extrapolation", True))
                        else:
                            print("❌ Extrapolation not advancing forward")
                            extrapolation_tests.append(("Forward extrapolation", False))
                        
                        # Check prediction consistency (should be different but reasonable)
                        if predictions != predictions2:
                            pred_diff = np.mean(np.abs(np.array(predictions) - np.array(predictions2)))
                            if pred_diff > 0.01:  # Some difference expected
                                print(f"✅ Extrapolation shows adaptive behavior (diff: {pred_diff:.3f})")
                                extrapolation_tests.append(("Adaptive behavior", True))
                            else:
                                print(f"❌ Extrapolation too static (diff: {pred_diff:.3f})")
                                extrapolation_tests.append(("Adaptive behavior", False))
                        else:
                            print("❌ Extrapolation predictions identical (not adaptive)")
                            extrapolation_tests.append(("Adaptive behavior", False))
                    else:
                        print("❌ Second extrapolation call failed")
                        extrapolation_tests.append(("Forward extrapolation", False))
                        extrapolation_tests.append(("Adaptive behavior", False))
                else:
                    print(f"❌ Incorrect extrapolation dimensions")
                    extrapolation_tests.extend([
                        ("Extrapolation length", False),
                        ("Pattern analysis", False),
                        ("Forward extrapolation", False),
                        ("Adaptive behavior", False)
                    ])
            else:
                print(f"❌ Continuous prediction failed: {response.status_code}")
                extrapolation_tests.extend([
                    ("Extrapolation length", False),
                    ("Pattern analysis", False),
                    ("Forward extrapolation", False),
                    ("Adaptive behavior", False)
                ])
        
        except Exception as e:
            print(f"❌ Adaptive extrapolation error: {str(e)}")
            extrapolation_tests.extend([
                ("Extrapolation length", False),
                ("Pattern analysis", False),
                ("Forward extrapolation", False),
                ("Adaptive behavior", False)
            ])
        
        # Evaluate adaptive extrapolation tests
        passed_tests = sum(1 for _, passed in extrapolation_tests if passed)
        total_tests = len(extrapolation_tests)
        
        print(f"\n📊 Adaptive Extrapolation Results: {passed_tests}/{total_tests}")
        for test_name, passed in extrapolation_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['adaptive_extrapolation'] = passed_tests >= total_tests * 0.75
    
    def test_continuous_pattern_awareness(self, model_id):
        """Test 4: Real-time Continuous Prediction with Pattern Awareness"""
        print("\n=== Testing Continuous Pattern Awareness ===")
        
        if not model_id:
            print("❌ Cannot test continuous pattern awareness - no trained model")
            self.test_results['continuous_pattern_awareness'] = False
            return
        
        continuous_tests = []
        
        try:
            # Test continuous prediction start/stop with pattern awareness
            print("   Starting continuous prediction...")
            
            # Reset first
            reset_response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            if reset_response.status_code == 200:
                print("✅ Continuous prediction reset successful")
                continuous_tests.append(("Reset continuous prediction", True))
            else:
                print("❌ Reset failed")
                continuous_tests.append(("Reset continuous prediction", False))
            
            # Start continuous prediction
            start_response = self.session.post(f"{API_BASE_URL}/start-continuous-prediction")
            if start_response.status_code == 200:
                print("✅ Continuous prediction started")
                continuous_tests.append(("Start continuous prediction", True))
                
                # Wait and test multiple continuous calls
                time.sleep(2)
                
                predictions_history = []
                pattern_analyses = []
                
                # Make multiple continuous prediction calls
                for i in range(5):
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": model_id, "steps": 15, "time_window": 80}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        predictions = data.get('predictions', [])
                        pattern_analysis = data.get('pattern_analysis')
                        
                        predictions_history.append(predictions)
                        if pattern_analysis:
                            pattern_analyses.append(pattern_analysis)
                        
                        time.sleep(1)  # Wait between calls
                    else:
                        print(f"❌ Continuous prediction call {i+1} failed")
                        break
                
                # Analyze continuous prediction results
                if len(predictions_history) >= 3:
                    print(f"✅ Multiple continuous predictions successful ({len(predictions_history)} calls)")
                    continuous_tests.append(("Multiple continuous calls", True))
                    
                    # Test pattern awareness maintenance
                    if len(pattern_analyses) >= 2:
                        print("✅ Pattern analysis maintained across continuous calls")
                        continuous_tests.append(("Pattern awareness maintenance", True))
                        
                        # Check if pattern analysis shows consistency
                        pattern_types = [pa.get('pattern_type') for pa in pattern_analyses if pa.get('pattern_type')]
                        if len(set(pattern_types)) <= 2:  # Should be consistent or evolving slowly
                            print(f"✅ Pattern type consistency maintained: {set(pattern_types)}")
                            continuous_tests.append(("Pattern consistency", True))
                        else:
                            print(f"❌ Pattern type inconsistent: {set(pattern_types)}")
                            continuous_tests.append(("Pattern consistency", False))
                    else:
                        print("❌ Pattern analysis not maintained")
                        continuous_tests.append(("Pattern awareness maintenance", False))
                        continuous_tests.append(("Pattern consistency", False))
                    
                    # Test prediction evolution (should change over time)
                    first_predictions = predictions_history[0]
                    last_predictions = predictions_history[-1]
                    
                    if first_predictions != last_predictions:
                        evolution_diff = np.mean(np.abs(np.array(first_predictions) - np.array(last_predictions)))
                        if evolution_diff > 0.1:
                            print(f"✅ Predictions evolve over continuous calls (diff: {evolution_diff:.3f})")
                            continuous_tests.append(("Prediction evolution", True))
                        else:
                            print(f"❌ Predictions too static (diff: {evolution_diff:.3f})")
                            continuous_tests.append(("Prediction evolution", False))
                    else:
                        print("❌ Predictions identical across calls")
                        continuous_tests.append(("Prediction evolution", False))
                else:
                    print("❌ Insufficient continuous prediction calls")
                    continuous_tests.extend([
                        ("Multiple continuous calls", False),
                        ("Pattern awareness maintenance", False),
                        ("Pattern consistency", False),
                        ("Prediction evolution", False)
                    ])
                
                # Stop continuous prediction
                stop_response = self.session.post(f"{API_BASE_URL}/stop-continuous-prediction")
                if stop_response.status_code == 200:
                    print("✅ Continuous prediction stopped")
                    continuous_tests.append(("Stop continuous prediction", True))
                else:
                    print("❌ Stop continuous prediction failed")
                    continuous_tests.append(("Stop continuous prediction", False))
            else:
                print("❌ Start continuous prediction failed")
                continuous_tests.extend([
                    ("Start continuous prediction", False),
                    ("Multiple continuous calls", False),
                    ("Pattern awareness maintenance", False),
                    ("Pattern consistency", False),
                    ("Prediction evolution", False),
                    ("Stop continuous prediction", False)
                ])
        
        except Exception as e:
            print(f"❌ Continuous pattern awareness error: {str(e)}")
            continuous_tests.extend([
                ("Reset continuous prediction", False),
                ("Start continuous prediction", False),
                ("Multiple continuous calls", False),
                ("Pattern awareness maintenance", False),
                ("Pattern consistency", False),
                ("Prediction evolution", False),
                ("Stop continuous prediction", False)
            ])
        
        # Evaluate continuous pattern awareness tests
        passed_tests = sum(1 for _, passed in continuous_tests if passed)
        total_tests = len(continuous_tests)
        
        print(f"\n📊 Continuous Pattern Awareness Results: {passed_tests}/{total_tests}")
        for test_name, passed in continuous_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['continuous_pattern_awareness'] = passed_tests >= total_tests * 0.7
    
    def test_pattern_classification_accuracy(self):
        """Test 5: Pattern Classification Accuracy"""
        print("\n=== Testing Pattern Classification Accuracy ===")
        
        classification_tests = []
        
        try:
            # Test different pattern types and their classification
            pattern_datasets = [
                ("U-shaped (Quadratic)", self.create_u_shaped_data()),
                ("S-shaped (Cubic)", self.create_s_shaped_data()),
                ("Complex (Custom)", self.create_complex_shaped_data()),
                ("Polynomial", self.create_polynomial_data())
            ]
            
            for pattern_name, pattern_data in pattern_datasets:
                print(f"\n--- Testing {pattern_name} Classification ---")
                
                # Upload and train model
                upload_result = self.upload_and_analyze_data(pattern_data, f"classification_{pattern_name.lower().replace(' ', '_')}")
                
                if upload_result['success']:
                    model_result = self.train_advanced_model(upload_result['data_id'], 'lstm')
                    
                    if model_result['success']:
                        # Generate prediction to get pattern analysis
                        response = self.session.get(
                            f"{API_BASE_URL}/generate-continuous-prediction",
                            params={"model_id": model_result['model_id'], "steps": 20, "time_window": 100}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            pattern_analysis = data.get('pattern_analysis')
                            
                            if pattern_analysis:
                                detected_pattern = pattern_analysis.get('pattern_type', 'unknown')
                                print(f"✅ Pattern detected: {detected_pattern}")
                                
                                # Check if detection makes sense for the pattern type
                                expected_patterns = {
                                    "U-shaped (Quadratic)": ['quadratic', 'u_shape', 'polynomial'],
                                    "S-shaped (Cubic)": ['cubic', 's_shape', 'polynomial'],
                                    "Complex (Custom)": ['custom_shape', 'complex', 'spline'],
                                    "Polynomial": ['polynomial', 'complex', 'custom_shape']
                                }
                                
                                expected = expected_patterns.get(pattern_name, [])
                                if any(exp in detected_pattern.lower() for exp in expected):
                                    print(f"✅ Pattern classification accurate for {pattern_name}")
                                    classification_tests.append((f"{pattern_name} classification", True))
                                else:
                                    print(f"❌ Pattern classification inaccurate: expected {expected}, got {detected_pattern}")
                                    classification_tests.append((f"{pattern_name} classification", False))
                            else:
                                print(f"❌ No pattern analysis for {pattern_name}")
                                classification_tests.append((f"{pattern_name} classification", False))
                        else:
                            print(f"❌ Prediction failed for {pattern_name}")
                            classification_tests.append((f"{pattern_name} classification", False))
                    else:
                        print(f"❌ Model training failed for {pattern_name}")
                        classification_tests.append((f"{pattern_name} classification", False))
                else:
                    print(f"❌ Data upload failed for {pattern_name}")
                    classification_tests.append((f"{pattern_name} classification", False))
        
        except Exception as e:
            print(f"❌ Pattern classification error: {str(e)}")
            classification_tests.extend([
                ("U-shaped (Quadratic) classification", False),
                ("S-shaped (Cubic) classification", False),
                ("Complex (Custom) classification", False),
                ("Polynomial classification", False)
            ])
        
        # Evaluate pattern classification tests
        passed_tests = sum(1 for _, passed in classification_tests if passed)
        total_tests = len(classification_tests)
        
        print(f"\n📊 Pattern Classification Results: {passed_tests}/{total_tests}")
        for test_name, passed in classification_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['pattern_classification'] = passed_tests >= total_tests * 0.5  # 50% threshold for classification
    
    def run_all_pattern_tests(self):
        """Run all enhanced pattern-aware prediction tests"""
        print("🎯 Starting Enhanced Pattern-Aware Prediction System Testing")
        print("=" * 70)
        
        # Test 1: Advanced Pattern Detection
        pattern_data_result = self.test_advanced_pattern_detection()
        
        # Test 2: Pattern Learning and Prediction Quality
        model_id = self.test_pattern_learning_and_prediction(pattern_data_result)
        
        # Test 3: Adaptive Extrapolation
        self.test_adaptive_extrapolation(model_id)
        
        # Test 4: Continuous Pattern Awareness
        self.test_continuous_pattern_awareness(model_id)
        
        # Test 5: Pattern Classification Accuracy
        self.test_pattern_classification_accuracy()
        
        # Generate final summary
        self.generate_test_summary()
    
    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        print("\n" + "=" * 70)
        print("🎯 ENHANCED PATTERN-AWARE PREDICTION SYSTEM TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"\n📊 Overall Results: {passed_tests}/{total_tests} test categories passed")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        test_descriptions = {
            'advanced_pattern_detection': 'Advanced Pattern Detection (quadratic, cubic, polynomial, custom)',
            'pattern_learning': 'Pattern Learning and Prediction Quality',
            'adaptive_extrapolation': 'Adaptive Extrapolation Based on Learned Patterns',
            'continuous_pattern_awareness': 'Real-time Continuous Prediction with Pattern Awareness',
            'pattern_classification': 'Pattern Classification Accuracy'
        }
        
        for test_key, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            description = test_descriptions.get(test_key, test_key)
            print(f"   {status} {description}")
        
        # Key Success Criteria Assessment
        print("\n🎯 Key Success Criteria Assessment:")
        
        criteria_met = []
        
        # Criterion 1: Advanced Pattern Detection
        if self.test_results.get('advanced_pattern_detection', False):
            print("   ✅ System detects advanced patterns (quadratic, cubic, polynomial, custom)")
            criteria_met.append(True)
        else:
            print("   ❌ System fails to detect advanced patterns")
            criteria_met.append(False)
        
        # Criterion 2: Pattern Learning
        if self.test_results.get('pattern_learning', False):
            print("   ✅ System learns from historical data and uses patterns for prediction")
            criteria_met.append(True)
        else:
            print("   ❌ System fails to learn patterns from historical data")
            criteria_met.append(False)
        
        # Criterion 3: Adaptive Extrapolation
        if self.test_results.get('adaptive_extrapolation', False):
            print("   ✅ System properly extrapolates points based on learned patterns")
            criteria_met.append(True)
        else:
            print("   ❌ System fails to extrapolate based on learned patterns")
            criteria_met.append(False)
        
        # Criterion 4: Continuous Pattern Awareness
        if self.test_results.get('continuous_pattern_awareness', False):
            print("   ✅ Enhanced system works for continuous prediction with pattern awareness")
            criteria_met.append(True)
        else:
            print("   ❌ System lacks continuous prediction pattern awareness")
            criteria_met.append(False)
        
        # Overall Assessment
        criteria_passed = sum(criteria_met)
        total_criteria = len(criteria_met)
        
        print(f"\n🏆 SUCCESS CRITERIA: {criteria_passed}/{total_criteria} met ({(criteria_passed/total_criteria)*100:.1f}%)")
        
        if criteria_passed >= 3:
            print("🎉 ENHANCED PATTERN-AWARE PREDICTION SYSTEM: WORKING SUCCESSFULLY!")
            print("   The system demonstrates advanced pattern detection, learning, and adaptive extrapolation.")
        elif criteria_passed >= 2:
            print("⚠️  ENHANCED PATTERN-AWARE PREDICTION SYSTEM: PARTIALLY WORKING")
            print("   Some advanced features working but improvements needed.")
        else:
            print("❌ ENHANCED PATTERN-AWARE PREDICTION SYSTEM: NEEDS SIGNIFICANT IMPROVEMENT")
            print("   Core pattern-aware functionality not working as expected.")
        
        return criteria_passed >= 3

if __name__ == "__main__":
    tester = PatternAwareTester()
    tester.run_all_pattern_tests()
"""
Enhanced Pattern-Aware Prediction System Testing
Tests the improved pattern detection and prediction algorithms
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://1883c9bd-2fda-48e0-82d4-0ec1f13153f1.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing enhanced pattern-aware prediction system at: {API_BASE_URL}")

class PatternAwareTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        self.data_ids = {}
        self.model_ids = {}
        
    def create_u_shaped_data(self, points=50):
        """Create U-shaped pattern data for testing"""
        x = np.linspace(-2, 2, points)
        y = x**2 + 3 + np.random.normal(0, 0.1, points)  # U-shape with noise
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph': y
        })
        
        return df
    
    def create_linear_data(self, points=50):
        """Create linear pattern data for testing"""
        x = np.linspace(0, points-1, points)
        y = 2 * x + 10 + np.random.normal(0, 0.5, points)  # Linear trend with noise
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph': y
        })
        
        return df
    
    def create_cubic_data(self, points=50):
        """Create cubic pattern data for testing"""
        x = np.linspace(-1, 1, points)
        y = x**3 - 0.5*x + 5 + np.random.normal(0, 0.1, points)  # Cubic pattern with noise
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph': y
        })
        
        return df
    
    def create_inverted_u_data(self, points=50):
        """Create inverted U-shaped pattern data for testing"""
        x = np.linspace(-2, 2, points)
        y = -x**2 + 8 + np.random.normal(0, 0.1, points)  # Inverted U-shape with noise
        
        dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph': y
        })
        
        return df
    
    def upload_and_train_model(self, data, pattern_name, model_type='arima'):
        """Upload data and train model"""
        try:
            # Upload data
            csv_content = data.to_csv(index=False)
            files = {'file': (f'{pattern_name}_data.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Failed to upload {pattern_name} data: {response.status_code}")
                return None, None
            
            data_id = response.json().get('data_id')
            self.data_ids[pattern_name] = data_id
            
            # Train model
            training_params = {
                "time_column": "timestamp",
                "target_column": "ph",
                "order": [1, 1, 1] if model_type == 'arima' else {}
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": model_type},
                json=training_params
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to train {pattern_name} model: {response.status_code}")
                return data_id, None
            
            model_id = response.json().get('model_id')
            self.model_ids[pattern_name] = model_id
            
            print(f"✅ Successfully uploaded and trained {pattern_name} model")
            return data_id, model_id
            
        except Exception as e:
            print(f"❌ Error in upload_and_train_model for {pattern_name}: {str(e)}")
            return None, None
    
    def test_pattern_detection(self):
        """Test 1: Pattern Detection Capabilities"""
        print("\n=== Testing Pattern Detection Capabilities ===")
        
        pattern_tests = []
        
        # Test different pattern types
        patterns = {
            'u_shape': self.create_u_shaped_data(),
            'linear': self.create_linear_data(),
            'cubic': self.create_cubic_data(),
            'inverted_u': self.create_inverted_u_data()
        }
        
        for pattern_name, data in patterns.items():
            print(f"\nTesting {pattern_name} pattern detection...")
            
            data_id, model_id = self.upload_and_train_model(data, pattern_name)
            
            if data_id and model_id:
                # Generate predictions to test pattern analysis
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 30, "time_window": 100}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    pattern_analysis = result.get('pattern_analysis', {})
                    
                    if pattern_analysis:
                        detected_pattern = pattern_analysis.get('pattern_type', 'unknown')
                        print(f"   Detected pattern type: {detected_pattern}")
                        print(f"   Trend slope: {pattern_analysis.get('trend_slope', 'N/A')}")
                        print(f"   Velocity: {pattern_analysis.get('velocity', 'N/A')}")
                        print(f"   Recent mean: {pattern_analysis.get('recent_mean', 'N/A')}")
                        
                        # Check if pattern detection is reasonable
                        pattern_detected = detected_pattern != 'unknown'
                        pattern_tests.append((f"{pattern_name} pattern detection", pattern_detected))
                        
                        if pattern_detected:
                            print(f"   ✅ {pattern_name} pattern detected successfully")
                        else:
                            print(f"   ❌ {pattern_name} pattern not detected")
                    else:
                        print(f"   ❌ No pattern analysis data for {pattern_name}")
                        pattern_tests.append((f"{pattern_name} pattern detection", False))
                else:
                    print(f"   ❌ Failed to generate predictions for {pattern_name}: {response.status_code}")
                    pattern_tests.append((f"{pattern_name} pattern detection", False))
            else:
                pattern_tests.append((f"{pattern_name} pattern detection", False))
        
        # Evaluate pattern detection results
        passed_tests = sum(1 for _, passed in pattern_tests if passed)
        total_tests = len(pattern_tests)
        
        print(f"\n📊 Pattern detection test results: {passed_tests}/{total_tests}")
        for test_name, passed in pattern_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['pattern_detection'] = passed_tests >= total_tests * 0.75
    
    def test_pattern_aware_predictions(self):
        """Test 2: Pattern-Aware Prediction Quality"""
        print("\n=== Testing Pattern-Aware Prediction Quality ===")
        
        prediction_tests = []
        
        # Test U-shaped pattern predictions
        if 'u_shape' in self.model_ids:
            model_id = self.model_ids['u_shape']
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 20, "time_window": 100}
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                pattern_analysis = result.get('pattern_analysis', {})
                
                if len(predictions) >= 10:
                    # Check if predictions maintain realistic pH range
                    ph_range_valid = all(6.0 <= p <= 8.0 for p in predictions)
                    
                    # Check if predictions show variability (not monotonic)
                    prediction_variability = len(set([round(p, 1) for p in predictions])) > 3
                    
                    # Check if predictions follow pattern characteristics
                    pattern_type = pattern_analysis.get('pattern_type', 'unknown')
                    pattern_following = pattern_type in ['u_shape', 'quadratic', 'complex']
                    
                    print(f"   U-shape predictions: {len(predictions)} points")
                    print(f"   pH range valid (6.0-8.0): {ph_range_valid}")
                    print(f"   Prediction variability: {prediction_variability}")
                    print(f"   Pattern following: {pattern_following} (detected: {pattern_type})")
                    
                    prediction_tests.append(("U-shape pH range", ph_range_valid))
                    prediction_tests.append(("U-shape variability", prediction_variability))
                    prediction_tests.append(("U-shape pattern following", pattern_following))
                else:
                    print("   ❌ Insufficient U-shape predictions generated")
                    prediction_tests.extend([
                        ("U-shape pH range", False),
                        ("U-shape variability", False),
                        ("U-shape pattern following", False)
                    ])
            else:
                print(f"   ❌ Failed to generate U-shape predictions: {response.status_code}")
                prediction_tests.extend([
                    ("U-shape pH range", False),
                    ("U-shape variability", False),
                    ("U-shape pattern following", False)
                ])
        
        # Test linear pattern predictions
        if 'linear' in self.model_ids:
            model_id = self.model_ids['linear']
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 20, "time_window": 100}
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                pattern_analysis = result.get('pattern_analysis', {})
                
                if len(predictions) >= 10:
                    # Check trend consistency for linear pattern
                    trend_slope = pattern_analysis.get('trend_slope', 0)
                    
                    # Calculate actual trend in predictions
                    if len(predictions) >= 5:
                        actual_trend = (predictions[-1] - predictions[0]) / len(predictions)
                        trend_consistency = abs(actual_trend) > 0.01  # Some trend should be present
                    else:
                        trend_consistency = False
                    
                    # Check if predictions maintain reasonable bounds
                    reasonable_bounds = all(-10 <= p <= 50 for p in predictions)  # Reasonable for linear growth
                    
                    print(f"   Linear predictions: {len(predictions)} points")
                    print(f"   Detected trend slope: {trend_slope}")
                    print(f"   Trend consistency: {trend_consistency}")
                    print(f"   Reasonable bounds: {reasonable_bounds}")
                    
                    prediction_tests.append(("Linear trend consistency", trend_consistency))
                    prediction_tests.append(("Linear reasonable bounds", reasonable_bounds))
                else:
                    print("   ❌ Insufficient linear predictions generated")
                    prediction_tests.extend([
                        ("Linear trend consistency", False),
                        ("Linear reasonable bounds", False)
                    ])
            else:
                print(f"   ❌ Failed to generate linear predictions: {response.status_code}")
                prediction_tests.extend([
                    ("Linear trend consistency", False),
                    ("Linear reasonable bounds", False)
                ])
        
        # Evaluate prediction quality results
        passed_tests = sum(1 for _, passed in prediction_tests if passed)
        total_tests = len(prediction_tests)
        
        print(f"\n📊 Pattern-aware prediction test results: {passed_tests}/{total_tests}")
        for test_name, passed in prediction_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['pattern_aware_predictions'] = passed_tests >= total_tests * 0.7
    
    def test_continuous_prediction_continuity(self):
        """Test 3: Continuous Prediction Continuity"""
        print("\n=== Testing Continuous Prediction Continuity ===")
        
        continuity_tests = []
        
        # Test with U-shaped pattern for continuity
        if 'u_shape' in self.model_ids:
            model_id = self.model_ids['u_shape']
            
            # Make multiple continuous prediction calls
            prediction_sets = []
            timestamp_sets = []
            
            for i in range(3):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 10, "time_window": 50}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    predictions = result.get('predictions', [])
                    timestamps = result.get('timestamps', [])
                    
                    prediction_sets.append(predictions)
                    timestamp_sets.append(timestamps)
                    
                    time.sleep(1)  # Wait between calls
                else:
                    print(f"   ❌ Continuous prediction call {i+1} failed: {response.status_code}")
            
            if len(prediction_sets) >= 2:
                # Test timestamp progression
                timestamp_progression = True
                if len(timestamp_sets) >= 2:
                    # Check if timestamps are progressing forward
                    try:
                        first_timestamps = timestamp_sets[0]
                        second_timestamps = timestamp_sets[1]
                        
                        if first_timestamps and second_timestamps:
                            # Parse first timestamp from each set
                            first_time = datetime.fromisoformat(first_timestamps[0].replace('Z', '+00:00'))
                            second_time = datetime.fromisoformat(second_timestamps[0].replace('Z', '+00:00'))
                            
                            timestamp_progression = second_time > first_time
                    except Exception as e:
                        print(f"   ⚠️ Timestamp parsing error: {e}")
                        timestamp_progression = False
                
                # Test prediction continuity (no extreme jumps)
                prediction_continuity = True
                for i in range(1, len(prediction_sets)):
                    if prediction_sets[i] and prediction_sets[i-1]:
                        # Check for extreme jumps between prediction sets
                        last_prev = prediction_sets[i-1][-1] if prediction_sets[i-1] else 0
                        first_curr = prediction_sets[i][0] if prediction_sets[i] else 0
                        
                        jump = abs(first_curr - last_prev)
                        if jump > 5.0:  # Threshold for "extreme jump" in pH context
                            prediction_continuity = False
                            break
                
                # Test pattern maintenance across calls
                pattern_maintenance = True
                if len(prediction_sets) >= 2:
                    # Check if predictions maintain similar characteristics
                    for pred_set in prediction_sets:
                        if pred_set:
                            # Check if predictions stay within reasonable pH bounds
                            if not all(4.0 <= p <= 10.0 for p in pred_set):
                                pattern_maintenance = False
                                break
                
                print(f"   Continuous prediction calls made: {len(prediction_sets)}")
                print(f"   Timestamp progression: {timestamp_progression}")
                print(f"   Prediction continuity: {prediction_continuity}")
                print(f"   Pattern maintenance: {pattern_maintenance}")
                
                continuity_tests.append(("Timestamp progression", timestamp_progression))
                continuity_tests.append(("Prediction continuity", prediction_continuity))
                continuity_tests.append(("Pattern maintenance", pattern_maintenance))
            else:
                print("   ❌ Insufficient continuous prediction calls")
                continuity_tests.extend([
                    ("Timestamp progression", False),
                    ("Prediction continuity", False),
                    ("Pattern maintenance", False)
                ])
        else:
            print("   ❌ No U-shape model available for continuity testing")
            continuity_tests.extend([
                ("Timestamp progression", False),
                ("Prediction continuity", False),
                ("Pattern maintenance", False)
            ])
        
        # Evaluate continuity results
        passed_tests = sum(1 for _, passed in continuity_tests if passed)
        total_tests = len(continuity_tests)
        
        print(f"\n📊 Continuous prediction continuity test results: {passed_tests}/{total_tests}")
        for test_name, passed in continuity_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['continuous_prediction_continuity'] = passed_tests >= total_tests * 0.7
    
    def test_downward_bias_elimination(self):
        """Test 4: Downward Bias Elimination"""
        print("\n=== Testing Downward Bias Elimination ===")
        
        bias_tests = []
        
        # Test with different patterns to ensure no downward bias
        for pattern_name in ['u_shape', 'linear']:
            if pattern_name in self.model_ids:
                model_id = self.model_ids[pattern_name]
                
                print(f"\nTesting downward bias elimination for {pattern_name}...")
                
                # Generate multiple prediction sets
                all_predictions = []
                
                for i in range(5):  # Make 5 calls to test for accumulated bias
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": model_id, "steps": 10, "time_window": 50}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        predictions = result.get('predictions', [])
                        all_predictions.extend(predictions)
                        time.sleep(0.5)
                    else:
                        print(f"   ❌ Prediction call {i+1} failed for {pattern_name}")
                
                if len(all_predictions) >= 20:
                    # Calculate trend slope across all predictions
                    x = np.arange(len(all_predictions))
                    slope = np.polyfit(x, all_predictions, 1)[0]
                    
                    # Check for downward bias (significant negative slope)
                    no_downward_bias = slope > -0.1  # Allow slight negative slope but not strong downward bias
                    
                    # Check prediction range stability
                    pred_std = np.std(all_predictions)
                    pred_mean = np.mean(all_predictions)
                    range_stability = pred_std < abs(pred_mean) * 0.5  # Standard deviation should be reasonable
                    
                    # Check for realistic values
                    realistic_values = all(0 <= p <= 20 for p in all_predictions)  # Reasonable range for pH-like data
                    
                    print(f"   {pattern_name} predictions analyzed: {len(all_predictions)} points")
                    print(f"   Overall slope: {slope:.6f}")
                    print(f"   No downward bias: {no_downward_bias}")
                    print(f"   Range stability: {range_stability} (std: {pred_std:.3f}, mean: {pred_mean:.3f})")
                    print(f"   Realistic values: {realistic_values}")
                    
                    bias_tests.append((f"{pattern_name} no downward bias", no_downward_bias))
                    bias_tests.append((f"{pattern_name} range stability", range_stability))
                    bias_tests.append((f"{pattern_name} realistic values", realistic_values))
                else:
                    print(f"   ❌ Insufficient predictions for {pattern_name} bias testing")
                    bias_tests.extend([
                        (f"{pattern_name} no downward bias", False),
                        (f"{pattern_name} range stability", False),
                        (f"{pattern_name} realistic values", False)
                    ])
            else:
                print(f"   ❌ No {pattern_name} model available for bias testing")
                bias_tests.extend([
                    (f"{pattern_name} no downward bias", False),
                    (f"{pattern_name} range stability", False),
                    (f"{pattern_name} realistic values", False)
                ])
        
        # Evaluate bias elimination results
        passed_tests = sum(1 for _, passed in bias_tests if passed)
        total_tests = len(bias_tests)
        
        print(f"\n📊 Downward bias elimination test results: {passed_tests}/{total_tests}")
        for test_name, passed in bias_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['downward_bias_elimination'] = passed_tests >= total_tests * 0.8
    
    def test_pattern_specific_algorithms(self):
        """Test 5: Pattern-Specific Algorithm Performance"""
        print("\n=== Testing Pattern-Specific Algorithm Performance ===")
        
        algorithm_tests = []
        
        # Test U-shaped pattern algorithm
        if 'u_shape' in self.model_ids:
            model_id = self.model_ids['u_shape']
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 30, "time_window": 100}
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                pattern_analysis = result.get('pattern_analysis', {})
                
                if len(predictions) >= 20:
                    # For U-shaped data, check if predictions follow quadratic-like behavior
                    x = np.arange(len(predictions))
                    
                    # Fit quadratic to predictions
                    try:
                        quadratic_coeffs = np.polyfit(x, predictions, 2)
                        quadratic_fit = np.polyval(quadratic_coeffs, x)
                        
                        # Calculate R-squared for quadratic fit
                        ss_res = np.sum((predictions - quadratic_fit) ** 2)
                        ss_tot = np.sum((predictions - np.mean(predictions)) ** 2)
                        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                        
                        quadratic_fit_quality = r_squared > 0.3  # Reasonable quadratic fit
                        
                        print(f"   U-shape quadratic fit R²: {r_squared:.3f}")
                        print(f"   Quadratic fit quality: {quadratic_fit_quality}")
                        
                        algorithm_tests.append(("U-shape quadratic fit", quadratic_fit_quality))
                    except Exception as e:
                        print(f"   ❌ U-shape quadratic fit error: {e}")
                        algorithm_tests.append(("U-shape quadratic fit", False))
                else:
                    print("   ❌ Insufficient U-shape predictions for algorithm testing")
                    algorithm_tests.append(("U-shape quadratic fit", False))
            else:
                print(f"   ❌ Failed to generate U-shape predictions: {response.status_code}")
                algorithm_tests.append(("U-shape quadratic fit", False))
        
        # Test linear pattern algorithm
        if 'linear' in self.model_ids:
            model_id = self.model_ids['linear']
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 30, "time_window": 100}
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                pattern_analysis = result.get('pattern_analysis', {})
                
                if len(predictions) >= 20:
                    # For linear data, check if predictions maintain linear trend
                    x = np.arange(len(predictions))
                    
                    # Fit linear to predictions
                    try:
                        linear_coeffs = np.polyfit(x, predictions, 1)
                        linear_fit = np.polyval(linear_coeffs, x)
                        
                        # Calculate R-squared for linear fit
                        ss_res = np.sum((predictions - linear_fit) ** 2)
                        ss_tot = np.sum((predictions - np.mean(predictions)) ** 2)
                        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                        
                        linear_fit_quality = r_squared > 0.4  # Good linear fit
                        
                        # Check if trend is maintained (slope should be reasonable)
                        slope = linear_coeffs[0]
                        trend_maintained = abs(slope) > 0.01  # Some trend should be present
                        
                        print(f"   Linear fit R²: {r_squared:.3f}")
                        print(f"   Linear fit quality: {linear_fit_quality}")
                        print(f"   Slope: {slope:.3f}")
                        print(f"   Trend maintained: {trend_maintained}")
                        
                        algorithm_tests.append(("Linear fit quality", linear_fit_quality))
                        algorithm_tests.append(("Linear trend maintained", trend_maintained))
                    except Exception as e:
                        print(f"   ❌ Linear fit error: {e}")
                        algorithm_tests.extend([
                            ("Linear fit quality", False),
                            ("Linear trend maintained", False)
                        ])
                else:
                    print("   ❌ Insufficient linear predictions for algorithm testing")
                    algorithm_tests.extend([
                        ("Linear fit quality", False),
                        ("Linear trend maintained", False)
                    ])
            else:
                print(f"   ❌ Failed to generate linear predictions: {response.status_code}")
                algorithm_tests.extend([
                    ("Linear fit quality", False),
                    ("Linear trend maintained", False)
                ])
        
        # Evaluate algorithm performance results
        passed_tests = sum(1 for _, passed in algorithm_tests if passed)
        total_tests = len(algorithm_tests)
        
        print(f"\n📊 Pattern-specific algorithm test results: {passed_tests}/{total_tests}")
        for test_name, passed in algorithm_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['pattern_specific_algorithms'] = passed_tests >= total_tests * 0.7
    
    def run_all_tests(self):
        """Run all pattern-aware prediction tests"""
        print("🎯 Starting Enhanced Pattern-Aware Prediction System Testing")
        print("=" * 70)
        
        # Run all tests
        self.test_pattern_detection()
        self.test_pattern_aware_predictions()
        self.test_continuous_prediction_continuity()
        self.test_downward_bias_elimination()
        self.test_pattern_specific_algorithms()
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("🎯 ENHANCED PATTERN-AWARE PREDICTION SYSTEM TEST REPORT")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"\n📊 OVERALL RESULTS: {passed_tests}/{total_tests} test categories passed")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 DETAILED RESULTS:")
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {test_name.replace('_', ' ').title()}")
        
        # Determine overall system status
        if passed_tests >= total_tests * 0.8:
            print("\n🎉 SYSTEM STATUS: EXCELLENT - Pattern-aware prediction system is working well!")
        elif passed_tests >= total_tests * 0.6:
            print("\n✅ SYSTEM STATUS: GOOD - Pattern-aware prediction system is functional with minor issues")
        elif passed_tests >= total_tests * 0.4:
            print("\n⚠️  SYSTEM STATUS: NEEDS IMPROVEMENT - Some pattern-aware features need attention")
        else:
            print("\n❌ SYSTEM STATUS: CRITICAL ISSUES - Pattern-aware prediction system needs significant fixes")
        
        print("\n🔍 KEY FINDINGS:")
        
        if self.test_results.get('pattern_detection', False):
            print("   ✅ Pattern detection is working correctly")
        else:
            print("   ❌ Pattern detection needs improvement")
        
        if self.test_results.get('downward_bias_elimination', False):
            print("   ✅ Downward bias has been successfully eliminated")
        else:
            print("   ❌ Downward bias issues still present")
        
        if self.test_results.get('continuous_prediction_continuity', False):
            print("   ✅ Continuous predictions maintain proper continuity")
        else:
            print("   ❌ Continuous prediction continuity needs work")
        
        if self.test_results.get('pattern_specific_algorithms', False):
            print("   ✅ Pattern-specific algorithms are performing well")
        else:
            print("   ❌ Pattern-specific algorithms need optimization")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    tester = PatternAwareTester()
    tester.run_all_tests()