#!/usr/bin/env python3
"""
Enhanced Pattern-Following Algorithm Testing
Tests the comprehensive bias correction, pattern-based prediction, trend stabilization, 
and error correction techniques to verify proper pattern following.
"""

import requests
import json
import pandas as pd
import numpy as np
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

print(f"Testing enhanced pattern-following algorithms at: {API_BASE_URL}")

class EnhancedPatternTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_ph_pattern_data(self, pattern_type="stable_with_trend"):
        """Create pH data with specific patterns for testing"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='H')
        
        if pattern_type == "stable_with_trend":
            # Stable pH with slight upward trend
            base_ph = 7.2
            trend = np.linspace(0, 0.3, 50)  # Slight upward trend
            noise = np.random.normal(0, 0.05, 50)  # Small noise
            ph_values = base_ph + trend + noise
            
        elif pattern_type == "cyclical":
            # Cyclical pH pattern (daily cycle)
            base_ph = 7.0
            cycle = 0.3 * np.sin(2 * np.pi * np.arange(50) / 24)  # Daily cycle
            trend = 0.1 * np.arange(50) / 50  # Small trend
            noise = np.random.normal(0, 0.03, 50)
            ph_values = base_ph + cycle + trend + noise
            
        elif pattern_type == "u_shaped":
            # U-shaped pattern (pH drops then recovers)
            x = np.linspace(-2, 2, 50)
            base_ph = 7.5
            u_shape = 0.2 * x**2  # U-shaped curve
            noise = np.random.normal(0, 0.04, 50)
            ph_values = base_ph - u_shape + noise
            
        elif pattern_type == "step_change":
            # Step change pattern
            ph_values = np.ones(50) * 7.0
            ph_values[25:] = 7.4  # Step up at midpoint
            noise = np.random.normal(0, 0.03, 50)
            ph_values += noise
            
        elif pattern_type == "complex_pattern":
            # Complex pattern with multiple components
            base_ph = 7.3
            trend = 0.15 * np.arange(50) / 50
            cycle1 = 0.2 * np.sin(2 * np.pi * np.arange(50) / 12)  # Short cycle
            cycle2 = 0.1 * np.sin(2 * np.pi * np.arange(50) / 30)  # Long cycle
            noise = np.random.normal(0, 0.04, 50)
            ph_values = base_ph + trend + cycle1 + cycle2 + noise
            
        # Ensure pH values are in realistic range
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'pH': ph_values
        })
        
        return df
    
    def test_multi_scale_pattern_analysis(self):
        """Test 1: Multi-scale pattern analysis functions"""
        print("\n=== Testing Multi-Scale Pattern Analysis ===")
        
        try:
            # Create complex pattern data
            df = self.create_ph_pattern_data("complex_pattern")
            csv_content = df.to_csv(index=False)
            
            # Upload data
            files = {'file': ('complex_pattern.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Data upload failed: {response.status_code}")
                self.test_results['multi_scale_analysis'] = False
                return
                
            data_id = response.json().get('data_id')
            print(f"✅ Complex pattern data uploaded (ID: {data_id})")
            
            # Train ARIMA model for pattern analysis
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
            )
            
            if response.status_code != 200:
                print(f"❌ Model training failed: {response.status_code}")
                self.test_results['multi_scale_analysis'] = False
                return
                
            model_id = response.json().get('model_id')
            print(f"✅ ARIMA model trained (ID: {model_id})")
            
            # Test continuous prediction with pattern analysis
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 20, "time_window": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                pattern_analysis = data.get('pattern_analysis', {})
                
                print("✅ Multi-scale pattern analysis successful")
                print(f"   Predictions generated: {len(predictions)}")
                
                # Check for pattern analysis components
                analysis_tests = []
                
                # Test for trend analysis
                if 'trend_slope' in pattern_analysis:
                    trend_slope = pattern_analysis['trend_slope']
                    print(f"   ✅ Trend slope detected: {trend_slope:.6f}")
                    analysis_tests.append(True)
                else:
                    print("   ❌ Trend slope missing")
                    analysis_tests.append(False)
                
                # Test for velocity analysis
                if 'velocity' in pattern_analysis:
                    velocity = pattern_analysis['velocity']
                    print(f"   ✅ Velocity analysis: {velocity:.6f}")
                    analysis_tests.append(True)
                else:
                    print("   ❌ Velocity analysis missing")
                    analysis_tests.append(False)
                
                # Test for pattern type detection
                if 'pattern_type' in pattern_analysis:
                    pattern_type = pattern_analysis['pattern_type']
                    print(f"   ✅ Pattern type detected: {pattern_type}")
                    analysis_tests.append(True)
                else:
                    print("   ❌ Pattern type detection missing")
                    analysis_tests.append(False)
                
                # Test for multi-scale components
                multi_scale_components = ['recent_trend', 'short_trend', 'overall_trend']
                multi_scale_detected = sum(1 for comp in multi_scale_components if comp in pattern_analysis)
                
                if multi_scale_detected >= 2:
                    print(f"   ✅ Multi-scale analysis: {multi_scale_detected}/3 components detected")
                    analysis_tests.append(True)
                else:
                    print(f"   ❌ Multi-scale analysis incomplete: {multi_scale_detected}/3 components")
                    analysis_tests.append(False)
                
                # Overall test result
                self.test_results['multi_scale_analysis'] = sum(analysis_tests) >= 3
                
            else:
                print(f"❌ Pattern analysis failed: {response.status_code}")
                self.test_results['multi_scale_analysis'] = False
                
        except Exception as e:
            print(f"❌ Multi-scale pattern analysis error: {str(e)}")
            self.test_results['multi_scale_analysis'] = False
    
    def test_bias_correction_historical_ranges(self):
        """Test 2: Enhanced bias correction maintains historical value ranges"""
        print("\n=== Testing Bias Correction & Historical Range Maintenance ===")
        
        try:
            # Create stable pH data with known range
            df = self.create_ph_pattern_data("stable_with_trend")
            historical_min = df['pH'].min()
            historical_max = df['pH'].max()
            historical_mean = df['pH'].mean()
            historical_std = df['pH'].std()
            
            print(f"   Historical pH range: {historical_min:.3f} - {historical_max:.3f}")
            print(f"   Historical mean: {historical_mean:.3f} ± {historical_std:.3f}")
            
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('stable_trend.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [1, 1, 1]}
            )
            model_id = response.json().get('model_id')
            
            # Test multiple prediction calls to check for bias accumulation
            bias_tests = []
            all_predictions = []
            
            for i in range(5):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 10, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    all_predictions.extend(predictions)
                    time.sleep(0.5)  # Small delay between calls
            
            if all_predictions:
                pred_min = min(all_predictions)
                pred_max = max(all_predictions)
                pred_mean = np.mean(all_predictions)
                pred_std = np.std(all_predictions)
                
                print(f"   Predicted pH range: {pred_min:.3f} - {pred_max:.3f}")
                print(f"   Predicted mean: {pred_mean:.3f} ± {pred_std:.3f}")
                
                # Test 1: Range maintenance (predictions within reasonable bounds)
                reasonable_lower = historical_min - 2 * historical_std
                reasonable_upper = historical_max + 2 * historical_std
                
                range_maintained = (pred_min >= reasonable_lower and pred_max <= reasonable_upper)
                print(f"   ✅ Range maintenance: {range_maintained}")
                bias_tests.append(range_maintained)
                
                # Test 2: Mean preservation (no significant bias)
                mean_deviation = abs(pred_mean - historical_mean)
                mean_preserved = mean_deviation <= historical_std
                print(f"   ✅ Mean preservation: {mean_preserved} (deviation: {mean_deviation:.3f})")
                bias_tests.append(mean_preserved)
                
                # Test 3: No downward bias (predictions don't consistently trend down)
                prediction_trend = np.polyfit(range(len(all_predictions)), all_predictions, 1)[0]
                no_downward_bias = prediction_trend > -0.01  # Allow small negative trend
                print(f"   ✅ No downward bias: {no_downward_bias} (trend: {prediction_trend:.6f})")
                bias_tests.append(no_downward_bias)
                
                # Test 4: Variability preservation
                variability_ratio = pred_std / historical_std
                variability_preserved = 0.5 <= variability_ratio <= 2.0  # Within reasonable range
                print(f"   ✅ Variability preservation: {variability_preserved} (ratio: {variability_ratio:.3f})")
                bias_tests.append(variability_preserved)
                
                self.test_results['bias_correction'] = sum(bias_tests) >= 3
                
            else:
                print("❌ No predictions generated")
                self.test_results['bias_correction'] = False
                
        except Exception as e:
            print(f"❌ Bias correction test error: {str(e)}")
            self.test_results['bias_correction'] = False
    
    def test_cyclical_pattern_detection(self):
        """Test 3: Improved cyclical pattern detection"""
        print("\n=== Testing Cyclical Pattern Detection ===")
        
        try:
            # Create cyclical pH data
            df = self.create_ph_pattern_data("cyclical")
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('cyclical_pattern.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
            )
            model_id = response.json().get('model_id')
            
            # Generate predictions and analyze for cyclical patterns
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 30, "time_window": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                pattern_analysis = data.get('pattern_analysis', {})
                
                cyclical_tests = []
                
                # Test 1: Cyclical pattern detection in analysis
                if 'pattern_type' in pattern_analysis:
                    pattern_type = pattern_analysis['pattern_type']
                    cyclical_detected = 'cycl' in pattern_type.lower() or 'complex' in pattern_type.lower()
                    print(f"   ✅ Pattern type detection: {pattern_type} (cyclical: {cyclical_detected})")
                    cyclical_tests.append(cyclical_detected)
                else:
                    print("   ❌ Pattern type not detected")
                    cyclical_tests.append(False)
                
                # Test 2: Predictions show cyclical behavior
                if len(predictions) >= 24:  # Need enough points to detect cycles
                    # Simple cycle detection: check for oscillations
                    diffs = np.diff(predictions)
                    sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i-1] < 0)
                    
                    # Expect some oscillations in cyclical data
                    cyclical_behavior = sign_changes >= 3
                    print(f"   ✅ Cyclical behavior in predictions: {cyclical_behavior} ({sign_changes} sign changes)")
                    cyclical_tests.append(cyclical_behavior)
                else:
                    print("   ❌ Insufficient predictions for cycle analysis")
                    cyclical_tests.append(False)
                
                # Test 3: Pattern continuation parameters
                if 'pattern_continuation' in pattern_analysis:
                    continuation = pattern_analysis['pattern_continuation']
                    has_continuation = isinstance(continuation, dict) and len(continuation) > 0
                    print(f"   ✅ Pattern continuation parameters: {has_continuation}")
                    cyclical_tests.append(has_continuation)
                else:
                    print("   ❌ Pattern continuation parameters missing")
                    cyclical_tests.append(False)
                
                self.test_results['cyclical_detection'] = sum(cyclical_tests) >= 2
                
            else:
                print(f"❌ Cyclical pattern prediction failed: {response.status_code}")
                self.test_results['cyclical_detection'] = False
                
        except Exception as e:
            print(f"❌ Cyclical pattern detection error: {str(e)}")
            self.test_results['cyclical_detection'] = False
    
    def test_adaptive_trend_decay(self):
        """Test 4: Adaptive trend decay follows historical trends better"""
        print("\n=== Testing Adaptive Trend Decay ===")
        
        try:
            # Create data with clear trend
            df = self.create_ph_pattern_data("stable_with_trend")
            historical_trend = np.polyfit(range(len(df)), df['pH'].values, 1)[0]
            print(f"   Historical trend slope: {historical_trend:.6f}")
            
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('trend_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [1, 1, 1]}
            )
            model_id = response.json().get('model_id')
            
            # Test trend decay over multiple prediction horizons
            trend_tests = []
            
            for steps in [10, 20, 30]:
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": steps, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    pattern_analysis = data.get('pattern_analysis', {})
                    
                    if predictions and len(predictions) >= 5:
                        # Calculate prediction trend
                        pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
                        
                        # Test trend consistency
                        trend_consistency = pattern_analysis.get('trend_consistency', 0)
                        
                        print(f"   Steps {steps}: pred_trend={pred_trend:.6f}, consistency={trend_consistency:.3f}")
                        
                        # Adaptive decay should maintain trend direction but with appropriate decay
                        trend_direction_maintained = (historical_trend * pred_trend >= 0)  # Same sign
                        reasonable_decay = abs(pred_trend) <= abs(historical_trend) * 2  # Not amplified too much
                        
                        trend_test = trend_direction_maintained and reasonable_decay
                        trend_tests.append(trend_test)
                        
                        print(f"   ✅ Trend decay test (steps {steps}): {trend_test}")
                    else:
                        trend_tests.append(False)
                else:
                    trend_tests.append(False)
            
            # Test adaptive decay parameters
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 20, "time_window": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                pattern_analysis = data.get('pattern_analysis', {})
                
                # Check for adaptive parameters
                adaptive_params = ['trend_consistency', 'stability_factor', 'bias_correction_factor']
                adaptive_detected = sum(1 for param in adaptive_params if param in pattern_analysis)
                
                print(f"   ✅ Adaptive parameters detected: {adaptive_detected}/3")
                trend_tests.append(adaptive_detected >= 2)
            
            self.test_results['adaptive_trend_decay'] = sum(trend_tests) >= len(trend_tests) * 0.7
            
        except Exception as e:
            print(f"❌ Adaptive trend decay test error: {str(e)}")
            self.test_results['adaptive_trend_decay'] = False
    
    def test_volatility_aware_adjustments(self):
        """Test 5: Volatility-aware adjustments maintain realistic variation"""
        print("\n=== Testing Volatility-Aware Adjustments ===")
        
        try:
            # Create data with specific volatility characteristics
            df = self.create_ph_pattern_data("complex_pattern")
            historical_volatility = df['pH'].std()
            historical_changes = np.diff(df['pH'].values)
            historical_change_std = np.std(historical_changes)
            
            print(f"   Historical volatility (std): {historical_volatility:.4f}")
            print(f"   Historical change volatility: {historical_change_std:.4f}")
            
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('volatility_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
            )
            model_id = response.json().get('model_id')
            
            # Generate multiple prediction sets to analyze volatility
            all_predictions = []
            volatility_tests = []
            
            for i in range(3):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 25, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    all_predictions.extend(predictions)
                    time.sleep(0.3)
            
            if all_predictions:
                pred_volatility = np.std(all_predictions)
                pred_changes = np.diff(all_predictions)
                pred_change_std = np.std(pred_changes)
                
                print(f"   Predicted volatility (std): {pred_volatility:.4f}")
                print(f"   Predicted change volatility: {pred_change_std:.4f}")
                
                # Test 1: Volatility preservation
                volatility_ratio = pred_volatility / historical_volatility
                volatility_preserved = 0.3 <= volatility_ratio <= 3.0  # Reasonable range
                print(f"   ✅ Volatility preservation: {volatility_preserved} (ratio: {volatility_ratio:.3f})")
                volatility_tests.append(volatility_preserved)
                
                # Test 2: Change volatility preservation
                change_ratio = pred_change_std / historical_change_std
                change_volatility_preserved = 0.2 <= change_ratio <= 4.0  # Reasonable range
                print(f"   ✅ Change volatility preservation: {change_volatility_preserved} (ratio: {change_ratio:.3f})")
                volatility_tests.append(change_volatility_preserved)
                
                # Test 3: Realistic variation (not too smooth, not too erratic)
                # Check for reasonable number of direction changes
                sign_changes = sum(1 for i in range(1, len(pred_changes)) if pred_changes[i] * pred_changes[i-1] < 0)
                expected_changes = len(pred_changes) * 0.2  # Expect some variation
                realistic_variation = sign_changes >= expected_changes
                print(f"   ✅ Realistic variation: {realistic_variation} ({sign_changes} changes, expected >= {expected_changes:.1f})")
                volatility_tests.append(realistic_variation)
                
                # Test 4: Check for volatility-aware parameters in analysis
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 15, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    pattern_analysis = data.get('pattern_analysis', {})
                    
                    volatility_params = ['volatility', 'stability_factor', 'recent_std']
                    volatility_param_count = sum(1 for param in volatility_params if param in pattern_analysis)
                    
                    volatility_aware = volatility_param_count >= 2
                    print(f"   ✅ Volatility-aware parameters: {volatility_aware} ({volatility_param_count}/3)")
                    volatility_tests.append(volatility_aware)
                
                self.test_results['volatility_adjustments'] = sum(volatility_tests) >= 3
                
            else:
                print("❌ No predictions generated for volatility analysis")
                self.test_results['volatility_adjustments'] = False
                
        except Exception as e:
            print(f"❌ Volatility-aware adjustments test error: {str(e)}")
            self.test_results['volatility_adjustments'] = False
    
    def test_enhanced_bounds_checking(self):
        """Test 6: Enhanced bounds checking keeps predictions within reasonable ranges"""
        print("\n=== Testing Enhanced Bounds Checking ===")
        
        try:
            # Create data with known bounds
            df = self.create_ph_pattern_data("u_shaped")
            data_min = df['pH'].min()
            data_max = df['pH'].max()
            data_mean = df['pH'].mean()
            data_std = df['pH'].std()
            
            # Define reasonable bounds (should be wider than historical but not unlimited)
            reasonable_lower = data_min - 2 * data_std
            reasonable_upper = data_max + 2 * data_std
            
            print(f"   Historical range: {data_min:.3f} - {data_max:.3f}")
            print(f"   Reasonable bounds: {reasonable_lower:.3f} - {reasonable_upper:.3f}")
            
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('bounds_test.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
            )
            model_id = response.json().get('model_id')
            
            # Test bounds checking with various prediction horizons
            bounds_tests = []
            all_predictions = []
            
            for steps in [10, 20, 30, 50]:  # Test different horizons
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": steps, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    all_predictions.extend(predictions)
                    
                    if predictions:
                        pred_min = min(predictions)
                        pred_max = max(predictions)
                        
                        # Test bounds for this horizon
                        within_reasonable_bounds = (pred_min >= reasonable_lower and pred_max <= reasonable_upper)
                        
                        print(f"   Steps {steps}: range {pred_min:.3f} - {pred_max:.3f}, within bounds: {within_reasonable_bounds}")
                        bounds_tests.append(within_reasonable_bounds)
                    else:
                        bounds_tests.append(False)
                else:
                    bounds_tests.append(False)
            
            # Test extreme case: very long prediction horizon
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 100, "time_window": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if predictions:
                    pred_min = min(predictions)
                    pred_max = max(predictions)
                    
                    # Even for very long horizons, should stay within expanded reasonable bounds
                    expanded_lower = data_min - 4 * data_std
                    expanded_upper = data_max + 4 * data_std
                    
                    extreme_bounds_test = (pred_min >= expanded_lower and pred_max <= expanded_upper)
                    print(f"   Extreme horizon (100 steps): range {pred_min:.3f} - {pred_max:.3f}, within expanded bounds: {extreme_bounds_test}")
                    bounds_tests.append(extreme_bounds_test)
                else:
                    bounds_tests.append(False)
            else:
                bounds_tests.append(False)
            
            # Test for bounds checking parameters in analysis
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 20, "time_window": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                pattern_analysis = data.get('pattern_analysis', {})
                
                # Check for bounds-related parameters
                bounds_params = ['mean', 'std', 'recent_mean', 'recent_std']
                bounds_param_count = sum(1 for param in bounds_params if param in pattern_analysis)
                
                bounds_aware = bounds_param_count >= 3
                print(f"   ✅ Bounds-aware parameters: {bounds_aware} ({bounds_param_count}/4)")
                bounds_tests.append(bounds_aware)
            
            # Overall bounds checking test
            self.test_results['bounds_checking'] = sum(bounds_tests) >= len(bounds_tests) * 0.8
            
        except Exception as e:
            print(f"❌ Enhanced bounds checking test error: {str(e)}")
            self.test_results['bounds_checking'] = False
    
    def test_pattern_preservation_score(self):
        """Test 7: Pattern preservation score improvements"""
        print("\n=== Testing Pattern Preservation Score ===")
        
        try:
            # Create data with clear patterns
            df = self.create_ph_pattern_data("cyclical")
            csv_content = df.to_csv(index=False)
            
            # Upload and train model
            files = {'file': ('pattern_preservation.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            data_id = response.json().get('data_id')
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
            )
            model_id = response.json().get('model_id')
            
            # Generate predictions and analyze pattern preservation
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 30, "time_window": 50}
            )
            
            preservation_tests = []
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                pattern_analysis = data.get('pattern_analysis', {})
                
                # Test 1: Pattern quality score
                if 'pattern_quality_score' in pattern_analysis:
                    quality_score = pattern_analysis['pattern_quality_score']
                    high_quality = quality_score >= 0.7  # Expect good quality
                    print(f"   ✅ Pattern quality score: {quality_score:.3f} (high quality: {high_quality})")
                    preservation_tests.append(high_quality)
                else:
                    print("   ❌ Pattern quality score missing")
                    preservation_tests.append(False)
                
                # Test 2: Pattern preservation metrics
                preservation_metrics = ['trend_consistency', 'pattern_continuation', 'stability_factor']
                metrics_present = sum(1 for metric in preservation_metrics if metric in pattern_analysis)
                
                metrics_adequate = metrics_present >= 2
                print(f"   ✅ Preservation metrics: {metrics_adequate} ({metrics_present}/3 present)")
                preservation_tests.append(metrics_adequate)
                
                # Test 3: Predictions maintain pattern characteristics
                if predictions and len(predictions) >= 20:
                    # Check if predictions show similar characteristics to historical data
                    historical_mean = df['pH'].mean()
                    historical_std = df['pH'].std()
                    
                    pred_mean = np.mean(predictions)
                    pred_std = np.std(predictions)
                    
                    mean_similarity = abs(pred_mean - historical_mean) <= historical_std
                    std_similarity = 0.5 <= (pred_std / historical_std) <= 2.0
                    
                    characteristics_preserved = mean_similarity and std_similarity
                    print(f"   ✅ Pattern characteristics preserved: {characteristics_preserved}")
                    print(f"      Mean similarity: {mean_similarity} ({pred_mean:.3f} vs {historical_mean:.3f})")
                    print(f"      Std similarity: {std_similarity} (ratio: {pred_std/historical_std:.3f})")
                    preservation_tests.append(characteristics_preserved)
                else:
                    preservation_tests.append(False)
                
                # Test 4: Pattern continuation quality
                if 'pattern_continuation' in pattern_analysis:
                    continuation = pattern_analysis['pattern_continuation']
                    if isinstance(continuation, dict):
                        continuation_quality = len(continuation) >= 2  # Has meaningful continuation data
                        print(f"   ✅ Pattern continuation quality: {continuation_quality}")
                        preservation_tests.append(continuation_quality)
                    else:
                        preservation_tests.append(False)
                else:
                    preservation_tests.append(False)
                
                self.test_results['pattern_preservation'] = sum(preservation_tests) >= 3
                
            else:
                print(f"❌ Pattern preservation test failed: {response.status_code}")
                self.test_results['pattern_preservation'] = False
                
        except Exception as e:
            print(f"❌ Pattern preservation score test error: {str(e)}")
            self.test_results['pattern_preservation'] = False
    
    def test_comprehensive_pattern_following(self):
        """Test 8: Comprehensive pattern following with complex scenarios"""
        print("\n=== Testing Comprehensive Pattern Following ===")
        
        try:
            comprehensive_tests = []
            
            # Test different pattern types
            pattern_types = ["stable_with_trend", "cyclical", "u_shaped", "step_change", "complex_pattern"]
            
            for pattern_type in pattern_types:
                print(f"\n   Testing pattern type: {pattern_type}")
                
                # Create and upload data
                df = self.create_ph_pattern_data(pattern_type)
                csv_content = df.to_csv(index=False)
                
                files = {'file': (f'{pattern_type}.csv', csv_content, 'text/csv')}
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code != 200:
                    comprehensive_tests.append(False)
                    continue
                    
                data_id = response.json().get('data_id')
                
                # Train model
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": "arima"},
                    json={"time_column": "timestamp", "target_column": "pH", "order": [2, 1, 2]}
                )
                
                if response.status_code != 200:
                    comprehensive_tests.append(False)
                    continue
                    
                model_id = response.json().get('model_id')
                
                # Test pattern following
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 25, "time_window": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    pattern_analysis = data.get('pattern_analysis', {})
                    
                    # Evaluate pattern following quality
                    pattern_detected = 'pattern_type' in pattern_analysis
                    reasonable_predictions = (
                        predictions and 
                        len(predictions) == 25 and 
                        all(6.0 <= p <= 8.0 for p in predictions)  # pH range check
                    )
                    
                    pattern_following_quality = pattern_detected and reasonable_predictions
                    print(f"      Pattern following quality: {pattern_following_quality}")
                    comprehensive_tests.append(pattern_following_quality)
                else:
                    comprehensive_tests.append(False)
            
            # Overall comprehensive test result
            passed_patterns = sum(comprehensive_tests)
            total_patterns = len(pattern_types)
            
            print(f"\n   ✅ Comprehensive pattern following: {passed_patterns}/{total_patterns} patterns handled correctly")
            self.test_results['comprehensive_pattern_following'] = passed_patterns >= total_patterns * 0.8
            
        except Exception as e:
            print(f"❌ Comprehensive pattern following test error: {str(e)}")
            self.test_results['comprehensive_pattern_following'] = False
    
    def run_all_tests(self):
        """Run all enhanced pattern-following algorithm tests"""
        print("🎯 ENHANCED PATTERN-FOLLOWING ALGORITHM TESTING")
        print("=" * 60)
        
        # Run all tests
        self.test_multi_scale_pattern_analysis()
        self.test_bias_correction_historical_ranges()
        self.test_cyclical_pattern_detection()
        self.test_adaptive_trend_decay()
        self.test_volatility_aware_adjustments()
        self.test_enhanced_bounds_checking()
        self.test_pattern_preservation_score()
        self.test_comprehensive_pattern_following()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 ENHANCED PATTERN-FOLLOWING TEST RESULTS")
        print("=" * 60)
        
        test_names = [
            ("Multi-Scale Pattern Analysis", "multi_scale_analysis"),
            ("Bias Correction & Historical Ranges", "bias_correction"),
            ("Cyclical Pattern Detection", "cyclical_detection"),
            ("Adaptive Trend Decay", "adaptive_trend_decay"),
            ("Volatility-Aware Adjustments", "volatility_adjustments"),
            ("Enhanced Bounds Checking", "bounds_checking"),
            ("Pattern Preservation Score", "pattern_preservation"),
            ("Comprehensive Pattern Following", "comprehensive_pattern_following")
        ]
        
        passed_tests = 0
        total_tests = len(test_names)
        
        for test_name, test_key in test_names:
            result = self.test_results.get(test_key, False)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            if result:
                passed_tests += 1
        
        print("=" * 60)
        success_rate = (passed_tests / total_tests) * 100
        print(f"🎯 OVERALL SUCCESS RATE: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 EXCELLENT: Enhanced pattern-following algorithms are working well!")
        elif success_rate >= 60:
            print("✅ GOOD: Most enhanced pattern-following features are functional")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Several pattern-following features need attention")
        
        return self.test_results

if __name__ == "__main__":
    tester = EnhancedPatternTester()
    results = tester.run_all_tests()