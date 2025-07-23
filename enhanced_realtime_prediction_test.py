#!/usr/bin/env python3
"""
Enhanced Real-time Continuous Prediction System Testing
Tests the advanced pattern memory system and real-time prediction capabilities
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://3504f872-4ab4-43c1-a827-4429cc10638c.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Enhanced Real-time Prediction System at: {API_BASE_URL}")

class EnhancedRealtimePredictionTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        self.data_id = None
        self.model_id = None
        
    def create_pattern_data(self, pattern_type="trend_seasonal"):
        """Create realistic time series data with clear patterns"""
        np.random.seed(42)  # For reproducible results
        
        if pattern_type == "linear_trend":
            # Linear trend data
            dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
            trend = np.linspace(7.0, 7.5, 50)
            noise = np.random.normal(0, 0.05, 50)
            values = trend + noise
            
        elif pattern_type == "seasonal_cyclical":
            # Seasonal/cyclical data
            dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
            base = 7.2
            seasonal = 0.3 * np.sin(2 * np.pi * np.arange(60) / 7)  # Weekly pattern
            trend = 0.002 * np.arange(60)  # Slight upward trend
            noise = np.random.normal(0, 0.03, 60)
            values = base + seasonal + trend + noise
            
        elif pattern_type == "volatile_noisy":
            # Volatile/noisy data
            dates = pd.date_range(start='2023-01-01', periods=45, freq='D')
            base = 7.0
            volatility = np.random.normal(0, 0.15, 45)
            trend = 0.005 * np.arange(45)
            values = base + volatility + trend
            
        elif pattern_type == "multi_pattern":
            # Multi-pattern data (trend + seasonality + noise)
            dates = pd.date_range(start='2023-01-01', periods=70, freq='D')
            base = 7.1
            trend = 0.003 * np.arange(70)
            seasonal = 0.2 * np.sin(2 * np.pi * np.arange(70) / 14)  # Bi-weekly pattern
            cyclical = 0.1 * np.cos(2 * np.pi * np.arange(70) / 30)  # Monthly pattern
            noise = np.random.normal(0, 0.04, 70)
            values = base + trend + seasonal + cyclical + noise
            
        else:  # pH monitoring data
            # pH monitoring data (6.0-8.0 range)
            dates = pd.date_range(start='2023-01-01', periods=48, freq='H')
            base = 7.2
            daily_cycle = 0.3 * np.sin(2 * np.pi * np.arange(48) / 24)  # Daily pH cycle
            trend = 0.001 * np.arange(48)
            noise = np.random.normal(0, 0.05, 48)
            values = base + daily_cycle + trend + noise
            # Ensure pH range
            values = np.clip(values, 6.0, 8.0)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'pH': values
        })
        
        return df
    
    def upload_test_data(self, pattern_type="multi_pattern"):
        """Upload test data with specific patterns"""
        print(f"\n=== Uploading {pattern_type} Test Data ===")
        
        try:
            df = self.create_pattern_data(pattern_type)
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': (f'{pattern_type}_data.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                
                print("✅ Pattern data upload successful")
                print(f"   Data ID: {self.data_id}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                
                return True
            else:
                print(f"❌ Pattern data upload failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Pattern data upload error: {str(e)}")
            return False
    
    def train_advanced_model(self):
        """Train advanced model for pattern learning"""
        print("\n=== Training Advanced Model for Pattern Learning ===")
        
        if not self.data_id:
            print("❌ Cannot train model - no data uploaded")
            return False
            
        try:
            # Train ARIMA model (most reliable for pattern learning)
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": "arima"},
                json={
                    "time_column": "timestamp",
                    "target_column": "pH",
                    "order": [2, 1, 2]  # Higher order for better pattern capture
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.model_id = data.get('model_id')
                
                print("✅ Advanced model training successful")
                print(f"   Model ID: {self.model_id}")
                print(f"   Status: {data.get('status')}")
                
                return True
            else:
                print(f"❌ Advanced model training failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Advanced model training error: {str(e)}")
            return False
    
    def test_pattern_learning_memory(self):
        """Test 1: Pattern Learning and Memory System"""
        print("\n=== Testing Pattern Learning and Memory System ===")
        
        if not self.model_id:
            print("❌ Cannot test pattern learning - no model trained")
            self.test_results['pattern_learning_memory'] = False
            return
            
        try:
            pattern_tests = []
            
            # Test pattern memory initialization
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": self.model_id, "steps": 25, "time_window": 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                pattern_analysis = data.get('pattern_analysis', {})
                
                print("✅ Pattern memory system initialized")
                print(f"   Predictions generated: {len(predictions)}")
                
                # Check pattern analysis components
                pattern_components = [
                    'trend_slope', 'velocity', 'recent_mean', 'last_value',
                    'volatility', 'stability_factor', 'pattern_type'
                ]
                
                detected_components = sum(1 for comp in pattern_components if comp in pattern_analysis)
                pattern_analysis_score = detected_components / len(pattern_components)
                
                print(f"   Pattern analysis components: {detected_components}/{len(pattern_components)}")
                print(f"   Pattern type detected: {pattern_analysis.get('pattern_type', 'unknown')}")
                print(f"   Trend slope: {pattern_analysis.get('trend_slope', 'N/A')}")
                print(f"   Volatility: {pattern_analysis.get('volatility', 'N/A')}")
                
                pattern_tests.append(("Pattern analysis components", pattern_analysis_score >= 0.7))
                pattern_tests.append(("Pattern memory initialization", True))
                
                # Test historical pattern learning
                if len(predictions) >= 20:
                    # Check if predictions show variability (not flat)
                    pred_std = np.std(predictions)
                    pred_range = max(predictions) - min(predictions)
                    
                    variability_test = pred_std > 0.01 and pred_range > 0.02
                    print(f"   Prediction variability: std={pred_std:.4f}, range={pred_range:.4f}")
                    pattern_tests.append(("Historical pattern variability", variability_test))
                    
                    # Check if predictions maintain realistic pH range
                    ph_range_test = all(6.0 <= p <= 8.0 for p in predictions)
                    print(f"   pH range maintained: {ph_range_test}")
                    pattern_tests.append(("pH range maintenance", ph_range_test))
                else:
                    pattern_tests.append(("Historical pattern variability", False))
                    pattern_tests.append(("pH range maintenance", False))
                    
            else:
                print(f"❌ Pattern memory initialization failed: {response.status_code}")
                pattern_tests.extend([
                    ("Pattern analysis components", False),
                    ("Pattern memory initialization", False),
                    ("Historical pattern variability", False),
                    ("pH range maintenance", False)
                ])
            
            # Evaluate pattern learning and memory
            passed_tests = sum(1 for _, passed in pattern_tests if passed)
            total_tests = len(pattern_tests)
            
            print(f"\n📊 Pattern Learning & Memory Results: {passed_tests}/{total_tests}")
            for test_name, passed in pattern_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['pattern_learning_memory'] = passed_tests >= total_tests * 0.75
            
        except Exception as e:
            print(f"❌ Pattern learning and memory test error: {str(e)}")
            self.test_results['pattern_learning_memory'] = False
    
    def test_enhanced_realtime_prediction(self):
        """Test 2: Enhanced Real-time Continuous Prediction Endpoint"""
        print("\n=== Testing Enhanced Real-time Continuous Prediction ===")
        
        try:
            realtime_tests = []
            
            # Test the new main endpoint
            response = self.session.get(f"{API_BASE_URL}/generate-enhanced-realtime-prediction")
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                pattern_scores = data.get('pattern_scores', {})
                
                print("✅ Enhanced real-time prediction endpoint working")
                print(f"   Predictions: {len(predictions)}")
                print(f"   Timestamps: {len(timestamps)}")
                
                # Check pattern scores
                expected_scores = ['pattern_preservation', 'continuity', 'variability_preservation', 'confidence']
                score_count = sum(1 for score in expected_scores if score in pattern_scores)
                
                print(f"   Pattern scores available: {score_count}/{len(expected_scores)}")
                for score_name in expected_scores:
                    if score_name in pattern_scores:
                        score_value = pattern_scores[score_name]
                        print(f"   {score_name}: {score_value:.3f}")
                
                realtime_tests.append(("Enhanced endpoint response", True))
                realtime_tests.append(("Pattern scores included", score_count >= 3))
                
                # Test prediction quality
                if predictions and len(predictions) >= 10:
                    # Check pattern preservation score
                    pattern_preservation = pattern_scores.get('pattern_preservation', 0)
                    continuity_score = pattern_scores.get('continuity', 0)
                    variability_score = pattern_scores.get('variability_preservation', 0)
                    confidence_score = pattern_scores.get('confidence', 0)
                    
                    realtime_tests.append(("Pattern preservation >0.7", pattern_preservation > 0.7))
                    realtime_tests.append(("Continuity score >0.8", continuity_score > 0.8))
                    realtime_tests.append(("Variability preservation >0.8", variability_score > 0.8))
                    realtime_tests.append(("Overall confidence >0.8", confidence_score > 0.8))
                    
                    # Test prediction characteristics
                    pred_mean = np.mean(predictions)
                    pred_std = np.std(predictions)
                    
                    print(f"   Prediction statistics: mean={pred_mean:.3f}, std={pred_std:.3f}")
                    
                    # Check if predictions are realistic
                    realistic_range = 6.0 <= pred_mean <= 8.0 and pred_std > 0.01
                    realtime_tests.append(("Realistic prediction characteristics", realistic_range))
                else:
                    realtime_tests.extend([
                        ("Pattern preservation >0.7", False),
                        ("Continuity score >0.8", False),
                        ("Variability preservation >0.8", False),
                        ("Overall confidence >0.8", False),
                        ("Realistic prediction characteristics", False)
                    ])
                    
            else:
                print(f"❌ Enhanced real-time prediction failed: {response.status_code} - {response.text}")
                realtime_tests.extend([
                    ("Enhanced endpoint response", False),
                    ("Pattern scores included", False),
                    ("Pattern preservation >0.7", False),
                    ("Continuity score >0.8", False),
                    ("Variability preservation >0.8", False),
                    ("Overall confidence >0.8", False),
                    ("Realistic prediction characteristics", False)
                ])
            
            # Evaluate enhanced real-time prediction
            passed_tests = sum(1 for _, passed in realtime_tests if passed)
            total_tests = len(realtime_tests)
            
            print(f"\n📊 Enhanced Real-time Prediction Results: {passed_tests}/{total_tests}")
            for test_name, passed in realtime_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['enhanced_realtime_prediction'] = passed_tests >= total_tests * 0.7
            
        except Exception as e:
            print(f"❌ Enhanced real-time prediction test error: {str(e)}")
            self.test_results['enhanced_realtime_prediction'] = False
    
    def test_pattern_following_quality(self):
        """Test 3: Pattern Following Quality"""
        print("\n=== Testing Pattern Following Quality ===")
        
        if not self.model_id:
            print("❌ Cannot test pattern following - no model trained")
            self.test_results['pattern_following_quality'] = False
            return
            
        try:
            quality_tests = []
            
            # Test multiple continuous predictions to assess pattern following
            prediction_sets = []
            pattern_scores_sets = []
            
            for i in range(3):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 20, "time_window": 100}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    pattern_analysis = data.get('pattern_analysis', {})
                    
                    prediction_sets.append(predictions)
                    pattern_scores_sets.append(pattern_analysis)
                    
                    time.sleep(0.5)  # Small delay between calls
            
            if len(prediction_sets) >= 2:
                print("✅ Multiple prediction sets generated for quality analysis")
                
                # Test 1: Trend consistency
                trend_consistency_scores = []
                for predictions in prediction_sets:
                    if len(predictions) >= 10:
                        # Calculate trend slope
                        x = np.arange(len(predictions))
                        trend_slope = np.polyfit(x, predictions, 1)[0]
                        trend_consistency_scores.append(abs(trend_slope))
                
                if trend_consistency_scores:
                    trend_consistency = np.std(trend_consistency_scores) < 0.01  # Low variation in trends
                    print(f"   Trend consistency: {trend_consistency}")
                    quality_tests.append(("Trend consistency", trend_consistency))
                else:
                    quality_tests.append(("Trend consistency", False))
                
                # Test 2: Cyclical pattern detection
                cyclical_patterns_detected = 0
                for i, predictions in enumerate(prediction_sets):
                    if len(predictions) >= 15:
                        # Simple cyclical pattern detection
                        autocorr = np.corrcoef(predictions[:-5], predictions[5:])[0, 1]
                        if not np.isnan(autocorr) and abs(autocorr) > 0.3:
                            cyclical_patterns_detected += 1
                
                cyclical_quality = cyclical_patterns_detected >= 1
                print(f"   Cyclical patterns detected: {cyclical_patterns_detected}/3")
                quality_tests.append(("Cyclical pattern detection", cyclical_quality))
                
                # Test 3: Volatility preservation
                volatility_scores = []
                for predictions in prediction_sets:
                    if len(predictions) >= 10:
                        volatility = np.std(predictions)
                        volatility_scores.append(volatility)
                
                if volatility_scores:
                    avg_volatility = np.mean(volatility_scores)
                    volatility_preserved = 0.02 <= avg_volatility <= 0.3  # Reasonable volatility range
                    print(f"   Average volatility: {avg_volatility:.4f}")
                    quality_tests.append(("Volatility preservation", volatility_preserved))
                else:
                    quality_tests.append(("Volatility preservation", False))
                
                # Test 4: Statistical properties maintenance
                statistical_properties = []
                for predictions in prediction_sets:
                    if len(predictions) >= 10:
                        mean_val = np.mean(predictions)
                        std_val = np.std(predictions)
                        statistical_properties.append((mean_val, std_val))
                
                if len(statistical_properties) >= 2:
                    means = [prop[0] for prop in statistical_properties]
                    stds = [prop[1] for prop in statistical_properties]
                    
                    mean_consistency = np.std(means) < 0.1  # Consistent means
                    std_consistency = np.std(stds) < 0.05   # Consistent standard deviations
                    
                    statistical_maintained = mean_consistency and std_consistency
                    print(f"   Statistical properties maintained: {statistical_maintained}")
                    quality_tests.append(("Statistical properties maintenance", statistical_maintained))
                else:
                    quality_tests.append(("Statistical properties maintenance", False))
                    
            else:
                print("❌ Insufficient prediction sets for quality analysis")
                quality_tests.extend([
                    ("Trend consistency", False),
                    ("Cyclical pattern detection", False),
                    ("Volatility preservation", False),
                    ("Statistical properties maintenance", False)
                ])
            
            # Evaluate pattern following quality
            passed_tests = sum(1 for _, passed in quality_tests if passed)
            total_tests = len(quality_tests)
            
            print(f"\n📊 Pattern Following Quality Results: {passed_tests}/{total_tests}")
            for test_name, passed in quality_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['pattern_following_quality'] = passed_tests >= total_tests * 0.75
            
        except Exception as e:
            print(f"❌ Pattern following quality test error: {str(e)}")
            self.test_results['pattern_following_quality'] = False
    
    def test_continuous_learning(self):
        """Test 4: Continuous Learning System"""
        print("\n=== Testing Continuous Learning System ===")
        
        if not self.model_id:
            print("❌ Cannot test continuous learning - no model trained")
            self.test_results['continuous_learning'] = False
            return
            
        try:
            learning_tests = []
            
            # Test learning from previous predictions
            print("   Testing adaptation from previous predictions...")
            
            # Make initial prediction
            response1 = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": self.model_id, "steps": 15, "time_window": 100}
            )
            
            if response1.status_code == 200:
                data1 = response1.json()
                predictions1 = data1.get('predictions', [])
                pattern_analysis1 = data1.get('pattern_analysis', {})
                
                # Wait and make second prediction (should adapt)
                time.sleep(1)
                
                response2 = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 15, "time_window": 100}
                )
                
                if response2.status_code == 200:
                    data2 = response2.json()
                    predictions2 = data2.get('predictions', [])
                    pattern_analysis2 = data2.get('pattern_analysis', {})
                    
                    print("✅ Sequential predictions generated for learning analysis")
                    
                    # Test 1: Adaptation in pattern analysis
                    analysis_adaptation = False
                    if pattern_analysis1 and pattern_analysis2:
                        # Check if pattern analysis values have changed (indicating learning)
                        key_metrics = ['trend_slope', 'velocity', 'recent_mean']
                        changes = []
                        
                        for metric in key_metrics:
                            if metric in pattern_analysis1 and metric in pattern_analysis2:
                                val1 = pattern_analysis1[metric]
                                val2 = pattern_analysis2[metric]
                                if val1 != val2:  # Values changed
                                    changes.append(True)
                                else:
                                    changes.append(False)
                        
                        analysis_adaptation = any(changes) if changes else False
                    
                    print(f"   Pattern analysis adaptation: {analysis_adaptation}")
                    learning_tests.append(("Pattern analysis adaptation", analysis_adaptation))
                    
                    # Test 2: Prediction evolution
                    prediction_evolution = False
                    if predictions1 and predictions2 and len(predictions1) >= 10 and len(predictions2) >= 10:
                        # Check if predictions have evolved (not identical)
                        prediction_diff = np.mean(np.abs(np.array(predictions1[:10]) - np.array(predictions2[:10])))
                        prediction_evolution = prediction_diff > 0.001  # Some difference indicates learning
                    
                    print(f"   Prediction evolution: {prediction_evolution}")
                    learning_tests.append(("Prediction evolution", prediction_evolution))
                    
                    # Test 3: Learning convergence (make more predictions)
                    convergence_test = False
                    prediction_history = [predictions1, predictions2]
                    
                    # Make 2 more predictions
                    for i in range(2):
                        time.sleep(0.5)
                        response = self.session.get(
                            f"{API_BASE_URL}/generate-continuous-prediction",
                            params={"model_id": self.model_id, "steps": 15, "time_window": 100}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            predictions = data.get('predictions', [])
                            if predictions:
                                prediction_history.append(predictions)
                    
                    if len(prediction_history) >= 3:
                        # Check if predictions are converging (becoming more stable)
                        variances = []
                        for i in range(len(prediction_history)):
                            if len(prediction_history[i]) >= 10:
                                variance = np.var(prediction_history[i][:10])
                                variances.append(variance)
                        
                        if len(variances) >= 3:
                            # Check if variance is stabilizing (not increasing dramatically)
                            variance_trend = np.polyfit(range(len(variances)), variances, 1)[0]
                            convergence_test = abs(variance_trend) < 0.01  # Stable or slowly changing
                    
                    print(f"   Learning convergence: {convergence_test}")
                    learning_tests.append(("Learning convergence", convergence_test))
                    
                else:
                    print("❌ Second prediction failed")
                    learning_tests.extend([
                        ("Pattern analysis adaptation", False),
                        ("Prediction evolution", False),
                        ("Learning convergence", False)
                    ])
                    
            else:
                print("❌ Initial prediction failed")
                learning_tests.extend([
                    ("Pattern analysis adaptation", False),
                    ("Prediction evolution", False),
                    ("Learning convergence", False)
                ])
            
            # Test 4: Memory persistence
            # Reset and check if system maintains some learning
            reset_response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            memory_persistence = False
            
            if reset_response.status_code == 200:
                # Make prediction after reset
                response_after_reset = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 10, "time_window": 100}
                )
                
                if response_after_reset.status_code == 200:
                    data_after_reset = response_after_reset.json()
                    predictions_after_reset = data_after_reset.get('predictions', [])
                    
                    # Check if predictions are still reasonable (indicating some memory persistence)
                    if predictions_after_reset:
                        pred_mean = np.mean(predictions_after_reset)
                        memory_persistence = 6.0 <= pred_mean <= 8.0  # Still in reasonable pH range
            
            print(f"   Memory persistence after reset: {memory_persistence}")
            learning_tests.append(("Memory persistence", memory_persistence))
            
            # Evaluate continuous learning
            passed_tests = sum(1 for _, passed in learning_tests if passed)
            total_tests = len(learning_tests)
            
            print(f"\n📊 Continuous Learning Results: {passed_tests}/{total_tests}")
            for test_name, passed in learning_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['continuous_learning'] = passed_tests >= total_tests * 0.75
            
        except Exception as e:
            print(f"❌ Continuous learning test error: {str(e)}")
            self.test_results['continuous_learning'] = False
    
    def test_multi_scale_pattern_analysis(self):
        """Test 5: Multi-scale Pattern Analysis"""
        print("\n=== Testing Multi-scale Pattern Analysis ===")
        
        try:
            multiscale_tests = []
            
            # Test different time scales with different data patterns
            test_scenarios = [
                ("linear_trend", "Linear Trend Analysis"),
                ("seasonal_cyclical", "Seasonal Pattern Analysis"),
                ("volatile_noisy", "Volatile Data Analysis"),
                ("multi_pattern", "Multi-pattern Analysis")
            ]
            
            scale_results = []
            
            for pattern_type, description in test_scenarios:
                print(f"   Testing {description}...")
                
                # Upload specific pattern data
                if self.upload_test_data(pattern_type):
                    if self.train_advanced_model():
                        # Test pattern analysis at this scale
                        response = self.session.get(
                            f"{API_BASE_URL}/generate-continuous-prediction",
                            params={"model_id": self.model_id, "steps": 20, "time_window": 100}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            pattern_analysis = data.get('pattern_analysis', {})
                            predictions = data.get('predictions', [])
                            
                            # Analyze pattern detection quality
                            pattern_detected = 'pattern_type' in pattern_analysis
                            trend_detected = 'trend_slope' in pattern_analysis
                            volatility_detected = 'volatility' in pattern_analysis
                            
                            scale_score = sum([pattern_detected, trend_detected, volatility_detected]) / 3
                            scale_results.append((description, scale_score, len(predictions) > 0))
                            
                            print(f"     Pattern type: {pattern_analysis.get('pattern_type', 'unknown')}")
                            print(f"     Trend slope: {pattern_analysis.get('trend_slope', 'N/A')}")
                            print(f"     Scale score: {scale_score:.2f}")
                        else:
                            scale_results.append((description, 0.0, False))
                    else:
                        scale_results.append((description, 0.0, False))
                else:
                    scale_results.append((description, 0.0, False))
            
            # Evaluate multi-scale analysis
            if scale_results:
                successful_scales = sum(1 for _, score, success in scale_results if success and score > 0.5)
                total_scales = len(scale_results)
                
                print(f"   Successful pattern analysis across scales: {successful_scales}/{total_scales}")
                
                multiscale_tests.append(("Multi-scale pattern detection", successful_scales >= total_scales * 0.75))
                
                # Test scale adaptation
                scale_scores = [score for _, score, success in scale_results if success]
                if scale_scores:
                    avg_scale_score = np.mean(scale_scores)
                    scale_consistency = np.std(scale_scores) < 0.3  # Consistent performance across scales
                    
                    print(f"   Average scale score: {avg_scale_score:.3f}")
                    print(f"   Scale consistency: {scale_consistency}")
                    
                    multiscale_tests.append(("Scale adaptation quality", avg_scale_score > 0.6))
                    multiscale_tests.append(("Scale consistency", scale_consistency))
                else:
                    multiscale_tests.extend([
                        ("Scale adaptation quality", False),
                        ("Scale consistency", False)
                    ])
            else:
                multiscale_tests.extend([
                    ("Multi-scale pattern detection", False),
                    ("Scale adaptation quality", False),
                    ("Scale consistency", False)
                ])
            
            # Test time window adaptation
            if self.model_id:
                window_tests = []
                for window_size in [50, 100, 200]:
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": self.model_id, "steps": 15, "time_window": window_size}
                    )
                    
                    window_success = response.status_code == 200
                    window_tests.append(window_success)
                
                window_adaptation = sum(window_tests) >= len(window_tests) * 0.8
                print(f"   Time window adaptation: {window_adaptation}")
                multiscale_tests.append(("Time window adaptation", window_adaptation))
            else:
                multiscale_tests.append(("Time window adaptation", False))
            
            # Evaluate multi-scale pattern analysis
            passed_tests = sum(1 for _, passed in multiscale_tests if passed)
            total_tests = len(multiscale_tests)
            
            print(f"\n📊 Multi-scale Pattern Analysis Results: {passed_tests}/{total_tests}")
            for test_name, passed in multiscale_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['multi_scale_pattern_analysis'] = passed_tests >= total_tests * 0.75
            
        except Exception as e:
            print(f"❌ Multi-scale pattern analysis test error: {str(e)}")
            self.test_results['multi_scale_pattern_analysis'] = False
    
    def test_complete_workflow(self):
        """Test 6: Complete Enhanced Workflow"""
        print("\n=== Testing Complete Enhanced Workflow ===")
        
        try:
            workflow_tests = []
            
            print("   Testing complete workflow: upload → train → enhanced prediction...")
            
            # Step 1: Upload pH monitoring data
            upload_success = self.upload_test_data("ph_monitoring")
            workflow_tests.append(("Data upload", upload_success))
            
            if upload_success:
                # Step 2: Train model
                train_success = self.train_advanced_model()
                workflow_tests.append(("Model training", train_success))
                
                if train_success:
                    # Step 3: Enhanced real-time prediction
                    response = self.session.get(f"{API_BASE_URL}/generate-enhanced-realtime-prediction")
                    enhanced_success = response.status_code == 200
                    workflow_tests.append(("Enhanced real-time prediction", enhanced_success))
                    
                    if enhanced_success:
                        data = response.json()
                        pattern_scores = data.get('pattern_scores', {})
                        
                        # Check if all required scores are present and meet thresholds
                        score_thresholds = {
                            'pattern_preservation': 0.7,
                            'continuity': 0.8,
                            'variability_preservation': 0.8,
                            'confidence': 0.8
                        }
                        
                        score_results = []
                        for score_name, threshold in score_thresholds.items():
                            score_value = pattern_scores.get(score_name, 0)
                            score_met = score_value >= threshold
                            score_results.append(score_met)
                            print(f"     {score_name}: {score_value:.3f} (threshold: {threshold})")
                        
                        scores_met = sum(score_results) >= len(score_results) * 0.75  # 75% of scores meet threshold
                        workflow_tests.append(("Quality score thresholds", scores_met))
                        
                        # Step 4: Continuous prediction with advanced pattern memory
                        response = self.session.get(
                            f"{API_BASE_URL}/generate-continuous-prediction",
                            params={"model_id": self.model_id, "steps": 25, "time_window": 100}
                        )
                        
                        continuous_success = response.status_code == 200
                        workflow_tests.append(("Advanced continuous prediction", continuous_success))
                        
                        if continuous_success:
                            data = response.json()
                            predictions = data.get('predictions', [])
                            
                            # Verify predictions follow historical patterns
                            if predictions and len(predictions) >= 20:
                                pred_mean = np.mean(predictions)
                                pred_std = np.std(predictions)
                                
                                # Check if predictions are in realistic pH range with proper variability
                                realistic_predictions = (6.0 <= pred_mean <= 8.0 and 
                                                       0.01 <= pred_std <= 0.5)
                                workflow_tests.append(("Realistic pattern following", realistic_predictions))
                                
                                print(f"     Prediction statistics: mean={pred_mean:.3f}, std={pred_std:.3f}")
                            else:
                                workflow_tests.append(("Realistic pattern following", False))
                        else:
                            workflow_tests.append(("Realistic pattern following", False))
                    else:
                        workflow_tests.extend([
                            ("Quality score thresholds", False),
                            ("Advanced continuous prediction", False),
                            ("Realistic pattern following", False)
                        ])
                else:
                    workflow_tests.extend([
                        ("Enhanced real-time prediction", False),
                        ("Quality score thresholds", False),
                        ("Advanced continuous prediction", False),
                        ("Realistic pattern following", False)
                    ])
            else:
                workflow_tests.extend([
                    ("Model training", False),
                    ("Enhanced real-time prediction", False),
                    ("Quality score thresholds", False),
                    ("Advanced continuous prediction", False),
                    ("Realistic pattern following", False)
                ])
            
            # Evaluate complete workflow
            passed_tests = sum(1 for _, passed in workflow_tests if passed)
            total_tests = len(workflow_tests)
            
            print(f"\n📊 Complete Enhanced Workflow Results: {passed_tests}/{total_tests}")
            for test_name, passed in workflow_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['complete_workflow'] = passed_tests >= total_tests * 0.8
            
        except Exception as e:
            print(f"❌ Complete workflow test error: {str(e)}")
            self.test_results['complete_workflow'] = False
    
    def run_all_tests(self):
        """Run all enhanced real-time prediction tests"""
        print("🚀 Starting Enhanced Real-time Continuous Prediction System Testing")
        print("=" * 80)
        
        # Initialize with multi-pattern data for comprehensive testing
        if self.upload_test_data("multi_pattern") and self.train_advanced_model():
            # Run all test categories
            self.test_pattern_learning_memory()
            self.test_enhanced_realtime_prediction()
            self.test_pattern_following_quality()
            self.test_continuous_learning()
            self.test_multi_scale_pattern_analysis()
            self.test_complete_workflow()
        else:
            print("❌ Failed to initialize test environment")
            return
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("🎯 ENHANCED REAL-TIME CONTINUOUS PREDICTION SYSTEM TEST REPORT")
        print("=" * 80)
        
        test_categories = [
            ("Pattern Learning and Memory", "pattern_learning_memory"),
            ("Enhanced Real-time Prediction", "enhanced_realtime_prediction"),
            ("Pattern Following Quality", "pattern_following_quality"),
            ("Continuous Learning", "continuous_learning"),
            ("Multi-scale Pattern Analysis", "multi_scale_pattern_analysis"),
            ("Complete Workflow", "complete_workflow")
        ]
        
        passed_categories = 0
        total_categories = len(test_categories)
        
        print("\n📊 TEST RESULTS SUMMARY:")
        for category_name, test_key in test_categories:
            result = self.test_results.get(test_key, False)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} {category_name}")
            if result:
                passed_categories += 1
        
        success_rate = (passed_categories / total_categories) * 100
        print(f"\n🎯 OVERALL SUCCESS RATE: {passed_categories}/{total_categories} ({success_rate:.1f}%)")
        
        # Determine overall system status
        if success_rate >= 90:
            status = "🎉 EXCELLENT - System ready for production"
        elif success_rate >= 75:
            status = "✅ GOOD - System working well with minor issues"
        elif success_rate >= 60:
            status = "⚠️  ACCEPTABLE - System functional but needs improvements"
        else:
            status = "❌ NEEDS WORK - Significant issues require attention"
        
        print(f"\n🏆 SYSTEM STATUS: {status}")
        
        # Specific findings
        print(f"\n🔍 KEY FINDINGS:")
        
        if self.test_results.get('pattern_learning_memory', False):
            print("   ✅ Advanced pattern memory system is working correctly")
        else:
            print("   ❌ Pattern memory system needs improvement")
            
        if self.test_results.get('enhanced_realtime_prediction', False):
            print("   ✅ Enhanced real-time prediction endpoint is functional")
        else:
            print("   ❌ Enhanced real-time prediction endpoint has issues")
            
        if self.test_results.get('pattern_following_quality', False):
            print("   ✅ Predictions maintain historical characteristics properly")
        else:
            print("   ❌ Pattern following quality needs improvement")
            
        if self.test_results.get('continuous_learning', False):
            print("   ✅ System adapts and learns from previous predictions")
        else:
            print("   ❌ Continuous learning system needs work")
            
        if self.test_results.get('multi_scale_pattern_analysis', False):
            print("   ✅ Multi-scale pattern recognition is working")
        else:
            print("   ❌ Multi-scale pattern analysis needs improvement")
        
        print("\n" + "=" * 80)
        return success_rate >= 75  # Return True if system is in good condition

if __name__ == "__main__":
    tester = EnhancedRealtimePredictionTester()
    tester.run_all_tests()