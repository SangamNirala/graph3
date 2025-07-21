#!/usr/bin/env python3
"""
Enhanced Real-time Graph Prediction System Testing
Tests the enhanced pattern analysis and prediction system with focus on pattern accuracy improvements
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://f576d419-9655-44df-95ef-dbabc9baf3ad.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing enhanced prediction system at: {API_BASE_URL}")

class EnhancedPredictionTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        self.test_results = {}
        
    def create_ph_dataset(self):
        """Create realistic pH time-series data for testing pattern analysis"""
        # Generate 49 samples of pH data with realistic patterns
        dates = pd.date_range(start='2024-01-01', periods=49, freq='H')
        
        # Create realistic pH data with trend, seasonality, and noise
        base_ph = 7.2
        trend = np.linspace(0, 0.6, 49)  # Slight upward trend
        seasonal = 0.3 * np.sin(2 * np.pi * np.arange(49) / 24)  # Daily cycle
        noise = np.random.normal(0, 0.05, 49)
        ph_values = base_ph + trend + seasonal + noise
        
        # Keep pH in realistic range (6.0-8.0)
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'pH': ph_values
        })
        
        return df
    
    def create_complex_pattern_dataset(self):
        """Create complex time-series data with multiple patterns for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create complex pattern with trend, seasonality, and cyclical components
        trend = np.linspace(1000, 1200, 100)  # Linear trend
        seasonal = 150 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly seasonality
        cyclical = 80 * np.sin(2 * np.pi * np.arange(100) / 30)  # Monthly cycle
        noise = np.random.normal(0, 25, 100)
        
        values = trend + seasonal + cyclical + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        return df
    
    def test_enhanced_data_preprocessing(self):
        """Test 1: Enhanced data preprocessing and quality validation"""
        print("\n=== Testing Enhanced Data Preprocessing and Quality Validation ===")
        
        try:
            # Create pH dataset for testing
            df = self.create_ph_dataset()
            csv_content = df.to_csv(index=False)
            
            # Test file upload with enhanced preprocessing
            files = {'file': ('ph_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                
                print("✅ Enhanced file upload successful")
                print(f"   Data ID: {self.data_id}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                
                # Test data quality report endpoint
                response = self.session.get(f"{API_BASE_URL}/data-quality-report")
                
                if response.status_code == 200:
                    quality_data = response.json()
                    quality_score = quality_data.get('quality_score', 0)
                    
                    print("✅ Data quality report successful")
                    print(f"   Quality score: {quality_score}")
                    print(f"   Status: {quality_data.get('status')}")
                    
                    # Validate quality report structure
                    if quality_score >= 90.0:  # Expect high quality for clean pH data
                        print("✅ Data quality validation working correctly")
                        self.test_results['enhanced_data_preprocessing'] = True
                    else:
                        print(f"❌ Unexpected quality score: {quality_score}")
                        self.test_results['enhanced_data_preprocessing'] = False
                        
                else:
                    print(f"❌ Data quality report failed: {response.status_code} - {response.text}")
                    self.test_results['enhanced_data_preprocessing'] = False
                    
            else:
                print(f"❌ Enhanced file upload failed: {response.status_code} - {response.text}")
                self.test_results['enhanced_data_preprocessing'] = False
                
        except Exception as e:
            print(f"❌ Enhanced data preprocessing test error: {str(e)}")
            self.test_results['enhanced_data_preprocessing'] = False
    
    def test_advanced_model_training(self):
        """Test 2: Advanced model training and hyperparameter optimization"""
        print("\n=== Testing Advanced Model Training and Hyperparameter Optimization ===")
        
        if not self.data_id:
            print("❌ Cannot test advanced model training - no data uploaded")
            self.test_results['advanced_model_training'] = False
            return
            
        try:
            # Test LSTM model training (most likely to work with small dataset)
            training_params = {
                "time_column": "timestamp",
                "target_column": "pH",
                "seq_len": 8,  # Adjusted for small dataset
                "pred_len": 3,
                "epochs": 10,
                "batch_size": 4
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": "lstm"},
                json=training_params
            )
            
            if response.status_code == 200:
                data = response.json()
                self.model_id = data.get('model_id')
                
                print("✅ Advanced LSTM model training successful")
                print(f"   Model ID: {self.model_id}")
                print(f"   Status: {data.get('status')}")
                print(f"   Performance grade: {data.get('evaluation_grade', 'N/A')}")
                
                # Test model performance endpoint
                response = self.session.get(f"{API_BASE_URL}/model-performance")
                
                if response.status_code == 200:
                    perf_data = response.json()
                    print("✅ Model performance retrieval successful")
                    print(f"   Performance metrics available: {len(perf_data.get('metrics', {}))}")
                    
                    # Test hyperparameter optimization
                    response = self.session.post(
                        f"{API_BASE_URL}/optimize-hyperparameters",
                        json={
                            "model_type": "lstm",
                            "n_trials": 5,  # Small number for testing
                            "time_column": "timestamp",
                            "target_column": "pH"
                        }
                    )
                    
                    if response.status_code == 200:
                        opt_data = response.json()
                        print("✅ Hyperparameter optimization successful")
                        print(f"   Best parameters found: {opt_data.get('best_params', {})}")
                        print(f"   Best score: {opt_data.get('best_score', 'N/A')}")
                        
                        self.test_results['advanced_model_training'] = True
                    else:
                        print(f"❌ Hyperparameter optimization failed: {response.status_code}")
                        self.test_results['advanced_model_training'] = False
                        
                else:
                    print(f"❌ Model performance retrieval failed: {response.status_code}")
                    self.test_results['advanced_model_training'] = False
                    
            else:
                print(f"❌ Advanced model training failed: {response.status_code} - {response.text}")
                self.test_results['advanced_model_training'] = False
                
        except Exception as e:
            print(f"❌ Advanced model training test error: {str(e)}")
            self.test_results['advanced_model_training'] = False
    
    def test_enhanced_pattern_analysis_endpoint(self):
        """Test 3: Enhanced pattern analysis endpoint"""
        print("\n=== Testing Enhanced Pattern Analysis Endpoint ===")
        
        if not self.model_id:
            print("❌ Cannot test pattern analysis - no model trained")
            self.test_results['enhanced_pattern_analysis'] = False
            return
            
        try:
            # Test enhanced pattern analysis endpoint
            response = self.session.get(f"{API_BASE_URL}/enhanced-pattern-analysis")
            
            if response.status_code == 200:
                data = response.json()
                pattern_analysis = data.get('pattern_analysis', {})
                recommendations = data.get('recommendations', {})
                
                print("✅ Enhanced pattern analysis successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Data length: {data.get('data_preview', {}).get('data_length', 'N/A')}")
                
                # Validate pattern analysis components
                expected_components = [
                    'trend_analysis', 'seasonal_analysis', 'cyclical_analysis',
                    'volatility_analysis', 'predictability', 'quality_score'
                ]
                
                components_found = []
                for component in expected_components:
                    if component in pattern_analysis:
                        components_found.append(component)
                        print(f"   ✅ {component}: Found")
                    else:
                        print(f"   ❌ {component}: Missing")
                
                # Validate recommendations
                if recommendations:
                    print("✅ Prediction recommendations generated:")
                    print(f"   Optimal method: {recommendations.get('optimal_prediction_method')}")
                    print(f"   Recommended steps: {recommendations.get('recommended_steps')}")
                    print(f"   Insights: {len(recommendations.get('insights', []))}")
                
                # Test passes if most components are found
                if len(components_found) >= len(expected_components) * 0.7:
                    print("✅ Pattern analysis components validation passed")
                    self.test_results['enhanced_pattern_analysis'] = True
                else:
                    print("❌ Pattern analysis components validation failed")
                    self.test_results['enhanced_pattern_analysis'] = False
                    
            else:
                print(f"❌ Enhanced pattern analysis failed: {response.status_code} - {response.text}")
                self.test_results['enhanced_pattern_analysis'] = False
                
        except Exception as e:
            print(f"❌ Enhanced pattern analysis test error: {str(e)}")
            self.test_results['enhanced_pattern_analysis'] = False
    
    def test_enhanced_continuous_prediction(self):
        """Test 4: Enhanced continuous prediction with pattern preservation"""
        print("\n=== Testing Enhanced Continuous Prediction ===")
        
        if not self.model_id:
            print("❌ Cannot test enhanced continuous prediction - no model trained")
            self.test_results['enhanced_continuous_prediction'] = False
            return
            
        try:
            # Test enhanced continuous prediction endpoint
            response = self.session.get(
                f"{API_BASE_URL}/generate-enhanced-continuous-prediction",
                params={"model_id": self.model_id, "steps": 20, "time_window": 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                pattern_analysis = data.get('pattern_analysis', {})
                enhancement_info = data.get('enhancement_info', {})
                
                print("✅ Enhanced continuous prediction successful")
                print(f"   Number of predictions: {len(predictions)}")
                print(f"   Number of timestamps: {len(timestamps)}")
                print(f"   Is enhanced: {data.get('is_enhanced', False)}")
                
                # Validate enhanced features
                if pattern_analysis:
                    print("✅ Pattern analysis included:")
                    print(f"   Prediction method: {pattern_analysis.get('prediction_method')}")
                    print(f"   Pattern preservation score: {pattern_analysis.get('pattern_preservation_score')}")
                    print(f"   Quality metrics: {len(pattern_analysis.get('quality_metrics', {}))}")
                
                if enhancement_info:
                    print("✅ Enhancement info included:")
                    print(f"   Trend strength: {enhancement_info.get('trend_strength')}")
                    print(f"   Seasonal strength: {enhancement_info.get('seasonal_strength')}")
                    print(f"   Predictability score: {enhancement_info.get('predictability_score')}")
                    print(f"   Pattern quality: {enhancement_info.get('pattern_quality')}")
                
                # Test pattern preservation - predictions should maintain realistic pH range
                if predictions:
                    ph_predictions = [p for p in predictions if isinstance(p, (int, float))]
                    if ph_predictions:
                        min_pred = min(ph_predictions)
                        max_pred = max(ph_predictions)
                        
                        if 5.5 <= min_pred <= 8.5 and 5.5 <= max_pred <= 8.5:
                            print("✅ Predictions maintain realistic pH range (5.5-8.5)")
                            pattern_preservation = True
                        else:
                            print(f"❌ Predictions outside realistic range: {min_pred:.2f} - {max_pred:.2f}")
                            pattern_preservation = False
                    else:
                        pattern_preservation = False
                else:
                    pattern_preservation = False
                
                # Test multiple calls for continuous behavior
                print("   Testing continuous prediction behavior...")
                response2 = self.session.get(
                    f"{API_BASE_URL}/generate-enhanced-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 20, "time_window": 100}
                )
                
                if response2.status_code == 200:
                    data2 = response2.json()
                    timestamps2 = data2.get('timestamps', [])
                    
                    # Check if timestamps advance (indicating continuous behavior)
                    if timestamps != timestamps2:
                        print("✅ Continuous prediction properly advances timestamps")
                        continuous_behavior = True
                    else:
                        print("❌ Continuous prediction not advancing timestamps")
                        continuous_behavior = False
                else:
                    continuous_behavior = False
                
                # Overall test result
                self.test_results['enhanced_continuous_prediction'] = (
                    pattern_preservation and continuous_behavior and 
                    bool(pattern_analysis) and bool(enhancement_info)
                )
                
            else:
                print(f"❌ Enhanced continuous prediction failed: {response.status_code} - {response.text}")
                self.test_results['enhanced_continuous_prediction'] = False
                
        except Exception as e:
            print(f"❌ Enhanced continuous prediction test error: {str(e)}")
            self.test_results['enhanced_continuous_prediction'] = False
    
    def test_pattern_accuracy_improvements(self):
        """Test 5: Pattern accuracy improvements with different data types"""
        print("\n=== Testing Pattern Accuracy Improvements ===")
        
        try:
            # Test with complex pattern dataset
            df = self.create_complex_pattern_dataset()
            csv_content = df.to_csv(index=False)
            
            # Upload complex pattern data
            files = {'file': ('complex_pattern.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                complex_data_id = data.get('data_id')
                
                print("✅ Complex pattern data uploaded successfully")
                
                # Train model on complex data
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": complex_data_id, "model_type": "arima"},
                    json={"time_column": "date", "target_column": "value", "order": [1, 1, 1]}
                )
                
                if response.status_code == 200:
                    model_data = response.json()
                    complex_model_id = model_data.get('model_id')
                    
                    print("✅ Model trained on complex pattern data")
                    
                    # Test pattern analysis on complex data
                    response = self.session.get(f"{API_BASE_URL}/enhanced-pattern-analysis")
                    
                    if response.status_code == 200:
                        pattern_data = response.json()
                        pattern_analysis = pattern_data.get('pattern_analysis', {})
                        
                        # Check if complex patterns are detected
                        trend_strength = pattern_analysis.get('trend_analysis', {}).get('trend_strength', 0)
                        seasonal_strength = pattern_analysis.get('seasonal_analysis', {}).get('seasonal_strength', 0)
                        
                        print(f"✅ Complex pattern analysis results:")
                        print(f"   Trend strength: {trend_strength}")
                        print(f"   Seasonal strength: {seasonal_strength}")
                        
                        # Test enhanced prediction on complex data
                        response = self.session.get(
                            f"{API_BASE_URL}/generate-enhanced-continuous-prediction",
                            params={"model_id": complex_model_id, "steps": 30, "time_window": 100}
                        )
                        
                        if response.status_code == 200:
                            pred_data = response.json()
                            predictions = pred_data.get('predictions', [])
                            enhancement_info = pred_data.get('enhancement_info', {})
                            
                            print("✅ Enhanced prediction on complex data successful")
                            print(f"   Predictions generated: {len(predictions)}")
                            print(f"   Pattern quality: {enhancement_info.get('pattern_quality', 'N/A')}")
                            
                            # Validate prediction quality
                            if predictions and len(predictions) == 30:
                                # Check for reasonable prediction values
                                pred_mean = np.mean(predictions)
                                pred_std = np.std(predictions)
                                
                                if 800 <= pred_mean <= 1400 and pred_std > 0:  # Reasonable for our test data
                                    print("✅ Predictions show realistic values and variability")
                                    self.test_results['pattern_accuracy_improvements'] = True
                                else:
                                    print(f"❌ Predictions unrealistic: mean={pred_mean:.2f}, std={pred_std:.2f}")
                                    self.test_results['pattern_accuracy_improvements'] = False
                            else:
                                print("❌ Prediction count mismatch")
                                self.test_results['pattern_accuracy_improvements'] = False
                                
                        else:
                            print(f"❌ Enhanced prediction on complex data failed: {response.status_code}")
                            self.test_results['pattern_accuracy_improvements'] = False
                            
                    else:
                        print(f"❌ Pattern analysis on complex data failed: {response.status_code}")
                        self.test_results['pattern_accuracy_improvements'] = False
                        
                else:
                    print(f"❌ Model training on complex data failed: {response.status_code}")
                    self.test_results['pattern_accuracy_improvements'] = False
                    
            else:
                print(f"❌ Complex pattern data upload failed: {response.status_code}")
                self.test_results['pattern_accuracy_improvements'] = False
                
        except Exception as e:
            print(f"❌ Pattern accuracy improvements test error: {str(e)}")
            self.test_results['pattern_accuracy_improvements'] = False
    
    def test_real_time_prediction_updates(self):
        """Test 6: Real-time prediction updates and frequency handling"""
        print("\n=== Testing Real-time Prediction Updates ===")
        
        if not self.model_id:
            print("❌ Cannot test real-time updates - no model trained")
            self.test_results['real_time_updates'] = False
            return
            
        try:
            # Test rapid successive calls to simulate real-time updates
            print("   Testing rapid successive prediction calls...")
            
            timestamps_list = []
            predictions_list = []
            response_times = []
            
            for i in range(5):  # Test 5 rapid calls
                start_time = time.time()
                
                response = self.session.get(
                    f"{API_BASE_URL}/generate-enhanced-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 10, "time_window": 50}
                )
                
                end_time = time.time()
                response_times.append(end_time - start_time)
                
                if response.status_code == 200:
                    data = response.json()
                    timestamps_list.append(data.get('timestamps', []))
                    predictions_list.append(data.get('predictions', []))
                    print(f"   Call {i+1}: ✅ Success ({response_times[-1]:.3f}s)")
                else:
                    print(f"   Call {i+1}: ❌ Failed ({response.status_code})")
                
                time.sleep(0.5)  # Brief pause between calls
            
            # Analyze results
            successful_calls = len([p for p in predictions_list if p])
            avg_response_time = np.mean(response_times) if response_times else 0
            
            print(f"✅ Real-time update test results:")
            print(f"   Successful calls: {successful_calls}/5")
            print(f"   Average response time: {avg_response_time:.3f}s")
            
            # Check if predictions are advancing (continuous behavior)
            advancing_timestamps = 0
            for i in range(1, len(timestamps_list)):
                if timestamps_list[i] and timestamps_list[i-1]:
                    if timestamps_list[i] != timestamps_list[i-1]:
                        advancing_timestamps += 1
            
            print(f"   Advancing timestamps: {advancing_timestamps}/{len(timestamps_list)-1}")
            
            # Test passes if most calls succeed and timestamps advance
            if successful_calls >= 4 and advancing_timestamps >= 3 and avg_response_time < 5.0:
                print("✅ Real-time prediction updates working correctly")
                self.test_results['real_time_updates'] = True
            else:
                print("❌ Real-time prediction updates not meeting requirements")
                self.test_results['real_time_updates'] = False
                
        except Exception as e:
            print(f"❌ Real-time updates test error: {str(e)}")
            self.test_results['real_time_updates'] = False
    
    def test_prediction_smoothness_and_continuity(self):
        """Test 7: Prediction smoothness and continuity"""
        print("\n=== Testing Prediction Smoothness and Continuity ===")
        
        if not self.model_id:
            print("❌ Cannot test smoothness - no model trained")
            self.test_results['prediction_smoothness'] = False
            return
            
        try:
            # Generate multiple prediction sequences to test smoothness
            prediction_sequences = []
            
            for i in range(3):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-enhanced-continuous-prediction",
                    params={"model_id": self.model_id, "steps": 15, "time_window": 100}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    if predictions:
                        prediction_sequences.append(predictions)
                        print(f"   Sequence {i+1}: {len(predictions)} predictions generated")
                
                time.sleep(1)  # Wait between sequences
            
            if len(prediction_sequences) >= 2:
                # Analyze smoothness within sequences
                smoothness_scores = []
                
                for seq in prediction_sequences:
                    if len(seq) > 1:
                        # Calculate smoothness as inverse of average absolute difference
                        diffs = [abs(seq[i+1] - seq[i]) for i in range(len(seq)-1)]
                        avg_diff = np.mean(diffs)
                        smoothness = 1.0 / (1.0 + avg_diff)  # Normalize to 0-1
                        smoothness_scores.append(smoothness)
                
                avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0
                print(f"✅ Smoothness analysis:")
                print(f"   Average smoothness score: {avg_smoothness:.3f}")
                
                # Test continuity between sequences
                continuity_gaps = []
                for i in range(1, len(prediction_sequences)):
                    if prediction_sequences[i] and prediction_sequences[i-1]:
                        # Check gap between end of previous and start of current
                        gap = abs(prediction_sequences[i][0] - prediction_sequences[i-1][-1])
                        continuity_gaps.append(gap)
                
                avg_gap = np.mean(continuity_gaps) if continuity_gaps else 0
                print(f"   Average continuity gap: {avg_gap:.3f}")
                
                # Test passes if smoothness is reasonable and gaps are not too large
                if avg_smoothness > 0.3 and avg_gap < 2.0:  # Thresholds for pH data
                    print("✅ Prediction smoothness and continuity acceptable")
                    self.test_results['prediction_smoothness'] = True
                else:
                    print("❌ Prediction smoothness or continuity issues detected")
                    self.test_results['prediction_smoothness'] = False
                    
            else:
                print("❌ Insufficient prediction sequences for smoothness testing")
                self.test_results['prediction_smoothness'] = False
                
        except Exception as e:
            print(f"❌ Prediction smoothness test error: {str(e)}")
            self.test_results['prediction_smoothness'] = False
    
    def run_enhanced_prediction_tests(self):
        """Run all enhanced prediction system tests"""
        print("🚀 Starting Enhanced Real-time Graph Prediction System Testing")
        print("=" * 70)
        
        # Run all enhanced tests
        self.test_enhanced_data_preprocessing()
        self.test_advanced_model_training()
        self.test_enhanced_pattern_analysis_endpoint()
        self.test_enhanced_continuous_prediction()
        self.test_pattern_accuracy_improvements()
        self.test_real_time_prediction_updates()
        self.test_prediction_smoothness_and_continuity()
        
        # Generate comprehensive test report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("🎯 ENHANCED PREDICTION SYSTEM TEST REPORT")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        print()
        
        # Detailed results
        test_categories = {
            'enhanced_data_preprocessing': 'Enhanced Data Preprocessing & Quality Validation',
            'advanced_model_training': 'Advanced Model Training & Hyperparameter Optimization',
            'enhanced_pattern_analysis': 'Enhanced Pattern Analysis Endpoint',
            'enhanced_continuous_prediction': 'Enhanced Continuous Prediction',
            'pattern_accuracy_improvements': 'Pattern Accuracy Improvements',
            'real_time_updates': 'Real-time Prediction Updates',
            'prediction_smoothness': 'Prediction Smoothness & Continuity'
        }
        
        for test_key, test_name in test_categories.items():
            result = self.test_results.get(test_key, False)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print()
        
        # Key findings
        print("🔍 Key Findings:")
        
        if self.test_results.get('enhanced_data_preprocessing', False):
            print("   ✅ Enhanced data preprocessing working correctly")
        else:
            print("   ❌ Enhanced data preprocessing has issues")
        
        if self.test_results.get('advanced_model_training', False):
            print("   ✅ Advanced model training and optimization functional")
        else:
            print("   ❌ Advanced model training needs attention")
        
        if self.test_results.get('enhanced_continuous_prediction', False):
            print("   ✅ Enhanced continuous prediction with pattern analysis working")
        else:
            print("   ❌ Enhanced continuous prediction system has issues")
        
        if self.test_results.get('pattern_accuracy_improvements', False):
            print("   ✅ Pattern accuracy improvements verified")
        else:
            print("   ❌ Pattern accuracy improvements not working as expected")
        
        # Recommendations
        print()
        print("💡 Recommendations:")
        
        if success_rate >= 80:
            print("   🎉 Enhanced prediction system is working well!")
            print("   📈 Pattern accuracy improvements are effective")
            print("   🔄 Real-time updates are functioning properly")
        elif success_rate >= 60:
            print("   ⚠️  Enhanced prediction system partially working")
            print("   🔧 Some components need attention")
        else:
            print("   🚨 Enhanced prediction system needs significant work")
            print("   🛠️  Multiple components require fixes")
        
        print()
        print("=" * 70)

if __name__ == "__main__":
    tester = EnhancedPredictionTester()
    tester.run_enhanced_prediction_tests()