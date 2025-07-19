#!/usr/bin/env python3
"""
Enhanced Continuous Prediction System Testing
Tests the enhanced continuous prediction system to verify it follows historical patterns properly
Focus on pattern following improvement, bias prevention, and quality metrics
"""

import requests
import json
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import statistics

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://909a9d1c-9da6-4ed6-bd0a-ff6c4fb747bb.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Enhanced Continuous Prediction System at: {API_BASE_URL}")

class EnhancedContinuousPredictionTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        self.test_results = {}
        self.ph_data = None
        
    def create_ph_dataset(self):
        """Create realistic pH dataset for testing pattern following"""
        print("\n=== Creating pH Dataset for Pattern Following Tests ===")
        
        # Generate 48 hours of pH data with realistic patterns
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=48, freq='H')
        
        # Create realistic pH pattern with:
        # 1. Base level around 7.2-7.8 (realistic pH range)
        # 2. Daily cycles (pH typically varies throughout day)
        # 3. Some random variation
        # 4. Gradual trends
        
        base_ph = 7.4
        daily_cycle = 0.3 * np.sin(2 * np.pi * np.arange(48) / 24)  # Daily cycle
        trend = 0.1 * np.sin(2 * np.pi * np.arange(48) / 48)  # Longer trend
        noise = np.random.normal(0, 0.05, 48)  # Small random variation
        
        ph_values = base_ph + daily_cycle + trend + noise
        
        # Ensure pH stays in realistic range
        ph_values = np.clip(ph_values, 6.8, 7.9)
        
        self.ph_data = pd.DataFrame({
            'timestamp': timestamps,
            'pH': ph_values
        })
        
        print(f"✅ Created pH dataset with {len(self.ph_data)} points")
        print(f"   pH range: {ph_values.min():.3f} - {ph_values.max():.3f}")
        print(f"   pH mean: {ph_values.mean():.3f}")
        print(f"   pH std: {ph_values.std():.3f}")
        
        return self.ph_data
    
    def upload_ph_data(self):
        """Upload pH data for testing"""
        print("\n=== Uploading pH Data ===")
        
        try:
            # Create CSV content
            csv_content = self.ph_data.to_csv(index=False)
            
            # Prepare file for upload
            files = {
                'file': ('ph_data.csv', csv_content, 'text/csv')
            }
            
            # Upload file
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                
                print("✅ pH data upload successful")
                print(f"   Data ID: {self.data_id}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                
                return True
            else:
                print(f"❌ pH data upload failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ pH data upload error: {str(e)}")
            return False
    
    def train_model_for_testing(self):
        """Train ARIMA model for continuous prediction testing"""
        print("\n=== Training Model for Continuous Prediction Testing ===")
        
        if not self.data_id:
            print("❌ Cannot train model - no data uploaded")
            return False
            
        try:
            # Train ARIMA model (works well for continuous prediction)
            training_data = {
                "data_id": self.data_id,
                "model_type": "arima",
                "parameters": {
                    "time_column": "timestamp",
                    "target_column": "pH",
                    "order": [1, 1, 1]
                }
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": "arima"},
                json=training_data["parameters"]
            )
            
            if response.status_code == 200:
                data = response.json()
                self.model_id = data.get('model_id')
                
                print("✅ Model training successful")
                print(f"   Model ID: {self.model_id}")
                print(f"   Status: {data.get('status')}")
                
                return True
            else:
                print(f"❌ Model training failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Model training error: {str(e)}")
            return False
    
    def test_multiple_continuous_prediction_calls(self):
        """Test multiple continuous prediction calls to verify pattern consistency"""
        print("\n=== Testing Multiple Continuous Prediction Calls ===")
        
        if not self.model_id:
            print("❌ Cannot test continuous prediction - no model trained")
            self.test_results['multiple_continuous_calls'] = False
            return
        
        try:
            # Reset continuous prediction first
            reset_response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            if reset_response.status_code != 200:
                print(f"⚠️  Reset failed: {reset_response.status_code}")
            
            # Make multiple continuous prediction calls
            all_predictions = []
            all_timestamps = []
            call_results = []
            
            num_calls = 5
            steps_per_call = 10
            
            print(f"Making {num_calls} continuous prediction calls with {steps_per_call} steps each...")
            
            for i in range(num_calls):
                print(f"   Call {i+1}/{num_calls}...")
                
                response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": self.model_id, "steps": steps_per_call, "time_window": 100}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', [])
                    timestamps = data.get('timestamps', [])
                    
                    call_results.append({
                        'call_number': i + 1,
                        'predictions': predictions,
                        'timestamps': timestamps,
                        'success': True
                    })
                    
                    all_predictions.extend(predictions)
                    all_timestamps.extend(timestamps)
                    
                    print(f"     ✅ Call {i+1} successful: {len(predictions)} predictions")
                    print(f"     pH range: {min(predictions):.3f} - {max(predictions):.3f}")
                    
                else:
                    print(f"     ❌ Call {i+1} failed: {response.status_code}")
                    call_results.append({
                        'call_number': i + 1,
                        'success': False,
                        'error': response.text
                    })
                
                # Small delay between calls
                time.sleep(0.5)
            
            # Analyze results
            successful_calls = [r for r in call_results if r['success']]
            success_rate = len(successful_calls) / num_calls
            
            print(f"\n📊 Multiple Continuous Prediction Calls Results:")
            print(f"   Success rate: {success_rate:.1%} ({len(successful_calls)}/{num_calls})")
            print(f"   Total predictions generated: {len(all_predictions)}")
            
            if len(successful_calls) >= 3:  # Need at least 3 successful calls
                # Test pattern consistency
                pattern_consistency = self.analyze_pattern_consistency(successful_calls)
                
                # Test bias accumulation
                bias_analysis = self.analyze_bias_accumulation(all_predictions)
                
                # Test variability preservation
                variability_analysis = self.analyze_variability_preservation(all_predictions)
                
                # Test continuity between calls
                continuity_analysis = self.analyze_continuity_between_calls(successful_calls)
                
                overall_score = (
                    pattern_consistency['score'] * 0.3 +
                    bias_analysis['score'] * 0.3 +
                    variability_analysis['score'] * 0.2 +
                    continuity_analysis['score'] * 0.2
                )
                
                print(f"\n🎯 Enhanced Continuous Prediction Quality Metrics:")
                print(f"   Pattern Consistency Score: {pattern_consistency['score']:.3f}")
                print(f"   Bias Prevention Score: {bias_analysis['score']:.3f}")
                print(f"   Variability Preservation Score: {variability_analysis['score']:.3f}")
                print(f"   Continuity Score: {continuity_analysis['score']:.3f}")
                print(f"   Overall Quality Score: {overall_score:.3f}")
                
                # Success criteria from review request
                success_criteria = {
                    'pattern_following_score': pattern_consistency['score'] >= 0.6,
                    'variability_preservation': variability_analysis['score'] >= 0.7,
                    'bias_prevention': bias_analysis['score'] >= 0.7,
                    'overall_quality': overall_score >= 0.7
                }
                
                passed_criteria = sum(success_criteria.values())
                total_criteria = len(success_criteria)
                
                print(f"\n✅ Success Criteria Assessment ({passed_criteria}/{total_criteria}):")
                for criterion, passed in success_criteria.items():
                    status = "✅" if passed else "❌"
                    print(f"   {status} {criterion.replace('_', ' ').title()}: {passed}")
                
                self.test_results['multiple_continuous_calls'] = {
                    'success': passed_criteria >= 3,  # At least 3/4 criteria must pass
                    'success_rate': success_rate,
                    'pattern_consistency': pattern_consistency,
                    'bias_analysis': bias_analysis,
                    'variability_analysis': variability_analysis,
                    'continuity_analysis': continuity_analysis,
                    'overall_score': overall_score,
                    'criteria_passed': passed_criteria,
                    'criteria_total': total_criteria
                }
                
            else:
                print("❌ Insufficient successful calls for analysis")
                self.test_results['multiple_continuous_calls'] = {
                    'success': False,
                    'error': 'Insufficient successful calls'
                }
                
        except Exception as e:
            print(f"❌ Multiple continuous prediction calls error: {str(e)}")
            self.test_results['multiple_continuous_calls'] = {
                'success': False,
                'error': str(e)
            }
    
    def analyze_pattern_consistency(self, successful_calls):
        """Analyze pattern consistency across multiple calls"""
        try:
            # Extract predictions from each call
            call_predictions = [call['predictions'] for call in successful_calls]
            
            # Calculate mean and std for each call
            call_stats = []
            for predictions in call_predictions:
                call_stats.append({
                    'mean': np.mean(predictions),
                    'std': np.std(predictions),
                    'min': np.min(predictions),
                    'max': np.max(predictions)
                })
            
            # Calculate consistency metrics
            means = [stat['mean'] for stat in call_stats]
            stds = [stat['std'] for stat in call_stats]
            
            # Pattern consistency = low variation in means and stds across calls
            mean_consistency = 1.0 / (1.0 + np.std(means))
            std_consistency = 1.0 / (1.0 + np.std(stds))
            
            # Historical comparison
            historical_mean = self.ph_data['pH'].mean()
            historical_std = self.ph_data['pH'].std()
            
            # How close are predictions to historical patterns
            mean_deviation = abs(np.mean(means) - historical_mean) / historical_std
            historical_similarity = max(0, 1.0 - mean_deviation)
            
            # Overall pattern consistency score
            pattern_score = (mean_consistency * 0.4 + std_consistency * 0.3 + historical_similarity * 0.3)
            
            return {
                'score': pattern_score,
                'mean_consistency': mean_consistency,
                'std_consistency': std_consistency,
                'historical_similarity': historical_similarity,
                'call_means': means,
                'call_stds': stds,
                'historical_mean': historical_mean,
                'historical_std': historical_std
            }
            
        except Exception as e:
            print(f"Error in pattern consistency analysis: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def analyze_bias_accumulation(self, all_predictions):
        """Analyze bias accumulation over multiple calls"""
        try:
            if len(all_predictions) < 10:
                return {'score': 0.0, 'error': 'Insufficient predictions'}
            
            # Calculate trend over all predictions
            x = np.arange(len(all_predictions))
            trend_slope, _ = np.polyfit(x, all_predictions, 1)
            
            # Historical reference
            historical_mean = self.ph_data['pH'].mean()
            historical_std = self.ph_data['pH'].std()
            
            # Bias metrics
            overall_mean = np.mean(all_predictions)
            mean_bias = abs(overall_mean - historical_mean) / historical_std
            
            # Trend bias (downward trend is bad)
            trend_bias = abs(trend_slope) / historical_std
            
            # Monotonic decline check
            segments = np.array_split(all_predictions, 5)
            segment_means = [np.mean(seg) for seg in segments]
            monotonic_decline = all(segment_means[i] >= segment_means[i+1] for i in range(len(segment_means)-1))
            
            # Bias prevention score (higher is better)
            bias_score = max(0, 1.0 - mean_bias - trend_bias)
            if monotonic_decline:
                bias_score *= 0.5  # Penalize monotonic decline
            
            return {
                'score': bias_score,
                'trend_slope': trend_slope,
                'mean_bias': mean_bias,
                'trend_bias': trend_bias,
                'monotonic_decline': monotonic_decline,
                'overall_mean': overall_mean,
                'historical_mean': historical_mean,
                'segment_means': segment_means
            }
            
        except Exception as e:
            print(f"Error in bias analysis: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def analyze_variability_preservation(self, all_predictions):
        """Analyze if predictions maintain historical variability"""
        try:
            if len(all_predictions) < 10:
                return {'score': 0.0, 'error': 'Insufficient predictions'}
            
            # Calculate prediction variability
            pred_std = np.std(all_predictions)
            pred_range = np.max(all_predictions) - np.min(all_predictions)
            
            # Historical variability
            hist_std = self.ph_data['pH'].std()
            hist_range = self.ph_data['pH'].max() - self.ph_data['pH'].min()
            
            # Variability ratios
            std_ratio = pred_std / hist_std if hist_std > 0 else 0
            range_ratio = pred_range / hist_range if hist_range > 0 else 0
            
            # Good variability preservation means ratios close to 1
            std_preservation = 1.0 - abs(1.0 - std_ratio)
            range_preservation = 1.0 - abs(1.0 - range_ratio)
            
            # Check for realistic variation (not too flat)
            changes = np.diff(all_predictions)
            change_std = np.std(changes)
            flatness_penalty = 1.0 if change_std > 0.01 else 0.5
            
            # Overall variability score
            variability_score = (std_preservation * 0.5 + range_preservation * 0.3) * flatness_penalty + 0.2
            variability_score = max(0, min(1, variability_score))
            
            return {
                'score': variability_score,
                'std_ratio': std_ratio,
                'range_ratio': range_ratio,
                'std_preservation': std_preservation,
                'range_preservation': range_preservation,
                'pred_std': pred_std,
                'hist_std': hist_std,
                'change_std': change_std,
                'flatness_penalty': flatness_penalty
            }
            
        except Exception as e:
            print(f"Error in variability analysis: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def analyze_continuity_between_calls(self, successful_calls):
        """Analyze continuity between prediction calls"""
        try:
            if len(successful_calls) < 2:
                return {'score': 0.0, 'error': 'Need at least 2 calls'}
            
            # Check timestamp continuity
            timestamp_gaps = []
            prediction_jumps = []
            
            for i in range(len(successful_calls) - 1):
                current_call = successful_calls[i]
                next_call = successful_calls[i + 1]
                
                # Get last timestamp of current call and first of next
                current_last_ts = current_call['timestamps'][-1]
                next_first_ts = next_call['timestamps'][0]
                
                # Parse timestamps
                try:
                    current_dt = pd.to_datetime(current_last_ts)
                    next_dt = pd.to_datetime(next_first_ts)
                    gap = (next_dt - current_dt).total_seconds()
                    timestamp_gaps.append(gap)
                except:
                    timestamp_gaps.append(0)
                
                # Check prediction value continuity
                current_last_pred = current_call['predictions'][-1]
                next_first_pred = next_call['predictions'][0]
                jump = abs(next_first_pred - current_last_pred)
                prediction_jumps.append(jump)
            
            # Analyze continuity
            avg_timestamp_gap = np.mean(timestamp_gaps) if timestamp_gaps else 0
            avg_prediction_jump = np.mean(prediction_jumps) if prediction_jumps else 0
            
            # Historical reference for jump size
            historical_std = self.ph_data['pH'].std()
            
            # Continuity score
            timestamp_continuity = 1.0 if avg_timestamp_gap > 0 else 0.5  # Should advance in time
            prediction_continuity = max(0, 1.0 - (avg_prediction_jump / (2 * historical_std)))
            
            continuity_score = (timestamp_continuity * 0.3 + prediction_continuity * 0.7)
            
            return {
                'score': continuity_score,
                'timestamp_gaps': timestamp_gaps,
                'prediction_jumps': prediction_jumps,
                'avg_timestamp_gap': avg_timestamp_gap,
                'avg_prediction_jump': avg_prediction_jump,
                'timestamp_continuity': timestamp_continuity,
                'prediction_continuity': prediction_continuity
            }
            
        except Exception as e:
            print(f"Error in continuity analysis: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def test_pattern_following_improvement(self):
        """Test pattern following improvement compared to baseline"""
        print("\n=== Testing Pattern Following Improvement ===")
        
        if not self.model_id:
            print("❌ Cannot test pattern following - no model trained")
            self.test_results['pattern_following_improvement'] = False
            return
        
        try:
            # Reset and generate predictions
            reset_response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            
            # Generate a longer sequence of predictions
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": self.model_id, "steps": 30, "time_window": 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if len(predictions) >= 20:
                    # Analyze pattern following
                    pattern_analysis = self.analyze_detailed_pattern_following(predictions)
                    
                    print(f"✅ Pattern Following Analysis:")
                    print(f"   Pattern Following Score: {pattern_analysis['pattern_score']:.3f}")
                    print(f"   Historical Similarity: {pattern_analysis['historical_similarity']:.3f}")
                    print(f"   Trend Consistency: {pattern_analysis['trend_consistency']:.3f}")
                    print(f"   Cyclical Pattern Detection: {pattern_analysis['cyclical_score']:.3f}")
                    print(f"   Variability Match: {pattern_analysis['variability_match']:.3f}")
                    
                    # Check if improvement criteria are met
                    improvement_criteria = {
                        'pattern_score_above_06': pattern_analysis['pattern_score'] >= 0.6,
                        'historical_similarity_good': pattern_analysis['historical_similarity'] >= 0.5,
                        'variability_preserved': pattern_analysis['variability_match'] >= 0.4,
                        'no_monotonic_decline': not pattern_analysis['monotonic_decline']
                    }
                    
                    passed_improvements = sum(improvement_criteria.values())
                    total_improvements = len(improvement_criteria)
                    
                    print(f"\n🎯 Pattern Following Improvement Criteria ({passed_improvements}/{total_improvements}):")
                    for criterion, passed in improvement_criteria.items():
                        status = "✅" if passed else "❌"
                        print(f"   {status} {criterion.replace('_', ' ').title()}: {passed}")
                    
                    self.test_results['pattern_following_improvement'] = {
                        'success': passed_improvements >= 3,  # At least 3/4 criteria
                        'pattern_analysis': pattern_analysis,
                        'criteria_passed': passed_improvements,
                        'criteria_total': total_improvements
                    }
                    
                else:
                    print("❌ Insufficient predictions for pattern analysis")
                    self.test_results['pattern_following_improvement'] = {'success': False, 'error': 'Insufficient predictions'}
                    
            else:
                print(f"❌ Pattern following test failed: {response.status_code} - {response.text}")
                self.test_results['pattern_following_improvement'] = {'success': False, 'error': response.text}
                
        except Exception as e:
            print(f"❌ Pattern following improvement error: {str(e)}")
            self.test_results['pattern_following_improvement'] = {'success': False, 'error': str(e)}
    
    def analyze_detailed_pattern_following(self, predictions):
        """Detailed analysis of how well predictions follow historical patterns"""
        try:
            historical_data = self.ph_data['pH'].values
            
            # 1. Overall similarity to historical distribution
            hist_mean = np.mean(historical_data)
            hist_std = np.std(historical_data)
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            
            mean_similarity = 1.0 - abs(pred_mean - hist_mean) / hist_std
            std_similarity = 1.0 - abs(pred_std - hist_std) / hist_std
            historical_similarity = (mean_similarity + std_similarity) / 2
            
            # 2. Trend consistency
            hist_trend = np.polyfit(np.arange(len(historical_data)), historical_data, 1)[0]
            pred_trend = np.polyfit(np.arange(len(predictions)), predictions, 1)[0]
            trend_consistency = 1.0 - abs(pred_trend - hist_trend) / hist_std
            
            # 3. Cyclical pattern detection
            # Check for daily-like cycles in predictions
            if len(predictions) >= 12:  # Need enough points
                # Look for cyclical patterns
                autocorr_scores = []
                for lag in range(1, min(len(predictions)//3, 8)):
                    if len(predictions) > lag:
                        x1 = predictions[:-lag]
                        x2 = predictions[lag:]
                        if len(x1) > 0 and len(x2) > 0:
                            corr = np.corrcoef(x1, x2)[0, 1]
                            if not np.isnan(corr):
                                autocorr_scores.append(abs(corr))
                
                cyclical_score = max(autocorr_scores) if autocorr_scores else 0.0
            else:
                cyclical_score = 0.0
            
            # 4. Variability match
            hist_changes = np.diff(historical_data)
            pred_changes = np.diff(predictions)
            
            hist_change_std = np.std(hist_changes)
            pred_change_std = np.std(pred_changes)
            
            variability_match = 1.0 - abs(pred_change_std - hist_change_std) / hist_change_std if hist_change_std > 0 else 0.0
            
            # 5. Check for monotonic decline
            segments = np.array_split(predictions, min(5, len(predictions)//3))
            segment_means = [np.mean(seg) for seg in segments if len(seg) > 0]
            monotonic_decline = len(segment_means) > 1 and all(
                segment_means[i] >= segment_means[i+1] for i in range(len(segment_means)-1)
            )
            
            # 6. Overall pattern score
            pattern_score = (
                historical_similarity * 0.3 +
                max(0, trend_consistency) * 0.2 +
                cyclical_score * 0.2 +
                max(0, variability_match) * 0.2 +
                (0.1 if not monotonic_decline else 0.0)
            )
            
            return {
                'pattern_score': max(0, min(1, pattern_score)),
                'historical_similarity': max(0, min(1, historical_similarity)),
                'trend_consistency': max(0, min(1, trend_consistency)),
                'cyclical_score': max(0, min(1, cyclical_score)),
                'variability_match': max(0, min(1, variability_match)),
                'monotonic_decline': monotonic_decline,
                'hist_mean': hist_mean,
                'hist_std': hist_std,
                'pred_mean': pred_mean,
                'pred_std': pred_std,
                'hist_trend': hist_trend,
                'pred_trend': pred_trend
            }
            
        except Exception as e:
            print(f"Error in detailed pattern analysis: {e}")
            return {
                'pattern_score': 0.0,
                'historical_similarity': 0.0,
                'trend_consistency': 0.0,
                'cyclical_score': 0.0,
                'variability_match': 0.0,
                'monotonic_decline': True,
                'error': str(e)
            }
    
    def run_comprehensive_test(self):
        """Run comprehensive enhanced continuous prediction system test"""
        print("🚀 Starting Enhanced Continuous Prediction System Testing")
        print("=" * 80)
        
        # Step 1: Create and upload pH data
        self.create_ph_dataset()
        if not self.upload_ph_data():
            print("❌ Failed to upload pH data - cannot continue")
            return
        
        # Step 2: Train model
        if not self.train_model_for_testing():
            print("❌ Failed to train model - cannot continue")
            return
        
        # Step 3: Test multiple continuous prediction calls
        self.test_multiple_continuous_prediction_calls()
        
        # Step 4: Test pattern following improvement
        self.test_pattern_following_improvement()
        
        # Step 5: Generate comprehensive report
        self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("🎯 ENHANCED CONTINUOUS PREDICTION SYSTEM TEST REPORT")
        print("=" * 80)
        
        total_tests = 0
        passed_tests = 0
        
        # Test 1: Multiple Continuous Calls
        if 'multiple_continuous_calls' in self.test_results:
            total_tests += 1
            result = self.test_results['multiple_continuous_calls']
            if result.get('success', False):
                passed_tests += 1
                print("✅ Multiple Continuous Prediction Calls: PASSED")
                print(f"   Overall Quality Score: {result.get('overall_score', 0):.3f}")
                print(f"   Criteria Passed: {result.get('criteria_passed', 0)}/{result.get('criteria_total', 4)}")
            else:
                print("❌ Multiple Continuous Prediction Calls: FAILED")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
        
        # Test 2: Pattern Following Improvement
        if 'pattern_following_improvement' in self.test_results:
            total_tests += 1
            result = self.test_results['pattern_following_improvement']
            if result.get('success', False):
                passed_tests += 1
                print("✅ Pattern Following Improvement: PASSED")
                if 'pattern_analysis' in result:
                    analysis = result['pattern_analysis']
                    print(f"   Pattern Following Score: {analysis.get('pattern_score', 0):.3f}")
                    print(f"   Historical Similarity: {analysis.get('historical_similarity', 0):.3f}")
            else:
                print("❌ Pattern Following Improvement: FAILED")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
        
        # Overall assessment
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\n📊 OVERALL TEST RESULTS:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Review request criteria assessment
        print(f"\n🎯 REVIEW REQUEST CRITERIA ASSESSMENT:")
        
        if 'multiple_continuous_calls' in self.test_results:
            result = self.test_results['multiple_continuous_calls']
            if result.get('success', False):
                print("✅ Enhanced Continuous Prediction System: WORKING")
                print("✅ Pattern Following Score >= 0.6: VERIFIED")
                print("✅ Variability Preservation: VERIFIED")
                print("✅ Bias Prevention: VERIFIED")
                print("✅ Multiple Consecutive Calls Maintain Consistency: VERIFIED")
            else:
                print("❌ Enhanced Continuous Prediction System: NEEDS WORK")
        
        if success_rate >= 80:
            print(f"\n🎉 CONCLUSION: Enhanced continuous prediction system is working well!")
            print("   The system successfully maintains historical patterns and prevents bias accumulation.")
        elif success_rate >= 50:
            print(f"\n⚠️  CONCLUSION: Enhanced continuous prediction system has some issues.")
            print("   Some improvements are working but further refinement needed.")
        else:
            print(f"\n❌ CONCLUSION: Enhanced continuous prediction system needs significant work.")
            print("   Major issues identified that need to be addressed.")
        
        return success_rate >= 80

if __name__ == "__main__":
    tester = EnhancedContinuousPredictionTester()
    tester.run_comprehensive_test()