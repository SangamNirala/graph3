#!/usr/bin/env python3
"""
Specific Quality Metrics Testing for Enhanced Continuous Prediction System
Tests the specific quality metrics mentioned in the review request
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://6292a650-5f0f-439b-bded-80d6a5caef50.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Quality Metrics at: {API_BASE_URL}")

class QualityMetricsTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        
    def create_ph_dataset(self):
        """Create realistic pH dataset for testing"""
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=48, freq='H')
        
        # Create realistic pH pattern
        base_ph = 7.4
        daily_cycle = 0.3 * np.sin(2 * np.pi * np.arange(48) / 24)
        trend = 0.1 * np.sin(2 * np.pi * np.arange(48) / 48)
        noise = np.random.normal(0, 0.05, 48)
        
        ph_values = base_ph + daily_cycle + trend + noise
        ph_values = np.clip(ph_values, 6.8, 7.9)
        
        self.ph_data = pd.DataFrame({
            'timestamp': timestamps,
            'pH': ph_values
        })
        
        return self.ph_data
    
    def upload_and_train(self):
        """Upload data and train model"""
        # Upload data
        csv_content = self.ph_data.to_csv(index=False)
        files = {'file': ('ph_data.csv', csv_content, 'text/csv')}
        
        response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
        if response.status_code != 200:
            return False
            
        self.data_id = response.json().get('data_id')
        
        # Train model
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
            self.model_id = response.json().get('model_id')
            return True
        return False
    
    def test_quality_metrics_requirements(self):
        """Test specific quality metrics from review request"""
        print("\n=== Testing Quality Metrics Requirements ===")
        
        # Reset continuous prediction
        self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
        
        # Make multiple calls to test consistency
        all_results = []
        
        for i in range(10):  # More calls for better statistics
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": self.model_id, "steps": 15, "time_window": 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                all_results.append({
                    'predictions': data.get('predictions', []),
                    'pattern_analysis': data.get('pattern_analysis', {}),
                    'prediction_method': data.get('prediction_method', 'unknown')
                })
            
            time.sleep(0.2)  # Small delay
        
        if len(all_results) < 5:
            print("❌ Insufficient successful calls for quality metrics testing")
            return False
        
        # Extract all predictions
        all_predictions = []
        pattern_scores = []
        variability_scores = []
        bias_scores = []
        overall_scores = []
        
        for result in all_results:
            all_predictions.extend(result['predictions'])
            
            # Extract quality metrics from pattern analysis
            pattern_analysis = result.get('pattern_analysis', {})
            
            # Try to extract scores from different possible locations
            if 'pattern_following_score' in pattern_analysis:
                pattern_scores.append(pattern_analysis['pattern_following_score'])
            elif 'quality_metrics' in pattern_analysis:
                quality_metrics = pattern_analysis['quality_metrics']
                if 'pattern_following_score' in quality_metrics:
                    pattern_scores.append(quality_metrics['pattern_following_score'])
            
            if 'variability_preservation' in pattern_analysis:
                variability_scores.append(pattern_analysis['variability_preservation'])
            elif 'quality_metrics' in pattern_analysis:
                quality_metrics = pattern_analysis['quality_metrics']
                if 'variability_preservation' in quality_metrics:
                    variability_scores.append(quality_metrics['variability_preservation'])
            
            if 'bias_prevention_score' in pattern_analysis:
                bias_scores.append(pattern_analysis['bias_prevention_score'])
            elif 'quality_metrics' in pattern_analysis:
                quality_metrics = pattern_analysis['quality_metrics']
                if 'bias_prevention_score' in quality_metrics:
                    bias_scores.append(quality_metrics['bias_prevention_score'])
        
        # Calculate our own quality metrics
        historical_data = self.ph_data['pH'].values
        
        # 1. Pattern Following Score
        calculated_pattern_score = self.calculate_pattern_following_score(all_predictions, historical_data)
        
        # 2. Variability Preservation Score
        calculated_variability_score = self.calculate_variability_preservation_score(all_predictions, historical_data)
        
        # 3. Bias Prevention Score
        calculated_bias_score = self.calculate_bias_prevention_score(all_predictions, historical_data)
        
        # 4. Overall Quality Score
        calculated_overall_score = (
            calculated_pattern_score * 0.4 +
            calculated_variability_score * 0.3 +
            calculated_bias_score * 0.3
        )
        
        print(f"\n🎯 QUALITY METRICS ASSESSMENT:")
        print(f"   Pattern Following Score: {calculated_pattern_score:.3f} (target: >= 0.6)")
        print(f"   Variability Preservation Score: {calculated_variability_score:.3f} (target: >= 0.7)")
        print(f"   Bias Prevention Score: {calculated_bias_score:.3f} (target: >= 0.7)")
        print(f"   Overall Quality Score: {calculated_overall_score:.3f} (target: >= 0.7)")
        
        # Check if targets are met
        targets_met = {
            'pattern_following': calculated_pattern_score >= 0.6,
            'variability_preservation': calculated_variability_score >= 0.7,
            'bias_prevention': calculated_bias_score >= 0.7,
            'overall_quality': calculated_overall_score >= 0.7
        }
        
        passed_targets = sum(targets_met.values())
        total_targets = len(targets_met)
        
        print(f"\n✅ TARGET ACHIEVEMENT ({passed_targets}/{total_targets}):")
        for target, met in targets_met.items():
            status = "✅" if met else "❌"
            print(f"   {status} {target.replace('_', ' ').title()}: {met}")
        
        # Additional analysis
        print(f"\n📊 ADDITIONAL ANALYSIS:")
        print(f"   Total predictions analyzed: {len(all_predictions)}")
        print(f"   Successful continuous calls: {len(all_results)}")
        print(f"   pH range in predictions: {min(all_predictions):.3f} - {max(all_predictions):.3f}")
        print(f"   Historical pH range: {historical_data.min():.3f} - {historical_data.max():.3f}")
        
        # Test for monotonic decline (should not happen)
        segments = np.array_split(all_predictions, 5)
        segment_means = [np.mean(seg) for seg in segments]
        monotonic_decline = all(segment_means[i] >= segment_means[i+1] for i in range(len(segment_means)-1))
        
        print(f"   Monotonic decline detected: {'❌ YES' if monotonic_decline else '✅ NO'}")
        print(f"   Segment means: {[f'{m:.3f}' for m in segment_means]}")
        
        return passed_targets >= 3  # At least 3/4 targets should be met
    
    def calculate_pattern_following_score(self, predictions, historical_data):
        """Calculate pattern following score"""
        try:
            # Compare statistical properties
            hist_mean = np.mean(historical_data)
            hist_std = np.std(historical_data)
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            
            # Mean similarity
            mean_similarity = 1.0 - abs(pred_mean - hist_mean) / hist_std
            
            # Std similarity
            std_similarity = 1.0 - abs(pred_std - hist_std) / hist_std
            
            # Range similarity
            hist_range = np.max(historical_data) - np.min(historical_data)
            pred_range = np.max(predictions) - np.min(predictions)
            range_similarity = 1.0 - abs(pred_range - hist_range) / hist_range
            
            # Trend similarity
            hist_trend = np.polyfit(np.arange(len(historical_data)), historical_data, 1)[0]
            pred_trend = np.polyfit(np.arange(len(predictions)), predictions, 1)[0]
            trend_similarity = 1.0 - abs(pred_trend - hist_trend) / hist_std
            
            # Combined score
            pattern_score = (
                mean_similarity * 0.3 +
                std_similarity * 0.3 +
                range_similarity * 0.2 +
                max(0, trend_similarity) * 0.2
            )
            
            return max(0, min(1, pattern_score))
            
        except Exception as e:
            print(f"Error calculating pattern following score: {e}")
            return 0.0
    
    def calculate_variability_preservation_score(self, predictions, historical_data):
        """Calculate variability preservation score"""
        try:
            # Standard deviation preservation
            hist_std = np.std(historical_data)
            pred_std = np.std(predictions)
            std_preservation = 1.0 - abs(pred_std - hist_std) / hist_std
            
            # Change variability preservation
            hist_changes = np.diff(historical_data)
            pred_changes = np.diff(predictions)
            
            hist_change_std = np.std(hist_changes)
            pred_change_std = np.std(pred_changes)
            
            change_preservation = 1.0 - abs(pred_change_std - hist_change_std) / hist_change_std
            
            # Variability score
            variability_score = (std_preservation * 0.6 + change_preservation * 0.4)
            
            return max(0, min(1, variability_score))
            
        except Exception as e:
            print(f"Error calculating variability preservation score: {e}")
            return 0.0
    
    def calculate_bias_prevention_score(self, predictions, historical_data):
        """Calculate bias prevention score"""
        try:
            # Check for trend bias
            pred_trend = np.polyfit(np.arange(len(predictions)), predictions, 1)[0]
            hist_std = np.std(historical_data)
            
            # Normalize trend by historical std
            normalized_trend = abs(pred_trend) / hist_std
            trend_bias_score = max(0, 1.0 - normalized_trend)
            
            # Check for mean bias
            hist_mean = np.mean(historical_data)
            pred_mean = np.mean(predictions)
            mean_bias = abs(pred_mean - hist_mean) / hist_std
            mean_bias_score = max(0, 1.0 - mean_bias)
            
            # Check for monotonic behavior
            segments = np.array_split(predictions, min(5, len(predictions)//3))
            segment_means = [np.mean(seg) for seg in segments if len(seg) > 0]
            
            if len(segment_means) > 1:
                monotonic_decline = all(segment_means[i] >= segment_means[i+1] for i in range(len(segment_means)-1))
                monotonic_increase = all(segment_means[i] <= segment_means[i+1] for i in range(len(segment_means)-1))
                monotonic_penalty = 0.5 if (monotonic_decline or monotonic_increase) else 0.0
            else:
                monotonic_penalty = 0.0
            
            # Combined bias prevention score
            bias_score = (trend_bias_score * 0.4 + mean_bias_score * 0.4 + (1.0 - monotonic_penalty) * 0.2)
            
            return max(0, min(1, bias_score))
            
        except Exception as e:
            print(f"Error calculating bias prevention score: {e}")
            return 0.0
    
    def run_quality_metrics_test(self):
        """Run the quality metrics test"""
        print("🎯 Starting Quality Metrics Testing for Enhanced Continuous Prediction System")
        print("=" * 80)
        
        # Create dataset
        print("Creating pH dataset...")
        self.create_ph_dataset()
        
        # Upload and train
        print("Uploading data and training model...")
        if not self.upload_and_train():
            print("❌ Failed to upload data or train model")
            return False
        
        # Test quality metrics
        success = self.test_quality_metrics_requirements()
        
        print("\n" + "=" * 80)
        if success:
            print("🎉 QUALITY METRICS TEST: PASSED")
            print("The enhanced continuous prediction system meets the quality requirements!")
        else:
            print("❌ QUALITY METRICS TEST: FAILED")
            print("The enhanced continuous prediction system needs improvement.")
        
        return success

if __name__ == "__main__":
    tester = QualityMetricsTester()
    tester.run_quality_metrics_test()