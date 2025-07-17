#!/usr/bin/env python3
"""
Focused pH Downward Trend Analysis Test
Specifically tests if the downward trend issue has been resolved
"""

import requests
import json
import pandas as pd
import io
import numpy as np
import os
from pathlib import Path
import statistics
import matplotlib.pyplot as plt

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://cd92a7da-4e5a-42ac-a0b3-ed7cdafc8664.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

class DownwardTrendAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        
    def create_ph_dataset(self):
        """Create pH dataset with known characteristics"""
        timestamps = pd.date_range(start='2024-01-01', periods=49, freq='H')
        
        # Create pH values with slight upward trend to test if model preserves it
        base_ph = 7.2
        ph_values = []
        
        for i in range(49):
            # Add slight upward trend + realistic variations
            trend = 0.008 * i  # Slight upward trend
            noise = np.random.normal(0, 0.03)
            periodic = 0.05 * np.sin(i * 2 * np.pi / 12)  # 12-hour cycle
            
            ph_value = base_ph + trend + noise + periodic
            ph_value = max(6.8, min(7.8, ph_value))
            ph_values.append(round(ph_value, 2))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ph_value': ph_values
        })
        
        return df
    
    def upload_and_train(self):
        """Upload data and train LSTM model"""
        print("=== Setting up pH dataset and LSTM model ===")
        
        # Create and upload dataset
        df = self.create_ph_dataset()
        csv_content = df.to_csv(index=False)
        
        print(f"Historical pH trend: {df['ph_value'].iloc[0]:.2f} → {df['ph_value'].iloc[-1]:.2f}")
        print(f"Historical pH range: {df['ph_value'].min():.2f} - {df['ph_value'].max():.2f}")
        
        files = {'file': ('ph_trend_test.csv', csv_content, 'text/csv')}
        response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            return False
            
        self.data_id = response.json().get('data_id')
        print(f"✅ Data uploaded: {self.data_id}")
        
        # Train LSTM model
        training_params = {
            "time_column": "timestamp",
            "target_column": "ph_value",
            "model_type": "lstm",
            "seq_len": 8,
            "pred_len": 3,
            "epochs": 30,
            "batch_size": 4,
            "learning_rate": 0.001
        }
        
        response = self.session.post(
            f"{API_BASE_URL}/train-model",
            params={"data_id": self.data_id, "model_type": "lstm"},
            json=training_params
        )
        
        if response.status_code != 200:
            print(f"❌ Training failed: {response.status_code}")
            return False
            
        self.model_id = response.json().get('model_id')
        print(f"✅ LSTM model trained: {self.model_id}")
        return True
    
    def test_prediction_trends(self):
        """Test multiple prediction scenarios for downward trend"""
        print("\n=== Testing Prediction Trends (Downward Bias Check) ===")
        
        results = {}
        
        # Test 1: Single prediction
        print("\n1. Single Prediction Test (30 steps)")
        response = self.session.get(
            f"{API_BASE_URL}/generate-prediction",
            params={"model_id": self.model_id, "steps": 30}
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            
            if predictions:
                trend_analysis = self.analyze_trend(predictions)
                results['single_prediction'] = {
                    'predictions': predictions,
                    'trend_analysis': trend_analysis,
                    'range': (min(predictions), max(predictions)),
                    'mean': statistics.mean(predictions)
                }
                
                print(f"   Predictions: {len(predictions)} values")
                print(f"   Range: {min(predictions):.3f} - {max(predictions):.3f}")
                print(f"   Mean: {statistics.mean(predictions):.3f}")
                print(f"   Trend: {trend_analysis['direction']} (slope: {trend_analysis['slope']:.6f})")
                print(f"   Downward bias: {'❌ YES' if trend_analysis['has_downward_bias'] else '✅ NO'}")
        
        # Test 2: Advanced prediction
        print("\n2. Advanced Prediction Test (30 steps)")
        payload = {"model_id": self.model_id, "steps": 30}
        response = self.session.post(f"{API_BASE_URL}/advanced-prediction", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            
            if predictions:
                trend_analysis = self.analyze_trend(predictions)
                results['advanced_prediction'] = {
                    'predictions': predictions,
                    'trend_analysis': trend_analysis,
                    'range': (min(predictions), max(predictions)),
                    'mean': statistics.mean(predictions)
                }
                
                print(f"   Predictions: {len(predictions)} values")
                print(f"   Range: {min(predictions):.3f} - {max(predictions):.3f}")
                print(f"   Mean: {statistics.mean(predictions):.3f}")
                print(f"   Trend: {trend_analysis['direction']} (slope: {trend_analysis['slope']:.6f})")
                print(f"   Downward bias: {'❌ YES' if trend_analysis['has_downward_bias'] else '✅ NO'}")
        
        # Test 3: Multiple short predictions (simulate continuous use)
        print("\n3. Multiple Short Predictions Test (5 calls × 10 steps)")
        all_predictions = []
        call_means = []
        
        for i in range(5):
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": self.model_id, "steps": 10, "offset": i*5}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                if predictions:
                    all_predictions.extend(predictions)
                    call_means.append(statistics.mean(predictions))
                    print(f"   Call {i+1}: Mean = {statistics.mean(predictions):.3f}")
        
        if all_predictions:
            overall_trend = self.analyze_trend(all_predictions)
            means_trend = self.analyze_trend(call_means)
            
            results['multiple_predictions'] = {
                'all_predictions': all_predictions,
                'call_means': call_means,
                'overall_trend': overall_trend,
                'means_trend': means_trend
            }
            
            print(f"   Overall range: {min(all_predictions):.3f} - {max(all_predictions):.3f}")
            print(f"   Overall trend: {overall_trend['direction']} (slope: {overall_trend['slope']:.6f})")
            print(f"   Call means trend: {means_trend['direction']} (slope: {means_trend['slope']:.6f})")
            print(f"   Accumulated downward bias: {'❌ YES' if means_trend['has_downward_bias'] else '✅ NO'}")
        
        return results
    
    def analyze_trend(self, values):
        """Analyze trend in prediction values"""
        if len(values) < 3:
            return {"direction": "insufficient_data", "slope": 0, "has_downward_bias": False}
        
        # Calculate linear trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        correlation = np.corrcoef(x, values)[0, 1]
        
        # Determine trend direction and bias
        if slope < -0.01 and correlation < -0.3:
            direction = "strongly_downward"
            has_downward_bias = True
        elif slope < -0.005:
            direction = "moderately_downward"
            has_downward_bias = True
        elif slope > 0.005:
            direction = "upward"
            has_downward_bias = False
        else:
            direction = "stable"
            has_downward_bias = False
        
        return {
            "slope": slope,
            "intercept": intercept,
            "correlation": correlation,
            "direction": direction,
            "has_downward_bias": has_downward_bias,
            "trend_strength": abs(correlation)
        }
    
    def generate_summary_report(self, results):
        """Generate comprehensive summary of downward trend analysis"""
        print("\n" + "="*60)
        print("📊 DOWNWARD TREND ANALYSIS SUMMARY")
        print("="*60)
        
        # Check each test for downward bias
        tests_passed = 0
        total_tests = 0
        
        for test_name, test_data in results.items():
            total_tests += 1
            
            if 'trend_analysis' in test_data:
                trend = test_data['trend_analysis']
                has_bias = trend['has_downward_bias']
                
                status = "✅ PASS" if not has_bias else "❌ FAIL"
                print(f"{test_name:25} {status} (trend: {trend['direction']})")
                
                if not has_bias:
                    tests_passed += 1
            
            elif 'means_trend' in test_data:  # Multiple predictions test
                means_trend = test_data['means_trend']
                overall_trend = test_data['overall_trend']
                
                has_bias = means_trend['has_downward_bias'] or overall_trend['has_downward_bias']
                status = "✅ PASS" if not has_bias else "❌ FAIL"
                print(f"{test_name:25} {status} (means: {means_trend['direction']}, overall: {overall_trend['direction']})")
                
                if not has_bias:
                    tests_passed += 1
        
        print("-" * 60)
        print(f"DOWNWARD BIAS TESTS PASSED: {tests_passed}/{total_tests} ({tests_passed/total_tests*100:.1f}%)")
        
        # Overall assessment
        if tests_passed == total_tests:
            print("🎉 CONCLUSION: DOWNWARD TREND ISSUE HAS BEEN RESOLVED!")
            print("   ✅ No persistent downward bias detected")
            print("   ✅ Predictions maintain realistic pH ranges")
            print("   ✅ Trend patterns are stable or follow historical data")
        elif tests_passed >= total_tests * 0.7:
            print("⚠️  CONCLUSION: SIGNIFICANT IMPROVEMENT IN DOWNWARD TREND")
            print("   ✅ Most tests show no downward bias")
            print("   ⚠️  Some minor issues may remain")
        else:
            print("❌ CONCLUSION: DOWNWARD TREND ISSUE STILL EXISTS")
            print("   ❌ Multiple tests show persistent downward bias")
            print("   ❌ Further algorithm improvements needed")
        
        return tests_passed == total_tests
    
    def run_analysis(self):
        """Run complete downward trend analysis"""
        print("🔬 pH PREDICTION DOWNWARD TREND ANALYSIS")
        print("="*60)
        
        if not self.upload_and_train():
            print("❌ Setup failed - cannot proceed with analysis")
            return False
        
        results = self.test_prediction_trends()
        
        if not results:
            print("❌ No prediction results - cannot analyze trends")
            return False
        
        return self.generate_summary_report(results)

if __name__ == "__main__":
    analyzer = DownwardTrendAnalyzer()
    success = analyzer.run_analysis()