#!/usr/bin/env python3
"""
Simple Pattern Analysis Test
Tests basic pattern analysis functionality to verify the enhanced algorithms
"""

import requests
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://fb94d99b-4b3b-4c52-8a8d-45c283b3e206.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing pattern analysis at: {API_BASE_URL}")

def create_simple_ph_data():
    """Create simple pH data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=30, freq='h')
    
    # Create pH data with slight trend and some variation
    base_ph = 7.2
    trend = np.linspace(0, 0.2, 30)  # Slight upward trend
    noise = np.random.normal(0, 0.05, 30)  # Small noise
    ph_values = base_ph + trend + noise
    
    # Ensure pH values are in realistic range
    ph_values = np.clip(ph_values, 6.0, 8.0)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'pH': ph_values
    })
    
    return df

def test_basic_pattern_analysis():
    """Test basic pattern analysis functionality"""
    print("\n=== Testing Basic Pattern Analysis ===")
    
    session = requests.Session()
    
    try:
        # Create and upload test data
        df = create_simple_ph_data()
        csv_content = df.to_csv(index=False)
        
        print(f"Created pH data: {len(df)} points, range {df['pH'].min():.3f} - {df['pH'].max():.3f}")
        
        # Upload data
        files = {'file': ('simple_ph_test.csv', csv_content, 'text/csv')}
        response = session.post(f"{API_BASE_URL}/upload-data", files=files)
        
        if response.status_code != 200:
            print(f"❌ Data upload failed: {response.status_code} - {response.text}")
            return False
            
        data_id = response.json().get('data_id')
        print(f"✅ Data uploaded successfully (ID: {data_id})")
        
        # Train ARIMA model
        response = session.post(
            f"{API_BASE_URL}/train-model",
            params={"data_id": data_id, "model_type": "arima"},
            json={"time_column": "timestamp", "target_column": "pH", "order": [1, 1, 1]}
        )
        
        if response.status_code != 200:
            print(f"❌ Model training failed: {response.status_code} - {response.text}")
            return False
            
        model_id = response.json().get('model_id')
        print(f"✅ ARIMA model trained successfully (ID: {model_id})")
        
        # Test basic prediction
        response = session.get(
            f"{API_BASE_URL}/generate-prediction",
            params={"model_id": model_id, "steps": 10}
        )
        
        if response.status_code != 200:
            print(f"❌ Basic prediction failed: {response.status_code} - {response.text}")
            return False
            
        data = response.json()
        predictions = data.get('predictions', [])
        timestamps = data.get('timestamps', [])
        
        print(f"✅ Basic prediction successful: {len(predictions)} predictions generated")
        print(f"   Sample predictions: {predictions[:3]}")
        
        # Test continuous prediction with pattern analysis
        response = session.get(
            f"{API_BASE_URL}/generate-continuous-prediction",
            params={"model_id": model_id, "steps": 15, "time_window": 30}
        )
        
        if response.status_code != 200:
            print(f"❌ Continuous prediction failed: {response.status_code} - {response.text}")
            return False
            
        data = response.json()
        predictions = data.get('predictions', [])
        pattern_analysis = data.get('pattern_analysis', {})
        
        print(f"✅ Continuous prediction successful: {len(predictions)} predictions")
        
        # Check for pattern analysis data
        if pattern_analysis:
            print("✅ Pattern analysis data present:")
            for key, value in pattern_analysis.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.6f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("❌ No pattern analysis data returned")
            
        # Test prediction quality
        if predictions:
            pred_min = min(predictions)
            pred_max = max(predictions)
            pred_mean = np.mean(predictions)
            
            # Check if predictions are in reasonable pH range
            reasonable_range = 6.0 <= pred_min and pred_max <= 8.0
            print(f"✅ Predictions in reasonable pH range: {reasonable_range}")
            print(f"   Prediction range: {pred_min:.3f} - {pred_max:.3f}")
            print(f"   Prediction mean: {pred_mean:.3f}")
            
            # Check for variability (not all same value)
            pred_std = np.std(predictions)
            has_variability = pred_std > 0.001
            print(f"✅ Predictions have variability: {has_variability} (std: {pred_std:.6f})")
            
            return reasonable_range and has_variability
        else:
            print("❌ No predictions generated")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return False

def test_downward_bias_check():
    """Test for downward bias in predictions"""
    print("\n=== Testing Downward Bias Check ===")
    
    session = requests.Session()
    
    try:
        # Create stable pH data
        df = create_simple_ph_data()
        historical_mean = df['pH'].mean()
        csv_content = df.to_csv(index=False)
        
        # Upload and train
        files = {'file': ('bias_test.csv', csv_content, 'text/csv')}
        response = session.post(f"{API_BASE_URL}/upload-data", files=files)
        data_id = response.json().get('data_id')
        
        response = session.post(
            f"{API_BASE_URL}/train-model",
            params={"data_id": data_id, "model_type": "arima"},
            json={"time_column": "timestamp", "target_column": "pH", "order": [1, 1, 1]}
        )
        model_id = response.json().get('model_id')
        
        # Make multiple prediction calls to check for bias accumulation
        all_predictions = []
        
        for i in range(3):
            response = session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 10, "time_window": 30}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                all_predictions.extend(predictions)
        
        if all_predictions:
            pred_mean = np.mean(all_predictions)
            
            # Calculate trend in predictions
            prediction_trend = np.polyfit(range(len(all_predictions)), all_predictions, 1)[0]
            
            print(f"Historical mean: {historical_mean:.3f}")
            print(f"Prediction mean: {pred_mean:.3f}")
            print(f"Prediction trend slope: {prediction_trend:.6f}")
            
            # Check for downward bias
            no_severe_downward_bias = prediction_trend > -0.01  # Allow small negative trend
            mean_reasonable = abs(pred_mean - historical_mean) <= 0.5  # Within reasonable range
            
            print(f"✅ No severe downward bias: {no_severe_downward_bias}")
            print(f"✅ Mean within reasonable range: {mean_reasonable}")
            
            return no_severe_downward_bias and mean_reasonable
        else:
            print("❌ No predictions for bias analysis")
            return False
            
    except Exception as e:
        print(f"❌ Bias test error: {str(e)}")
        return False

def test_pattern_following_basic():
    """Test basic pattern following capability"""
    print("\n=== Testing Basic Pattern Following ===")
    
    session = requests.Session()
    
    try:
        # Create data with clear trend
        dates = pd.date_range(start='2023-01-01', periods=25, freq='h')
        trend = np.linspace(7.0, 7.5, 25)  # Clear upward trend
        noise = np.random.normal(0, 0.02, 25)  # Small noise
        ph_values = trend + noise
        
        df = pd.DataFrame({
            'timestamp': dates,
            'pH': ph_values
        })
        
        historical_trend = np.polyfit(range(len(df)), df['pH'].values, 1)[0]
        print(f"Historical trend slope: {historical_trend:.6f}")
        
        csv_content = df.to_csv(index=False)
        
        # Upload and train
        files = {'file': ('trend_test.csv', csv_content, 'text/csv')}
        response = session.post(f"{API_BASE_URL}/upload-data", files=files)
        data_id = response.json().get('data_id')
        
        response = session.post(
            f"{API_BASE_URL}/train-model",
            params={"data_id": data_id, "model_type": "arima"},
            json={"time_column": "timestamp", "target_column": "pH", "order": [1, 1, 1]}
        )
        model_id = response.json().get('model_id')
        
        # Generate predictions
        response = session.get(
            f"{API_BASE_URL}/generate-continuous-prediction",
            params={"model_id": model_id, "steps": 15, "time_window": 25}
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            
            if predictions and len(predictions) >= 5:
                # Check if predictions follow the trend direction
                prediction_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
                
                print(f"Prediction trend slope: {prediction_trend:.6f}")
                
                # Trend direction should be similar (both positive or both negative)
                trend_direction_maintained = (historical_trend * prediction_trend >= 0)
                
                # Predictions should show some progression (not flat)
                pred_range = max(predictions) - min(predictions)
                has_progression = pred_range > 0.01
                
                print(f"✅ Trend direction maintained: {trend_direction_maintained}")
                print(f"✅ Has progression: {has_progression} (range: {pred_range:.4f})")
                
                return trend_direction_maintained and has_progression
            else:
                print("❌ Insufficient predictions for trend analysis")
                return False
        else:
            print(f"❌ Pattern following test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Pattern following test error: {str(e)}")
        return False

def main():
    """Run all simple pattern tests"""
    print("🎯 SIMPLE PATTERN ANALYSIS TESTING")
    print("=" * 50)
    
    tests = [
        ("Basic Pattern Analysis", test_basic_pattern_analysis),
        ("Downward Bias Check", test_downward_bias_check),
        ("Basic Pattern Following", test_pattern_following_basic)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("🎯 SIMPLE PATTERN TEST RESULTS")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    success_rate = (passed / len(tests)) * 100
    print(f"\n🎯 SUCCESS RATE: {passed}/{len(tests)} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT: Pattern analysis is working well!")
    elif success_rate >= 60:
        print("✅ GOOD: Basic pattern functionality is working")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Pattern analysis needs attention")
    
    return results

if __name__ == "__main__":
    main()