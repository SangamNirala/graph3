#!/usr/bin/env python3
"""
Quick test for continuous prediction downward bias
"""

import requests
import json
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from scipy import stats

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://f54db828-52d2-4e14-b664-3ae23427df52.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Continuous Prediction at: {API_BASE_URL}")

def create_test_data():
    """Create simple pH test data"""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='h')
    # Linear upward trend with noise
    values = np.linspace(6.8, 7.4, 50) + np.random.normal(0, 0.1, 50)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'ph_value': values
    })
    return df

def test_continuous_prediction_bias():
    """Test continuous prediction for downward bias"""
    session = requests.Session()
    
    print("\n=== Step 1: Upload Test Data ===")
    df = create_test_data()
    csv_content = df.to_csv(index=False)
    
    files = {'file': ('test_ph.csv', csv_content, 'text/csv')}
    response = session.post(f"{API_BASE_URL}/upload-data", files=files)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        return False
    
    data_id = response.json().get('data_id')
    print(f"✅ Data uploaded: {data_id}")
    
    print("\n=== Step 2: Train ARIMA Model ===")
    training_params = {
        "time_column": "timestamp",
        "target_column": "ph_value",
        "order": [1, 1, 1]
    }
    
    response = session.post(
        f"{API_BASE_URL}/train-model",
        params={"data_id": data_id, "model_type": "arima"},
        json=training_params
    )
    
    if response.status_code != 200:
        print(f"❌ Training failed: {response.status_code}")
        return False
    
    model_id = response.json().get('model_id')
    print(f"✅ Model trained: {model_id}")
    
    print("\n=== Step 3: Test Continuous Prediction Bias ===")
    
    # Reset continuous predictions
    session.post(f"{API_BASE_URL}/reset-continuous-prediction")
    
    all_predictions = []
    num_calls = 10
    
    for i in range(num_calls):
        print(f"   Call {i+1}/{num_calls}...")
        
        response = session.get(
            f"{API_BASE_URL}/generate-continuous-prediction",
            params={
                "model_id": model_id,
                "steps": 5,
                "time_window": 30
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            
            if predictions:
                # Handle different response formats
                if isinstance(predictions, list) and len(predictions) > 0:
                    if isinstance(predictions[0], dict):
                        # Format: [{'value': x, 'timestamp': y}, ...]
                        pred_values = [p['value'] for p in predictions]
                    else:
                        # Format: [x, y, z, ...] (direct values)
                        pred_values = predictions
                else:
                    pred_values = []
                
                if pred_values:
                    all_predictions.extend(pred_values)
                    
                    print(f"      Predictions: {len(pred_values)} values")
                    print(f"      Range: {min(pred_values):.3f} - {max(pred_values):.3f}")
                    print(f"      Mean: {np.mean(pred_values):.3f}")
                else:
                    print(f"      ❌ No prediction values found")
            else:
                print(f"      ❌ No predictions returned")
        else:
            print(f"      ❌ Call failed: {response.status_code}")
        
        time.sleep(0.5)
    
    print("\n=== Step 4: Analyze Bias ===")
    if len(all_predictions) > 5:
        predictions = np.array(all_predictions)
        
        # Calculate trend slope
        x = np.arange(len(predictions))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        
        print(f"   Total predictions: {len(predictions)}")
        print(f"   Mean: {mean_pred:.6f}")
        print(f"   Std: {std_pred:.6f}")
        print(f"   Range: {min_pred:.6f} - {max_pred:.6f}")
        print(f"   Trend slope: {slope:.6f}")
        print(f"   R-squared: {r_value**2:.6f}")
        
        # Bias assessment
        bias_threshold = 0.01
        
        if abs(slope) <= bias_threshold:
            print(f"✅ NO DOWNWARD BIAS DETECTED (slope: {slope:.6f})")
            return True
        else:
            if slope < -bias_threshold:
                print(f"❌ DOWNWARD BIAS DETECTED (slope: {slope:.6f})")
            else:
                print(f"⚠️ UPWARD BIAS DETECTED (slope: {slope:.6f})")
            return False
    else:
        print("❌ Insufficient predictions for analysis")
        return False

if __name__ == "__main__":
    print("🎯 QUICK CONTINUOUS PREDICTION BIAS TEST")
    print("=" * 50)
    
    result = test_continuous_prediction_bias()
    
    print("\n" + "=" * 50)
    if result:
        print("🎉 CONTINUOUS PREDICTION BIAS: RESOLVED!")
    else:
        print("❌ CONTINUOUS PREDICTION BIAS: STILL PRESENT")