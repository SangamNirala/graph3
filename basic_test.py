#!/usr/bin/env python3
"""
Simple test for basic prediction functionality
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://4f76f575-d5bb-4a63-b0d6-32438c43963e.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Basic Prediction at: {API_BASE_URL}")

def create_test_data():
    """Create simple pH test data"""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='h')
    # Linear upward trend with noise
    values = np.linspace(6.8, 7.4, 30) + np.random.normal(0, 0.05, 30)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'ph_value': values
    })
    return df

def test_basic_prediction():
    """Test basic prediction functionality"""
    session = requests.Session()
    
    print("\n=== Step 1: Upload Test Data ===")
    df = create_test_data()
    csv_content = df.to_csv(index=False)
    
    files = {'file': ('test_ph.csv', csv_content, 'text/csv')}
    response = session.post(f"{API_BASE_URL}/upload-data", files=files)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")
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
        print(f"❌ Training failed: {response.status_code} - {response.text}")
        return False
    
    model_id = response.json().get('model_id')
    print(f"✅ Model trained: {model_id}")
    
    print("\n=== Step 3: Test Basic Prediction ===")
    
    response = session.get(
        f"{API_BASE_URL}/generate-prediction",
        params={"model_id": model_id, "steps": 10}
    )
    
    if response.status_code == 200:
        data = response.json()
        predictions = data.get('predictions', [])
        
        if predictions:
            print(f"✅ Basic prediction successful: {len(predictions)} predictions")
            
            # Handle different response formats
            if isinstance(predictions, list) and len(predictions) > 0:
                if isinstance(predictions[0], dict):
                    pred_values = [p['value'] for p in predictions]
                else:
                    pred_values = predictions
                
                print(f"   Range: {min(pred_values):.3f} - {max(pred_values):.3f}")
                print(f"   Mean: {np.mean(pred_values):.3f}")
                return True
            else:
                print("❌ No prediction values found")
                return False
        else:
            print("❌ No predictions returned")
            return False
    else:
        print(f"❌ Basic prediction failed: {response.status_code} - {response.text}")
        return False

def test_ph_simulation():
    """Test pH simulation endpoints"""
    session = requests.Session()
    
    print("\n=== Step 4: Test pH Simulation ===")
    
    response = session.get(f"{API_BASE_URL}/ph-simulation")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ pH simulation working: pH={data.get('ph_value', 'N/A'):.3f}")
        return True
    else:
        print(f"❌ pH simulation failed: {response.status_code}")
        return False

def test_pattern_analysis():
    """Test pattern analysis endpoints"""
    session = requests.Session()
    
    print("\n=== Step 5: Test Pattern Analysis ===")
    
    response = session.get(f"{API_BASE_URL}/advanced-pattern-analysis")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Pattern analysis working")
        return True
    else:
        print(f"❌ Pattern analysis failed: {response.status_code}")
        return False

if __name__ == "__main__":
    print("🎯 BASIC PREDICTION FUNCTIONALITY TEST")
    print("=" * 50)
    
    results = []
    
    # Test basic prediction
    results.append(test_basic_prediction())
    
    # Test pH simulation
    results.append(test_ph_simulation())
    
    # Test pattern analysis
    results.append(test_pattern_analysis())
    
    print("\n" + "=" * 50)
    success_count = sum(results)
    total_tests = len(results)
    
    print(f"SUCCESS RATE: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    
    if success_count == total_tests:
        print("🎉 ALL BASIC FUNCTIONALITY WORKING!")
    elif success_count >= total_tests * 0.5:
        print("⚠️ PARTIAL FUNCTIONALITY WORKING")
    else:
        print("❌ MAJOR ISSUES DETECTED")