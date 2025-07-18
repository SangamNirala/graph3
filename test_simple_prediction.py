#!/usr/bin/env python3
"""
Simple test to identify the issue with the advanced pH prediction
"""

import requests
import json
import numpy as np
import pandas as pd

# Backend URL
BACKEND_URL = "http://localhost:8001"
API_BASE_URL = f"{BACKEND_URL}/api"

def test_simple_prediction():
    """Test if basic prediction works"""
    print("Testing simple prediction...")
    
    # Create simple test data
    data = {
        'time_step': list(range(30)),
        'pH': [7.0 + 0.1 * np.sin(i/3) + 0.05 * np.random.randn() for i in range(30)]
    }
    
    df = pd.DataFrame(data)
    csv_file = '/tmp/simple_test.csv'
    df.to_csv(csv_file, index=False)
    
    # Upload data
    try:
        with open(csv_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/upload-data", files=files)
        
        print(f"Upload response: {response.status_code}")
        if response.status_code == 200:
            print("✅ Upload successful")
        else:
            print(f"❌ Upload failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return
    
    # Test status endpoint first
    try:
        response = requests.get(f"{API_BASE_URL}/prediction-system-status")
        print(f"Status response: {response.status_code}")
        if response.status_code == 200:
            print("✅ Status endpoint works")
            print(f"Status: {response.json()}")
        else:
            print(f"❌ Status failed: {response.text}")
    except Exception as e:
        print(f"❌ Status error: {e}")
    
    # Test simple prediction
    try:
        response = requests.get(f"{API_BASE_URL}/generate-advanced-ph-prediction?steps=5&maintain_patterns=true")
        print(f"Prediction response: {response.status_code}")
        if response.status_code == 200:
            print("✅ Advanced prediction works")
            result = response.json()
            print(f"Predictions: {result.get('predictions', [])}")
        else:
            print(f"❌ Advanced prediction failed: {response.text}")
    except Exception as e:
        print(f"❌ Advanced prediction error: {e}")

if __name__ == "__main__":
    test_simple_prediction()