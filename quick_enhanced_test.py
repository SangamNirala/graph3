#!/usr/bin/env python3
"""
Quick test for enhanced endpoints after fixes
"""

import requests
import json
import pandas as pd
import io
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://91a5cae0-5aba-4c20-b7d2-24f3a6c5da09.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

def test_enhanced_endpoints():
    session = requests.Session()
    
    # Create and upload pH data
    dates = pd.date_range(start='2024-01-01', periods=49, freq='H')
    ph_values = [7.2 + 0.1 * i + 0.05 * (i % 24) for i in range(49)]
    df = pd.DataFrame({'timestamp': dates, 'pH': ph_values})
    csv_content = df.to_csv(index=False)
    
    print("1. Testing file upload...")
    files = {'file': ('ph_data.csv', csv_content, 'text/csv')}
    response = session.post(f"{API_BASE_URL}/upload-data", files=files)
    
    if response.status_code == 200:
        data_id = response.json().get('data_id')
        print(f"✅ Upload successful: {data_id}")
        
        print("2. Testing LSTM model training...")
        response = session.post(
            f"{API_BASE_URL}/train-model",
            params={"data_id": data_id, "model_type": "lstm"},
            json={"time_column": "timestamp", "target_column": "pH", "seq_len": 8, "pred_len": 3, "epochs": 5}
        )
        
        if response.status_code == 200:
            model_id = response.json().get('model_id')
            print(f"✅ LSTM training successful: {model_id}")
            
            print("3. Testing enhanced pattern analysis...")
            response = session.get(f"{API_BASE_URL}/enhanced-pattern-analysis")
            
            if response.status_code == 200:
                print("✅ Enhanced pattern analysis working!")
                data = response.json()
                print(f"   Pattern analysis keys: {list(data.get('pattern_analysis', {}).keys())}")
            else:
                print(f"❌ Enhanced pattern analysis failed: {response.status_code} - {response.text}")
            
            print("4. Testing enhanced continuous prediction...")
            response = session.get(
                f"{API_BASE_URL}/generate-enhanced-continuous-prediction",
                params={"model_id": model_id, "steps": 10, "time_window": 50}
            )
            
            if response.status_code == 200:
                print("✅ Enhanced continuous prediction working!")
                data = response.json()
                print(f"   Predictions: {len(data.get('predictions', []))}")
                print(f"   Is enhanced: {data.get('is_enhanced', False)}")
            else:
                print(f"❌ Enhanced continuous prediction failed: {response.status_code} - {response.text}")
                
        else:
            print(f"❌ LSTM training failed: {response.status_code} - {response.text}")
    else:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_enhanced_endpoints()