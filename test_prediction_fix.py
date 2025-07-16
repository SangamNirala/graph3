#!/usr/bin/env python3
"""
Focused test for prediction generation after fix
"""

import requests
import json
import pandas as pd
import io
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / 'frontend' / '.env')

BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://665fa10b-70b2-4d36-a661-3f7b1bc0b244.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

def create_sample_data():
    """Create realistic time-series sample data for testing"""
    import numpy as np
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    trend = np.linspace(1000, 1200, 50)
    seasonal = 100 * np.sin(2 * np.pi * np.arange(50) / 7)
    noise = np.random.normal(0, 30, 50)
    sales = trend + seasonal + noise
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales
    })
    return df

def test_prediction_fix():
    session = requests.Session()
    
    print("=== Testing Prediction Generation Fix ===")
    
    # 1. Upload data
    df = create_sample_data()
    csv_content = df.to_csv(index=False)
    files = {'file': ('sales_data.csv', csv_content, 'text/csv')}
    
    response = session.post(f"{API_BASE_URL}/upload-data", files=files)
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return False
    
    data_id = response.json()['data_id']
    print(f"✅ Data uploaded: {data_id}")
    
    # 2. Train Prophet model
    response = session.post(
        f"{API_BASE_URL}/train-model",
        params={"data_id": data_id, "model_type": "prophet"},
        json={"time_column": "date", "target_column": "sales"}
    )
    if response.status_code != 200:
        print(f"❌ Prophet training failed: {response.text}")
        return False
    
    prophet_model_id = response.json()['model_id']
    print(f"✅ Prophet model trained: {prophet_model_id}")
    
    # 3. Test Prophet prediction
    response = session.get(
        f"{API_BASE_URL}/generate-prediction",
        params={"model_id": prophet_model_id, "steps": 5}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Prophet prediction successful: {len(data['predictions'])} predictions")
        print(f"   Sample predictions: {data['predictions'][:3]}")
    else:
        print(f"❌ Prophet prediction failed: {response.text}")
    
    # 4. Train ARIMA model
    response = session.post(
        f"{API_BASE_URL}/train-model",
        params={"data_id": data_id, "model_type": "arima"},
        json={"time_column": "date", "target_column": "sales", "order": [1, 1, 1]}
    )
    if response.status_code != 200:
        print(f"❌ ARIMA training failed: {response.text}")
        return False
    
    arima_model_id = response.json()['model_id']
    print(f"✅ ARIMA model trained: {arima_model_id}")
    
    # 5. Test ARIMA prediction (this was failing before)
    response = session.get(
        f"{API_BASE_URL}/generate-prediction",
        params={"model_id": arima_model_id, "steps": 5}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✅ ARIMA prediction successful: {len(data['predictions'])} predictions")
        print(f"   Sample predictions: {data['predictions'][:3]}")
        return True
    else:
        print(f"❌ ARIMA prediction failed: {response.text}")
        return False

if __name__ == "__main__":
    success = test_prediction_fix()
    if success:
        print("🎉 Prediction generation fix successful!")
    else:
        print("⚠️ Prediction generation still has issues")