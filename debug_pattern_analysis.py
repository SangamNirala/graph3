#!/usr/bin/env python3
"""
Debug Pattern Analysis System
"""

import requests
import json
import pandas as pd
import numpy as np
import io
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://16359d47-48b7-46cc-a21d-6ad29245d1fd.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Debugging pattern analysis at: {API_BASE_URL}")

def create_clear_u_shaped_data(points=100):
    """Create very clear U-shaped pattern data"""
    x = np.linspace(-3, 3, points)
    y = x**2 + 6.5 + np.random.normal(0, 0.05, points)  # Clear U-shape, pH range 6.0-8.0
    
    dates = pd.date_range(start='2023-01-01', periods=points, freq='D')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'ph': y
    })
    
    return df

def debug_pattern_analysis():
    """Debug the pattern analysis system"""
    session = requests.Session()
    
    # Create very clear U-shaped data
    data = create_clear_u_shaped_data(100)
    print(f"Created U-shaped data with {len(data)} points")
    print(f"pH range: {data['ph'].min():.2f} - {data['ph'].max():.2f}")
    print(f"Data shape: {data.shape}")
    
    # Upload data
    csv_content = data.to_csv(index=False)
    files = {'file': ('debug_u_shape.csv', csv_content, 'text/csv')}
    
    response = session.post(f"{API_BASE_URL}/upload-data", files=files)
    
    if response.status_code != 200:
        print(f"❌ Failed to upload data: {response.status_code}")
        print(response.text)
        return
    
    upload_result = response.json()
    data_id = upload_result.get('data_id')
    print(f"✅ Data uploaded successfully. Data ID: {data_id}")
    
    # Train ARIMA model
    training_params = {
        "time_column": "timestamp",
        "target_column": "ph",
        "order": [1, 1, 1]
    }
    
    response = session.post(
        f"{API_BASE_URL}/train-model",
        params={"data_id": data_id, "model_type": "arima"},
        json=training_params
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to train model: {response.status_code}")
        print(response.text)
        return
    
    train_result = response.json()
    model_id = train_result.get('model_id')
    print(f"✅ Model trained successfully. Model ID: {model_id}")
    
    # Test continuous prediction with detailed output
    response = session.get(
        f"{API_BASE_URL}/generate-continuous-prediction",
        params={"model_id": model_id, "steps": 20, "time_window": 100}
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to generate predictions: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    
    print("\n=== PREDICTION RESULT ANALYSIS ===")
    print(f"Response keys: {list(result.keys())}")
    
    predictions = result.get('predictions', [])
    print(f"Number of predictions: {len(predictions)}")
    if predictions:
        print(f"Prediction range: {min(predictions):.3f} - {max(predictions):.3f}")
        print(f"First 5 predictions: {predictions[:5]}")
    
    pattern_analysis = result.get('pattern_analysis', {})
    print(f"\nPattern analysis keys: {list(pattern_analysis.keys())}")
    
    for key, value in pattern_analysis.items():
        print(f"  {key}: {value}")
    
    system_metrics = result.get('system_metrics', {})
    print(f"\nSystem metrics keys: {list(system_metrics.keys())}")
    
    for key, value in system_metrics.items():
        print(f"  {key}: {value}")
    
    prediction_method = result.get('prediction_method', 'unknown')
    print(f"\nPrediction method: {prediction_method}")
    
    # Check if we're getting fallback predictions
    if pattern_analysis.get('primary_pattern') == 'fallback':
        print("\n⚠️  WARNING: System is using fallback predictions!")
        print("This indicates the pattern analysis system is not working properly.")
    elif pattern_analysis.get('primary_pattern') == 'unknown':
        print("\n⚠️  WARNING: Pattern detected as 'unknown'!")
        print("This indicates pattern detection needs improvement.")
    else:
        print(f"\n✅ Pattern detected: {pattern_analysis.get('primary_pattern')}")
    
    # Test multiple calls to check continuity
    print("\n=== TESTING CONTINUITY ===")
    for i in range(3):
        response = session.get(
            f"{API_BASE_URL}/generate-continuous-prediction",
            params={"model_id": model_id, "steps": 10, "time_window": 50}
        )
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            timestamps = result.get('timestamps', [])
            
            print(f"Call {i+1}: {len(predictions)} predictions, first timestamp: {timestamps[0] if timestamps else 'None'}")
            if predictions:
                print(f"  Prediction range: {min(predictions):.3f} - {max(predictions):.3f}")
        else:
            print(f"Call {i+1}: Failed with status {response.status_code}")

if __name__ == "__main__":
    debug_pattern_analysis()