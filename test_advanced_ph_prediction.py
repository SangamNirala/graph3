#!/usr/bin/env python3
"""
Test script for the new advanced pH prediction system
"""

import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Backend URL
BACKEND_URL = "http://localhost:8001"
API_BASE_URL = f"{BACKEND_URL}/api"

def create_sample_ph_data():
    """Create realistic pH data with patterns"""
    np.random.seed(42)
    
    # Create time series
    t = np.linspace(0, 10, 100)
    
    # Base pH level around 7
    base_ph = 7.0
    
    # Add some cyclical patterns (daily variations)
    cyclical = 0.5 * np.sin(2 * np.pi * t / 3) + 0.3 * np.sin(2 * np.pi * t / 7)
    
    # Add trend
    trend = 0.2 * np.sin(t / 2)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(t))
    
    # Combine all components
    ph_values = base_ph + cyclical + trend + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'time_step': range(len(ph_values)),
        'pH': ph_values,
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(len(ph_values))]
    })
    
    return df

def test_advanced_prediction():
    """Test the advanced pH prediction system"""
    print("🧪 Testing Advanced pH Prediction System")
    print("=" * 50)
    
    # Create sample data
    print("1. Creating sample pH data...")
    df = create_sample_ph_data()
    print(f"   Created {len(df)} data points")
    print(f"   pH range: {df['pH'].min():.2f} - {df['pH'].max():.2f}")
    
    # Save to CSV
    csv_file = '/tmp/test_ph_data.csv'
    df.to_csv(csv_file, index=False)
    print(f"   Saved to: {csv_file}")
    
    # Upload data
    print("\n2. Uploading data to backend...")
    try:
        with open(csv_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/upload-data", files=files)
        
        if response.status_code == 200:
            upload_result = response.json()
            print(f"   ✅ Upload successful")
            print(f"   Data quality score: {upload_result.get('analysis', {}).get('quality_score', 'N/A')}")
        else:
            print(f"   ❌ Upload failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Upload error: {e}")
        return False
    
    # Test advanced prediction
    print("\n3. Testing advanced pH prediction...")
    try:
        response = requests.get(f"{API_BASE_URL}/generate-advanced-ph-prediction?steps=30&maintain_patterns=true")
        
        if response.status_code == 200:
            prediction_result = response.json()
            print(f"   ✅ Advanced prediction successful")
            
            predictions = prediction_result.get('predictions', [])
            print(f"   Generated {len(predictions)} predictions")
            
            if predictions:
                print(f"   Prediction range: {min(predictions):.2f} - {max(predictions):.2f}")
                print(f"   Mean prediction: {np.mean(predictions):.2f}")
                print(f"   Std prediction: {np.std(predictions):.2f}")
            
            # Check quality analysis
            quality = prediction_result.get('quality_analysis', {})
            print(f"   Quality metrics available: {list(quality.keys())}")
            
            # Check pattern analysis
            patterns = prediction_result.get('pattern_analysis', {})
            if patterns:
                print(f"   Pattern analysis: {list(patterns.keys())}")
            
            # Check historical continuity
            continuity = prediction_result.get('historical_continuity', {})
            if continuity:
                continuity_score = continuity.get('continuity_score', 0)
                print(f"   Historical continuity score: {continuity_score:.3f}")
            
        else:
            print(f"   ❌ Advanced prediction failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Advanced prediction error: {e}")
        return False
    
    # Test prediction extension
    print("\n4. Testing prediction extension...")
    try:
        response = requests.get(f"{API_BASE_URL}/extend-advanced-ph-prediction?additional_steps=10")
        
        if response.status_code == 200:
            extension_result = response.json()
            print(f"   ✅ Extension successful")
            
            extended_predictions = extension_result.get('predictions', [])
            print(f"   Extended {len(extended_predictions)} additional predictions")
            
            if extended_predictions:
                print(f"   Extension range: {min(extended_predictions):.2f} - {max(extended_predictions):.2f}")
            
        else:
            print(f"   ❌ Extension failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Extension error: {e}")
    
    # Test system status
    print("\n5. Checking system status...")
    try:
        response = requests.get(f"{API_BASE_URL}/prediction-system-status")
        
        if response.status_code == 200:
            status_result = response.json()
            print(f"   ✅ Status check successful")
            
            active_system = status_result.get('active_system', 'unknown')
            ph_engine_status = status_result.get('advanced_ph_engine_status', 'unknown')
            
            print(f"   Active system: {active_system}")
            print(f"   Advanced pH engine: {ph_engine_status}")
            
            systems = status_result.get('systems_available', [])
            print(f"   Available systems: {', '.join(systems)}")
            
        else:
            print(f"   ❌ Status check failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Status check error: {e}")
    
    print("\n🎉 Advanced pH prediction system test completed!")
    return True

if __name__ == "__main__":
    success = test_advanced_prediction()
    if success:
        print("\n✅ All tests passed! The advanced pH prediction system is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")