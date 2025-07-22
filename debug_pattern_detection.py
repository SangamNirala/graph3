#!/usr/bin/env python3
"""
Debug script to test pattern detection and prediction generation
"""

import numpy as np
import requests  
import pandas as pd
import json
from io import StringIO

# Create test data with different patterns
def create_square_wave(length=50, period=10, amplitude=2, offset=5):
    """Create a square wave pattern"""
    x = np.arange(length)
    square = amplitude * np.sign(np.sin(2 * np.pi * x / period)) + offset
    return square

def create_triangular_wave(length=50, period=10, amplitude=2, offset=5):
    """Create a triangular wave pattern"""
    x = np.arange(length)
    triangle = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * x / period)) + offset
    return triangle

def create_sine_wave(length=50, period=10, amplitude=2, offset=5):
    """Create a sine wave pattern"""
    x = np.arange(length)
    sine = amplitude * np.sin(2 * np.pi * x / period) + offset
    return sine

def upload_data_and_test_prediction(data, pattern_name):
    """Upload data and test prediction"""
    print(f"\n{'='*50}")
    print(f"Testing {pattern_name}")
    print(f"{'='*50}")
    
    # Create CSV data
    timestamps = pd.date_range('2023-01-01', periods=len(data), freq='H')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': data
    })
    
    csv_string = df.to_csv(index=False)
    
    try:
        # Upload data
        files = {'file': ('test_data.csv', StringIO(csv_string), 'text/csv')}
        upload_response = requests.post('http://localhost:8001/api/upload-data', files=files)
        
        if upload_response.status_code == 200:
            print("✅ Data uploaded successfully")
            print(f"Upload response: {upload_response.json()}")
        else:
            print(f"❌ Upload failed: {upload_response.status_code} - {upload_response.text}")
            return
        
        # Test universal waveform prediction
        print("\n🌊 Testing Universal Waveform Prediction...")
        prediction_response = requests.get(
            'http://localhost:8001/api/generate-universal-waveform-prediction',
            params={'steps': 20, 'learning_mode': 'comprehensive'}
        )
        
        if prediction_response.status_code == 200:
            result = prediction_response.json()
            print("✅ Universal waveform prediction successful")
            
            # Analyze the results
            detected_patterns = result.get('waveform_analysis', {}).get('detected_patterns', {})
            print(f"📊 Detected patterns: {detected_patterns}")
            
            pattern_complexity = result.get('waveform_analysis', {}).get('pattern_complexity', 0)
            print(f"📈 Pattern complexity: {pattern_complexity}")
            
            predictions = result.get('predictions', [])
            print(f"🔮 Generated {len(predictions)} predictions")
            print(f"📉 First 5 predictions: {predictions[:5]}")
            print(f"📉 Last 5 predictions: {predictions[-5:]}")
            
            # Check if predictions match the pattern
            if len(predictions) >= 10:
                # Simple pattern analysis
                pred_array = np.array(predictions[:10])
                
                # Check if predictions are constant (square wave characteristic)
                if np.std(pred_array) < 0.1:
                    print("🟦 Predictions appear constant (possible square wave detection)")
                elif np.all(np.diff(pred_array) > 0) or np.all(np.diff(pred_array) < 0):
                    print("📈 Predictions are monotonic (possible triangular/sawtooth detection)")
                elif len(set(pred_array.round(2))) > 5:
                    print("🌊 Predictions show variety (possible sine/complex pattern detection)")
                else:
                    print("❓ Predictions pattern unclear")
            
        else:
            print(f"❌ Universal waveform prediction failed: {prediction_response.status_code}")
            print(f"Error: {prediction_response.text}")
        
    except Exception as e:
        print(f"❌ Error testing {pattern_name}: {e}")

def main():
    print("🧪 Debug Pattern Detection and Prediction Generation")
    print("=" * 60)
    
    # Test different patterns
    test_patterns = [
        (create_square_wave(), "Square Wave"),
        (create_triangular_wave(), "Triangular Wave"), 
        (create_sine_wave(), "Sine Wave")
    ]
    
    for data, name in test_patterns:
        upload_data_and_test_prediction(data, name)
    
    print(f"\n{'='*60}")
    print("🏁 Testing completed!")

if __name__ == "__main__":
    main()