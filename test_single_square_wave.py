#!/usr/bin/env python3
"""
Test single square wave to debug the core issue
"""

import numpy as np
import requests  
import pandas as pd
import json
from io import StringIO

def test_single_square_wave():
    print("🔧 Testing Single Square Wave for Core Issue Debug")
    print("=" * 50)
    
    # Simple, clear square wave
    square_wave = np.array([5, 7, 7, 7, 7, 3, 3, 3, 3, 7, 7, 7, 7, 3, 3, 3, 3])
    print(f"Square wave data: {square_wave}")
    
    # Create CSV data
    timestamps = pd.date_range('2023-01-01', periods=len(square_wave), freq='h')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': square_wave
    })
    
    csv_string = df.to_csv(index=False)
    
    try:
        # Upload data
        files = {'file': ('test_data.csv', StringIO(csv_string), 'text/csv')}
        upload_response = requests.post('http://localhost:8001/api/upload-data', files=files)
        
        if upload_response.status_code == 200:
            print("✅ Data uploaded successfully")
        else:
            print(f"❌ Upload failed: {upload_response.status_code}")
            return
        
        # Test universal waveform prediction
        print("\n🌊 Testing Universal Waveform Prediction...")
        prediction_response = requests.get(
            'http://localhost:8001/api/generate-universal-waveform-prediction',
            params={'steps': 10, 'learning_mode': 'comprehensive'}
        )
        
        print(f"Response status: {prediction_response.status_code}")
        
        if prediction_response.status_code == 200:
            result = prediction_response.json()
            print("✅ Universal waveform prediction successful!")
            
            # Print key results
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            
            detected_patterns = result.get('waveform_analysis', {}).get('detected_patterns', {})
            print(f"📊 Detected patterns: {list(detected_patterns.keys())}")
            
            # Show confidence for each pattern type
            for pattern_type, pattern_info in detected_patterns.items():
                confidence = pattern_info.get('confidence', 0) if isinstance(pattern_info, dict) else 0
                print(f"  - {pattern_type}: {confidence:.3f} confidence")
            
            predictions = result.get('predictions', [])
            print(f"🔮 Generated {len(predictions)} predictions")
            print(f"📈 Predictions: {predictions}")
            
            # Success!
            print("\n🎉 SUCCESS! The API is working without 500 errors.")
            
        else:
            print(f"❌ Prediction failed: {prediction_response.status_code}")
            print(f"Error response: {prediction_response.text}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_single_square_wave()