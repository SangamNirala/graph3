#!/usr/bin/env python3
"""
Focused Enhanced Real-Time Prediction System v3 Testing
Quick test of the new enhanced real-time prediction system
"""

import requests
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"🎯 FOCUSED ENHANCED REAL-TIME PREDICTION SYSTEM v3 TESTING")
print(f"Testing at: {API_BASE_URL}")
print("=" * 70)

def test_v3_endpoint_basic():
    """Test basic v3 endpoint functionality"""
    print("\n=== Testing Enhanced v3 Endpoint ===")
    
    try:
        # Test with minimal parameters
        response = requests.get(
            f"{API_BASE_URL}/generate-enhanced-realtime-prediction-v3",
            params={"steps": 5, "time_window": 30, "maintain_patterns": True},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ v3 endpoint responding")
            print(f"   Response structure: {list(data.keys())}")
            
            # Check predictions
            predictions = data.get('predictions', [])
            print(f"   Predictions generated: {len(predictions)}")
            
            # Check system metrics
            system_metrics = data.get('system_metrics', {})
            if system_metrics:
                print(f"   System metrics available: {list(system_metrics.keys())}")
                print(f"   Pattern stability: {system_metrics.get('pattern_stability', 'N/A')}")
                print(f"   Recent accuracy: {system_metrics.get('recent_accuracy', 'N/A')}")
                print(f"   System running: {system_metrics.get('is_running', 'N/A')}")
            
            # Check confidence intervals
            confidence_intervals = data.get('confidence_intervals', [])
            print(f"   Confidence intervals: {len(confidence_intervals)} available")
            
            return True
            
        else:
            print(f"❌ v3 endpoint failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}...")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ v3 endpoint timeout (>30s)")
        return False
    except Exception as e:
        print(f"❌ v3 endpoint error: {str(e)}")
        return False

def test_pattern_learning_basic():
    """Test basic pattern learning capabilities"""
    print("\n=== Testing Pattern Learning ===")
    
    try:
        # Create simple pattern data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='H')
        # Simple sine wave pattern
        values = 7.0 + 0.5 * np.sin(2 * np.pi * np.arange(30) / 12) + np.random.normal(0, 0.1, 30)
        df = pd.DataFrame({'timestamp': dates, 'value': values})
        csv_content = df.to_csv(index=False)
        
        # Upload data
        files = {'file': ('pattern_test.csv', csv_content, 'text/csv')}
        response = requests.post(f"{API_BASE_URL}/upload-data", files=files, timeout=20)
        
        if response.status_code == 200:
            data_id = response.json().get('data_id')
            print(f"✅ Pattern data uploaded: {data_id}")
            
            # Train model
            training_params = {
                "time_column": "timestamp",
                "target_column": "value",
                "order": [1, 1, 1]
            }
            
            response = requests.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json=training_params,
                timeout=30
            )
            
            if response.status_code == 200:
                model_id = response.json().get('model_id')
                print(f"✅ Model trained: {model_id}")
                
                # Test pattern learning with v3
                response = requests.get(
                    f"{API_BASE_URL}/generate-enhanced-realtime-prediction-v3",
                    params={"steps": 10, "time_window": 20, "maintain_patterns": True},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    system_metrics = data.get('system_metrics', {})
                    
                    pattern_stability = system_metrics.get('pattern_stability', 0)
                    recent_accuracy = system_metrics.get('recent_accuracy', 0)
                    
                    print(f"✅ Pattern learning test completed")
                    print(f"   Pattern stability: {pattern_stability:.3f}")
                    print(f"   Recent accuracy: {recent_accuracy:.3f}")
                    
                    # Success criteria
                    success = pattern_stability >= 0.3 and recent_accuracy >= 0.4
                    print(f"   {'✅' if success else '⚠️'} Pattern learning: {'WORKING' if success else 'NEEDS IMPROVEMENT'}")
                    
                    return success
                else:
                    print(f"❌ Pattern prediction failed: {response.status_code}")
                    return False
            else:
                print(f"❌ Model training failed: {response.status_code}")
                return False
        else:
            print(f"❌ Data upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Pattern learning test error: {str(e)}")
        return False

def test_fallback_system():
    """Test fallback system"""
    print("\n=== Testing Fallback System ===")
    
    try:
        # Test v2 fallback
        response = requests.get(
            f"{API_BASE_URL}/generate-enhanced-realtime-prediction-v2",
            params={"steps": 5, "time_window": 30, "maintain_patterns": True},
            timeout=20
        )
        
        v2_working = response.status_code == 200
        print(f"{'✅' if v2_working else '❌'} v2 fallback: {'WORKING' if v2_working else 'FAILED'}")
        
        # Test standard continuous prediction fallback
        response = requests.get(
            f"{API_BASE_URL}/generate-continuous-prediction",
            params={"model_type": "fallback", "steps": 5, "time_window": 30},
            timeout=20
        )
        
        v1_working = response.status_code == 200
        print(f"{'✅' if v1_working else '❌'} Standard fallback: {'WORKING' if v1_working else 'FAILED'}")
        
        return v2_working or v1_working
        
    except Exception as e:
        print(f"❌ Fallback test error: {str(e)}")
        return False

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    try:
        # Test multiple consecutive calls for learning
        learning_results = []
        
        for i in range(3):
            response = requests.get(
                f"{API_BASE_URL}/generate-enhanced-realtime-prediction-v3",
                params={"steps": 5, "time_window": 30, "maintain_patterns": True},
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                system_metrics = data.get('system_metrics', {})
                learning_results.append({
                    'accuracy': system_metrics.get('recent_accuracy', 0),
                    'stability': system_metrics.get('pattern_stability', 0),
                    'adaptation_events': system_metrics.get('adaptation_events', 0)
                })
                print(f"   Call {i+1}: accuracy={system_metrics.get('recent_accuracy', 0):.3f}")
            else:
                learning_results.append(None)
            
            time.sleep(1)  # Brief pause between calls
        
        # Analyze learning
        valid_results = [r for r in learning_results if r is not None]
        if len(valid_results) >= 2:
            avg_accuracy = np.mean([r['accuracy'] for r in valid_results])
            avg_stability = np.mean([r['stability'] for r in valid_results])
            
            print(f"✅ Real-time learning test completed")
            print(f"   Average accuracy: {avg_accuracy:.3f}")
            print(f"   Average stability: {avg_stability:.3f}")
            
            success = avg_accuracy >= 0.4 and avg_stability >= 0.3
            print(f"   {'✅' if success else '⚠️'} Real-time learning: {'WORKING' if success else 'NEEDS IMPROVEMENT'}")
            
            return success
        else:
            print("❌ Insufficient results for learning analysis")
            return False
            
    except Exception as e:
        print(f"❌ Advanced features test error: {str(e)}")
        return False

def main():
    """Run focused tests"""
    print("Starting focused Enhanced Real-Time Prediction System v3 tests...")
    
    results = {}
    
    # Run tests
    results['v3_endpoint'] = test_v3_endpoint_basic()
    results['pattern_learning'] = test_pattern_learning_basic()
    results['fallback_system'] = test_fallback_system()
    results['advanced_features'] = test_advanced_features()
    
    # Generate report
    print("\n" + "=" * 70)
    print("🎯 FOCUSED TEST REPORT")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    success_rate = passed / total if total > 0 else 0
    
    print(f"\n📊 RESULTS: {passed}/{total} tests passed ({success_rate:.1%})")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 ASSESSMENT:")
    if success_rate >= 0.75:
        print("   ✅ Enhanced Real-Time Prediction System v3 is WORKING WELL")
        print("   🚀 Advanced pattern learning capabilities are functional")
        print("   📈 System shows good real-time adaptation")
    elif success_rate >= 0.5:
        print("   ⚠️ Enhanced Real-Time Prediction System v3 is PARTIALLY WORKING")
        print("   🔧 Some components need optimization")
        print("   📊 Core functionality is available")
    else:
        print("   ❌ Enhanced Real-Time Prediction System v3 NEEDS IMPROVEMENT")
        print("   🛠️ Multiple components require fixes")
        print("   📋 Review system implementation")
    
    print("=" * 70)

if __name__ == "__main__":
    main()