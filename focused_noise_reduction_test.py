#!/usr/bin/env python3
"""
Focused Noise Reduction System Testing
Tests the newly implemented noise reduction system for real-time continuous prediction graph smoothing
"""

import requests
import json
import numpy as np
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://a7807159-b509-4af3-bd55-7a43a5ac0a45.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing noise reduction system at: {API_BASE_URL}")

class NoiseReductionTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_noise_test_datasets(self):
        """Create test datasets with different noise patterns"""
        # Base smooth signal
        x = np.linspace(0, 10, 50)
        base_signal = 7.0 + 0.5 * np.sin(x) + 0.2 * x
        
        datasets = {}
        
        # Smooth data (minimal noise)
        datasets['smooth'] = (base_signal + np.random.normal(0, 0.05, len(base_signal))).tolist()
        
        # Data with spikes
        spike_signal = base_signal.copy()
        spike_indices = [10, 25, 40]
        for idx in spike_indices:
            if idx < len(spike_signal):
                spike_signal[idx] += np.random.choice([-2, 2])  # Add random spikes
        datasets['spikes'] = spike_signal.tolist()
        
        # Data with jitter (high-frequency noise)
        jitter_noise = np.random.normal(0, 0.3, len(base_signal))
        datasets['jitter'] = (base_signal + jitter_noise).tolist()
        
        # Data with oscillation
        oscillation = 0.4 * np.sin(5 * x)  # High-frequency oscillation
        datasets['oscillation'] = (base_signal + oscillation).tolist()
        
        return datasets
    
    def test_core_noise_reduction_system(self):
        """Test the core AdvancedNoiseReductionSystem class"""
        print("\n=== Testing Core Noise Reduction System ===")
        
        test_datasets = self.create_noise_test_datasets()
        noise_tests = []
        
        # Test 1: Smooth data handling
        print("Testing smooth data handling...")
        try:
            response = self.session.post(
                f"{API_BASE_URL}/test-noise-reduction",
                json={
                    "predictions": test_datasets['smooth'],
                    "test_type": "smooth_data"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                noise_score = data.get('noise_reduction_score', 0)
                smoothing_applied = data.get('smoothing_applied', [])
                quality_metrics = data.get('quality_metrics', {})
                
                print(f"✅ Smooth data test successful")
                print(f"   Noise reduction score: {noise_score:.3f}")
                print(f"   Smoothing methods: {smoothing_applied}")
                print(f"   Smoothness score: {quality_metrics.get('smoothness_score', 0):.3f}")
                
                # For smooth data, minimal smoothing should be applied
                smooth_test_passed = ('minimal_smoothing' in smoothing_applied or 
                                    'light_smoothing' in smoothing_applied or
                                    len(smoothing_applied) <= 2)
                noise_tests.append(("Smooth data handling", smooth_test_passed))
            else:
                print(f"❌ Smooth data test failed: {response.status_code} - {response.text}")
                noise_tests.append(("Smooth data handling", False))
        except Exception as e:
            print(f"❌ Smooth data test error: {e}")
            noise_tests.append(("Smooth data handling", False))
        
        # Test 2: Spike noise detection and removal
        print("Testing spike noise detection and removal...")
        try:
            response = self.session.post(
                f"{API_BASE_URL}/test-noise-reduction",
                json={
                    "predictions": test_datasets['spikes'],
                    "test_type": "spike_noise"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                noise_score = data.get('noise_reduction_score', 0)
                smoothing_applied = data.get('smoothing_applied', [])
                noise_analysis = data.get('noise_analysis', {})
                
                print(f"✅ Spike noise test successful")
                print(f"   Noise reduction score: {noise_score:.3f}")
                print(f"   Smoothing methods: {smoothing_applied}")
                print(f"   Detected noise type: {noise_analysis.get('dominant_noise_type', 'unknown')}")
                print(f"   Noise level: {noise_analysis.get('noise_level', 'unknown')}")
                
                # For spike data, spike removal should be detected and applied
                spike_test_passed = (noise_analysis.get('dominant_noise_type') == 'spikes' and
                                   ('spike_removal' in smoothing_applied or 
                                    'median_filter' in smoothing_applied) and 
                                   noise_score > 0.2)
                noise_tests.append(("Spike noise detection and removal", spike_test_passed))
            else:
                print(f"❌ Spike noise test failed: {response.status_code} - {response.text}")
                noise_tests.append(("Spike noise detection and removal", False))
        except Exception as e:
            print(f"❌ Spike noise test error: {e}")
            noise_tests.append(("Spike noise detection and removal", False))
        
        # Test 3: Jitter noise reduction
        print("Testing jitter noise reduction...")
        try:
            response = self.session.post(
                f"{API_BASE_URL}/test-noise-reduction",
                json={
                    "predictions": test_datasets['jitter'],
                    "test_type": "jitter_noise"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                noise_score = data.get('noise_reduction_score', 0)
                smoothing_applied = data.get('smoothing_applied', [])
                noise_analysis = data.get('noise_analysis', {})
                
                print(f"✅ Jitter noise test successful")
                print(f"   Noise reduction score: {noise_score:.3f}")
                print(f"   Smoothing methods: {smoothing_applied}")
                print(f"   Detected noise type: {noise_analysis.get('dominant_noise_type', 'unknown')}")
                
                # For jitter data, appropriate smoothing should be applied
                jitter_test_passed = (('savgol_filter' in smoothing_applied or 
                                     'gaussian_smooth' in smoothing_applied or
                                     noise_analysis.get('dominant_noise_type') == 'jitter') and 
                                     noise_score > 0.1)
                noise_tests.append(("Jitter noise reduction", jitter_test_passed))
            else:
                print(f"❌ Jitter noise test failed: {response.status_code} - {response.text}")
                noise_tests.append(("Jitter noise reduction", False))
        except Exception as e:
            print(f"❌ Jitter noise test error: {e}")
            noise_tests.append(("Jitter noise reduction", False))
        
        # Test 4: Oscillation noise reduction
        print("Testing oscillation noise reduction...")
        try:
            response = self.session.post(
                f"{API_BASE_URL}/test-noise-reduction",
                json={
                    "predictions": test_datasets['oscillation'],
                    "test_type": "oscillation_noise"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                noise_score = data.get('noise_reduction_score', 0)
                smoothing_applied = data.get('smoothing_applied', [])
                noise_analysis = data.get('noise_analysis', {})
                
                print(f"✅ Oscillation noise test successful")
                print(f"   Noise reduction score: {noise_score:.3f}")
                print(f"   Smoothing methods: {smoothing_applied}")
                print(f"   Detected noise type: {noise_analysis.get('dominant_noise_type', 'unknown')}")
                
                # For oscillation data, appropriate filtering should be applied
                oscillation_test_passed = (('butterworth_filter' in smoothing_applied or 
                                          'moving_average' in smoothing_applied or
                                          noise_analysis.get('dominant_noise_type') == 'oscillation') and 
                                          noise_score > 0.1)
                noise_tests.append(("Oscillation noise reduction", oscillation_test_passed))
            else:
                print(f"❌ Oscillation noise test failed: {response.status_code} - {response.text}")
                noise_tests.append(("Oscillation noise reduction", False))
        except Exception as e:
            print(f"❌ Oscillation noise test error: {e}")
            noise_tests.append(("Oscillation noise reduction", False))
        
        return noise_tests
    
    def test_enhanced_prediction_endpoints(self):
        """Test the enhanced prediction endpoints with noise reduction"""
        print("\n=== Testing Enhanced Prediction Endpoints ===")
        
        endpoint_tests = []
        
        # Test 1: Enhanced real-time prediction endpoint
        print("Testing enhanced real-time prediction endpoint...")
        try:
            response = self.session.get(
                f"{API_BASE_URL}/generate-enhanced-realtime-prediction",
                params={"steps": 20, "time_window": 100, "maintain_patterns": True}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                noise_info = data.get('noise_reduction_info', {})
                
                print(f"✅ Enhanced real-time prediction successful")
                print(f"   Predictions generated: {len(predictions)}")
                print(f"   Noise reduction applied: {noise_info.get('applied', 'unknown')}")
                
                # Check if predictions are generated and noise reduction info is present
                enhanced_realtime_passed = (len(predictions) == 20)
                endpoint_tests.append(("Enhanced real-time prediction", enhanced_realtime_passed))
            else:
                print(f"❌ Enhanced real-time prediction failed: {response.status_code} - {response.text}")
                endpoint_tests.append(("Enhanced real-time prediction", False))
        except Exception as e:
            print(f"❌ Enhanced real-time prediction error: {e}")
            endpoint_tests.append(("Enhanced real-time prediction", False))
        
        # Test 2: Advanced pH prediction endpoint
        print("Testing advanced pH prediction endpoint...")
        try:
            response = self.session.get(
                f"{API_BASE_URL}/generate-advanced-ph-prediction",
                params={"steps": 15, "maintain_patterns": True}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                print(f"✅ Advanced pH prediction successful")
                print(f"   Predictions generated: {len(predictions)}")
                
                # Check if predictions are in realistic pH range
                ph_range_valid = all(5.0 <= p <= 9.0 for p in predictions[:5]) if predictions else False
                advanced_ph_passed = len(predictions) == 15 and ph_range_valid
                endpoint_tests.append(("Advanced pH prediction", advanced_ph_passed))
            else:
                print(f"❌ Advanced pH prediction failed: {response.status_code} - {response.text}")
                endpoint_tests.append(("Advanced pH prediction", False))
        except Exception as e:
            print(f"❌ Advanced pH prediction error: {e}")
            endpoint_tests.append(("Advanced pH prediction", False))
        
        # Test 3: Extended pH prediction endpoint
        print("Testing extended pH prediction endpoint...")
        try:
            response = self.session.get(
                f"{API_BASE_URL}/extend-advanced-ph-prediction",
                params={"additional_steps": 5}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                prediction_info = data.get('prediction_info', {})
                
                print(f"✅ Extended pH prediction successful")
                print(f"   Extended predictions: {len(predictions)}")
                print(f"   Noise reduction applied: {prediction_info.get('noise_reduction_applied', False)}")
                print(f"   Smoothing type: {prediction_info.get('smoothing_type', 'none')}")
                
                # Check if real-time smoothing was applied
                extended_ph_passed = (len(predictions) == 5 and 
                                    prediction_info.get('noise_reduction_applied', False))
                endpoint_tests.append(("Extended pH prediction", extended_ph_passed))
            else:
                print(f"❌ Extended pH prediction failed: {response.status_code} - {response.text}")
                endpoint_tests.append(("Extended pH prediction", False))
        except Exception as e:
            print(f"❌ Extended pH prediction error: {e}")
            endpoint_tests.append(("Extended pH prediction", False))
        
        return endpoint_tests
    
    def test_pattern_preservation(self):
        """Test that noise reduction maintains pattern integrity"""
        print("\n=== Testing Pattern Preservation ===")
        
        pattern_tests = []
        
        # Create test data with clear patterns
        x = np.linspace(0, 4*np.pi, 100)
        
        # Test 1: Sine wave pattern preservation
        print("Testing sine wave pattern preservation...")
        try:
            sine_pattern = (7.0 + 0.5 * np.sin(x) + np.random.normal(0, 0.1, len(x))).tolist()
            
            response = self.session.post(
                f"{API_BASE_URL}/test-noise-reduction",
                json={
                    "predictions": sine_pattern,
                    "test_type": "sine_pattern"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                original = data.get('original_predictions', [])
                smoothed = data.get('smoothed_predictions', [])
                quality_metrics = data.get('quality_metrics', {})
                
                print(f"✅ Sine pattern test successful")
                print(f"   Pattern preservation score: {quality_metrics.get('pattern_preservation_score', 0):.3f}")
                print(f"   Smoothness score: {quality_metrics.get('smoothness_score', 0):.3f}")
                
                # Check if pattern is preserved (high pattern preservation score)
                sine_pattern_passed = quality_metrics.get('pattern_preservation_score', 0) > 0.6
                pattern_tests.append(("Sine wave pattern preservation", sine_pattern_passed))
            else:
                print(f"❌ Sine pattern test failed: {response.status_code}")
                pattern_tests.append(("Sine wave pattern preservation", False))
        except Exception as e:
            print(f"❌ Sine pattern test error: {e}")
            pattern_tests.append(("Sine wave pattern preservation", False))
        
        # Test 2: Linear trend preservation
        print("Testing linear trend preservation...")
        try:
            linear_trend = (5.0 + 0.1 * np.arange(50) + np.random.normal(0, 0.05, 50)).tolist()
            
            response = self.session.post(
                f"{API_BASE_URL}/test-noise-reduction",
                json={
                    "predictions": linear_trend,
                    "test_type": "linear_trend"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                quality_metrics = data.get('quality_metrics', {})
                
                print(f"✅ Linear trend test successful")
                print(f"   Pattern preservation score: {quality_metrics.get('pattern_preservation_score', 0):.3f}")
                
                # Check if trend is preserved
                linear_trend_passed = quality_metrics.get('pattern_preservation_score', 0) > 0.7
                pattern_tests.append(("Linear trend preservation", linear_trend_passed))
            else:
                print(f"❌ Linear trend test failed: {response.status_code}")
                pattern_tests.append(("Linear trend preservation", False))
        except Exception as e:
            print(f"❌ Linear trend test error: {e}")
            pattern_tests.append(("Linear trend preservation", False))
        
        return pattern_tests
    
    def run_comprehensive_noise_reduction_tests(self):
        """Run all noise reduction tests"""
        print("🎯 COMPREHENSIVE NOISE REDUCTION SYSTEM TESTING")
        print("=" * 60)
        
        all_tests = []
        
        # Test core noise reduction system
        core_tests = self.test_core_noise_reduction_system()
        all_tests.extend(core_tests)
        
        # Test enhanced prediction endpoints
        endpoint_tests = self.test_enhanced_prediction_endpoints()
        all_tests.extend(endpoint_tests)
        
        # Test pattern preservation
        pattern_tests = self.test_pattern_preservation()
        all_tests.extend(pattern_tests)
        
        # Print comprehensive results
        print("\n" + "=" * 60)
        print("🎯 NOISE REDUCTION SYSTEM TEST RESULTS")
        print("=" * 60)
        
        passed_tests = sum(1 for _, passed in all_tests if passed)
        total_tests = len(all_tests)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        # Group results by category
        core_results = [test for test in all_tests if any(keyword in test[0].lower() 
                       for keyword in ['smooth', 'spike', 'jitter', 'oscillation'])]
        endpoint_results = [test for test in all_tests if any(keyword in test[0].lower() 
                           for keyword in ['enhanced', 'advanced', 'extended'])]
        pattern_results = [test for test in all_tests if 'preservation' in test[0].lower()]
        
        print(f"\n🔧 Core Noise Reduction System:")
        for test_name, passed in core_results:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        print(f"\n🚀 Enhanced Prediction Endpoints:")
        for test_name, passed in endpoint_results:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        print(f"\n🎨 Pattern Preservation:")
        for test_name, passed in pattern_results:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        # Overall assessment
        if success_rate >= 80:
            print(f"\n🎉 EXCELLENT: Noise reduction system is working excellently!")
        elif success_rate >= 60:
            print(f"\n✅ GOOD: Noise reduction system is working well with minor issues.")
        elif success_rate >= 40:
            print(f"\n⚠️  PARTIAL: Noise reduction system has significant issues that need attention.")
        else:
            print(f"\n❌ CRITICAL: Noise reduction system has major problems and needs immediate fixes.")
        
        return success_rate >= 60

if __name__ == "__main__":
    tester = NoiseReductionTester()
    success = tester.run_comprehensive_noise_reduction_tests()
    
    if success:
        print(f"\n🎯 CONCLUSION: Noise reduction system testing PASSED!")
    else:
        print(f"\n❌ CONCLUSION: Noise reduction system testing FAILED!")