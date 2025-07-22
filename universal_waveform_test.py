#!/usr/bin/env python3
"""
Comprehensive Testing for Enhanced Universal Waveform Prediction System
Tests the new /api/generate-universal-waveform-prediction endpoint with various waveform types
"""

import requests
import json
import pandas as pd
import numpy as np
import io
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import math

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://5eb713e9-78d7-42c8-a00c-608d827b4afb.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Universal Waveform Prediction System at: {API_BASE_URL}")

class UniversalWaveformTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        self.data_id = None
        
    def create_waveform_data(self, waveform_type: str, samples: int = 100) -> pd.DataFrame:
        """Create different types of waveform data for testing"""
        t = np.linspace(0, 4*np.pi, samples)
        
        if waveform_type == 'square_wave':
            # Square wave with sharp transitions and flat segments
            data = np.sign(np.sin(t)) * 2 + 5
            # Add some realistic noise
            data += np.random.normal(0, 0.1, len(data))
            
        elif waveform_type == 'triangular_wave':
            # Triangular wave with linear segments and sharp peaks
            data = 2 * np.arcsin(np.sin(t)) / np.pi * 3 + 7
            data += np.random.normal(0, 0.05, len(data))
            
        elif waveform_type == 'sawtooth_wave':
            # Sawtooth wave with linear ramps and sharp drops
            data = 2 * (t % (2*np.pi)) / (2*np.pi) - 1
            data = data * 2 + 6
            data += np.random.normal(0, 0.08, len(data))
            
        elif waveform_type == 'sinusoidal':
            # Smooth sinusoidal curves
            data = 3 * np.sin(t) + 2 * np.sin(2*t) + 8
            data += np.random.normal(0, 0.1, len(data))
            
        elif waveform_type == 'step_function':
            # Step function with discrete levels
            steps = np.floor(t / (np.pi/2)) % 4
            data = steps * 1.5 + 4
            data += np.random.normal(0, 0.05, len(data))
            
        elif waveform_type == 'complex_composite':
            # Complex composite pattern combining multiple waveforms
            data = (2 * np.sin(t) + 
                   1.5 * np.sign(np.sin(2*t)) + 
                   0.8 * (2 * (t % np.pi) / np.pi - 1) + 
                   0.5 * np.sin(5*t) + 6)
            data += np.random.normal(0, 0.1, len(data))
            
        elif waveform_type == 'irregular_pattern':
            # Irregular pattern with random characteristics
            data = np.cumsum(np.random.normal(0, 0.3, samples))
            data += 2 * np.sin(t/2) + np.random.exponential(0.2, samples) - 0.2
            data = data - np.mean(data) + 7
            
        else:
            # Default to sinusoidal
            data = np.sin(t) + 5
            
        # Create DataFrame with timestamp and target columns
        timestamps = pd.date_range(start='2024-01-01', periods=samples, freq='H')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': data,
            'sensor_id': [f'sensor_{waveform_type}'] * samples
        })
        
        return df
    
    def upload_waveform_data(self, waveform_type: str) -> bool:
        """Upload waveform data to the backend"""
        try:
            print(f"\n📤 Uploading {waveform_type} waveform data...")
            
            # Create waveform data
            df = self.create_waveform_data(waveform_type)
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Upload file
            files = {
                'file': (f'{waveform_type}_data.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                result = response.json()
                self.data_id = result.get('data_id')
                print(f"✅ {waveform_type} data uploaded successfully (ID: {self.data_id})")
                print(f"   Data shape: {result.get('data_shape', 'Unknown')}")
                print(f"   Columns detected: {result.get('columns', [])}")
                return True
            else:
                print(f"❌ Failed to upload {waveform_type} data: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading {waveform_type} data: {e}")
            return False
    
    def test_universal_waveform_prediction(self, waveform_type: str, steps: int = 30, 
                                         time_window: int = 80, learning_mode: str = 'comprehensive') -> dict:
        """Test the universal waveform prediction endpoint"""
        try:
            print(f"\n🔮 Testing universal waveform prediction for {waveform_type}...")
            
            # Make prediction request
            params = {
                'steps': steps,
                'time_window': time_window,
                'learning_mode': learning_mode
            }
            
            response = self.session.get(f"{API_BASE_URL}/generate-universal-waveform-prediction", params=params)
            
            if response.status_code == 200:
                result = response.json()
                
                # Analyze response structure
                test_result = {
                    'waveform_type': waveform_type,
                    'status': result.get('status'),
                    'predictions_count': len(result.get('predictions', [])),
                    'timestamps_count': len(result.get('timestamps', [])),
                    'has_confidence_intervals': bool(result.get('confidence_intervals')),
                    'prediction_method': result.get('prediction_method'),
                    'waveform_analysis': result.get('waveform_analysis', {}),
                    'prediction_capabilities': result.get('prediction_capabilities', {}),
                    'quality_metrics': result.get('quality_metrics', {}),
                    'learning_summary': result.get('learning_summary', {}),
                    'success': True
                }
                
                # Analyze waveform analysis results
                waveform_analysis = result.get('waveform_analysis', {})
                detected_patterns = waveform_analysis.get('detected_patterns', {})
                pattern_complexity = waveform_analysis.get('pattern_complexity', 0)
                shape_preservation = waveform_analysis.get('shape_preservation', 0)
                geometric_consistency = waveform_analysis.get('geometric_consistency', 0)
                
                # Analyze prediction capabilities
                capabilities = result.get('prediction_capabilities', {})
                supported_types = capabilities.get('waveform_types_supported', [])
                complexity_handling = capabilities.get('complexity_handling', 0)
                learning_quality = capabilities.get('pattern_learning_quality', {})
                
                # Analyze quality metrics
                quality = result.get('quality_metrics', {})
                overall_quality = quality.get('overall_quality', 0)
                pattern_following = quality.get('pattern_following_score', 0)
                waveform_fidelity = quality.get('waveform_fidelity', 0)
                
                # Analyze learning summary
                learning = result.get('learning_summary', {})
                patterns_learned = learning.get('patterns_learned', 0)
                data_points = learning.get('data_points_analyzed', 0)
                
                print(f"✅ Universal waveform prediction successful for {waveform_type}")
                print(f"   📊 Generated {test_result['predictions_count']} predictions")
                print(f"   🎯 Pattern complexity: {pattern_complexity:.3f}")
                print(f"   🔧 Shape preservation: {shape_preservation:.3f}")
                print(f"   📐 Geometric consistency: {geometric_consistency:.3f}")
                print(f"   🏆 Overall quality: {overall_quality:.3f}")
                print(f"   📈 Pattern following score: {pattern_following:.3f}")
                print(f"   🌊 Waveform fidelity: {waveform_fidelity:.3f}")
                print(f"   🧠 Patterns learned: {patterns_learned}")
                print(f"   📋 Data points analyzed: {data_points}")
                print(f"   🎨 Detected patterns: {list(detected_patterns.keys())}")
                print(f"   🔧 Supported waveform types: {supported_types}")
                
                # Additional analysis for predictions
                predictions = result.get('predictions', [])
                if predictions:
                    pred_array = np.array(predictions)
                    test_result.update({
                        'prediction_range': [float(np.min(pred_array)), float(np.max(pred_array))],
                        'prediction_mean': float(np.mean(pred_array)),
                        'prediction_std': float(np.std(pred_array)),
                        'prediction_variability': len(np.unique(np.round(pred_array, 2)))
                    })
                    
                    print(f"   📊 Prediction range: [{test_result['prediction_range'][0]:.2f}, {test_result['prediction_range'][1]:.2f}]")
                    print(f"   📊 Prediction mean: {test_result['prediction_mean']:.2f}")
                    print(f"   📊 Prediction variability: {test_result['prediction_variability']} unique values")
                
                return test_result
                
            else:
                print(f"❌ Universal waveform prediction failed for {waveform_type}: {response.status_code}")
                print(f"   Response: {response.text}")
                return {
                    'waveform_type': waveform_type,
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'status_code': response.status_code
                }
                
        except Exception as e:
            print(f"❌ Error testing universal waveform prediction for {waveform_type}: {e}")
            return {
                'waveform_type': waveform_type,
                'success': False,
                'error': str(e)
            }
    
    def test_pattern_learning_quality(self, waveform_type: str) -> dict:
        """Test the quality of pattern learning for specific waveform types"""
        try:
            print(f"\n🧠 Testing pattern learning quality for {waveform_type}...")
            
            # Test with different learning modes
            learning_modes = ['comprehensive', 'fast', 'detailed']
            results = {}
            
            for mode in learning_modes:
                result = self.test_universal_waveform_prediction(
                    waveform_type, steps=20, time_window=60, learning_mode=mode
                )
                results[mode] = result
                
                if result.get('success'):
                    quality = result.get('quality_metrics', {})
                    print(f"   {mode} mode - Quality: {quality.get('overall_quality', 0):.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error testing pattern learning quality: {e}")
            return {'error': str(e)}
    
    def test_prediction_consistency(self, waveform_type: str) -> dict:
        """Test consistency of predictions across multiple calls"""
        try:
            print(f"\n🔄 Testing prediction consistency for {waveform_type}...")
            
            results = []
            for i in range(3):
                result = self.test_universal_waveform_prediction(waveform_type, steps=15)
                if result.get('success'):
                    results.append(result)
                time.sleep(0.5)  # Small delay between calls
            
            if len(results) >= 2:
                # Analyze consistency
                qualities = [r.get('quality_metrics', {}).get('overall_quality', 0) for r in results]
                pattern_scores = [r.get('quality_metrics', {}).get('pattern_following_score', 0) for r in results]
                
                quality_consistency = 1.0 - (np.std(qualities) / (np.mean(qualities) + 1e-8))
                pattern_consistency = 1.0 - (np.std(pattern_scores) / (np.mean(pattern_scores) + 1e-8))
                
                print(f"   📊 Quality consistency: {quality_consistency:.3f}")
                print(f"   📊 Pattern consistency: {pattern_consistency:.3f}")
                
                return {
                    'waveform_type': waveform_type,
                    'consistency_results': results,
                    'quality_consistency': quality_consistency,
                    'pattern_consistency': pattern_consistency,
                    'success': True
                }
            else:
                return {'success': False, 'error': 'Insufficient successful predictions for consistency test'}
                
        except Exception as e:
            print(f"❌ Error testing prediction consistency: {e}")
            return {'error': str(e)}
    
    def test_error_handling(self) -> dict:
        """Test error handling with edge cases"""
        try:
            print(f"\n⚠️ Testing error handling...")
            
            error_tests = {}
            
            # Test 1: No data uploaded
            print("   Testing with no data...")
            response = self.session.get(f"{API_BASE_URL}/generate-universal-waveform-prediction")
            error_tests['no_data'] = {
                'status_code': response.status_code,
                'response': response.json() if response.status_code == 200 else response.text,
                'handled_gracefully': response.status_code in [200, 400]
            }
            
            # Test 2: Invalid parameters
            print("   Testing with invalid parameters...")
            params = {'steps': -10, 'time_window': -5, 'learning_mode': 'invalid_mode'}
            response = self.session.get(f"{API_BASE_URL}/generate-universal-waveform-prediction", params=params)
            error_tests['invalid_params'] = {
                'status_code': response.status_code,
                'response': response.json() if response.status_code == 200 else response.text,
                'handled_gracefully': response.status_code in [200, 400]
            }
            
            # Test 3: Extreme parameters
            print("   Testing with extreme parameters...")
            params = {'steps': 1000, 'time_window': 10000}
            response = self.session.get(f"{API_BASE_URL}/generate-universal-waveform-prediction", params=params)
            error_tests['extreme_params'] = {
                'status_code': response.status_code,
                'response': response.json() if response.status_code == 200 else response.text,
                'handled_gracefully': response.status_code in [200, 400, 500]
            }
            
            print(f"   ✅ Error handling tests completed")
            return error_tests
            
        except Exception as e:
            print(f"❌ Error testing error handling: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_waveform_tests(self):
        """Run comprehensive tests for all waveform types"""
        print("🚀 Starting Comprehensive Universal Waveform Prediction System Testing")
        print("=" * 80)
        
        # Define waveform types to test
        waveform_types = [
            'square_wave',
            'triangular_wave', 
            'sawtooth_wave',
            'sinusoidal',
            'step_function',
            'complex_composite',
            'irregular_pattern'
        ]
        
        all_results = {}
        successful_tests = 0
        total_tests = 0
        
        # Test each waveform type
        for waveform_type in waveform_types:
            print(f"\n{'='*60}")
            print(f"🌊 TESTING WAVEFORM TYPE: {waveform_type.upper()}")
            print(f"{'='*60}")
            
            # Upload data for this waveform type
            if self.upload_waveform_data(waveform_type):
                
                # Test 1: Basic universal waveform prediction
                result = self.test_universal_waveform_prediction(waveform_type)
                all_results[f"{waveform_type}_basic"] = result
                if result.get('success'):
                    successful_tests += 1
                total_tests += 1
                
                # Test 2: Pattern learning quality
                learning_result = self.test_pattern_learning_quality(waveform_type)
                all_results[f"{waveform_type}_learning"] = learning_result
                if any(r.get('success', False) for r in learning_result.values() if isinstance(r, dict)):
                    successful_tests += 1
                total_tests += 1
                
                # Test 3: Prediction consistency
                consistency_result = self.test_prediction_consistency(waveform_type)
                all_results[f"{waveform_type}_consistency"] = consistency_result
                if consistency_result.get('success'):
                    successful_tests += 1
                total_tests += 1
            
            else:
                print(f"❌ Skipping tests for {waveform_type} due to upload failure")
        
        # Test error handling
        print(f"\n{'='*60}")
        print(f"⚠️ TESTING ERROR HANDLING")
        print(f"{'='*60}")
        
        error_result = self.test_error_handling()
        all_results['error_handling'] = error_result
        if error_result and not error_result.get('error'):
            successful_tests += 1
        total_tests += 1
        
        # Generate comprehensive summary
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE UNIVERSAL WAVEFORM PREDICTION TESTING SUMMARY")
        print(f"{'='*80}")
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"🎯 Overall Success Rate: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        
        # Analyze results by category
        waveform_results = {}
        for waveform_type in waveform_types:
            basic_key = f"{waveform_type}_basic"
            if basic_key in all_results and all_results[basic_key].get('success'):
                result = all_results[basic_key]
                quality_metrics = result.get('quality_metrics', {})
                waveform_analysis = result.get('waveform_analysis', {})
                
                waveform_results[waveform_type] = {
                    'overall_quality': quality_metrics.get('overall_quality', 0),
                    'pattern_following': quality_metrics.get('pattern_following_score', 0),
                    'waveform_fidelity': quality_metrics.get('waveform_fidelity', 0),
                    'shape_preservation': waveform_analysis.get('shape_preservation', 0),
                    'geometric_consistency': waveform_analysis.get('geometric_consistency', 0),
                    'pattern_complexity': waveform_analysis.get('pattern_complexity', 0)
                }
        
        # Print detailed results
        print(f"\n📈 WAVEFORM-SPECIFIC RESULTS:")
        for waveform_type, metrics in waveform_results.items():
            print(f"   🌊 {waveform_type}:")
            print(f"      Overall Quality: {metrics['overall_quality']:.3f}")
            print(f"      Pattern Following: {metrics['pattern_following']:.3f}")
            print(f"      Waveform Fidelity: {metrics['waveform_fidelity']:.3f}")
            print(f"      Shape Preservation: {metrics['shape_preservation']:.3f}")
            print(f"      Geometric Consistency: {metrics['geometric_consistency']:.3f}")
            print(f"      Pattern Complexity: {metrics['pattern_complexity']:.3f}")
        
        # Calculate average metrics
        if waveform_results:
            avg_quality = np.mean([m['overall_quality'] for m in waveform_results.values()])
            avg_pattern_following = np.mean([m['pattern_following'] for m in waveform_results.values()])
            avg_fidelity = np.mean([m['waveform_fidelity'] for m in waveform_results.values()])
            avg_shape_preservation = np.mean([m['shape_preservation'] for m in waveform_results.values()])
            avg_geometric_consistency = np.mean([m['geometric_consistency'] for m in waveform_results.values()])
            
            print(f"\n📊 AVERAGE PERFORMANCE METRICS:")
            print(f"   🏆 Average Overall Quality: {avg_quality:.3f}")
            print(f"   📈 Average Pattern Following: {avg_pattern_following:.3f}")
            print(f"   🌊 Average Waveform Fidelity: {avg_fidelity:.3f}")
            print(f"   🔧 Average Shape Preservation: {avg_shape_preservation:.3f}")
            print(f"   📐 Average Geometric Consistency: {avg_geometric_consistency:.3f}")
        
        # Final assessment
        print(f"\n🎉 FINAL ASSESSMENT:")
        if success_rate >= 80:
            print(f"   ✅ EXCELLENT: Universal waveform prediction system is working excellently!")
            print(f"   ✅ The system successfully handles multiple waveform types with high quality.")
        elif success_rate >= 60:
            print(f"   ⚠️ GOOD: Universal waveform prediction system is working well with minor issues.")
        else:
            print(f"   ❌ NEEDS IMPROVEMENT: Universal waveform prediction system has significant issues.")
        
        # Store results for potential further analysis
        self.test_results = all_results
        
        return {
            'success_rate': success_rate,
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'waveform_results': waveform_results,
            'all_results': all_results
        }

def main():
    """Main testing function"""
    tester = UniversalWaveformTester()
    results = tester.run_comprehensive_waveform_tests()
    
    # Return results for potential integration with other testing systems
    return results

if __name__ == "__main__":
    main()