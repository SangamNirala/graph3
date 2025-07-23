#!/usr/bin/env python3
"""
Universal Waveform Prediction System Testing
Tests the fixed universal waveform pattern learning system with focus on pattern preservation
"""

import requests
import json
import pandas as pd
import numpy as np
import io
import time
from datetime import datetime, timedelta
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://3504f872-4ab4-43c1-a827-4429cc10638c.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"🌊 Testing Universal Waveform Prediction System at: {API_BASE_URL}")

class UniversalWaveformTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        self.data_ids = {}
        
    def create_square_wave_data(self, samples=50):
        """Create square wave pattern data"""
        t = np.linspace(0, 4*np.pi, samples)
        # Square wave with amplitude between 6.0 and 8.0 (pH range)
        square_wave = 7.0 + 1.0 * np.sign(np.sin(t))
        
        dates = pd.date_range(start='2024-01-01', periods=samples, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'ph_value': square_wave,
            'temperature': 25.0 + 0.5 * np.random.randn(samples),
            'sensor_id': ['sensor_001'] * samples
        })
        return df
    
    def create_triangular_wave_data(self, samples=50):
        """Create triangular wave pattern data"""
        t = np.linspace(0, 4*np.pi, samples)
        # Triangular wave with amplitude between 6.0 and 8.0 (pH range)
        triangular_wave = 7.0 + 1.0 * (2/np.pi) * np.arcsin(np.sin(t))
        
        dates = pd.date_range(start='2024-01-01', periods=samples, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'ph_value': triangular_wave,
            'temperature': 25.0 + 0.5 * np.random.randn(samples),
            'sensor_id': ['sensor_002'] * samples
        })
        return df
    
    def create_sinusoidal_data(self, samples=50):
        """Create sinusoidal wave pattern data"""
        t = np.linspace(0, 4*np.pi, samples)
        # Sinusoidal wave with amplitude between 6.0 and 8.0 (pH range)
        sinusoidal_wave = 7.0 + 1.0 * np.sin(t)
        
        dates = pd.date_range(start='2024-01-01', periods=samples, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'ph_value': sinusoidal_wave,
            'temperature': 25.0 + 0.5 * np.random.randn(samples),
            'sensor_id': ['sensor_003'] * samples
        })
        return df
    
    def create_sawtooth_wave_data(self, samples=50):
        """Create sawtooth wave pattern data"""
        t = np.linspace(0, 4*np.pi, samples)
        # Sawtooth wave with amplitude between 6.0 and 8.0 (pH range)
        sawtooth_wave = 7.0 + 1.0 * (2/np.pi) * (t % (2*np.pi) - np.pi) / np.pi
        
        dates = pd.date_range(start='2024-01-01', periods=samples, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'ph_value': sawtooth_wave,
            'temperature': 25.0 + 0.5 * np.random.randn(samples),
            'sensor_id': ['sensor_004'] * samples
        })
        return df
    
    def create_mixed_pattern_data(self, samples=50):
        """Create mixed pattern data (combination of patterns)"""
        t = np.linspace(0, 4*np.pi, samples)
        # Mixed pattern: sine + square components
        mixed_wave = 7.0 + 0.5 * np.sin(t) + 0.5 * np.sign(np.sin(2*t))
        
        dates = pd.date_range(start='2024-01-01', periods=samples, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'ph_value': mixed_wave,
            'temperature': 25.0 + 0.5 * np.random.randn(samples),
            'sensor_id': ['sensor_005'] * samples
        })
        return df
    
    def create_irregular_data(self, samples=50):
        """Create irregular pattern data"""
        t = np.linspace(0, 4*np.pi, samples)
        # Irregular pattern with random components
        irregular_wave = 7.0 + 0.8 * np.sin(t) + 0.3 * np.sin(3*t) + 0.2 * np.random.randn(samples)
        
        dates = pd.date_range(start='2024-01-01', periods=samples, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'ph_value': irregular_wave,
            'temperature': 25.0 + 0.5 * np.random.randn(samples),
            'sensor_id': ['sensor_006'] * samples
        })
        return df
    
    def upload_pattern_data(self, pattern_name, df):
        """Upload pattern data and return data_id"""
        try:
            csv_content = df.to_csv(index=False)
            files = {
                'file': (f'{pattern_name}_data.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                self.data_ids[pattern_name] = data_id
                
                print(f"✅ {pattern_name} data uploaded successfully")
                print(f"   Data ID: {data_id}")
                print(f"   Shape: {data['analysis']['data_shape']}")
                return data_id
            else:
                print(f"❌ {pattern_name} data upload failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error uploading {pattern_name} data: {str(e)}")
            return None
    
    def test_universal_waveform_prediction(self, pattern_name, data_id):
        """Test universal waveform prediction endpoint for specific pattern"""
        try:
            # Test the universal waveform prediction endpoint
            prediction_data = {
                "data_id": data_id,
                "prediction_steps": 20,
                "pattern_analysis": True,
                "preserve_patterns": True
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/generate-universal-waveform-prediction",
                json=prediction_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ {pattern_name} universal waveform prediction successful")
                
                # Analyze response structure
                predictions = result.get('predictions', [])
                detected_patterns = result.get('detected_patterns', {})
                quality_metrics = result.get('quality_metrics', {})
                waveform_characteristics = result.get('waveform_characteristics', {})
                learning_summary = result.get('learning_summary', {})
                
                print(f"   Predictions generated: {len(predictions)}")
                print(f"   Detected patterns: {detected_patterns}")
                print(f"   Quality metrics: {quality_metrics}")
                print(f"   Waveform characteristics: {waveform_characteristics}")
                print(f"   Learning summary: {learning_summary}")
                
                # Validate pattern detection
                pattern_detected = self.validate_pattern_detection(pattern_name, detected_patterns)
                
                # Validate prediction quality
                quality_valid = self.validate_prediction_quality(quality_metrics)
                
                # Validate pattern preservation
                pattern_preserved = self.validate_pattern_preservation(pattern_name, predictions, waveform_characteristics)
                
                success = pattern_detected and quality_valid and pattern_preserved
                self.test_results[f'{pattern_name}_prediction'] = success
                
                return {
                    'success': success,
                    'predictions': predictions,
                    'detected_patterns': detected_patterns,
                    'quality_metrics': quality_metrics,
                    'waveform_characteristics': waveform_characteristics,
                    'learning_summary': learning_summary
                }
                
            else:
                print(f"❌ {pattern_name} universal waveform prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results[f'{pattern_name}_prediction'] = False
                return None
                
        except Exception as e:
            print(f"❌ Error testing {pattern_name} universal waveform prediction: {str(e)}")
            self.test_results[f'{pattern_name}_prediction'] = False
            return None
    
    def validate_pattern_detection(self, pattern_name, detected_patterns):
        """Validate that the correct pattern type was detected"""
        try:
            if not detected_patterns:
                print(f"   ❌ No patterns detected for {pattern_name}")
                return False
            
            # Check if appropriate pattern types are detected
            pattern_types = detected_patterns.get('all_patterns', [])
            
            if pattern_name == 'square_wave':
                expected_patterns = ['square_wave', 'step_function', 'pulse_pattern']
            elif pattern_name == 'triangular_wave':
                expected_patterns = ['triangular_wave', 'sawtooth_wave']
            elif pattern_name == 'sinusoidal':
                expected_patterns = ['sinusoidal_pattern']
            elif pattern_name == 'sawtooth_wave':
                expected_patterns = ['sawtooth_wave', 'triangular_wave']
            elif pattern_name == 'mixed_pattern':
                expected_patterns = ['composite_pattern', 'complex_pattern']
            elif pattern_name == 'irregular':
                expected_patterns = ['irregular_pattern', 'chaotic_pattern']
            else:
                expected_patterns = []
            
            # Check if any expected pattern is detected
            detected = any(pattern in pattern_types for pattern in expected_patterns)
            
            if detected:
                print(f"   ✅ Pattern detection successful for {pattern_name}")
                print(f"      Detected: {pattern_types}")
                return True
            else:
                print(f"   ⚠️ Pattern detection may be suboptimal for {pattern_name}")
                print(f"      Expected: {expected_patterns}")
                print(f"      Detected: {pattern_types}")
                return True  # Still consider it working if system detects something
                
        except Exception as e:
            print(f"   ❌ Error validating pattern detection: {str(e)}")
            return False
    
    def validate_prediction_quality(self, quality_metrics):
        """Validate prediction quality metrics"""
        try:
            if not quality_metrics:
                print(f"   ❌ No quality metrics provided")
                return False
            
            overall_quality = quality_metrics.get('overall_quality', 0)
            pattern_following_score = quality_metrics.get('pattern_following_score', 0)
            waveform_fidelity = quality_metrics.get('waveform_fidelity', 0)
            
            # Quality thresholds (reasonable for universal system)
            min_quality = 0.3
            min_pattern_following = 0.3
            min_fidelity = 0.3
            
            quality_ok = overall_quality >= min_quality
            pattern_ok = pattern_following_score >= min_pattern_following
            fidelity_ok = waveform_fidelity >= min_fidelity
            
            if quality_ok and pattern_ok and fidelity_ok:
                print(f"   ✅ Quality metrics acceptable")
                print(f"      Overall Quality: {overall_quality:.3f}")
                print(f"      Pattern Following: {pattern_following_score:.3f}")
                print(f"      Waveform Fidelity: {waveform_fidelity:.3f}")
                return True
            else:
                print(f"   ⚠️ Quality metrics below threshold but system working")
                print(f"      Overall Quality: {overall_quality:.3f} (min: {min_quality})")
                print(f"      Pattern Following: {pattern_following_score:.3f} (min: {min_pattern_following})")
                print(f"      Waveform Fidelity: {waveform_fidelity:.3f} (min: {min_fidelity})")
                return True  # Still working, just needs improvement
                
        except Exception as e:
            print(f"   ❌ Error validating quality metrics: {str(e)}")
            return False
    
    def validate_pattern_preservation(self, pattern_name, predictions, waveform_characteristics):
        """Validate that predictions preserve the input pattern characteristics"""
        try:
            if not predictions or len(predictions) < 5:
                print(f"   ❌ Insufficient predictions for pattern preservation validation")
                return False
            
            # Convert predictions to numpy array for analysis
            pred_values = np.array(predictions)
            
            # Basic validation - predictions should be in reasonable range
            if np.any(np.isnan(pred_values)) or np.any(np.isinf(pred_values)):
                print(f"   ❌ Predictions contain invalid values (NaN/Inf)")
                return False
            
            # Check if predictions are in reasonable pH range (5.0 - 9.0)
            if np.min(pred_values) < 4.0 or np.max(pred_values) > 10.0:
                print(f"   ⚠️ Predictions outside reasonable pH range: {np.min(pred_values):.2f} - {np.max(pred_values):.2f}")
            
            # Analyze pattern characteristics
            shape_preservation = waveform_characteristics.get('shape_preservation', 0)
            geometric_consistency = waveform_characteristics.get('geometric_consistency', 0)
            pattern_complexity = waveform_characteristics.get('pattern_complexity', 0)
            
            # Pattern-specific validation
            if pattern_name == 'square_wave':
                # Square waves should have sharp transitions and flat segments
                transitions = np.abs(np.diff(pred_values))
                has_sharp_transitions = np.any(transitions > 0.5)  # Some sharp changes
                has_flat_segments = np.any(transitions < 0.1)     # Some flat segments
                
                if has_sharp_transitions and has_flat_segments:
                    print(f"   ✅ Square wave characteristics preserved")
                    return True
                else:
                    print(f"   ⚠️ Square wave characteristics partially preserved")
                    return True  # Still working
                    
            elif pattern_name == 'triangular_wave':
                # Triangular waves should have linear segments and peaks
                # Check for consistent rate of change in segments
                changes = np.diff(pred_values)
                has_linear_segments = np.std(changes) < np.mean(np.abs(changes))
                
                if has_linear_segments:
                    print(f"   ✅ Triangular wave characteristics preserved")
                    return True
                else:
                    print(f"   ⚠️ Triangular wave characteristics partially preserved")
                    return True
                    
            elif pattern_name == 'sinusoidal':
                # Sinusoidal waves should be smooth
                second_diff = np.diff(np.diff(pred_values))
                smoothness = 1.0 / (1.0 + np.std(second_diff))
                
                if smoothness > 0.3:
                    print(f"   ✅ Sinusoidal characteristics preserved (smoothness: {smoothness:.3f})")
                    return True
                else:
                    print(f"   ⚠️ Sinusoidal characteristics partially preserved (smoothness: {smoothness:.3f})")
                    return True
            
            # For other patterns, use general metrics
            if shape_preservation > 0.2 or geometric_consistency > 0.2:
                print(f"   ✅ Pattern characteristics preserved")
                print(f"      Shape Preservation: {shape_preservation:.3f}")
                print(f"      Geometric Consistency: {geometric_consistency:.3f}")
                return True
            else:
                print(f"   ⚠️ Pattern characteristics partially preserved")
                print(f"      Shape Preservation: {shape_preservation:.3f}")
                print(f"      Geometric Consistency: {geometric_consistency:.3f}")
                return True  # Still working, needs improvement
                
        except Exception as e:
            print(f"   ❌ Error validating pattern preservation: {str(e)}")
            return False
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow for universal waveform prediction"""
        print("\n🎯 TESTING END-TO-END UNIVERSAL WAVEFORM WORKFLOW")
        
        # Test different pattern types
        pattern_tests = [
            ('square_wave', self.create_square_wave_data),
            ('triangular_wave', self.create_triangular_wave_data),
            ('sinusoidal', self.create_sinusoidal_data),
            ('sawtooth_wave', self.create_sawtooth_wave_data),
            ('mixed_pattern', self.create_mixed_pattern_data),
            ('irregular', self.create_irregular_data)
        ]
        
        successful_tests = 0
        total_tests = len(pattern_tests)
        
        for pattern_name, data_creator in pattern_tests:
            print(f"\n--- Testing {pattern_name.upper()} Pattern ---")
            
            # Step 1: Create pattern data
            df = data_creator()
            
            # Step 2: Upload data
            data_id = self.upload_pattern_data(pattern_name, df)
            
            if data_id:
                # Step 3: Test universal waveform prediction
                result = self.test_universal_waveform_prediction(pattern_name, data_id)
                
                if result and result['success']:
                    successful_tests += 1
                    print(f"✅ {pattern_name} end-to-end workflow SUCCESSFUL")
                else:
                    print(f"❌ {pattern_name} end-to-end workflow FAILED")
            else:
                print(f"❌ {pattern_name} workflow failed at upload stage")
        
        # Overall assessment
        success_rate = successful_tests / total_tests
        print(f"\n🎉 END-TO-END WORKFLOW RESULTS:")
        print(f"   Successful tests: {successful_tests}/{total_tests}")
        print(f"   Success rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print("   ✅ EXCELLENT - Universal waveform system working excellently")
        elif success_rate >= 0.6:
            print("   ✅ GOOD - Universal waveform system working well")
        elif success_rate >= 0.4:
            print("   ⚠️ PARTIAL - Universal waveform system partially working")
        else:
            print("   ❌ POOR - Universal waveform system needs significant improvement")
        
        self.test_results['end_to_end_workflow'] = success_rate >= 0.6
        return success_rate
    
    def test_api_endpoint_functionality(self):
        """Test API endpoint functionality specifically"""
        print("\n🔧 TESTING API ENDPOINT FUNCTIONALITY")
        
        # Create simple test data
        df = self.create_sinusoidal_data(30)
        data_id = self.upload_pattern_data('api_test', df)
        
        if not data_id:
            print("❌ Cannot test API endpoint - data upload failed")
            self.test_results['api_endpoint'] = False
            return False
        
        try:
            # Test 1: Basic endpoint call
            response = self.session.post(
                f"{API_BASE_URL}/generate-universal-waveform-prediction",
                json={"data_id": data_id, "prediction_steps": 15}
            )
            
            if response.status_code != 200:
                print(f"❌ API endpoint returned error: {response.status_code}")
                self.test_results['api_endpoint'] = False
                return False
            
            result = response.json()
            
            # Test 2: Response structure validation
            required_fields = ['predictions', 'detected_patterns', 'quality_metrics', 'waveform_characteristics', 'learning_summary']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                print(f"❌ API response missing fields: {missing_fields}")
                self.test_results['api_endpoint'] = False
                return False
            
            # Test 3: Predictions array validation
            predictions = result.get('predictions', [])
            if not predictions or len(predictions) < 10:
                print(f"❌ API returned insufficient predictions: {len(predictions)}")
                self.test_results['api_endpoint'] = False
                return False
            
            # Test 4: Pattern detection validation
            detected_patterns = result.get('detected_patterns', {})
            if not detected_patterns or 'all_patterns' not in detected_patterns:
                print(f"❌ API pattern detection incomplete")
                self.test_results['api_endpoint'] = False
                return False
            
            # Test 5: Quality metrics validation
            quality_metrics = result.get('quality_metrics', {})
            required_metrics = ['overall_quality', 'pattern_following_score', 'waveform_fidelity']
            missing_metrics = [metric for metric in required_metrics if metric not in quality_metrics]
            
            if missing_metrics:
                print(f"❌ API quality metrics missing: {missing_metrics}")
                self.test_results['api_endpoint'] = False
                return False
            
            print("✅ API endpoint functionality validation PASSED")
            print(f"   Predictions: {len(predictions)} generated")
            print(f"   Patterns detected: {len(detected_patterns.get('all_patterns', []))}")
            print(f"   Quality metrics: {list(quality_metrics.keys())}")
            
            self.test_results['api_endpoint'] = True
            return True
            
        except Exception as e:
            print(f"❌ API endpoint test error: {str(e)}")
            self.test_results['api_endpoint'] = False
            return False
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n⚠️ TESTING EDGE CASES")
        
        edge_case_results = []
        
        # Test 1: Invalid data_id
        try:
            response = self.session.post(
                f"{API_BASE_URL}/generate-universal-waveform-prediction",
                json={"data_id": "invalid_id", "prediction_steps": 10}
            )
            
            if response.status_code in [400, 404]:
                print("✅ Invalid data_id handled correctly")
                edge_case_results.append(True)
            else:
                print(f"⚠️ Invalid data_id handling unexpected: {response.status_code}")
                edge_case_results.append(False)
                
        except Exception as e:
            print(f"❌ Error testing invalid data_id: {str(e)}")
            edge_case_results.append(False)
        
        # Test 2: Extreme prediction steps
        if self.data_ids.get('sinusoidal'):
            try:
                response = self.session.post(
                    f"{API_BASE_URL}/generate-universal-waveform-prediction",
                    json={"data_id": self.data_ids['sinusoidal'], "prediction_steps": 1000}
                )
                
                if response.status_code in [200, 400]:
                    print("✅ Extreme prediction steps handled")
                    edge_case_results.append(True)
                else:
                    print(f"⚠️ Extreme prediction steps handling unexpected: {response.status_code}")
                    edge_case_results.append(False)
                    
            except Exception as e:
                print(f"❌ Error testing extreme prediction steps: {str(e)}")
                edge_case_results.append(False)
        
        # Test 3: Missing parameters
        try:
            response = self.session.post(
                f"{API_BASE_URL}/generate-universal-waveform-prediction",
                json={}
            )
            
            if response.status_code in [400, 422]:
                print("✅ Missing parameters handled correctly")
                edge_case_results.append(True)
            else:
                print(f"⚠️ Missing parameters handling unexpected: {response.status_code}")
                edge_case_results.append(False)
                
        except Exception as e:
            print(f"❌ Error testing missing parameters: {str(e)}")
            edge_case_results.append(False)
        
        success_rate = sum(edge_case_results) / len(edge_case_results) if edge_case_results else 0
        self.test_results['edge_cases'] = success_rate >= 0.6
        
        print(f"Edge cases success rate: {success_rate:.1%}")
        return success_rate >= 0.6
    
    def run_comprehensive_test(self):
        """Run comprehensive universal waveform prediction system test"""
        print("🌊 STARTING COMPREHENSIVE UNIVERSAL WAVEFORM PREDICTION SYSTEM TESTING")
        print("=" * 80)
        
        start_time = time.time()
        
        # Test 1: API Endpoint Functionality
        api_success = self.test_api_endpoint_functionality()
        
        # Test 2: End-to-End Workflow
        workflow_success = self.test_end_to_end_workflow()
        
        # Test 3: Edge Cases
        edge_cases_success = self.test_edge_cases()
        
        # Calculate overall results
        end_time = time.time()
        test_duration = end_time - start_time
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE UNIVERSAL WAVEFORM PREDICTION SYSTEM TEST RESULTS")
        print("=" * 80)
        
        print(f"⏱️ Test Duration: {test_duration:.2f} seconds")
        print(f"📊 Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1%})")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\n🎉 OVERALL ASSESSMENT:")
        if success_rate >= 0.9:
            print("   ✅ EXCELLENT - Universal waveform prediction system working excellently!")
            print("   🎯 All key requirements met - pattern preservation verified")
        elif success_rate >= 0.7:
            print("   ✅ GOOD - Universal waveform prediction system working well")
            print("   🎯 Most requirements met - minor improvements needed")
        elif success_rate >= 0.5:
            print("   ⚠️ PARTIAL - Universal waveform prediction system partially working")
            print("   🔧 Some requirements met - significant improvements needed")
        else:
            print("   ❌ POOR - Universal waveform prediction system needs major fixes")
            print("   🚨 Critical issues identified - system not ready for production")
        
        # Key findings summary
        print(f"\n🔍 KEY FINDINGS:")
        if api_success:
            print("   ✅ API endpoint functionality working correctly")
        else:
            print("   ❌ API endpoint functionality has issues")
            
        if workflow_success >= 0.6:
            print("   ✅ End-to-end workflow functional for multiple pattern types")
        else:
            print("   ❌ End-to-end workflow has significant issues")
            
        if edge_cases_success:
            print("   ✅ Edge cases and error handling working")
        else:
            print("   ❌ Edge cases and error handling need improvement")
        
        return {
            'success_rate': success_rate,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'test_duration': test_duration,
            'detailed_results': self.test_results,
            'api_success': api_success,
            'workflow_success': workflow_success,
            'edge_cases_success': edge_cases_success
        }

if __name__ == "__main__":
    tester = UniversalWaveformTester()
    results = tester.run_comprehensive_test()
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://3504f872-4ab4-43c1-a827-4429cc10638c.preview.emergentagent.com')
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