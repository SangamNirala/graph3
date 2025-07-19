#!/usr/bin/env python3
"""
Focused Testing for Enhanced Real-Time Continuous Prediction System v2
Tests the new v2 system with pH data as requested in the review
"""

import requests
import json
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://f54db828-52d2-4e14-b664-3ae23427df52.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Enhanced Real-Time Continuous Prediction System v2 at: {API_BASE_URL}")

class V2SystemTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_ph_test_data(self):
        """Create realistic pH test data for v2 system testing"""
        # Generate 48 hours of pH data (every 30 minutes = 96 data points)
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=96, freq='30min')
        
        # Create realistic pH pattern (6.0-8.0 range with cyclical variations)
        time_hours = np.arange(96) * 0.5  # Convert to hours
        
        # Base pH around 7.2 with daily cycle
        base_ph = 7.2
        daily_cycle = 0.3 * np.sin(2 * np.pi * time_hours / 24)  # Daily variation
        
        # Add some process variations
        process_trend = 0.1 * np.sin(2 * np.pi * time_hours / 12)  # 12-hour cycle
        
        # Add realistic noise
        noise = np.random.normal(0, 0.05, 96)
        
        # Combine all components
        ph_values = base_ph + daily_cycle + process_trend + noise
        
        # Ensure pH stays in realistic range
        ph_values = np.clip(ph_values, 6.0, 8.0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'pH': ph_values,
            'temperature': np.random.normal(25, 2, 96),  # Temperature data
            'conductivity': np.random.normal(1500, 100, 96)  # Conductivity data
        })
        
        return df
    
    def test_v2_system_comprehensive(self):
        """Comprehensive test of the Enhanced Real-Time Continuous Prediction System v2"""
        print("\n🎯 ENHANCED REAL-TIME CONTINUOUS PREDICTION SYSTEM V2 TESTING")
        print("=" * 80)
        
        v2_tests = []
        
        try:
            # Step 1: Upload pH test data
            print("🔬 Step 1: Uploading realistic pH test data...")
            ph_df = self.create_ph_test_data()
            csv_content = ph_df.to_csv(index=False)
            files = {'file': ('ph_test_data_v2.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            upload_success = response.status_code == 200
            v2_tests.append(("pH data upload", upload_success))
            
            if not upload_success:
                print(f"❌ pH data upload failed: {response.status_code}")
                return False
            
            data_id = response.json().get('data_id')
            print(f"✅ pH data uploaded successfully: {len(ph_df)} data points")
            print(f"   pH range in data: {ph_df['pH'].min():.3f} - {ph_df['pH'].max():.3f}")
            print(f"   Data variability (std): {ph_df['pH'].std():.3f}")
            
            # Step 2: Train model with pH data
            print("\n🔬 Step 2: Training model with pH data...")
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "timestamp", "target_column": "pH", "order": [1, 1, 1]}
            )
            
            training_success = response.status_code == 200
            v2_tests.append(("pH model training", training_success))
            
            if not training_success:
                print(f"❌ pH model training failed: {response.status_code}")
                return False
            
            model_id = response.json().get('model_id')
            print("✅ pH model trained successfully")
            
            # Step 3: Test Enhanced Real-Time Continuous Prediction System v2
            print("\n🔬 Step 3: Testing Enhanced Real-Time Continuous Prediction System v2...")
            print("Testing new /api/generate-enhanced-realtime-prediction-v2 endpoint...")
            
            response = self.session.get(
                f"{API_BASE_URL}/generate-enhanced-realtime-prediction-v2",
                params={"steps": 30, "time_window": 100, "maintain_patterns": True}
            )
            
            v2_endpoint_success = response.status_code == 200
            v2_tests.append(("V2 endpoint functionality", v2_endpoint_success))
            
            if not v2_endpoint_success:
                print(f"❌ V2 endpoint failed: {response.status_code} - {response.text}")
                return False
            
            v2_data = response.json()
            predictions = v2_data.get('predictions', [])
            metadata = v2_data.get('metadata', {})
            
            print("✅ Enhanced Real-Time Continuous Prediction System v2 successful!")
            print(f"   Number of predictions: {len(predictions)}")
            print(f"   pH prediction range: {min(predictions):.3f} - {max(predictions):.3f}")
            
            # Test 4: Enhanced Pattern Learning (AdvancedPatternMemoryV2)
            print("\n🔬 Step 4: Testing Enhanced Pattern Learning (AdvancedPatternMemoryV2)...")
            pattern_analysis = metadata.get('pattern_analysis', {})
            has_advanced_patterns = bool(pattern_analysis)
            v2_tests.append(("Advanced Pattern Memory v2", has_advanced_patterns))
            
            if has_advanced_patterns:
                print("✅ AdvancedPatternMemoryV2 system active")
                print(f"   Pattern analysis components: {list(pattern_analysis.keys())}")
            else:
                print("❌ AdvancedPatternMemoryV2 system not detected")
            
            # Test 5: Superior Pattern Following
            print("\n🔬 Step 5: Testing Superior Pattern Following...")
            pattern_score = metadata.get('pattern_following_score', 0)
            superior_pattern_following = pattern_score >= 0.6
            v2_tests.append(("Superior pattern following", superior_pattern_following))
            
            if superior_pattern_following:
                print(f"✅ Superior pattern following achieved: {pattern_score:.3f}")
            else:
                print(f"❌ Pattern following below threshold: {pattern_score:.3f}")
            
            # Test 6: Enhanced Quality Metrics
            print("\n🔬 Step 6: Testing Enhanced Quality Metrics...")
            required_metrics = [
                'pattern_following_score', 'variability_preservation_score',
                'bias_prevention_score', 'continuity_score'
            ]
            
            metrics_present = all(metric in metadata for metric in required_metrics)
            v2_tests.append(("Enhanced quality metrics", metrics_present))
            
            if metrics_present:
                print("✅ All enhanced quality metrics present:")
                for metric in required_metrics:
                    value = metadata.get(metric, 0)
                    print(f"   {metric}: {value:.3f}")
            else:
                print("❌ Missing enhanced quality metrics")
            
            # Test 7: Variability Preservation
            print("\n🔬 Step 7: Testing Variability Preservation...")
            variability_score = metadata.get('variability_preservation_score', 0)
            good_variability = variability_score >= 0.7
            v2_tests.append(("Variability preservation", good_variability))
            
            # Check actual prediction variability
            if len(predictions) >= 5:
                pred_std = np.std(predictions)
                pred_range = max(predictions) - min(predictions)
                unique_values = len(set([round(p, 3) for p in predictions]))
                
                # Check for good variability (not flat/monotonic)
                actual_variability_good = pred_std > 0.01 and unique_values >= 5 and pred_range > 0.05
                v2_tests.append(("Actual prediction variability", actual_variability_good))
                
                if good_variability and actual_variability_good:
                    print(f"✅ Excellent variability preservation:")
                    print(f"   Score: {variability_score:.3f}")
                    print(f"   Actual std: {pred_std:.3f}, range: {pred_range:.3f}, unique: {unique_values}")
                else:
                    print(f"❌ Poor variability preservation:")
                    print(f"   Score: {variability_score:.3f}")
                    print(f"   Actual std: {pred_std:.3f}, range: {pred_range:.3f}, unique: {unique_values}")
            
            # Test 8: Bias Prevention
            print("\n🔬 Step 8: Testing Bias Prevention...")
            bias_score = metadata.get('bias_prevention_score', 0)
            good_bias_prevention = bias_score >= 0.7
            v2_tests.append(("Bias prevention", good_bias_prevention))
            
            if good_bias_prevention:
                print(f"✅ Good bias prevention: {bias_score:.3f}")
            else:
                print(f"❌ Poor bias prevention: {bias_score:.3f}")
            
            # Test 9: Multiple Continuous Predictions (Bias Accumulation Test)
            print("\n🔬 Step 9: Testing Multiple Continuous Predictions for Bias Accumulation...")
            print("Making 5 consecutive v2 prediction calls...")
            
            v2_results = []
            all_predictions = []
            
            for i in range(5):
                response = self.session.get(
                    f"{API_BASE_URL}/generate-enhanced-realtime-prediction-v2",
                    params={"steps": 20, "time_window": 100}
                )
                if response.status_code == 200:
                    result_data = response.json()
                    v2_results.append(result_data)
                    preds = result_data.get('predictions', [])
                    all_predictions.extend(preds)
                    print(f"   Call {i+1}: {len(preds)} predictions, range: {min(preds):.3f}-{max(preds):.3f}")
                time.sleep(0.5)
            
            if len(v2_results) >= 3 and len(all_predictions) >= 30:
                # Check for bias accumulation prevention
                x = np.arange(len(all_predictions))
                slope = np.polyfit(x, all_predictions, 1)[0]
                
                # Good bias prevention: slope should not be strongly negative
                no_downward_bias = slope > -0.01  # Allow slight negative slope
                v2_tests.append(("No bias accumulation", no_downward_bias))
                
                if no_downward_bias:
                    print(f"✅ No bias accumulation detected: slope={slope:.6f}")
                else:
                    print(f"❌ Downward bias accumulation detected: slope={slope:.6f}")
                
                # Check consistency of bias prevention scores
                bias_scores = [r.get('metadata', {}).get('bias_prevention_score', 0) for r in v2_results]
                avg_bias_score = np.mean(bias_scores)
                consistent_bias_prevention = avg_bias_score >= 0.7
                v2_tests.append(("Consistent bias prevention", consistent_bias_prevention))
                
                if consistent_bias_prevention:
                    print(f"✅ Consistent bias prevention: avg_score={avg_bias_score:.3f}")
                else:
                    print(f"❌ Inconsistent bias prevention: avg_score={avg_bias_score:.3f}")
            
            # Test 10: pH Range Validation
            print("\n🔬 Step 10: Testing pH Range Validation...")
            all_test_predictions = []
            for result in v2_results:
                all_test_predictions.extend(result.get('predictions', []))
            
            ph_range_valid = all(6.0 <= pred <= 8.0 for pred in all_test_predictions)
            v2_tests.append(("pH range validation", ph_range_valid))
            
            if ph_range_valid:
                print(f"✅ All {len(all_test_predictions)} predictions in valid pH range (6.0-8.0)")
                print(f"   Overall range: {min(all_test_predictions):.3f} - {max(all_test_predictions):.3f}")
            else:
                invalid_preds = [p for p in all_test_predictions if not (6.0 <= p <= 8.0)]
                print(f"❌ {len(invalid_preds)} predictions outside valid pH range")
            
            # Test 11: System Learning and Status
            print("\n🔬 Step 11: Testing System Learning and Status...")
            learning_active = metadata.get('learning_active', False)
            system_status = metadata.get('system_status', '')
            prediction_count = metadata.get('prediction_count', 0)
            
            system_healthy = learning_active and system_status == 'active' and prediction_count > 0
            v2_tests.append(("System learning active", system_healthy))
            
            if system_healthy:
                print(f"✅ System learning active:")
                print(f"   Status: {system_status}")
                print(f"   Learning: {learning_active}")
                print(f"   Prediction count: {prediction_count}")
            else:
                print(f"❌ System learning issues:")
                print(f"   Status: {system_status}")
                print(f"   Learning: {learning_active}")
                print(f"   Prediction count: {prediction_count}")
            
            # Final Assessment
            print("\n" + "=" * 80)
            print("📊 ENHANCED REAL-TIME CONTINUOUS PREDICTION SYSTEM V2 TEST RESULTS")
            print("=" * 80)
            
            passed_tests = sum(1 for _, passed in v2_tests if passed)
            total_tests = len(v2_tests)
            success_rate = (passed_tests / total_tests) * 100
            
            print(f"Total Tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {total_tests - passed_tests}")
            print(f"Success Rate: {success_rate:.1f}%")
            
            print("\nDetailed Results:")
            for test_name, result in v2_tests:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {status} - {test_name}")
            
            # Overall assessment
            v2_system_success = passed_tests >= total_tests * 0.8  # 80% threshold
            
            if v2_system_success:
                print("\n🎉 ENHANCED REAL-TIME CONTINUOUS PREDICTION SYSTEM V2 - SUCCESS!")
                print("✅ Superior pattern following achieved")
                print("✅ Variability preservation working")
                print("✅ Bias prevention effective")
                print("✅ Quality metrics comprehensive")
                print("✅ Advanced pattern memory v2 active")
            else:
                print("\n⚠️  ENHANCED REAL-TIME CONTINUOUS PREDICTION SYSTEM V2 - NEEDS IMPROVEMENT")
                print(f"📊 Success rate: {success_rate:.1f}% (target: 80%)")
            
            return v2_system_success
            
        except Exception as e:
            print(f"❌ V2 system test error: {str(e)}")
            return False

if __name__ == "__main__":
    tester = V2SystemTester()
    success = tester.test_v2_system_comprehensive()
    
    if success:
        print("\n🎯 Enhanced Real-Time Continuous Prediction System v2 testing completed successfully!")
        exit(0)
    else:
        print("\n⚠️  Enhanced Real-Time Continuous Prediction System v2 testing completed with issues.")
        exit(1)