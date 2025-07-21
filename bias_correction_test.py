#!/usr/bin/env python3
"""
Bias Correction Algorithm Testing
Specifically tests the bias correction methods mentioned in the review request
"""

import requests
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
from pathlib import Path
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://f576d419-9655-44df-95ef-dbabc9baf3ad.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing bias correction algorithms at: {API_BASE_URL}")

class BiasCorrectTester:
    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False
        self.test_results = {}
        
    def test_bias_correction_with_extreme_data(self):
        """Test bias correction with data that would normally cause downward bias"""
        print("\n=== Testing Bias Correction with Extreme Data ===")
        
        try:
            # Create data with strong downward trend that should be corrected
            timestamps = [datetime.now() - timedelta(hours=50-i) for i in range(50)]
            
            # Create strongly declining pH data
            base_values = np.linspace(7.8, 6.2, 50)  # Strong downward trend
            noise = np.random.normal(0, 0.05, 50)
            ph_values = base_values + noise
            ph_values = np.clip(ph_values, 6.0, 8.0)
            
            df = pd.DataFrame({'timestamp': timestamps, 'ph_value': ph_values})
            
            print(f"Created extreme declining data: {ph_values[0]:.2f} → {ph_values[-1]:.2f}")
            print(f"Historical trend slope: {np.polyfit(range(50), ph_values, 1)[0]:.6f}")
            
            # Upload and test
            csv_content = df.to_csv(index=False)
            files = {'file': ('extreme_decline_ph.csv', csv_content, 'text/csv')}
            
            upload_response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if upload_response.status_code == 200:
                data_id = upload_response.json().get('data_id')
                
                # Test advanced prediction (should apply bias correction)
                pred_response = self.session.post(
                    f"{API_BASE_URL}/advanced-prediction",
                    params={"data_id": data_id, "steps": 20}
                )
                
                if pred_response.status_code == 200:
                    predictions = pred_response.json().get('predictions', [])
                    
                    if predictions:
                        pred_slope = np.polyfit(range(len(predictions)), predictions, 1)[0]
                        pred_mean = np.mean(predictions)
                        
                        print(f"Prediction trend slope: {pred_slope:.6f}")
                        print(f"Prediction mean: {pred_mean:.3f}")
                        
                        # Check if bias correction worked
                        if pred_slope > -0.005:  # Much less negative than input
                            print("✅ Bias correction working - prevented extreme downward trend")
                            self.test_results['extreme_bias_correction'] = True
                        else:
                            print(f"❌ Bias correction failed - still strong downward trend: {pred_slope:.6f}")
                            self.test_results['extreme_bias_correction'] = False
                            
                        # Check if predictions stay in reasonable range
                        if all(6.0 <= p <= 8.0 for p in predictions):
                            print("✅ Predictions maintained realistic pH range")
                        else:
                            print("❌ Predictions went outside realistic range")
                            
                    else:
                        print("❌ No predictions generated")
                        self.test_results['extreme_bias_correction'] = False
                else:
                    print(f"❌ Prediction failed: {pred_response.status_code}")
                    self.test_results['extreme_bias_correction'] = False
            else:
                print(f"❌ Upload failed: {upload_response.status_code}")
                self.test_results['extreme_bias_correction'] = False
                
        except Exception as e:
            print(f"❌ Extreme bias correction test error: {str(e)}")
            self.test_results['extreme_bias_correction'] = False
    
    def test_mean_reversion_capability(self):
        """Test if predictions revert to historical mean over time"""
        print("\n=== Testing Mean Reversion Capability ===")
        
        try:
            # Create data with outlier at the end
            timestamps = [datetime.now() - timedelta(hours=30-i) for i in range(30)]
            
            # Normal pH around 7.2, but last few values are outliers
            ph_values = np.full(30, 7.2) + np.random.normal(0, 0.1, 30)
            ph_values[-3:] = [6.0, 5.8, 5.9]  # Extreme low outliers at end
            ph_values = np.clip(ph_values, 5.5, 8.5)
            
            df = pd.DataFrame({'timestamp': timestamps, 'ph_value': ph_values})
            
            historical_mean = np.mean(ph_values[:-3])  # Mean without outliers
            recent_mean = np.mean(ph_values[-3:])      # Mean of outliers
            
            print(f"Historical mean (without outliers): {historical_mean:.3f}")
            print(f"Recent mean (outliers): {recent_mean:.3f}")
            
            # Upload and test
            csv_content = df.to_csv(index=False)
            files = {'file': ('outlier_ph.csv', csv_content, 'text/csv')}
            
            upload_response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if upload_response.status_code == 200:
                data_id = upload_response.json().get('data_id')
                
                # Test prediction - should revert toward historical mean
                pred_response = self.session.post(
                    f"{API_BASE_URL}/advanced-prediction",
                    params={"data_id": data_id, "steps": 15}
                )
                
                if pred_response.status_code == 200:
                    predictions = pred_response.json().get('predictions', [])
                    
                    if predictions:
                        pred_mean = np.mean(predictions)
                        
                        # Check if predictions move toward historical mean
                        distance_to_historical = abs(pred_mean - historical_mean)
                        distance_to_recent = abs(pred_mean - recent_mean)
                        
                        print(f"Prediction mean: {pred_mean:.3f}")
                        print(f"Distance to historical mean: {distance_to_historical:.3f}")
                        print(f"Distance to recent outliers: {distance_to_recent:.3f}")
                        
                        if distance_to_historical < distance_to_recent:
                            print("✅ Mean reversion working - predictions closer to historical mean")
                            self.test_results['mean_reversion'] = True
                        else:
                            print("❌ Mean reversion not working - predictions follow outliers")
                            self.test_results['mean_reversion'] = False
                            
                    else:
                        print("❌ No predictions generated")
                        self.test_results['mean_reversion'] = False
                else:
                    print(f"❌ Prediction failed: {pred_response.status_code}")
                    self.test_results['mean_reversion'] = False
            else:
                print(f"❌ Upload failed: {upload_response.status_code}")
                self.test_results['mean_reversion'] = False
                
        except Exception as e:
            print(f"❌ Mean reversion test error: {str(e)}")
            self.test_results['mean_reversion'] = False
    
    def test_bounds_checking_algorithm(self):
        """Test enhanced bounds checking to prevent unrealistic values"""
        print("\n=== Testing Enhanced Bounds Checking ===")
        
        try:
            # Create data that might lead to out-of-bounds predictions
            timestamps = [datetime.now() - timedelta(hours=25-i) for i in range(25)]
            
            # Create data with extreme values that might cause extrapolation issues
            ph_values = [7.0] * 20 + [8.5, 8.7, 8.9, 9.1, 9.3]  # Extreme upward trend at end
            ph_values = np.array(ph_values)
            
            df = pd.DataFrame({'timestamp': timestamps, 'ph_value': ph_values})
            
            print(f"Data ends with extreme values: {ph_values[-5:]}")
            
            # Upload and test
            csv_content = df.to_csv(index=False)
            files = {'file': ('extreme_values_ph.csv', csv_content, 'text/csv')}
            
            upload_response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if upload_response.status_code == 200:
                data_id = upload_response.json().get('data_id')
                
                # Test prediction - should be bounded
                pred_response = self.session.post(
                    f"{API_BASE_URL}/advanced-prediction",
                    params={"data_id": data_id, "steps": 20}
                )
                
                if pred_response.status_code == 200:
                    predictions = pred_response.json().get('predictions', [])
                    
                    if predictions:
                        min_pred = min(predictions)
                        max_pred = max(predictions)
                        
                        print(f"Prediction range: {min_pred:.3f} to {max_pred:.3f}")
                        
                        # Check if bounds checking worked
                        if 5.0 <= min_pred <= 9.0 and 5.0 <= max_pred <= 9.0:
                            print("✅ Enhanced bounds checking working - predictions in reasonable range")
                            self.test_results['bounds_checking'] = True
                        else:
                            print(f"❌ Bounds checking failed - predictions outside reasonable range")
                            self.test_results['bounds_checking'] = False
                            
                        # Check for realistic pH range (stricter)
                        if all(5.5 <= p <= 8.5 for p in predictions):
                            print("✅ All predictions in realistic pH range (5.5-8.5)")
                        else:
                            out_of_range = [p for p in predictions if not (5.5 <= p <= 8.5)]
                            print(f"⚠️  {len(out_of_range)} predictions outside realistic range: {out_of_range[:3]}...")
                            
                    else:
                        print("❌ No predictions generated")
                        self.test_results['bounds_checking'] = False
                else:
                    print(f"❌ Prediction failed: {pred_response.status_code}")
                    self.test_results['bounds_checking'] = False
            else:
                print(f"❌ Upload failed: {upload_response.status_code}")
                self.test_results['bounds_checking'] = False
                
        except Exception as e:
            print(f"❌ Bounds checking test error: {str(e)}")
            self.test_results['bounds_checking'] = False
    
    def test_variability_preservation(self):
        """Test if predictions maintain natural variability from historical data"""
        print("\n=== Testing Variability Preservation ===")
        
        try:
            # Create data with specific variability characteristics
            timestamps = [datetime.now() - timedelta(hours=40-i) for i in range(40)]
            
            # Create data with known variability
            base_ph = 7.3
            variability = 0.15  # Standard deviation
            ph_values = np.random.normal(base_ph, variability, 40)
            ph_values = np.clip(ph_values, 6.0, 8.0)
            
            df = pd.DataFrame({'timestamp': timestamps, 'ph_value': ph_values})
            
            historical_std = np.std(ph_values)
            print(f"Historical variability (std): {historical_std:.4f}")
            
            # Upload and test
            csv_content = df.to_csv(index=False)
            files = {'file': ('variable_ph.csv', csv_content, 'text/csv')}
            
            upload_response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if upload_response.status_code == 200:
                data_id = upload_response.json().get('data_id')
                
                # Test prediction - should preserve variability
                pred_response = self.session.post(
                    f"{API_BASE_URL}/advanced-prediction",
                    params={"data_id": data_id, "steps": 25}
                )
                
                if pred_response.status_code == 200:
                    predictions = pred_response.json().get('predictions', [])
                    
                    if predictions:
                        pred_std = np.std(predictions)
                        variability_ratio = pred_std / historical_std
                        
                        print(f"Prediction variability (std): {pred_std:.4f}")
                        print(f"Variability preservation ratio: {variability_ratio:.3f}")
                        
                        # Check if variability is reasonably preserved
                        if 0.3 <= variability_ratio <= 2.0:  # Allow some change but not extreme
                            print("✅ Variability reasonably preserved")
                            self.test_results['variability_preservation'] = True
                        else:
                            print(f"❌ Variability not preserved - ratio: {variability_ratio:.3f}")
                            self.test_results['variability_preservation'] = False
                            
                        # Check for minimum variability (not too flat)
                        if pred_std >= 0.02:
                            print("✅ Predictions show sufficient variability")
                        else:
                            print(f"❌ Predictions too flat - std: {pred_std:.6f}")
                            
                    else:
                        print("❌ No predictions generated")
                        self.test_results['variability_preservation'] = False
                else:
                    print(f"❌ Prediction failed: {pred_response.status_code}")
                    self.test_results['variability_preservation'] = False
            else:
                print(f"❌ Upload failed: {upload_response.status_code}")
                self.test_results['variability_preservation'] = False
                
        except Exception as e:
            print(f"❌ Variability preservation test error: {str(e)}")
            self.test_results['variability_preservation'] = False
    
    def run_all_tests(self):
        """Run all bias correction algorithm tests"""
        print("🔧 Starting Bias Correction Algorithm Testing")
        print("=" * 60)
        
        self.test_bias_correction_with_extreme_data()
        self.test_mean_reversion_capability()
        self.test_bounds_checking_algorithm()
        self.test_variability_preservation()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 BIAS CORRECTION ALGORITHM TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        # Critical assessment
        print("\n🔍 BIAS CORRECTION ASSESSMENT:")
        
        if passed_tests == total_tests:
            print("✅ ALL BIAS CORRECTION ALGORITHMS WORKING CORRECTLY")
            print("   The enhanced bias correction system is functioning as designed")
        elif passed_tests >= total_tests * 0.75:
            print("⚠️  MOST BIAS CORRECTION ALGORITHMS WORKING")
            print("   Minor improvements needed in some areas")
        else:
            print("❌ SIGNIFICANT BIAS CORRECTION ISSUES DETECTED")
            print("   Major improvements needed in bias correction algorithms")
        
        return self.test_results

if __name__ == "__main__":
    tester = BiasCorrectTester()
    results = tester.run_all_tests()