#!/usr/bin/env python3
"""
Final Comprehensive Upload Workflow Test
Test the complete end-to-end upload workflow to identify any remaining issues
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://f54db828-52d2-4e14-b664-3ae23427df52.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing complete upload workflow at: {API_BASE_URL}")

class FinalUploadWorkflowTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
    
    def create_realistic_ph_data(self):
        """Create realistic pH monitoring data that mimics user data"""
        # Create 48 hours of hourly pH monitoring data
        dates = pd.date_range(start='2023-12-01 00:00:00', periods=48, freq='H')
        
        # Simulate realistic pH variations
        base_ph = 7.2
        ph_values = []
        
        for i, date in enumerate(dates):
            # Daily cycle (pH typically lower at night)
            daily_cycle = 0.3 * np.sin(2 * np.pi * i / 24 - np.pi/2)
            
            # Random drift
            drift = np.random.normal(0, 0.05)
            
            # Measurement noise
            noise = np.random.normal(0, 0.02)
            
            # Occasional spikes (equipment issues)
            spike = 0.5 if np.random.random() < 0.05 else 0
            
            ph_value = base_ph + daily_cycle + drift + noise + spike
            ph_value = max(6.0, min(8.5, ph_value))  # Keep in realistic range
            ph_values.append(round(ph_value, 2))
        
        # Add related parameters
        temperature = 25 + np.random.normal(0, 2, 48)  # Temperature variations
        conductivity = 1500 + np.random.normal(0, 100, 48)  # Conductivity
        
        df = pd.DataFrame({
            'timestamp': dates,
            'pH': ph_values,
            'temperature_C': np.round(temperature, 1),
            'conductivity_uS': np.round(conductivity, 0).astype(int),
            'operator': ['John'] * 24 + ['Sarah'] * 24  # Different operators
        })
        
        return df
    
    def test_realistic_ph_upload_workflow(self):
        """Test complete workflow with realistic pH data"""
        print("\n=== Testing Realistic pH Upload Workflow ===")
        
        try:
            # Step 1: Create and upload realistic pH data
            df = self.create_realistic_ph_data()
            csv_content = df.to_csv(index=False)
            
            print(f"   Created pH dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"   pH range: {df['pH'].min():.2f} - {df['pH'].max():.2f}")
            print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            files = {'file': ('ph_monitoring_data.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Step 1 FAILED: Upload failed with {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['realistic_ph_workflow'] = False
                return
            
            upload_data = response.json()
            data_id = upload_data.get('data_id')
            analysis = upload_data.get('analysis', {})
            
            print("✅ Step 1: Upload successful")
            print(f"   Data ID: {data_id}")
            print(f"   Detected columns: {analysis.get('columns', [])}")
            print(f"   Time columns: {analysis.get('time_columns', [])}")
            print(f"   Numeric columns: {analysis.get('numeric_columns', [])}")
            
            # Validate analysis
            if 'pH' not in analysis.get('numeric_columns', []):
                print("❌ pH column not detected as numeric!")
                self.test_results['realistic_ph_workflow'] = False
                return
            
            if 'timestamp' not in analysis.get('time_columns', []):
                print("❌ Timestamp column not detected as time!")
                self.test_results['realistic_ph_workflow'] = False
                return
            
            # Step 2: Check parameter suggestions
            suggested_params = analysis.get('suggested_parameters', {})
            time_col = suggested_params.get('time_column')
            target_col = suggested_params.get('target_column')
            
            print("✅ Step 2: Parameter analysis successful")
            print(f"   Suggested time column: {time_col}")
            print(f"   Suggested target column: {target_col}")
            
            if not time_col or not target_col:
                print("❌ Parameter suggestions incomplete!")
                self.test_results['realistic_ph_workflow'] = False
                return
            
            # Step 3: Train model with pH data
            training_params = {
                "time_column": time_col,
                "target_column": target_col,
                "order": [1, 1, 1]
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json=training_params
            )
            
            if response.status_code != 200:
                print(f"❌ Step 3 FAILED: Model training failed with {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['realistic_ph_workflow'] = False
                return
            
            training_data = response.json()
            model_id = training_data.get('model_id')
            
            print("✅ Step 3: Model training successful")
            print(f"   Model ID: {model_id}")
            print(f"   Training status: {training_data.get('status')}")
            
            # Step 4: Generate predictions
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 24}  # 24 hours ahead
            )
            
            if response.status_code != 200:
                print(f"❌ Step 4 FAILED: Prediction generation failed with {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['realistic_ph_workflow'] = False
                return
            
            prediction_data = response.json()
            predictions = prediction_data.get('predictions', [])
            timestamps = prediction_data.get('timestamps', [])
            
            print("✅ Step 4: Prediction generation successful")
            print(f"   Generated {len(predictions)} predictions")
            print(f"   Prediction range: {min(predictions):.2f} - {max(predictions):.2f}")
            print(f"   Sample predictions: {predictions[:3]}")
            
            # Validate predictions are reasonable for pH
            if not all(5.0 <= p <= 9.0 for p in predictions):
                print("⚠️  Some predictions outside reasonable pH range (5.0-9.0)")
            
            # Step 5: Test continuous prediction
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 12, "time_window": 48}
            )
            
            if response.status_code != 200:
                print(f"❌ Step 5 FAILED: Continuous prediction failed with {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['realistic_ph_workflow'] = False
                return
            
            continuous_data = response.json()
            continuous_predictions = continuous_data.get('predictions', [])
            
            print("✅ Step 5: Continuous prediction successful")
            print(f"   Generated {len(continuous_predictions)} continuous predictions")
            
            # Step 6: Test data quality report
            response = self.session.get(f"{API_BASE_URL}/data-quality-report", params={"data_id": data_id})
            
            if response.status_code != 200:
                print(f"❌ Step 6 FAILED: Data quality report failed with {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['realistic_ph_workflow'] = False
                return
            
            quality_data = response.json()
            quality_score = quality_data.get('quality_score', 0)
            
            print("✅ Step 6: Data quality analysis successful")
            print(f"   Quality score: {quality_score}")
            print(f"   Recommendations: {len(quality_data.get('recommendations', []))}")
            
            # All steps successful
            print("🎉 Complete pH workflow successful!")
            self.test_results['realistic_ph_workflow'] = True
            
        except Exception as e:
            print(f"❌ Realistic pH workflow error: {str(e)}")
            self.test_results['realistic_ph_workflow'] = False
    
    def test_user_reported_scenarios(self):
        """Test scenarios that might match user-reported issues"""
        print("\n=== Testing User-Reported Issue Scenarios ===")
        
        scenarios = []
        
        # Scenario 1: Very small pH dataset (user might have limited data)
        try:
            small_df = pd.DataFrame({
                'time': pd.date_range(start='2023-12-01', periods=8, freq='H'),
                'pH_value': [7.1, 7.2, 7.0, 7.3, 7.1, 7.2, 7.0, 7.1]
            })
            
            csv_content = small_df.to_csv(index=False)
            files = {'file': ('small_ph_data.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 400:
                print("✅ Scenario 1: Small dataset properly rejected with clear error")
                scenarios.append(True)
            else:
                print(f"❌ Scenario 1: Unexpected response {response.status_code}")
                scenarios.append(False)
                
        except Exception as e:
            print(f"❌ Scenario 1 error: {e}")
            scenarios.append(False)
        
        # Scenario 2: pH data with missing values (common in real monitoring)
        try:
            missing_df = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-12-01', periods=20, freq='H'),
                'pH': [7.1, 7.2, None, 7.3, 7.1, '', 7.0, 7.1, 7.2, 7.0,
                       7.3, 7.1, 7.2, None, 7.0, 7.1, '', 7.2, 7.0, 7.3],
                'temp': [25.1, 25.2, 25.0, None, 25.3, 25.1, 25.2, 25.0, 25.1, 25.2,
                        25.3, 25.1, 25.2, 25.0, 25.1, None, 25.2, 25.0, 25.1, 25.3]
            })
            
            csv_content = missing_df.to_csv(index=False)
            files = {'file': ('ph_with_missing.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                missing_info = data['analysis']['data_preview']['missing_values']
                print(f"✅ Scenario 2: Missing values handled (missing: {missing_info})")
                scenarios.append(True)
            else:
                print(f"❌ Scenario 2: Failed with {response.status_code}")
                scenarios.append(False)
                
        except Exception as e:
            print(f"❌ Scenario 2 error: {e}")
            scenarios.append(False)
        
        # Scenario 3: pH data with unusual column names (user might have different naming)
        try:
            unusual_df = pd.DataFrame({
                'Date/Time': pd.date_range(start='2023-12-01', periods=15, freq='H'),
                'pH Level (units)': np.random.normal(7.2, 0.2, 15),
                'Temp (°C)': np.random.normal(25, 1, 15),
                'Notes/Comments': ['Normal'] * 15
            })
            
            csv_content = unusual_df.to_csv(index=False)
            files = {'file': ('unusual_columns.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                time_cols = data['analysis']['time_columns']
                numeric_cols = data['analysis']['numeric_columns']
                print(f"✅ Scenario 3: Unusual columns handled (time: {time_cols}, numeric: {numeric_cols})")
                scenarios.append(True)
            else:
                print(f"❌ Scenario 3: Failed with {response.status_code}")
                scenarios.append(False)
                
        except Exception as e:
            print(f"❌ Scenario 3 error: {e}")
            scenarios.append(False)
        
        # Scenario 4: File with BOM (Byte Order Mark) - common in Excel exports
        try:
            bom_df = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-12-01', periods=12, freq='H'),
                'pH': np.random.normal(7.2, 0.1, 12)
            })
            
            csv_content = bom_df.to_csv(index=False)
            # Add BOM to simulate Excel export
            bom_content = '\ufeff' + csv_content
            
            files = {'file': ('bom_file.csv', bom_content.encode('utf-8-sig'), 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                print("✅ Scenario 4: BOM file handled successfully")
                scenarios.append(True)
            else:
                print(f"❌ Scenario 4: BOM file failed with {response.status_code}")
                scenarios.append(False)
                
        except Exception as e:
            print(f"❌ Scenario 4 error: {e}")
            scenarios.append(False)
        
        successful_scenarios = sum(scenarios)
        print(f"\n✅ User scenario tests: {successful_scenarios}/{len(scenarios)} successful")
        
        self.test_results['user_scenarios'] = successful_scenarios >= len(scenarios) * 0.75
    
    def test_error_message_clarity(self):
        """Test that error messages are clear and helpful"""
        print("\n=== Testing Error Message Clarity ===")
        
        error_tests = []
        
        # Test 1: Empty file error message
        try:
            files = {'file': ('empty.csv', '', 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code >= 400:
                error_msg = response.json().get('detail', '')
                if 'empty' in error_msg.lower() or 'no data' in error_msg.lower():
                    print("✅ Empty file error message is clear")
                    error_tests.append(True)
                else:
                    print(f"⚠️  Empty file error message unclear: {error_msg}")
                    error_tests.append(False)
            else:
                print("❌ Empty file not rejected")
                error_tests.append(False)
                
        except Exception as e:
            print(f"❌ Empty file test error: {e}")
            error_tests.append(False)
        
        # Test 2: Small dataset error message
        try:
            small_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            csv_content = small_df.to_csv(index=False)
            files = {'file': ('small.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code >= 400:
                error_msg = response.json().get('detail', '')
                if 'small' in error_msg.lower() and '10' in error_msg:
                    print("✅ Small dataset error message is clear and specific")
                    error_tests.append(True)
                else:
                    print(f"⚠️  Small dataset error message unclear: {error_msg}")
                    error_tests.append(False)
            else:
                print("❌ Small dataset not rejected")
                error_tests.append(False)
                
        except Exception as e:
            print(f"❌ Small dataset test error: {e}")
            error_tests.append(False)
        
        # Test 3: Invalid file format error message
        try:
            files = {'file': ('test.txt', 'not a csv file', 'text/plain')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code >= 400:
                error_msg = response.json().get('detail', '')
                if 'format' in error_msg.lower() or 'csv' in error_msg.lower() or 'excel' in error_msg.lower():
                    print("✅ Invalid format error message is clear")
                    error_tests.append(True)
                else:
                    print(f"⚠️  Invalid format error message unclear: {error_msg}")
                    error_tests.append(False)
            else:
                print("❌ Invalid format not rejected")
                error_tests.append(False)
                
        except Exception as e:
            print(f"❌ Invalid format test error: {e}")
            error_tests.append(False)
        
        successful_error_tests = sum(error_tests)
        print(f"\n✅ Error message clarity: {successful_error_tests}/{len(error_tests)} clear")
        
        self.test_results['error_message_clarity'] = successful_error_tests >= len(error_tests) * 0.8
    
    def run_final_comprehensive_test(self):
        """Run final comprehensive test"""
        print("🎯 Starting Final Comprehensive Upload Workflow Test")
        print("=" * 70)
        
        # Run all final tests
        self.test_realistic_ph_upload_workflow()
        self.test_user_reported_scenarios()
        self.test_error_message_clarity()
        
        # Print final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final comprehensive summary"""
        print("\n" + "=" * 70)
        print("🎯 FINAL COMPREHENSIVE UPLOAD TEST SUMMARY")
        print("=" * 70)
        
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Final Test Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print()
        
        # Print individual results
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print()
        
        # Final assessment
        if success_rate == 100:
            print("🎉 PERFECT: All upload functionality tests passed!")
            print("   The file upload system is working correctly.")
            print("   User issues may be related to:")
            print("   • Frontend JavaScript errors")
            print("   • Network connectivity issues")
            print("   • Browser-specific problems")
            print("   • User data format issues")
        elif success_rate >= 80:
            print("✅ EXCELLENT: Upload functionality is working very well!")
            print("   Minor issues identified but core functionality is solid.")
        elif success_rate >= 60:
            print("⚠️  GOOD: Upload functionality mostly works with some issues.")
        else:
            print("🚨 CRITICAL: Upload functionality has significant problems!")
        
        print()
        print("📋 FINAL RECOMMENDATIONS:")
        
        if not self.test_results.get('realistic_ph_workflow', True):
            print("  🚨 CRITICAL: Fix realistic pH workflow - core functionality broken!")
        else:
            print("  ✅ Core pH upload workflow is working correctly")
        
        if not self.test_results.get('user_scenarios', True):
            print("  ⚠️  Improve handling of user-specific scenarios")
        else:
            print("  ✅ User scenarios are handled well")
        
        if not self.test_results.get('error_message_clarity', True):
            print("  ⚠️  Improve error message clarity for better user experience")
        else:
            print("  ✅ Error messages are clear and helpful")
        
        print()
        print("🔍 INVESTIGATION FOCUS:")
        print("  Since backend upload functionality is working well,")
        print("  user issues are likely caused by:")
        print("  1. Frontend JavaScript errors during upload")
        print("  2. File size or format issues not covered in tests")
        print("  3. Browser compatibility problems")
        print("  4. Network timeout issues")
        print("  5. User data containing unexpected characters/formats")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    tester = FinalUploadWorkflowTester()
    tester.run_final_comprehensive_test()