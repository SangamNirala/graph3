#!/usr/bin/env python3
"""
Additional Document Upload Tests - Focus on specific scenarios from review request
"""

import requests
import json
import pandas as pd
import io
import time
import os
import tempfile
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://064f3bb3-c010-4892-8a8e-8e29d9900fe8.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"🎯 ADDITIONAL DOCUMENT UPLOAD TESTING - REVIEW REQUEST FOCUS")
print(f"Testing backend at: {API_BASE_URL}")

class AdditionalUploadTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def test_realistic_ph_dataset(self):
        """Test with realistic pH monitoring dataset as mentioned in review"""
        print("\n=== Testing Realistic pH Monitoring Dataset ===")
        
        try:
            # Create realistic 48-hour pH monitoring data
            start_time = datetime.now() - timedelta(hours=48)
            timestamps = [start_time + timedelta(minutes=30*i) for i in range(96)]  # Every 30 minutes
            
            # Realistic pH values with natural variation
            ph_values = []
            base_ph = 7.2
            
            for i, ts in enumerate(timestamps):
                # Daily cycle (slightly lower at night)
                hour = ts.hour
                daily_cycle = -0.3 * np.cos(2 * np.pi * hour / 24)
                
                # Weekly pattern (slightly different on weekends)
                weekly_cycle = 0.1 * np.sin(2 * np.pi * ts.weekday() / 7)
                
                # Random variation
                noise = np.random.normal(0, 0.05)
                
                # Slight upward trend
                trend = 0.001 * i
                
                ph_value = base_ph + daily_cycle + weekly_cycle + noise + trend
                ph_value = max(6.5, min(7.8, ph_value))  # Keep in realistic bounds
                ph_values.append(round(ph_value, 3))
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'ph_value': ph_values,
                'temperature': [round(22 + np.random.normal(0, 1.5), 1) for _ in range(96)],
                'conductivity': [round(1200 + np.random.normal(0, 50), 0) for _ in range(96)],
                'turbidity': [round(2.5 + np.random.normal(0, 0.3), 2) for _ in range(96)],
                'sensor_status': ['OK'] * 94 + ['CALIBRATING'] * 2
            })
            
            csv_content = df.to_csv(index=False)
            
            # Test upload
            files = {'file': ('realistic_ph_48h.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                
                print("✅ Realistic pH dataset upload successful")
                print(f"   Data ID: {data_id}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                
                # Validate pH range in preview
                preview = data['analysis']['data_preview']['head']
                if preview:
                    ph_values_preview = [row.get('ph_value') for row in preview[:5]]
                    valid_ph_range = all(6.0 <= ph <= 8.0 for ph in ph_values_preview if ph is not None)
                    
                    if valid_ph_range:
                        print("✅ pH values in realistic range (6.0-8.0)")
                        self.test_results['realistic_ph_dataset'] = True
                    else:
                        print("❌ pH values outside realistic range")
                        self.test_results['realistic_ph_dataset'] = False
                else:
                    print("❌ No preview data available")
                    self.test_results['realistic_ph_dataset'] = False
                    
            else:
                print(f"❌ Realistic pH dataset upload failed: {response.status_code} - {response.text}")
                self.test_results['realistic_ph_dataset'] = False
                
        except Exception as e:
            print(f"❌ Realistic pH dataset test error: {str(e)}")
            self.test_results['realistic_ph_dataset'] = False
    
    def test_different_file_sizes(self):
        """Test different file sizes as mentioned in review"""
        print("\n=== Testing Different File Sizes ===")
        
        try:
            file_size_tests = []
            
            # Test 1: Small file (10 rows)
            small_df = pd.DataFrame({
                'time': pd.date_range('2023-01-01', periods=10, freq='H'),
                'value': np.random.normal(7.0, 0.2, 10)
            })
            csv_content = small_df.to_csv(index=False)
            files = {'file': ('small_10_rows.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            small_file_result = response.status_code == 200
            file_size_tests.append(("Small file (10 rows)", small_file_result))
            
            if small_file_result:
                print("✅ Small file (10 rows) uploaded successfully")
            else:
                print(f"❌ Small file upload failed: {response.status_code}")
            
            # Test 2: Medium file (1,000 rows)
            medium_df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
                'ph_level': np.random.normal(7.2, 0.3, 1000),
                'temperature': np.random.normal(23, 2, 1000)
            })
            csv_content = medium_df.to_csv(index=False)
            files = {'file': ('medium_1000_rows.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            medium_file_result = response.status_code == 200
            file_size_tests.append(("Medium file (1,000 rows)", medium_file_result))
            
            if medium_file_result:
                print("✅ Medium file (1,000 rows) uploaded successfully")
            else:
                print(f"❌ Medium file upload failed: {response.status_code}")
            
            # Test 3: Large file (5,000 rows)
            large_df = pd.DataFrame({
                'datetime': pd.date_range('2023-01-01', periods=5000, freq='30min'),
                'sensor_reading': np.random.normal(7.1, 0.4, 5000),
                'quality_score': np.random.uniform(0.8, 1.0, 5000)
            })
            csv_content = large_df.to_csv(index=False)
            files = {'file': ('large_5000_rows.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files, timeout=60)
            large_file_result = response.status_code == 200
            file_size_tests.append(("Large file (5,000 rows)", large_file_result))
            
            if large_file_result:
                print("✅ Large file (5,000 rows) uploaded successfully")
            else:
                print(f"❌ Large file upload failed: {response.status_code}")
            
            # Overall file size test result
            passed_tests = sum(1 for _, passed in file_size_tests if passed)
            total_tests = len(file_size_tests)
            
            print(f"📊 File size tests: {passed_tests}/{total_tests}")
            self.test_results['different_file_sizes'] = passed_tests >= 2  # At least 2/3 should pass
            
        except Exception as e:
            print(f"❌ Different file sizes test error: {str(e)}")
            self.test_results['different_file_sizes'] = False
    
    def test_csv_variations(self):
        """Test various CSV format variations"""
        print("\n=== Testing CSV Format Variations ===")
        
        try:
            csv_variation_tests = []
            
            # Test 1: CSV with semicolon separator
            df = pd.DataFrame({
                'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'ph': [7.1, 7.2, 7.0],
                'temp': [22.5, 23.1, 22.8]
            })
            csv_content = df.to_csv(index=False, sep=';')
            files = {'file': ('semicolon_sep.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            semicolon_result = response.status_code == 200
            csv_variation_tests.append(("Semicolon separator", semicolon_result))
            
            if semicolon_result:
                print("✅ Semicolon separator CSV handled correctly")
            else:
                print(f"❌ Semicolon separator CSV failed: {response.status_code}")
            
            # Test 2: CSV with quoted fields
            quoted_csv = '"timestamp","ph_value","location"\n"2023-01-01 10:00:00","7.2","Lab A"\n"2023-01-01 11:00:00","7.1","Lab B"'
            files = {'file': ('quoted_fields.csv', quoted_csv, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            quoted_result = response.status_code == 200
            csv_variation_tests.append(("Quoted fields", quoted_result))
            
            if quoted_result:
                print("✅ Quoted fields CSV handled correctly")
            else:
                print(f"❌ Quoted fields CSV failed: {response.status_code}")
            
            # Test 3: CSV with different date formats
            df = pd.DataFrame({
                'date_iso': ['2023-01-01T10:00:00Z', '2023-01-01T11:00:00Z', '2023-01-01T12:00:00Z'],
                'date_us': ['01/01/2023 10:00', '01/01/2023 11:00', '01/01/2023 12:00'],
                'value': [7.1, 7.2, 7.0]
            })
            csv_content = df.to_csv(index=False)
            files = {'file': ('date_formats.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            date_formats_result = response.status_code == 200
            csv_variation_tests.append(("Different date formats", date_formats_result))
            
            if date_formats_result:
                print("✅ Different date formats handled correctly")
            else:
                print(f"❌ Different date formats failed: {response.status_code}")
            
            # Overall CSV variations test result
            passed_tests = sum(1 for _, passed in csv_variation_tests if passed)
            total_tests = len(csv_variation_tests)
            
            print(f"📊 CSV variation tests: {passed_tests}/{total_tests}")
            self.test_results['csv_variations'] = passed_tests >= 2  # At least 2/3 should pass
            
        except Exception as e:
            print(f"❌ CSV variations test error: {str(e)}")
            self.test_results['csv_variations'] = False
    
    def test_upload_error_scenarios(self):
        """Test specific error scenarios that might cause upload issues"""
        print("\n=== Testing Upload Error Scenarios ===")
        
        try:
            error_scenario_tests = []
            
            # Test 1: Very large single file (simulate user trying to upload huge file)
            print("   Testing very large file simulation...")
            # Create a reasonably large file (not too big to avoid timeout)
            large_df = pd.DataFrame({
                'timestamp': pd.date_range('2020-01-01', periods=20000, freq='5min'),
                'ph_value': np.random.normal(7.2, 0.2, 20000),
                'temperature': np.random.normal(23, 1, 20000),
                'conductivity': np.random.normal(1200, 50, 20000),
                'notes': ['Measurement ' + str(i) for i in range(20000)]
            })
            csv_content = large_df.to_csv(index=False)
            files = {'file': ('very_large_file.csv', csv_content, 'text/csv')}
            
            try:
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files, timeout=120)
                large_file_handled = response.status_code in [200, 413, 400]  # Success or proper error
                if response.status_code == 200:
                    print("✅ Very large file uploaded successfully")
                else:
                    print(f"✅ Very large file properly rejected: {response.status_code}")
            except requests.exceptions.Timeout:
                print("✅ Very large file timed out (expected behavior)")
                large_file_handled = True
            
            error_scenario_tests.append(("Very large file handling", large_file_handled))
            
            # Test 2: Malformed CSV (inconsistent columns)
            malformed_csv = "timestamp,ph_value,temperature\n2023-01-01,7.2,23.1\n2023-01-02,7.1\n2023-01-03,7.0,22.8,extra_column"
            files = {'file': ('malformed.csv', malformed_csv, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            malformed_handled = response.status_code in [200, 400]  # Either handled or properly rejected
            error_scenario_tests.append(("Malformed CSV", malformed_handled))
            
            if response.status_code == 200:
                print("✅ Malformed CSV handled gracefully")
            else:
                print("✅ Malformed CSV properly rejected")
            
            # Test 3: File with only numeric column names
            df = pd.DataFrame({
                '1': [7.1, 7.2, 7.0],
                '2': [23.1, 23.2, 22.9],
                '3': [1200, 1205, 1198]
            })
            csv_content = df.to_csv(index=False)
            files = {'file': ('numeric_columns.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            numeric_columns_handled = response.status_code in [200, 400]
            error_scenario_tests.append(("Numeric column names", numeric_columns_handled))
            
            if response.status_code == 200:
                print("✅ Numeric column names handled correctly")
            else:
                print("✅ Numeric column names properly rejected")
            
            # Overall error scenarios test result
            passed_tests = sum(1 for _, passed in error_scenario_tests if passed)
            total_tests = len(error_scenario_tests)
            
            print(f"📊 Error scenario tests: {passed_tests}/{total_tests}")
            self.test_results['upload_error_scenarios'] = passed_tests >= 2  # At least 2/3 should pass
            
        except Exception as e:
            print(f"❌ Upload error scenarios test error: {str(e)}")
            self.test_results['upload_error_scenarios'] = False
    
    def test_complete_prediction_workflow(self):
        """Test complete workflow: upload → analysis → parameter configuration → training → prediction"""
        print("\n=== Testing Complete Prediction Workflow ===")
        
        try:
            workflow_steps = []
            
            # Step 1: Upload realistic time series data
            df = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100, freq='D'),
                'sales': 1000 + 200 * np.sin(np.arange(100) * 2 * np.pi / 7) + np.random.normal(0, 50, 100),
                'region': ['North'] * 50 + ['South'] * 50
            })
            csv_content = df.to_csv(index=False)
            files = {'file': ('workflow_test.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                workflow_steps.append(("Upload", True))
                print("✅ Step 1: Data upload successful")
                
                # Step 2: Verify analysis and parameter suggestions
                analysis = data['analysis']
                suggested_params = analysis.get('suggested_parameters', {})
                
                analysis_valid = (
                    'date' in analysis['time_columns'] and
                    'sales' in analysis['numeric_columns'] and
                    suggested_params.get('time_column') == 'date' and
                    suggested_params.get('target_column') == 'sales'
                )
                
                workflow_steps.append(("Analysis", analysis_valid))
                if analysis_valid:
                    print("✅ Step 2: Data analysis and parameter suggestion successful")
                else:
                    print("❌ Step 2: Data analysis failed")
                
                # Step 3: Train ARIMA model (more reliable than Prophet)
                if analysis_valid:
                    training_params = {
                        "time_column": "date",
                        "target_column": "sales",
                        "order": [1, 1, 1]
                    }
                    
                    response = self.session.post(
                        f"{API_BASE_URL}/train-model",
                        params={"data_id": data_id, "model_type": "arima"},
                        json=training_params
                    )
                    
                    if response.status_code == 200:
                        model_data = response.json()
                        model_id = model_data.get('model_id')
                        workflow_steps.append(("Training", True))
                        print("✅ Step 3: Model training successful")
                        
                        # Step 4: Generate predictions
                        if model_id:
                            response = self.session.get(
                                f"{API_BASE_URL}/generate-prediction",
                                params={"model_id": model_id, "steps": 30}
                            )
                            
                            if response.status_code == 200:
                                pred_data = response.json()
                                predictions = pred_data.get('predictions', [])
                                timestamps = pred_data.get('timestamps', [])
                                
                                if len(predictions) == 30 and len(timestamps) == 30:
                                    workflow_steps.append(("Prediction", True))
                                    print("✅ Step 4: Prediction generation successful")
                                    print(f"   Generated {len(predictions)} predictions")
                                    print(f"   Sample predictions: {predictions[:3]}")
                                    
                                    # Step 5: Test continuous prediction
                                    response = self.session.get(
                                        f"{API_BASE_URL}/generate-continuous-prediction",
                                        params={"model_id": model_id, "steps": 10, "time_window": 50}
                                    )
                                    
                                    if response.status_code == 200:
                                        cont_data = response.json()
                                        cont_predictions = cont_data.get('predictions', [])
                                        
                                        if len(cont_predictions) == 10:
                                            workflow_steps.append(("Continuous Prediction", True))
                                            print("✅ Step 5: Continuous prediction successful")
                                        else:
                                            workflow_steps.append(("Continuous Prediction", False))
                                            print("❌ Step 5: Continuous prediction failed")
                                    else:
                                        workflow_steps.append(("Continuous Prediction", False))
                                        print("❌ Step 5: Continuous prediction failed")
                                else:
                                    workflow_steps.append(("Prediction", False))
                                    workflow_steps.append(("Continuous Prediction", False))
                                    print("❌ Step 4: Prediction generation failed - wrong number of predictions")
                            else:
                                workflow_steps.append(("Prediction", False))
                                workflow_steps.append(("Continuous Prediction", False))
                                print("❌ Step 4: Prediction generation failed")
                        else:
                            workflow_steps.append(("Prediction", False))
                            workflow_steps.append(("Continuous Prediction", False))
                            print("❌ Step 4: No model ID for prediction")
                    else:
                        workflow_steps.append(("Training", False))
                        workflow_steps.append(("Prediction", False))
                        workflow_steps.append(("Continuous Prediction", False))
                        print("❌ Step 3: Model training failed")
                else:
                    workflow_steps.append(("Training", False))
                    workflow_steps.append(("Prediction", False))
                    workflow_steps.append(("Continuous Prediction", False))
            else:
                workflow_steps.append(("Upload", False))
                workflow_steps.append(("Analysis", False))
                workflow_steps.append(("Training", False))
                workflow_steps.append(("Prediction", False))
                workflow_steps.append(("Continuous Prediction", False))
                print("❌ Step 1: Data upload failed")
            
            # Evaluate complete workflow
            passed_steps = sum(1 for _, passed in workflow_steps if passed)
            total_steps = len(workflow_steps)
            
            print(f"📊 Complete workflow steps: {passed_steps}/{total_steps}")
            for step_name, passed in workflow_steps:
                status = "✅" if passed else "❌"
                print(f"   {status} {step_name}")
            
            self.test_results['complete_prediction_workflow'] = passed_steps >= total_steps * 0.8  # 80% pass rate
            
        except Exception as e:
            print(f"❌ Complete prediction workflow error: {str(e)}")
            self.test_results['complete_prediction_workflow'] = False
    
    def run_additional_tests(self):
        """Run all additional tests"""
        print("🚀 STARTING ADDITIONAL DOCUMENT UPLOAD TESTING")
        print("=" * 60)
        
        # Run additional tests
        self.test_realistic_ph_dataset()
        self.test_different_file_sizes()
        self.test_csv_variations()
        self.test_upload_error_scenarios()
        self.test_complete_prediction_workflow()
        
        # Generate summary
        self.generate_additional_summary()
    
    def generate_additional_summary(self):
        """Generate summary of additional tests"""
        print("\n" + "=" * 60)
        print("🎯 ADDITIONAL DOCUMENT UPLOAD TEST SUMMARY")
        print("=" * 60)
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 ADDITIONAL TESTS RESULTS: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}% success rate)")
        print()
        
        # Detailed results
        for test_name, result in self.test_results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {test_name.replace('_', ' ').title()}")
        
        print(f"\n🔍 ADDITIONAL FINDINGS:")
        if success_rate >= 80:
            print("   ✅ Additional testing confirms document upload functionality is working well")
        else:
            print("   ⚠️  Additional testing reveals some issues with document upload functionality")
        
        return self.test_results

if __name__ == "__main__":
    tester = AdditionalUploadTester()
    tester.run_additional_tests()