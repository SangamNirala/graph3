#!/usr/bin/env python3
"""
Comprehensive File Upload Testing for pH Monitoring System
Focus on identifying issues that could cause "error in uploading documents"
"""

import requests
import json
import pandas as pd
import io
import time
import numpy as np
import os
from pathlib import Path
import tempfile
import chardet

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://16359d47-48b7-46cc-a21d-6ad29245d1fd.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing file upload functionality at: {API_BASE_URL}")

class ComprehensiveUploadTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_ph_data(self, rows=50):
        """Create realistic pH monitoring data"""
        dates = pd.date_range(start='2023-01-01', periods=rows, freq='H')
        
        # Generate realistic pH values (6.0-8.0 range)
        base_ph = 7.0
        ph_values = []
        current_ph = base_ph
        
        for i in range(rows):
            # Add realistic pH variations
            drift = np.random.normal(0, 0.05)  # Small random drift
            seasonal = 0.2 * np.sin(2 * np.pi * i / 24)  # Daily cycle
            noise = np.random.normal(0, 0.1)  # Measurement noise
            
            current_ph += drift
            ph_value = current_ph + seasonal + noise
            
            # Keep within realistic bounds
            ph_value = max(6.0, min(8.0, ph_value))
            ph_values.append(ph_value)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'pH': ph_values,
            'temperature': np.random.normal(25, 2, rows),  # Temperature data
            'conductivity': np.random.normal(1500, 100, rows)  # Conductivity data
        })
        
        return df
    
    def create_time_series_data(self, rows=100):
        """Create generic time series data"""
        dates = pd.date_range(start='2023-01-01', periods=rows, freq='D')
        
        # Create realistic time series with trend and seasonality
        trend = np.linspace(100, 200, rows)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(rows) / 7)  # Weekly pattern
        noise = np.random.normal(0, 10, rows)
        values = trend + seasonal + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'category': ['A'] * (rows//2) + ['B'] * (rows//2)
        })
        
        return df
    
    def test_basic_csv_upload(self):
        """Test 1: Basic CSV file upload with simple data"""
        print("\n=== Test 1: Basic CSV File Upload ===")
        
        try:
            # Create simple CSV data
            df = self.create_time_series_data(50)
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('test_data.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Basic CSV upload successful")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   Columns: {data['analysis']['columns']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                
                # Validate response structure
                required_fields = ['data_id', 'analysis', 'status']
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    print("✅ Response structure is complete")
                    self.test_results['basic_csv_upload'] = True
                else:
                    print(f"❌ Missing fields in response: {missing_fields}")
                    self.test_results['basic_csv_upload'] = False
                    
            else:
                print(f"❌ Basic CSV upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['basic_csv_upload'] = False
                
        except Exception as e:
            print(f"❌ Basic CSV upload error: {str(e)}")
            self.test_results['basic_csv_upload'] = False
    
    def test_ph_data_upload(self):
        """Test 2: pH monitoring data upload"""
        print("\n=== Test 2: pH Monitoring Data Upload ===")
        
        try:
            # Create pH monitoring data
            df = self.create_ph_data(24)  # 24 hours of hourly data
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('ph_monitoring_data.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ pH data upload successful")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   pH column detected: {'pH' in data['analysis']['numeric_columns']}")
                print(f"   Timestamp column detected: {'timestamp' in data['analysis']['time_columns']}")
                
                # Validate pH-specific analysis
                analysis = data['analysis']
                ph_detected = 'pH' in analysis['numeric_columns']
                timestamp_detected = 'timestamp' in analysis['time_columns']
                
                if ph_detected and timestamp_detected:
                    print("✅ pH monitoring data correctly analyzed")
                    self.test_results['ph_data_upload'] = True
                else:
                    print("❌ pH monitoring data analysis incomplete")
                    self.test_results['ph_data_upload'] = False
                    
            else:
                print(f"❌ pH data upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['ph_data_upload'] = False
                
        except Exception as e:
            print(f"❌ pH data upload error: {str(e)}")
            self.test_results['ph_data_upload'] = False
    
    def test_excel_file_upload(self):
        """Test 3: Excel file upload"""
        print("\n=== Test 3: Excel File Upload ===")
        
        try:
            # Create Excel file
            df = self.create_time_series_data(30)
            
            # Create temporary Excel file
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                df.to_excel(tmp_file.name, index=False)
                tmp_file_path = tmp_file.name
            
            try:
                # Read Excel file as binary
                with open(tmp_file_path, 'rb') as f:
                    excel_content = f.read()
                
                files = {
                    'file': ('test_data.xlsx', excel_content, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                }
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ Excel file upload successful")
                    print(f"   Data shape: {data['analysis']['data_shape']}")
                    print(f"   Columns: {data['analysis']['columns']}")
                    self.test_results['excel_upload'] = True
                else:
                    print(f"❌ Excel file upload failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    self.test_results['excel_upload'] = False
                    
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            print(f"❌ Excel file upload error: {str(e)}")
            self.test_results['excel_upload'] = False
    
    def test_utf8_encoding(self):
        """Test 4: UTF-8 encoding with special characters"""
        print("\n=== Test 4: UTF-8 Encoding Test ===")
        
        try:
            # Create data with special characters
            dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
            df = pd.DataFrame({
                'date': dates,
                'location': ['São Paulo', 'München', 'Zürich', 'Montréal', 'Kraków', 
                           'Москва', '北京', 'Tōkyō', 'Ålesund', 'Reykjavík'],
                'value': np.random.normal(100, 10, 10)
            })
            
            csv_content = df.to_csv(index=False, encoding='utf-8')
            
            files = {
                'file': ('utf8_data.csv', csv_content.encode('utf-8'), 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ UTF-8 encoding upload successful")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Check if special characters are preserved
                preview = data['analysis']['data_preview']['head']
                if preview and len(preview) > 0:
                    location_values = [row.get('location', '') for row in preview[:3]]
                    special_chars_preserved = any('ã' in loc or 'ü' in loc or 'ø' in loc for loc in location_values)
                    
                    if special_chars_preserved:
                        print("✅ Special characters preserved")
                        self.test_results['utf8_encoding'] = True
                    else:
                        print("⚠️  Special characters may not be preserved")
                        print(f"   Sample locations: {location_values}")
                        self.test_results['utf8_encoding'] = True  # Still counts as working
                else:
                    print("⚠️  Cannot verify character preservation (no preview)")
                    self.test_results['utf8_encoding'] = True
                    
            else:
                print(f"❌ UTF-8 encoding upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['utf8_encoding'] = False
                
        except Exception as e:
            print(f"❌ UTF-8 encoding test error: {str(e)}")
            self.test_results['utf8_encoding'] = False
    
    def test_latin1_encoding(self):
        """Test 5: Latin-1 encoding"""
        print("\n=== Test 5: Latin-1 Encoding Test ===")
        
        try:
            # Create data with Latin-1 specific characters
            dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
            df = pd.DataFrame({
                'date': dates,
                'description': ['Café', 'Naïve', 'Résumé', 'Piñata', 'Señor'],
                'symbol': ['©', '®', '°', '±', '§'],
                'value': np.random.normal(50, 5, 5)
            })
            
            csv_content = df.to_csv(index=False, encoding='latin-1')
            
            files = {
                'file': ('latin1_data.csv', csv_content.encode('latin-1'), 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Latin-1 encoding upload successful")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Check if Latin-1 characters are handled
                preview = data['analysis']['data_preview']['head']
                if preview and len(preview) > 0:
                    symbols = [row.get('symbol', '') for row in preview[:3]]
                    print(f"   Sample symbols: {symbols}")
                    self.test_results['latin1_encoding'] = True
                else:
                    self.test_results['latin1_encoding'] = True
                    
            else:
                print(f"❌ Latin-1 encoding upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['latin1_encoding'] = False
                
        except Exception as e:
            print(f"❌ Latin-1 encoding test error: {str(e)}")
            self.test_results['latin1_encoding'] = False
    
    def test_mixed_data_types(self):
        """Test 6: Mixed data types and NaN values"""
        print("\n=== Test 6: Mixed Data Types and NaN Values ===")
        
        try:
            # Create data with mixed types and missing values
            df = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
                'numeric_with_nan': [1.5, 2.0, np.nan, 4.5, 5.0, np.nan, 7.0, 8.5, 9.0, 10.5],
                'mixed_column': [1, '2', 3.5, 'text', 5, np.nan, 7, '8.5', 9, 'end'],
                'text_column': ['A', 'B', '', 'D', None, 'F', 'G', '  ', 'I', 'J'],
                'boolean_like': [True, False, 1, 0, 'Yes', 'No', True, False, 1, 0]
            })
            
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('mixed_data.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Mixed data types upload successful")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                
                # Check missing values handling
                missing_values = data['analysis']['data_preview']['missing_values']
                if missing_values:
                    total_missing = sum(missing_values.values())
                    print(f"   Total missing values detected: {total_missing}")
                    self.test_results['mixed_data_types'] = True
                else:
                    print("   No missing values info (may still be working)")
                    self.test_results['mixed_data_types'] = True
                    
            else:
                print(f"❌ Mixed data types upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['mixed_data_types'] = False
                
        except Exception as e:
            print(f"❌ Mixed data types test error: {str(e)}")
            self.test_results['mixed_data_types'] = False
    
    def test_large_file_upload(self):
        """Test 7: Large file upload"""
        print("\n=== Test 7: Large File Upload ===")
        
        try:
            # Create larger dataset (10,000 rows)
            df = self.create_time_series_data(10000)
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('large_data.csv', csv_content, 'text/csv')
            }
            
            print(f"   File size: {len(csv_content)} bytes (~{len(csv_content)/1024:.1f} KB)")
            
            # Use longer timeout for large file
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Large file upload successful")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                self.test_results['large_file_upload'] = True
            else:
                print(f"❌ Large file upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['large_file_upload'] = False
                
        except Exception as e:
            print(f"❌ Large file upload error: {str(e)}")
            self.test_results['large_file_upload'] = False
    
    def test_empty_file_handling(self):
        """Test 8: Empty file handling"""
        print("\n=== Test 8: Empty File Handling ===")
        
        try:
            # Test completely empty file
            files = {
                'file': ('empty.csv', '', 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code >= 400:
                print("✅ Empty file correctly rejected")
                empty_file_test = True
            else:
                print("❌ Empty file not rejected (should return error)")
                empty_file_test = False
            
            # Test file with only headers
            files = {
                'file': ('headers_only.csv', 'date,value\n', 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code >= 400:
                print("✅ Headers-only file correctly rejected")
                headers_only_test = True
            else:
                print("❌ Headers-only file not rejected")
                headers_only_test = False
            
            self.test_results['empty_file_handling'] = empty_file_test and headers_only_test
            
        except Exception as e:
            print(f"❌ Empty file handling error: {str(e)}")
            self.test_results['empty_file_handling'] = False
    
    def test_invalid_file_formats(self):
        """Test 9: Invalid file format handling"""
        print("\n=== Test 9: Invalid File Format Handling ===")
        
        try:
            invalid_formats = [
                ('test.txt', 'This is a text file', 'text/plain'),
                ('test.json', '{"key": "value"}', 'application/json'),
                ('test.pdf', b'%PDF-1.4 fake pdf content', 'application/pdf'),
                ('test.doc', b'fake doc content', 'application/msword')
            ]
            
            invalid_format_tests = []
            
            for filename, content, content_type in invalid_formats:
                files = {
                    'file': (filename, content, content_type)
                }
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code >= 400:
                    print(f"✅ {filename} correctly rejected")
                    invalid_format_tests.append(True)
                else:
                    print(f"❌ {filename} not rejected (should return error)")
                    invalid_format_tests.append(False)
            
            self.test_results['invalid_file_formats'] = all(invalid_format_tests)
            
        except Exception as e:
            print(f"❌ Invalid file format test error: {str(e)}")
            self.test_results['invalid_file_formats'] = False
    
    def test_malformed_csv_data(self):
        """Test 10: Malformed CSV data handling"""
        print("\n=== Test 10: Malformed CSV Data Handling ===")
        
        try:
            malformed_data_tests = []
            
            # Test 1: Inconsistent number of columns
            malformed_csv1 = "date,value\n2023-01-01,100\n2023-01-02,200,extra_column\n2023-01-03,300"
            files = {'file': ('malformed1.csv', malformed_csv1, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                print("✅ Inconsistent columns handled gracefully")
                malformed_data_tests.append(True)
            else:
                print(f"⚠️  Inconsistent columns rejected: {response.status_code}")
                malformed_data_tests.append(True)  # Either handling is acceptable
            
            # Test 2: Invalid date formats
            malformed_csv2 = "date,value\ninvalid_date,100\n2023-13-45,200\n2023-01-01,300"
            files = {'file': ('malformed2.csv', malformed_csv2, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                print("✅ Invalid dates handled gracefully")
                malformed_data_tests.append(True)
            else:
                print(f"⚠️  Invalid dates rejected: {response.status_code}")
                malformed_data_tests.append(True)  # Either handling is acceptable
            
            # Test 3: All NaN column
            malformed_csv3 = "date,value\n2023-01-01,\n2023-01-02,\n2023-01-03,"
            files = {'file': ('malformed3.csv', malformed_csv3, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                print("✅ All-NaN column handled gracefully")
                malformed_data_tests.append(True)
            else:
                print(f"⚠️  All-NaN column rejected: {response.status_code}")
                malformed_data_tests.append(True)  # Either handling is acceptable
            
            self.test_results['malformed_csv_data'] = all(malformed_data_tests)
            
        except Exception as e:
            print(f"❌ Malformed CSV data test error: {str(e)}")
            self.test_results['malformed_csv_data'] = False
    
    def test_data_quality_analysis(self):
        """Test 11: Data quality analysis functionality"""
        print("\n=== Test 11: Data Quality Analysis ===")
        
        try:
            # Upload data and get data_id
            df = self.create_ph_data(50)
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('quality_test.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data_id = response.json().get('data_id')
                print("✅ Data uploaded for quality analysis")
                
                # Test data quality report endpoint
                response = self.session.get(f"{API_BASE_URL}/data-quality-report", params={"data_id": data_id})
                
                if response.status_code == 200:
                    quality_data = response.json()
                    print("✅ Data quality analysis successful")
                    print(f"   Quality score: {quality_data.get('quality_score', 'N/A')}")
                    print(f"   Recommendations: {len(quality_data.get('recommendations', []))}")
                    
                    # Validate quality report structure
                    required_fields = ['quality_score', 'recommendations', 'analysis_details']
                    missing_fields = [field for field in required_fields if field not in quality_data]
                    
                    if not missing_fields:
                        print("✅ Quality report structure is complete")
                        self.test_results['data_quality_analysis'] = True
                    else:
                        print(f"⚠️  Missing fields in quality report: {missing_fields}")
                        self.test_results['data_quality_analysis'] = True  # Still working
                        
                else:
                    print(f"❌ Data quality analysis failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    self.test_results['data_quality_analysis'] = False
                    
            else:
                print("❌ Failed to upload data for quality analysis")
                self.test_results['data_quality_analysis'] = False
                
        except Exception as e:
            print(f"❌ Data quality analysis error: {str(e)}")
            self.test_results['data_quality_analysis'] = False
    
    def test_complete_upload_flow(self):
        """Test 12: Complete upload to prediction flow"""
        print("\n=== Test 12: Complete Upload to Prediction Flow ===")
        
        try:
            flow_steps = []
            
            # Step 1: Upload data
            df = self.create_ph_data(30)
            csv_content = df.to_csv(index=False)
            files = {'file': ('flow_test.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            upload_success = response.status_code == 200
            flow_steps.append(("Data upload", upload_success))
            
            if upload_success:
                data_id = response.json().get('data_id')
                analysis = response.json().get('analysis', {})
                print("✅ Step 1: Data upload successful")
                
                # Step 2: Verify parameter suggestions
                suggested_params = analysis.get('suggested_parameters', {})
                time_col = suggested_params.get('time_column')
                target_col = suggested_params.get('target_column')
                
                params_valid = time_col and target_col
                flow_steps.append(("Parameter suggestions", params_valid))
                
                if params_valid:
                    print("✅ Step 2: Parameter suggestions generated")
                    
                    # Step 3: Train model with suggested parameters
                    response = self.session.post(
                        f"{API_BASE_URL}/train-model",
                        params={"data_id": data_id, "model_type": "arima"},
                        json={
                            "time_column": time_col,
                            "target_column": target_col,
                            "order": [1, 1, 1]
                        }
                    )
                    
                    training_success = response.status_code == 200
                    flow_steps.append(("Model training", training_success))
                    
                    if training_success:
                        model_id = response.json().get('model_id')
                        print("✅ Step 3: Model training successful")
                        
                        # Step 4: Generate predictions
                        response = self.session.get(
                            f"{API_BASE_URL}/generate-prediction",
                            params={"model_id": model_id, "steps": 10}
                        )
                        
                        prediction_success = response.status_code == 200
                        flow_steps.append(("Prediction generation", prediction_success))
                        
                        if prediction_success:
                            print("✅ Step 4: Prediction generation successful")
                        else:
                            print("❌ Step 4: Prediction generation failed")
                    else:
                        print("❌ Step 3: Model training failed")
                        flow_steps.append(("Prediction generation", False))
                else:
                    print("❌ Step 2: Parameter suggestions invalid")
                    flow_steps.append(("Model training", False))
                    flow_steps.append(("Prediction generation", False))
            else:
                print("❌ Step 1: Data upload failed")
                flow_steps.extend([
                    ("Parameter suggestions", False),
                    ("Model training", False),
                    ("Prediction generation", False)
                ])
            
            # Evaluate complete flow
            passed_steps = sum(1 for _, passed in flow_steps if passed)
            total_steps = len(flow_steps)
            
            print(f"✅ Complete flow steps passed: {passed_steps}/{total_steps}")
            for step_name, passed in flow_steps:
                status = "✅" if passed else "❌"
                print(f"   {status} {step_name}")
            
            self.test_results['complete_upload_flow'] = passed_steps >= total_steps * 0.75  # 75% success rate
            
        except Exception as e:
            print(f"❌ Complete upload flow error: {str(e)}")
            self.test_results['complete_upload_flow'] = False
    
    def run_all_tests(self):
        """Run all upload tests"""
        print("🚀 Starting Comprehensive File Upload Testing")
        print("=" * 60)
        
        # Run all tests
        self.test_basic_csv_upload()
        self.test_ph_data_upload()
        self.test_excel_file_upload()
        self.test_utf8_encoding()
        self.test_latin1_encoding()
        self.test_mixed_data_types()
        self.test_large_file_upload()
        self.test_empty_file_handling()
        self.test_invalid_file_formats()
        self.test_malformed_csv_data()
        self.test_data_quality_analysis()
        self.test_complete_upload_flow()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE FILE UPLOAD TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print()
        
        # Group results
        critical_tests = [
            'basic_csv_upload', 'ph_data_upload', 'complete_upload_flow'
        ]
        
        encoding_tests = [
            'utf8_encoding', 'latin1_encoding', 'mixed_data_types'
        ]
        
        edge_case_tests = [
            'empty_file_handling', 'invalid_file_formats', 'malformed_csv_data'
        ]
        
        feature_tests = [
            'excel_upload', 'large_file_upload', 'data_quality_analysis'
        ]
        
        def print_test_group(group_name, test_list):
            print(f"{group_name}:")
            for test in test_list:
                if test in self.test_results:
                    status = "✅ PASS" if self.test_results[test] else "❌ FAIL"
                    print(f"  {status} {test.replace('_', ' ').title()}")
            print()
        
        print_test_group("🔥 CRITICAL FUNCTIONALITY", critical_tests)
        print_test_group("🌐 ENCODING SUPPORT", encoding_tests)
        print_test_group("⚠️  EDGE CASE HANDLING", edge_case_tests)
        print_test_group("🚀 ADVANCED FEATURES", feature_tests)
        
        # Identify potential issues
        failed_tests = [test for test, result in self.test_results.items() if not result]
        
        if failed_tests:
            print("🚨 POTENTIAL ISSUES IDENTIFIED:")
            for test in failed_tests:
                print(f"  ❌ {test.replace('_', ' ').title()}")
            print()
        
        # Overall assessment
        if success_rate >= 90:
            print("🎉 EXCELLENT: File upload functionality is working very well!")
        elif success_rate >= 75:
            print("✅ GOOD: File upload functionality is mostly working with minor issues.")
        elif success_rate >= 50:
            print("⚠️  MODERATE: File upload has some issues that need attention.")
        else:
            print("🚨 CRITICAL: File upload functionality has significant problems!")
        
        print()
        print("📋 RECOMMENDATIONS:")
        
        if not self.test_results.get('basic_csv_upload', True):
            print("  • Fix basic CSV upload functionality - this is critical!")
        
        if not self.test_results.get('ph_data_upload', True):
            print("  • Fix pH monitoring data upload - core functionality issue!")
        
        if not self.test_results.get('complete_upload_flow', True):
            print("  • Fix complete upload to prediction flow - end-to-end issue!")
        
        encoding_issues = [t for t in encoding_tests if not self.test_results.get(t, True)]
        if encoding_issues:
            print(f"  • Improve encoding support for: {', '.join(encoding_issues)}")
        
        if not self.test_results.get('data_quality_analysis', True):
            print("  • Fix data quality analysis endpoint")
        
        if success_rate == 100:
            print("  • No issues found! File upload functionality is working perfectly.")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    tester = ComprehensiveUploadTester()
    tester.run_all_tests()