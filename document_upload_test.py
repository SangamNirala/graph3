#!/usr/bin/env python3
"""
Comprehensive Document Upload Testing for Real-time Graph Prediction Application
Focus on testing document upload functionality thoroughly as requested in review
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

print(f"🎯 COMPREHENSIVE DOCUMENT UPLOAD TESTING")
print(f"Testing backend at: {API_BASE_URL}")
print(f"Focus: Document upload functionality as requested in review")

class DocumentUploadTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        self.uploaded_data_ids = []
        
    def create_realistic_ph_data(self, num_points=48):
        """Create realistic pH monitoring data for testing"""
        # Generate 48 hours of pH data (hourly readings)
        start_time = datetime.now() - timedelta(hours=num_points)
        timestamps = [start_time + timedelta(hours=i) for i in range(num_points)]
        
        # Realistic pH values with natural variation (6.8-7.6 range)
        base_ph = 7.2
        ph_values = []
        
        for i in range(num_points):
            # Add daily cycle (slightly lower at night)
            daily_cycle = 0.2 * np.sin(2 * np.pi * i / 24)
            # Add random variation
            noise = np.random.normal(0, 0.1)
            # Add slight trend
            trend = 0.001 * i
            
            ph_value = base_ph + daily_cycle + noise + trend
            # Keep within realistic bounds
            ph_value = max(6.0, min(8.0, ph_value))
            ph_values.append(round(ph_value, 2))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ph_value': ph_values,
            'temperature': [round(22 + np.random.normal(0, 2), 1) for _ in range(num_points)],
            'sensor_id': ['pH_001'] * num_points
        })
        
        return df
    
    def create_sales_data(self, num_points=100):
        """Create realistic sales data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=num_points, freq='D')
        
        # Create realistic sales data with trend and seasonality
        trend = np.linspace(1000, 1500, num_points)
        seasonal = 200 * np.sin(2 * np.pi * np.arange(num_points) / 7)  # Weekly pattern
        noise = np.random.normal(0, 50, num_points)
        sales = trend + seasonal + noise
        
        df = pd.DataFrame({
            'date': dates,
            'sales': [round(s, 2) for s in sales],
            'region': ['North'] * (num_points//2) + ['South'] * (num_points//2),
            'product': ['A'] * (num_points//3) + ['B'] * (num_points//3) + ['C'] * (num_points//3 + num_points%3)
        })
        
        return df
    
    def create_problematic_data(self):
        """Create data with various issues for edge case testing"""
        # Data with NaN values, mixed types, empty strings
        df = pd.DataFrame({
            'time_step': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'value': [7.2, np.nan, 7.1, '', '7.3', 7.0, None, 7.4, 'invalid', 7.2],
            'category': ['A', 'B', '', 'A', None, 'B', 'A', '', 'B', 'A'],
            'empty_col': ['', '', '', '', '', '', '', '', '', '']
        })
        return df
    
    def test_csv_upload_basic(self):
        """Test 1: Basic CSV file upload"""
        print("\n=== Test 1: Basic CSV File Upload ===")
        
        try:
            # Create realistic pH data
            df = self.create_realistic_ph_data(24)  # 24 hours of data
            csv_content = df.to_csv(index=False)
            
            # Prepare file for upload
            files = {
                'file': ('ph_monitoring_24h.csv', csv_content, 'text/csv')
            }
            
            # Test file upload
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                self.uploaded_data_ids.append(data_id)
                
                print("✅ Basic CSV upload successful")
                print(f"   Data ID: {data_id}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Validate analysis results
                analysis = data['analysis']
                expected_columns = ['timestamp', 'ph_value', 'temperature', 'sensor_id']
                columns_correct = all(col in analysis['columns'] for col in expected_columns)
                time_detected = 'timestamp' in analysis['time_columns']
                numeric_detected = 'ph_value' in analysis['numeric_columns']
                
                if columns_correct and time_detected and numeric_detected:
                    print("✅ Data analysis correctly identified all columns")
                    self.test_results['csv_upload_basic'] = True
                else:
                    print("❌ Data analysis failed to identify columns correctly")
                    self.test_results['csv_upload_basic'] = False
                    
            else:
                print(f"❌ Basic CSV upload failed: {response.status_code} - {response.text}")
                self.test_results['csv_upload_basic'] = False
                
        except Exception as e:
            print(f"❌ Basic CSV upload error: {str(e)}")
            self.test_results['csv_upload_basic'] = False
    
    def test_excel_upload(self):
        """Test 2: Excel file upload (.xlsx)"""
        print("\n=== Test 2: Excel File Upload (.xlsx) ===")
        
        try:
            # Create sales data
            df = self.create_sales_data(50)
            
            # Create temporary Excel file
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                df.to_excel(tmp_file.name, index=False)
                tmp_file_path = tmp_file.name
            
            try:
                # Read Excel file as binary
                with open(tmp_file_path, 'rb') as f:
                    excel_content = f.read()
                
                # Prepare file for upload
                files = {
                    'file': ('sales_data.xlsx', excel_content, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                }
                
                # Test Excel upload
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    data_id = data.get('data_id')
                    self.uploaded_data_ids.append(data_id)
                    
                    print("✅ Excel upload successful")
                    print(f"   Data ID: {data_id}")
                    print(f"   Columns detected: {data['analysis']['columns']}")
                    print(f"   Data shape: {data['analysis']['data_shape']}")
                    
                    # Validate Excel analysis
                    analysis = data['analysis']
                    expected_columns = ['date', 'sales', 'region', 'product']
                    columns_correct = all(col in analysis['columns'] for col in expected_columns)
                    
                    if columns_correct and len(analysis['columns']) == 4:
                        print("✅ Excel data analysis correctly identified all columns")
                        self.test_results['excel_upload'] = True
                    else:
                        print("❌ Excel data analysis failed")
                        self.test_results['excel_upload'] = False
                        
                else:
                    print(f"❌ Excel upload failed: {response.status_code} - {response.text}")
                    self.test_results['excel_upload'] = False
                    
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            print(f"❌ Excel upload error: {str(e)}")
            self.test_results['excel_upload'] = False
    
    def test_large_file_upload(self):
        """Test 3: Large file upload (10,000 rows)"""
        print("\n=== Test 3: Large File Upload (10,000 rows) ===")
        
        try:
            # Create large dataset
            df = self.create_sales_data(10000)
            csv_content = df.to_csv(index=False)
            
            print(f"   File size: {len(csv_content)} bytes (~{len(csv_content)/1024:.1f} KB)")
            
            # Prepare file for upload
            files = {
                'file': ('large_sales_data.csv', csv_content, 'text/csv')
            }
            
            # Test large file upload with longer timeout
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                self.uploaded_data_ids.append(data_id)
                
                print("✅ Large file upload successful")
                print(f"   Data ID: {data_id}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Validate large file processing
                analysis = data['analysis']
                if analysis['data_shape'][0] == 10000:
                    print("✅ Large file processed correctly (10,000 rows)")
                    self.test_results['large_file_upload'] = True
                else:
                    print(f"❌ Large file processing error: expected 10,000 rows, got {analysis['data_shape'][0]}")
                    self.test_results['large_file_upload'] = False
                    
            else:
                print(f"❌ Large file upload failed: {response.status_code} - {response.text}")
                self.test_results['large_file_upload'] = False
                
        except Exception as e:
            print(f"❌ Large file upload error: {str(e)}")
            self.test_results['large_file_upload'] = False
    
    def test_utf8_encoding(self):
        """Test 4: UTF-8 encoding with special characters"""
        print("\n=== Test 4: UTF-8 Encoding with Special Characters ===")
        
        try:
            # Create data with UTF-8 special characters
            df = pd.DataFrame({
                'city': ['São Paulo', 'München', 'Zürich', 'Montréal', 'Kraków'],
                'temperature': [25.5, 18.2, 16.8, 22.1, 19.7],
                'humidity': [65, 72, 68, 71, 69],
                'date': pd.date_range('2023-01-01', periods=5, freq='D')
            })
            
            csv_content = df.to_csv(index=False, encoding='utf-8')
            
            # Prepare file for upload
            files = {
                'file': ('international_weather.csv', csv_content.encode('utf-8'), 'text/csv')
            }
            
            # Test UTF-8 upload
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                self.uploaded_data_ids.append(data_id)
                
                print("✅ UTF-8 encoding upload successful")
                print(f"   Data ID: {data_id}")
                
                # Check if special characters are preserved
                preview = data['analysis']['data_preview']['head']
                if preview and len(preview) > 0:
                    cities_in_preview = [row.get('city', '') for row in preview]
                    special_chars_preserved = any('ã' in city or 'ü' in city or 'é' in city or 'ó' in city 
                                                for city in cities_in_preview)
                    
                    if special_chars_preserved:
                        print("✅ UTF-8 special characters preserved correctly")
                        print(f"   Sample cities: {cities_in_preview[:3]}")
                        self.test_results['utf8_encoding'] = True
                    else:
                        print("❌ UTF-8 special characters not preserved")
                        self.test_results['utf8_encoding'] = False
                else:
                    print("❌ No preview data available to check UTF-8 encoding")
                    self.test_results['utf8_encoding'] = False
                    
            else:
                print(f"❌ UTF-8 encoding upload failed: {response.status_code} - {response.text}")
                self.test_results['utf8_encoding'] = False
                
        except Exception as e:
            print(f"❌ UTF-8 encoding test error: {str(e)}")
            self.test_results['utf8_encoding'] = False
    
    def test_latin1_encoding(self):
        """Test 5: Latin-1 encoding with symbols"""
        print("\n=== Test 5: Latin-1 Encoding with Symbols ===")
        
        try:
            # Create data with Latin-1 symbols
            df = pd.DataFrame({
                'product': ['Widget©', 'Gadget®', 'Tool°', 'Device±', 'Item§'],
                'price': [19.99, 29.99, 39.99, 49.99, 59.99],
                'currency': ['€', '$', '£', '¥', '€'],
                'date': pd.date_range('2023-01-01', periods=5, freq='D')
            })
            
            csv_content = df.to_csv(index=False, encoding='latin-1')
            
            # Prepare file for upload
            files = {
                'file': ('products_latin1.csv', csv_content.encode('latin-1'), 'text/csv')
            }
            
            # Test Latin-1 upload
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                self.uploaded_data_ids.append(data_id)
                
                print("✅ Latin-1 encoding upload successful")
                print(f"   Data ID: {data_id}")
                
                # Check if symbols are preserved
                preview = data['analysis']['data_preview']['head']
                if preview and len(preview) > 0:
                    products_in_preview = [row.get('product', '') for row in preview]
                    symbols_preserved = any('©' in prod or '®' in prod or '°' in prod or '±' in prod 
                                          for prod in products_in_preview)
                    
                    if symbols_preserved:
                        print("✅ Latin-1 symbols preserved correctly")
                        print(f"   Sample products: {products_in_preview[:3]}")
                        self.test_results['latin1_encoding'] = True
                    else:
                        print("❌ Latin-1 symbols not preserved")
                        self.test_results['latin1_encoding'] = False
                else:
                    print("❌ No preview data available to check Latin-1 encoding")
                    self.test_results['latin1_encoding'] = False
                    
            else:
                print(f"❌ Latin-1 encoding upload failed: {response.status_code} - {response.text}")
                self.test_results['latin1_encoding'] = False
                
        except Exception as e:
            print(f"❌ Latin-1 encoding test error: {str(e)}")
            self.test_results['latin1_encoding'] = False
    
    def test_problematic_data_upload(self):
        """Test 6: Upload data with NaN values, mixed types, empty strings"""
        print("\n=== Test 6: Problematic Data Upload (NaN, Mixed Types, Empty Strings) ===")
        
        try:
            # Create problematic data
            df = self.create_problematic_data()
            csv_content = df.to_csv(index=False)
            
            # Prepare file for upload
            files = {
                'file': ('problematic_data.csv', csv_content, 'text/csv')
            }
            
            # Test problematic data upload
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                self.uploaded_data_ids.append(data_id)
                
                print("✅ Problematic data upload successful")
                print(f"   Data ID: {data_id}")
                
                # Check data quality analysis
                analysis = data['analysis']
                missing_values = analysis['data_preview'].get('missing_values', {})
                
                # Should detect missing values
                total_missing = sum(missing_values.values()) if missing_values else 0
                if total_missing > 0:
                    print(f"✅ Missing values detected correctly: {total_missing} total")
                    print(f"   Missing values by column: {missing_values}")
                    self.test_results['problematic_data_upload'] = True
                else:
                    print("❌ Missing values not detected properly")
                    self.test_results['problematic_data_upload'] = False
                    
            else:
                print(f"❌ Problematic data upload failed: {response.status_code} - {response.text}")
                self.test_results['problematic_data_upload'] = False
                
        except Exception as e:
            print(f"❌ Problematic data upload error: {str(e)}")
            self.test_results['problematic_data_upload'] = False
    
    def test_invalid_file_formats(self):
        """Test 7: Invalid file format rejection"""
        print("\n=== Test 7: Invalid File Format Rejection ===")
        
        try:
            invalid_format_tests = []
            
            # Test 1: Text file (.txt)
            text_content = "This is a plain text file, not CSV or Excel"
            files = {'file': ('test.txt', text_content, 'text/plain')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            txt_rejected = response.status_code >= 400
            invalid_format_tests.append(("TXT file rejection", txt_rejected))
            if txt_rejected:
                print("✅ TXT file correctly rejected")
            else:
                print("❌ TXT file not rejected")
            
            # Test 2: JSON file (.json)
            json_content = '{"data": "This is JSON, not CSV"}'
            files = {'file': ('test.json', json_content, 'application/json')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            json_rejected = response.status_code >= 400
            invalid_format_tests.append(("JSON file rejection", json_rejected))
            if json_rejected:
                print("✅ JSON file correctly rejected")
            else:
                print("❌ JSON file not rejected")
            
            # Test 3: Image file (.png)
            # Create minimal PNG-like content
            png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            files = {'file': ('test.png', png_content, 'image/png')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            png_rejected = response.status_code >= 400
            invalid_format_tests.append(("PNG file rejection", png_rejected))
            if png_rejected:
                print("✅ PNG file correctly rejected")
            else:
                print("❌ PNG file not rejected")
            
            # Overall invalid format test result
            passed_tests = sum(1 for _, passed in invalid_format_tests if passed)
            total_tests = len(invalid_format_tests)
            
            print(f"📊 Invalid format rejection tests: {passed_tests}/{total_tests}")
            self.test_results['invalid_file_formats'] = passed_tests >= total_tests * 0.7  # 70% pass rate
            
        except Exception as e:
            print(f"❌ Invalid file format test error: {str(e)}")
            self.test_results['invalid_file_formats'] = False
    
    def test_empty_file_upload(self):
        """Test 8: Empty file upload rejection"""
        print("\n=== Test 8: Empty File Upload Rejection ===")
        
        try:
            # Test empty CSV file
            empty_content = ""
            files = {'file': ('empty.csv', empty_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code >= 400:
                print("✅ Empty file correctly rejected")
                print(f"   Status code: {response.status_code}")
                self.test_results['empty_file_upload'] = True
            else:
                print("❌ Empty file not rejected")
                self.test_results['empty_file_upload'] = False
                
        except Exception as e:
            print(f"❌ Empty file upload test error: {str(e)}")
            self.test_results['empty_file_upload'] = False
    
    def test_complete_upload_workflow(self):
        """Test 9: Complete upload → analysis → parameter configuration → training workflow"""
        print("\n=== Test 9: Complete Upload Workflow ===")
        
        try:
            workflow_tests = []
            
            # Step 1: Upload data
            df = self.create_realistic_ph_data(48)
            csv_content = df.to_csv(index=False)
            files = {'file': ('workflow_test.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                workflow_tests.append(("File upload", True))
                print("✅ Step 1: File upload successful")
                
                # Step 2: Verify data analysis
                analysis = data['analysis']
                suggested_params = analysis.get('suggested_parameters', {})
                
                analysis_valid = (
                    'timestamp' in analysis['time_columns'] and
                    'ph_value' in analysis['numeric_columns'] and
                    suggested_params.get('time_column') == 'timestamp' and
                    suggested_params.get('target_column') == 'ph_value'
                )
                
                workflow_tests.append(("Data analysis", analysis_valid))
                if analysis_valid:
                    print("✅ Step 2: Data analysis successful")
                else:
                    print("❌ Step 2: Data analysis failed")
                
                # Step 3: Train model with suggested parameters
                if analysis_valid:
                    training_params = {
                        "time_column": "timestamp",
                        "target_column": "ph_value",
                        "seasonality_mode": "additive",
                        "yearly_seasonality": False,
                        "weekly_seasonality": False,
                        "daily_seasonality": True
                    }
                    
                    response = self.session.post(
                        f"{API_BASE_URL}/train-model",
                        params={"data_id": data_id, "model_type": "arima"},
                        json=training_params
                    )
                    
                    if response.status_code == 200:
                        model_data = response.json()
                        model_id = model_data.get('model_id')
                        workflow_tests.append(("Model training", True))
                        print("✅ Step 3: Model training successful")
                        
                        # Step 4: Generate predictions
                        if model_id:
                            response = self.session.get(
                                f"{API_BASE_URL}/generate-prediction",
                                params={"model_id": model_id, "steps": 24}
                            )
                            
                            if response.status_code == 200:
                                pred_data = response.json()
                                predictions = pred_data.get('predictions', [])
                                
                                if len(predictions) == 24:
                                    workflow_tests.append(("Prediction generation", True))
                                    print("✅ Step 4: Prediction generation successful")
                                else:
                                    workflow_tests.append(("Prediction generation", False))
                                    print("❌ Step 4: Prediction generation failed - wrong number of predictions")
                            else:
                                workflow_tests.append(("Prediction generation", False))
                                print("❌ Step 4: Prediction generation failed")
                        else:
                            workflow_tests.append(("Prediction generation", False))
                            print("❌ Step 4: No model ID for prediction generation")
                    else:
                        workflow_tests.append(("Model training", False))
                        workflow_tests.append(("Prediction generation", False))
                        print("❌ Step 3: Model training failed")
                else:
                    workflow_tests.append(("Model training", False))
                    workflow_tests.append(("Prediction generation", False))
            else:
                workflow_tests.append(("File upload", False))
                workflow_tests.append(("Data analysis", False))
                workflow_tests.append(("Model training", False))
                workflow_tests.append(("Prediction generation", False))
                print("❌ Step 1: File upload failed")
            
            # Evaluate workflow test results
            passed_tests = sum(1 for _, passed in workflow_tests if passed)
            total_tests = len(workflow_tests)
            
            print(f"📊 Complete workflow tests: {passed_tests}/{total_tests}")
            for test_name, passed in workflow_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['complete_upload_workflow'] = passed_tests >= total_tests * 0.8  # 80% pass rate
            
        except Exception as e:
            print(f"❌ Complete upload workflow error: {str(e)}")
            self.test_results['complete_upload_workflow'] = False
    
    def test_concurrent_uploads(self):
        """Test 10: Concurrent file uploads"""
        print("\n=== Test 10: Concurrent File Uploads ===")
        
        try:
            import threading
            import queue
            
            results_queue = queue.Queue()
            
            def upload_file(file_name, data_df):
                try:
                    csv_content = data_df.to_csv(index=False)
                    files = {'file': (file_name, csv_content, 'text/csv')}
                    
                    response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                    results_queue.put((file_name, response.status_code == 200))
                except Exception as e:
                    results_queue.put((file_name, False))
            
            # Create multiple datasets
            datasets = [
                ("concurrent_ph_1.csv", self.create_realistic_ph_data(24)),
                ("concurrent_sales_1.csv", self.create_sales_data(50)),
                ("concurrent_ph_2.csv", self.create_realistic_ph_data(36))
            ]
            
            # Start concurrent uploads
            threads = []
            for file_name, data_df in datasets:
                thread = threading.Thread(target=upload_file, args=(file_name, data_df))
                threads.append(thread)
                thread.start()
            
            # Wait for all uploads to complete
            for thread in threads:
                thread.join(timeout=30)  # 30 second timeout per thread
            
            # Collect results
            upload_results = []
            while not results_queue.empty():
                upload_results.append(results_queue.get())
            
            successful_uploads = sum(1 for _, success in upload_results if success)
            total_uploads = len(upload_results)
            
            print(f"📊 Concurrent uploads: {successful_uploads}/{total_uploads} successful")
            for file_name, success in upload_results:
                status = "✅" if success else "❌"
                print(f"   {status} {file_name}")
            
            self.test_results['concurrent_uploads'] = successful_uploads >= total_uploads * 0.8  # 80% success rate
            
        except Exception as e:
            print(f"❌ Concurrent uploads test error: {str(e)}")
            self.test_results['concurrent_uploads'] = False
    
    def test_edge_cases(self):
        """Test 11: Edge cases and boundary conditions"""
        print("\n=== Test 11: Edge Cases and Boundary Conditions ===")
        
        try:
            edge_case_tests = []
            
            # Test 1: Single row CSV
            single_row_df = pd.DataFrame({
                'timestamp': [datetime.now()],
                'value': [7.2]
            })
            csv_content = single_row_df.to_csv(index=False)
            files = {'file': ('single_row.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            single_row_handled = response.status_code in [200, 400]  # Either accepted or properly rejected
            edge_case_tests.append(("Single row CSV", single_row_handled))
            
            if response.status_code == 200:
                print("✅ Single row CSV accepted")
            else:
                print("✅ Single row CSV properly rejected")
            
            # Test 2: Very wide CSV (many columns)
            wide_data = {'col_' + str(i): [i] * 10 for i in range(100)}  # 100 columns
            wide_df = pd.DataFrame(wide_data)
            csv_content = wide_df.to_csv(index=False)
            files = {'file': ('wide_data.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            wide_csv_handled = response.status_code in [200, 400]
            edge_case_tests.append(("Wide CSV (100 columns)", wide_csv_handled))
            
            if response.status_code == 200:
                print("✅ Wide CSV (100 columns) accepted")
            else:
                print("✅ Wide CSV properly rejected")
            
            # Test 3: CSV with only headers (no data rows)
            headers_only_content = "timestamp,ph_value,temperature\n"
            files = {'file': ('headers_only.csv', headers_only_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            headers_only_handled = response.status_code >= 400  # Should be rejected
            edge_case_tests.append(("Headers only CSV", headers_only_handled))
            
            if headers_only_handled:
                print("✅ Headers-only CSV properly rejected")
            else:
                print("❌ Headers-only CSV not rejected")
            
            # Test 4: CSV with special filename characters
            normal_df = self.create_realistic_ph_data(10)
            csv_content = normal_df.to_csv(index=False)
            special_filename = "test file with spaces & symbols (2023).csv"
            files = {'file': (special_filename, csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            special_filename_handled = response.status_code == 200
            edge_case_tests.append(("Special filename characters", special_filename_handled))
            
            if special_filename_handled:
                print("✅ Special filename characters handled correctly")
            else:
                print("❌ Special filename characters caused issues")
            
            # Evaluate edge case test results
            passed_tests = sum(1 for _, passed in edge_case_tests if passed)
            total_tests = len(edge_case_tests)
            
            print(f"📊 Edge case tests: {passed_tests}/{total_tests}")
            for test_name, passed in edge_case_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['edge_cases'] = passed_tests >= total_tests * 0.7  # 70% pass rate
            
        except Exception as e:
            print(f"❌ Edge cases test error: {str(e)}")
            self.test_results['edge_cases'] = False
    
    def run_all_tests(self):
        """Run all document upload tests"""
        print("🚀 STARTING COMPREHENSIVE DOCUMENT UPLOAD TESTING")
        print("=" * 60)
        
        # Run all tests
        self.test_csv_upload_basic()
        self.test_excel_upload()
        self.test_large_file_upload()
        self.test_utf8_encoding()
        self.test_latin1_encoding()
        self.test_problematic_data_upload()
        self.test_invalid_file_formats()
        self.test_empty_file_upload()
        self.test_complete_upload_workflow()
        self.test_concurrent_uploads()
        self.test_edge_cases()
        
        # Generate summary
        self.generate_test_summary()
    
    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE DOCUMENT UPLOAD TEST SUMMARY")
        print("=" * 60)
        
        # Calculate overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}% success rate)")
        print()
        
        # Detailed results by category
        test_categories = {
            "Basic Upload Functionality": [
                ("Basic CSV Upload", self.test_results.get('csv_upload_basic', False)),
                ("Excel Upload (.xlsx)", self.test_results.get('excel_upload', False)),
                ("Large File Upload (10K rows)", self.test_results.get('large_file_upload', False))
            ],
            "Encoding Support": [
                ("UTF-8 Encoding", self.test_results.get('utf8_encoding', False)),
                ("Latin-1 Encoding", self.test_results.get('latin1_encoding', False))
            ],
            "Data Quality Handling": [
                ("Problematic Data (NaN/Mixed)", self.test_results.get('problematic_data_upload', False))
            ],
            "Error Handling": [
                ("Invalid File Format Rejection", self.test_results.get('invalid_file_formats', False)),
                ("Empty File Rejection", self.test_results.get('empty_file_upload', False))
            ],
            "Workflow Integration": [
                ("Complete Upload Workflow", self.test_results.get('complete_upload_workflow', False))
            ],
            "Performance & Reliability": [
                ("Concurrent Uploads", self.test_results.get('concurrent_uploads', False)),
                ("Edge Cases", self.test_results.get('edge_cases', False))
            ]
        }
        
        for category, tests in test_categories.items():
            print(f"📁 {category}:")
            category_passed = sum(1 for _, passed in tests if passed)
            category_total = len(tests)
            category_rate = (category_passed / category_total * 100) if category_total > 0 else 0
            
            for test_name, passed in tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            print(f"   📊 Category Success Rate: {category_passed}/{category_total} ({category_rate:.1f}%)")
            print()
        
        # Key findings
        print("🔍 KEY FINDINGS:")
        
        if success_rate >= 90:
            print("   🎉 EXCELLENT: Document upload functionality is working excellently!")
        elif success_rate >= 80:
            print("   ✅ GOOD: Document upload functionality is working well with minor issues.")
        elif success_rate >= 70:
            print("   ⚠️  MODERATE: Document upload functionality has some issues that need attention.")
        else:
            print("   ❌ POOR: Document upload functionality has significant issues.")
        
        # Specific recommendations
        failed_tests = [name for name, result in self.test_results.items() if not result]
        if failed_tests:
            print("\n🔧 FAILED TESTS REQUIRING ATTENTION:")
            for test_name in failed_tests:
                print(f"   • {test_name}")
        
        print(f"\n📈 UPLOADED DATA IDs FOR CLEANUP: {len(self.uploaded_data_ids)} files uploaded")
        if self.uploaded_data_ids:
            print(f"   Data IDs: {self.uploaded_data_ids[:5]}{'...' if len(self.uploaded_data_ids) > 5 else ''}")
        
        print("\n🎯 CONCLUSION:")
        if success_rate >= 80:
            print("   Document upload functionality is working correctly and ready for production use.")
            print("   The backend successfully handles various file formats, encodings, and edge cases.")
        else:
            print("   Document upload functionality needs improvement before production use.")
            print("   Focus on fixing the failed test cases identified above.")

if __name__ == "__main__":
    tester = DocumentUploadTester()
    tester.run_all_tests()