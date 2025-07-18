#!/usr/bin/env python3
"""
Comprehensive File Upload Testing for Document Upload Functionality
Tests all scenarios that might prevent users from uploading documents
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
import openpyxl
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://91a5cae0-5aba-4c20-b7d2-24f3a6c5da09.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing file upload at: {API_BASE_URL}")

class FileUploadTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_simple_csv_data(self):
        """Create simple CSV data for basic testing"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        values = np.random.normal(100, 10, 50)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'category': ['A'] * 25 + ['B'] * 25
        })
        
        return df
    
    def create_utf8_csv_data(self):
        """Create CSV data with UTF-8 special characters"""
        cities = ['São Paulo', 'München', 'Zürich', 'Montréal', 'Kraków', 'Москва', '北京', 'العربية'] * 3  # Repeat to get 24 rows
        dates = pd.date_range(start='2023-01-01', periods=len(cities), freq='D')
        values = np.random.normal(100, 10, len(cities))
        descriptions = ['Sunny ☀️', 'Cloudy ☁️', 'Rainy 🌧️', 'Snowy ❄️', 'Windy 💨', 'Foggy 🌫️', 'Hot 🔥', 'Cold 🥶'] * 3
        
        df = pd.DataFrame({
            'date': dates,
            'city': cities,
            'temperature': values,
            'description': descriptions
        })
        
        return df
    
    def create_latin1_csv_data(self):
        """Create CSV data with Latin-1 encoding characters"""
        symbols = ['©', '®', '°', '±', '§', '¶', '½', '¼'] * 3  # Repeat to get 24 rows
        dates = pd.date_range(start='2023-01-01', periods=len(symbols), freq='D')
        values = np.random.normal(50, 5, len(symbols))
        units = ['°C', '±5%', '©2023', '®Brand', '§1.1', '¶Note', '½Cup', '¼Tsp'] * 3
        
        df = pd.DataFrame({
            'date': dates,
            'symbol': symbols,
            'measurement': values,
            'unit': units
        })
        
        return df
    
    def create_mixed_data_types_csv(self):
        """Create CSV with mixed data types and NaN values"""
        # Create 15 rows to meet minimum requirement
        data = {
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'] * 3,
            'numeric_col': [1.5, 2.0, np.nan, 4.5, 'invalid'] * 3,
            'text_col': ['hello', '', 'world', None, 'test'] * 3,
            'mixed_col': [1, 'two', 3.0, np.nan, '5'] * 3,
            'boolean_col': [True, False, 'yes', 'no', np.nan] * 3
        }
        
        df = pd.DataFrame(data)
        return df
    
    def create_large_csv_data(self, size='medium'):
        """Create CSV data of different sizes"""
        if size == 'small':
            rows = 10
        elif size == 'medium':
            rows = 1000
        elif size == 'large':
            rows = 10000
        else:
            rows = 100
            
        dates = pd.date_range(start='2023-01-01', periods=rows, freq='H')
        values = np.random.normal(100, 15, rows)
        categories = np.random.choice(['A', 'B', 'C', 'D'], rows)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'category': categories,
            'random_data': np.random.random(rows)
        })
        
        return df
    
    def create_excel_data(self):
        """Create Excel data for testing"""
        df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'Sales': np.random.normal(1000, 100, 30),
            'Region': ['North', 'South', 'East', 'West'] * 7 + ['North', 'South'],
            'Product': ['A', 'B', 'C'] * 10
        })
        
        return df
    
    def create_empty_csv(self):
        """Create empty CSV file"""
        return pd.DataFrame()
    
    def create_csv_with_special_chars_in_filename(self):
        """Create CSV with special characters in data"""
        # Create 15 rows to meet minimum requirement
        special_chars = ['test@email.com', 'file#name.txt', 'path\\to\\file', 'query?param=value', 'anchor#section'] * 3
        dates = pd.date_range(start='2023-01-01', periods=len(special_chars), freq='D')
        values = list(range(1, len(special_chars) + 1))
        
        df = pd.DataFrame({
            'date': dates,
            'special_chars': special_chars,
            'values': values
        })
        
        return df
    
    def test_simple_csv_upload(self):
        """Test 1: Upload a simple CSV file and verify it processes correctly"""
        print("\n=== Test 1: Simple CSV Upload ===")
        
        try:
            df = self.create_simple_csv_data()
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('simple_data.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get('analysis', {})
                
                print("✅ Simple CSV upload successful")
                print(f"   Data shape: {analysis.get('data_shape')}")
                print(f"   Columns: {analysis.get('columns')}")
                print(f"   Time columns: {analysis.get('time_columns')}")
                print(f"   Numeric columns: {analysis.get('numeric_columns')}")
                
                # Validate analysis
                if ('timestamp' in analysis.get('time_columns', []) and 
                    'value' in analysis.get('numeric_columns', []) and
                    analysis.get('data_shape', [0, 0])[0] == 50):
                    print("✅ Data analysis correct")
                    self.test_results['simple_csv_upload'] = True
                else:
                    print("❌ Data analysis incorrect")
                    self.test_results['simple_csv_upload'] = False
                    
            else:
                print(f"❌ Simple CSV upload failed: {response.status_code} - {response.text}")
                self.test_results['simple_csv_upload'] = False
                
        except Exception as e:
            print(f"❌ Simple CSV upload error: {str(e)}")
            self.test_results['simple_csv_upload'] = False
    
    def test_utf8_encoding_upload(self):
        """Test 2a: Test file upload with UTF-8 encoding"""
        print("\n=== Test 2a: UTF-8 Encoding Upload ===")
        
        try:
            df = self.create_utf8_csv_data()
            csv_content = df.to_csv(index=False, encoding='utf-8')
            
            files = {
                'file': ('utf8_data.csv', csv_content.encode('utf-8'), 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get('analysis', {})
                
                print("✅ UTF-8 encoding upload successful")
                print(f"   Data shape: {analysis.get('data_shape')}")
                
                # Check if special characters are preserved
                preview = analysis.get('data_preview', {}).get('head', [])
                if preview and any('São Paulo' in str(row.get('city', '')) for row in preview):
                    print("✅ UTF-8 special characters preserved")
                    self.test_results['utf8_encoding_upload'] = True
                else:
                    print("❌ UTF-8 special characters not preserved")
                    self.test_results['utf8_encoding_upload'] = False
                    
            else:
                print(f"❌ UTF-8 encoding upload failed: {response.status_code} - {response.text}")
                self.test_results['utf8_encoding_upload'] = False
                
        except Exception as e:
            print(f"❌ UTF-8 encoding upload error: {str(e)}")
            self.test_results['utf8_encoding_upload'] = False
    
    def test_latin1_encoding_upload(self):
        """Test 2b: Test file upload with Latin-1 encoding"""
        print("\n=== Test 2b: Latin-1 Encoding Upload ===")
        
        try:
            df = self.create_latin1_csv_data()
            csv_content = df.to_csv(index=False, encoding='latin-1')
            
            files = {
                'file': ('latin1_data.csv', csv_content.encode('latin-1'), 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get('analysis', {})
                
                print("✅ Latin-1 encoding upload successful")
                print(f"   Data shape: {analysis.get('data_shape')}")
                
                # Check if Latin-1 characters are preserved
                preview = analysis.get('data_preview', {}).get('head', [])
                if preview and any('©' in str(row.get('symbol', '')) for row in preview):
                    print("✅ Latin-1 special characters preserved")
                    self.test_results['latin1_encoding_upload'] = True
                else:
                    print("❌ Latin-1 special characters not preserved")
                    self.test_results['latin1_encoding_upload'] = False
                    
            else:
                print(f"❌ Latin-1 encoding upload failed: {response.status_code} - {response.text}")
                self.test_results['latin1_encoding_upload'] = False
                
        except Exception as e:
            print(f"❌ Latin-1 encoding upload error: {str(e)}")
            self.test_results['latin1_encoding_upload'] = False
    
    def test_mixed_data_types_upload(self):
        """Test 3: Test file upload with mixed data types and NaN values"""
        print("\n=== Test 3: Mixed Data Types and NaN Values Upload ===")
        
        try:
            df = self.create_mixed_data_types_csv()
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('mixed_data.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get('analysis', {})
                
                print("✅ Mixed data types upload successful")
                print(f"   Data shape: {analysis.get('data_shape')}")
                print(f"   Numeric columns: {analysis.get('numeric_columns')}")
                
                # Check missing values handling
                missing_values = analysis.get('data_preview', {}).get('missing_values', {})
                total_missing = sum(missing_values.values()) if missing_values else 0
                
                if total_missing > 0:
                    print(f"✅ Missing values detected and handled: {total_missing} total")
                    self.test_results['mixed_data_types_upload'] = True
                else:
                    print("❌ Missing values not properly detected")
                    self.test_results['mixed_data_types_upload'] = False
                    
            else:
                print(f"❌ Mixed data types upload failed: {response.status_code} - {response.text}")
                self.test_results['mixed_data_types_upload'] = False
                
        except Exception as e:
            print(f"❌ Mixed data types upload error: {str(e)}")
            self.test_results['mixed_data_types_upload'] = False
    
    def test_different_file_sizes(self):
        """Test 4: Test file upload with different file sizes"""
        print("\n=== Test 4: Different File Sizes Upload ===")
        
        size_tests = []
        
        for size in ['small', 'medium', 'large']:
            try:
                print(f"\n   Testing {size} file size...")
                df = self.create_large_csv_data(size)
                csv_content = df.to_csv(index=False)
                
                files = {
                    'file': (f'{size}_data.csv', csv_content, 'text/csv')
                }
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    analysis = data.get('analysis', {})
                    data_shape = analysis.get('data_shape', [0, 0])
                    
                    print(f"   ✅ {size.capitalize()} file upload successful - {data_shape[0]} rows")
                    size_tests.append(True)
                else:
                    print(f"   ❌ {size.capitalize()} file upload failed: {response.status_code}")
                    size_tests.append(False)
                    
            except Exception as e:
                print(f"   ❌ {size.capitalize()} file upload error: {str(e)}")
                size_tests.append(False)
        
        # Overall result
        passed_tests = sum(size_tests)
        total_tests = len(size_tests)
        
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            print(f"✅ File size tests passed: {passed_tests}/{total_tests}")
            self.test_results['different_file_sizes'] = True
        else:
            print(f"❌ File size tests failed: {passed_tests}/{total_tests}")
            self.test_results['different_file_sizes'] = False
    
    def test_excel_file_upload(self):
        """Test 5: Test file upload with Excel files (.xlsx)"""
        print("\n=== Test 5: Excel File Upload ===")
        
        try:
            df = self.create_excel_data()
            
            # Create temporary Excel file
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                df.to_excel(tmp_file.name, index=False)
                
                # Read the Excel file content
                with open(tmp_file.name, 'rb') as excel_file:
                    excel_content = excel_file.read()
                
                files = {
                    'file': ('excel_data.xlsx', excel_content, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                }
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    analysis = data.get('analysis', {})
                    
                    print("✅ Excel file upload successful")
                    print(f"   Data shape: {analysis.get('data_shape')}")
                    print(f"   Columns: {analysis.get('columns')}")
                    
                    # Validate Excel data processing
                    if ('Date' in analysis.get('time_columns', []) and 
                        'Sales' in analysis.get('numeric_columns', []) and
                        analysis.get('data_shape', [0, 0])[0] == 30):
                        print("✅ Excel data analysis correct")
                        self.test_results['excel_file_upload'] = True
                    else:
                        print("❌ Excel data analysis incorrect")
                        self.test_results['excel_file_upload'] = False
                        
                else:
                    print(f"❌ Excel file upload failed: {response.status_code} - {response.text}")
                    self.test_results['excel_file_upload'] = False
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
        except Exception as e:
            print(f"❌ Excel file upload error: {str(e)}")
            self.test_results['excel_file_upload'] = False
    
    def test_invalid_file_formats(self):
        """Test 6: Test error handling for invalid file formats"""
        print("\n=== Test 6: Invalid File Formats ===")
        
        invalid_formats = [
            ('test.txt', 'This is a text file', 'text/plain'),
            ('test.json', '{"key": "value"}', 'application/json'),
            ('test.pdf', b'%PDF-1.4 fake pdf content', 'application/pdf'),
            ('test.doc', b'fake word document', 'application/msword')
        ]
        
        format_tests = []
        
        for filename, content, content_type in invalid_formats:
            try:
                files = {
                    'file': (filename, content, content_type)
                }
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code >= 400:
                    print(f"   ✅ {filename} correctly rejected (status: {response.status_code})")
                    format_tests.append(True)
                else:
                    print(f"   ❌ {filename} incorrectly accepted (status: {response.status_code})")
                    format_tests.append(False)
                    
            except Exception as e:
                print(f"   ❌ Error testing {filename}: {str(e)}")
                format_tests.append(False)
        
        # Overall result
        passed_tests = sum(format_tests)
        total_tests = len(format_tests)
        
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            print(f"✅ Invalid format tests passed: {passed_tests}/{total_tests}")
            self.test_results['invalid_file_formats'] = True
        else:
            print(f"❌ Invalid format tests failed: {passed_tests}/{total_tests}")
            self.test_results['invalid_file_formats'] = False
    
    def test_empty_file_upload(self):
        """Test 7: Test upload with empty files"""
        print("\n=== Test 7: Empty File Upload ===")
        
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
                print("❌ Empty file incorrectly accepted")
                empty_file_test = False
            
            # Test CSV with only headers
            files = {
                'file': ('headers_only.csv', 'col1,col2,col3\n', 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code >= 400:
                print("✅ Headers-only file correctly rejected")
                headers_only_test = True
            else:
                print("❌ Headers-only file incorrectly accepted")
                headers_only_test = False
            
            # Test DataFrame that becomes empty after cleaning
            df = self.create_empty_csv()
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('empty_dataframe.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code >= 400:
                print("✅ Empty DataFrame correctly rejected")
                empty_df_test = True
            else:
                print("❌ Empty DataFrame incorrectly accepted")
                empty_df_test = False
            
            # Overall result
            empty_tests = [empty_file_test, headers_only_test, empty_df_test]
            passed_tests = sum(empty_tests)
            
            if passed_tests >= 2:  # At least 2/3 should pass
                print(f"✅ Empty file tests passed: {passed_tests}/3")
                self.test_results['empty_file_upload'] = True
            else:
                print(f"❌ Empty file tests failed: {passed_tests}/3")
                self.test_results['empty_file_upload'] = False
                
        except Exception as e:
            print(f"❌ Empty file upload error: {str(e)}")
            self.test_results['empty_file_upload'] = False
    
    def test_special_characters_upload(self):
        """Test 8: Test upload with files that have special characters"""
        print("\n=== Test 8: Special Characters Upload ===")
        
        try:
            df = self.create_csv_with_special_chars_in_filename()
            csv_content = df.to_csv(index=False)
            
            # Test with special characters in filename
            special_filename = 'test_file_with_special_chars_@#$%^&()_+.csv'
            
            files = {
                'file': (special_filename, csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get('analysis', {})
                
                print("✅ File with special characters in filename uploaded successfully")
                print(f"   Data shape: {analysis.get('data_shape')}")
                
                # Check if special characters in data are handled
                preview = analysis.get('data_preview', {}).get('head', [])
                if preview and any('@' in str(row.get('special_chars', '')) for row in preview):
                    print("✅ Special characters in data preserved")
                    self.test_results['special_characters_upload'] = True
                else:
                    print("❌ Special characters in data not preserved")
                    self.test_results['special_characters_upload'] = False
                    
            else:
                print(f"❌ Special characters upload failed: {response.status_code} - {response.text}")
                self.test_results['special_characters_upload'] = False
                
        except Exception as e:
            print(f"❌ Special characters upload error: {str(e)}")
            self.test_results['special_characters_upload'] = False
    
    def test_data_analysis_and_suggestions(self):
        """Test 9: Verify the data analysis and parameter suggestion works correctly"""
        print("\n=== Test 9: Data Analysis and Parameter Suggestions ===")
        
        try:
            df = self.create_simple_csv_data()
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('analysis_test.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get('analysis', {})
                
                print("✅ Data analysis successful")
                
                # Check analysis components
                analysis_tests = []
                
                # Test column detection
                columns = analysis.get('columns', [])
                if len(columns) == 3 and 'timestamp' in columns:
                    print("✅ Column detection correct")
                    analysis_tests.append(True)
                else:
                    print("❌ Column detection incorrect")
                    analysis_tests.append(False)
                
                # Test time column detection
                time_columns = analysis.get('time_columns', [])
                if 'timestamp' in time_columns:
                    print("✅ Time column detection correct")
                    analysis_tests.append(True)
                else:
                    print("❌ Time column detection incorrect")
                    analysis_tests.append(False)
                
                # Test numeric column detection
                numeric_columns = analysis.get('numeric_columns', [])
                if 'value' in numeric_columns:
                    print("✅ Numeric column detection correct")
                    analysis_tests.append(True)
                else:
                    print("❌ Numeric column detection incorrect")
                    analysis_tests.append(False)
                
                # Test data preview
                data_preview = analysis.get('data_preview', {})
                if 'head' in data_preview and 'describe' in data_preview:
                    print("✅ Data preview generated")
                    analysis_tests.append(True)
                else:
                    print("❌ Data preview not generated")
                    analysis_tests.append(False)
                
                # Test parameter suggestions
                suggested_params = analysis.get('suggested_parameters', {})
                if (suggested_params.get('time_column') == 'timestamp' and
                    suggested_params.get('target_column') == 'value'):
                    print("✅ Parameter suggestions correct")
                    analysis_tests.append(True)
                else:
                    print("❌ Parameter suggestions incorrect")
                    analysis_tests.append(False)
                
                # Overall result
                passed_tests = sum(analysis_tests)
                total_tests = len(analysis_tests)
                
                if passed_tests >= total_tests * 0.8:  # 80% pass rate
                    print(f"✅ Data analysis tests passed: {passed_tests}/{total_tests}")
                    self.test_results['data_analysis_and_suggestions'] = True
                else:
                    print(f"❌ Data analysis tests failed: {passed_tests}/{total_tests}")
                    self.test_results['data_analysis_and_suggestions'] = False
                    
            else:
                print(f"❌ Data analysis failed: {response.status_code} - {response.text}")
                self.test_results['data_analysis_and_suggestions'] = False
                
        except Exception as e:
            print(f"❌ Data analysis error: {str(e)}")
            self.test_results['data_analysis_and_suggestions'] = False
    
    def test_complete_upload_flow(self):
        """Test 10: Test the complete upload flow from frontend perspective"""
        print("\n=== Test 10: Complete Upload Flow ===")
        
        try:
            flow_tests = []
            
            # Step 1: Upload file
            df = self.create_simple_csv_data()
            csv_content = df.to_csv(index=False)
            
            files = {
                'file': ('flow_test.csv', csv_content, 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id')
                analysis = data.get('analysis', {})
                
                print("✅ Step 1: File upload successful")
                flow_tests.append(True)
                
                # Step 2: Verify data analysis is complete
                if (analysis.get('time_columns') and 
                    analysis.get('numeric_columns') and
                    analysis.get('suggested_parameters')):
                    print("✅ Step 2: Data analysis complete")
                    flow_tests.append(True)
                else:
                    print("❌ Step 2: Data analysis incomplete")
                    flow_tests.append(False)
                
                # Step 3: Test model training with uploaded data
                training_params = {
                    "time_column": "timestamp",
                    "target_column": "value",
                    "seasonality_mode": "additive"
                }
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": "prophet"},
                    json=training_params
                )
                
                if response.status_code == 200:
                    model_data = response.json()
                    model_id = model_data.get('model_id')
                    
                    print("✅ Step 3: Model training successful")
                    flow_tests.append(True)
                    
                    # Step 4: Test prediction generation
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-prediction",
                        params={"model_id": model_id, "steps": 10}
                    )
                    
                    if response.status_code == 200:
                        pred_data = response.json()
                        predictions = pred_data.get('predictions', [])
                        
                        if len(predictions) == 10:
                            print("✅ Step 4: Prediction generation successful")
                            flow_tests.append(True)
                        else:
                            print("❌ Step 4: Prediction generation incomplete")
                            flow_tests.append(False)
                    else:
                        print("❌ Step 4: Prediction generation failed")
                        flow_tests.append(False)
                        
                else:
                    print("❌ Step 3: Model training failed")
                    flow_tests.append(False)
                    flow_tests.append(False)  # Skip prediction test
                    
            else:
                print("❌ Step 1: File upload failed")
                flow_tests.extend([False, False, False])  # All subsequent steps fail
            
            # Overall result
            passed_tests = sum(flow_tests)
            total_tests = len(flow_tests)
            
            if passed_tests >= total_tests * 0.8:  # 80% pass rate
                print(f"✅ Complete upload flow tests passed: {passed_tests}/{total_tests}")
                self.test_results['complete_upload_flow'] = True
            else:
                print(f"❌ Complete upload flow tests failed: {passed_tests}/{total_tests}")
                self.test_results['complete_upload_flow'] = False
                
        except Exception as e:
            print(f"❌ Complete upload flow error: {str(e)}")
            self.test_results['complete_upload_flow'] = False
    
    def run_all_tests(self):
        """Run all file upload tests"""
        print("🚀 Starting Comprehensive File Upload Testing")
        print("=" * 60)
        
        # Run all test scenarios
        self.test_simple_csv_upload()
        self.test_utf8_encoding_upload()
        self.test_latin1_encoding_upload()
        self.test_mixed_data_types_upload()
        self.test_different_file_sizes()
        self.test_excel_file_upload()
        self.test_invalid_file_formats()
        self.test_empty_file_upload()
        self.test_special_characters_upload()
        self.test_data_analysis_and_suggestions()
        self.test_complete_upload_flow()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 FILE UPLOAD TEST SUMMARY")
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
            print(f"  {status} {test_name.replace('_', ' ').title()}")
        
        # Identify critical issues
        critical_failures = []
        for test_name, result in self.test_results.items():
            if not result and test_name in ['simple_csv_upload', 'complete_upload_flow', 'data_analysis_and_suggestions']:
                critical_failures.append(test_name.replace('_', ' ').title())
        
        if critical_failures:
            print(f"\n🚨 CRITICAL ISSUES IDENTIFIED:")
            for failure in critical_failures:
                print(f"  - {failure}")
        else:
            print(f"\n✅ No critical issues identified in file upload functionality")
        
        return passed_tests, total_tests

if __name__ == "__main__":
    tester = FileUploadTester()
    tester.run_all_tests()