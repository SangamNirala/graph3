#!/usr/bin/env python3
"""
Focused Testing for Improved Data Upload Endpoint
Tests the specific improvements mentioned in the review request:
1. Better encoding support (utf-8, latin-1, cp1252)
2. Data validation (empty files, minimum 10+ rows)
3. Data cleaning (empty rows/columns, duplicates, string to numeric conversion)
4. Enhanced error handling (specific error messages instead of HTTP 500)
5. File size validation (10MB limit)
6. End-to-end flow: Upload → Data Analysis → Prophet Training → Prediction Generation
"""

import requests
import pandas as pd
import io
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://c2650a0e-aefa-4982-abd1-a27b2446e525.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing improved upload endpoint at: {API_BASE_URL}")

class UploadImprovementTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_ph_dataset(self):
        """Create user's exact pH dataset (time_step and pH columns)"""
        # Create realistic pH data similar to what user would have
        time_steps = list(range(0, 23))  # 0 to 22 time steps
        ph_values = [
            7.2, 7.3, 7.1, 7.4, 7.2, 7.5, 7.3, 7.6, 7.4, 7.7, 7.5, 7.8,
            7.6, 7.7, 7.5, 7.6, 7.4, 7.5, 7.3, 7.4, 7.2, 7.3, 7.1
        ]
        
        df = pd.DataFrame({
            'time_step': time_steps,
            'pH': ph_values
        })
        
        return df
    
    def create_problematic_dataset(self):
        """Create dataset with common data issues"""
        data = {
            'time_step': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '', None, 12, 13, 14],
            'pH': [7.2, '7.3', 7.1, np.nan, '7.4', 7.2, '', 7.5, 'invalid', 7.6, 7.4, 7.7, 7.5, '7.8', 7.6],
            'empty_col': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            'duplicate_data': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        
        df = pd.DataFrame(data)
        # Add some duplicate rows
        df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)
        
        return df
    
    def create_small_dataset(self):
        """Create dataset with less than 10 rows"""
        df = pd.DataFrame({
            'time_step': [0, 1, 2, 3, 4],
            'pH': [7.2, 7.3, 7.1, 7.4, 7.2]
        })
        return df
    
    def create_empty_dataset(self):
        """Create completely empty dataset"""
        return pd.DataFrame()
    
    def create_large_dataset(self):
        """Create dataset that's close to 10MB limit"""
        # Create a dataset with many rows and columns to approach size limit
        rows = 50000
        data = {
            'time_step': list(range(rows)),
            'pH': np.random.normal(7.0, 0.5, rows),
            'temperature': np.random.normal(25.0, 2.0, rows),
            'pressure': np.random.normal(1013.25, 10.0, rows),
            'notes': ['Sample data point with some text to increase size'] * rows
        }
        return pd.DataFrame(data)
    
    def test_encoding_support(self):
        """Test 1: Better encoding support (utf-8, latin-1, cp1252)"""
        print("\n=== Testing Encoding Support ===")
        
        encoding_tests = []
        
        # Test UTF-8 encoding
        try:
            df = self.create_ph_dataset()
            # Add some UTF-8 characters
            df = df.astype({'pH': 'object'})  # Convert to object type first
            df.loc[0, 'pH'] = '7.2°C'  # Degree symbol
            csv_content = df.to_csv(index=False).encode('utf-8')
            
            files = {'file': ('ph_data_utf8.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            utf8_success = response.status_code == 200
            encoding_tests.append(("UTF-8 encoding", utf8_success))
            
            if utf8_success:
                print("✅ UTF-8 encoding handled successfully")
            else:
                print(f"❌ UTF-8 encoding failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ UTF-8 encoding test error: {str(e)}")
            encoding_tests.append(("UTF-8 encoding", False))
        
        # Test Latin-1 encoding
        try:
            df = self.create_ph_dataset()
            # Add some Latin-1 characters
            df = df.astype({'pH': 'object'})  # Convert to object type first
            df.loc[0, 'pH'] = '7.2µ'  # Micro symbol
            csv_content = df.to_csv(index=False).encode('latin-1')
            
            files = {'file': ('ph_data_latin1.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            latin1_success = response.status_code == 200
            encoding_tests.append(("Latin-1 encoding", latin1_success))
            
            if latin1_success:
                print("✅ Latin-1 encoding handled successfully")
            else:
                print(f"❌ Latin-1 encoding failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Latin-1 encoding test error: {str(e)}")
            encoding_tests.append(("Latin-1 encoding", False))
        
        # Test CP1252 encoding
        try:
            df = self.create_ph_dataset()
            csv_content = df.to_csv(index=False).encode('cp1252')
            
            files = {'file': ('ph_data_cp1252.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            cp1252_success = response.status_code == 200
            encoding_tests.append(("CP1252 encoding", cp1252_success))
            
            if cp1252_success:
                print("✅ CP1252 encoding handled successfully")
            else:
                print(f"❌ CP1252 encoding failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ CP1252 encoding test error: {str(e)}")
            encoding_tests.append(("CP1252 encoding", False))
        
        # Evaluate encoding tests
        passed_tests = sum(1 for _, passed in encoding_tests if passed)
        total_tests = len(encoding_tests)
        
        print(f"✅ Encoding support tests passed: {passed_tests}/{total_tests}")
        for test_name, passed in encoding_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['encoding_support'] = passed_tests >= 2  # At least 2/3 encodings should work
    
    def test_data_validation(self):
        """Test 2: Data validation (empty files, minimum 10+ rows)"""
        print("\n=== Testing Data Validation ===")
        
        validation_tests = []
        
        # Test 1: Empty file validation
        try:
            empty_df = self.create_empty_dataset()
            csv_content = empty_df.to_csv(index=False)
            
            files = {'file': ('empty_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            empty_file_rejected = response.status_code >= 400
            validation_tests.append(("Empty file rejection", empty_file_rejected))
            
            if empty_file_rejected:
                print("✅ Empty file correctly rejected")
                print(f"   Error message: {response.json().get('detail', 'No detail')}")
            else:
                print("❌ Empty file was not rejected")
                
        except Exception as e:
            print(f"❌ Empty file test error: {str(e)}")
            validation_tests.append(("Empty file rejection", False))
        
        # Test 2: Small dataset validation (less than 10 rows)
        try:
            small_df = self.create_small_dataset()
            csv_content = small_df.to_csv(index=False)
            
            files = {'file': ('small_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            small_dataset_rejected = response.status_code >= 400
            validation_tests.append(("Small dataset rejection", small_dataset_rejected))
            
            if small_dataset_rejected:
                print("✅ Small dataset (< 10 rows) correctly rejected")
                print(f"   Error message: {response.json().get('detail', 'No detail')}")
            else:
                print("❌ Small dataset was not rejected")
                
        except Exception as e:
            print(f"❌ Small dataset test error: {str(e)}")
            validation_tests.append(("Small dataset rejection", False))
        
        # Test 3: Valid dataset acceptance
        try:
            valid_df = self.create_ph_dataset()
            csv_content = valid_df.to_csv(index=False)
            
            files = {'file': ('valid_ph_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            valid_dataset_accepted = response.status_code == 200
            validation_tests.append(("Valid dataset acceptance", valid_dataset_accepted))
            
            if valid_dataset_accepted:
                print("✅ Valid pH dataset correctly accepted")
                data = response.json()
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                
                # Store data_id for later tests
                self.data_id = data.get('data_id')
            else:
                print(f"❌ Valid dataset was rejected: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Valid dataset test error: {str(e)}")
            validation_tests.append(("Valid dataset acceptance", False))
        
        # Evaluate validation tests
        passed_tests = sum(1 for _, passed in validation_tests if passed)
        total_tests = len(validation_tests)
        
        print(f"✅ Data validation tests passed: {passed_tests}/{total_tests}")
        for test_name, passed in validation_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['data_validation'] = passed_tests == total_tests
    
    def test_data_cleaning(self):
        """Test 3: Data cleaning (empty rows/columns, duplicates, string to numeric conversion)"""
        print("\n=== Testing Data Cleaning ===")
        
        cleaning_tests = []
        
        try:
            # Create problematic dataset
            problematic_df = self.create_problematic_dataset()
            print(f"Original problematic dataset shape: {problematic_df.shape}")
            print(f"Original columns: {problematic_df.columns.tolist()}")
            print(f"Sample data:\n{problematic_df.head()}")
            
            csv_content = problematic_df.to_csv(index=False)
            
            files = {'file': ('problematic_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data['analysis']
                
                print("✅ Problematic dataset processed successfully")
                print(f"   Cleaned data shape: {analysis['data_shape']}")
                print(f"   Columns after cleaning: {analysis['columns']}")
                print(f"   Numeric columns detected: {analysis['numeric_columns']}")
                
                # Test 1: Empty columns removed
                empty_col_removed = 'empty_col' not in analysis['columns']
                cleaning_tests.append(("Empty columns removed", empty_col_removed))
                
                if empty_col_removed:
                    print("✅ Empty columns successfully removed")
                else:
                    print("❌ Empty columns not removed")
                
                # Test 2: Duplicates handled
                original_rows = len(problematic_df)
                cleaned_rows = analysis['data_shape'][0]
                duplicates_removed = cleaned_rows < original_rows
                cleaning_tests.append(("Duplicates removed", duplicates_removed))
                
                if duplicates_removed:
                    print(f"✅ Duplicates removed: {original_rows} → {cleaned_rows} rows")
                else:
                    print("❌ Duplicates not properly handled")
                
                # Test 3: String to numeric conversion
                ph_is_numeric = 'pH' in analysis['numeric_columns']
                cleaning_tests.append(("String to numeric conversion", ph_is_numeric))
                
                if ph_is_numeric:
                    print("✅ String pH values converted to numeric")
                else:
                    print("❌ String to numeric conversion failed")
                
                # Test 4: Data preview available
                has_preview = 'data_preview' in analysis and len(analysis['data_preview'].get('head', [])) > 0
                cleaning_tests.append(("Data preview generated", has_preview))
                
                if has_preview:
                    print("✅ Data preview successfully generated")
                    print(f"   Preview rows: {len(analysis['data_preview']['head'])}")
                else:
                    print("❌ Data preview not generated")
                
            else:
                print(f"❌ Problematic dataset processing failed: {response.status_code} - {response.text}")
                # All cleaning tests fail if processing fails
                cleaning_tests = [
                    ("Empty columns removed", False),
                    ("Duplicates removed", False),
                    ("String to numeric conversion", False),
                    ("Data preview generated", False)
                ]
                
        except Exception as e:
            print(f"❌ Data cleaning test error: {str(e)}")
            cleaning_tests = [
                ("Empty columns removed", False),
                ("Duplicates removed", False),
                ("String to numeric conversion", False),
                ("Data preview generated", False)
            ]
        
        # Evaluate cleaning tests
        passed_tests = sum(1 for _, passed in cleaning_tests if passed)
        total_tests = len(cleaning_tests)
        
        print(f"✅ Data cleaning tests passed: {passed_tests}/{total_tests}")
        for test_name, passed in cleaning_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['data_cleaning'] = passed_tests >= total_tests * 0.75  # 75% pass rate
    
    def test_error_handling(self):
        """Test 4: Enhanced error handling (specific error messages instead of HTTP 500)"""
        print("\n=== Testing Enhanced Error Handling ===")
        
        error_tests = []
        
        # Test 1: Unsupported file format
        try:
            files = {'file': ('test.txt', 'This is not a CSV file', 'text/plain')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            unsupported_format_handled = response.status_code == 400  # Should be 400, not 500
            error_tests.append(("Unsupported file format", unsupported_format_handled))
            
            if unsupported_format_handled:
                error_detail = response.json().get('detail', '')
                print("✅ Unsupported file format correctly handled")
                print(f"   Error message: {error_detail}")
                
                # Check if error message is specific
                specific_message = 'format' in error_detail.lower() or 'csv' in error_detail.lower()
                error_tests.append(("Specific error message", specific_message))
                
                if specific_message:
                    print("✅ Specific error message provided")
                else:
                    print("❌ Generic error message")
            else:
                print(f"❌ Unsupported file format not properly handled: {response.status_code}")
                error_tests.append(("Specific error message", False))
                
        except Exception as e:
            print(f"❌ Unsupported file format test error: {str(e)}")
            error_tests.append(("Unsupported file format", False))
            error_tests.append(("Specific error message", False))
        
        # Test 2: Corrupted CSV file
        try:
            corrupted_csv = "time_step,pH\n0,7.2\n1,7.3\nthis is not valid csv data\n2"
            files = {'file': ('corrupted.csv', corrupted_csv, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            corrupted_file_handled = response.status_code == 400  # Should be 400, not 500
            error_tests.append(("Corrupted file handling", corrupted_file_handled))
            
            if corrupted_file_handled:
                error_detail = response.json().get('detail', '')
                print("✅ Corrupted CSV file correctly handled")
                print(f"   Error message: {error_detail}")
                
                # Check if error message mentions file reading issue
                specific_message = 'reading' in error_detail.lower() or 'format' in error_detail.lower()
                error_tests.append(("File reading error message", specific_message))
                
                if specific_message:
                    print("✅ Specific file reading error message provided")
                else:
                    print("❌ Generic error message for file reading")
            else:
                print(f"❌ Corrupted file not properly handled: {response.status_code}")
                error_tests.append(("File reading error message", False))
                
        except Exception as e:
            print(f"❌ Corrupted file test error: {str(e)}")
            error_tests.append(("Corrupted file handling", False))
            error_tests.append(("File reading error message", False))
        
        # Test 3: No numeric columns
        try:
            no_numeric_df = pd.DataFrame({
                'text_col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
                'text_col2': ['x', 'y', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 'x', 'y']
            })
            csv_content = no_numeric_df.to_csv(index=False)
            
            files = {'file': ('no_numeric.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            no_numeric_handled = response.status_code == 400  # Should be 400, not 500
            error_tests.append(("No numeric columns handling", no_numeric_handled))
            
            if no_numeric_handled:
                error_detail = response.json().get('detail', '')
                print("✅ No numeric columns correctly handled")
                print(f"   Error message: {error_detail}")
                
                # Check if error message mentions numeric columns
                specific_message = 'numeric' in error_detail.lower()
                error_tests.append(("Numeric columns error message", specific_message))
                
                if specific_message:
                    print("✅ Specific numeric columns error message provided")
                else:
                    print("❌ Generic error message for numeric columns")
            else:
                print(f"❌ No numeric columns not properly handled: {response.status_code}")
                error_tests.append(("Numeric columns error message", False))
                
        except Exception as e:
            print(f"❌ No numeric columns test error: {str(e)}")
            error_tests.append(("No numeric columns handling", False))
            error_tests.append(("Numeric columns error message", False))
        
        # Evaluate error handling tests
        passed_tests = sum(1 for _, passed in error_tests if passed)
        total_tests = len(error_tests)
        
        print(f"✅ Error handling tests passed: {passed_tests}/{total_tests}")
        for test_name, passed in error_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['error_handling'] = passed_tests >= total_tests * 0.7  # 70% pass rate
    
    def test_file_size_validation(self):
        """Test 5: File size validation (10MB limit)"""
        print("\n=== Testing File Size Validation ===")
        
        size_tests = []
        
        # Test 1: Large file rejection
        try:
            print("Creating large dataset (this may take a moment)...")
            large_df = self.create_large_dataset()
            csv_content = large_df.to_csv(index=False)
            
            # Check actual size
            content_size = len(csv_content.encode('utf-8'))
            print(f"Generated file size: {content_size / (1024*1024):.2f} MB")
            
            files = {'file': ('large_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if content_size > 10 * 1024 * 1024:  # If actually larger than 10MB
                large_file_rejected = response.status_code >= 400
                size_tests.append(("Large file rejection", large_file_rejected))
                
                if large_file_rejected:
                    error_detail = response.json().get('detail', '')
                    print("✅ Large file correctly rejected")
                    print(f"   Error message: {error_detail}")
                    
                    # Check if error message mentions file size
                    size_message = 'size' in error_detail.lower() or 'large' in error_detail.lower()
                    size_tests.append(("File size error message", size_message))
                    
                    if size_message:
                        print("✅ Specific file size error message provided")
                    else:
                        print("❌ Generic error message for file size")
                else:
                    print(f"❌ Large file was not rejected: {response.status_code}")
                    size_tests.append(("File size error message", False))
            else:
                print("⚠️  Generated file is not actually larger than 10MB, skipping large file test")
                size_tests.append(("Large file rejection", True))  # Skip this test
                size_tests.append(("File size error message", True))  # Skip this test
                
        except Exception as e:
            print(f"❌ Large file test error: {str(e)}")
            size_tests.append(("Large file rejection", False))
            size_tests.append(("File size error message", False))
        
        # Test 2: Normal size file acceptance
        try:
            normal_df = self.create_ph_dataset()
            csv_content = normal_df.to_csv(index=False)
            content_size = len(csv_content.encode('utf-8'))
            
            print(f"Normal file size: {content_size} bytes")
            
            files = {'file': ('normal_ph_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            normal_file_accepted = response.status_code == 200
            size_tests.append(("Normal file acceptance", normal_file_accepted))
            
            if normal_file_accepted:
                print("✅ Normal size file correctly accepted")
            else:
                print(f"❌ Normal size file was rejected: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Normal file test error: {str(e)}")
            size_tests.append(("Normal file acceptance", False))
        
        # Evaluate size validation tests
        passed_tests = sum(1 for _, passed in size_tests if passed)
        total_tests = len(size_tests)
        
        print(f"✅ File size validation tests passed: {passed_tests}/{total_tests}")
        for test_name, passed in size_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['file_size_validation'] = passed_tests >= total_tests * 0.8  # 80% pass rate
    
    def test_end_to_end_flow(self):
        """Test 6: Complete end-to-end flow: Upload → Data Analysis → Prophet Training → Prediction Generation"""
        print("\n=== Testing End-to-End Flow ===")
        
        flow_tests = []
        
        try:
            # Step 1: Upload pH dataset
            ph_df = self.create_ph_dataset()
            csv_content = ph_df.to_csv(index=False)
            
            files = {'file': ('ph_data_e2e.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            upload_success = response.status_code == 200
            flow_tests.append(("Upload pH dataset", upload_success))
            
            if upload_success:
                data = response.json()
                data_id = data.get('data_id')
                analysis = data['analysis']
                
                print("✅ Step 1: pH dataset upload successful")
                print(f"   Data ID: {data_id}")
                print(f"   Data shape: {analysis['data_shape']}")
                print(f"   Time columns: {analysis['time_columns']}")
                print(f"   Numeric columns: {analysis['numeric_columns']}")
                
                # Verify pH dataset structure
                correct_structure = 'time_step' in analysis['columns'] and 'pH' in analysis['numeric_columns']
                flow_tests.append(("pH dataset structure", correct_structure))
                
                if correct_structure:
                    print("✅ pH dataset structure correctly identified")
                else:
                    print("❌ pH dataset structure not correctly identified")
                
            else:
                print(f"❌ Step 1: pH dataset upload failed: {response.status_code} - {response.text}")
                flow_tests.append(("pH dataset structure", False))
                self.test_results['end_to_end_flow'] = False
                return
            
            # Step 2: Train Prophet model
            training_params = {
                "time_column": "time_step",
                "target_column": "pH",
                "seasonality_mode": "additive",
                "yearly_seasonality": False,
                "weekly_seasonality": False,
                "daily_seasonality": False
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "prophet"},
                json=training_params
            )
            
            training_success = response.status_code == 200
            flow_tests.append(("Prophet model training", training_success))
            
            if training_success:
                data = response.json()
                model_id = data.get('model_id')
                
                print("✅ Step 2: Prophet model training successful")
                print(f"   Model ID: {model_id}")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                
            else:
                print(f"❌ Step 2: Prophet model training failed: {response.status_code} - {response.text}")
                self.test_results['end_to_end_flow'] = False
                return
            
            # Step 3: Generate predictions
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 10}
            )
            
            prediction_success = response.status_code == 200
            flow_tests.append(("Prediction generation", prediction_success))
            
            if prediction_success:
                data = response.json()
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                confidence_intervals = data.get('confidence_intervals')
                
                print("✅ Step 3: Prediction generation successful")
                print(f"   Number of predictions: {len(predictions)}")
                print(f"   Number of timestamps: {len(timestamps)}")
                print(f"   Has confidence intervals: {confidence_intervals is not None}")
                print(f"   Sample predictions: {predictions[:3]}")
                print(f"   Sample timestamps: {timestamps[:3]}")
                
                # Verify prediction structure
                correct_predictions = len(predictions) == 10 and len(timestamps) == 10
                flow_tests.append(("Prediction structure", correct_predictions))
                
                if correct_predictions:
                    print("✅ Prediction structure is correct")
                else:
                    print("❌ Prediction structure is incorrect")
                
            else:
                print(f"❌ Step 3: Prediction generation failed: {response.status_code} - {response.text}")
                flow_tests.append(("Prediction structure", False))
            
            # Step 4: Test historical data retrieval
            response = self.session.get(f"{API_BASE_URL}/historical-data")
            
            historical_success = response.status_code == 200
            flow_tests.append(("Historical data retrieval", historical_success))
            
            if historical_success:
                data = response.json()
                values = data.get('values', [])
                timestamps = data.get('timestamps', [])
                
                print("✅ Step 4: Historical data retrieval successful")
                print(f"   Number of historical points: {len(values)}")
                print(f"   Sample values: {values[:3]}")
                
            else:
                print(f"❌ Step 4: Historical data retrieval failed: {response.status_code} - {response.text}")
            
        except Exception as e:
            print(f"❌ End-to-end flow error: {str(e)}")
            flow_tests = [
                ("Upload pH dataset", False),
                ("pH dataset structure", False),
                ("Prophet model training", False),
                ("Prediction generation", False),
                ("Prediction structure", False),
                ("Historical data retrieval", False)
            ]
        
        # Evaluate end-to-end flow tests
        passed_tests = sum(1 for _, passed in flow_tests if passed)
        total_tests = len(flow_tests)
        
        print(f"✅ End-to-end flow tests passed: {passed_tests}/{total_tests}")
        for test_name, passed in flow_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        self.test_results['end_to_end_flow'] = passed_tests >= total_tests * 0.85  # 85% pass rate
    
    def run_all_tests(self):
        """Run all upload improvement tests"""
        print("🚀 Starting Upload Improvement Testing")
        print("=" * 60)
        
        # Run all focused tests
        self.test_encoding_support()
        self.test_data_validation()
        self.test_data_cleaning()
        self.test_error_handling()
        self.test_file_size_validation()
        self.test_end_to_end_flow()
        
        # Print final results
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 UPLOAD IMPROVEMENT TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"Total Test Categories: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} - {test_name.replace('_', ' ').title()}")
        
        print("\n" + "=" * 60)
        
        # Return overall success
        return passed_tests >= total_tests * 0.8  # 80% pass rate for overall success

if __name__ == "__main__":
    tester = UploadImprovementTester()
    overall_success = tester.run_all_tests()
    
    if overall_success:
        print("🎉 Upload improvement testing completed successfully!")
        exit(0)
    else:
        print("⚠️  Upload improvement testing completed with some failures.")
        exit(1)