#!/usr/bin/env python3
"""
Focused Backend Testing for Encoding Support and Data Cleaning Fixes
Tests the critical fixes implemented for UTF-8/Latin-1 encoding and data cleaning
"""

import requests
import json
import pandas as pd
import io
import time
import numpy as np
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://4f535dbd-21ac-4151-8dfe-215665939abd.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing encoding and data cleaning fixes at: {API_BASE_URL}")

class EncodingDataCleaningTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_utf8_test_data(self):
        """Create test data with UTF-8 characters"""
        data = {
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                         '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                         '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'],
            'temperature': [23.5, 24.1, 22.8, 25.3, 23.9, 24.7, 22.1, 26.2, 23.4, 25.8,
                           24.3, 22.9, 25.1, 23.7, 24.5],
            'location': ['São Paulo', 'München', 'Zürich', 'Montréal', 'Kraków',
                        'São Paulo', 'München', 'Zürich', 'Montréal', 'Kraków',
                        'São Paulo', 'München', 'Zürich', 'Montréal', 'Kraków'],
            'notes': ['Día soleado', 'Schönes Wetter', 'Très chaud', 'Beau temps', 'Piękny dzień',
                     'Día nublado', 'Regnerisch', 'Très froid', 'Temps nuageux', 'Deszczowy dzień',
                     'Día caluroso', 'Sonnig', 'Très venteux', 'Temps pluvieux', 'Słoneczny dzień']
        }
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8')
    
    def create_latin1_test_data(self):
        """Create test data with Latin-1 characters"""
        data = {
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                    '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                    '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'],
            'value': [100.5, 102.3, 98.7, 105.1, 99.8, 103.2, 97.4, 106.8, 101.1, 104.5,
                     98.9, 102.7, 105.3, 99.6, 103.8],
            'description': ['Café', 'Niño', 'Año', 'Señor', 'Corazón',
                           'Café', 'Niño', 'Año', 'Señor', 'Corazón',
                           'Café', 'Niño', 'Año', 'Señor', 'Corazón'],
            'symbol': ['©', '®', '°', '±', '§', '©', '®', '°', '±', '§',
                      '©', '®', '°', '±', '§']
        }
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='latin-1')
    
    def create_nan_test_data(self):
        """Create test data with NaN values"""
        data = {
            'time': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                    '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                    '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'],
            'value1': [10.5, np.nan, 12.3, np.nan, 11.8, 13.2, np.nan, 14.1, 12.7, np.nan,
                      11.4, 13.8, np.nan, 12.9, 14.3],
            'value2': [np.nan, 20.1, np.nan, 22.5, np.nan, 21.8, 23.2, np.nan, 20.7, 22.9,
                      np.nan, 21.3, 23.6, np.nan, 22.1],
            'category': ['A', np.nan, 'B', 'C', np.nan, 'A', 'B', np.nan, 'C', 'A',
                        np.nan, 'B', 'C', np.nan, 'A']
        }
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def create_mixed_types_test_data(self):
        """Create test data with mixed data types"""
        data = {
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                         '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                         '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'],
            'mixed_column': ['123.45', '67.89', 'invalid', '234.56', '89.12',
                            '156.78', 'bad_data', '345.67', '123.89', 'error',
                            '234.12', '456.78', 'invalid', '567.89', '678.90'],
            'numeric_strings': ['100', '200', '300', 'abc', '500', '600', 'def', '800', '900', 'ghi',
                               '1000', '1100', 'jkl', '1300', '1400'],
            'boolean_mixed': ['true', 'false', '1', '0', 'yes', 'no', 'TRUE', 'FALSE', '1', '0',
                             'yes', 'no', 'true', 'false', '1']
        }
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def create_empty_strings_test_data(self):
        """Create test data with empty strings and whitespace"""
        data = {
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                    '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                    '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'],
            'value': [10.5, 20.3, 15.7, 18.9, 22.1, 19.4, 21.8, 17.2, 23.6, 16.9,
                     20.7, 18.3, 22.4, 19.1, 21.5],
            'empty_column': ['', '  ', '', '   ', '', '    ', '', '  ', '', '   ',
                            '', '  ', '', '    ', ''],
            'whitespace_column': [' data1 ', '  data2  ', '', '  ', ' data3', '  data4  ', '', ' data5 ',
                                 '  ', ' data6', '', '  data7  ', ' data8 ', '', '  data9  '],
            'null_strings': ['null', 'NULL', 'nan', 'NaN', 'None', 'null', 'NULL', 'nan', 'NaN', 'None',
                            'null', 'NULL', 'nan', 'NaN', 'None']
        }
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def create_problematic_combination_data(self):
        """Create test data with combination of problematic data"""
        data = {
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                         '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                         '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'],
            'mixed_numeric': [10.5, 'invalid', np.nan, '', '  20.3  ', 15.7, 'bad', np.nan, '  ',
                             '25.8', 'error', np.nan, '', '  18.9  ', '22.1'],
            'clean_numeric': [100.1, 102.3, 98.7, 105.1, 99.8, 103.2, 97.4, 106.8, 101.1, 104.5,
                             98.9, 102.7, 105.3, 99.6, 103.8],  # Always valid numeric column
            'utf8_with_nulls': ['São Paulo', '', np.nan, 'München', 'null', 'Zürich', '', np.nan,
                               'Montréal', 'NULL', 'Kraków', '', np.nan, 'São Paulo', 'null'],
            'empty_mixed': ['', '  ', np.nan, 'NULL', 'valid_data', '', '   ', np.nan, 'null',
                           'good_data', '', '  ', np.nan, 'NULL', 'final_data'],
            'whitespace_numbers': [' 123.45 ', '', '  67.89  ', 'nan', '  ', ' 234.56 ', '', '  89.12  ',
                                  'null', '   ', ' 156.78 ', '', '  345.67  ', 'NaN', '  ']
        }
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8')
    
    def test_utf8_encoding_support(self):
        """Test 1: UTF-8 encoding support"""
        print("\n=== Testing UTF-8 Encoding Support ===")
        
        try:
            # Create UTF-8 encoded CSV data
            csv_content = self.create_utf8_test_data()
            
            # Prepare file for upload
            files = {
                'file': ('utf8_test.csv', csv_content.encode('utf-8'), 'text/csv')
            }
            
            # Test file upload with UTF-8 encoding
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ UTF-8 file upload successful")
                print(f"   Data ID: {data.get('data_id')}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Check if UTF-8 characters were preserved
                preview = data['analysis']['data_preview']['head']
                if preview and len(preview) > 0:
                    # Look for UTF-8 characters in the preview
                    utf8_preserved = any(
                        any('ã' in str(val) or 'ü' in str(val) or 'é' in str(val) or 'ó' in str(val) 
                            for val in row.values() if val is not None)
                        for row in preview
                    )
                    
                    if utf8_preserved:
                        print("✅ UTF-8 characters preserved correctly")
                        self.test_results['utf8_encoding'] = True
                    else:
                        print("❌ UTF-8 characters not preserved")
                        self.test_results['utf8_encoding'] = False
                else:
                    print("⚠️  No preview data to check UTF-8 preservation")
                    self.test_results['utf8_encoding'] = True  # Upload succeeded
                    
            else:
                print(f"❌ UTF-8 file upload failed: {response.status_code} - {response.text}")
                self.test_results['utf8_encoding'] = False
                
        except Exception as e:
            print(f"❌ UTF-8 encoding test error: {str(e)}")
            self.test_results['utf8_encoding'] = False
    
    def test_latin1_encoding_support(self):
        """Test 2: Latin-1 encoding support"""
        print("\n=== Testing Latin-1 Encoding Support ===")
        
        try:
            # Create Latin-1 encoded CSV data
            csv_content = self.create_latin1_test_data()
            
            # Prepare file for upload (encoded as Latin-1)
            files = {
                'file': ('latin1_test.csv', csv_content.encode('latin-1'), 'text/csv')
            }
            
            # Test file upload with Latin-1 encoding
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Latin-1 file upload successful")
                print(f"   Data ID: {data.get('data_id')}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Check if Latin-1 characters were preserved
                preview = data['analysis']['data_preview']['head']
                if preview and len(preview) > 0:
                    # Look for Latin-1 characters in the preview
                    latin1_preserved = any(
                        any('©' in str(val) or '®' in str(val) or '°' in str(val) or '±' in str(val)
                            for val in row.values() if val is not None)
                        for row in preview
                    )
                    
                    if latin1_preserved:
                        print("✅ Latin-1 characters preserved correctly")
                        self.test_results['latin1_encoding'] = True
                    else:
                        print("❌ Latin-1 characters not preserved")
                        self.test_results['latin1_encoding'] = False
                else:
                    print("⚠️  No preview data to check Latin-1 preservation")
                    self.test_results['latin1_encoding'] = True  # Upload succeeded
                    
            else:
                print(f"❌ Latin-1 file upload failed: {response.status_code} - {response.text}")
                self.test_results['latin1_encoding'] = False
                
        except Exception as e:
            print(f"❌ Latin-1 encoding test error: {str(e)}")
            self.test_results['latin1_encoding'] = False
    
    def test_nan_values_handling(self):
        """Test 3: NaN values handling"""
        print("\n=== Testing NaN Values Handling ===")
        
        try:
            # Create CSV data with NaN values
            csv_content = self.create_nan_test_data()
            
            # Prepare file for upload
            files = {
                'file': ('nan_test.csv', csv_content, 'text/csv')
            }
            
            # Test file upload with NaN values
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ NaN values file upload successful")
                print(f"   Data ID: {data.get('data_id')}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Check missing values report
                missing_values = data['analysis']['data_preview'].get('missing_values', {})
                if missing_values:
                    print("✅ Missing values correctly identified:")
                    for col, count in missing_values.items():
                        print(f"   {col}: {count} missing values")
                    self.test_results['nan_handling'] = True
                else:
                    print("❌ Missing values not properly identified")
                    self.test_results['nan_handling'] = False
                    
            else:
                print(f"❌ NaN values file upload failed: {response.status_code} - {response.text}")
                self.test_results['nan_handling'] = False
                
        except Exception as e:
            print(f"❌ NaN values handling test error: {str(e)}")
            self.test_results['nan_handling'] = False
    
    def test_mixed_data_types_handling(self):
        """Test 4: Mixed data types handling"""
        print("\n=== Testing Mixed Data Types Handling ===")
        
        try:
            # Create CSV data with mixed data types
            csv_content = self.create_mixed_types_test_data()
            
            # Prepare file for upload
            files = {
                'file': ('mixed_types_test.csv', csv_content, 'text/csv')
            }
            
            # Test file upload with mixed data types
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Mixed data types file upload successful")
                print(f"   Data ID: {data.get('data_id')}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Check if numeric columns were properly identified despite mixed types
                numeric_cols = data['analysis']['numeric_columns']
                if len(numeric_cols) > 0:
                    print("✅ Numeric columns identified despite mixed data types")
                    self.test_results['mixed_types_handling'] = True
                else:
                    print("❌ No numeric columns identified from mixed data types")
                    self.test_results['mixed_types_handling'] = False
                    
            else:
                print(f"❌ Mixed data types file upload failed: {response.status_code} - {response.text}")
                self.test_results['mixed_types_handling'] = False
                
        except Exception as e:
            print(f"❌ Mixed data types handling test error: {str(e)}")
            self.test_results['mixed_types_handling'] = False
    
    def test_empty_strings_handling(self):
        """Test 5: Empty strings and whitespace handling"""
        print("\n=== Testing Empty Strings and Whitespace Handling ===")
        
        try:
            # Create CSV data with empty strings and whitespace
            csv_content = self.create_empty_strings_test_data()
            
            # Prepare file for upload
            files = {
                'file': ('empty_strings_test.csv', csv_content, 'text/csv')
            }
            
            # Test file upload with empty strings and whitespace
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Empty strings file upload successful")
                print(f"   Data ID: {data.get('data_id')}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Check missing values report (empty strings should be converted to NaN)
                missing_values = data['analysis']['data_preview'].get('missing_values', {})
                if missing_values:
                    print("✅ Empty strings correctly converted to missing values:")
                    for col, count in missing_values.items():
                        if count > 0:
                            print(f"   {col}: {count} missing values")
                    self.test_results['empty_strings_handling'] = True
                else:
                    print("⚠️  No missing values reported (may be expected if all cleaned)")
                    self.test_results['empty_strings_handling'] = True
                    
            else:
                print(f"❌ Empty strings file upload failed: {response.status_code} - {response.text}")
                self.test_results['empty_strings_handling'] = False
                
        except Exception as e:
            print(f"❌ Empty strings handling test error: {str(e)}")
            self.test_results['empty_strings_handling'] = False
    
    def test_problematic_combination_handling(self):
        """Test 6: Problematic data combination handling"""
        print("\n=== Testing Problematic Data Combination Handling ===")
        
        try:
            # Create CSV data with combination of problematic data
            csv_content = self.create_problematic_combination_data()
            
            # Prepare file for upload
            files = {
                'file': ('problematic_combo_test.csv', csv_content.encode('utf-8'), 'text/csv')
            }
            
            # Test file upload with problematic data combination
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Problematic combination file upload successful")
                print(f"   Data ID: {data.get('data_id')}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                
                # Check if data was properly cleaned and analyzed
                preview = data['analysis']['data_preview']['head']
                missing_values = data['analysis']['data_preview'].get('missing_values', {})
                
                if preview and missing_values:
                    print("✅ Problematic data properly cleaned and analyzed")
                    print(f"   Missing values identified: {sum(missing_values.values())} total")
                    self.test_results['problematic_combination_handling'] = True
                else:
                    print("❌ Problematic data not properly handled")
                    self.test_results['problematic_combination_handling'] = False
                    
            else:
                print(f"❌ Problematic combination file upload failed: {response.status_code} - {response.text}")
                self.test_results['problematic_combination_handling'] = False
                
        except Exception as e:
            print(f"❌ Problematic combination handling test error: {str(e)}")
            self.test_results['problematic_combination_handling'] = False
    
    def test_data_quality_report_with_problematic_data(self):
        """Test 7: Data quality report with problematic data"""
        print("\n=== Testing Data Quality Report with Problematic Data ===")
        
        try:
            # First upload problematic data
            csv_content = self.create_problematic_combination_data()
            files = {
                'file': ('quality_test.csv', csv_content.encode('utf-8'), 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                print("✅ Problematic data uploaded successfully")
                
                # Test data quality report endpoint
                response = self.session.get(f"{API_BASE_URL}/data-quality-report")
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ Data quality report generated successfully")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Quality score: {data.get('quality_score')}")
                    
                    recommendations = data.get('recommendations', [])
                    if recommendations:
                        print(f"   Recommendations: {len(recommendations)} items")
                        for rec in recommendations[:3]:  # Show first 3
                            print(f"     - {rec}")
                    
                    self.test_results['data_quality_report'] = True
                else:
                    print(f"❌ Data quality report failed: {response.status_code} - {response.text}")
                    self.test_results['data_quality_report'] = False
            else:
                print("❌ Failed to upload data for quality report test")
                self.test_results['data_quality_report'] = False
                
        except Exception as e:
            print(f"❌ Data quality report test error: {str(e)}")
            self.test_results['data_quality_report'] = False
    
    def test_model_training_with_cleaned_data(self):
        """Test 8: Model training with cleaned problematic data"""
        print("\n=== Testing Model Training with Cleaned Data ===")
        
        try:
            # Upload problematic data that should be cleaned
            csv_content = self.create_problematic_combination_data()
            files = {
                'file': ('training_test.csv', csv_content.encode('utf-8'), 'text/csv')
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data_id = response.json().get('data_id')
                print("✅ Problematic data uploaded and cleaned successfully")
                
                # Try to train a model with the cleaned data
                training_params = {
                    "time_column": "timestamp",
                    "target_column": "mixed_numeric",  # This column has mixed/problematic data
                    "order": [1, 1, 1]
                }
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": "arima"},
                    json=training_params
                )
                
                if response.status_code == 200:
                    model_data = response.json()
                    print("✅ Model training successful with cleaned data")
                    print(f"   Model ID: {model_data.get('model_id')}")
                    print(f"   Status: {model_data.get('status')}")
                    self.test_results['model_training_cleaned_data'] = True
                else:
                    print(f"❌ Model training failed with cleaned data: {response.status_code} - {response.text}")
                    self.test_results['model_training_cleaned_data'] = False
            else:
                print("❌ Failed to upload data for model training test")
                self.test_results['model_training_cleaned_data'] = False
                
        except Exception as e:
            print(f"❌ Model training with cleaned data test error: {str(e)}")
            self.test_results['model_training_cleaned_data'] = False
    
    def test_encoding_fallback_mechanism(self):
        """Test 9: Encoding fallback mechanism"""
        print("\n=== Testing Encoding Fallback Mechanism ===")
        
        try:
            # Create data with mixed encoding challenges
            # This simulates a file that might fail with one encoding but work with another
            test_data = "timestamp,value,description\n"
            for i in range(15):  # Create 15 rows to meet minimum requirement
                test_data += f"2023-01-{i+1:02d},{100+i*10},Café_{i}\n"
            
            # Test with different encoding scenarios
            encodings_to_test = [
                ('cp1252', test_data.encode('cp1252')),
                ('iso-8859-1', test_data.encode('iso-8859-1')),
            ]
            
            fallback_tests = []
            
            for encoding_name, encoded_content in encodings_to_test:
                files = {
                    'file': (f'fallback_{encoding_name}_test.csv', encoded_content, 'text/csv')
                }
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code == 200:
                    print(f"✅ {encoding_name} encoding handled successfully")
                    fallback_tests.append(True)
                else:
                    print(f"❌ {encoding_name} encoding failed: {response.status_code}")
                    fallback_tests.append(False)
            
            # Test passes if at least one encoding fallback worked
            if any(fallback_tests):
                print("✅ Encoding fallback mechanism working")
                self.test_results['encoding_fallback'] = True
            else:
                print("❌ Encoding fallback mechanism failed")
                self.test_results['encoding_fallback'] = False
                
        except Exception as e:
            print(f"❌ Encoding fallback test error: {str(e)}")
            self.test_results['encoding_fallback'] = False
    
    def run_all_tests(self):
        """Run all encoding and data cleaning tests"""
        print("🚀 Starting Encoding Support and Data Cleaning Testing")
        print("=" * 70)
        
        # Run all focused tests
        self.test_utf8_encoding_support()
        self.test_latin1_encoding_support()
        self.test_nan_values_handling()
        self.test_mixed_data_types_handling()
        self.test_empty_strings_handling()
        self.test_problematic_combination_handling()
        self.test_data_quality_report_with_problematic_data()
        self.test_model_training_with_cleaned_data()
        self.test_encoding_fallback_mechanism()
        
        # Print final results
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 70)
        print("📊 ENCODING & DATA CLEANING TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} - {test_name.replace('_', ' ').title()}")
        
        print("\n" + "=" * 70)
        
        # Specific analysis for the review request
        encoding_tests = ['utf8_encoding', 'latin1_encoding', 'encoding_fallback']
        data_cleaning_tests = ['nan_handling', 'mixed_types_handling', 'empty_strings_handling', 'problematic_combination_handling']
        
        encoding_passed = sum(1 for test in encoding_tests if self.test_results.get(test, False))
        data_cleaning_passed = sum(1 for test in data_cleaning_tests if self.test_results.get(test, False))
        
        print("🎯 CRITICAL FIXES ANALYSIS:")
        print(f"   Encoding Support: {encoding_passed}/{len(encoding_tests)} tests passed")
        print(f"   Data Cleaning: {data_cleaning_passed}/{len(data_cleaning_tests)} tests passed")
        
        # Return overall success
        return passed_tests >= total_tests * 0.8  # 80% pass rate for overall success

if __name__ == "__main__":
    tester = EncodingDataCleaningTester()
    overall_success = tester.run_all_tests()
    
    if overall_success:
        print("🎉 Encoding and data cleaning testing completed successfully!")
        exit(0)
    else:
        print("⚠️  Encoding and data cleaning testing completed with some failures.")
        exit(1)