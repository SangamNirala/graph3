#!/usr/bin/env python3
"""
Additional Edge Case Testing for File Upload Issues
Focus on specific scenarios that might cause user upload problems
"""

import requests
import json
import pandas as pd
import io
import numpy as np
import os
from pathlib import Path
import tempfile

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://6292a650-5f0f-439b-bded-80d6a5caef50.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing edge cases at: {API_BASE_URL}")

class EdgeCaseTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
    
    def test_very_small_datasets(self):
        """Test with datasets smaller than minimum requirements"""
        print("\n=== Testing Very Small Datasets ===")
        
        try:
            # Test with 5 rows (below minimum)
            small_df = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
                'value': [1, 2, 3, 4, 5]
            })
            
            csv_content = small_df.to_csv(index=False)
            files = {'file': ('small_data.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 400:
                print("✅ Small dataset correctly rejected with proper error message")
                print(f"   Error: {response.json().get('detail', 'No detail')}")
                self.test_results['small_dataset_handling'] = True
            elif response.status_code == 200:
                print("⚠️  Small dataset accepted (may cause issues later)")
                self.test_results['small_dataset_handling'] = True
            else:
                print(f"❌ Unexpected response for small dataset: {response.status_code}")
                self.test_results['small_dataset_handling'] = False
                
        except Exception as e:
            print(f"❌ Small dataset test error: {str(e)}")
            self.test_results['small_dataset_handling'] = False
    
    def test_duplicate_column_names(self):
        """Test with duplicate column names"""
        print("\n=== Testing Duplicate Column Names ===")
        
        try:
            # Create CSV with duplicate column names
            csv_content = "date,value,value,date\n2023-01-01,100,200,2023-01-01\n2023-01-02,150,250,2023-01-02\n"
            files = {'file': ('duplicate_cols.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                columns = data['analysis']['columns']
                print(f"✅ Duplicate columns handled: {columns}")
                self.test_results['duplicate_columns'] = True
            else:
                print(f"❌ Duplicate columns failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['duplicate_columns'] = False
                
        except Exception as e:
            print(f"❌ Duplicate columns test error: {str(e)}")
            self.test_results['duplicate_columns'] = False
    
    def test_special_characters_in_filename(self):
        """Test with special characters in filename"""
        print("\n=== Testing Special Characters in Filename ===")
        
        try:
            df = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=15, freq='D'),
                'value': np.random.normal(100, 10, 15)
            })
            
            csv_content = df.to_csv(index=False)
            
            # Test various special characters in filename
            special_filenames = [
                'test file with spaces.csv',
                'test-file-with-dashes.csv',
                'test_file_with_underscores.csv',
                'test.file.with.dots.csv',
                'test(file)with(parentheses).csv',
                'test[file]with[brackets].csv'
            ]
            
            special_filename_tests = []
            
            for filename in special_filenames:
                files = {'file': (filename, csv_content, 'text/csv')}
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code == 200:
                    print(f"✅ {filename} uploaded successfully")
                    special_filename_tests.append(True)
                else:
                    print(f"❌ {filename} failed: {response.status_code}")
                    special_filename_tests.append(False)
            
            self.test_results['special_filenames'] = all(special_filename_tests)
            
        except Exception as e:
            print(f"❌ Special filenames test error: {str(e)}")
            self.test_results['special_filenames'] = False
    
    def test_extremely_long_column_names(self):
        """Test with extremely long column names"""
        print("\n=== Testing Extremely Long Column Names ===")
        
        try:
            # Create data with very long column names
            long_col_name = "this_is_an_extremely_long_column_name_that_might_cause_issues_in_some_systems_" * 3
            
            df = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=15, freq='D'),
                long_col_name: np.random.normal(100, 10, 15),
                'normal_column': np.random.normal(50, 5, 15)
            })
            
            csv_content = df.to_csv(index=False)
            files = {'file': ('long_columns.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                columns = data['analysis']['columns']
                print(f"✅ Long column names handled successfully")
                print(f"   Column count: {len(columns)}")
                self.test_results['long_column_names'] = True
            else:
                print(f"❌ Long column names failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['long_column_names'] = False
                
        except Exception as e:
            print(f"❌ Long column names test error: {str(e)}")
            self.test_results['long_column_names'] = False
    
    def test_numeric_column_names(self):
        """Test with numeric column names"""
        print("\n=== Testing Numeric Column Names ===")
        
        try:
            # Create CSV with numeric column names
            csv_content = "1,2,3,4\n2023-01-01,100,200,300\n2023-01-02,150,250,350\n"
            files = {'file': ('numeric_cols.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                columns = data['analysis']['columns']
                print(f"✅ Numeric column names handled: {columns}")
                self.test_results['numeric_column_names'] = True
            else:
                print(f"❌ Numeric column names failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['numeric_column_names'] = False
                
        except Exception as e:
            print(f"❌ Numeric column names test error: {str(e)}")
            self.test_results['numeric_column_names'] = False
    
    def test_different_date_formats(self):
        """Test with different date formats"""
        print("\n=== Testing Different Date Formats ===")
        
        try:
            date_format_tests = []
            
            # Test various date formats
            date_formats = [
                ('ISO format', ['2023-01-01', '2023-01-02', '2023-01-03']),
                ('US format', ['01/01/2023', '01/02/2023', '01/03/2023']),
                ('EU format', ['01.01.2023', '02.01.2023', '03.01.2023']),
                ('Timestamp', ['2023-01-01 12:00:00', '2023-01-02 12:00:00', '2023-01-03 12:00:00']),
                ('Mixed format', ['2023-01-01', '01/02/2023', '03.01.2023'])
            ]
            
            for format_name, dates in date_formats:
                df = pd.DataFrame({
                    'date': dates + ['2023-01-04'] * (15 - len(dates)),  # Pad to 15 rows
                    'value': np.random.normal(100, 10, 15)
                })
                
                csv_content = df.to_csv(index=False)
                files = {'file': (f'{format_name.lower().replace(" ", "_")}.csv', csv_content, 'text/csv')}
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    time_cols = data['analysis']['time_columns']
                    print(f"✅ {format_name} handled successfully (time cols: {time_cols})")
                    date_format_tests.append(True)
                else:
                    print(f"❌ {format_name} failed: {response.status_code}")
                    date_format_tests.append(False)
            
            self.test_results['different_date_formats'] = sum(date_format_tests) >= len(date_format_tests) * 0.8
            
        except Exception as e:
            print(f"❌ Date formats test error: {str(e)}")
            self.test_results['different_date_formats'] = False
    
    def test_extreme_values(self):
        """Test with extreme numeric values"""
        print("\n=== Testing Extreme Numeric Values ===")
        
        try:
            # Create data with extreme values
            df = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=15, freq='D'),
                'very_large': [1e10, 1e15, 1e20] + [100] * 12,
                'very_small': [1e-10, 1e-15, 1e-20] + [1] * 12,
                'negative': [-1e10, -1e6, -1000] + [-10] * 12,
                'normal': np.random.normal(100, 10, 15)
            })
            
            csv_content = df.to_csv(index=False)
            files = {'file': ('extreme_values.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                numeric_cols = data['analysis']['numeric_columns']
                print(f"✅ Extreme values handled successfully")
                print(f"   Numeric columns detected: {len(numeric_cols)}")
                self.test_results['extreme_values'] = True
            else:
                print(f"❌ Extreme values failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['extreme_values'] = False
                
        except Exception as e:
            print(f"❌ Extreme values test error: {str(e)}")
            self.test_results['extreme_values'] = False
    
    def test_unicode_content(self):
        """Test with various Unicode characters"""
        print("\n=== Testing Unicode Content ===")
        
        try:
            # Create data with various Unicode characters
            df = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=15, freq='D'),
                'emoji_data': ['😀', '🚀', '💡', '🔥', '⭐'] + ['normal'] * 10,
                'chinese': ['北京', '上海', '广州', '深圳', '杭州'] + ['test'] * 10,
                'arabic': ['مرحبا', 'العالم', 'اختبار', 'بيانات', 'تحليل'] + ['test'] * 10,
                'value': np.random.normal(100, 10, 15)
            })
            
            csv_content = df.to_csv(index=False, encoding='utf-8')
            files = {'file': ('unicode_content.csv', csv_content.encode('utf-8'), 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Unicode content handled successfully")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                self.test_results['unicode_content'] = True
            else:
                print(f"❌ Unicode content failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results['unicode_content'] = False
                
        except Exception as e:
            print(f"❌ Unicode content test error: {str(e)}")
            self.test_results['unicode_content'] = False
    
    def test_concurrent_uploads(self):
        """Test concurrent file uploads"""
        print("\n=== Testing Concurrent Uploads ===")
        
        try:
            import threading
            import time
            
            # Create test data
            df = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=20, freq='D'),
                'value': np.random.normal(100, 10, 20)
            })
            csv_content = df.to_csv(index=False)
            
            results = []
            
            def upload_file(file_id):
                try:
                    files = {'file': (f'concurrent_{file_id}.csv', csv_content, 'text/csv')}
                    response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                    results.append(response.status_code == 200)
                except Exception as e:
                    print(f"   Upload {file_id} error: {e}")
                    results.append(False)
            
            # Start 3 concurrent uploads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=upload_file, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            successful_uploads = sum(results)
            print(f"✅ Concurrent uploads: {successful_uploads}/3 successful")
            
            self.test_results['concurrent_uploads'] = successful_uploads >= 2  # At least 2/3 should succeed
            
        except Exception as e:
            print(f"❌ Concurrent uploads test error: {str(e)}")
            self.test_results['concurrent_uploads'] = False
    
    def test_file_size_limits(self):
        """Test file size limits"""
        print("\n=== Testing File Size Limits ===")
        
        try:
            # Test with progressively larger files
            sizes = [1000, 5000, 10000, 50000]  # Number of rows
            size_tests = []
            
            for size in sizes:
                print(f"   Testing with {size} rows...")
                
                # Create large dataset
                df = pd.DataFrame({
                    'date': pd.date_range(start='2023-01-01', periods=size, freq='H'),
                    'value1': np.random.normal(100, 10, size),
                    'value2': np.random.normal(200, 20, size),
                    'value3': np.random.normal(300, 30, size),
                    'category': ['A', 'B', 'C'] * (size // 3 + 1)
                })[:size]  # Ensure exact size
                
                csv_content = df.to_csv(index=False)
                file_size_mb = len(csv_content) / (1024 * 1024)
                
                files = {'file': (f'size_test_{size}.csv', csv_content, 'text/csv')}
                
                try:
                    response = self.session.post(f"{API_BASE_URL}/upload-data", files=files, timeout=120)
                    
                    if response.status_code == 200:
                        print(f"   ✅ {size} rows ({file_size_mb:.1f} MB) uploaded successfully")
                        size_tests.append(True)
                    else:
                        print(f"   ❌ {size} rows failed: {response.status_code}")
                        size_tests.append(False)
                        
                except requests.exceptions.Timeout:
                    print(f"   ⏰ {size} rows timed out (may be too large)")
                    size_tests.append(False)
                except Exception as e:
                    print(f"   ❌ {size} rows error: {str(e)}")
                    size_tests.append(False)
            
            successful_sizes = sum(size_tests)
            print(f"✅ File size tests: {successful_sizes}/{len(sizes)} successful")
            
            self.test_results['file_size_limits'] = successful_sizes >= len(sizes) * 0.75
            
        except Exception as e:
            print(f"❌ File size limits test error: {str(e)}")
            self.test_results['file_size_limits'] = False
    
    def run_all_edge_case_tests(self):
        """Run all edge case tests"""
        print("🔍 Starting Edge Case Testing for File Upload")
        print("=" * 60)
        
        # Run all edge case tests
        self.test_very_small_datasets()
        self.test_duplicate_column_names()
        self.test_special_characters_in_filename()
        self.test_extremely_long_column_names()
        self.test_numeric_column_names()
        self.test_different_date_formats()
        self.test_extreme_values()
        self.test_unicode_content()
        self.test_concurrent_uploads()
        self.test_file_size_limits()
        
        # Print summary
        self.print_edge_case_summary()
    
    def print_edge_case_summary(self):
        """Print edge case test summary"""
        print("\n" + "=" * 60)
        print("🔍 EDGE CASE TESTING SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Edge Case Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print()
        
        # Print individual results
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print()
        
        # Identify critical edge case failures
        critical_edge_cases = [
            'small_dataset_handling',
            'different_date_formats',
            'unicode_content'
        ]
        
        failed_critical = [test for test in critical_edge_cases if not self.test_results.get(test, True)]
        
        if failed_critical:
            print("🚨 CRITICAL EDGE CASE FAILURES:")
            for test in failed_critical:
                print(f"  ❌ {test.replace('_', ' ').title()}")
            print()
        
        # Overall assessment
        if success_rate >= 90:
            print("🎉 EXCELLENT: Edge case handling is very robust!")
        elif success_rate >= 75:
            print("✅ GOOD: Most edge cases are handled well.")
        elif success_rate >= 50:
            print("⚠️  MODERATE: Some edge cases need attention.")
        else:
            print("🚨 CRITICAL: Edge case handling needs significant improvement!")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    tester = EdgeCaseTester()
    tester.run_all_edge_case_tests()