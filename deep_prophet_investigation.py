#!/usr/bin/env python3
"""
Deep Investigation of Prophet HTTP 500 Errors
Focus on missing values and other edge cases
"""

import requests
import json
import pandas as pd
import numpy as np
import io
import time
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://5a3adf14-acc4-45e4-8b35-2ee37c5def6f.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Deep investigation of Prophet errors at: {API_BASE_URL}")

class ProphetErrorInvestigator:
    def __init__(self):
        self.session = requests.Session()
        
    def create_ph_dataset_with_missing_values(self):
        """Create pH dataset with missing values that causes HTTP 500"""
        time_steps = list(range(23))
        ph_values = [
            7.5, 7.577645714, 7.65, 7.712132034, 7.759807621, 7.789777748, 
            7.8, 7.789777748, 7.759807621, 7.712132034, 7.65, 7.577645714, 
            7.5, 7.422354286, 7.35, 7.287867966, 7.240192379, 7.210222252, 
            7.2, 7.210222252, 7.240192379, 7.287867966, 7.35
        ]
        
        from datetime import datetime, timedelta
        start_date = datetime(2023, 1, 1)
        timestamps = [start_date + timedelta(days=i) for i in time_steps]
        
        df = pd.DataFrame({
            'time_step': time_steps,
            'timestamp': timestamps,
            'pH': ph_values
        })
        
        # Introduce missing values
        df.loc[5, 'pH'] = None
        df.loc[10, 'pH'] = None
        df.loc[15, 'pH'] = np.nan
        
        return df
    
    def test_missing_values_scenarios(self):
        """Test different missing value scenarios"""
        print("\n🔍 Testing Missing Values Scenarios")
        print("=" * 50)
        
        scenarios = []
        
        # Scenario 1: NaN values
        df1 = self.create_ph_dataset_with_missing_values()
        scenarios.append(("NaN values", df1))
        
        # Scenario 2: Empty string values
        df2 = self.create_ph_dataset_with_missing_values()
        df2.loc[5, 'pH'] = ""
        df2.loc[10, 'pH'] = ""
        scenarios.append(("Empty string values", df2))
        
        # Scenario 3: Zero values (edge case)
        df3 = self.create_ph_dataset_with_missing_values()
        df3.loc[5, 'pH'] = 0
        df3.loc[10, 'pH'] = 0
        scenarios.append(("Zero values", df3))
        
        # Scenario 4: Extreme pH values
        df4 = self.create_ph_dataset_with_missing_values()
        df4.loc[5, 'pH'] = 15.0  # Invalid pH
        df4.loc[10, 'pH'] = -2.0  # Invalid pH
        scenarios.append(("Extreme pH values", df4))
        
        # Scenario 5: Mixed data types
        df5 = self.create_ph_dataset_with_missing_values()
        df5.loc[5, 'pH'] = "invalid"
        df5.loc[10, 'pH'] = "N/A"
        scenarios.append(("Mixed data types", df5))
        
        results = {}
        
        for scenario_name, df in scenarios:
            print(f"\n--- Testing: {scenario_name} ---")
            
            try:
                # Upload data
                csv_content = df.to_csv(index=False)
                files = {'file': (f'{scenario_name}.csv', csv_content, 'text/csv')}
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code != 200:
                    print(f"❌ Data upload failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    results[scenario_name] = "upload_failed"
                    continue
                
                data_id = response.json().get('data_id')
                print(f"✅ Data uploaded successfully. ID: {data_id}")
                
                # Try Prophet training
                training_params = {
                    "time_column": "timestamp",
                    "target_column": "pH",
                    "seasonality_mode": "additive",
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False
                }
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": "prophet"},
                    json=training_params
                )
                
                print(f"Training response status: {response.status_code}")
                
                if response.status_code == 200:
                    print("✅ Prophet training successful")
                    results[scenario_name] = "success"
                elif response.status_code == 500:
                    print("🚨 HTTP 500 ERROR FOUND!")
                    try:
                        error_data = response.json()
                        print(f"   Error details: {error_data}")
                        error_detail = error_data.get('detail', '')
                        results[scenario_name] = f"http_500: {error_detail}"
                    except:
                        print(f"   Raw error: {response.text}")
                        results[scenario_name] = f"http_500: {response.text}"
                else:
                    print(f"❌ Other error: {response.status_code}")
                    results[scenario_name] = f"error_{response.status_code}"
                    
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                results[scenario_name] = f"exception: {str(e)}"
        
        return results
    
    def test_data_preprocessing_edge_cases(self):
        """Test edge cases in data preprocessing"""
        print("\n🔍 Testing Data Preprocessing Edge Cases")
        print("=" * 50)
        
        edge_cases = []
        
        # Edge case 1: Single data point
        df1 = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01')],
            'pH': [7.0]
        })
        edge_cases.append(("Single data point", df1))
        
        # Edge case 2: Two data points
        df2 = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=2, freq='D'),
            'pH': [7.0, 7.1]
        })
        edge_cases.append(("Two data points", df2))
        
        # Edge case 3: All same values
        df3 = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
            'pH': [7.0] * 10
        })
        edge_cases.append(("All same values", df3))
        
        # Edge case 4: Unsorted timestamps
        df4 = pd.DataFrame({
            'timestamp': ['2023-01-05', '2023-01-01', '2023-01-03', '2023-01-02', '2023-01-04'],
            'pH': [7.0, 7.1, 7.2, 7.3, 7.4]
        })
        edge_cases.append(("Unsorted timestamps", df4))
        
        # Edge case 5: Future timestamps
        df5 = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=5, freq='D'),
            'pH': [7.0, 7.1, 7.2, 7.1, 7.0]
        })
        edge_cases.append(("Future timestamps", df5))
        
        results = {}
        
        for case_name, df in edge_cases:
            print(f"\n--- Testing: {case_name} ---")
            print(f"Data shape: {df.shape}")
            print(f"Data preview:\n{df}")
            
            try:
                # Upload and test
                csv_content = df.to_csv(index=False)
                files = {'file': (f'{case_name}.csv', csv_content, 'text/csv')}
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code != 200:
                    print(f"❌ Upload failed: {response.status_code}")
                    results[case_name] = "upload_failed"
                    continue
                
                data_id = response.json().get('data_id')
                print(f"✅ Upload successful. ID: {data_id}")
                
                # Test Prophet training
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": "prophet"},
                    json={"time_column": "timestamp", "target_column": "pH"}
                )
                
                if response.status_code == 200:
                    print("✅ Prophet training successful")
                    results[case_name] = "success"
                elif response.status_code == 500:
                    print("🚨 HTTP 500 ERROR!")
                    try:
                        error_data = response.json()
                        print(f"   Error: {error_data}")
                        results[case_name] = f"http_500: {error_data.get('detail', '')}"
                    except:
                        results[case_name] = f"http_500: {response.text}"
                else:
                    print(f"❌ Error {response.status_code}")
                    results[case_name] = f"error_{response.status_code}"
                    
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                results[case_name] = f"exception: {str(e)}"
        
        return results
    
    def test_column_name_edge_cases(self):
        """Test edge cases with column names"""
        print("\n🔍 Testing Column Name Edge Cases")
        print("=" * 50)
        
        # Create base data
        base_data = {
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
            'pH': [7.0 + 0.1*i for i in range(10)]
        }
        
        column_cases = []
        
        # Case 1: Column names with spaces
        df1 = pd.DataFrame(base_data)
        df1.columns = ['time stamp', 'p H']
        column_cases.append(("Spaces in column names", df1, 'time stamp', 'p H'))
        
        # Case 2: Column names with special characters
        df2 = pd.DataFrame(base_data)
        df2.columns = ['time-stamp', 'pH_value']
        column_cases.append(("Special chars in column names", df2, 'time-stamp', 'pH_value'))
        
        # Case 3: Non-existent column names
        df3 = pd.DataFrame(base_data)
        column_cases.append(("Non-existent columns", df3, 'nonexistent_time', 'nonexistent_ph'))
        
        # Case 4: Swapped column types
        df4 = pd.DataFrame({
            'pH': pd.date_range('2023-01-01', periods=10, freq='D'),
            'timestamp': [7.0 + 0.1*i for i in range(10)]
        })
        column_cases.append(("Swapped column types", df4, 'pH', 'timestamp'))
        
        results = {}
        
        for case_name, df, time_col, target_col in column_cases:
            print(f"\n--- Testing: {case_name} ---")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Time column: '{time_col}', Target column: '{target_col}'")
            
            try:
                # Upload data
                csv_content = df.to_csv(index=False)
                files = {'file': (f'{case_name}.csv', csv_content, 'text/csv')}
                
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code != 200:
                    print(f"❌ Upload failed: {response.status_code}")
                    results[case_name] = "upload_failed"
                    continue
                
                data_id = response.json().get('data_id')
                print(f"✅ Upload successful. ID: {data_id}")
                
                # Test Prophet training
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": "prophet"},
                    json={"time_column": time_col, "target_column": target_col}
                )
                
                if response.status_code == 200:
                    print("✅ Prophet training successful")
                    results[case_name] = "success"
                elif response.status_code == 500:
                    print("🚨 HTTP 500 ERROR!")
                    try:
                        error_data = response.json()
                        print(f"   Error: {error_data}")
                        results[case_name] = f"http_500: {error_data.get('detail', '')}"
                    except:
                        results[case_name] = f"http_500: {response.text}"
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")
                    results[case_name] = f"error_{response.status_code}"
                    
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                results[case_name] = f"exception: {str(e)}"
        
        return results
    
    def run_deep_investigation(self):
        """Run comprehensive investigation"""
        print("🕵️ Starting Deep Prophet Error Investigation")
        print("=" * 70)
        
        all_results = {}
        
        # Test 1: Missing values scenarios
        print("\n1️⃣ MISSING VALUES INVESTIGATION")
        missing_results = self.test_missing_values_scenarios()
        all_results.update(missing_results)
        
        # Test 2: Data preprocessing edge cases
        print("\n2️⃣ DATA PREPROCESSING EDGE CASES")
        preprocessing_results = self.test_data_preprocessing_edge_cases()
        all_results.update(preprocessing_results)
        
        # Test 3: Column name edge cases
        print("\n3️⃣ COLUMN NAME EDGE CASES")
        column_results = self.test_column_name_edge_cases()
        all_results.update(column_results)
        
        # Print comprehensive summary
        self.print_investigation_summary(all_results)
        
        return all_results
    
    def print_investigation_summary(self, results):
        """Print investigation summary"""
        print("\n" + "=" * 70)
        print("🔍 DEEP INVESTIGATION SUMMARY")
        print("=" * 70)
        
        total_tests = len(results)
        http_500_errors = sum(1 for result in results.values() if "http_500" in str(result))
        upload_failures = sum(1 for result in results.values() if "upload_failed" in str(result))
        successes = sum(1 for result in results.values() if result == "success")
        other_errors = total_tests - http_500_errors - upload_failures - successes
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Successful: {successes}")
        print(f"🚨 HTTP 500 Errors: {http_500_errors}")
        print(f"📤 Upload Failures: {upload_failures}")
        print(f"❌ Other Errors: {other_errors}")
        
        print(f"\nHTTP 500 Error Rate: {(http_500_errors/total_tests)*100:.1f}%")
        
        print("\n📋 DETAILED RESULTS:")
        for test_name, result in results.items():
            if "http_500" in str(result):
                print(f"🚨 HTTP 500 - {test_name}: {result}")
            elif result == "success":
                print(f"✅ SUCCESS - {test_name}")
            elif "upload_failed" in str(result):
                print(f"📤 UPLOAD FAIL - {test_name}")
            else:
                print(f"❌ OTHER - {test_name}: {result}")
        
        print("\n🎯 KEY FINDINGS:")
        if http_500_errors > 0:
            print(f"✅ CONFIRMED: Found {http_500_errors} scenarios causing HTTP 500 errors")
            print("   These reproduce the user's reported Prophet training issue")
            
            # Identify patterns
            error_patterns = {}
            for test_name, result in results.items():
                if "http_500" in str(result):
                    if "missing" in test_name.lower() or "nan" in test_name.lower():
                        error_patterns["Missing/NaN values"] = error_patterns.get("Missing/NaN values", 0) + 1
                    elif "column" in test_name.lower():
                        error_patterns["Column issues"] = error_patterns.get("Column issues", 0) + 1
                    elif "data" in test_name.lower():
                        error_patterns["Data issues"] = error_patterns.get("Data issues", 0) + 1
            
            print("\n📊 ERROR PATTERNS:")
            for pattern, count in error_patterns.items():
                print(f"   - {pattern}: {count} cases")
        else:
            print("❌ No HTTP 500 errors found in this investigation")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    investigator = ProphetErrorInvestigator()
    results = investigator.run_deep_investigation()
    
    # Count HTTP 500 errors
    http_500_count = sum(1 for result in results.values() if "http_500" in str(result))
    
    if http_500_count > 0:
        print(f"\n🚨 INVESTIGATION COMPLETE: Found {http_500_count} HTTP 500 error scenarios")
        print("   This confirms and reproduces the user's reported Prophet training issue")
        exit(1)
    else:
        print("\n✅ No HTTP 500 errors found in deep investigation")
        exit(0)