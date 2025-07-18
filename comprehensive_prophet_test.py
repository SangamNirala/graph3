#!/usr/bin/env python3
"""
Comprehensive Prophet Model Testing
Tests various scenarios to reproduce the HTTP 500 error
"""

import requests
import json
import pandas as pd
import io
import time
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://99b9a80c-dec7-4b9b-8839-ca8a46e41fb1.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Prophet model scenarios at: {API_BASE_URL}")

class ComprehensiveProphetTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_ph_dataset_exact(self):
        """Create the exact pH dataset from user's description"""
        time_steps = list(range(23))  # 0 to 22
        ph_values = [
            7.5, 7.577645714, 7.65, 7.712132034, 7.759807621, 7.789777748, 
            7.8, 7.789777748, 7.759807621, 7.712132034, 7.65, 7.577645714, 
            7.5, 7.422354286, 7.35, 7.287867966, 7.240192379, 7.210222252, 
            7.2, 7.210222252, 7.240192379, 7.287867966, 7.35
        ]
        
        df = pd.DataFrame({
            'time_step': time_steps,
            'pH': ph_values
        })
        
        return df
    
    def create_ph_dataset_with_timestamps(self):
        """Create pH dataset with proper timestamps"""
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
        
        return df
    
    def test_scenario(self, scenario_name, df, time_column, target_column, extra_params=None):
        """Test a specific Prophet training scenario"""
        print(f"\n=== Testing Scenario: {scenario_name} ===")
        
        try:
            # Upload data
            csv_content = df.to_csv(index=False)
            files = {'file': (f'{scenario_name}.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Data upload failed: {response.status_code}")
                return False
            
            data_id = response.json().get('data_id')
            print(f"✅ Data uploaded successfully. ID: {data_id}")
            
            # Prepare training parameters
            training_params = {
                "time_column": time_column,
                "target_column": target_column,
                "seasonality_mode": "additive",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False
            }
            
            if extra_params:
                training_params.update(extra_params)
            
            print(f"Training parameters: {training_params}")
            
            # Train Prophet model
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "prophet"},
                json=training_params
            )
            
            print(f"Training response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Prophet training successful")
                print(f"   Model ID: {data.get('model_id')}")
                print(f"   Message: {data.get('message')}")
                return True
            elif response.status_code == 500:
                print(f"❌ Prophet training failed with HTTP 500 - INTERNAL SERVER ERROR")
                try:
                    error_data = response.json()
                    print(f"   Error details: {error_data}")
                except:
                    print(f"   Raw error response: {response.text}")
                return False
            else:
                print(f"❌ Prophet training failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Scenario test error: {str(e)}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def test_all_scenarios(self):
        """Test all possible scenarios that might cause HTTP 500"""
        print("🔍 Starting Comprehensive Prophet Model Testing")
        print("=" * 70)
        
        # Scenario 1: Exact user data with time_step column
        df1 = self.create_ph_dataset_exact()
        self.test_results['exact_user_data_time_step'] = self.test_scenario(
            "Exact User Data (time_step)", df1, "time_step", "pH"
        )
        
        # Scenario 2: Exact user data with timestamp column
        df2 = self.create_ph_dataset_with_timestamps()
        self.test_results['exact_user_data_timestamp'] = self.test_scenario(
            "Exact User Data (timestamp)", df2, "timestamp", "pH"
        )
        
        # Scenario 3: Test with different seasonality modes
        self.test_results['multiplicative_seasonality'] = self.test_scenario(
            "Multiplicative Seasonality", df2, "timestamp", "pH", 
            {"seasonality_mode": "multiplicative"}
        )
        
        # Scenario 4: Test with no seasonality
        self.test_results['no_seasonality'] = self.test_scenario(
            "No Seasonality", df2, "timestamp", "pH", 
            {
                "yearly_seasonality": False,
                "weekly_seasonality": False,
                "daily_seasonality": False
            }
        )
        
        # Scenario 5: Test with all seasonality enabled
        self.test_results['all_seasonality'] = self.test_scenario(
            "All Seasonality", df2, "timestamp", "pH", 
            {
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": True
            }
        )
        
        # Scenario 6: Test with minimal data (edge case)
        df_minimal = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='D'),
            'pH': [7.0, 7.1, 7.2, 7.1, 7.0]
        })
        self.test_results['minimal_data'] = self.test_scenario(
            "Minimal Data (5 points)", df_minimal, "timestamp", "pH"
        )
        
        # Scenario 7: Test with missing values
        df_missing = df2.copy()
        df_missing.loc[5, 'pH'] = None
        df_missing.loc[10, 'pH'] = None
        self.test_results['missing_values'] = self.test_scenario(
            "Data with Missing Values", df_missing, "timestamp", "pH"
        )
        
        # Scenario 8: Test with duplicate timestamps
        df_duplicates = df2.copy()
        df_duplicates.loc[5, 'timestamp'] = df_duplicates.loc[4, 'timestamp']
        self.test_results['duplicate_timestamps'] = self.test_scenario(
            "Duplicate Timestamps", df_duplicates, "timestamp", "pH"
        )
        
        # Scenario 9: Test with irregular time intervals
        df_irregular = pd.DataFrame({
            'timestamp': [
                '2023-01-01', '2023-01-03', '2023-01-06', '2023-01-10', 
                '2023-01-15', '2023-01-21', '2023-01-28', '2023-02-05'
            ],
            'pH': [7.0, 7.1, 7.2, 7.3, 7.2, 7.1, 7.0, 6.9]
        })
        self.test_results['irregular_intervals'] = self.test_scenario(
            "Irregular Time Intervals", df_irregular, "timestamp", "pH"
        )
        
        # Scenario 10: Test with very small pH variations
        df_small_var = df2.copy()
        df_small_var['pH'] = 7.0 + (df_small_var['pH'] - 7.0) * 0.01  # Scale down variations
        self.test_results['small_variations'] = self.test_scenario(
            "Small pH Variations", df_small_var, "timestamp", "pH"
        )
        
        # Print comprehensive results
        self.print_comprehensive_results()
        
        return self.test_results
    
    def test_specific_user_request_format(self):
        """Test the exact format mentioned in the user's request"""
        print("\n🎯 Testing Exact User Request Format")
        print("=" * 50)
        
        # Create data exactly as user described
        df = self.create_ph_dataset_exact()
        
        print("User's data format:")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(f"Sample data:\n{df.head()}")
        
        # Test the exact API call format from user's description
        try:
            # Upload data
            csv_content = df.to_csv(index=False)
            files = {'file': ('ph_data.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Upload failed: {response.status_code}")
                return False
            
            data_id = response.json().get('data_id')
            print(f"✅ Data uploaded. ID: {data_id}")
            
            # Test the exact URL format from user's description
            test_url = f"{API_BASE_URL}/train-model?data_id={data_id}&model_type=prophet"
            print(f"Testing URL: {test_url}")
            
            # Test with different parameter combinations
            param_combinations = [
                {"time_column": "time_step", "target_column": "pH"},
                {"time_column": "time_step", "target_column": "pH", "seasonality_mode": "additive"},
                {"time_column": "time_step", "target_column": "pH", "seasonality_mode": "multiplicative"},
            ]
            
            for i, params in enumerate(param_combinations, 1):
                print(f"\n--- Parameter Combination {i} ---")
                print(f"Parameters: {params}")
                
                response = self.session.post(test_url, json=params)
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 500:
                    print("🚨 FOUND HTTP 500 ERROR!")
                    try:
                        error_data = response.json()
                        print(f"Error details: {error_data}")
                    except:
                        print(f"Raw error: {response.text}")
                    return True  # Found the error!
                elif response.status_code == 200:
                    print("✅ Training successful")
                    data = response.json()
                    print(f"Model ID: {data.get('model_id')}")
                else:
                    print(f"❌ Other error: {response.status_code}")
                    print(f"Response: {response.text}")
            
            return False  # No 500 error found
            
        except Exception as e:
            print(f"❌ Test error: {str(e)}")
            return False
    
    def print_comprehensive_results(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE PROPHET TEST RESULTS")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Scenarios Tested: {total_tests}")
        print(f"Successful: {passed_tests}")
        print(f"Failed (HTTP 500): {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for scenario, result in self.test_results.items():
            status = "✅ SUCCESS" if result else "❌ HTTP 500 ERROR"
            print(f"  {status} - {scenario.replace('_', ' ').title()}")
        
        # Analysis
        print("\n🔍 ANALYSIS:")
        if failed_tests == 0:
            print("✅ No HTTP 500 errors found in any scenario")
            print("   Prophet model training appears to be working correctly")
        else:
            print(f"❌ Found {failed_tests} scenarios that cause HTTP 500 errors")
            print("   These scenarios reproduce the user's reported issue:")
            for scenario, result in self.test_results.items():
                if not result:
                    print(f"   - {scenario.replace('_', ' ').title()}")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    tester = ComprehensiveProphetTester()
    
    # Test all scenarios
    results = tester.test_all_scenarios()
    
    # Test specific user request format
    found_500_error = tester.test_specific_user_request_format()
    
    # Final assessment
    failed_scenarios = sum(1 for result in results.values() if not result)
    
    if failed_scenarios > 0 or found_500_error:
        print(f"\n🚨 CONFIRMED: Found {failed_scenarios} scenarios causing HTTP 500 errors")
        print("   This reproduces the user's reported Prophet training issue")
        exit(1)
    else:
        print("\n✅ No HTTP 500 errors found - Prophet training working correctly")
        exit(0)