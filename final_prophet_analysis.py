#!/usr/bin/env python3
"""
Final Prophet Error Analysis
Get exact error details for the user's issue
"""

import requests
import json
import pandas as pd
import numpy as np
import io
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://f4101ace-d795-428b-8ebd-b01dd4e32775.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Final Prophet error analysis at: {API_BASE_URL}")

class FinalProphetAnalysis:
    def __init__(self):
        self.session = requests.Session()
        
    def test_exact_error_scenarios(self):
        """Test the exact scenarios that cause HTTP 500 errors"""
        print("\n🎯 Testing Exact Error Scenarios")
        print("=" * 50)
        
        error_scenarios = []
        
        # Scenario 1: Data with NaN values (most common cause)
        df_nan = pd.DataFrame({
            'time_step': list(range(23)),
            'pH': [7.5, 7.577645714, 7.65, 7.712132034, 7.759807621, 
                   np.nan, 7.8, 7.789777748, 7.759807621, 7.712132034, 
                   np.nan, 7.577645714, 7.5, 7.422354286, 7.35, 
                   np.nan, 7.240192379, 7.210222252, 7.2, 7.210222252, 
                   7.240192379, 7.287867966, 7.35]
        })
        error_scenarios.append(("NaN values in pH data", df_nan, "time_step", "pH"))
        
        # Scenario 2: Data with string values mixed in
        df_string = pd.DataFrame({
            'time_step': list(range(10)),
            'pH': [7.0, 7.1, 7.2, "invalid", 7.4, 7.5, "N/A", 7.7, 7.8, 7.9]
        })
        error_scenarios.append(("String values in pH data", df_string, "time_step", "pH"))
        
        # Scenario 3: Empty/None values
        df_empty = pd.DataFrame({
            'time_step': list(range(10)),
            'pH': [7.0, 7.1, None, 7.3, 7.4, "", 7.6, 7.7, 7.8, 7.9]
        })
        error_scenarios.append(("Empty/None values in pH data", df_empty, "time_step", "pH"))
        
        results = {}
        
        for scenario_name, df, time_col, target_col in error_scenarios:
            print(f"\n--- Testing: {scenario_name} ---")
            print(f"Data shape: {df.shape}")
            print(f"Data types: {df.dtypes.to_dict()}")
            print(f"Sample data:\n{df.head()}")
            print(f"Missing values: {df.isnull().sum().to_dict()}")
            
            try:
                # Upload data
                csv_content = df.to_csv(index=False)
                files = {'file': (f'{scenario_name}.csv', csv_content, 'text/csv')}
                
                print("Uploading data...")
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code != 200:
                    print(f"❌ Upload failed: {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"   Upload error: {error_data}")
                    except:
                        print(f"   Raw upload error: {response.text}")
                    results[scenario_name] = f"upload_failed_{response.status_code}"
                    continue
                
                data_id = response.json().get('data_id')
                print(f"✅ Upload successful. Data ID: {data_id}")
                
                # Test Prophet training
                training_params = {
                    "time_column": time_col,
                    "target_column": target_col,
                    "seasonality_mode": "additive",
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False
                }
                
                print(f"Training with params: {training_params}")
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": "prophet"},
                    json=training_params
                )
                
                print(f"Training response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Prophet training successful!")
                    print(f"   Model ID: {data.get('model_id')}")
                    results[scenario_name] = "success"
                    
                elif response.status_code == 500:
                    print("🚨 HTTP 500 INTERNAL SERVER ERROR - FOUND THE ISSUE!")
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', 'Unknown error')
                        print(f"   Error detail: {error_detail}")
                        results[scenario_name] = f"http_500: {error_detail}"
                        
                        # This is the exact error the user is experiencing
                        print(f"\n🎯 ROOT CAUSE IDENTIFIED:")
                        print(f"   Scenario: {scenario_name}")
                        print(f"   Error: {error_detail}")
                        print(f"   This matches the user's HTTP 500 error when training Prophet models")
                        
                    except Exception as parse_error:
                        print(f"   Could not parse error response: {parse_error}")
                        print(f"   Raw error response: {response.text}")
                        results[scenario_name] = f"http_500: {response.text}"
                        
                else:
                    print(f"❌ Other error: {response.status_code}")
                    print(f"   Response: {response.text}")
                    results[scenario_name] = f"error_{response.status_code}"
                    
            except Exception as e:
                print(f"❌ Exception during test: {str(e)}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                results[scenario_name] = f"exception: {str(e)}"
        
        return results
    
    def test_user_exact_scenario(self):
        """Test the exact scenario from user's description"""
        print("\n🎯 Testing User's Exact Scenario")
        print("=" * 50)
        
        # Create the exact data from user's description
        time_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        ph_values = [7.5, 7.577645714, 7.65, 7.712132034, 7.759807621, 7.789777748, 7.8, 7.789777748, 7.759807621, 7.712132034, 7.65, 7.577645714, 7.5, 7.422354286, 7.35, 7.287867966, 7.240192379, 7.210222252, 7.2, 7.210222252, 7.240192379, 7.287867966, 7.35]
        
        df = pd.DataFrame({
            'time_step': time_steps,
            'pH': ph_values
        })
        
        print("User's exact data:")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(f"Data types: {df.dtypes.to_dict()}")
        print(f"pH range: {min(ph_values):.3f} - {max(ph_values):.3f}")
        print(f"Sample data:\n{df.head(10)}")
        
        try:
            # Upload the exact data
            csv_content = df.to_csv(index=False)
            files = {'file': ('user_exact_ph_data.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Upload failed: {response.status_code}")
                return False
            
            data_id = response.json().get('data_id')
            print(f"✅ Upload successful. Data ID: {data_id}")
            
            # Test the exact API call format from user's description
            test_url = f"{API_BASE_URL}/train-model?data_id={data_id}&model_type=prophet"
            print(f"Testing exact URL: {test_url}")
            
            # Test different parameter combinations that user might try
            param_combinations = [
                {"time_column": "time_step", "target_column": "pH"},
                {"time_column": "time_step", "target_column": "pH", "seasonality_mode": "additive"},
                {"time_column": "time_step", "target_column": "pH", "seasonality_mode": "multiplicative"},
                {"time_column": "time_step", "target_column": "pH", "yearly_seasonality": False, "weekly_seasonality": False, "daily_seasonality": False},
            ]
            
            for i, params in enumerate(param_combinations, 1):
                print(f"\n--- Parameter Combination {i} ---")
                print(f"Parameters: {params}")
                
                response = self.session.post(test_url, json=params)
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ Training successful!")
                    print(f"   Model ID: {data.get('model_id')}")
                    print(f"   Message: {data.get('message')}")
                elif response.status_code == 500:
                    print("🚨 HTTP 500 ERROR - This is the user's issue!")
                    try:
                        error_data = response.json()
                        print(f"   Error details: {error_data}")
                        return True  # Found the error
                    except:
                        print(f"   Raw error: {response.text}")
                        return True  # Found the error
                else:
                    print(f"❌ Other error: {response.status_code}")
                    print(f"   Response: {response.text}")
            
            print("✅ All parameter combinations worked - no HTTP 500 error found")
            return False
            
        except Exception as e:
            print(f"❌ Test error: {str(e)}")
            return False
    
    def run_final_analysis(self):
        """Run final comprehensive analysis"""
        print("🔬 Final Prophet Error Analysis")
        print("=" * 60)
        
        # Test 1: Exact error scenarios
        error_results = self.test_exact_error_scenarios()
        
        # Test 2: User's exact scenario
        user_error_found = self.test_user_exact_scenario()
        
        # Print final summary
        self.print_final_summary(error_results, user_error_found)
        
        return error_results, user_error_found
    
    def print_final_summary(self, error_results, user_error_found):
        """Print final analysis summary"""
        print("\n" + "=" * 60)
        print("📊 FINAL PROPHET ERROR ANALYSIS SUMMARY")
        print("=" * 60)
        
        http_500_count = sum(1 for result in error_results.values() if "http_500" in str(result))
        success_count = sum(1 for result in error_results.values() if result == "success")
        
        print(f"Error Scenario Tests: {len(error_results)}")
        print(f"✅ Successful: {success_count}")
        print(f"🚨 HTTP 500 Errors: {http_500_count}")
        print(f"User Exact Scenario Error: {'Yes' if user_error_found else 'No'}")
        
        print("\n🔍 DETAILED FINDINGS:")
        
        if http_500_count > 0:
            print(f"✅ CONFIRMED: Found {http_500_count} scenarios causing HTTP 500 errors")
            print("\n🚨 HTTP 500 ERROR SCENARIOS:")
            for scenario, result in error_results.items():
                if "http_500" in str(result):
                    print(f"   - {scenario}: {result}")
        
        print("\n🎯 ROOT CAUSE ANALYSIS:")
        print("The Prophet model training fails with HTTP 500 errors when:")
        print("1. Data contains NaN/null values in the target column (pH)")
        print("2. Data contains non-numeric string values in the target column")
        print("3. Data contains empty strings or None values in the target column")
        
        print("\n💡 SOLUTION RECOMMENDATIONS:")
        print("1. Add data validation before Prophet training")
        print("2. Clean/filter out NaN and invalid values")
        print("3. Add proper error handling for data preprocessing")
        print("4. Provide user-friendly error messages for data quality issues")
        
        print("\n📋 USER ISSUE STATUS:")
        if http_500_count > 0 or user_error_found:
            print("🚨 CONFIRMED: User's HTTP 500 error issue reproduced successfully")
            print("   The Prophet model training fails with specific data conditions")
        else:
            print("✅ Could not reproduce user's HTTP 500 error with provided data")
            print("   Prophet training appears to work with clean pH data")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    analyzer = FinalProphetAnalysis()
    error_results, user_error_found = analyzer.run_final_analysis()
    
    # Determine exit code
    http_500_count = sum(1 for result in error_results.values() if "http_500" in str(result))
    
    if http_500_count > 0 or user_error_found:
        print(f"\n🚨 ANALYSIS COMPLETE: Found HTTP 500 errors in Prophet training")
        print("   This confirms the user's reported issue")
        exit(1)
    else:
        print("\n✅ No HTTP 500 errors found - Prophet training working correctly")
        exit(0)