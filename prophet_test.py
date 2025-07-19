#!/usr/bin/env python3
"""
Focused Prophet Model Training Test
Tests the specific Prophet model training issue with pH time series data
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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://f54db828-52d2-4e14-b664-3ae23427df52.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing Prophet model at: {API_BASE_URL}")

class ProphetTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        
    def create_ph_dataset(self):
        """Create pH dataset similar to user's data"""
        # Create the exact pH dataset from the review request
        time_steps = list(range(23))  # 0 to 22
        ph_values = [
            7.5, 7.577645714, 7.65, 7.712132034, 7.759807621, 7.789777748, 
            7.8, 7.789777748, 7.759807621, 7.712132034, 7.65, 7.577645714, 
            7.5, 7.422354286, 7.35, 7.287867966, 7.240192379, 7.210222252, 
            7.2, 7.210222252, 7.240192379, 7.287867966, 7.35
        ]
        
        # Create timestamps for the data (daily intervals)
        from datetime import datetime, timedelta
        start_date = datetime(2023, 1, 1)
        timestamps = [start_date + timedelta(days=i) for i in time_steps]
        
        df = pd.DataFrame({
            'time_step': time_steps,
            'timestamp': timestamps,
            'pH': ph_values
        })
        
        print(f"Created pH dataset with {len(df)} samples")
        print(f"pH range: {min(ph_values):.3f} - {max(ph_values):.3f}")
        print(f"Sample data:")
        print(df.head())
        
        return df
    
    def test_ph_data_upload(self):
        """Test uploading pH dataset"""
        print("\n=== Testing pH Data Upload ===")
        
        try:
            # Create pH dataset
            df = self.create_ph_dataset()
            csv_content = df.to_csv(index=False)
            
            # Prepare file for upload
            files = {
                'file': ('ph_data.csv', csv_content, 'text/csv')
            }
            
            # Test file upload
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                
                print("✅ pH data upload successful")
                print(f"   Data ID: {self.data_id}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   Suggested parameters: {data['analysis']['suggested_parameters']}")
                
                return True
                
            else:
                print(f"❌ pH data upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ pH data upload error: {str(e)}")
            return False
    
    def test_prophet_training_with_ph_data(self):
        """Test Prophet model training with pH data - this is the main test"""
        print("\n=== Testing Prophet Model Training with pH Data ===")
        
        if not self.data_id:
            print("❌ Cannot test Prophet training - no pH data uploaded")
            return False
            
        try:
            # Test different time column options
            time_column_options = ['timestamp', 'time_step']
            target_column = 'pH'
            
            for time_col in time_column_options:
                print(f"\n--- Testing with time_column: {time_col} ---")
                
                # Prepare training parameters
                training_params = {
                    "time_column": time_col,
                    "target_column": target_column,
                    "seasonality_mode": "additive",
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False
                }
                
                print(f"Training parameters: {training_params}")
                
                # Test Prophet model training
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": self.data_id, "model_type": "prophet"},
                    json=training_params
                )
                
                print(f"Response status: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    data = response.json()
                    model_id = data.get('model_id')
                    
                    print(f"✅ Prophet training successful with {time_col}")
                    print(f"   Model ID: {model_id}")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Message: {data.get('message')}")
                    
                    # Store the successful model_id
                    if not self.model_id:
                        self.model_id = model_id
                    
                    return True
                    
                elif response.status_code == 500:
                    print(f"❌ Prophet training failed with HTTP 500 - INTERNAL SERVER ERROR")
                    print(f"   Time column: {time_col}")
                    print(f"   Target column: {target_column}")
                    
                    # Try to get more details from the response
                    try:
                        error_data = response.json()
                        print(f"   Error details: {error_data}")
                    except:
                        print(f"   Raw error response: {response.text}")
                    
                    # Continue testing with other time columns
                    continue
                    
                else:
                    print(f"❌ Prophet training failed with status {response.status_code}")
                    print(f"   Response: {response.text}")
                    continue
            
            # If we get here, all time column options failed
            print("❌ Prophet training failed with all time column options")
            return False
                
        except Exception as e:
            print(f"❌ Prophet training error: {str(e)}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def test_prophet_training_detailed_debug(self):
        """Detailed debugging of Prophet training issues"""
        print("\n=== Detailed Prophet Training Debug ===")
        
        if not self.data_id:
            print("❌ Cannot debug Prophet training - no pH data uploaded")
            return False
        
        try:
            # Test with minimal parameters first
            minimal_params = {
                "time_column": "timestamp",
                "target_column": "pH"
            }
            
            print(f"Testing with minimal parameters: {minimal_params}")
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": "prophet"},
                json=minimal_params
            )
            
            print(f"Minimal params response status: {response.status_code}")
            
            if response.status_code == 500:
                print("❌ Even minimal parameters cause HTTP 500 error")
                try:
                    error_data = response.json()
                    print(f"Error details: {error_data}")
                    
                    # Check if it's a specific Prophet error
                    error_detail = error_data.get('detail', '')
                    if 'Prophet' in error_detail:
                        print("🔍 This appears to be a Prophet-specific error")
                    elif 'data' in error_detail.lower():
                        print("🔍 This appears to be a data preparation error")
                    elif 'column' in error_detail.lower():
                        print("🔍 This appears to be a column-related error")
                    else:
                        print("🔍 Generic server error - need to check backend logs")
                        
                except Exception as parse_error:
                    print(f"Could not parse error response: {parse_error}")
                    print(f"Raw response: {response.text}")
            
            # Test with different seasonality settings
            seasonality_variants = [
                {"seasonality_mode": "additive"},
                {"seasonality_mode": "multiplicative"},
                {"yearly_seasonality": False, "weekly_seasonality": False, "daily_seasonality": False},
            ]
            
            for variant in seasonality_variants:
                test_params = {
                    "time_column": "timestamp",
                    "target_column": "pH",
                    **variant
                }
                
                print(f"\nTesting seasonality variant: {variant}")
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": self.data_id, "model_type": "prophet"},
                    json=test_params
                )
                
                print(f"Variant response status: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"✅ Success with variant: {variant}")
                    return True
                elif response.status_code == 500:
                    try:
                        error_data = response.json()
                        print(f"❌ Failed with variant: {variant}")
                        print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                    except:
                        print(f"❌ Failed with variant: {variant} - Could not parse error")
            
            return False
            
        except Exception as e:
            print(f"❌ Debug test error: {str(e)}")
            return False
    
    def test_data_preparation_for_prophet(self):
        """Test if the data preparation is causing issues"""
        print("\n=== Testing Data Preparation for Prophet ===")
        
        if not self.data_id:
            print("❌ Cannot test data preparation - no pH data uploaded")
            return False
        
        # Create a simple test to see if we can access the uploaded data
        try:
            # Try to get historical data (this uses the same data preparation logic)
            response = self.session.get(f"{API_BASE_URL}/historical-data")
            
            print(f"Historical data response status: {response.status_code}")
            
            if response.status_code == 400:
                print("❌ Cannot get historical data - no model trained (expected)")
                print("   This suggests data upload worked but model training is the issue")
                return True
            elif response.status_code == 200:
                print("✅ Historical data accessible - data preparation likely working")
                return True
            else:
                print(f"❌ Unexpected historical data response: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Data preparation test error: {str(e)}")
            return False
    
    def test_arima_with_same_data(self):
        """Test ARIMA with the same pH data to isolate Prophet-specific issues"""
        print("\n=== Testing ARIMA with Same pH Data ===")
        
        if not self.data_id:
            print("❌ Cannot test ARIMA - no pH data uploaded")
            return False
            
        try:
            # Test ARIMA with the same data
            arima_params = {
                "time_column": "timestamp",
                "target_column": "pH",
                "order": [1, 1, 1]
            }
            
            print(f"ARIMA parameters: {arima_params}")
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": "arima"},
                json=arima_params
            )
            
            print(f"ARIMA response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ ARIMA training successful with same pH data")
                print(f"   Model ID: {data.get('model_id')}")
                print("   This suggests the issue is Prophet-specific, not data-related")
                return True
            else:
                print(f"❌ ARIMA training also failed: {response.status_code}")
                print(f"   Response: {response.text}")
                print("   This suggests a broader data preparation issue")
                return False
                
        except Exception as e:
            print(f"❌ ARIMA test error: {str(e)}")
            return False
    
    def run_prophet_focused_tests(self):
        """Run all Prophet-focused tests"""
        print("🔍 Starting Focused Prophet Model Testing")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Upload pH data
        results['ph_data_upload'] = self.test_ph_data_upload()
        
        # Test 2: Data preparation check
        results['data_preparation'] = self.test_data_preparation_for_prophet()
        
        # Test 3: ARIMA comparison test
        results['arima_comparison'] = self.test_arima_with_same_data()
        
        # Test 4: Prophet training (main test)
        results['prophet_training'] = self.test_prophet_training_with_ph_data()
        
        # Test 5: Detailed debugging
        if not results['prophet_training']:
            results['prophet_debug'] = self.test_prophet_training_detailed_debug()
        
        # Print summary
        self.print_prophet_test_summary(results)
        
        return results
    
    def print_prophet_test_summary(self, results):
        """Print Prophet test summary"""
        print("\n" + "=" * 60)
        print("📊 PROPHET MODEL TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} - {test_name.replace('_', ' ').title()}")
        
        # Specific analysis
        print("\n🔍 ANALYSIS:")
        
        if not results.get('ph_data_upload', False):
            print("❌ CRITICAL: pH data upload failed - cannot proceed with Prophet testing")
        elif not results.get('prophet_training', False):
            print("❌ CRITICAL: Prophet model training failed with HTTP 500 error")
            print("   This confirms the user's reported issue")
            
            if results.get('arima_comparison', False):
                print("✅ ARIMA works with same data - issue is Prophet-specific")
            else:
                print("❌ ARIMA also fails - broader data preparation issue")
                
            if results.get('data_preparation', False):
                print("✅ Data preparation appears to work - issue is in Prophet training logic")
            else:
                print("❌ Data preparation may be the root cause")
        else:
            print("✅ Prophet model training working correctly")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    tester = ProphetTester()
    results = tester.run_prophet_focused_tests()
    
    # Focus on the main issue
    if not results.get('prophet_training', False):
        print("\n🚨 CONFIRMED: Prophet model training fails with HTTP 500 errors")
        print("   This matches the user's reported issue exactly")
        exit(1)
    else:
        print("\n✅ Prophet model training working correctly")
        exit(0)