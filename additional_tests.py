#!/usr/bin/env python3
"""
Additional Data Quality Report Testing
"""

import requests
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://2a292801-ad4e-4bb1-a343-8bf056863289.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing data quality report at: {API_BASE_URL}")

class DataQualityTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        
    def upload_test_data(self):
        """Upload test data first"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        values = np.random.normal(100, 10, 50)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'ph_value': values,
            'category': ['A'] * 25 + ['B'] * 25
        })
        
        csv_content = df.to_csv(index=False)
        
        files = {
            'file': ('test_data.csv', csv_content, 'text/csv')
        }
        
        response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
        
        if response.status_code == 200:
            data = response.json()
            self.data_id = data.get('data_id')
            print("✅ Test data uploaded successfully")
            return True
        else:
            print(f"❌ Test data upload failed: {response.status_code}")
            return False
    
    def test_data_quality_report(self):
        """Test data quality report endpoint"""
        print("\n=== Testing Data Quality Report Endpoint ===")
        
        if not self.data_id:
            print("❌ No data uploaded for quality report test")
            return False
            
        try:
            response = self.session.get(f"{API_BASE_URL}/data-quality-report")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Data quality report successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Quality Score: {data.get('quality_score')}")
                print(f"   Recommendations: {len(data.get('recommendations', []))}")
                
                # Validate response structure
                if (data.get('status') == 'success' and 
                    'quality_score' in data and
                    'recommendations' in data):
                    print("✅ Data quality report structure correct")
                    return True
                else:
                    print("❌ Data quality report structure incorrect")
                    return False
                    
            else:
                print(f"❌ Data quality report failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Data quality report error: {str(e)}")
            return False
    
    def test_supported_models_endpoint(self):
        """Test supported models endpoint"""
        print("\n=== Testing Supported Models Endpoint ===")
        
        try:
            response = self.session.get(f"{API_BASE_URL}/supported-models")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Supported models endpoint successful")
                print(f"   Models: {data.get('models', [])}")
                
                # Validate response structure
                models = data.get('models', [])
                if len(models) > 0 and 'prophet' in models:
                    print("✅ Supported models list correct")
                    return True
                else:
                    print("❌ Supported models list incorrect")
                    return False
                    
            else:
                print(f"❌ Supported models endpoint failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Supported models endpoint error: {str(e)}")
            return False
    
    def run_tests(self):
        """Run all additional tests"""
        print("🚀 Starting Additional Data Quality Testing")
        print("=" * 50)
        
        results = []
        
        # Upload test data first
        if self.upload_test_data():
            results.append(self.test_data_quality_report())
        else:
            results.append(False)
            
        results.append(self.test_supported_models_endpoint())
        
        # Print summary
        passed = sum(results)
        total = len(results)
        
        print(f"\n📊 Additional Tests Summary: {passed}/{total} passed ({(passed/total)*100:.1f}%)")
        
        return passed, total

if __name__ == "__main__":
    tester = DataQualityTester()
    tester.run_tests()