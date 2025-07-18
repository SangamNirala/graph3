#!/usr/bin/env python3
"""
Focused Backend Testing for Prediction Flow Issues
Tests the specific prediction flow that frontend buttons depend on
"""

import requests
import json
import pandas as pd
import io
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://c8772c28-6b4b-4343-84fa-effeefd86ff0.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing prediction flow at: {API_BASE_URL}")

class PredictionFlowTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        
    def create_sample_data(self):
        """Create realistic time-series sample data for testing"""
        # Generate 100 days of daily sales data with trend and seasonality
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create realistic sales data with trend and weekly seasonality
        import numpy as np
        trend = np.linspace(1000, 1500, 100)
        seasonal = 200 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly pattern
        noise = np.random.normal(0, 50, 100)
        sales = trend + seasonal + noise
        
        df = pd.DataFrame({
            'date': dates,
            'sales': sales,
            'region': ['North'] * 50 + ['South'] * 50
        })
        
        return df
    
    def test_complete_prediction_flow(self):
        """Test the complete prediction flow that frontend buttons depend on"""
        print("\n=== TESTING COMPLETE PREDICTION FLOW ===")
        
        # Step 1: Upload data
        print("\n1. Testing File Upload...")
        df = self.create_sample_data()
        csv_content = df.to_csv(index=False)
        
        files = {'file': ('sales_data.csv', csv_content, 'text/csv')}
        response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
        
        if response.status_code != 200:
            print(f"❌ File upload failed: {response.status_code} - {response.text}")
            return False
            
        data = response.json()
        self.data_id = data.get('data_id')
        print(f"✅ File uploaded successfully. Data ID: {self.data_id}")
        
        # Step 2: Train ARIMA model (since it's more reliable than Prophet)
        print("\n2. Testing Model Training (ARIMA)...")
        training_params = {
            "time_column": "date",
            "target_column": "sales",
            "order": [1, 1, 1]
        }
        
        response = self.session.post(
            f"{API_BASE_URL}/train-model",
            params={"data_id": self.data_id, "model_type": "arima"},
            json=training_params
        )
        
        if response.status_code != 200:
            print(f"❌ Model training failed: {response.status_code} - {response.text}")
            return False
            
        data = response.json()
        self.model_id = data.get('model_id')
        print(f"✅ Model trained successfully. Model ID: {self.model_id}")
        
        # Step 3: Test Historical Data (Left Graph)
        print("\n3. Testing Historical Data Retrieval (Left Graph)...")
        response = self.session.get(f"{API_BASE_URL}/historical-data")
        
        if response.status_code != 200:
            print(f"❌ Historical data retrieval failed: {response.status_code} - {response.text}")
            return False
            
        historical_data = response.json()
        print(f"✅ Historical data retrieved: {len(historical_data.get('values', []))} points")
        print(f"   Sample historical values: {historical_data.get('values', [])[:3]}")
        print(f"   Sample historical timestamps: {historical_data.get('timestamps', [])[:3]}")
        
        # Step 4: Test Initial Prediction Generation (Right Graph)
        print("\n4. Testing Initial Prediction Generation (Right Graph)...")
        response = self.session.get(
            f"{API_BASE_URL}/generate-prediction",
            params={"model_id": self.model_id, "steps": 30}
        )
        
        if response.status_code != 200:
            print(f"❌ Initial prediction generation failed: {response.status_code} - {response.text}")
            return False
            
        initial_predictions = response.json()
        print(f"✅ Initial predictions generated: {len(initial_predictions.get('predictions', []))} points")
        print(f"   Sample prediction values: {initial_predictions.get('predictions', [])[:3]}")
        print(f"   Sample prediction timestamps: {initial_predictions.get('timestamps', [])[:3]}")
        
        # Step 5: Test Continuous Prediction Flow
        print("\n5. Testing Continuous Prediction Flow...")
        
        # 5a: Reset continuous prediction
        response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
        if response.status_code != 200:
            print(f"❌ Reset continuous prediction failed: {response.status_code}")
            return False
        print("✅ Continuous prediction reset")
        
        # 5b: Start continuous prediction
        response = self.session.post(f"{API_BASE_URL}/start-continuous-prediction")
        if response.status_code != 200:
            print(f"❌ Start continuous prediction failed: {response.status_code}")
            return False
        print("✅ Continuous prediction started")
        
        # 5c: Generate multiple continuous predictions to test extrapolation
        print("   Testing continuous extrapolation...")
        predictions_list = []
        
        for i in range(5):
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": self.model_id, "steps": 10, "time_window": 100}
            )
            
            if response.status_code != 200:
                print(f"❌ Continuous prediction {i+1} failed: {response.status_code}")
                return False
                
            prediction_data = response.json()
            predictions_list.append(prediction_data)
            print(f"   ✅ Continuous prediction {i+1}: {len(prediction_data.get('predictions', []))} points")
            print(f"      First timestamp: {prediction_data.get('timestamps', ['N/A'])[0]}")
            
            time.sleep(0.5)  # Small delay between calls
        
        # Verify extrapolation is working
        if len(predictions_list) >= 2:
            first_timestamps = predictions_list[0].get('timestamps', [])
            second_timestamps = predictions_list[1].get('timestamps', [])
            
            if first_timestamps != second_timestamps:
                print("✅ Continuous prediction properly extrapolates forward")
            else:
                print("❌ Continuous prediction NOT extrapolating (same timestamps)")
                return False
        
        # 5d: Stop continuous prediction
        response = self.session.post(f"{API_BASE_URL}/stop-continuous-prediction")
        if response.status_code != 200:
            print(f"❌ Stop continuous prediction failed: {response.status_code}")
            return False
        print("✅ Continuous prediction stopped")
        
        # Step 6: Test pH Simulation (Additional feature)
        print("\n6. Testing pH Simulation Integration...")
        
        # Test real-time pH
        response = self.session.get(f"{API_BASE_URL}/ph-simulation")
        if response.status_code != 200:
            print(f"❌ pH simulation failed: {response.status_code}")
            return False
            
        ph_data = response.json()
        print(f"✅ pH simulation working: pH={ph_data.get('ph_value')}, Confidence={ph_data.get('confidence')}")
        
        # Test pH history
        response = self.session.get(f"{API_BASE_URL}/ph-simulation-history")
        if response.status_code != 200:
            print(f"❌ pH history failed: {response.status_code}")
            return False
            
        ph_history = response.json()
        print(f"✅ pH history working: {len(ph_history.get('data', []))} data points")
        
        print("\n🎉 COMPLETE PREDICTION FLOW TEST PASSED!")
        print("   ✅ File upload working")
        print("   ✅ Model training working")
        print("   ✅ Historical data retrieval working (Left Graph)")
        print("   ✅ Initial prediction generation working (Right Graph)")
        print("   ✅ Continuous prediction extrapolation working")
        print("   ✅ pH simulation integration working")
        
        return True
    
    def test_frontend_button_scenarios(self):
        """Test specific scenarios that frontend buttons would trigger"""
        print("\n=== TESTING FRONTEND BUTTON SCENARIOS ===")
        
        if not self.model_id:
            print("❌ No model available for frontend button testing")
            return False
        
        # Scenario 1: "Generate Predictions" button
        print("\n1. Testing 'Generate Predictions' Button Scenario...")
        response = self.session.get(
            f"{API_BASE_URL}/generate-prediction",
            params={"model_id": self.model_id, "steps": 50}
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            timestamps = data.get('timestamps', [])
            
            if len(predictions) == 50 and len(timestamps) == 50:
                print("✅ 'Generate Predictions' button scenario works correctly")
                print(f"   Generated {len(predictions)} predictions")
                print(f"   Time range: {timestamps[0]} to {timestamps[-1]}")
            else:
                print("❌ 'Generate Predictions' button scenario failed - incorrect data structure")
                return False
        else:
            print(f"❌ 'Generate Predictions' button scenario failed: {response.status_code}")
            return False
        
        # Scenario 2: "Start Continuous Prediction" button
        print("\n2. Testing 'Start Continuous Prediction' Button Scenario...")
        
        # Reset first
        self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
        
        # Start continuous prediction
        response = self.session.post(f"{API_BASE_URL}/start-continuous-prediction")
        if response.status_code != 200:
            print(f"❌ 'Start Continuous Prediction' button scenario failed: {response.status_code}")
            return False
        
        print("✅ 'Start Continuous Prediction' button scenario works")
        
        # Test multiple continuous calls (simulating frontend polling)
        print("   Simulating frontend polling for continuous updates...")
        for i in range(3):
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": self.model_id, "steps": 20, "time_window": 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Continuous update {i+1}: {len(data.get('predictions', []))} predictions")
            else:
                print(f"   ❌ Continuous update {i+1} failed: {response.status_code}")
                return False
            
            time.sleep(1)  # Simulate frontend polling interval
        
        # Stop continuous prediction
        response = self.session.post(f"{API_BASE_URL}/stop-continuous-prediction")
        if response.status_code == 200:
            print("✅ 'Stop Continuous Prediction' works correctly")
        else:
            print(f"❌ 'Stop Continuous Prediction' failed: {response.status_code}")
            return False
        
        print("\n🎉 FRONTEND BUTTON SCENARIOS TEST PASSED!")
        return True

def main():
    tester = PredictionFlowTester()
    
    print("🚀 Starting Focused Prediction Flow Testing")
    print("=" * 60)
    
    # Test complete prediction flow
    flow_success = tester.test_complete_prediction_flow()
    
    if flow_success:
        # Test frontend button scenarios
        button_success = tester.test_frontend_button_scenarios()
        
        if button_success:
            print("\n" + "=" * 60)
            print("🎉 ALL PREDICTION FLOW TESTS PASSED!")
            print("   The backend APIs are working correctly for:")
            print("   - File upload and data analysis")
            print("   - Model training (ARIMA)")
            print("   - Historical data retrieval (Left Graph)")
            print("   - Initial prediction generation (Right Graph)")
            print("   - Continuous prediction extrapolation")
            print("   - pH simulation integration")
            print("   - Frontend button scenarios")
            print("\n   If frontend buttons are not working, the issue is likely:")
            print("   1. Frontend-backend communication problems")
            print("   2. Frontend state management issues")
            print("   3. Frontend graph rendering problems")
            print("   4. Frontend error handling not displaying errors")
            print("=" * 60)
            return True
        else:
            print("\n❌ Frontend button scenarios failed")
            return False
    else:
        print("\n❌ Prediction flow test failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)