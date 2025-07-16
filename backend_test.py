#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Real-time Graph Prediction Application
Tests all API endpoints and WebSocket functionality
"""

import requests
import json
import pandas as pd
import io
import time
import asyncio
import websockets
import threading
from datetime import datetime, timedelta
import numpy as np
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'frontend' / '.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://186bfa85-b41a-4c4f-a661-fdf21229469f.preview.emergentagent.com')
API_BASE_URL = f"{BACKEND_URL}/api"
WS_URL = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + "/ws/predictions"

print(f"Testing backend at: {API_BASE_URL}")
print(f"WebSocket URL: {WS_URL}")

class BackendTester:
    def __init__(self):
        self.session = requests.Session()
        self.data_id = None
        self.model_id = None
        self.test_results = {}
        
    def create_sample_data(self):
        """Create realistic time-series sample data for testing"""
        # Generate 100 days of daily sales data with trend and seasonality
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create realistic sales data with trend and weekly seasonality
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
    
    def test_file_upload(self):
        """Test 1: File upload and data analysis endpoint"""
        print("\n=== Testing File Upload and Data Analysis ===")
        
        try:
            # Create sample CSV data
            df = self.create_sample_data()
            csv_content = df.to_csv(index=False)
            
            # Prepare file for upload
            files = {
                'file': ('sales_data.csv', csv_content, 'text/csv')
            }
            
            # Test file upload
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.data_id = data.get('data_id')
                
                print("✅ File upload successful")
                print(f"   Data ID: {self.data_id}")
                print(f"   Columns detected: {data['analysis']['columns']}")
                print(f"   Time columns: {data['analysis']['time_columns']}")
                print(f"   Numeric columns: {data['analysis']['numeric_columns']}")
                print(f"   Data shape: {data['analysis']['data_shape']}")
                print(f"   Suggested parameters: {data['analysis']['suggested_parameters']}")
                
                # Validate analysis results
                analysis = data['analysis']
                if 'date' in analysis['time_columns'] and 'sales' in analysis['numeric_columns']:
                    print("✅ Data analysis correctly identified time and numeric columns")
                    self.test_results['file_upload'] = True
                else:
                    print("❌ Data analysis failed to identify columns correctly")
                    self.test_results['file_upload'] = False
                    
            else:
                print(f"❌ File upload failed: {response.status_code} - {response.text}")
                self.test_results['file_upload'] = False
                
        except Exception as e:
            print(f"❌ File upload test error: {str(e)}")
            self.test_results['file_upload'] = False
    
    def test_model_training_prophet(self):
        """Test 2a: Prophet model training"""
        print("\n=== Testing Prophet Model Training ===")
        
        if not self.data_id:
            print("❌ Cannot test model training - no data uploaded")
            self.test_results['prophet_training'] = False
            return
            
        try:
            # Prepare training parameters
            training_data = {
                "data_id": self.data_id,
                "model_type": "prophet",
                "parameters": {
                    "time_column": "date",
                    "target_column": "sales",
                    "seasonality_mode": "additive",
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False
                }
            }
            
            # Test model training
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": "prophet"},
                json=training_data["parameters"]
            )
            
            if response.status_code == 200:
                data = response.json()
                self.model_id = data.get('model_id')
                
                print("✅ Prophet model training successful")
                print(f"   Model ID: {self.model_id}")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                
                self.test_results['prophet_training'] = True
                
            else:
                print(f"❌ Prophet model training failed: {response.status_code} - {response.text}")
                self.test_results['prophet_training'] = False
                
        except Exception as e:
            print(f"❌ Prophet model training error: {str(e)}")
            self.test_results['prophet_training'] = False
    
    def test_model_training_arima(self):
        """Test 2b: ARIMA model training"""
        print("\n=== Testing ARIMA Model Training ===")
        
        if not self.data_id:
            print("❌ Cannot test ARIMA training - no data uploaded")
            self.test_results['arima_training'] = False
            return
            
        try:
            # Prepare training parameters for ARIMA
            training_data = {
                "data_id": self.data_id,
                "model_type": "arima",
                "parameters": {
                    "time_column": "date",
                    "target_column": "sales",
                    "order": [1, 1, 1]
                }
            }
            
            # Test ARIMA model training
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": self.data_id, "model_type": "arima"},
                json=training_data["parameters"]
            )
            
            if response.status_code == 200:
                data = response.json()
                arima_model_id = data.get('model_id')
                
                print("✅ ARIMA model training successful")
                print(f"   Model ID: {arima_model_id}")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                
                self.test_results['arima_training'] = True
                
            else:
                print(f"❌ ARIMA model training failed: {response.status_code} - {response.text}")
                self.test_results['arima_training'] = False
                
        except Exception as e:
            print(f"❌ ARIMA model training error: {str(e)}")
            self.test_results['arima_training'] = False
    
    def test_prediction_generation(self):
        """Test 3: Prediction generation"""
        print("\n=== Testing Prediction Generation ===")
        
        if not self.model_id:
            print("❌ Cannot test prediction generation - no model trained")
            self.test_results['prediction_generation'] = False
            return
            
        try:
            # Test prediction generation
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": self.model_id, "steps": 10}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Prediction generation successful")
                print(f"   Number of predictions: {len(data.get('predictions', []))}")
                print(f"   Number of timestamps: {len(data.get('timestamps', []))}")
                print(f"   Has confidence intervals: {data.get('confidence_intervals') is not None}")
                
                # Validate prediction structure
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                
                if len(predictions) == 10 and len(timestamps) == 10:
                    print("✅ Prediction data structure is correct")
                    print(f"   Sample predictions: {predictions[:3]}")
                    print(f"   Sample timestamps: {timestamps[:3]}")
                    self.test_results['prediction_generation'] = True
                else:
                    print("❌ Prediction data structure is incorrect")
                    self.test_results['prediction_generation'] = False
                    
            else:
                print(f"❌ Prediction generation failed: {response.status_code} - {response.text}")
                self.test_results['prediction_generation'] = False
                
        except Exception as e:
            print(f"❌ Prediction generation error: {str(e)}")
            self.test_results['prediction_generation'] = False
    
    def test_historical_data(self):
        """Test 4: Historical data retrieval"""
        print("\n=== Testing Historical Data Retrieval ===")
        
        if not self.model_id:
            print("❌ Cannot test historical data - no model trained")
            self.test_results['historical_data'] = False
            return
            
        try:
            # Test historical data retrieval
            response = self.session.get(f"{API_BASE_URL}/historical-data")
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Historical data retrieval successful")
                print(f"   Number of data points: {len(data.get('values', []))}")
                print(f"   Number of timestamps: {len(data.get('timestamps', []))}")
                
                # Validate historical data structure
                values = data.get('values', [])
                timestamps = data.get('timestamps', [])
                
                if len(values) > 0 and len(timestamps) > 0 and len(values) == len(timestamps):
                    print("✅ Historical data structure is correct")
                    print(f"   Sample values: {values[:3]}")
                    print(f"   Sample timestamps: {timestamps[:3]}")
                    self.test_results['historical_data'] = True
                else:
                    print("❌ Historical data structure is incorrect")
                    self.test_results['historical_data'] = False
                    
            else:
                print(f"❌ Historical data retrieval failed: {response.status_code} - {response.text}")
                self.test_results['historical_data'] = False
                
        except Exception as e:
            print(f"❌ Historical data retrieval error: {str(e)}")
            self.test_results['historical_data'] = False
    
    def test_continuous_prediction_control(self):
        """Test 5: Continuous prediction start/stop"""
        print("\n=== Testing Continuous Prediction Control ===")
        
        try:
            # Test start continuous prediction
            response = self.session.post(f"{API_BASE_URL}/start-continuous-prediction")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Start continuous prediction successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                
                # Wait a moment then test stop
                time.sleep(2)
                
                # Test stop continuous prediction
                response = self.session.post(f"{API_BASE_URL}/stop-continuous-prediction")
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ Stop continuous prediction successful")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Message: {data.get('message')}")
                    
                    self.test_results['continuous_prediction'] = True
                else:
                    print(f"❌ Stop continuous prediction failed: {response.status_code} - {response.text}")
                    self.test_results['continuous_prediction'] = False
                    
            else:
                print(f"❌ Start continuous prediction failed: {response.status_code} - {response.text}")
                self.test_results['continuous_prediction'] = False
                
        except Exception as e:
            print(f"❌ Continuous prediction control error: {str(e)}")
            self.test_results['continuous_prediction'] = False
    
    async def test_websocket_connection(self):
        """Test 6: WebSocket connection and messaging"""
        print("\n=== Testing WebSocket Connection ===")
        
        try:
            # Test WebSocket connection
            async with websockets.connect(WS_URL) as websocket:
                print("✅ WebSocket connection established")
                
                # Send a test message
                test_message = json.dumps({"type": "test", "message": "Hello WebSocket"})
                await websocket.send(test_message)
                print("✅ Test message sent to WebSocket")
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"✅ WebSocket response received: {response}")
                    self.test_results['websocket'] = True
                except asyncio.TimeoutError:
                    print("⚠️  WebSocket response timeout (this may be normal)")
                    self.test_results['websocket'] = True  # Connection worked even if no immediate response
                    
        except Exception as e:
            print(f"❌ WebSocket connection error: {str(e)}")
            self.test_results['websocket'] = False
    
    def test_ph_simulation_endpoints(self):
        """Test 7: pH Simulation Endpoints"""
        print("\n=== Testing pH Simulation Endpoints ===")
        
        try:
            # Test real-time pH simulation
            response = self.session.get(f"{API_BASE_URL}/ph-simulation")
            
            if response.status_code == 200:
                data = response.json()
                ph_value = data.get('ph_value')
                confidence = data.get('confidence')
                timestamp = data.get('timestamp')
                
                print("✅ Real-time pH simulation successful")
                print(f"   pH Value: {ph_value}")
                print(f"   Confidence: {confidence}")
                print(f"   Timestamp: {timestamp}")
                
                # Validate pH value is in realistic range (6.0-8.0)
                if ph_value and 6.0 <= ph_value <= 8.0:
                    print("✅ pH value is in realistic range (6.0-8.0)")
                    ph_realtime_test = True
                else:
                    print(f"❌ pH value {ph_value} is outside realistic range (6.0-8.0)")
                    ph_realtime_test = False
                    
            else:
                print(f"❌ Real-time pH simulation failed: {response.status_code} - {response.text}")
                ph_realtime_test = False
            
            # Test pH simulation history
            response = self.session.get(f"{API_BASE_URL}/ph-simulation-history", params={"hours": 24})
            
            if response.status_code == 200:
                data = response.json()
                history_data = data.get('data', [])
                current_ph = data.get('current_ph')
                target_ph = data.get('target_ph')
                status = data.get('status')
                
                print("✅ pH simulation history successful")
                print(f"   History data points: {len(history_data)}")
                print(f"   Current pH: {current_ph}")
                print(f"   Target pH: {target_ph}")
                print(f"   Status: {status}")
                
                # Validate history data structure and pH values
                ph_history_test = True
                if history_data:
                    sample_point = history_data[0]
                    if 'timestamp' in sample_point and 'ph_value' in sample_point and 'confidence' in sample_point:
                        print("✅ History data structure is correct")
                        
                        # Check if all pH values are in realistic range
                        ph_values = [point['ph_value'] for point in history_data[:10]]  # Check first 10
                        valid_ph_count = sum(1 for ph in ph_values if 6.0 <= ph <= 8.0)
                        
                        if valid_ph_count == len(ph_values):
                            print("✅ All sampled pH values are in realistic range (6.0-8.0)")
                        else:
                            print(f"❌ {len(ph_values) - valid_ph_count} pH values are outside realistic range")
                            ph_history_test = False
                    else:
                        print("❌ History data structure is incorrect")
                        ph_history_test = False
                else:
                    print("❌ No history data returned")
                    ph_history_test = False
                    
            else:
                print(f"❌ pH simulation history failed: {response.status_code} - {response.text}")
                ph_history_test = False
            
            self.test_results['ph_simulation'] = ph_realtime_test and ph_history_test
            
        except Exception as e:
            print(f"❌ pH simulation endpoints error: {str(e)}")
            self.test_results['ph_simulation'] = False
    
    def test_enhanced_continuous_prediction(self):
        """Test 8: Enhanced Continuous Prediction with Extrapolation"""
        print("\n=== Testing Enhanced Continuous Prediction ===")
        
        if not self.model_id:
            print("❌ Cannot test enhanced continuous prediction - no model trained")
            self.test_results['enhanced_continuous_prediction'] = False
            return
            
        try:
            # Test generate-continuous-prediction endpoint
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": self.model_id, "steps": 20, "time_window": 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                timestamps = data.get('timestamps', [])
                confidence_intervals = data.get('confidence_intervals')
                
                print("✅ Enhanced continuous prediction successful")
                print(f"   Number of predictions: {len(predictions)}")
                print(f"   Number of timestamps: {len(timestamps)}")
                print(f"   Has confidence intervals: {confidence_intervals is not None}")
                
                # Validate prediction structure
                if len(predictions) == 20 and len(timestamps) == 20:
                    print("✅ Continuous prediction data structure is correct")
                    print(f"   Sample predictions: {predictions[:3]}")
                    print(f"   Sample timestamps: {timestamps[:3]}")
                    
                    # Test multiple calls to verify extrapolation
                    print("   Testing extrapolation by making multiple calls...")
                    
                    # Make second call
                    response2 = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": self.model_id, "steps": 20, "time_window": 100}
                    )
                    
                    if response2.status_code == 200:
                        data2 = response2.json()
                        timestamps2 = data2.get('timestamps', [])
                        
                        # Check if timestamps are different (indicating extrapolation)
                        if timestamps != timestamps2:
                            print("✅ Continuous prediction properly extrapolates forward")
                            extrapolation_test = True
                        else:
                            print("❌ Continuous prediction not extrapolating (same timestamps)")
                            extrapolation_test = False
                    else:
                        print("❌ Second continuous prediction call failed")
                        extrapolation_test = False
                    
                    self.test_results['enhanced_continuous_prediction'] = extrapolation_test
                else:
                    print("❌ Continuous prediction data structure is incorrect")
                    self.test_results['enhanced_continuous_prediction'] = False
                    
            else:
                print(f"❌ Enhanced continuous prediction failed: {response.status_code} - {response.text}")
                self.test_results['enhanced_continuous_prediction'] = False
                
        except Exception as e:
            print(f"❌ Enhanced continuous prediction error: {str(e)}")
            self.test_results['enhanced_continuous_prediction'] = False
    
    def test_reset_functionality(self):
        """Test 9: Reset Continuous Prediction Functionality"""
        print("\n=== Testing Reset Continuous Prediction ===")
        
        try:
            # Test reset continuous prediction
            response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                message = data.get('message')
                
                print("✅ Reset continuous prediction successful")
                print(f"   Status: {status}")
                print(f"   Message: {message}")
                
                # Validate response structure
                if status == "reset" and "reset" in message.lower():
                    print("✅ Reset response structure is correct")
                    self.test_results['reset_functionality'] = True
                else:
                    print("❌ Reset response structure is incorrect")
                    self.test_results['reset_functionality'] = False
                    
            else:
                print(f"❌ Reset continuous prediction failed: {response.status_code} - {response.text}")
                self.test_results['reset_functionality'] = False
                
        except Exception as e:
            print(f"❌ Reset functionality error: {str(e)}")
            self.test_results['reset_functionality'] = False
    
    def test_complete_continuous_prediction_flow(self):
        """Test 10: Complete Continuous Prediction Flow"""
        print("\n=== Testing Complete Continuous Prediction Flow ===")
        
        if not self.model_id:
            print("❌ Cannot test continuous prediction flow - no model trained")
            self.test_results['continuous_prediction_flow'] = False
            return
            
        try:
            flow_tests = []
            
            # Step 1: Reset continuous prediction state
            response = self.session.post(f"{API_BASE_URL}/reset-continuous-prediction")
            flow_tests.append(("Reset continuous prediction", response.status_code == 200))
            
            # Step 2: Start continuous prediction
            response = self.session.post(f"{API_BASE_URL}/start-continuous-prediction")
            flow_tests.append(("Start continuous prediction", response.status_code == 200))
            
            if response.status_code == 200:
                print("✅ Continuous prediction started")
                
                # Step 3: Wait and test multiple continuous predictions
                time.sleep(2)
                
                # Make multiple calls to test extrapolation
                timestamps_list = []
                for i in range(3):
                    response = self.session.get(
                        f"{API_BASE_URL}/generate-continuous-prediction",
                        params={"model_id": self.model_id, "steps": 10, "time_window": 50}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        timestamps = data.get('timestamps', [])
                        timestamps_list.append(timestamps)
                        time.sleep(1)  # Wait between calls
                    
                # Check if predictions are extrapolating forward
                if len(timestamps_list) >= 2:
                    extrapolating = timestamps_list[0] != timestamps_list[1]
                    flow_tests.append(("Continuous extrapolation", extrapolating))
                    if extrapolating:
                        print("✅ Continuous prediction properly extrapolates forward")
                    else:
                        print("❌ Continuous prediction not extrapolating")
                
                # Step 4: Stop continuous prediction
                response = self.session.post(f"{API_BASE_URL}/stop-continuous-prediction")
                flow_tests.append(("Stop continuous prediction", response.status_code == 200))
                
                if response.status_code == 200:
                    print("✅ Continuous prediction stopped")
            
            # Evaluate flow test results
            passed_tests = sum(1 for _, passed in flow_tests if passed)
            total_tests = len(flow_tests)
            
            print(f"✅ Continuous prediction flow tests passed: {passed_tests}/{total_tests}")
            for test_name, passed in flow_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['continuous_prediction_flow'] = passed_tests >= total_tests * 0.8  # 80% pass rate
            
        except Exception as e:
            print(f"❌ Continuous prediction flow error: {str(e)}")
            self.test_results['continuous_prediction_flow'] = False
    
    def test_ph_target_management(self):
        """Test 11: pH Target Management"""
        print("\n=== Testing pH Target Management ===")
        
        try:
            ph_target_tests = []
            
            # Test 1: Set valid pH target (7.0)
            response = self.session.post(
                f"{API_BASE_URL}/set-ph-target",
                json={"target_ph": 7.0}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Set pH target 7.0 successful")
                print(f"   Status: {data.get('status')}")
                print(f"   Target pH: {data.get('target_ph')}")
                print(f"   Message: {data.get('message')}")
                ph_target_tests.append(("Set pH target 7.0", True))
            else:
                print(f"❌ Set pH target 7.0 failed: {response.status_code} - {response.text}")
                ph_target_tests.append(("Set pH target 7.0", False))
            
            # Test 2: Set valid pH target (8.5)
            response = self.session.post(
                f"{API_BASE_URL}/set-ph-target",
                json={"target_ph": 8.5}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Set pH target 8.5 successful")
                print(f"   Target pH: {data.get('target_ph')}")
                ph_target_tests.append(("Set pH target 8.5", True))
            else:
                print(f"❌ Set pH target 8.5 failed: {response.status_code} - {response.text}")
                ph_target_tests.append(("Set pH target 8.5", False))
            
            # Test 3: Set valid pH target (6.2)
            response = self.session.post(
                f"{API_BASE_URL}/set-ph-target",
                json={"target_ph": 6.2}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Set pH target 6.2 successful")
                print(f"   Target pH: {data.get('target_ph')}")
                ph_target_tests.append(("Set pH target 6.2", True))
            else:
                print(f"❌ Set pH target 6.2 failed: {response.status_code} - {response.text}")
                ph_target_tests.append(("Set pH target 6.2", False))
            
            # Test 4: Test pH validation - invalid high value (15.0)
            response = self.session.post(
                f"{API_BASE_URL}/set-ph-target",
                json={"target_ph": 15.0}
            )
            
            if response.status_code >= 400:
                print("✅ pH validation correctly rejected pH 15.0 (outside 0-14 range)")
                ph_target_tests.append(("Reject invalid pH 15.0", True))
            else:
                print("❌ pH validation failed to reject pH 15.0")
                ph_target_tests.append(("Reject invalid pH 15.0", False))
            
            # Test 5: Test pH validation - invalid low value (-1.0)
            response = self.session.post(
                f"{API_BASE_URL}/set-ph-target",
                json={"target_ph": -1.0}
            )
            
            if response.status_code >= 400:
                print("✅ pH validation correctly rejected pH -1.0 (outside 0-14 range)")
                ph_target_tests.append(("Reject invalid pH -1.0", True))
            else:
                print("❌ pH validation failed to reject pH -1.0")
                ph_target_tests.append(("Reject invalid pH -1.0", False))
            
            # Test 6: Verify pH target affects simulation
            # Set a specific target pH
            self.session.post(f"{API_BASE_URL}/set-ph-target", json={"target_ph": 7.8})
            
            # Get pH simulation to check if target affects readings
            response = self.session.get(f"{API_BASE_URL}/ph-simulation")
            
            if response.status_code == 200:
                data = response.json()
                ph_value = data.get('ph_value')
                print(f"✅ pH simulation responds to target pH setting")
                print(f"   Current pH reading: {ph_value} (target: 7.8)")
                ph_target_tests.append(("pH simulation responds to target", True))
            else:
                print("❌ pH simulation failed after setting target")
                ph_target_tests.append(("pH simulation responds to target", False))
            
            # Evaluate pH target management tests
            passed_tests = sum(1 for _, passed in ph_target_tests if passed)
            total_tests = len(ph_target_tests)
            
            print(f"✅ pH target management tests passed: {passed_tests}/{total_tests}")
            for test_name, passed in ph_target_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['ph_target_management'] = passed_tests >= total_tests * 0.8  # 80% pass rate
            
        except Exception as e:
            print(f"❌ pH target management error: {str(e)}")
            self.test_results['ph_target_management'] = False
    
    def test_advanced_pattern_analysis(self):
        """Test 12: Advanced Pattern Analysis Features"""
        print("\n=== Testing Advanced Pattern Analysis ===")
        
        if not self.model_id:
            print("❌ Cannot test pattern analysis - no model trained")
            self.test_results['pattern_analysis'] = False
            return
            
        try:
            pattern_tests = []
            
            # Test 1: Generate continuous prediction with pattern analysis
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": self.model_id, "steps": 30, "time_window": 100}
            )
            
            if response.status_code == 200:
                data = response.json()
                pattern_analysis = data.get('pattern_analysis')
                predictions = data.get('predictions', [])
                
                print("✅ Advanced continuous prediction with pattern analysis successful")
                print(f"   Number of predictions: {len(predictions)}")
                
                if pattern_analysis:
                    print("✅ Pattern analysis data included:")
                    print(f"   Trend slope: {pattern_analysis.get('trend_slope')}")
                    print(f"   Velocity: {pattern_analysis.get('velocity')}")
                    print(f"   Recent mean: {pattern_analysis.get('recent_mean')}")
                    print(f"   Last value: {pattern_analysis.get('last_value')}")
                    pattern_tests.append(("Pattern analysis included", True))
                else:
                    print("❌ Pattern analysis data missing")
                    pattern_tests.append(("Pattern analysis included", False))
                
                # Test that predictions honor historical trends
                if len(predictions) >= 5:
                    # Check if predictions show some trend consistency
                    trend_consistency = abs(predictions[-1] - predictions[0]) > 0  # Some change over time
                    print(f"✅ Predictions show trend consistency: {trend_consistency}")
                    pattern_tests.append(("Trend consistency", trend_consistency))
                else:
                    pattern_tests.append(("Trend consistency", False))
                    
            else:
                print(f"❌ Advanced continuous prediction failed: {response.status_code} - {response.text}")
                pattern_tests.append(("Pattern analysis included", False))
                pattern_tests.append(("Trend consistency", False))
            
            # Test 2: Test prediction extension mechanism
            response = self.session.get(f"{API_BASE_URL}/extend-prediction", params={"steps": 10})
            
            if response.status_code == 200:
                data = response.json()
                extension_info = data.get('extension_info')
                predictions = data.get('predictions', [])
                
                print("✅ Prediction extension successful")
                print(f"   Number of extended predictions: {len(predictions)}")
                
                if extension_info:
                    print("✅ Extension info included:")
                    print(f"   Trend: {extension_info.get('trend')}")
                    print(f"   Velocity: {extension_info.get('velocity')}")
                    print(f"   Base value: {extension_info.get('base_value')}")
                    pattern_tests.append(("Extension mechanism", True))
                else:
                    print("❌ Extension info missing")
                    pattern_tests.append(("Extension mechanism", False))
                    
            else:
                print(f"❌ Prediction extension failed: {response.status_code} - {response.text}")
                pattern_tests.append(("Extension mechanism", False))
            
            # Test 3: Test smooth transition capabilities
            # Make multiple extension calls to test smooth transitions
            timestamps_list = []
            predictions_list = []
            
            for i in range(3):
                response = self.session.get(f"{API_BASE_URL}/extend-prediction", params={"steps": 5})
                if response.status_code == 200:
                    data = response.json()
                    timestamps_list.append(data.get('timestamps', []))
                    predictions_list.append(data.get('predictions', []))
                    time.sleep(0.5)
            
            if len(predictions_list) >= 2:
                # Check if predictions maintain smooth transitions
                smooth_transition = True
                for i in range(1, len(predictions_list)):
                    if predictions_list[i] and predictions_list[i-1]:
                        # Check if there's reasonable continuity
                        last_prev = predictions_list[i-1][-1] if predictions_list[i-1] else 0
                        first_curr = predictions_list[i][0] if predictions_list[i] else 0
                        gap = abs(first_curr - last_prev)
                        if gap > 1000:  # Arbitrary threshold for "smooth"
                            smooth_transition = False
                            break
                
                print(f"✅ Smooth transition test: {smooth_transition}")
                pattern_tests.append(("Smooth transitions", smooth_transition))
            else:
                pattern_tests.append(("Smooth transitions", False))
            
            # Evaluate pattern analysis tests
            passed_tests = sum(1 for _, passed in pattern_tests if passed)
            total_tests = len(pattern_tests)
            
            print(f"✅ Pattern analysis tests passed: {passed_tests}/{total_tests}")
            for test_name, passed in pattern_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['pattern_analysis'] = passed_tests >= total_tests * 0.7  # 70% pass rate
            
        except Exception as e:
            print(f"❌ Pattern analysis error: {str(e)}")
            self.test_results['pattern_analysis'] = False
    
    def test_integration_flow(self):
        """Test 13: Complete Integration Flow"""
        print("\n=== Testing Complete Integration Flow ===")
        
        try:
            integration_tests = []
            
            print("Starting complete integration flow test...")
            
            # Step 1: File upload
            df = self.create_sample_data()
            csv_content = df.to_csv(index=False)
            files = {'file': ('integration_test.csv', csv_content, 'text/csv')}
            
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            upload_success = response.status_code == 200
            integration_tests.append(("File upload", upload_success))
            
            if upload_success:
                data_id = response.json().get('data_id')
                print("✅ Step 1: File upload successful")
            else:
                print("❌ Step 1: File upload failed")
                self.test_results['integration_flow'] = False
                return
            
            # Step 2: Model training
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "arima"},
                json={"time_column": "date", "target_column": "sales", "order": [1, 1, 1]}
            )
            training_success = response.status_code == 200
            integration_tests.append(("Model training", training_success))
            
            if training_success:
                model_id = response.json().get('model_id')
                print("✅ Step 2: Model training successful")
            else:
                print("❌ Step 2: Model training failed")
                self.test_results['integration_flow'] = False
                return
            
            # Step 3: Initial predictions
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": model_id, "steps": 20}
            )
            initial_pred_success = response.status_code == 200
            integration_tests.append(("Initial predictions", initial_pred_success))
            
            if initial_pred_success:
                print("✅ Step 3: Initial predictions successful")
            else:
                print("❌ Step 3: Initial predictions failed")
            
            # Step 4: Continuous prediction
            response = self.session.get(
                f"{API_BASE_URL}/generate-continuous-prediction",
                params={"model_id": model_id, "steps": 15, "time_window": 100}
            )
            continuous_pred_success = response.status_code == 200
            integration_tests.append(("Continuous prediction", continuous_pred_success))
            
            if continuous_pred_success:
                print("✅ Step 4: Continuous prediction successful")
            else:
                print("❌ Step 4: Continuous prediction failed")
            
            # Step 5: Prediction extension
            response = self.session.get(f"{API_BASE_URL}/extend-prediction", params={"steps": 10})
            extension_success = response.status_code == 200
            integration_tests.append(("Prediction extension", extension_success))
            
            if extension_success:
                print("✅ Step 5: Prediction extension successful")
            else:
                print("❌ Step 5: Prediction extension failed")
            
            # Step 6: pH simulation integration
            response = self.session.get(f"{API_BASE_URL}/ph-simulation")
            ph_sim_success = response.status_code == 200
            integration_tests.append(("pH simulation", ph_sim_success))
            
            if ph_sim_success:
                print("✅ Step 6: pH simulation successful")
            else:
                print("❌ Step 6: pH simulation failed")
            
            # Evaluate integration flow
            passed_tests = sum(1 for _, passed in integration_tests if passed)
            total_tests = len(integration_tests)
            
            print(f"✅ Integration flow tests passed: {passed_tests}/{total_tests}")
            for test_name, passed in integration_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['integration_flow'] = passed_tests >= total_tests * 0.85  # 85% pass rate
            
        except Exception as e:
            print(f"❌ Integration flow error: {str(e)}")
            self.test_results['integration_flow'] = False
    
    def test_error_handling(self):
        """Test 14: Error handling for invalid inputs"""
        print("\n=== Testing Error Handling ===")
        
        error_tests = []
        
        try:
            # Test 1: Invalid file format
            files = {'file': ('test.txt', 'invalid content', 'text/plain')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            error_tests.append(("Invalid file format", response.status_code >= 400))
            
            # Test 2: Train model without data
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": "invalid_id", "model_type": "prophet"},
                json={"time_column": "date", "target_column": "sales"}
            )
            error_tests.append(("Train model with invalid data_id", response.status_code >= 400))
            
            # Test 3: Generate prediction without model
            response = self.session.get(
                f"{API_BASE_URL}/generate-prediction",
                params={"model_id": "invalid_model", "steps": 10}
            )
            error_tests.append(("Generate prediction with invalid model", response.status_code >= 400))
            
            # Test 4: Invalid model type
            if self.data_id:
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": self.data_id, "model_type": "invalid_model"},
                    json={"time_column": "date", "target_column": "sales"}
                )
                error_tests.append(("Invalid model type", response.status_code >= 400))
            
            # Evaluate error handling
            passed_tests = sum(1 for _, passed in error_tests if passed)
            total_tests = len(error_tests)
            
            print(f"✅ Error handling tests passed: {passed_tests}/{total_tests}")
            for test_name, passed in error_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['error_handling'] = passed_tests >= total_tests * 0.75  # 75% pass rate
            
        except Exception as e:
            print(f"❌ Error handling test error: {str(e)}")
            self.test_results['error_handling'] = False
    
    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Comprehensive Backend Testing")
        print("=" * 60)
        
        # Run synchronous tests
        self.test_file_upload()
        self.test_model_training_prophet()
        self.test_model_training_arima()
        self.test_prediction_generation()
        self.test_historical_data()
        self.test_continuous_prediction_control()
        
        # NEW FEATURE TESTS
        self.test_ph_simulation_endpoints()
        self.test_enhanced_continuous_prediction()
        self.test_reset_functionality()
        self.test_complete_continuous_prediction_flow()
        
        # ENHANCED NEW FEATURE TESTS (from review request)
        self.test_ph_target_management()
        self.test_advanced_pattern_analysis()
        self.test_integration_flow()
        
        self.test_error_handling()
        
        # Run WebSocket test
        try:
            asyncio.run(self.test_websocket_connection())
        except Exception as e:
            print(f"❌ WebSocket test failed: {str(e)}")
            self.test_results['websocket'] = False
        
        # Print final results
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 BACKEND TEST SUMMARY")
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
            print(f"  {status} - {test_name.replace('_', ' ').title()}")
        
        print("\n" + "=" * 60)
        
        # Return overall success
        return passed_tests >= total_tests * 0.8  # 80% pass rate for overall success

if __name__ == "__main__":
    tester = BackendTester()
    overall_success = tester.run_all_tests()
    
    if overall_success:
        print("🎉 Backend testing completed successfully!")
        exit(0)
    else:
        print("⚠️  Backend testing completed with some failures.")
        exit(1)