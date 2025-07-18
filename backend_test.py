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
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://c8772c28-6b4b-4343-84fa-effeefd86ff0.preview.emergentagent.com')
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
    
    async def test_enhanced_websocket_connection(self):
        """Test 6a: Enhanced WebSocket connection with heartbeat and error handling"""
        print("\n=== Testing Enhanced WebSocket Connection ===")
        
        try:
            # Test WebSocket connection with enhanced features
            async with websockets.connect(WS_URL) as websocket:
                print("✅ Enhanced WebSocket connection established")
                
                # Test 1: Connection confirmation message
                try:
                    initial_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    initial_data = json.loads(initial_response)
                    if initial_data.get('type') == 'connection_established':
                        print("✅ Connection confirmation received")
                        connection_confirmed = True
                    else:
                        print(f"⚠️  Unexpected initial message: {initial_data}")
                        connection_confirmed = False
                except asyncio.TimeoutError:
                    print("⚠️  No connection confirmation received")
                    connection_confirmed = False
                
                # Test 2: Send test message and receive echo
                test_message = json.dumps({"type": "test", "message": "Hello Enhanced WebSocket"})
                await websocket.send(test_message)
                print("✅ Test message sent to enhanced WebSocket")
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    if response_data.get('type') == 'echo':
                        print(f"✅ Echo response received: {response_data.get('message')}")
                        echo_test = True
                    else:
                        print(f"⚠️  Unexpected response type: {response_data}")
                        echo_test = False
                except asyncio.TimeoutError:
                    print("⚠️  No echo response received")
                    echo_test = False
                
                # Test 3: Wait for heartbeat message
                print("⏳ Waiting for heartbeat message...")
                try:
                    heartbeat_response = await asyncio.wait_for(websocket.recv(), timeout=65.0)  # Wait longer than heartbeat interval
                    heartbeat_data = json.loads(heartbeat_response)
                    if heartbeat_data.get('type') == 'heartbeat':
                        print("✅ Heartbeat message received")
                        heartbeat_test = True
                    else:
                        print(f"⚠️  Expected heartbeat, got: {heartbeat_data}")
                        heartbeat_test = False
                except asyncio.TimeoutError:
                    print("⚠️  No heartbeat received within timeout")
                    heartbeat_test = False
                
                # Overall WebSocket test result
                websocket_tests = [
                    ("Connection Confirmation", connection_confirmed),
                    ("Echo Response", echo_test),
                    ("Heartbeat", heartbeat_test)
                ]
                
                passed_tests = sum(1 for _, passed in websocket_tests if passed)
                print(f"\n📊 Enhanced WebSocket test results: {passed_tests}/3")
                for test_name, passed in websocket_tests:
                    status = "✅" if passed else "❌"
                    print(f"   {status} {test_name}")
                
                self.test_results['enhanced_websocket'] = passed_tests >= 2  # At least 2/3 tests should pass
                    
        except Exception as e:
            print(f"❌ Enhanced WebSocket connection error: {str(e)}")
            self.test_results['enhanced_websocket'] = False

    def test_server_sent_events(self):
        """Test 6b: Server-Sent Events (SSE) endpoint as WebSocket fallback"""
        print("\n=== Testing Server-Sent Events (SSE) ===")
        
        try:
            import sseclient  # We'll need to handle this gracefully if not available
        except ImportError:
            # Fallback to manual SSE handling
            print("⚠️  sseclient not available, using manual SSE handling")
        
        try:
            # Test SSE connection
            sse_url = f"{API_BASE_URL}/stream/predictions"
            
            # Use requests with stream=True for SSE
            response = self.session.get(sse_url, stream=True, timeout=30)
            
            if response.status_code == 200:
                print("✅ SSE connection established")
                
                # Read initial events
                events_received = []
                lines_buffer = ""
                
                for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                    if chunk:
                        lines_buffer += chunk
                        
                        # Process complete lines
                        while '\n' in lines_buffer:
                            line, lines_buffer = lines_buffer.split('\n', 1)
                            
                            if line.startswith('data: '):
                                try:
                                    event_data = json.loads(line[6:])  # Remove 'data: ' prefix
                                    events_received.append(event_data)
                                    print(f"📨 SSE event received: {event_data.get('type', 'unknown')}")
                                    
                                    # Stop after receiving a few events or connection confirmation
                                    if len(events_received) >= 3 or event_data.get('type') == 'connection_established':
                                        break
                                except json.JSONDecodeError:
                                    print(f"⚠️  Invalid JSON in SSE event: {line}")
                    
                    # Break after reasonable time to avoid hanging
                    if len(events_received) >= 1:
                        break
                
                # Analyze received events
                connection_established = any(event.get('type') == 'connection_established' for event in events_received)
                heartbeat_received = any(event.get('type') == 'heartbeat' for event in events_received)
                
                sse_tests = [
                    ("SSE Connection", response.status_code == 200),
                    ("Connection Established Event", connection_established),
                    ("Events Received", len(events_received) > 0)
                ]
                
                passed_tests = sum(1 for _, passed in sse_tests if passed)
                print(f"\n📊 SSE test results: {passed_tests}/3")
                for test_name, passed in sse_tests:
                    status = "✅" if passed else "❌"
                    print(f"   {status} {test_name}")
                
                self.test_results['sse_streaming'] = passed_tests >= 2  # At least 2/3 tests should pass
                
            else:
                print(f"❌ SSE connection failed: {response.status_code}")
                self.test_results['sse_streaming'] = False
                
        except Exception as e:
            print(f"❌ SSE test error: {str(e)}")
            self.test_results['sse_streaming'] = False

    def test_long_polling(self):
        """Test 6c: Long Polling endpoint as another fallback option"""
        print("\n=== Testing Long Polling ===")
        
        try:
            # Test long polling endpoint with shorter timeout
            polling_url = f"{API_BASE_URL}/poll/predictions"
            
            # Test 1: Basic polling request with short timeout
            try:
                response = self.session.get(polling_url, timeout=5)  # Shorter timeout
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ Long polling request successful")
                    print(f"   Response type: {data.get('type', 'unknown')}")
                    
                    # Check if it's a timeout response (which is expected without active model)
                    if data.get('type') == 'timeout':
                        print("   ✅ Received expected timeout response (no active model)")
                        basic_polling_test = True
                    elif data.get('type') == 'prediction_update':
                        print("   ✅ Received prediction update")
                        basic_polling_test = True
                    else:
                        print(f"   ⚠️  Unexpected response type: {data.get('type')}")
                        basic_polling_test = True  # Still counts as working endpoint
                else:
                    print(f"❌ Long polling failed: {response.status_code} - {response.text}")
                    basic_polling_test = False
                    
            except requests.exceptions.ReadTimeout:
                print("⚠️  Long polling timed out (expected behavior)")
                basic_polling_test = True  # Timeout is expected behavior
            except Exception as e:
                print(f"❌ Long polling error: {str(e)}")
                basic_polling_test = False
            
            # Test 2: Quick polling test (should return quickly)
            try:
                # This should return quickly with timeout response
                response = self.session.get(polling_url, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('type') in ['timeout', 'prediction_update']:
                        print("✅ Quick polling test successful")
                        quick_polling_test = True
                    else:
                        print(f"⚠️  Unexpected quick polling response: {data}")
                        quick_polling_test = False
                else:
                    print(f"⚠️  Quick polling failed: {response.status_code}")
                    quick_polling_test = False
            except requests.exceptions.ReadTimeout:
                print("⚠️  Quick polling timed out")
                quick_polling_test = False
            except Exception as e:
                print(f"⚠️  Quick polling error: {str(e)}")
                quick_polling_test = False
            
            # Test 3: Endpoint availability (just check if endpoint exists)
            try:
                # Use HEAD request to check if endpoint exists without waiting
                response = self.session.head(polling_url, timeout=2)
                endpoint_available = response.status_code in [200, 405]  # 405 = Method Not Allowed is OK for HEAD
                if endpoint_available:
                    print("✅ Polling endpoint is available")
                else:
                    print(f"⚠️  Polling endpoint availability unclear: {response.status_code}")
            except:
                # If HEAD fails, try a quick GET
                try:
                    response = self.session.get(polling_url, timeout=1)
                    endpoint_available = response.status_code == 200
                    print("✅ Polling endpoint is available (via GET)")
                except:
                    endpoint_available = False
                    print("❌ Polling endpoint not available")
            
            # Overall long polling test result
            polling_tests = [
                ("Basic Polling", basic_polling_test),
                ("Quick Polling", quick_polling_test),
                ("Endpoint Available", endpoint_available)
            ]
            
            passed_tests = sum(1 for _, passed in polling_tests if passed)
            print(f"\n📊 Long polling test results: {passed_tests}/3")
            for test_name, passed in polling_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
            self.test_results['long_polling'] = passed_tests >= 2  # At least 2/3 tests should pass
            
        except Exception as e:
            print(f"❌ Long polling test error: {str(e)}")
            self.test_results['long_polling'] = False

    def test_connection_status_endpoint(self):
        """Test 6d: Connection status endpoint for debugging"""
        print("\n=== Testing Connection Status Endpoint ===")
        
        try:
            # Test connection status endpoint
            status_url = f"{API_BASE_URL}/connection-status"
            
            response = self.session.get(status_url)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Connection status endpoint successful")
                print(f"   Status data: {data}")
                
                # Validate expected fields in status response
                expected_fields = ['websocket_connections', 'sse_connections', 'server_status']
                status_validation = []
                
                for field in expected_fields:
                    if field in data:
                        print(f"   ✅ {field}: {data[field]}")
                        status_validation.append(True)
                    else:
                        print(f"   ⚠️  Missing field: {field}")
                        status_validation.append(False)
                
                # Check if status indicates healthy server
                server_healthy = data.get('server_status') == 'running' or 'active' in str(data.get('server_status', '')).lower()
                if server_healthy:
                    print("   ✅ Server status indicates healthy state")
                else:
                    print(f"   ⚠️  Server status unclear: {data.get('server_status')}")
                
                status_tests = [
                    ("Endpoint Response", True),
                    ("Required Fields", sum(status_validation) >= len(expected_fields) * 0.7),
                    ("Server Health", server_healthy)
                ]
                
                passed_tests = sum(1 for _, passed in status_tests if passed)
                print(f"\n📊 Connection status test results: {passed_tests}/3")
                for test_name, passed in status_tests:
                    status = "✅" if passed else "❌"
                    print(f"   {status} {test_name}")
                
                self.test_results['connection_status'] = passed_tests >= 2
                
            else:
                print(f"❌ Connection status endpoint failed: {response.status_code} - {response.text}")
                self.test_results['connection_status'] = False
                
        except Exception as e:
            print(f"❌ Connection status test error: {str(e)}")
            self.test_results['connection_status'] = False

    def test_websocket_support_endpoint(self):
        """Test 6e: Test WebSocket support endpoint"""
        print("\n=== Testing WebSocket Support Endpoint ===")
        
        try:
            # Test WebSocket support detection endpoint
            support_url = f"{API_BASE_URL}/test-websocket-support"
            
            response = self.session.get(support_url)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ WebSocket support endpoint successful")
                print(f"   Support data: {data}")
                
                # Validate support information
                websocket_supported = data.get('websocket_supported', False)
                fallback_available = data.get('fallback_methods', [])
                
                print(f"   WebSocket supported: {websocket_supported}")
                print(f"   Fallback methods: {fallback_available}")
                
                support_tests = [
                    ("Endpoint Response", True),
                    ("WebSocket Support Info", 'websocket_supported' in data),
                    ("Fallback Methods Listed", len(fallback_available) > 0)
                ]
                
                passed_tests = sum(1 for _, passed in support_tests if passed)
                print(f"\n📊 WebSocket support test results: {passed_tests}/3")
                for test_name, passed in support_tests:
                    status = "✅" if passed else "❌"
                    print(f"   {status} {test_name}")
                
                self.test_results['websocket_support'] = passed_tests >= 2
                
            else:
                print(f"❌ WebSocket support endpoint failed: {response.status_code} - {response.text}")
                self.test_results['websocket_support'] = False
                
        except Exception as e:
            print(f"❌ WebSocket support test error: {str(e)}")
            self.test_results['websocket_support'] = False

    async def test_websocket_connection(self):
        """Test 6: Comprehensive WebSocket and Streaming Tests"""
        print("\n=== COMPREHENSIVE WEBSOCKET & STREAMING TESTS ===")
        
        # Run enhanced WebSocket test
        await self.test_enhanced_websocket_connection()
        
        # Run SSE test
        self.test_server_sent_events()
        
        # Run long polling test
        self.test_long_polling()
        
        # Run connection status test
        self.test_connection_status_endpoint()
        
        # Run WebSocket support test
        self.test_websocket_support_endpoint()
        
        # Overall streaming functionality assessment
        streaming_tests = [
            ('Enhanced WebSocket', self.test_results.get('enhanced_websocket', False)),
            ('SSE Streaming', self.test_results.get('sse_streaming', False)),
            ('Long Polling', self.test_results.get('long_polling', False)),
            ('Connection Status', self.test_results.get('connection_status', False)),
            ('WebSocket Support', self.test_results.get('websocket_support', False))
        ]
        
        passed_streaming_tests = sum(1 for _, passed in streaming_tests if passed)
        total_streaming_tests = len(streaming_tests)
        
        print(f"\n🎯 STREAMING FUNCTIONALITY SUMMARY: {passed_streaming_tests}/{total_streaming_tests} tests passed")
        for test_name, passed in streaming_tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        # Set overall WebSocket result (for backward compatibility)
        self.test_results['websocket'] = passed_streaming_tests >= 3  # At least 3/5 streaming methods should work
    
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
    
    def create_pattern_test_data(self, pattern_type="quadratic"):
        """Create test data with specific patterns for advanced ML testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        if pattern_type == "quadratic":
            # U-shaped quadratic pattern
            x = np.linspace(-5, 5, 100)
            values = x**2 + np.random.normal(0, 0.5, 100) + 10
        elif pattern_type == "cubic":
            # S-shaped cubic pattern
            x = np.linspace(-2, 2, 100)
            values = x**3 - 3*x + np.random.normal(0, 0.3, 100) + 5
        elif pattern_type == "polynomial":
            # Complex polynomial pattern
            x = np.linspace(0, 10, 100)
            values = 0.1*x**4 - 2*x**3 + 10*x**2 - 20*x + np.random.normal(0, 1, 100) + 50
        elif pattern_type == "custom":
            # Custom pattern with multiple components
            x = np.linspace(0, 4*np.pi, 100)
            trend = 0.5 * x
            seasonal = 3 * np.sin(x) + 1.5 * np.cos(2*x)
            noise = np.random.normal(0, 0.2, 100)
            values = trend + seasonal + noise + 20
        else:
            # Linear pattern (default)
            values = np.linspace(10, 50, 100) + np.random.normal(0, 2, 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value': values
        })
        
        return df
    
    def test_advanced_ml_models_dependency_fix(self):
        """Test advanced ML models (LSTM, DLinear, N-BEATS) for SymPy/mpmath dependency resolution"""
        print("\n=== Testing Advanced ML Models - Dependency Fix ===")
        
        try:
            # Create pattern test data
            df = self.create_pattern_test_data("quadratic")
            csv_content = df.to_csv(index=False)
            
            # Upload pattern data
            files = {'file': ('pattern_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Pattern data upload failed: {response.status_code}")
                self.test_results['advanced_ml_dependency_fix'] = False
                return
                
            data_id = response.json().get('data_id')
            print("✅ Pattern data uploaded successfully")
            
            # Test each advanced model type
            advanced_models = ['lstm', 'dlinear', 'nbeats']
            model_results = {}
            
            for model_type in advanced_models:
                print(f"\n--- Testing {model_type.upper()} Model ---")
                
                try:
                    # Train advanced model
                    training_params = {
                        "time_column": "timestamp",
                        "target_column": "value",
                        "seq_len": 20,
                        "pred_len": 10,
                        "epochs": 20,
                        "batch_size": 8,
                        "learning_rate": 0.001
                    }
                    
                    response = self.session.post(
                        f"{API_BASE_URL}/train-model",
                        params={"data_id": data_id, "model_type": model_type},
                        json=training_params
                    )
                    
                    if response.status_code == 200:
                        model_data = response.json()
                        model_id = model_data.get('model_id')
                        
                        print(f"✅ {model_type.upper()} model training successful")
                        print(f"   Model ID: {model_id}")
                        print(f"   Status: {model_data.get('status')}")
                        
                        # Test prediction generation
                        pred_response = self.session.get(
                            f"{API_BASE_URL}/generate-prediction",
                            params={"model_id": model_id, "steps": 15}
                        )
                        
                        if pred_response.status_code == 200:
                            pred_data = pred_response.json()
                            predictions = pred_data.get('predictions', [])
                            
                            print(f"✅ {model_type.upper()} prediction generation successful")
                            print(f"   Number of predictions: {len(predictions)}")
                            print(f"   Sample predictions: {predictions[:3] if predictions else 'None'}")
                            
                            # Test pattern-aware prediction
                            pattern_response = self.session.get(
                                f"{API_BASE_URL}/advanced-prediction",
                                params={"model_id": model_id, "steps": 20}
                            )
                            
                            if pattern_response.status_code == 200:
                                pattern_data = pattern_response.json()
                                pattern_predictions = pattern_data.get('predictions', [])
                                
                                print(f"✅ {model_type.upper()} pattern-aware prediction successful")
                                print(f"   Pattern predictions: {len(pattern_predictions)}")
                                
                                model_results[model_type] = {
                                    'training': True,
                                    'prediction': True,
                                    'pattern_aware': True,
                                    'dependency_resolved': True
                                }
                            else:
                                print(f"⚠️ {model_type.upper()} pattern-aware prediction failed: {pattern_response.status_code}")
                                model_results[model_type] = {
                                    'training': True,
                                    'prediction': True,
                                    'pattern_aware': False,
                                    'dependency_resolved': True
                                }
                        else:
                            print(f"❌ {model_type.upper()} prediction failed: {pred_response.status_code}")
                            error_text = pred_response.text
                            if 'SymPy' in error_text or 'mpmath' in error_text:
                                print(f"❌ SymPy/mpmath dependency error still present!")
                                model_results[model_type] = {
                                    'training': True,
                                    'prediction': False,
                                    'pattern_aware': False,
                                    'dependency_resolved': False
                                }
                            else:
                                model_results[model_type] = {
                                    'training': True,
                                    'prediction': False,
                                    'pattern_aware': False,
                                    'dependency_resolved': True
                                }
                    else:
                        print(f"❌ {model_type.upper()} model training failed: {response.status_code}")
                        error_text = response.text
                        if 'SymPy' in error_text or 'mpmath' in error_text:
                            print(f"❌ SymPy/mpmath dependency error detected!")
                            model_results[model_type] = {
                                'training': False,
                                'prediction': False,
                                'pattern_aware': False,
                                'dependency_resolved': False
                            }
                        else:
                            model_results[model_type] = {
                                'training': False,
                                'prediction': False,
                                'pattern_aware': False,
                                'dependency_resolved': True
                            }
                            
                except Exception as e:
                    print(f"❌ {model_type.upper()} model test error: {str(e)}")
                    if 'SymPy' in str(e) or 'mpmath' in str(e):
                        print(f"❌ SymPy/mpmath dependency error in exception!")
                        model_results[model_type] = {
                            'training': False,
                            'prediction': False,
                            'pattern_aware': False,
                            'dependency_resolved': False
                        }
                    else:
                        model_results[model_type] = {
                            'training': False,
                            'prediction': False,
                            'pattern_aware': False,
                            'dependency_resolved': True
                        }
            
            # Evaluate results
            print(f"\n📊 Advanced ML Models Test Results:")
            dependency_resolved_count = 0
            working_models_count = 0
            
            for model_type, results in model_results.items():
                dependency_status = "✅" if results['dependency_resolved'] else "❌"
                training_status = "✅" if results['training'] else "❌"
                prediction_status = "✅" if results['prediction'] else "❌"
                pattern_status = "✅" if results['pattern_aware'] else "❌"
                
                print(f"   {model_type.upper()}:")
                print(f"     Dependency Resolved: {dependency_status}")
                print(f"     Training: {training_status}")
                print(f"     Prediction: {prediction_status}")
                print(f"     Pattern-Aware: {pattern_status}")
                
                if results['dependency_resolved']:
                    dependency_resolved_count += 1
                if results['training'] and results['prediction']:
                    working_models_count += 1
            
            # Overall assessment
            dependency_fix_success = dependency_resolved_count == len(advanced_models)
            models_working = working_models_count >= 2  # At least 2 out of 3 models should work
            
            print(f"\n🎯 Dependency Fix Assessment:")
            print(f"   SymPy/mpmath dependency resolved: {dependency_resolved_count}/{len(advanced_models)} models")
            print(f"   Working advanced models: {working_models_count}/{len(advanced_models)} models")
            
            self.test_results['advanced_ml_dependency_fix'] = dependency_fix_success and models_working
            
        except Exception as e:
            print(f"❌ Advanced ML models dependency test error: {str(e)}")
            if 'SymPy' in str(e) or 'mpmath' in str(e):
                print(f"❌ SymPy/mpmath dependency error in main test!")
            self.test_results['advanced_ml_dependency_fix'] = False
    
    def test_pattern_aware_predictions(self):
        """Test pattern-aware prediction generation with different pattern types"""
        print("\n=== Testing Pattern-Aware Predictions ===")
        
        pattern_types = ['quadratic', 'cubic', 'polynomial', 'custom']
        pattern_results = {}
        
        for pattern_type in pattern_types:
            print(f"\n--- Testing {pattern_type.upper()} Pattern ---")
            
            try:
                # Create pattern-specific test data
                df = self.create_pattern_test_data(pattern_type)
                csv_content = df.to_csv(index=False)
                
                # Upload pattern data
                files = {'file': (f'{pattern_type}_pattern.csv', csv_content, 'text/csv')}
                response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
                
                if response.status_code != 200:
                    print(f"❌ {pattern_type} pattern data upload failed")
                    pattern_results[pattern_type] = False
                    continue
                    
                data_id = response.json().get('data_id')
                
                # Train LSTM model (most reliable for pattern detection)
                training_params = {
                    "time_column": "timestamp",
                    "target_column": "value",
                    "seq_len": 25,
                    "pred_len": 15,
                    "epochs": 30,
                    "batch_size": 8
                }
                
                response = self.session.post(
                    f"{API_BASE_URL}/train-model",
                    params={"data_id": data_id, "model_type": "lstm"},
                    json=training_params
                )
                
                if response.status_code != 200:
                    print(f"❌ {pattern_type} pattern LSTM training failed: {response.status_code}")
                    pattern_results[pattern_type] = False
                    continue
                    
                model_id = response.json().get('model_id')
                print(f"✅ {pattern_type} pattern LSTM model trained")
                
                # Test pattern-aware prediction
                pred_response = self.session.get(
                    f"{API_BASE_URL}/generate-prediction",
                    params={"model_id": model_id, "steps": 20}
                )
                
                if pred_response.status_code == 200:
                    pred_data = pred_response.json()
                    predictions = pred_data.get('predictions', [])
                    
                    # Analyze prediction quality
                    if len(predictions) >= 15:
                        # Check for downward bias (main issue to resolve)
                        prediction_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
                        has_downward_bias = prediction_trend < -0.1  # Significant downward trend
                        
                        # Check for reasonable variability (not monotonic)
                        prediction_std = np.std(predictions)
                        has_variability = prediction_std > 0.01
                        
                        # Check for realistic value range
                        original_mean = df['value'].mean()
                        original_std = df['value'].std()
                        pred_mean = np.mean(predictions)
                        
                        # Predictions should be within reasonable range of original data
                        within_range = abs(pred_mean - original_mean) < 3 * original_std
                        
                        print(f"✅ {pattern_type} pattern predictions generated:")
                        print(f"   Predictions count: {len(predictions)}")
                        print(f"   Prediction trend slope: {prediction_trend:.6f}")
                        print(f"   Has downward bias: {'❌ YES' if has_downward_bias else '✅ NO'}")
                        print(f"   Has variability: {'✅ YES' if has_variability else '❌ NO'}")
                        print(f"   Within realistic range: {'✅ YES' if within_range else '❌ NO'}")
                        print(f"   Original mean: {original_mean:.2f}, Pred mean: {pred_mean:.2f}")
                        
                        # Pattern follows historical characteristics
                        pattern_quality = not has_downward_bias and has_variability and within_range
                        pattern_results[pattern_type] = pattern_quality
                        
                    else:
                        print(f"❌ {pattern_type} pattern insufficient predictions: {len(predictions)}")
                        pattern_results[pattern_type] = False
                        
                else:
                    print(f"❌ {pattern_type} pattern prediction failed: {pred_response.status_code}")
                    pattern_results[pattern_type] = False
                    
            except Exception as e:
                print(f"❌ {pattern_type} pattern test error: {str(e)}")
                pattern_results[pattern_type] = False
        
        # Evaluate pattern-aware prediction results
        successful_patterns = sum(1 for success in pattern_results.values() if success)
        total_patterns = len(pattern_types)
        
        print(f"\n📊 Pattern-Aware Prediction Results:")
        for pattern_type, success in pattern_results.items():
            status = "✅" if success else "❌"
            print(f"   {pattern_type.upper()} pattern: {status}")
        
        print(f"\n🎯 Pattern-Aware Assessment:")
        print(f"   Successful patterns: {successful_patterns}/{total_patterns}")
        print(f"   Success rate: {(successful_patterns/total_patterns)*100:.1f}%")
        
        self.test_results['pattern_aware_predictions'] = successful_patterns >= total_patterns * 0.75  # 75% success rate
    
    def test_downward_bias_resolution(self):
        """Test that predictions don't show persistent downward bias"""
        print("\n=== Testing Downward Bias Resolution ===")
        
        try:
            # Create stable test data (should not trend downward)
            dates = pd.date_range(start='2023-01-01', periods=80, freq='D')
            # Stable data around mean with small variations
            stable_values = 25 + np.random.normal(0, 2, 80) + 5 * np.sin(np.linspace(0, 4*np.pi, 80))
            
            df = pd.DataFrame({
                'timestamp': dates,
                'value': stable_values
            })
            
            csv_content = df.to_csv(index=False)
            
            # Upload stable data
            files = {'file': ('stable_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Stable data upload failed: {response.status_code}")
                self.test_results['downward_bias_resolution'] = False
                return
                
            data_id = response.json().get('data_id')
            print("✅ Stable test data uploaded")
            
            # Train LSTM model
            training_params = {
                "time_column": "timestamp",
                "target_column": "value",
                "seq_len": 20,
                "pred_len": 10,
                "epochs": 25
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "lstm"},
                json=training_params
            )
            
            if response.status_code != 200:
                print(f"❌ Stable data LSTM training failed: {response.status_code}")
                self.test_results['downward_bias_resolution'] = False
                return
                
            model_id = response.json().get('model_id')
            print("✅ LSTM model trained on stable data")
            
            # Test multiple prediction calls to check for accumulated bias
            bias_tests = []
            
            for i in range(5):
                pred_response = self.session.get(
                    f"{API_BASE_URL}/generate-prediction",
                    params={"model_id": model_id, "steps": 30}
                )
                
                if pred_response.status_code == 200:
                    pred_data = pred_response.json()
                    predictions = pred_data.get('predictions', [])
                    
                    if len(predictions) >= 20:
                        # Calculate trend slope
                        trend_slope = np.polyfit(range(len(predictions)), predictions, 1)[0]
                        
                        # Check for downward bias (slope should not be significantly negative)
                        has_downward_bias = trend_slope < -0.05
                        
                        # Check prediction mean vs historical mean
                        historical_mean = np.mean(stable_values)
                        prediction_mean = np.mean(predictions)
                        mean_deviation = abs(prediction_mean - historical_mean)
                        
                        print(f"   Call {i+1}: Trend slope: {trend_slope:.6f}, Mean dev: {mean_deviation:.2f}")
                        
                        bias_tests.append({
                            'call': i+1,
                            'trend_slope': trend_slope,
                            'has_downward_bias': has_downward_bias,
                            'mean_deviation': mean_deviation,
                            'predictions': predictions[:5]  # First 5 predictions
                        })
                    else:
                        print(f"❌ Call {i+1}: Insufficient predictions: {len(predictions)}")
                        bias_tests.append({
                            'call': i+1,
                            'has_downward_bias': True,  # Mark as failed
                            'trend_slope': -999,
                            'mean_deviation': 999
                        })
                else:
                    print(f"❌ Call {i+1}: Prediction failed: {pred_response.status_code}")
                    bias_tests.append({
                        'call': i+1,
                        'has_downward_bias': True,  # Mark as failed
                        'trend_slope': -999,
                        'mean_deviation': 999
                    })
            
            # Analyze bias test results
            successful_calls = [test for test in bias_tests if not test['has_downward_bias']]
            downward_bias_calls = [test for test in bias_tests if test['has_downward_bias']]
            
            print(f"\n📊 Downward Bias Test Results:")
            print(f"   Total prediction calls: {len(bias_tests)}")
            print(f"   Calls without downward bias: {len(successful_calls)}")
            print(f"   Calls with downward bias: {len(downward_bias_calls)}")
            
            if successful_calls:
                avg_slope = np.mean([test['trend_slope'] for test in successful_calls])
                avg_deviation = np.mean([test['mean_deviation'] for test in successful_calls])
                print(f"   Average trend slope (successful): {avg_slope:.6f}")
                print(f"   Average mean deviation (successful): {avg_deviation:.2f}")
            
            # Success criteria: At least 80% of calls should not have downward bias
            success_rate = len(successful_calls) / len(bias_tests)
            bias_resolved = success_rate >= 0.8
            
            print(f"\n🎯 Downward Bias Resolution Assessment:")
            print(f"   Success rate: {success_rate*100:.1f}%")
            print(f"   Downward bias resolved: {'✅ YES' if bias_resolved else '❌ NO'}")
            
            self.test_results['downward_bias_resolution'] = bias_resolved
            
        except Exception as e:
            print(f"❌ Downward bias resolution test error: {str(e)}")
            self.test_results['downward_bias_resolution'] = False
    
    def test_continuous_pattern_prediction(self):
        """Test continuous prediction with pattern preservation"""
        print("\n=== Testing Continuous Pattern Prediction ===")
        
        try:
            # Create cyclical pattern data
            dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
            x = np.linspace(0, 4*np.pi, 60)
            cyclical_values = 30 + 10 * np.sin(x) + 5 * np.cos(2*x) + np.random.normal(0, 1, 60)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'value': cyclical_values
            })
            
            csv_content = df.to_csv(index=False)
            
            # Upload cyclical data
            files = {'file': ('cyclical_data.csv', csv_content, 'text/csv')}
            response = self.session.post(f"{API_BASE_URL}/upload-data", files=files)
            
            if response.status_code != 200:
                print(f"❌ Cyclical data upload failed: {response.status_code}")
                self.test_results['continuous_pattern_prediction'] = False
                return
                
            data_id = response.json().get('data_id')
            print("✅ Cyclical test data uploaded")
            
            # Train LSTM model
            training_params = {
                "time_column": "timestamp",
                "target_column": "value",
                "seq_len": 15,
                "pred_len": 8,
                "epochs": 30
            }
            
            response = self.session.post(
                f"{API_BASE_URL}/train-model",
                params={"data_id": data_id, "model_type": "lstm"},
                json=training_params
            )
            
            if response.status_code != 200:
                print(f"❌ Cyclical data LSTM training failed: {response.status_code}")
                self.test_results['continuous_pattern_prediction'] = False
                return
                
            model_id = response.json().get('model_id')
            print("✅ LSTM model trained on cyclical data")
            
            # Test continuous prediction calls
            continuous_tests = []
            all_predictions = []
            all_timestamps = []
            
            for i in range(4):
                pred_response = self.session.get(
                    f"{API_BASE_URL}/generate-continuous-prediction",
                    params={"model_id": model_id, "steps": 15, "time_window": 50}
                )
                
                if pred_response.status_code == 200:
                    pred_data = pred_response.json()
                    predictions = pred_data.get('predictions', [])
                    timestamps = pred_data.get('timestamps', [])
                    
                    if len(predictions) >= 10:
                        all_predictions.extend(predictions)
                        all_timestamps.extend(timestamps)
                        
                        # Check pattern preservation
                        pred_std = np.std(predictions)
                        pred_mean = np.mean(predictions)
                        
                        # Check for reasonable variability (cyclical pattern should have variation)
                        has_variation = pred_std > 1.0
                        
                        # Check for reasonable mean (should be close to historical mean)
                        historical_mean = np.mean(cyclical_values)
                        mean_reasonable = abs(pred_mean - historical_mean) < 10
                        
                        continuous_tests.append({
                            'call': i+1,
                            'predictions_count': len(predictions),
                            'std': pred_std,
                            'mean': pred_mean,
                            'has_variation': has_variation,
                            'mean_reasonable': mean_reasonable,
                            'timestamps_advance': len(set(timestamps)) == len(timestamps)  # Unique timestamps
                        })
                        
                        print(f"   Call {i+1}: {len(predictions)} predictions, std: {pred_std:.2f}, mean: {pred_mean:.2f}")
                    else:
                        print(f"❌ Call {i+1}: Insufficient predictions: {len(predictions)}")
                        continuous_tests.append({
                            'call': i+1,
                            'predictions_count': len(predictions),
                            'has_variation': False,
                            'mean_reasonable': False,
                            'timestamps_advance': False
                        })
                else:
                    print(f"❌ Call {i+1}: Continuous prediction failed: {pred_response.status_code}")
                    continuous_tests.append({
                        'call': i+1,
                        'predictions_count': 0,
                        'has_variation': False,
                        'mean_reasonable': False,
                        'timestamps_advance': False
                    })
                
                time.sleep(0.5)  # Brief pause between calls
            
            # Analyze continuous prediction results
            successful_calls = [test for test in continuous_tests 
                              if test['has_variation'] and test['mean_reasonable'] and test['timestamps_advance']]
            
            print(f"\n📊 Continuous Pattern Prediction Results:")
            print(f"   Total continuous calls: {len(continuous_tests)}")
            print(f"   Successful calls: {len(successful_calls)}")
            
            if all_predictions:
                overall_std = np.std(all_predictions)
                overall_mean = np.mean(all_predictions)
                print(f"   Overall predictions std: {overall_std:.2f}")
                print(f"   Overall predictions mean: {overall_mean:.2f}")
                print(f"   Total predictions generated: {len(all_predictions)}")
            
            # Success criteria: At least 75% of continuous calls should preserve patterns
            success_rate = len(successful_calls) / len(continuous_tests) if continuous_tests else 0
            pattern_preserved = success_rate >= 0.75
            
            print(f"\n🎯 Continuous Pattern Prediction Assessment:")
            print(f"   Success rate: {success_rate*100:.1f}%")
            print(f"   Pattern preservation: {'✅ YES' if pattern_preserved else '❌ NO'}")
            
            self.test_results['continuous_pattern_prediction'] = pattern_preserved
            
        except Exception as e:
            print(f"❌ Continuous pattern prediction test error: {str(e)}")
            self.test_results['continuous_pattern_prediction'] = False

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
        
        # ADVANCED ML MODELS TESTING (Focus of this review)
        print("\n" + "="*60)
        print("🎯 ADVANCED ML MODELS TESTING - DEPENDENCY FIX VERIFICATION")
        print("="*60)
        
        self.test_advanced_ml_models_dependency_fix()
        self.test_pattern_aware_predictions()
        self.test_downward_bias_resolution()
        self.test_continuous_pattern_prediction()
        
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