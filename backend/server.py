from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import io
import json
import asyncio
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.utils
import aiofiles
import random
import math
from datetime import datetime, timedelta

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Define Models
class DataAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    columns: List[str]
    time_columns: List[str]
    numeric_columns: List[str]
    data_shape: tuple
    data_preview: Dict[str, Any]
    suggested_parameters: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelTraining(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data_id: str
    model_type: str
    parameters: Dict[str, Any]
    training_status: str = "pending"
    model_performance: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PredictionData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    prediction_values: List[float]
    timestamps: List[str]
    confidence_intervals: Optional[List[Dict[str, float]]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Global variables to store current model and data
current_model = None
current_data = None
current_scaler = None
prediction_task = None
continuous_predictions = []
ph_simulation_data = []
target_ph = 7.6  # Global target pH value

# Utility functions
def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze uploaded data and suggest parameters"""
    analysis = {
        'columns': df.columns.tolist(),
        'time_columns': [],
        'numeric_columns': [],
        'data_shape': df.shape,
        'data_preview': {}
    }
    
    # Identify time columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(100))
                analysis['time_columns'].append(col)
            except:
                pass
        elif 'date' in col.lower() or 'time' in col.lower():
            analysis['time_columns'].append(col)
    
    # Identify numeric columns
    analysis['numeric_columns'] = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create data preview
    analysis['data_preview'] = {
        'head': df.head(10).to_dict('records'),
        'describe': df.describe().to_dict() if len(analysis['numeric_columns']) > 0 else {},
        'missing_values': df.isnull().sum().to_dict()
    }
    
    # Suggest parameters
    analysis['suggested_parameters'] = {
        'time_column': analysis['time_columns'][0] if analysis['time_columns'] else None,
        'target_column': analysis['numeric_columns'][0] if analysis['numeric_columns'] else None,
        'model_type': 'prophet',
        'prediction_horizon': 30,
        'train_test_split': 0.8,
        'seasonality_mode': 'additive'
    }
    
    return analysis

def prepare_data_for_model(df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
    """Prepare data for time series modeling"""
    # Create a copy and sort by time
    data = df[[time_col, target_col]].copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col)
    
    # Remove duplicates and handle missing values
    data = data.drop_duplicates(subset=[time_col])
    data = data.dropna()
    
    return data

def train_prophet_model(data: pd.DataFrame, time_col: str, target_col: str, params: Dict[str, Any]):
    """Train Prophet model"""
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_data = data.rename(columns={time_col: 'ds', target_col: 'y'})
    
    # Create and train model
    model = Prophet(
        seasonality_mode=params.get('seasonality_mode', 'additive'),
        yearly_seasonality=params.get('yearly_seasonality', True),
        weekly_seasonality=params.get('weekly_seasonality', True),
        daily_seasonality=params.get('daily_seasonality', False)
    )
    
    model.fit(prophet_data)
    return model

def train_arima_model(data: pd.DataFrame, time_col: str, target_col: str, params: Dict[str, Any]):
    """Train ARIMA model"""
    # Prepare data
    ts_data = data.set_index(time_col)[target_col]
    
    # Create and train ARIMA model
    order = params.get('order', (1, 1, 1))
    model = ARIMA(ts_data, order=order)
    fitted_model = model.fit()
    
    return fitted_model

def generate_ph_simulation_data(duration_hours: int = 24) -> List[Dict]:
    """Generate realistic pH simulation data for testing"""
    data = []
    start_time = datetime.now() - timedelta(hours=duration_hours)
    
    # Base pH level around 7.0 with realistic variations
    base_ph = 7.0
    current_ph = base_ph
    
    for i in range(duration_hours * 60):  # Generate data every minute
        # Add realistic pH fluctuations
        # Small random walk + periodic variations
        noise = random.gauss(0, 0.02)  # Small random noise
        periodic = 0.1 * math.sin(i * 2 * math.pi / (24 * 60))  # Daily cycle
        trend = 0.05 * math.sin(i * 2 * math.pi / (4 * 60))  # 4-hour cycle
        
        current_ph += noise + periodic/100 + trend/100
        
        # Keep pH in realistic range (6.0-8.0)
        current_ph = max(6.0, min(8.0, current_ph))
        
        timestamp = start_time + timedelta(minutes=i)
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'ph_value': round(current_ph, 2),
            'confidence': random.uniform(85, 98)  # Simulation confidence
        })
    
    return data

def simulate_real_time_ph():
    """Generate a single real-time pH reading"""
    global ph_simulation_data
    
    if not ph_simulation_data:
        ph_simulation_data = generate_ph_simulation_data(24)
    
    # Get current pH based on simulation
    current_index = len(ph_simulation_data) - 1
    if current_index < len(ph_simulation_data):
        base_reading = ph_simulation_data[current_index]
        
        # Add small real-time variation
        ph_variation = random.gauss(0, 0.01)
        current_ph = base_reading['ph_value'] + ph_variation
        current_ph = max(6.0, min(8.0, current_ph))
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ph_value': round(current_ph, 2),
            'confidence': random.uniform(88, 97)
        }
    
    # Fallback to random realistic pH
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ph_value': round(random.uniform(6.8, 7.2), 2),
        'confidence': random.uniform(85, 95)
    }

# API Routes
@api_router.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload and analyze data file"""
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Analyze data
        analysis = analyze_data(df)
        
        # Create analysis record
        data_analysis = DataAnalysis(
            filename=file.filename,
            columns=analysis['columns'],
            time_columns=analysis['time_columns'],
            numeric_columns=analysis['numeric_columns'],
            data_shape=analysis['data_shape'],
            data_preview=analysis['data_preview'],
            suggested_parameters=analysis['suggested_parameters']
        )
        
        # Store in database
        await db.data_analyses.insert_one(data_analysis.dict())
        
        # Store dataframe globally for model training
        global current_data
        current_data = df
        
        return {
            "status": "success",
            "data_id": data_analysis.id,
            "analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/train-model")
async def train_model(data_id: str, model_type: str, parameters: Dict[str, Any]):
    """Train predictive model"""
    try:
        global current_model, current_data, current_scaler
        
        print(f"Training model with data_id: {data_id}, model_type: {model_type}")
        print(f"Parameters: {parameters}")
        
        if current_data is None:
            print("Error: No data uploaded - current_data is None")
            raise HTTPException(status_code=400, detail="No data uploaded. Please upload data first.")
        
        print(f"Current data shape: {current_data.shape}")
        print(f"Current data columns: {current_data.columns.tolist()}")
        
        # Get parameters
        time_col = parameters.get('time_column')
        target_col = parameters.get('target_column')
        
        print(f"Time column: {time_col}, Target column: {target_col}")
        
        if not time_col or not target_col:
            raise HTTPException(status_code=400, detail="Time column and target column are required")
        
        # Check if columns exist in the data
        if time_col not in current_data.columns:
            raise HTTPException(status_code=400, detail=f"Time column '{time_col}' not found in data. Available columns: {current_data.columns.tolist()}")
        
        if target_col not in current_data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found in data. Available columns: {current_data.columns.tolist()}")
        
        # Prepare data
        print("Preparing data for model training...")
        prepared_data = prepare_data_for_model(current_data, time_col, target_col)
        print(f"Prepared data shape: {prepared_data.shape}")
        
        # Train model based on type
        if model_type == 'prophet':
            print("Training Prophet model...")
            try:
                model = train_prophet_model(prepared_data, time_col, target_col, parameters)
                print("Prophet model trained successfully")
            except Exception as prophet_error:
                print(f"Prophet training error: {prophet_error}")
                raise HTTPException(status_code=500, detail=f"Prophet model training failed: {str(prophet_error)}")
                
        elif model_type == 'arima':
            print("Training ARIMA model...")
            try:
                model = train_arima_model(prepared_data, time_col, target_col, parameters)
                print("ARIMA model trained successfully")
            except Exception as arima_error:
                print(f"ARIMA training error: {arima_error}")
                raise HTTPException(status_code=500, detail=f"ARIMA model training failed: {str(arima_error)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type. Use 'prophet' or 'arima'")
        
        # Store model globally
        current_model = {
            'model': model,
            'model_type': model_type,
            'time_col': time_col,
            'target_col': target_col,
            'parameters': parameters,
            'data': prepared_data
        }
        
        # Create training record
        training_record = ModelTraining(
            data_id=data_id,
            model_type=model_type,
            parameters=parameters,
            training_status="completed",
            model_performance={"status": "trained"}
        )
        
        await db.model_trainings.insert_one(training_record.dict())
        
        print(f"Model training completed successfully. Model ID: {training_record.id}")
        
        return {
            "status": "success",
            "model_id": training_record.id,
            "message": f"{model_type.upper()} model trained successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"General training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@api_router.get("/generate-prediction")
async def generate_prediction(model_id: str, steps: int = 30, offset: int = 0):
    """Generate predictions using trained model with optional offset for continuous prediction"""
    try:
        global current_model, continuous_predictions
        
        if current_model is None:
            raise HTTPException(status_code=400, detail="No model trained")
        
        model = current_model['model']
        model_type = current_model['model_type']
        data = current_model['data']
        
        if model_type == 'prophet':
            # Create future dataframe with offset
            future = model.make_future_dataframe(periods=steps + offset)
            forecast = model.predict(future)
            
            # Extract predictions (skip offset, take next 'steps' predictions)
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps + offset).head(steps)
            
            result = {
                'timestamps': predictions['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'predictions': predictions['yhat'].tolist(),
                'confidence_intervals': [
                    {'lower': row['yhat_lower'], 'upper': row['yhat_upper']} 
                    for _, row in predictions.iterrows()
                ]
            }
            
        elif model_type == 'arima':
            # Generate ARIMA predictions with offset
            forecast = model.forecast(steps=steps + offset)
            
            # Create timestamps - handle the case where data index might not be datetime
            time_col = current_model['time_col']
            original_data = current_model['data']
            
            # Get the last timestamp from the original data
            if time_col in original_data.columns:
                last_timestamp = pd.to_datetime(original_data[time_col].iloc[-1])
                # Try to infer frequency from the time series
                time_series = pd.to_datetime(original_data[time_col])
                freq = pd.infer_freq(time_series)
            else:
                last_timestamp = pd.to_datetime(original_data.index[-1])
                freq = pd.infer_freq(pd.to_datetime(original_data.index))
            
            # Default to daily frequency if inference fails
            if freq is None:
                freq = 'D'
            
            # Create future timestamps with offset
            try:
                future_timestamps = pd.date_range(
                    start=last_timestamp + pd.Timedelta(days=1 + offset), 
                    periods=steps, 
                    freq=freq
                )
            except:
                # Fallback to daily frequency
                future_timestamps = pd.date_range(
                    start=last_timestamp + pd.Timedelta(days=1 + offset), 
                    periods=steps, 
                    freq='D'
                )
            
            # Take the forecasted values (skip offset, take next 'steps' values)
            prediction_values = forecast.tolist()[offset:offset + steps] if offset < len(forecast) else forecast.tolist()[-steps:]
            
            result = {
                'timestamps': future_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'predictions': prediction_values,
                'confidence_intervals': None
            }
        
        # Store prediction for continuous use
        continuous_predictions.append(result)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/generate-continuous-prediction")
async def generate_continuous_prediction(model_id: str, steps: int = 30, time_window: int = 100):
    """Generate continuous predictions that extrapolate forward"""
    try:
        global current_model, continuous_predictions
        
        if current_model is None:
            raise HTTPException(status_code=400, detail="No model trained")
        
        model = current_model['model']
        model_type = current_model['model_type']
        data = current_model['data']
        
        # Calculate how many predictions to generate based on time window
        prediction_offset = len(continuous_predictions) * 5  # Each call advances by 5 steps
        
        if model_type == 'prophet':
            # Create future dataframe with increasing offset
            future = model.make_future_dataframe(periods=steps + prediction_offset)
            forecast = model.predict(future)
            
            # Extract predictions from the end
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
            
            result = {
                'timestamps': predictions['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'predictions': predictions['yhat'].tolist(),
                'confidence_intervals': [
                    {'lower': row['yhat_lower'], 'upper': row['yhat_upper']} 
                    for _, row in predictions.iterrows()
                ]
            }
            
        elif model_type == 'arima':
            # Generate ARIMA predictions with increasing offset
            forecast = model.forecast(steps=steps + prediction_offset)
            
            # Create timestamps
            time_col = current_model['time_col']
            original_data = current_model['data']
            
            # Get the last timestamp from the original data
            if time_col in original_data.columns:
                last_timestamp = pd.to_datetime(original_data[time_col].iloc[-1])
                # Try to infer frequency from the time series
                time_series = pd.to_datetime(original_data[time_col])
                freq = pd.infer_freq(time_series)
            else:
                last_timestamp = pd.to_datetime(original_data.index[-1])
                freq = pd.infer_freq(pd.to_datetime(original_data.index))
            
            # Default to daily frequency if inference fails
            if freq is None:
                freq = 'D'
            
            # Create future timestamps starting from last prediction
            try:
                future_timestamps = pd.date_range(
                    start=last_timestamp + pd.Timedelta(days=1 + prediction_offset), 
                    periods=steps, 
                    freq=freq
                )
            except:
                # Fallback to daily frequency
                future_timestamps = pd.date_range(
                    start=last_timestamp + pd.Timedelta(days=1 + prediction_offset), 
                    periods=steps, 
                    freq='D'
                )
            
            # Take the forecasted values from the end
            prediction_values = forecast.tolist()[-steps:]
            
            result = {
                'timestamps': future_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'predictions': prediction_values,
                'confidence_intervals': None
            }
        
        # Store prediction for continuous use - THIS WAS MISSING!
        continuous_predictions.append(result)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/ph-simulation")
async def get_ph_simulation():
    """Get simulated pH readings"""
    try:
        ph_reading = simulate_real_time_ph()
        return ph_reading
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/ph-simulation-history")
async def get_ph_simulation_history(hours: int = 24):
    """Get historical pH simulation data"""
    try:
        global ph_simulation_data
        if not ph_simulation_data:
            ph_simulation_data = generate_ph_simulation_data(hours)
        
        return {
            'data': ph_simulation_data,
            'current_ph': ph_simulation_data[-1]['ph_value'] if ph_simulation_data else 7.0,
            'target_ph': target_ph,  # Use global target pH
            'status': 'Connected'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/set-ph-target")
async def set_ph_target(request: dict):
    """Set target pH for monitoring"""
    try:
        global target_ph
        target_ph = float(request.get('target_ph', 7.6))
        
        # Validate pH range
        if target_ph < 0 or target_ph > 14:
            raise HTTPException(status_code=400, detail="pH must be between 0 and 14")
        
        return {
            'status': 'success',
            'target_ph': target_ph,
            'message': f'Target pH set to {target_ph}'
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pH value")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/historical-data")
async def get_historical_data():
    """Get historical data for plotting"""
    try:
        global current_model
        
        if current_model is None:
            raise HTTPException(status_code=400, detail="No model trained")
        
        data = current_model['data']
        time_col = current_model['time_col']
        target_col = current_model['target_col']
        
        result = {
            'timestamps': data[time_col].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'values': data[target_col].tolist()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for real-time predictions
@app.websocket("/ws/predictions")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process real-time prediction request
            await manager.send_personal_message(f"Received: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Continuous prediction task
async def continuous_prediction_task():
    """Background task for continuous predictions"""
    while True:
        if current_model is not None:
            try:
                # Generate continuous prediction that extrapolates forward
                prediction = await generate_continuous_prediction("current", 30, 100)
                
                # Also get real-time pH simulation
                ph_reading = simulate_real_time_ph()
                
                # Broadcast to all connected clients
                await manager.broadcast(json.dumps({
                    'type': 'prediction_update',
                    'data': prediction,
                    'ph_reading': ph_reading
                }))
                
            except Exception as e:
                print(f"Error in continuous prediction: {e}")
        
        await asyncio.sleep(1)  # Update every 1 second for smoother experience

@api_router.post("/start-continuous-prediction")
async def start_continuous_prediction():
    """Start continuous prediction updates"""
    global prediction_task, continuous_predictions
    
    # Reset continuous predictions
    continuous_predictions = []
    
    if prediction_task is None:
        prediction_task = asyncio.create_task(continuous_prediction_task())
    
    return {"status": "started", "message": "Continuous prediction started"}

@api_router.post("/stop-continuous-prediction")
async def stop_continuous_prediction():
    """Stop continuous prediction updates"""
    global prediction_task, continuous_predictions
    
    if prediction_task is not None:
        prediction_task.cancel()
        prediction_task = None
    
    # Reset continuous predictions
    continuous_predictions = []
    
    return {"status": "stopped", "message": "Continuous prediction stopped"}

@api_router.post("/reset-continuous-prediction")
async def reset_continuous_prediction():
    """Reset continuous prediction state"""
    global continuous_predictions
    continuous_predictions = []
    return {"status": "reset", "message": "Continuous prediction reset"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()