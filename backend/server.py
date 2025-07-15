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
        
        if current_data is None:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        # Get parameters
        time_col = parameters.get('time_column')
        target_col = parameters.get('target_column')
        
        if not time_col or not target_col:
            raise HTTPException(status_code=400, detail="Time column and target column are required")
        
        # Prepare data
        prepared_data = prepare_data_for_model(current_data, time_col, target_col)
        
        # Train model based on type
        if model_type == 'prophet':
            model = train_prophet_model(prepared_data, time_col, target_col, parameters)
        elif model_type == 'arima':
            model = train_arima_model(prepared_data, time_col, target_col, parameters)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
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
        
        return {
            "status": "success",
            "model_id": training_record.id,
            "message": "Model trained successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/generate-prediction")
async def generate_prediction(model_id: str, steps: int = 30):
    """Generate predictions using trained model"""
    try:
        global current_model
        
        if current_model is None:
            raise HTTPException(status_code=400, detail="No model trained")
        
        model = current_model['model']
        model_type = current_model['model_type']
        data = current_model['data']
        
        if model_type == 'prophet':
            # Create future dataframe
            future = model.make_future_dataframe(periods=steps)
            forecast = model.predict(future)
            
            # Extract predictions
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
            # Generate ARIMA predictions
            forecast = model.forecast(steps=steps)
            
            # Create timestamps
            last_timestamp = data.index[-1]
            freq = pd.infer_freq(data.index)
            future_timestamps = pd.date_range(
                start=last_timestamp + pd.Timedelta(freq), 
                periods=steps, 
                freq=freq
            )
            
            result = {
                'timestamps': future_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'predictions': forecast.tolist(),
                'confidence_intervals': None
            }
        
        return result
        
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
                # Generate new prediction
                prediction = await generate_prediction("current", 1)
                
                # Broadcast to all connected clients
                await manager.broadcast(json.dumps({
                    'type': 'prediction_update',
                    'data': prediction
                }))
                
            except Exception as e:
                print(f"Error in continuous prediction: {e}")
        
        await asyncio.sleep(2)  # Update every 2 seconds

@api_router.post("/start-continuous-prediction")
async def start_continuous_prediction():
    """Start continuous prediction updates"""
    global prediction_task
    
    if prediction_task is None:
        prediction_task = asyncio.create_task(continuous_prediction_task())
    
    return {"status": "started", "message": "Continuous prediction started"}

@api_router.post("/stop-continuous-prediction")
async def stop_continuous_prediction():
    """Stop continuous prediction updates"""
    global prediction_task
    
    if prediction_task is not None:
        prediction_task.cancel()
        prediction_task = None
    
    return {"status": "stopped", "message": "Continuous prediction stopped"}

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