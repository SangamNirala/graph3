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

# Import new advanced models
from advanced_models import (
    AdvancedTimeSeriesForecaster, 
    ModelEnsemble, 
    create_advanced_forecasting_models,
    optimize_hyperparameters
)
from data_preprocessing import (
    AdvancedTimeSeriesPreprocessor,
    TimeSeriesValidator,
    preprocess_time_series_data
)
from model_evaluation import (
    TimeSeriesEvaluator,
    evaluate_model_performance
)

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

# Enhanced ML variables
current_advanced_model = None
current_ensemble_model = None
current_preprocessor = None
model_evaluation_results = {}
supported_advanced_models = ['dlinear', 'nbeats', 'lstm', 'lightgbm', 'xgboost', 'ensemble']

def analyze_historical_patterns(data, time_col, target_col):
    """Analyze historical data patterns for advanced extrapolation"""
    try:
        # Convert time column to datetime if not already
        if time_col in data.columns:
            data[time_col] = pd.to_datetime(data[time_col])
            data = data.set_index(time_col)
        
        target_values = data[target_col].values
        
        # Calculate key statistics
        mean_value = np.mean(target_values)
        std_value = np.std(target_values)
        
        # Calculate trend (linear regression slope)
        x = np.arange(len(target_values))
        trend_slope = np.polyfit(x, target_values, 1)[0]
        
        # Calculate moving averages for different windows
        ma_5 = pd.Series(target_values).rolling(window=min(5, len(target_values))).mean()
        ma_10 = pd.Series(target_values).rolling(window=min(10, len(target_values))).mean()
        
        # Calculate velocity (rate of change)
        velocity = np.diff(target_values)
        avg_velocity = np.mean(velocity) if len(velocity) > 0 else 0
        
        # Calculate acceleration (rate of change of velocity)
        acceleration = np.diff(velocity) if len(velocity) > 1 else [0]
        avg_acceleration = np.mean(acceleration) if len(acceleration) > 0 else 0
        
        # Detect seasonality/periodicity
        recent_values = target_values[-min(50, len(target_values)):]  # Last 50 points
        
        patterns = {
            'mean': mean_value,
            'std': std_value,
            'trend_slope': trend_slope,
            'velocity': avg_velocity,
            'acceleration': avg_acceleration,
            'recent_mean': np.mean(recent_values),
            'recent_std': np.std(recent_values),
            'last_value': target_values[-1],
            'last_5_values': target_values[-5:].tolist(),
            'ma_5_last': ma_5.iloc[-1] if not ma_5.empty else mean_value,
            'ma_10_last': ma_10.iloc[-1] if not ma_10.empty else mean_value,
        }
        
        return patterns
        
    except Exception as e:
        print(f"Error analyzing patterns: {e}")
        return None

def generate_advanced_extrapolation(patterns, steps=30, time_step=1):
    """Generate advanced extrapolation based on historical patterns"""
    try:
        if not patterns:
            return []
            
        predictions = []
        current_value = patterns['last_value']
        current_velocity = patterns['velocity']
        current_acceleration = patterns['acceleration']
        
        # Weight factors for different components
        trend_weight = 0.3
        velocity_weight = 0.4
        acceleration_weight = 0.2
        noise_weight = 0.1
        
        for i in range(steps):
            # Trend component
            trend_component = patterns['trend_slope'] * time_step * trend_weight
            
            # Velocity component (momentum)
            velocity_component = current_velocity * velocity_weight
            
            # Acceleration component (curvature)
            acceleration_component = current_acceleration * acceleration_weight
            
            # Add controlled noise for realism
            noise_component = np.random.normal(0, patterns['recent_std'] * 0.1) * noise_weight
            
            # Combine all components
            prediction = current_value + trend_component + velocity_component + acceleration_component + noise_component
            
            # Apply gentle mean reversion to prevent extreme drift
            mean_reversion = (patterns['recent_mean'] - prediction) * 0.05
            prediction += mean_reversion
            
            predictions.append(prediction)
            
            # Update for next iteration
            current_value = prediction
            current_velocity = current_velocity * 0.95 + (prediction - patterns['last_value']) * 0.05  # Decay velocity
            current_acceleration = current_acceleration * 0.9  # Decay acceleration
            
        return predictions
        
    except Exception as e:
        print(f"Error generating extrapolation: {e}")
        return []

def create_smooth_transition(historical_data, predicted_data, transition_points=5):
    """Create smooth transition between historical and predicted data"""
    try:
        if len(historical_data) < transition_points:
            return predicted_data
        
        # Get the last few points from historical data
        transition_hist = historical_data[-transition_points:]
        
        # Calculate smoothing weights
        weights = np.linspace(0.8, 0.2, transition_points)
        
        # Apply smoothing to first few predicted points
        smoothed_predictions = predicted_data.copy()
        
        for i in range(min(transition_points, len(predicted_data))):
            if i < len(transition_hist):
                # Blend historical trend with prediction
                hist_trend = transition_hist[i] if i < len(transition_hist) else transition_hist[-1]
                smoothed_predictions[i] = (
                    smoothed_predictions[i] * (1 - weights[i]) +
                    hist_trend * weights[i]
                )
        
        return smoothed_predictions
        
    except Exception as e:
        print(f"Error creating smooth transition: {e}")
        return predicted_data

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

def train_advanced_model(data: pd.DataFrame, time_col: str, target_col: str, params: Dict[str, Any]):
    """Train advanced ML models (DLinear, N-BEATS, LSTM, etc.)"""
    global current_advanced_model, current_preprocessor, model_evaluation_results
    
    try:
        model_type = params.get('model_type', 'dlinear')
        seq_len = params.get('seq_len', 50)
        pred_len = params.get('pred_len', 30)
        
        print(f"Training advanced model: {model_type}")
        
        # Enhanced preprocessing
        preprocessing_config = {
            'scaling_method': params.get('scaling_method', 'standard'),
            'denoise': params.get('denoise', True),
            'denoise_method': params.get('denoise_method', 'savgol'),
            'create_features': params.get('create_features', False),  # Keep simple for now
            'outlier_method': params.get('outlier_method', 'zscore')
        }
        
        # Preprocess data
        preprocessing_results = preprocess_time_series_data(
            data, time_col, target_col, preprocessing_config
        )
        
        current_preprocessor = preprocessing_results['preprocessor']
        processed_data = preprocessing_results['preprocessing_results']['processed_data']
        
        print(f"Data preprocessing completed. Quality score: {preprocessing_results['validation_results']['quality_score']:.2f}")
        
        # Create advanced model
        if model_type == 'ensemble':
            # Create ensemble of multiple models
            models = create_advanced_forecasting_models(seq_len, pred_len)
            current_advanced_model = ModelEnsemble(models)
            
            # Train ensemble
            training_results = current_advanced_model.fit(processed_data, time_col, target_col)
            
        else:
            # Create single advanced model
            current_advanced_model = AdvancedTimeSeriesForecaster(seq_len, pred_len, model_type)
            
            # Prepare data for training
            data_dict = current_advanced_model.prepare_data(processed_data, time_col, target_col)
            
            # Train model based on type
            if model_type in ['dlinear', 'nbeats', 'lstm']:
                training_results = current_advanced_model.train_pytorch_model(
                    data_dict['X_train'], data_dict['y_train'],
                    data_dict['X_test'], data_dict['y_test'],
                    epochs=params.get('epochs', 100),
                    batch_size=params.get('batch_size', 32),
                    lr=params.get('learning_rate', 0.001)
                )
            else:
                training_results = current_advanced_model.train_gradient_boosting(
                    data_dict['X_train'], data_dict['y_train'],
                    data_dict['X_test'], data_dict['y_test']
                )
            
            # Evaluate model
            evaluation_results = evaluate_model_performance(
                current_advanced_model, processed_data, time_col, target_col
            )
            model_evaluation_results = evaluation_results
            
        print(f"Advanced model training completed: {model_type}")
        return {
            'model': current_advanced_model,
            'preprocessor': current_preprocessor,
            'training_results': training_results,
            'evaluation_results': model_evaluation_results,
            'model_type': model_type
        }
        
    except Exception as e:
        print(f"Error training advanced model: {e}")
        raise e

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
    """Generate a single real-time pH reading that moves toward target pH"""
    global ph_simulation_data, target_ph
    
    if not ph_simulation_data:
        ph_simulation_data = generate_ph_simulation_data(24)
    
    # Get current pH based on simulation
    current_index = len(ph_simulation_data) - 1
    if current_index < len(ph_simulation_data):
        base_reading = ph_simulation_data[current_index]
        current_ph = base_reading['ph_value']
        
        # Add drift toward target pH
        ph_drift = (target_ph - current_ph) * 0.01  # Slow drift toward target
        ph_variation = random.gauss(0, 0.01)  # Random variation
        
        # Combine drift and variation
        new_ph = current_ph + ph_drift + ph_variation
        new_ph = max(6.0, min(8.0, new_ph))  # Keep within realistic bounds
        
        # Update the simulation data for next iteration
        ph_simulation_data[current_index]['ph_value'] = new_ph
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ph_value': round(new_ph, 2),
            'confidence': random.uniform(88, 97)
        }
    
    # Fallback that still respects target pH
    fallback_ph = target_ph + random.gauss(0, 0.1)
    fallback_ph = max(6.0, min(8.0, fallback_ph))
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ph_value': round(fallback_ph, 2),
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
    """Generate continuous predictions with advanced pattern-based extrapolation"""
    try:
        global current_model, continuous_predictions
        
        if current_model is None:
            raise HTTPException(status_code=400, detail="No model trained")
        
        model = current_model['model']
        model_type = current_model['model_type']
        data = current_model['data']
        time_col = current_model['time_col']
        target_col = current_model['target_col']
        
        # Analyze historical patterns
        patterns = analyze_historical_patterns(data, time_col, target_col)
        
        if not patterns:
            # Fallback to basic prediction
            return await generate_basic_prediction(model, model_type, data, steps)
        
        # Generate advanced extrapolation
        advanced_predictions = generate_advanced_extrapolation(patterns, steps)
        
        # Create smooth transition from historical data
        historical_values = data[target_col].values
        smoothed_predictions = create_smooth_transition(historical_values, advanced_predictions)
        
        # Calculate prediction offset for continuous extension
        prediction_offset = len(continuous_predictions) * 5  # Each call advances by 5 steps
        
        # Create timestamps for predictions
        if time_col in data.columns:
            last_timestamp = pd.to_datetime(data[time_col].iloc[-1])
            time_series = pd.to_datetime(data[time_col])
            freq = pd.infer_freq(time_series)
        else:
            last_timestamp = pd.to_datetime(data.index[-1])
            freq = pd.infer_freq(pd.to_datetime(data.index))
        
        # Default to daily frequency if inference fails
        if freq is None:
            freq = 'D'
        
        # Create future timestamps
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
        
        # Create result with advanced predictions
        result = {
            'timestamps': future_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'predictions': smoothed_predictions[:steps],
            'confidence_intervals': None,
            'pattern_analysis': {
                'trend_slope': patterns['trend_slope'],
                'velocity': patterns['velocity'],
                'recent_mean': patterns['recent_mean'],
                'last_value': patterns['last_value']
            }
        }
        
        # Store prediction for continuous use
        continuous_predictions.append(result)
        
        return result
        
    except Exception as e:
        print(f"Error in advanced continuous prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_basic_prediction(model, model_type, data, steps):
    """Fallback basic prediction method"""
    try:
        if model_type == 'prophet':
            future = model.make_future_dataframe(periods=steps)
            forecast = model.predict(future)
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
            
            return {
                'timestamps': predictions['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'predictions': predictions['yhat'].tolist(),
                'confidence_intervals': [
                    {'lower': row['yhat_lower'], 'upper': row['yhat_upper']} 
                    for _, row in predictions.iterrows()
                ]
            }
            
        elif model_type == 'arima':
            forecast = model.forecast(steps=steps)
            
            # Create basic timestamps
            last_timestamp = pd.to_datetime(data.index[-1] if hasattr(data, 'index') else '2023-01-01')
            future_timestamps = pd.date_range(
                start=last_timestamp + pd.Timedelta(days=1), 
                periods=steps, 
                freq='D'
            )
            
            return {
                'timestamps': future_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'predictions': forecast.tolist(),
                'confidence_intervals': None
            }
            
    except Exception as e:
        print(f"Error in basic prediction: {e}")
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

@api_router.get("/extend-prediction")
async def extend_prediction(steps: int = 5):
    """Extend current predictions with new points that follow the trend"""
    try:
        global current_model, continuous_predictions
        
        if current_model is None:
            raise HTTPException(status_code=400, detail="No model trained")
        
        if not continuous_predictions:
            # No existing predictions, generate initial ones
            return await generate_continuous_prediction("current", steps, 100)
        
        # Get the last prediction to continue from
        last_prediction = continuous_predictions[-1]
        last_values = last_prediction['predictions']
        
        # Analyze trend of recent predictions
        if len(last_values) >= 3:
            # Calculate trend from last few points
            x = np.arange(len(last_values))
            trend = np.polyfit(x[-3:], last_values[-3:], 1)[0]  # Linear trend from last 3 points
            velocity = np.mean(np.diff(last_values[-3:]))  # Average velocity
        else:
            trend = 0
            velocity = 0
        
        # Generate new predictions that follow the trend
        new_predictions = []
        last_value = last_values[-1]
        
        for i in range(steps):
            # Apply trend with some decay and noise
            trend_component = trend * (1 - i * 0.1)  # Decay trend over time
            velocity_component = velocity * (1 - i * 0.05)  # Decay velocity
            noise = np.random.normal(0, 0.1)  # Small noise
            
            next_value = last_value + trend_component + velocity_component + noise
            new_predictions.append(next_value)
            last_value = next_value
        
        # Create timestamps for new predictions
        data = current_model['data']
        time_col = current_model['time_col']
        
        if time_col in data.columns:
            last_timestamp = pd.to_datetime(last_prediction['timestamps'][-1])
            time_series = pd.to_datetime(data[time_col])
            freq = pd.infer_freq(time_series)
        else:
            last_timestamp = pd.to_datetime(last_prediction['timestamps'][-1])
            freq = 'D'
        
        if freq is None:
            freq = 'D'
        
        # Create future timestamps
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(days=1), 
            periods=steps, 
            freq=freq
        )
        
        result = {
            'timestamps': future_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'predictions': new_predictions,
            'confidence_intervals': None,
            'extension_info': {
                'trend': trend,
                'velocity': velocity,
                'base_value': last_values[-1]
            }
        }
        
        # Store the extension
        continuous_predictions.append(result)
        
        return result
        
    except Exception as e:
        print(f"Error extending predictions: {e}")
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