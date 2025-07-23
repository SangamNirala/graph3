# Advanced pH Monitoring and Prediction System

A comprehensive real-time pH monitoring and prediction system with advanced machine learning capabilities, universal waveform pattern learning, and noise reduction algorithms.

## 🌟 Features

### Core Functionality
- **Real-time pH Monitoring**: Three-panel dashboard for comprehensive pH monitoring
- **Advanced ML Predictions**: Multiple models including LSTM, Prophet, ARIMA, DLinear, N-BEATS
- **Universal Waveform Learning**: Learns and reproduces any pattern complexity (square wave, triangular, sinusoidal, etc.)
- **Noise Reduction**: Advanced algorithms with Savitzky-Golay, Gaussian, Butterworth filters
- **Pattern Following**: Enhanced algorithms that maintain historical patterns and eliminate downward bias
- **File Upload**: CSV/Excel support with automatic data analysis and column detection
- **Continuous Prediction**: Real-time extrapolation with smooth transitions
- **Interactive Controls**: pH target slider, time window controls, enhanced visual smoothing

### Advanced Features  
- **Enhanced Pattern Analysis**: Multi-scale pattern detection and learning
- **Adaptive Smoothing**: Pattern-aware smoothing (none for square waves, low for triangular, medium for sinusoidal)
- **Quality Validation**: Comprehensive data quality scoring and recommendations
- **Model Comparison**: Automated model performance comparison and selection
- **Hyperparameter Optimization**: Automated optimization using Optuna
- **Confidence Intervals**: Statistical confidence bounds for all predictions

## 🏗️ System Architecture

### Backend (FastAPI)
- **API Server**: FastAPI with automatic OpenAPI documentation
- **Database**: MongoDB for data persistence  
- **ML Models**: PyTorch, Prophet, Scikit-learn, LightGBM, XGBoost
- **Processing**: Advanced noise reduction and pattern analysis engines
- **WebSocket**: Real-time data streaming (note: may have infrastructure limitations)

### Frontend (React)
- **UI Framework**: React 19 with modern hooks
- **Styling**: Tailwind CSS for responsive design
- **Visualization**: Plotly.js for interactive charts and graphs
- **File Upload**: React Dropzone for drag-and-drop file uploads
- **Real-time Updates**: WebSocket integration for live data

### Database (MongoDB)
- **Local Instance**: MongoDB running on default port 27017
- **Data Storage**: Time series data, model states, user uploads
- **Indexing**: Optimized for time-based queries

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ 
- Node.js 20+
- Yarn package manager
- Docker (for MongoDB)
- Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Database Setup (MongoDB with Docker)

#### Install Docker (if not available)
```bash
sudo apt update
sudo apt install -y docker.io
```

#### Start MongoDB Container
```bash
# Start MongoDB in Docker container
docker run -d --name mongodb -p 27017:27017 mongo:latest

# Verify MongoDB is running
docker ps
```

You should see output like:
```
CONTAINER ID   IMAGE          COMMAND                  CREATED       STATUS       PORTS                                             NAMES
6e0a0ee5ec42   mongo:latest   "docker-entrypoint.s…"   1 minute ago  Up 1 minute  0.0.0.0:27017->27017/tcp, [::]:27017->27017/tcp  mongodb
```

### 3. Backend Setup

#### Set Environment Variables
The backend `.env` file should contain:
```bash
# backend/.env
MONGO_URL="mongodb://localhost:27017"
DB_NAME="test_database"
```

#### Install Python Dependencies
```bash
cd /app/backend
pip install -r requirements.txt
```

#### Start Backend
```bash
cd /app/backend
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Frontend Setup

#### Environment Configuration
The frontend `.env` file is already configured:
```bash
# frontend/.env (already configured)
REACT_APP_BACKEND_URL=https://ee04ac22-cb45-4b61-832c-93de71320985.preview.emergentagent.com
WDS_SOCKET_PORT=443
```

#### Install Node Dependencies and Start
```bash
cd /app/frontend
yarn install
yarn start
```

### 5. Complete System Startup (3 Terminals Required)

#### Terminal 1: MongoDB (already running via Docker)
```bash
# Verify MongoDB is running
docker ps

# If not running, start it:
docker run -d --name mongodb -p 27017:27017 mongo:latest
```

#### Terminal 2: Backend
```bash
cd /app/backend
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

#### Terminal 3: Frontend
```bash
cd /app/frontend
yarn start
```

#### Verify Services
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001 
- **API Documentation**: http://localhost:8001/docs
- **MongoDB**: mongodb://localhost:27017 (running in Docker)

## 📊 Usage Guide

### 1. Data Upload
- Navigate to the application frontend
- Use the file upload area to upload CSV or Excel files
- Supported formats: CSV with timestamp and pH columns
- The system automatically detects columns and analyzes data quality

### 2. Model Training  
- Configure parameters (timestamp column, pH column, model type)
- Choose from available models:
  - **LSTM**: Deep learning for complex patterns
  - **Prophet**: Time series with seasonality
  - **ARIMA**: Traditional statistical forecasting
  - **DLinear**: Linear decomposition model
  - **N-BEATS**: Neural basis expansion

### 3. Prediction Generation
- **Single Predictions**: Generate point forecasts
- **Continuous Predictions**: Real-time extrapolation
- **Advanced Predictions**: Ensemble methods with confidence intervals
- **Universal Waveform**: Pattern-aware predictions that preserve input characteristics

### 4. Interactive Dashboard
- **Left Panel**: Historical pH data visualization
- **Center Panel**: Real-time pH sensor readings with controls
- **Right Panel**: LSTM predictions with noise reduction

### 5. Pattern Learning
- Upload different waveform types (square, triangular, sinusoidal)
- System automatically detects pattern characteristics  
- Predictions maintain input pattern shapes
- Enhanced visual smoothing adapts to pattern type

## 🔧 Configuration

### Backend Configuration  
```python
# Key configuration files:
- server.py: Main FastAPI application
- requirements.txt: Python dependencies
- .env: Environment variables (MONGO_URL, DB_NAME)
```

### Frontend Configuration  
```javascript
// Key configuration files:
- package.json: Node.js dependencies
- tailwind.config.js: Styling configuration
- .env: Environment variables (REACT_APP_BACKEND_URL)
```

### Docker MongoDB Configuration
```bash
# MongoDB running in Docker container
# Container name: mongodb
# Port mapping: 0.0.0.0:27017->27017/tcp
# Image: mongo:latest

# Management commands:
docker ps                    # Check if container is running
docker stop mongodb          # Stop MongoDB container
docker start mongodb         # Start MongoDB container
docker restart mongodb       # Restart MongoDB container
```

## 🧪 Testing

### Backend Testing
```bash
cd /app
python backend_test.py
python comprehensive_ph_test.py  
python universal_waveform_test.py
```

### Frontend Testing
```bash
cd frontend
yarn test
```

### Integration Testing
```bash
# Test complete workflow
python test_ph_dataset.py
python focused_prediction_test.py
```

## 🔍 API Endpoints

### Data Management
- `POST /api/upload-data`: Upload CSV/Excel files
- `GET /api/data-quality-report`: Get data quality analysis

### Model Training
- `POST /api/train-model`: Train ML models
- `GET /api/supported-models`: List available models
- `GET /api/model-performance`: Get model metrics

### Predictions
- `POST /api/generate-prediction`: Basic predictions
- `POST /api/generate-advanced-prediction`: Ensemble predictions  
- `POST /api/generate-universal-waveform-prediction`: Pattern-aware predictions
- `POST /api/generate-enhanced-realtime-prediction`: Noise-reduced predictions

### Real-time Features
- `POST /api/start-continuous-prediction`: Start real-time predictions
- `POST /api/stop-continuous-prediction`: Stop real-time predictions
- `GET /api/ph-simulation`: Get simulated pH readings

## 🐛 Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check backend logs in Terminal 2
# Common fixes:
cd /app/backend
pip install -r requirements.txt

# Check if all dependencies are installed
python -c "import pymongo; print('MongoDB driver OK')"
```

#### Frontend Won't Start  
```bash
# Check frontend logs in Terminal 3
# Common fixes:
cd /app/frontend
yarn install

# Clear cache if needed
yarn cache clean
```

#### MongoDB Connection Issues
```bash
# Check if MongoDB container is running
docker ps

# If not running, start MongoDB
docker run -d --name mongodb -p 27017:27017 mongo:latest

# If container exists but stopped
docker start mongodb

# Check MongoDB logs
docker logs mongodb

# Test MongoDB connection
docker exec -it mongodb mongosh --eval "db.runCommand({ping: 1})"
```

#### Port Already in Use Errors
```bash
# Check what's using the ports
sudo lsof -i :27017  # MongoDB
sudo lsof -i :8001   # Backend
sudo lsof -i :3000   # Frontend

# Kill processes if needed
sudo kill -9 <PID>
```

#### Docker Issues
```bash
# If Docker command fails
sudo service docker start

# If container name conflicts
docker rm mongodb
docker run -d --name mongodb -p 27017:27017 mongo:latest

# Check Docker status
docker --version
docker ps -a
```

### Performance Optimization
- For large datasets, use data preprocessing and quality validation
- Enable noise reduction for smoother real-time predictions  
- Use appropriate model types for your data characteristics
- Monitor memory usage with advanced ML models

## 📁 Project Structure

```
/app/
├── backend/                 # FastAPI backend
│   ├── server.py           # Main API server
│   ├── requirements.txt    # Python dependencies
│   ├── advanced_models.py  # ML model implementations
│   ├── data_preprocessing.py # Data processing pipeline
│   ├── advanced_noise_reduction.py # Noise reduction algorithms
│   ├── universal_waveform_learning.py # Pattern learning system
│   └── .env               # Environment variables
├── frontend/               # React frontend  
│   ├── src/
│   │   ├── App.js         # Main React application
│   │   └── components/    # React components
│   ├── package.json       # Node.js dependencies
│   ├── tailwind.config.js # Styling configuration
│   └── .env              # Environment variables
├── tests/                 # Test files
├── README.md             # This file
└── test_result.md        # Testing documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test_result.md file for known issues
3. Check API documentation at http://localhost:8001/docs
4. Contact the development team

## 🔄 Recent Updates

- ✅ Fixed universal waveform pattern learning (no more sine wave fallback)
- ✅ Implemented comprehensive noise reduction system  
- ✅ Resolved pH prediction downward trend bias
- ✅ Enhanced pattern-following algorithms
- ✅ Added adaptive visual smoothing
- ✅ Improved continuous prediction extrapolation
- ⚠️ WebSocket real-time streaming has infrastructure limitations
- ⚠️ Prophet model may have stan_backend compatibility issues

---

**Made with ❤️ using FastAPI, React, and advanced ML techniques**
