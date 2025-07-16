# Real-Time Graph Prediction Application

A full-stack web application that allows users to upload time-series data, train machine learning models (Prophet/ARIMA), and generate real-time predictions with interactive graph visualizations.

## Features

- **File Upload**: Drag-and-drop CSV/Excel file upload with automatic data analysis
- **ML Model Training**: Support for Prophet and ARIMA time-series forecasting models
- **Interactive Dashboard**: Three-panel pH monitoring dashboard with real-time visualizations
- **Prediction Generation**: Static prediction generation and continuous real-time extrapolation
- **Time Window Control**: Slider to control the number of data points displayed
- **Real-time pH Simulation**: Integrated pH sensor simulation with live data feeds

## Tech Stack

- **Frontend**: React 18+ with Tailwind CSS
- **Backend**: FastAPI (Python) with async support
- **Database**: MongoDB
- **ML Libraries**: Prophet, ARIMA (statsmodels), scikit-learn
- **Visualization**: Custom HTML5 Canvas charts

## Prerequisites

Before running the application, make sure you have the following installed:

- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **MongoDB** (running locally or connection string)
- **Yarn** package manager
- **pip** for Python package management

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd real-time-graph-prediction
```

### 2. Backend Setup

#### 2.1 Navigate to backend directory
```bash
cd backend
```

#### 2.2 Create Python virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

#### 2.3 Install Python dependencies
```bash
pip install -r requirements.txt
```

#### 2.4 Set up environment variables
Create a `.env` file in the `backend` directory:
```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=prediction_app
```

**Note**: Replace `MONGO_URL` with your MongoDB connection string if using a remote database.

#### 2.5 Start MongoDB (if running locally)
```bash
# On macOS with Homebrew
brew services start mongodb/brew/mongodb-community

# On Ubuntu/Debian
sudo systemctl start mongod

# On Windows
# Start MongoDB service from Services panel or command line
```

### 3. Frontend Setup

#### 3.1 Navigate to frontend directory
```bash
cd ../frontend
```

#### 3.2 Install Node.js dependencies
```bash
yarn install
```

#### 3.3 Set up environment variables
Create a `.env` file in the `frontend` directory:
```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

**Note**: Update the backend URL if your backend is running on a different port or domain.

## Running the Application

### Method 1: Manual Startup (Development)

#### 1. Start the Backend
```bash
cd backend
python server.py
```
The backend will start on `http://localhost:8001`

#### 2. Start the Frontend (in a new terminal)
```bash
cd frontend
yarn start
```
The frontend will start on `http://localhost:3000`

### Method 2: Using Supervisor (Production-like)

If you have supervisor installed:

```bash
# Start all services
sudo supervisorctl start all

# Check status
sudo supervisorctl status

# Stop all services
sudo supervisorctl stop all
```

## Usage Guide

### 1. Upload Data
1. Navigate to `http://localhost:3000`
2. Drag and drop a CSV/Excel file with time-series data
3. The file should contain at least:
   - A time column (date/datetime)
   - A numeric target column for prediction

### 2. Configure Parameters
1. Select the **Time Column** from your data
2. Select the **Target Column** (numeric values to predict)
3. Choose **Model Type** (Prophet recommended for seasonal data, ARIMA for stationary data)
4. Set **Prediction Horizon** (number of future points to predict)
5. Click **Train Model**

### 3. Generate Predictions
1. After training, you'll see a three-panel dashboard
2. **Left Panel**: Shows historical data from your uploaded file
3. **Right Panel**: Shows prediction results
4. **Middle Panel**: Control buttons and pH simulation

#### Static Predictions
- Click **"Generate Predictions"** to create a one-time prediction
- View results in the right panel with green visualization

#### Continuous Predictions
- Click **"Start Continuous Prediction"** for real-time extrapolation
- Watch as predictions continuously extend forward
- Use **"Stop Continuous Prediction"** to halt the process

### 4. Time Window Control
- Use the slider at the bottom to control how many data points are displayed
- Range: 20-200 data points
- Affects both historical and prediction graphs

## Sample Data Format

Your CSV file should look like this:

```csv
date,sales,region
2023-01-01,1000,North
2023-01-02,1050,North
2023-01-03,980,North
...
```

**Requirements**:
- Time column: Any date/datetime format
- Target column: Numeric values
- Additional columns: Optional (will be ignored)

## Troubleshooting

### Common Issues

#### 1. Backend won't start
- Check if MongoDB is running
- Verify Python dependencies are installed
- Check the `.env` file configuration

#### 2. Frontend won't start
- Run `yarn install` to ensure all dependencies are installed
- Check if the backend URL in `.env` is correct
- Verify Node.js version is 16 or higher

#### 3. File upload fails
- Ensure the CSV file has proper headers
- Check for time and numeric columns
- File size should be reasonable (< 10MB)

#### 4. Model training fails
- Ensure you have selected valid time and target columns
- Check if the data has enough points (minimum 10-20 recommended)
- For Prophet: requires at least 2 periods of data

#### 5. Graphs not showing
- Check browser console for JavaScript errors
- Verify API calls are successful (Network tab)
- Ensure data is properly formatted

### Debug Mode

To enable debug logging:

**Backend**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Frontend**:
Open browser developer tools and check the console for debug messages.

## API Endpoints

### Core Endpoints
- `POST /api/upload-data` - Upload and analyze data file
- `POST /api/train-model` - Train ML model
- `GET /api/generate-prediction` - Generate static predictions
- `GET /api/generate-continuous-prediction` - Generate continuous predictions
- `GET /api/historical-data` - Get historical data

### Control Endpoints
- `POST /api/start-continuous-prediction` - Start continuous prediction
- `POST /api/stop-continuous-prediction` - Stop continuous prediction
- `POST /api/reset-continuous-prediction` - Reset prediction state

### Simulation Endpoints
- `GET /api/ph-simulation` - Get current pH reading
- `GET /api/ph-simulation-history` - Get pH history data

## Project Structure

```
real-time-graph-prediction/
├── README.md
├── backend/
│   ├── server.py              # FastAPI application
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Backend environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js            # Main React component
│   │   ├── App.css           # Styles
│   │   └── index.js          # React entry point
│   ├── public/
│   ├── package.json          # Node.js dependencies
│   └── .env                  # Frontend environment variables
└── test_result.md            # Testing documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test_result.md file for detailed testing information
3. Open an issue in the repository

---

**Note**: This application is designed for educational and demonstration purposes. For production use, consider implementing proper authentication, error handling, and security measures.
