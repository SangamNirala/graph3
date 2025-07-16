import React, { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [currentStep, setCurrentStep] = useState('upload');
  const [uploadedData, setUploadedData] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [parameters, setParameters] = useState({});
  const [modelId, setModelId] = useState(null);
  const [historicalData, setHistoricalData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [verticalOffset, setVerticalOffset] = useState(0);
  const [predictionOffset, setPredictionOffset] = useState(0);
  const [continuousPredictions, setContinuousPredictions] = useState([]);
  const [websocket, setWebsocket] = useState(null);

  // File upload handler
  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API}/upload-data`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setUploadedData(result);
        setAnalysis(result.analysis);
        setParameters(result.analysis.suggested_parameters);
        setCurrentStep('parameters');
      } else {
        alert('Error uploading file');
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Error uploading file');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    multiple: false
  });

  // Train model
  const handleTrainModel = async () => {
    if (!uploadedData || !parameters.time_column || !parameters.target_column) {
      alert('Please select time and target columns');
      return;
    }

    setIsTraining(true);
    try {
      const response = await fetch(`${API}/train-model?data_id=${uploadedData.data_id}&model_type=${parameters.model_type}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(parameters),
      });

      if (response.ok) {
        const result = await response.json();
        setModelId(result.model_id);
        setCurrentStep('prediction');
        
        // Load historical data
        await loadHistoricalData();
      } else {
        alert('Error training model');
      }
    } catch (error) {
      console.error('Training error:', error);
      alert('Error training model');
    } finally {
      setIsTraining(false);
    }
  };

  // Load historical data
  const loadHistoricalData = async () => {
    try {
      const response = await fetch(`${API}/historical-data`);
      if (response.ok) {
        const data = await response.json();
        setHistoricalData(data);
      }
    } catch (error) {
      console.error('Error loading historical data:', error);
    }
  };

  // Generate predictions
  const generatePredictions = async () => {
    if (!modelId) return;

    try {
      const response = await fetch(`${API}/generate-prediction?model_id=${modelId}&steps=50`);
      if (response.ok) {
        const data = await response.json();
        setPredictionData(data);
      }
    } catch (error) {
      console.error('Error generating predictions:', error);
    }
  };

  // Simple chart component using Canvas
  const SimpleChart = ({ data, title, color = '#3B82F6' }) => {
    const canvasRef = React.useRef(null);
    
    React.useEffect(() => {
      if (!data || !canvasRef.current) return;
      
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Draw axes
      ctx.strokeStyle = '#ddd';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(50, height - 50);
      ctx.lineTo(width - 50, height - 50);
      ctx.moveTo(50, 50);
      ctx.lineTo(50, height - 50);
      ctx.stroke();
      
      // Draw data
      if (data.values && data.values.length > 0) {
        const maxVal = Math.max(...data.values);
        const minVal = Math.min(...data.values);
        const range = maxVal - minVal || 1;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        data.values.forEach((value, index) => {
          const x = 50 + (index / (data.values.length - 1)) * (width - 100);
          const y = height - 50 - ((value - minVal) / range) * (height - 100);
          
          if (index === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        
        ctx.stroke();
      }
      
      // Draw title
      ctx.fillStyle = '#333';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(title, width / 2, 30);
      
    }, [data, title, color]);
    
    return <canvas ref={canvasRef} width={400} height={300} className="border border-gray-300 rounded" />;
  };

  // Start continuous prediction
  const startContinuousPrediction = async () => {
    setIsPredicting(true);
    
    try {
      // Start continuous prediction on backend
      await fetch(`${API}/start-continuous-prediction`, { method: 'POST' });
      
      // Generate initial predictions
      await generatePredictions();
      
      // Set up polling for updates instead of WebSocket
      const interval = setInterval(async () => {
        if (isPredicting) {
          await generatePredictions();
        }
      }, 3000);
      
      setWebsocket(interval);
      
    } catch (error) {
      console.error('Error starting continuous prediction:', error);
    }
  };

  // Stop continuous prediction
  const stopContinuousPrediction = async () => {
    setIsPredicting(false);
    
    if (websocket) {
      clearInterval(websocket);
      setWebsocket(null);
    }
    
    try {
      await fetch(`${API}/stop-continuous-prediction`, { method: 'POST' });
    } catch (error) {
      console.error('Error stopping continuous prediction:', error);
    }
  };

  // Cleanup interval on component unmount
  useEffect(() => {
    return () => {
      if (websocket) {
        clearInterval(websocket);
      }
    };
  }, [websocket]);

  // Render upload step
  const renderUploadStep = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">
          🚀 Real-Time Graph Prediction
        </h1>
        
        <div className="bg-white rounded-lg shadow-xl p-8">
          <h2 className="text-2xl font-semibold mb-6 text-gray-700">Upload Your Data</h2>
          
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
              ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
          >
            <input {...getInputProps()} />
            <div className="text-gray-600">
              <div className="text-6xl mb-4">📊</div>
              {isDragActive ? (
                <p className="text-lg">Drop the file here...</p>
              ) : (
                <div>
                  <p className="text-lg mb-2">Drag and drop your data file here, or click to browse</p>
                  <p className="text-sm text-gray-500">Supports CSV, Excel files</p>
                </div>
              )}
            </div>
          </div>
          
          {analysis && (
            <div className="mt-8 p-6 bg-gray-50 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">📈 Data Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p><strong>File:</strong> {uploadedData.analysis.columns.length} columns, {uploadedData.analysis.data_shape[0]} rows</p>
                  <p><strong>Time columns:</strong> {analysis.time_columns.join(', ') || 'None detected'}</p>
                  <p><strong>Numeric columns:</strong> {analysis.numeric_columns.join(', ') || 'None detected'}</p>
                </div>
                <div>
                  <p><strong>Suggested target:</strong> {analysis.suggested_parameters.target_column || 'None'}</p>
                  <p><strong>Suggested time:</strong> {analysis.suggested_parameters.time_column || 'None'}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  // Render parameters step
  const renderParametersStep = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">
          ⚙️ Configure Model Parameters
        </h1>
        
        <div className="bg-white rounded-lg shadow-xl p-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Time Column
              </label>
              <select
                value={parameters.time_column || ''}
                onChange={(e) => setParameters({...parameters, time_column: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select time column</option>
                {analysis?.time_columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Target Column
              </label>
              <select
                value={parameters.target_column || ''}
                onChange={(e) => setParameters({...parameters, target_column: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select target column</option>
                {analysis?.numeric_columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Type
              </label>
              <select
                value={parameters.model_type || 'prophet'}
                onChange={(e) => setParameters({...parameters, model_type: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="prophet">Prophet (Recommended)</option>
                <option value="arima">ARIMA</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Prediction Horizon
              </label>
              <input
                type="number"
                value={parameters.prediction_horizon || 30}
                onChange={(e) => setParameters({...parameters, prediction_horizon: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                min="1"
                max="365"
              />
            </div>
          </div>
          
          <div className="mt-8 flex justify-between">
            <button
              onClick={() => setCurrentStep('upload')}
              className="px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
            >
              ← Back
            </button>
            
            <button
              onClick={handleTrainModel}
              disabled={isTraining || !parameters.time_column || !parameters.target_column}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
            >
              {isTraining ? '🔄 Training...' : '🚀 Train Model'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  // Render prediction step
  const renderPredictionStep = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">
          📊 Real-Time Predictions
        </h1>
        
        <div className="bg-white rounded-lg shadow-xl p-6">
          <div className="mb-6 flex justify-between items-center">
            <button
              onClick={() => setCurrentStep('parameters')}
              className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
            >
              ← Back
            </button>
            
            <div className="flex space-x-4">
              <button
                onClick={generatePredictions}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                📈 Generate Predictions
              </button>
              
              {!isPredicting ? (
                <button
                  onClick={startContinuousPrediction}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  ▶️ Start Continuous Prediction
                </button>
              ) : (
                <button
                  onClick={stopContinuousPrediction}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  ⏹️ Stop Continuous Prediction
                </button>
              )}
            </div>
          </div>
          
          {/* Status Indicator */}
          {isPredicting && (
            <div className="mb-4 text-center">
              <div className="inline-flex items-center px-4 py-2 bg-green-100 text-green-800 rounded-full">
                <div className="w-3 h-3 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                Live Predictions Active
              </div>
            </div>
          )}
          
          {/* Graphs Container */}
          <div className="flex flex-col lg:flex-row space-y-6 lg:space-y-0 lg:space-x-6">
            {/* Historical Data Graph */}
            <div className="flex-1">
              <h3 className="text-lg font-semibold mb-4 text-center">📈 Historical Data</h3>
              {historicalData ? (
                <SimpleChart 
                  data={historicalData} 
                  title="Historical Data"
                  color="#3B82F6"
                />
              ) : (
                <div className="border border-gray-300 rounded h-64 flex items-center justify-center text-gray-500">
                  No historical data available
                </div>
              )}
            </div>
            
            {/* Vertical Slider */}
            <div className="flex lg:flex-col justify-center items-center px-4">
              <label className="text-sm font-medium text-gray-700 mb-2 lg:transform lg:-rotate-90">
                Vertical Pan
              </label>
              <input
                type="range"
                min="-100"
                max="100"
                value={verticalOffset}
                onChange={(e) => setVerticalOffset(Number(e.target.value))}
                className="w-32 lg:w-32 lg:transform lg:-rotate-90"
              />
              <div className="text-xs text-gray-500 mt-2 lg:transform lg:-rotate-90">
                {verticalOffset}
              </div>
            </div>
            
            {/* Predictions Graph */}
            <div className="flex-1">
              <h3 className="text-lg font-semibold mb-4 text-center">
                🔮 Predictions {isPredicting && <span className="text-green-600">(Live)</span>}
              </h3>
              {predictionData ? (
                <SimpleChart 
                  data={{ values: predictionData.predictions }} 
                  title="Predictions"
                  color="#10B981"
                />
              ) : (
                <div className="border border-gray-300 rounded h-64 flex items-center justify-center text-gray-500">
                  Click "Generate Predictions" to start
                </div>
              )}
            </div>
          </div>
          
          {/* Data Summary */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-800 mb-2">📊 Historical Data</h4>
              <p className="text-sm text-blue-700">
                {historicalData ? `${historicalData.values.length} data points` : 'No data loaded'}
              </p>
            </div>
            
            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="font-semibold text-green-800 mb-2">🔮 Predictions</h4>
              <p className="text-sm text-green-700">
                {predictionData ? `${predictionData.predictions.length} predictions generated` : 'No predictions yet'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Main render
  return (
    <div className="App">
      {currentStep === 'upload' && renderUploadStep()}
      {currentStep === 'parameters' && renderParametersStep()}
      {currentStep === 'prediction' && renderPredictionStep()}
    </div>
  );
}

export default App;