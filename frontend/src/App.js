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
  const [timeWindow, setTimeWindow] = useState(50);
  const [continuousPredictions, setContinuousPredictions] = useState([]);
  const [websocket, setWebsocket] = useState(null);
  const [phData, setPhData] = useState({ current_ph: 7.0, target_ph: 7.6, status: 'Connected' });
  const [realtimePhReadings, setRealtimePhReadings] = useState([]);
  const [lstmPredictions, setLstmPredictions] = useState([]);
  const [predictionConfidence, setPredictionConfidence] = useState(67);

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

  // Load pH simulation data
  const loadPhSimulation = async () => {
    try {
      const response = await fetch(`${API}/ph-simulation-history?hours=24`);
      if (response.ok) {
        const data = await response.json();
        setPhData(data);
        setRealtimePhReadings(data.data.slice(-100)); // Keep last 100 readings
      }
    } catch (error) {
      console.error('Error loading pH simulation:', error);
    }
  };

  // Generate continuous predictions with proper extrapolation
  const generateContinuousPredictions = async () => {
    if (!modelId) return;

    try {
      const response = await fetch(`${API}/generate-continuous-prediction?model_id=${modelId}&steps=30&time_window=${timeWindow}`);
      if (response.ok) {
        const data = await response.json();
        return data;
      }
    } catch (error) {
      console.error('Error generating continuous predictions:', error);
    }
    return null;
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

  // Generate predictions with time window
  const generatePredictions = async (window = 0) => {
    if (!modelId) return;

    try {
      const response = await fetch(`${API}/generate-prediction?model_id=${modelId}&steps=30&offset=${window}`);
      if (response.ok) {
        const data = await response.json();
        return data;
      }
    } catch (error) {
      console.error('Error generating predictions:', error);
    }
    return null;
  };

  // Update predictions based on time window
  const updatePredictionsWithWindow = async () => {
    if (isPredicting) {
      const newPredictions = await generateContinuousPredictions();
      if (newPredictions) {
        setPredictionData(newPredictions);
        setLstmPredictions(prev => {
          const updated = [...prev, ...newPredictions.predictions];
          return updated.slice(-timeWindow); // Keep only data within time window
        });
      }
    }
  };

  // Update predictions when time window changes
  useEffect(() => {
    if (isPredicting && predictionData) {
      updatePredictionsWithWindow();
    }
  }, [timeWindow]);

  // Simple chart component for pH monitoring dashboard
  const PhChart = ({ data, title, color = '#3B82F6', showAnimation = false, currentValue = null }) => {
    const canvasRef = React.useRef(null);
    const [animationFrame, setAnimationFrame] = React.useState(0);
    
    React.useEffect(() => {
      if (!data || !canvasRef.current) return;
      
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Draw gradient background
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, 'rgba(59, 130, 246, 0.1)');
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0.05)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
      
      // Draw grid
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      for (let i = 0; i <= 10; i++) {
        const x = 50 + (i / 10) * (width - 100);
        const y = 50 + (i / 10) * (height - 100);
        ctx.beginPath();
        ctx.moveTo(x, 50);
        ctx.lineTo(x, height - 50);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(50, y);
        ctx.lineTo(width - 50, y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      
      // Draw axes
      ctx.strokeStyle = '#374151';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(50, height - 50);
      ctx.lineTo(width - 50, height - 50);
      ctx.moveTo(50, 50);
      ctx.lineTo(50, height - 50);
      ctx.stroke();
      
      // Draw data
      const values = data.values || data;
      if (values && values.length > 0) {
        const maxVal = Math.max(...values);
        const minVal = Math.min(...values);
        const range = maxVal - minVal || 1;
        
        // Draw area fill
        ctx.fillStyle = color.replace('rgb', 'rgba').replace(')', ', 0.3)');
        ctx.beginPath();
        ctx.moveTo(50, height - 50);
        
        values.forEach((value, index) => {
          const x = 50 + (index / (values.length - 1)) * (width - 100);
          const y = height - 50 - ((value - minVal) / range) * (height - 100);
          ctx.lineTo(x, y);
        });
        
        ctx.lineTo(width - 50, height - 50);
        ctx.closePath();
        ctx.fill();
        
        // Draw line
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        
        values.forEach((value, index) => {
          const x = 50 + (index / (values.length - 1)) * (width - 100);
          const y = height - 50 - ((value - minVal) / range) * (height - 100);
          
          if (index === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        
        ctx.stroke();
        
        // Draw data points
        ctx.fillStyle = color;
        values.forEach((value, index) => {
          const x = 50 + (index / (values.length - 1)) * (width - 100);
          const y = height - 50 - ((value - minVal) / range) * (height - 100);
          
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, 2 * Math.PI);
          ctx.fill();
        });
        
        // Show animation effect for live data
        if (showAnimation && isPredicting) {
          const pulseSize = 3 + Math.sin(animationFrame * 0.1) * 2;
          const lastIndex = values.length - 1;
          const x = 50 + (lastIndex / (values.length - 1)) * (width - 100);
          const y = height - 50 - ((values[lastIndex] - minVal) / range) * (height - 100);
          
          ctx.strokeStyle = '#10B981';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, pulseSize, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
      
      // Draw title
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(title, width / 2, 25);
      
      // Draw current value if provided
      if (currentValue !== null) {
        ctx.fillStyle = '#059669';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'right';
        ctx.fillText(`Current: ${currentValue}`, width - 60, 45);
      }
      
    }, [data, title, color, showAnimation, animationFrame, isPredicting, currentValue]);
    
    React.useEffect(() => {
      let animationId;
      if (showAnimation && isPredicting) {
        const animate = () => {
          setAnimationFrame(prev => prev + 1);
          animationId = requestAnimationFrame(animate);
        };
        animationId = requestAnimationFrame(animate);
      }
      return () => {
        if (animationId) {
          cancelAnimationFrame(animationId);
        }
      };
    }, [showAnimation, isPredicting]);
    
    return <canvas ref={canvasRef} width={360} height={280} className="border border-gray-300 rounded-lg shadow-sm" />;
  };

  // Start continuous prediction with proper extrapolation
  const startContinuousPrediction = async () => {
    setIsPredicting(true);
    setContinuousPredictions([]);
    
    // Load pH simulation data
    await loadPhSimulation();
    
    try {
      // Reset backend continuous prediction state
      await fetch(`${API}/reset-continuous-prediction`, { method: 'POST' });
      
      // Start continuous prediction on backend
      await fetch(`${API}/start-continuous-prediction`, { method: 'POST' });
      
      // Generate initial predictions
      const initialPredictions = await generateContinuousPredictions();
      if (initialPredictions) {
        setPredictionData(initialPredictions);
        setLstmPredictions(initialPredictions.predictions);
      }
      
      // Set up interval for continuous predictions that extrapolate
      const interval = setInterval(async () => {
        try {
          if (isPredicting) {
            // Generate new continuous predictions
            const newPredictions = await generateContinuousPredictions();
            if (newPredictions) {
              setPredictionData(newPredictions);
              setLstmPredictions(prev => {
                const updated = [...prev, ...newPredictions.predictions];
                return updated.slice(-timeWindow); // Keep within time window
              });
            }
            
            // Update pH simulation
            const phResponse = await fetch(`${API}/ph-simulation`);
            if (phResponse.ok) {
              const phReading = await phResponse.json();
              setPhData(prev => ({ ...prev, current_ph: phReading.ph_value }));
              setRealtimePhReadings(prev => {
                const updated = [...prev, phReading];
                return updated.slice(-timeWindow); // Keep within time window
              });
            }
          }
        } catch (error) {
          console.error('Error in continuous prediction loop:', error);
        }
      }, 1000); // Update every 1 second for smooth extrapolation
      
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
                onClick={async () => {
                  const predictions = await generatePredictions(0);
                  if (predictions) setPredictionData(predictions);
                }}
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
              <div className="text-center">
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Prediction Control
                </label>
                <input
                  type="range"
                  min="-100"
                  max="100"
                  value={verticalOffset}
                  onChange={(e) => setVerticalOffset(Number(e.target.value))}
                  className="w-32 lg:w-32 lg:transform lg:-rotate-90"
                />
                <div className="text-xs text-gray-500 mt-2">
                  Offset: {verticalOffset}
                </div>
                <div className="text-xs text-green-600 mt-1">
                  {isPredicting ? '🔄 Live' : '⏸️ Static'}
                </div>
              </div>
            </div>
            
            {/* Predictions Graph */}
            <div className="flex-1">
              <h3 className="text-lg font-semibold mb-4 text-center">
                🔮 Predictions {isPredicting && <span className="text-green-600 animate-pulse">(Live)</span>}
              </h3>
              {predictionData ? (
                <SimpleChart 
                  data={{ values: predictionData.predictions }} 
                  title="Continuous Predictions"
                  color="#10B981"
                  showAnimation={true}
                />
              ) : (
                <div className="border border-gray-300 rounded h-64 flex items-center justify-center text-gray-500">
                  Click "Generate Predictions" to start
                </div>
              )}
            </div>
          </div>
          
          {/* Data Summary */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
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
            
            <div className="bg-purple-50 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-800 mb-2">⚙️ Control</h4>
              <p className="text-sm text-purple-700">
                Slider offset: {verticalOffset}
                <br />
                Prediction offset: {predictionOffset}
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