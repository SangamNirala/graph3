import React, { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';

// Lazy load Plotly to reduce initial bundle size
const Plot = React.lazy(() => import('react-plotly.js'));

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

  // Start continuous prediction
  const startContinuousPrediction = async () => {
    setIsPredicting(true);
    
    try {
      // Start continuous prediction on backend
      await fetch(`${API}/start-continuous-prediction`, { method: 'POST' });
      
      // Connect to WebSocket
      const ws = new WebSocket(`${BACKEND_URL.replace('http', 'ws')}/ws/predictions`);
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'prediction_update') {
          setPredictionData(prev => ({
            ...prev,
            ...data.data
          }));
        }
      };
      
      ws.onopen = () => {
        console.log('WebSocket connected');
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
      };
      
      setWebsocket(ws);
      
      // Also generate initial predictions
      await generatePredictions();
      
    } catch (error) {
      console.error('Error starting continuous prediction:', error);
    }
  };

  // Stop continuous prediction
  const stopContinuousPrediction = async () => {
    setIsPredicting(false);
    
    if (websocket) {
      websocket.close();
      setWebsocket(null);
    }
    
    try {
      await fetch(`${API}/stop-continuous-prediction`, { method: 'POST' });
    } catch (error) {
      console.error('Error stopping continuous prediction:', error);
    }
  };

  // Cleanup WebSocket on component unmount
  useEffect(() => {
    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, [websocket]);

  // Render upload step
  const renderUploadStep = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">
          Real-Time Graph Prediction
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
              <svg className="mx-auto h-16 w-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
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
              <h3 className="text-lg font-semibold mb-4">Data Analysis</h3>
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
          Configure Model Parameters
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
                <option value="prophet">Prophet</option>
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
              Back
            </button>
            
            <button
              onClick={handleTrainModel}
              disabled={isTraining || !parameters.time_column || !parameters.target_column}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
            >
              {isTraining ? 'Training...' : 'Train Model'}
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
          Real-Time Predictions
        </h1>
        
        <div className="bg-white rounded-lg shadow-xl p-6">
          <div className="mb-6 flex justify-between items-center">
            <button
              onClick={() => setCurrentStep('parameters')}
              className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
            >
              Back
            </button>
            
            <div className="flex space-x-4">
              <button
                onClick={generatePredictions}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Generate Predictions
              </button>
              
              {!isPredicting ? (
                <button
                  onClick={startContinuousPrediction}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  Start Continuous Prediction
                </button>
              ) : (
                <button
                  onClick={stopContinuousPrediction}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  Stop Continuous Prediction
                </button>
              )}
            </div>
          </div>
          
          {/* Graphs Container */}
          <div className="flex space-x-4">
            {/* Historical Data Graph */}
            <div className="flex-1">
              <h3 className="text-lg font-semibold mb-4 text-center">Historical Data</h3>
              {historicalData && (
                <React.Suspense fallback={<div className="text-center py-8">Loading graph...</div>}>
                  <Plot
                    data={[
                      {
                        x: historicalData.timestamps,
                        y: historicalData.values.map(v => v + verticalOffset),
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#3B82F6' },
                        name: 'Historical Data'
                      }
                    ]}
                    layout={{
                      width: 500,
                      height: 400,
                      title: 'Historical Data',
                      xaxis: { title: 'Time' },
                      yaxis: { title: 'Value' },
                      margin: { l: 50, r: 50, t: 50, b: 50 }
                    }}
                    config={{ responsive: true }}
                  />
                </React.Suspense>
              )}
            </div>
            
            {/* Vertical Slider */}
            <div className="flex flex-col justify-center items-center px-4">
              <label className="text-sm font-medium text-gray-700 mb-2 transform -rotate-90">
                Vertical Pan
              </label>
              <input
                type="range"
                min="-100"
                max="100"
                value={verticalOffset}
                onChange={(e) => setVerticalOffset(Number(e.target.value))}
                className="w-32 transform -rotate-90"
                style={{ writingMode: 'bt-lr' }}
              />
            </div>
            
            {/* Predictions Graph */}
            <div className="flex-1">
              <h3 className="text-lg font-semibold mb-4 text-center">
                Predictions {isPredicting && <span className="text-green-600">(Live)</span>}
              </h3>
              {predictionData && (
                <React.Suspense fallback={<div className="text-center py-8">Loading predictions...</div>}>
                  <Plot
                    data={[
                      {
                        x: predictionData.timestamps,
                        y: predictionData.predictions.map(p => p + verticalOffset * 0.1),
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#10B981' },
                        name: 'Predictions'
                      },
                      ...(predictionData.confidence_intervals ? [{
                        x: predictionData.timestamps,
                        y: predictionData.confidence_intervals.map(ci => ci.upper + verticalOffset * 0.1),
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: 'rgba(16, 185, 129, 0.3)' },
                        name: 'Upper Confidence',
                        showlegend: false
                      }, {
                        x: predictionData.timestamps,
                        y: predictionData.confidence_intervals.map(ci => ci.lower + verticalOffset * 0.1),
                        type: 'scatter',
                        mode: 'lines',
                        fill: 'tonexty',
                        fillcolor: 'rgba(16, 185, 129, 0.1)',
                        line: { color: 'rgba(16, 185, 129, 0.3)' },
                        name: 'Lower Confidence',
                        showlegend: false
                      }] : [])
                    ]}
                    layout={{
                      width: 500,
                      height: 400,
                      title: 'Predictions',
                      xaxis: { title: 'Time' },
                      yaxis: { title: 'Value' },
                      margin: { l: 50, r: 50, t: 50, b: 50 }
                    }}
                    config={{ responsive: true }}
                  />
                </React.Suspense>
              )}
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