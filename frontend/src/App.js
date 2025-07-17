import React, { useState, useEffect, useCallback, useRef } from 'react';
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
  const isPredictingRef = useRef(false);
  const [timeWindow, setTimeWindow] = useState(50);
  const [continuousPredictions, setContinuousPredictions] = useState([]);
  const [websocket, setWebsocket] = useState(null);
  const [phData, setPhData] = useState({ current_ph: 7.0, target_ph: 7.6, status: 'Connected' });
  const [targetPh, setTargetPh] = useState(7.6);
  const [isAdjustingPh, setIsAdjustingPh] = useState(false);
  const [realtimePhReadings, setRealtimePhReadings] = useState([]);
  const [lstmPredictions, setLstmPredictions] = useState([]);
  const [predictionConfidence, setPredictionConfidence] = useState(67);
  
  // Enhanced ML state
  const [supportedModels, setSupportedModels] = useState([]);
  const [modelPerformance, setModelPerformance] = useState(null);
  const [dataQuality, setDataQuality] = useState(null);
  const [isAdvancedMode, setIsAdvancedMode] = useState(false);
  const [modelComparison, setModelComparison] = useState(null);
  const [isOptimizing, setIsOptimizing] = useState(false);

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

  // Load supported models
  const loadSupportedModels = async () => {
    try {
      const response = await fetch(`${API}/supported-models`);
      if (response.ok) {
        const result = await response.json();
        setSupportedModels(result);
      }
    } catch (error) {
      console.error('Error loading supported models:', error);
    }
  };

  // Load data quality report
  const loadDataQualityReport = async () => {
    try {
      const response = await fetch(`${API}/data-quality-report`);
      if (response.ok) {
        const result = await response.json();
        setDataQuality(result);
      }
    } catch (error) {
      console.error('Error loading data quality report:', error);
    }
  };

  // Load model performance
  const loadModelPerformance = async () => {
    try {
      const response = await fetch(`${API}/model-performance`);
      if (response.ok) {
        const result = await response.json();
        setModelPerformance(result);
      }
    } catch (error) {
      console.error('Error loading model performance:', error);
    }
  };

  // Compare models
  const compareModels = async () => {
    try {
      const response = await fetch(`${API}/model-comparison`);
      if (response.ok) {
        const result = await response.json();
        setModelComparison(result);
      }
    } catch (error) {
      console.error('Error comparing models:', error);
    }
  };

  // Optimize hyperparameters
  const optimizeHyperparameters = async (modelType) => {
    setIsOptimizing(true);
    try {
      const response = await fetch(`${API}/optimize-hyperparameters?model_type=${modelType}&n_trials=20`, {
        method: 'POST'
      });
      if (response.ok) {
        const result = await response.json();
        alert(`Hyperparameter optimization completed!\nBest parameters: ${JSON.stringify(result.best_parameters)}\nBest score: ${result.best_score.toFixed(4)}`);
      }
    } catch (error) {
      console.error('Error optimizing hyperparameters:', error);
    } finally {
      setIsOptimizing(false);
    }
  };

  // Load enhanced features on component mount
  useEffect(() => {
    loadSupportedModels();
  }, []);

  // Load quality report and performance after data upload
  useEffect(() => {
    if (uploadedData) {
      loadDataQualityReport();
    }
  }, [uploadedData]);

  // Load model performance after training
  useEffect(() => {
    if (modelId && currentStep === 'prediction') {
      loadModelPerformance();
    }
  }, [modelId, currentStep]);

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

    if (parameters.time_column === parameters.target_column) {
      alert('Time column and target column cannot be the same. Please select different columns.');
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
        
        // Show performance metrics for advanced models
        if (result.performance_metrics) {
          const metrics = result.performance_metrics;
          alert(`Model trained successfully!\nPerformance Grade: ${result.evaluation_grade}\nRMSE: ${metrics.rmse?.toFixed(4) || 'N/A'}\nMAE: ${metrics.mae?.toFixed(4) || 'N/A'}\nR²: ${metrics.r2?.toFixed(4) || 'N/A'}`);
        }
        
        // Load historical data
        await loadHistoricalData();
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        alert(`Error training model: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Training error:', error);
      alert(`Error training model: ${error.message || 'Network error'}`);
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

  // Handle pH target adjustment
  const handlePhTargetChange = async (newTarget) => {
    setTargetPh(newTarget);
    setIsAdjustingPh(true);
    
    try {
      // Update target pH on backend
      const response = await fetch(`${API}/set-ph-target`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ target_ph: newTarget }),
      });

      if (response.ok) {
        setPhData(prev => ({ ...prev, target_ph: newTarget }));
      }
    } catch (error) {
      console.error('Error setting pH target:', error);
    } finally {
      setIsAdjustingPh(false);
    }
  };

  // Generate continuous predictions with proper extrapolation
  const generateContinuousPredictions = async () => {
    if (!modelId) return null;

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
    if (!modelId) return null;

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
      if (newPredictions && newPredictions.predictions) {
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
      
      // Process data - handle multiple formats
      let values = [];
      if (Array.isArray(data)) {
        values = data;
      } else if (data && data.values) {
        values = data.values;
      } else if (data && data.predictions) {
        values = data.predictions;
      }
      
      // Ensure we have valid numeric values
      values = values.filter(v => v !== null && v !== undefined && !isNaN(v));
      
      console.log('PhChart rendering values:', values.length, 'data points:', values.slice(0, 5));
      
      if (values && values.length > 0) {
        const maxVal = Math.max(...values);
        const minVal = Math.min(...values);
        const range = maxVal - minVal || 1;
        
        console.log('PhChart range:', { minVal, maxVal, range });
        
        // Convert hex color to rgba for area fill
        const hexToRgba = (hex, alpha = 0.3) => {
          const r = parseInt(hex.slice(1, 3), 16);
          const g = parseInt(hex.slice(3, 5), 16);
          const b = parseInt(hex.slice(5, 7), 16);
          return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        };
        
        // Draw area fill
        ctx.fillStyle = hexToRgba(color, 0.3);
        ctx.beginPath();
        ctx.moveTo(50, height - 50);
        
        values.forEach((value, index) => {
          const x = 50 + (index / Math.max(values.length - 1, 1)) * (width - 100);
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
          const x = 50 + (index / Math.max(values.length - 1, 1)) * (width - 100);
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
          const x = 50 + (index / Math.max(values.length - 1, 1)) * (width - 100);
          const y = height - 50 - ((value - minVal) / range) * (height - 100);
          
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, 2 * Math.PI);
          ctx.fill();
        });
        
        // Show animation effect for live data
        if (showAnimation && isPredicting) {
          const pulseSize = 3 + Math.sin(animationFrame * 0.1) * 2;
          const lastIndex = values.length - 1;
          const x = 50 + (lastIndex / Math.max(values.length - 1, 1)) * (width - 100);
          const y = height - 50 - ((values[lastIndex] - minVal) / range) * (height - 100);
          
          ctx.strokeStyle = '#10B981';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, pulseSize, 0, 2 * Math.PI);
          ctx.stroke();
        }
      } else {
        // Draw "No Data" message
        ctx.fillStyle = '#9CA3AF';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No data available', width / 2, height / 2);
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
    isPredictingRef.current = true;
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
        setLstmPredictions(initialPredictions.predictions || []);
      }
      
      // Set up interval for continuous predictions that extrapolate
      const interval = setInterval(async () => {
        try {
          if (isPredictingRef.current) {
            // Use the new extend-prediction endpoint for smoother continuous prediction
            const extensionResponse = await fetch(`${API}/extend-prediction?steps=5`);
            if (extensionResponse.ok) {
              const extensionData = await extensionResponse.json();
              
              // Update predictions with smooth extension
              setLstmPredictions(prev => {
                const updated = [...prev, ...extensionData.predictions];
                return updated.slice(-timeWindow); // Keep within time window
              });
              
              // Update prediction data for display
              setPredictionData(extensionData);
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
      }, 2000); // Update every 2 seconds for smooth but not overwhelming updates
      
      setWebsocket(interval);
      
    } catch (error) {
      console.error('Error starting continuous prediction:', error);
    }
  };

  // Stop continuous prediction
  const stopContinuousPrediction = async () => {
    setIsPredicting(false);
    isPredictingRef.current = false;
    
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

  // Load pH simulation data when component mounts
  useEffect(() => {
    if (currentStep === 'prediction' && modelId) {
      loadPhSimulation();
    }
  }, [currentStep, modelId]);

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
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">
          ⚙️ Configure Model Parameters
        </h1>
        
        <div className="bg-white rounded-lg shadow-xl p-8">
          {/* Toggle for Advanced Mode */}
          <div className="mb-6">
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={isAdvancedMode}
                onChange={(e) => setIsAdvancedMode(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm font-medium text-gray-700">
                🚀 Enable Advanced ML Models (DLinear, N-BEATS, LSTM, Ensemble)
              </span>
            </label>
          </div>

          {/* Data Quality Report */}
          {dataQuality && (
            <div className="mb-6 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold text-blue-800 mb-2">📊 Data Quality Report</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="font-medium">Quality Score:</span>
                  <span className={`ml-2 px-2 py-1 rounded ${
                    dataQuality.validation_results.quality_score >= 80 ? 'bg-green-100 text-green-800' :
                    dataQuality.validation_results.quality_score >= 60 ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {dataQuality.validation_results.quality_score.toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="font-medium">Total Rows:</span>
                  <span className="ml-2">{dataQuality.validation_results.total_rows}</span>
                </div>
                <div>
                  <span className="font-medium">Missing Values:</span>
                  <span className="ml-2">{Object.values(dataQuality.validation_results.missing_values).reduce((a, b) => a + b, 0)}</span>
                </div>
              </div>
              {dataQuality.recommendations && dataQuality.recommendations.length > 0 && (
                <div className="mt-2">
                  <span className="font-medium text-blue-800">Recommendations:</span>
                  <ul className="list-disc list-inside text-sm text-blue-700 mt-1">
                    {dataQuality.recommendations.slice(0, 3).map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          
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
                <option value="prophet">Prophet (Traditional)</option>
                <option value="arima">ARIMA (Traditional)</option>
                {isAdvancedMode && supportedModels.advanced_models && supportedModels.advanced_models.map(model => (
                  <option key={model} value={model}>
                    {model.toUpperCase()} (Advanced)
                  </option>
                ))}
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

            {/* Advanced Parameters */}
            {isAdvancedMode && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Sequence Length
                  </label>
                  <input
                    type="number"
                    value={parameters.seq_len || 50}
                    onChange={(e) => setParameters({...parameters, seq_len: parseInt(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="10"
                    max="200"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Training Epochs
                  </label>
                  <input
                    type="number"
                    value={parameters.epochs || 100}
                    onChange={(e) => setParameters({...parameters, epochs: parseInt(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="10"
                    max="500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    step="0.001"
                    value={parameters.learning_rate || 0.001}
                    onChange={(e) => setParameters({...parameters, learning_rate: parseFloat(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="0.0001"
                    max="0.1"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Batch Size
                  </label>
                  <select
                    value={parameters.batch_size || 32}
                    onChange={(e) => setParameters({...parameters, batch_size: parseInt(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value={16}>16</option>
                    <option value={32}>32</option>
                    <option value={64}>64</option>
                  </select>
                </div>
              </>
            )}
          </div>
          
          {/* Advanced Actions */}
          {isAdvancedMode && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-800 mb-3">🔧 Advanced Actions</h3>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => optimizeHyperparameters(parameters.model_type)}
                  disabled={isOptimizing || !parameters.model_type}
                  className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 transition-colors text-sm"
                >
                  {isOptimizing ? '🔄 Optimizing...' : '⚡ Optimize Hyperparameters'}
                </button>
                
                <button
                  onClick={compareModels}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm"
                >
                  📊 Compare Models
                </button>
              </div>
            </div>
          )}
          
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

  // Render prediction step with three-panel pH monitoring dashboard layout
  const renderPredictionStep = () => (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-4 mb-6">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-800">pH Monitoring Dashboard</h1>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">Real-time pH sensor monitoring with LSTM predictions</span>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-sm font-medium text-green-700">Connected</span>
              </div>
            </div>
          </div>
        </div>

        {/* Three-panel layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Real-time pH Sensor Readings */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-800">Real-time pH Sensor Readings</h2>
              <span className="text-sm text-green-600 font-medium">Live</span>
            </div>
            
            <div className="mb-4">
              {historicalData && historicalData.values && historicalData.values.length > 0 ? (
                <PhChart 
                  data={historicalData.values.slice(-timeWindow)} 
                  title="Historical Data"
                  color="#3B82F6"
                  showAnimation={false}
                />
              ) : realtimePhReadings.length > 0 ? (
                <PhChart 
                  data={realtimePhReadings.map(r => r.ph_value)} 
                  title="Real-time pH Readings"
                  color="#3B82F6"
                  showAnimation={isPredicting}
                  currentValue={phData.current_ph}
                />
              ) : (
                <div className="border border-gray-300 rounded h-64 flex items-center justify-center text-gray-500">
                  <div className="text-center">
                    <div className="text-6xl mb-2">📊</div>
                    <p>No sensor data available</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Middle Panel - pH Control Panel */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-800">pH Control Panel</h2>
              <button
                onClick={() => setCurrentStep('parameters')}
                className="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
              >
                ← Back
              </button>
            </div>
            
            <div className="text-center mb-6">
              <div className="text-5xl font-bold text-blue-600 mb-2">
                {phData.current_ph.toFixed(2)}
              </div>
              <div className="text-sm text-gray-600 mb-1">Current pH Level</div>
              <div className="text-sm text-gray-500">Target pH: {phData.target_ph}</div>
            </div>

            {/* pH Control Slider */}
            <div className="mb-6">
              {/* Visual pH Scale */}
              <div className="flex justify-center mb-4">
                <div className="relative">
                  <div className="w-4 h-64 bg-gradient-to-t from-red-500 via-yellow-500 to-green-500 rounded-full"></div>
                  {/* Current pH Indicator */}
                  <div 
                    className="absolute w-6 h-6 bg-blue-600 rounded-full border-2 border-white shadow-lg transform -translate-x-1 transition-all duration-300"
                    style={{ 
                      top: `${((14 - phData.current_ph) / 14) * 100}%`,
                      marginTop: '-12px'
                    }}
                  ></div>
                  {/* Target pH Indicator */}
                  <div 
                    className="absolute w-4 h-4 bg-orange-500 rounded-full border-2 border-white shadow-md transform -translate-x-0.5 transition-all duration-300"
                    style={{ 
                      top: `${((14 - targetPh) / 14) * 100}%`,
                      marginTop: '-8px'
                    }}
                  ></div>
                </div>
              </div>
              
              {/* Interactive pH Slider */}
              <div className="flex justify-center mb-4">
                <div className="relative">
                  <input
                    type="range"
                    min="0"
                    max="14"
                    step="0.1"
                    value={targetPh}
                    onChange={(e) => {
                      const newTarget = parseFloat(e.target.value);
                      handlePhTargetChange(newTarget);
                    }}
                    className="w-64 h-4 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-vertical"
                    style={{
                      transform: 'rotate(-90deg)',
                      transformOrigin: 'center'
                    }}
                  />
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-xs text-gray-500 mb-2">pH Scale</div>
                <div className="flex justify-between text-xs text-gray-400 mb-2">
                  <span>0</span>
                  <span>7</span>
                  <span>14</span>
                </div>
                <div className="text-xs text-gray-600">
                  <span className="inline-block w-3 h-3 bg-blue-600 rounded-full mr-1"></span>
                  Current: {phData.current_ph.toFixed(2)}
                  <span className="mx-2">|</span>
                  <span className="inline-block w-3 h-3 bg-orange-500 rounded-full mr-1"></span>
                  Target: {targetPh.toFixed(1)}
                  {isAdjustingPh && <span className="ml-2 text-blue-500">Adjusting...</span>}
                </div>
              </div>
            </div>

            {/* Control Buttons */}
            <div className="space-y-3">
              <button
                onClick={async () => {
                  // Load historical data first
                  await loadHistoricalData();
                  
                  // Then generate predictions
                  const predictions = await generatePredictions(0);
                  if (predictions) {
                    setPredictionData(predictions);
                    setLstmPredictions(predictions.predictions);
                  }
                }}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                📈 Generate Predictions
              </button>
              
              {!isPredicting ? (
                <button
                  onClick={startContinuousPrediction}
                  className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  ▶️ Start Continuous Prediction
                </button>
              ) : (
                <button
                  onClick={stopContinuousPrediction}
                  className="w-full px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  ⏹️ Stop Continuous Prediction
                </button>
              )}
            </div>

            {/* Time Window Control */}
            <div className="mt-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Time Window Control
              </label>
              <input
                type="range"
                min="20"
                max="200"
                value={timeWindow}
                onChange={(e) => setTimeWindow(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>20 pts</span>
                <span>{timeWindow} pts</span>
                <span>200 pts</span>
              </div>
            </div>
          </div>

          {/* Right Panel - LSTM Predictions */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-800">LSTM Predictions</h2>
              <span className="text-sm text-gray-600">Next 30 Steps</span>
            </div>
            
            <div className="mb-4">
              {lstmPredictions.length > 0 ? (
                <PhChart 
                  data={lstmPredictions.slice(-timeWindow)} 
                  title="LSTM Predictions"
                  color="#10B981"
                  showAnimation={isPredicting}
                />
              ) : predictionData && predictionData.predictions && predictionData.predictions.length > 0 ? (
                <PhChart 
                  data={predictionData.predictions.slice(-timeWindow)} 
                  title="Generated Predictions"
                  color="#10B981"
                  showAnimation={false}
                />
              ) : (
                <div className="border border-gray-300 rounded h-64 flex items-center justify-center text-gray-500">
                  <div className="text-center">
                    <div className="text-6xl mb-2">🔮</div>
                    <p>Click "Generate Predictions" to start</p>
                  </div>
                </div>
              )}
            </div>

            <div className="text-center">
              <div className="text-sm text-gray-600 mb-2">Prediction Confidence</div>
              <div className="text-2xl font-bold text-green-600">
                {predictionConfidence}%
              </div>
            </div>

            {/* Status Indicator */}
            {isPredicting && (
              <div className="mt-4 text-center">
                <div className="inline-flex items-center px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                  Live Predictions Active
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Data Summary */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-sm text-gray-600 mb-1">Historical Data</div>
            <div className="text-lg font-semibold text-blue-600">
              {historicalData ? `${historicalData.values.length} points` : 'No data'}
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-sm text-gray-600 mb-1">Realtime Readings</div>
            <div className="text-lg font-semibold text-green-600">
              {realtimePhReadings.length} points
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-sm text-gray-600 mb-1">Predictions</div>
            <div className="text-lg font-semibold text-purple-600">
              {lstmPredictions.length} points
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-sm text-gray-600 mb-1">Time Window</div>
            <div className="text-lg font-semibold text-indigo-600">
              {timeWindow} points
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