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
  
  // Slider-responsive graph data for right panel
  const [sliderGraphData, setSliderGraphData] = useState([]);
  const [lastSliderChange, setLastSliderChange] = useState(Date.now());
  const [previousPh, setPreviousPh] = useState(7.6); // Track previous pH for smooth transitions
  const [isSliderActive, setIsSliderActive] = useState(false); // Track if slider is being actively moved
  const [sliderTransitionSpeed, setSliderTransitionSpeed] = useState(200); // Dynamic transition speed
  
  // Enhanced continuous prediction updates with visual buffering for smooth transitions
  const [predictionBuffer, setPredictionBuffer] = useState([]);
  const [smoothTransition, setSmoothTransition] = useState(true);

  // Buffer and smooth prediction updates
  const updatePredictionsSmooth = (newPredictions) => {
    if (!newPredictions || !newPredictions.predictions) return;
    
    setLstmPredictions(prev => {
      // Smooth transition between old and new predictions
      const updated = [...prev, ...newPredictions.predictions];
      const windowed = updated.slice(-timeWindow);
      
      // Apply additional frontend smoothing for visual continuity
      if (smoothTransition && prev.length > 0 && windowed.length > 5) {
        // Create smooth transition between the last few old points and new points
        const transitionZone = Math.min(3, Math.floor(windowed.length * 0.1));
        for (let i = 0; i < transitionZone && i < windowed.length - 1; i++) {
          const weight = (i + 1) / (transitionZone + 1);
          const idx = windowed.length - transitionZone + i;
          if (idx > 0) {
            windowed[idx] = windowed[idx-1] * (1 - weight) + windowed[idx] * weight;
          }
        }
      }
      
      return windowed;
    });
  };
  
  // Enhanced ML state
  const [supportedModels, setSupportedModels] = useState([]);
  const [modelPerformance, setModelPerformance] = useState(null);
  const [dataQuality, setDataQuality] = useState(null);
  const [isAdvancedMode, setIsAdvancedMode] = useState(false);
  const [modelComparison, setModelComparison] = useState(null);
  const [isOptimizing, setIsOptimizing] = useState(false);
  
  // Toast notification state
  const [toast, setToast] = useState({ show: false, message: '', type: 'info' });
  
  // Toast notification function
  const showToast = (message, type = 'info') => {
    setToast({ show: true, message, type });
    setTimeout(() => setToast({ show: false, message: '', type: 'info' }), 5000);
  };

  // Initialize test mode for slider functionality demonstration
  const initializeTestMode = () => {
    console.log('🚀 Starting test mode initialization...');
    
    try {
      // Set up test data to show the dashboard
      setCurrentStep('prediction'); // Go directly to prediction step (string, not number)
      setModelId('test-model');
      
      // Initialize with basic target pH
      setTargetPh(7.6);
      setPhData({ current_ph: 7.0, target_ph: 7.6, status: 'Connected' });
      
      // Initialize slider graph with test data
      setTimeout(() => {
        const testSliderData = generateSliderGraphData(7.6, 30);
        setSliderGraphData(testSliderData);
        console.log('📊 Slider graph data initialized:', testSliderData.length, 'points');
      }, 100);
      
      // Set some basic historical data for left panel
      const testHistoricalData = Array.from({ length: 24 }, (_, i) => 7.0 + Math.sin(i * 0.5) * 0.3);
      setHistoricalData({ values: testHistoricalData });
      
      console.log('✅ Test mode initialized - Dashboard ready');
    } catch (error) {
      console.error('❌ Error in test mode initialization:', error);
    }
  };

  // Initialize test mode on component mount for development/testing
  useEffect(() => {
    // Check if we want to enable test mode (could be controlled by URL parameter or environment)
    const urlParams = new URLSearchParams(window.location.search);
    const testMode = urlParams.get('test') === 'true' || window.location.hostname === 'localhost';
    
    if (testMode) {
      initializeTestMode();
    }
  }, []);

  // File upload handler
  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    const supportedTypes = ['text/csv', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel'];
    if (!supportedTypes.includes(file.type) && !file.name.match(/\.(csv|xlsx|xls)$/i)) {
      showToast('Unsupported file type. Please upload CSV (.csv) or Excel (.xlsx, .xls) files only.', 'error');
      return;
    }

    // Validate file size (limit to 50MB)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      showToast('File too large. Please upload files smaller than 50MB.', 'error');
      return;
    }

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
        showToast('File uploaded successfully!', 'success');
      } else {
        // Get detailed error message from response
        let errorMessage = 'Error uploading file';
        try {
          const errorData = await response.json();
          if (errorData.detail) {
            errorMessage = `Upload failed: ${errorData.detail}`;
          } else if (errorData.error) {
            errorMessage = `Upload failed: ${errorData.error}`;
          } else if (errorData.message) {
            errorMessage = `Upload failed: ${errorData.message}`;
          }
        } catch (e) {
          // If we can't parse JSON, show HTTP status
          errorMessage = `Upload failed: HTTP ${response.status} - ${response.statusText}`;
        }
        
        showToast(errorMessage, 'error');
        console.error('Upload failed:', {
          status: response.status,
          statusText: response.statusText,
          file: file.name,
          fileSize: file.size,
          fileType: file.type
        });
      }
    } catch (error) {
      console.error('Upload error:', error);
      
      // Provide specific error messages based on error type
      let errorMessage = 'Error uploading file';
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage = 'Network error: Unable to connect to server. Please check your internet connection.';
      } else if (error.name === 'TypeError' && error.message.includes('JSON')) {
        errorMessage = 'Server response error: Invalid response format.';
      } else if (error.message.includes('timeout')) {
        errorMessage = 'Upload timeout: File upload took too long. Please try with a smaller file.';
      } else if (error.message.includes('abort')) {
        errorMessage = 'Upload cancelled: File upload was interrupted.';
      } else {
        errorMessage = `Upload failed: ${error.message}`;
      }
      
      showToast(errorMessage, 'error');
    }
  }, [showToast]);

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

  // Load enhanced pattern analysis
  const loadEnhancedPatternAnalysis = async () => {
    try {
      const response = await fetch(`${API}/enhanced-pattern-analysis`);
      if (response.ok) {
        const result = await response.json();
        console.log('Enhanced pattern analysis:', result);
        
        // Display pattern insights in console for debugging
        if (result.recommendations && result.recommendations.insights) {
          console.log('Pattern Analysis Insights:', result.recommendations.insights);
        }
        
        return result;
      }
    } catch (error) {
      console.error('Error loading enhanced pattern analysis:', error);
    }
    return null;
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
        showToast(`Hyperparameter optimization completed! Best parameters: ${JSON.stringify(result.best_parameters)}, Best score: ${result.best_score.toFixed(4)}`, 'success');
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
      showToast('Missing required fields: Please select both time and target columns', 'error');
      return;
    }

    if (parameters.time_column === parameters.target_column) {
      showToast('Invalid column selection: Time column and target column cannot be the same. Please select different columns.', 'error');
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
          showToast(`Model trained successfully! Performance Grade: ${result.evaluation_grade}, RMSE: ${metrics.rmse?.toFixed(4) || 'N/A'}, MAE: ${metrics.mae?.toFixed(4) || 'N/A'}, R²: ${metrics.r2?.toFixed(4) || 'N/A'}`, 'success');
        } else {
          showToast('Model trained successfully!', 'success');
        }
        
        // Load historical data
        await loadHistoricalData();
      } else {
        // Get detailed error message from response
        let errorMessage = 'Training failed';
        try {
          const errorData = await response.json();
          if (errorData.detail) {
            errorMessage = `Training failed: ${errorData.detail}`;
          } else if (errorData.error) {
            errorMessage = `Training failed: ${errorData.error}`;
          } else if (errorData.message) {
            errorMessage = `Training failed: ${errorData.message}`;
          }
        } catch (e) {
          errorMessage = `Training failed: HTTP ${response.status} - ${response.statusText}`;
        }
        
        showToast(errorMessage, 'error');
        console.error('Training failed:', {
          status: response.status,
          statusText: response.statusText,
          modelType: parameters.model_type,
          dataId: uploadedData.data_id
        });
      }
    } catch (error) {
      console.error('Training error:', error);
      
      // Provide specific error messages based on error type
      let errorMessage = 'Training failed';
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage = 'Network error: Unable to connect to server. Please check your internet connection.';
      } else if (error.name === 'TypeError' && error.message.includes('JSON')) {
        errorMessage = 'Server response error: Invalid response format.';
      } else if (error.message.includes('timeout')) {
        errorMessage = 'Training timeout: Model training took too long. Please try with a smaller dataset.';
      } else if (error.message.includes('abort')) {
        errorMessage = 'Training cancelled: Model training was interrupted.';
      } else {
        errorMessage = `Training failed: ${error.message}`;
      }
      
      showToast(errorMessage, 'error');
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

  // Generate slider-responsive graph data based on targetPh and historical pattern
  const generateSliderGraphData = (targetPhValue, numPoints = 30) => {
    const currentTime = Date.now();
    const timeSinceLastChange = currentTime - lastSliderChange;
    
    // Extract pattern from historical data if available
    const getHistoricalPattern = () => {
      if (!historicalData || !historicalData.values || historicalData.values.length === 0) {
        // Default sine wave pattern if no historical data
        return Array.from({ length: numPoints }, (_, i) => Math.sin(i * 0.2) * 0.5);
      }
      
      const histValues = historicalData.values;
      const histMean = histValues.reduce((sum, val) => sum + val, 0) / histValues.length;
      
      // Extract the pattern (deviation from mean) and normalize it
      const pattern = histValues.map(val => val - histMean);
      const maxDeviation = Math.max(...pattern.map(Math.abs));
      const normalizedPattern = pattern.map(p => maxDeviation > 0 ? p / maxDeviation : 0);
      
      // Interpolate/resample to match numPoints
      if (normalizedPattern.length === numPoints) {
        return normalizedPattern;
      }
      
      const resampledPattern = [];
      for (let i = 0; i < numPoints; i++) {
        const sourceIndex = (i / (numPoints - 1)) * (normalizedPattern.length - 1);
        const lowerIndex = Math.floor(sourceIndex);
        const upperIndex = Math.min(lowerIndex + 1, normalizedPattern.length - 1);
        const fraction = sourceIndex - lowerIndex;
        
        const interpolatedValue = normalizedPattern[lowerIndex] * (1 - fraction) + 
                                 normalizedPattern[upperIndex] * fraction;
        resampledPattern.push(interpolatedValue);
      }
      
      return resampledPattern;
    };
    
    const basePattern = getHistoricalPattern();
    
    // If slider hasn't moved recently (more than 3 seconds), show stable pattern at target pH
    if (timeSinceLastChange > 3000) {
      // Apply the historical pattern scaled to the target pH level
      const patternAmplitude = 0.3; // Reduced amplitude for stable state
      return basePattern.map(patternValue => targetPhValue + (patternValue * patternAmplitude));
    }
    
    // If slider is actively moving, create dynamic transition toward target pH with pattern
    const graphData = [];
    const startPh = previousPh;
    
    for (let i = 0; i < numPoints; i++) {
      const progress = i / (numPoints - 1); // 0 to 1
      const patternValue = basePattern[i] || 0;
      
      if (timeSinceLastChange < 1000) {
        // Very recent change - show dramatic movement toward target pH following pattern
        const phDifference = targetPhValue - startPh;
        
        // Base transition value
        let baseValue = startPh + (phDifference * Math.pow(progress, 0.7));
        
        // Apply historical pattern with increasing amplitude as we approach target
        const patternAmplitude = 0.6 * progress; // Pattern becomes stronger as we progress
        const patternContribution = patternValue * patternAmplitude;
        
        let value = baseValue + patternContribution;
        
        // Add some transitional variation that decreases as we approach target
        const variationIntensity = (1 - progress) * 0.15;
        value += Math.sin(i * 0.25) * variationIntensity;
        
        // Ensure final 20% of points smoothly approach target with full pattern
        if (i > numPoints * 0.8) {
          const finalProgress = (i - numPoints * 0.8) / (numPoints * 0.2);
          const targetWithPattern = targetPhValue + (patternValue * 0.4);
          value = value * (1 - finalProgress) + targetWithPattern * finalProgress;
        }
        
        graphData.push(Math.max(0, Math.min(14, value)));
      } else {
        // Recent change (1-3 seconds) - show gradual convergence to pattern at target pH
        const convergenceFactor = (timeSinceLastChange - 1000) / 2000; // 0 to 1 over 2 seconds
        
        // Base value at target pH
        const baseValue = targetPhValue;
        
        // Apply historical pattern with decreasing random variation
        const patternAmplitude = 0.4 + (0.2 * (1 - convergenceFactor)); // 0.4 to 0.6 amplitude
        const patternContribution = patternValue * patternAmplitude;
        
        // Add diminishing random oscillation
        const oscillation = Math.sin(i * 0.3) * 0.1 * (1 - convergenceFactor);
        const randomNoise = (Math.random() - 0.5) * 0.05 * (1 - convergenceFactor);
        
        const value = baseValue + patternContribution + oscillation + randomNoise;
        graphData.push(Math.max(0, Math.min(14, value)));
      }
    }
    
    return graphData;
  };

  // Generate continuous predictions with proper extrapolation
  const generateContinuousPredictions = async () => {
    if (!modelId) {
      console.error('No model available for continuous predictions');
      return null;
    }

    try {
      // Try new enhanced real-time prediction first (highest priority)
      const enhancedRealtimeResponse = await fetch(`${API}/generate-enhanced-realtime-prediction?steps=30&time_window=${timeWindow}&maintain_patterns=true`);
      if (enhancedRealtimeResponse.ok) {
        const enhancedRealtimeData = await enhancedRealtimeResponse.json();
        console.log('Enhanced real-time prediction result:', enhancedRealtimeData);
        console.log('Pattern following score:', enhancedRealtimeData.metadata?.pattern_following_score);
        console.log('Continuity score:', enhancedRealtimeData.metadata?.continuity_score);
        return enhancedRealtimeData;
      }
      
      // Try advanced pH prediction as fallback
      const advancedResponse = await fetch(`${API}/generate-advanced-ph-prediction?steps=30&maintain_patterns=true`);
      if (advancedResponse.ok) {
        const advancedData = await advancedResponse.json();
        console.log('Advanced pH prediction result:', advancedData);
        return advancedData;
      }
      
      // Try enhanced prediction as fallback
      const enhancedResponse = await fetch(`${API}/generate-enhanced-continuous-prediction?model_id=${modelId}&steps=30&time_window=${timeWindow}`);
      if (enhancedResponse.ok) {
        const enhancedData = await enhancedResponse.json();
        console.log('Enhanced prediction result:', enhancedData);
        return enhancedData;
      }
      
      // Fallback to standard prediction
      const response = await fetch(`${API}/generate-continuous-prediction?model_id=${modelId}&steps=30&time_window=${timeWindow}`);
      if (response.ok) {
        const data = await response.json();
        return data;
      } else {
        // Log error details but don't alert for continuous predictions (they run in background)
        console.error('Continuous prediction failed:', {
          status: response.status,
          statusText: response.statusText,
          modelId: modelId,
          timeWindow: timeWindow
        });
        
        // Try to get error details
        try {
          const errorData = await response.json();
          console.error('Error details:', errorData);
        } catch (e) {
          console.error('Could not parse error response');
        }
      }
    } catch (error) {
      console.error('Error generating continuous predictions:', error);
      
      // Log additional details for debugging
      console.error('Continuous prediction error details:', {
        errorName: error.name,
        errorMessage: error.message,
        modelId: modelId,
        timeWindow: timeWindow
      });
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
    if (!modelId) {
      showToast('No model available: Please train a model first before generating predictions.', 'error');
      return null;
    }

    try {
      // Try new advanced pH prediction first
      const advancedResponse = await fetch(`${API}/generate-advanced-ph-prediction?steps=30&maintain_patterns=true`);
      if (advancedResponse.ok) {
        const advancedData = await advancedResponse.json();
        console.log('Advanced pH prediction result:', advancedData);
        return advancedData;
      }
      
      // Fallback to standard prediction
      const response = await fetch(`${API}/generate-prediction?model_id=${modelId}&steps=30&offset=${window}`);
      if (response.ok) {
        const data = await response.json();
        return data;
      } else {
        // Get detailed error message from response
        let errorMessage = 'Prediction generation failed';
        try {
          const errorData = await response.json();
          if (errorData.detail) {
            errorMessage = `Prediction failed: ${errorData.detail}`;
          } else if (errorData.error) {
            errorMessage = `Prediction failed: ${errorData.error}`;
          } else if (errorData.message) {
            errorMessage = `Prediction failed: ${errorData.message}`;
          }
        } catch (e) {
          errorMessage = `Prediction failed: HTTP ${response.status} - ${response.statusText}`;
        }
        
        showToast(errorMessage, 'error');
        console.error('Prediction generation failed:', {
          status: response.status,
          statusText: response.statusText,
          modelId: modelId,
          window: window
        });
      }
    } catch (error) {
      console.error('Error generating predictions:', error);
      
      // Provide specific error messages based on error type
      let errorMessage = 'Prediction generation failed';
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage = 'Network error: Unable to connect to server. Please check your internet connection.';
      } else if (error.name === 'TypeError' && error.message.includes('JSON')) {
        errorMessage = 'Server response error: Invalid response format.';
      } else if (error.message.includes('timeout')) {
        errorMessage = 'Prediction timeout: Generation took too long. Please try again.';
      } else if (error.message.includes('abort')) {
        errorMessage = 'Prediction cancelled: Generation was interrupted.';
      } else {
        errorMessage = `Prediction failed: ${error.message}`;
      }
      
      showToast(errorMessage, 'error');
    }
    return null;
  };

  // Update predictions based on time window
  const updatePredictionsWithWindow = async () => {
    if (isPredicting) {
      const newPredictions = await generateContinuousPredictions();
      if (newPredictions && newPredictions.predictions) {
        setPredictionData(newPredictions);
        updatePredictionsSmooth(newPredictions);
      }
    }
  };

  // Update predictions when time window changes
  useEffect(() => {
    if (isPredicting && predictionData) {
      updatePredictionsWithWindow();
    }
  }, [timeWindow]);

  // Update slider graph data when targetPh changes
  useEffect(() => {
    // Update previous pH before changing
    setPreviousPh(prev => prev); // Keep current previous value for this update
    
    setLastSliderChange(Date.now());
    
    // Immediate update when slider moves
    const updateSliderGraph = () => {
      const newGraphData = generateSliderGraphData(targetPh, timeWindow);
      setSliderGraphData(newGraphData);
    };
    
    updateSliderGraph();
    
    // Set up interval to continuously update the graph for smooth transition
    const interval = setInterval(() => {
      updateSliderGraph();
    }, 200); // Update every 200ms for smooth animation
    
    // Clear interval after 5 seconds (when graph should be settled)
    const timeout = setTimeout(() => {
      clearInterval(interval);
      // Final update to ensure horizontal line
      const finalData = generateSliderGraphData(targetPh, timeWindow);
      setSliderGraphData(finalData);
      // Update previous pH to current target for next transition
      setPreviousPh(targetPh);
    }, 5000);
    
    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
  }, [targetPh, timeWindow]);

  // Enhanced smooth chart component for pH monitoring dashboard with advanced noise reduction
  const PhChart = ({ data, title, color = '#3B82F6', showAnimation = false, currentValue = null, smoothLevel = 'high', useFixedScale = false }) => {
    const canvasRef = React.useRef(null);
    const [animationFrame, setAnimationFrame] = React.useState(0);
    const [visualBuffer, setVisualBuffer] = React.useState([]);
    
    // Frontend visual smoothing function using interpolation
    const applyVisualSmoothing = (values, level = 'high') => {
      if (!values || values.length < 3) return values;
      
      const smoothed = [...values];
      
      // Apply different smoothing levels
      switch (level) {
        case 'high':
          // Apply multiple passes of smoothing for maximum smoothness
          for (let pass = 0; pass < 2; pass++) {
            for (let i = 1; i < smoothed.length - 1; i++) {
              smoothed[i] = (smoothed[i-1] * 0.25 + smoothed[i] * 0.5 + smoothed[i+1] * 0.25);
            }
          }
          break;
        case 'medium':
          // Single pass smoothing
          for (let i = 1; i < smoothed.length - 1; i++) {
            smoothed[i] = (smoothed[i-1] * 0.2 + smoothed[i] * 0.6 + smoothed[i+1] * 0.2);
          }
          break;
        case 'low':
          // Light smoothing
          for (let i = 1; i < smoothed.length - 1; i++) {
            smoothed[i] = (smoothed[i-1] * 0.1 + smoothed[i] * 0.8 + smoothed[i+1] * 0.1);
          }
          break;
      }
      
      return smoothed;
    };

    // Create interpolated points for smooth curves
    const createInterpolatedCurve = (points) => {
      if (points.length < 2) return points;
      
      const interpolated = [];
      
      for (let i = 0; i < points.length - 1; i++) {
        interpolated.push(points[i]);
        
        // Add interpolated points between existing points
        const steps = 3; // Number of interpolation steps
        for (let j = 1; j < steps; j++) {
          const t = j / steps;
          const interpolatedPoint = {
            x: points[i].x + t * (points[i + 1].x - points[i].x),
            y: points[i].y + t * (points[i + 1].y - points[i].y)
          };
          interpolated.push(interpolatedPoint);
        }
      }
      
      interpolated.push(points[points.length - 1]);
      return interpolated;
    };

    // Draw smooth cubic Bezier curve
    const drawSmoothCurve = (ctx, points, tension = 0.3) => {
      if (points.length < 2) return;
      
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      
      for (let i = 0; i < points.length - 1; i++) {
        const cp1x = points[i].x + (points[i + 1].x - points[i].x) * tension;
        const cp1y = points[i].y + (points[i + 1].y - points[i].y) * tension;
        
        const cp2x = points[i + 1].x - (points[i + 1].x - points[i].x) * tension;
        const cp2y = points[i + 1].y - (points[i + 1].y - points[i].y) * tension;
        
        if (i === 0) {
          ctx.quadraticCurveTo(cp1x, cp1y, points[i + 1].x, points[i + 1].y);
        } else {
          ctx.bezierCurveTo(
            cp1x, cp1y,
            cp2x, cp2y,
            points[i + 1].x, points[i + 1].y
          );
        }
      }
    };

    // Draw smooth area fill
    const drawSmoothAreaFill = (ctx, points, color, alpha = 0.3) => {
      if (points.length < 2) return;
      
      const hexToRgba = (hex, alpha = 0.3) => {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
      };
      
      ctx.fillStyle = hexToRgba(color, alpha);
      ctx.beginPath();
      ctx.moveTo(points[0].x, ctx.canvas.height - 70); // Updated for new margin
      
      // Draw smooth curve to create the top boundary
      ctx.lineTo(points[0].x, points[0].y);
      drawSmoothCurve(ctx, points, 0.2);
      
      // Close the path along the bottom
      ctx.lineTo(points[points.length - 1].x, ctx.canvas.height - 70); // Updated for new margin
      ctx.closePath();
      ctx.fill();
    };
    
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
      
      // Draw grid lines (adjusted for new margins)
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      for (let i = 0; i <= 10; i++) {
        const x = 70 + (i / 10) * (width - 100);
        const y = 30 + (i / 10) * (height - 100);
        ctx.beginPath();
        ctx.moveTo(x, 30);
        ctx.lineTo(x, height - 70);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(70, y);
        ctx.lineTo(width - 30, y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      
      // Draw axes
      ctx.strokeStyle = '#374151';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(70, height - 70);
      ctx.lineTo(width - 30, height - 70);
      ctx.moveTo(70, 30);
      ctx.lineTo(70, height - 70);
      ctx.stroke();
      
      // Add axis labels and titles
      ctx.fillStyle = '#374151';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      
      // Y-axis labels (pH values)
      ctx.textAlign = 'right';
      for (let i = 0; i <= 14; i += 2) {
        const y = height - 70 - (i / 14) * (height - 100);
        ctx.fillText(i.toString(), 65, y + 4);
      }
      
      // Y-axis title
      ctx.save();
      ctx.translate(20, height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.font = 'bold 14px Arial';
      ctx.fillText('pH Level', 0, 0);
      ctx.restore();
      
      // X-axis labels (time points or data points)
      ctx.textAlign = 'center';
      const numXLabels = 5;
      for (let i = 0; i < numXLabels; i++) {
        const x = 70 + (i / (numXLabels - 1)) * (width - 100);
        const labelValue = useFixedScale ? 
          `T${Math.round((i / (numXLabels - 1)) * 30)}` : // Time points for slider graph
          `${Math.round((i / (numXLabels - 1)) * 24)}h`; // Hours for historical data
        ctx.fillText(labelValue, x, height - 50);
      }
      
      // X-axis title
      ctx.textAlign = 'center';
      ctx.font = 'bold 14px Arial';
      const xAxisTitle = useFixedScale ? 'Time Points' : 'Time (Hours)';
      ctx.fillText(xAxisTitle, width / 2, height - 25);
      
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
      
      console.log('Enhanced PhChart rendering values:', values.length, 'data points:', values.slice(0, 5));
      
      if (values && values.length > 0) {
        // Apply frontend visual smoothing
        const smoothedValues = applyVisualSmoothing(values, smoothLevel);
        
        // Use fixed pH scale (0-14) for slider-responsive graph, dynamic scale for others
        let maxVal, minVal, range;
        if (useFixedScale) {
          // Fixed pH scale from 0 to 14
          minVal = 0;
          maxVal = 14;
          range = 14;
        } else {
          // Dynamic scale based on data
          maxVal = Math.max(...smoothedValues);
          minVal = Math.min(...smoothedValues);
          range = maxVal - minVal || 1;
        }
        
        console.log('Enhanced PhChart range:', { minVal, maxVal, range, smoothLevel, useFixedScale });
        
        // Convert values to coordinate points (adjusted for new margins)
        const points = smoothedValues.map((value, index) => ({
          x: 70 + (index / Math.max(smoothedValues.length - 1, 1)) * (width - 100),
          y: height - 70 - ((value - minVal) / range) * (height - 100)
        }));
        
        // Create interpolated points for extra smoothness
        const interpolatedPoints = createInterpolatedCurve(points);
        
        // Draw smooth area fill
        drawSmoothAreaFill(ctx, interpolatedPoints, color, 0.3);
        
        // Draw smooth curve line
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Enable antialiasing for smoother lines
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        
        // Draw the main smooth curve
        drawSmoothCurve(ctx, points, 0.3);
        ctx.stroke();
        
        // Draw data points (less prominent for smoother look)
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.7;
        points.forEach((point, index) => {
          // Only show every few points to reduce visual noise
          if (index % 2 === 0 || index === points.length - 1) {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
            ctx.fill();
          }
        });
        ctx.globalAlpha = 1.0;
        
        // Enhanced animation effect for live data with smooth pulsing
        if (showAnimation && isPredicting && points.length > 0) {
          const pulseSize = 4 + Math.sin(animationFrame * 0.08) * 2;
          const lastPoint = points[points.length - 1];
          
          // Glowing effect
          ctx.shadowColor = '#10B981';
          ctx.shadowBlur = 10;
          ctx.strokeStyle = '#10B981';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(lastPoint.x, lastPoint.y, pulseSize, 0, 2 * Math.PI);
          ctx.stroke();
          
          // Reset shadow
          ctx.shadowBlur = 0;
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
      
      // Draw smoothing level indicator
      if (smoothLevel !== 'medium') {
        ctx.fillStyle = '#6B7280';
        ctx.font = '10px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Smooth: ${smoothLevel}`, 60, height - 15);
      }
      
    }, [data, title, color, showAnimation, animationFrame, isPredicting, currentValue, smoothLevel]);
    
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
            // Use the new advanced pH prediction extension endpoint for smoother continuous prediction
            const extensionResponse = await fetch(`${API}/extend-advanced-ph-prediction?additional_steps=5`);
            if (extensionResponse.ok) {
              const extensionData = await extensionResponse.json();
              
              // Update predictions with smooth extension
              updatePredictionsSmooth(extensionData);
              
              // Update prediction data for display
              setPredictionData(extensionData);
            } else {
              // Fallback to standard extension
              const fallbackResponse = await fetch(`${API}/extend-prediction?steps=5`);
              if (fallbackResponse.ok) {
                const fallbackData = await fallbackResponse.json();
                
                updatePredictionsSmooth(fallbackData);
                
                setPredictionData(fallbackData);
              }
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
                  smoothLevel="medium"
                />
              ) : realtimePhReadings.length > 0 ? (
                <PhChart 
                  data={realtimePhReadings.map(r => r.ph_value)} 
                  title="Real-time pH Readings"
                  color="#3B82F6"
                  showAnimation={isPredicting}
                  currentValue={phData.current_ph}
                  smoothLevel="medium"
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
                  // Load enhanced pattern analysis first
                  await loadEnhancedPatternAnalysis();
                  
                  // Load historical data
                  await loadHistoricalData();
                  
                  // Then generate predictions
                  const predictions = await generatePredictions(0);
                  if (predictions && predictions.predictions) {
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
                  onClick={async () => {
                    // Load pattern analysis before starting continuous prediction
                    await loadEnhancedPatternAnalysis();
                    startContinuousPrediction();
                  }}
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

            {/* Visual Smoothing Control */}
            <div className="mt-4">
              <label className="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={smoothTransition}
                  onChange={(e) => setSmoothTransition(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm text-gray-700">
                  🎛️ Enhanced Visual Smoothing (reduces noise in graph lines)
                </span>
              </label>
            </div>
          </div>

          {/* Right Panel - Slider-Responsive Graph */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-800">Slider-Responsive Graph</h2>
              <span className="text-sm text-gray-600">Follows pH Slider</span>
            </div>
            
            <div className="mb-4">
              {sliderGraphData.length > 0 ? (
                <PhChart 
                  data={sliderGraphData} 
                  title={`pH Level: ${targetPh.toFixed(1)}`}
                  color="#10B981"
                  showAnimation={true}
                  smoothLevel="high"
                  useFixedScale={true}
                />
              ) : lstmPredictions.length > 0 ? (
                <PhChart 
                  data={lstmPredictions.slice(-timeWindow)} 
                  title="LSTM Predictions"
                  color="#10B981"
                  showAnimation={isPredicting}
                  smoothLevel="high"
                />
              ) : predictionData && predictionData.predictions && predictionData.predictions.length > 0 ? (
                <PhChart 
                  data={predictionData.predictions.slice(-timeWindow)} 
                  title="Generated Predictions"
                  color="#10B981"
                  showAnimation={false}
                  smoothLevel="high"
                />
              ) : (
                <div className="border border-gray-300 rounded h-64 flex items-center justify-center text-gray-500">
                  <div className="text-center">
                    <div className="text-6xl mb-2">🎯</div>
                    <p>Move the pH slider to see responsive graph</p>
                  </div>
                </div>
              )}
            </div>

            <div className="text-center">
              <div className="text-sm text-gray-600 mb-2">Current Target pH</div>
              <div className="text-2xl font-bold text-green-600">
                {targetPh.toFixed(1)}
              </div>
            </div>

            {/* Status Indicator */}
            <div className="mt-4 text-center">
              <div className="inline-flex items-center px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                Graph Synced with Slider
              </div>
            </div>
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
      
      {/* Toast Notification */}
      {toast.show && (
        <div className={`fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg transition-all duration-300 ${
          toast.type === 'success' 
            ? 'bg-green-100 border border-green-400 text-green-700' 
            : toast.type === 'error' 
            ? 'bg-red-100 border border-red-400 text-red-700' 
            : 'bg-blue-100 border border-blue-400 text-blue-700'
        }`}>
          <div className="flex items-center">
            <div className="mr-3">
              {toast.type === 'success' && <div className="text-2xl">✅</div>}
              {toast.type === 'error' && <div className="text-2xl">❌</div>}
              {toast.type === 'info' && <div className="text-2xl">ℹ️</div>}
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium">{toast.message}</p>
            </div>
            <button
              onClick={() => setToast({ show: false, message: '', type: 'info' })}
              className="ml-4 text-gray-400 hover:text-gray-600"
            >
              <div className="text-lg">×</div>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;