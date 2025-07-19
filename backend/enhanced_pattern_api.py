"""
Advanced Pattern Learning API Endpoints
New endpoints for improved historical pattern learning and continuous prediction
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Create new API router for enhanced pattern learning endpoints
enhanced_api_router = APIRouter()

# Import the enhanced systems
from universal_pattern_learning import UniversalPatternLearning
from enhanced_pattern_aware_prediction import EnhancedPatternAwarePredictionEngine
from enhanced_realtime_continuous_system import EnhancedRealTimeContinuousPredictionSystem

# Global enhanced systems (to be initialized)
enhanced_universal_learner = None
enhanced_pattern_predictor = None
enhanced_realtime_system = None

@enhanced_api_router.post("/initialize-enhanced-pattern-learning")
async def initialize_enhanced_pattern_learning():
    """Initialize enhanced pattern learning systems"""
    try:
        global enhanced_universal_learner, enhanced_pattern_predictor, enhanced_realtime_system
        
        # Initialize systems
        enhanced_universal_learner = UniversalPatternLearning(memory_size=5000)
        enhanced_pattern_predictor = EnhancedPatternAwarePredictionEngine()
        enhanced_realtime_system = EnhancedRealTimeContinuousPredictionSystem(
            enhanced_universal_learner, enhanced_pattern_predictor
        )
        
        return {
            'status': 'success',
            'message': 'Enhanced pattern learning systems initialized',
            'systems_initialized': [
                'UniversalPatternLearning',
                'EnhancedPatternAwarePredictionEngine', 
                'EnhancedRealTimeContinuousPredictionSystem'
            ]
        }
        
    except Exception as e:
        logger.error(f"Error initializing enhanced pattern learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@enhanced_api_router.post("/learn-historical-patterns")
async def learn_historical_patterns(data: Dict[str, Any]):
    """Learn patterns from historical data with enhanced algorithms"""
    try:
        global enhanced_universal_learner
        
        if enhanced_universal_learner is None:
            raise HTTPException(status_code=400, detail="Enhanced pattern learning system not initialized")
        
        # Extract data from request
        historical_values = np.array(data['values'])
        timestamps = np.array(data.get('timestamps', [])) if data.get('timestamps') else None
        context = data.get('context', {})
        
        # Learn patterns
        learning_result = enhanced_universal_learner.learn_patterns(
            historical_values,
            timestamps=timestamps,
            pattern_context=context
        )
        
        return {
            'status': 'success',
            'learning_result': learning_result,
            'patterns_learned': learning_result.get('patterns_learned', 0),
            'learning_quality': learning_result.get('learning_quality', {}),
            'ready_for_prediction': learning_result.get('ready_for_prediction', False)
        }
        
    except Exception as e:
        logger.error(f"Error learning historical patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@enhanced_api_router.post("/generate-enhanced-pattern-predictions")
async def generate_enhanced_pattern_predictions(data: Dict[str, Any]):
    """Generate predictions with enhanced pattern awareness"""
    try:
        global enhanced_pattern_predictor, enhanced_universal_learner
        
        if not enhanced_pattern_predictor or not enhanced_universal_learner:
            raise HTTPException(status_code=400, detail="Enhanced systems not initialized")
        
        # Extract parameters
        historical_values = np.array(data['values'])
        steps = data.get('steps', 30)
        previous_predictions = data.get('previous_predictions', [])
        confidence_level = data.get('confidence_level', 0.95)
        
        # Learn patterns first
        pattern_result = enhanced_universal_learner.learn_patterns(
            historical_values,
            pattern_context={'data_type': 'prediction_request', 'priority': 'high_accuracy'}
        )
        
        # Generate enhanced predictions
        prediction_result = enhanced_pattern_predictor.generate_pattern_aware_predictions(
            data=historical_values,
            steps=steps,
            patterns=pattern_result.get('pattern_analysis', {}),
            previous_predictions=previous_predictions,
            confidence_level=confidence_level
        )
        
        return {
            'status': 'success',
            'predictions': prediction_result['predictions'],
            'confidence_intervals': prediction_result.get('confidence_intervals', []),
            'quality_metrics': prediction_result.get('quality_metrics', {}),
            'pattern_analysis': prediction_result.get('pattern_analysis', {}),
            'enhancement_info': {
                'pattern_following_score': prediction_result.get('pattern_following_score', 0.0),
                'continuity_score': prediction_result.get('continuity_score', 0.0),
                'prediction_method': prediction_result.get('prediction_method', 'enhanced_pattern_aware')
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating enhanced pattern predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@enhanced_api_router.post("/initialize-realtime-continuous-system")
async def initialize_realtime_continuous_system(data: Dict[str, Any]):
    """Initialize real-time continuous prediction system for right panel"""
    try:
        global enhanced_realtime_system
        
        if enhanced_realtime_system is None:
            raise HTTPException(status_code=400, detail="Real-time system not initialized")
        
        # Extract historical data
        historical_values = np.array(data['values'])
        timestamps = np.array(data.get('timestamps', [])) if data.get('timestamps') else None
        context = data.get('context', {'target': 'right_panel_graph'})
        
        # Initialize system with historical data
        init_result = enhanced_realtime_system.initialize_with_historical_data(
            historical_values,
            timestamps=timestamps,
            context=context
        )
        
        return {
            'status': 'success',
            'initialization_result': init_result,
            'ready_for_continuous_prediction': init_result.get('ready_for_continuous_prediction', False),
            'system_confidence': init_result.get('system_confidence', 0.0)
        }
        
    except Exception as e:
        logger.error(f"Error initializing real-time continuous system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@enhanced_api_router.post("/generate-realtime-continuous-predictions")
async def generate_realtime_continuous_predictions(data: Dict[str, Any]):
    """Generate continuous predictions optimized for right panel graph"""
    try:
        global enhanced_realtime_system
        
        if enhanced_realtime_system is None:
            raise HTTPException(status_code=400, detail="Real-time system not initialized")
        
        # Extract parameters
        steps = data.get('steps', 30)
        previous_predictions = data.get('previous_predictions', [])
        real_time_context = data.get('context', {})
        
        # Generate enhanced continuous predictions
        prediction_result = enhanced_realtime_system.generate_enhanced_continuous_predictions(
            steps=steps,
            previous_predictions=previous_predictions,
            real_time_context=real_time_context
        )
        
        return {
            'status': 'success',
            'predictions': prediction_result['predictions'],
            'confidence_intervals': prediction_result.get('confidence_intervals', []),
            'quality_metrics': prediction_result.get('quality_metrics', {}),
            'enhancement_info': prediction_result.get('enhancement_info', {}),
            'real_time_adaptations': prediction_result.get('real_time_adaptations_applied', {}),
            'prediction_method': prediction_result.get('prediction_method', 'enhanced_realtime_continuous')
        }
        
    except Exception as e:
        logger.error(f"Error generating real-time continuous predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@enhanced_api_router.get("/enhanced-system-status")
async def get_enhanced_system_status():
    """Get status of enhanced pattern learning systems"""
    try:
        global enhanced_universal_learner, enhanced_pattern_predictor, enhanced_realtime_system
        
        status = {
            'universal_pattern_learner': {
                'initialized': enhanced_universal_learner is not None,
                'pattern_library_size': len(enhanced_universal_learner.pattern_library) if enhanced_universal_learner else 0,
                'learned_patterns_count': len(enhanced_universal_learner.learned_patterns) if enhanced_universal_learner else 0
            },
            'enhanced_pattern_predictor': {
                'initialized': enhanced_pattern_predictor is not None,
                'strategy_performance': dict(enhanced_pattern_predictor.strategy_performance) if enhanced_pattern_predictor else {},
                'prediction_history_length': len(enhanced_pattern_predictor.prediction_history) if enhanced_pattern_predictor else 0
            },
            'enhanced_realtime_system': {
                'initialized': enhanced_realtime_system is not None,
                'system_status': enhanced_realtime_system.get_system_status() if enhanced_realtime_system else {}
            }
        }
        
        return {
            'status': 'success',
            'systems_status': status,
            'overall_health': 'healthy' if all([
                enhanced_universal_learner is not None,
                enhanced_pattern_predictor is not None,
                enhanced_realtime_system is not None
            ]) else 'partial'
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@enhanced_api_router.post("/reset-enhanced-systems")
async def reset_enhanced_systems():
    """Reset all enhanced pattern learning systems"""
    try:
        global enhanced_universal_learner, enhanced_pattern_predictor, enhanced_realtime_system
        
        reset_results = {}
        
        if enhanced_universal_learner:
            # Reset universal pattern learner
            enhanced_universal_learner.pattern_library.clear()
            enhanced_universal_learner.learned_patterns.clear()
            enhanced_universal_learner.pattern_performance.clear()
            reset_results['universal_pattern_learner'] = 'reset'
        
        if enhanced_pattern_predictor:
            # Reset enhanced pattern predictor
            enhanced_pattern_predictor.strategy_performance.clear()
            enhanced_pattern_predictor.pattern_performance.clear()
            enhanced_pattern_predictor.prediction_history.clear()
            reset_results['enhanced_pattern_predictor'] = 'reset'
        
        if enhanced_realtime_system:
            # Reset real-time system
            enhanced_realtime_system.reset_system()
            reset_results['enhanced_realtime_system'] = 'reset'
        
        return {
            'status': 'success',
            'message': 'Enhanced systems reset successfully',
            'reset_results': reset_results
        }
        
    except Exception as e:
        logger.error(f"Error resetting enhanced systems: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@enhanced_api_router.get("/pattern-learning-metrics")
async def get_pattern_learning_metrics():
    """Get detailed metrics about pattern learning performance"""
    try:
        global enhanced_universal_learner, enhanced_pattern_predictor, enhanced_realtime_system
        
        metrics = {}
        
        if enhanced_universal_learner:
            metrics['pattern_learning'] = {
                'pattern_library_size': len(enhanced_universal_learner.pattern_library),
                'learned_patterns_count': len(enhanced_universal_learner.learned_patterns),
                'performance_history': list(enhanced_universal_learner.performance_history),
                'pattern_accuracy': dict(enhanced_universal_learner.pattern_accuracy)
            }
        
        if enhanced_pattern_predictor:
            metrics['prediction_engine'] = {
                'strategy_performance': dict(enhanced_pattern_predictor.strategy_performance),
                'pattern_performance': dict(enhanced_pattern_predictor.pattern_performance),
                'prediction_history_length': len(enhanced_pattern_predictor.prediction_history)
            }
        
        if enhanced_realtime_system:
            metrics['realtime_system'] = enhanced_realtime_system.get_system_status()
        
        return {
            'status': 'success',
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting pattern learning metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))