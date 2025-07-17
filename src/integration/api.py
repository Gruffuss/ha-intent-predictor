"""
FastAPI REST API for HA Intent Predictor system.

Provides endpoints for:
- Prediction queries
- System status and health
- Manual training triggers
- Configuration updates
- Real-time metrics
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import asyncio
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ..learning.adaptive_predictor import AdaptiveOccupancyPredictor
from ..integration.monitoring import PerformanceMonitor
from ..storage.timeseries_db import TimescaleDBManager
from ..storage.feature_store import RedisFeatureStore

logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionRequest(BaseModel):
    room: str = Field(..., description="Room identifier")
    horizon: int = Field(..., description="Prediction horizon in minutes")
    context: Optional[Dict] = Field(None, description="Additional context")

class PredictionResponse(BaseModel):
    room: str
    horizon: int
    probability: float = Field(..., ge=0, le=1, description="Occupancy probability")
    uncertainty: float = Field(..., ge=0, le=1, description="Prediction uncertainty")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    contributing_factors: List[str] = Field(default_factory=list)
    model_agreement: float = Field(..., ge=0, le=1, description="Model ensemble agreement")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchPredictionRequest(BaseModel):
    rooms: List[str] = Field(..., description="List of room identifiers")
    horizons: List[int] = Field(..., description="List of prediction horizons")
    include_uncertainty: bool = Field(True, description="Include uncertainty estimates")

class SystemStatus(BaseModel):
    status: str = Field(..., description="System status")
    uptime: float = Field(..., description="System uptime in seconds")
    active_models: int = Field(..., description="Number of active models")
    total_predictions: int = Field(..., description="Total predictions made")
    average_accuracy: float = Field(..., description="Average prediction accuracy")
    memory_usage: float = Field(..., description="Memory usage percentage")
    cpu_usage: float = Field(..., description="CPU usage percentage")
    last_training: Optional[datetime] = Field(None, description="Last training time")

class RoomStatus(BaseModel):
    room: str
    current_occupancy: bool
    occupancy_probability: float
    last_activity: datetime
    prediction_accuracy: float
    model_performance: Dict[str, float]
    active_patterns: List[str]
    anomaly_score: float

class TrainingRequest(BaseModel):
    room: Optional[str] = Field(None, description="Specific room to train (all if None)")
    force_retrain: bool = Field(False, description="Force complete retraining")
    include_historical: bool = Field(True, description="Include historical data")

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    components: Dict[str, str]
    version: str = "1.0.0"

class MetricsResponse(BaseModel):
    timestamp: datetime
    system_metrics: Dict[str, float]
    prediction_metrics: Dict[str, float]
    room_metrics: Dict[str, Dict[str, float]]
    performance_metrics: Dict[str, float]


def create_api_app(system_instance) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        system_instance: Main system instance for accessing components
    
    Returns:
        Configured FastAPI application
    """
    
    app = FastAPI(
        title="HA Intent Predictor API",
        description="Adaptive ML-based occupancy prediction system for Home Assistant",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store system instance for route access
    app.state.system = system_instance
    
    # Dependency to get system components
    def get_system():
        return app.state.system
    
    def get_predictor() -> AdaptiveOccupancyPredictor:
        return app.state.system.components.get('predictor')
    
    def get_monitor() -> PerformanceMonitor:
        return app.state.system.components.get('monitor')
    
    def get_timeseries_db() -> TimescaleDBManager:
        return app.state.system.components.get('timeseries_db')
    
    def get_feature_store() -> RedisFeatureStore:
        return app.state.system.components.get('feature_store')
    
    # Health check endpoint
    @app.get("/health", response_model=HealthCheck)
    async def health_check(system = Depends(get_system)):
        """System health check"""
        
        components = {}
        overall_status = "healthy"
        
        # Check each component
        for component_name, component in system.components.items():
            try:
                if hasattr(component, 'health_check'):
                    component_status = await component.health_check()
                    components[component_name] = component_status
                else:
                    components[component_name] = "running"
            except Exception as e:
                components[component_name] = f"error: {str(e)}"
                overall_status = "degraded"
        
        return HealthCheck(
            status=overall_status,
            timestamp=datetime.now(),
            components=components
        )
    
    # System status endpoint
    @app.get("/status", response_model=SystemStatus)
    async def get_system_status(
        system = Depends(get_system),
        monitor = Depends(get_monitor)
    ):
        """Get comprehensive system status"""
        
        try:
            # Get system metrics
            metrics = await monitor.get_current_metrics()
            
            return SystemStatus(
                status="running" if system.running else "stopped",
                uptime=metrics.get('uptime', 0),
                active_models=metrics.get('active_models', 0),
                total_predictions=metrics.get('total_predictions', 0),
                average_accuracy=metrics.get('average_accuracy', 0.0),
                memory_usage=metrics.get('memory_usage', 0.0),
                cpu_usage=metrics.get('cpu_usage', 0.0),
                last_training=metrics.get('last_training')
            )
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            raise HTTPException(status_code=500, detail="Failed to get system status")
    
    # Single prediction endpoint
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_occupancy(
        request: PredictionRequest,
        predictor = Depends(get_predictor)
    ):
        """Get occupancy prediction for a room"""
        
        if not predictor:
            raise HTTPException(status_code=503, detail="Predictor not available")
        
        try:
            # Get prediction
            prediction = await predictor.predict_occupancy(
                room_id=request.room,
                horizon_minutes=request.horizon,
                context=request.context or {}
            )
            
            return PredictionResponse(
                room=request.room,
                horizon=request.horizon,
                probability=prediction['probability'],
                uncertainty=prediction.get('uncertainty', 0.0),
                confidence=1.0 - prediction.get('uncertainty', 0.0),
                contributing_factors=prediction.get('contributing_factors', []),
                model_agreement=prediction.get('model_agreement', 0.0),
                explanation=prediction.get('explanation')
            )
            
        except Exception as e:
            logger.error(f"Prediction error for {request.room}: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Batch prediction endpoint
    @app.post("/predict/batch")
    async def predict_batch(
        request: BatchPredictionRequest,
        predictor = Depends(get_predictor)
    ):
        """Get batch predictions for multiple rooms and horizons"""
        
        if not predictor:
            raise HTTPException(status_code=503, detail="Predictor not available")
        
        try:
            predictions = []
            
            # Process all combinations
            for room in request.rooms:
                for horizon in request.horizons:
                    prediction = await predictor.predict_occupancy(
                        room_id=room,
                        horizon_minutes=horizon
                    )
                    
                    pred_response = PredictionResponse(
                        room=room,
                        horizon=horizon,
                        probability=prediction['probability'],
                        uncertainty=prediction.get('uncertainty', 0.0),
                        confidence=1.0 - prediction.get('uncertainty', 0.0),
                        contributing_factors=prediction.get('contributing_factors', []),
                        model_agreement=prediction.get('model_agreement', 0.0),
                        explanation=prediction.get('explanation')
                    )
                    
                    predictions.append(pred_response)
            
            return {"predictions": predictions}
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    # Room status endpoint
    @app.get("/rooms/{room_id}/status", response_model=RoomStatus)
    async def get_room_status(
        room_id: str,
        predictor = Depends(get_predictor),
        db = Depends(get_timeseries_db)
    ):
        """Get detailed status for a specific room"""
        
        try:
            # Get current occupancy
            current_prediction = await predictor.predict_occupancy(
                room_id=room_id,
                horizon_minutes=0  # Current state
            )
            
            # Get room metrics
            room_metrics = await predictor.get_room_metrics(room_id)
            
            # Get last activity
            last_activity = await db.fetchrow("""
                SELECT MAX(timestamp) as last_activity
                FROM sensor_events
                WHERE room = $1 AND state = 'on'
            """, room_id)
            
            return RoomStatus(
                room=room_id,
                current_occupancy=current_prediction['probability'] > 0.5,
                occupancy_probability=current_prediction['probability'],
                last_activity=last_activity['last_activity'] if last_activity else datetime.now(),
                prediction_accuracy=room_metrics.get('accuracy', 0.0),
                model_performance=room_metrics.get('model_performance', {}),
                active_patterns=room_metrics.get('active_patterns', []),
                anomaly_score=room_metrics.get('anomaly_score', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error getting room status for {room_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get room status: {str(e)}")
    
    # List all rooms
    @app.get("/rooms")
    async def list_rooms(db = Depends(get_timeseries_db)):
        """List all available rooms"""
        
        try:
            rooms = await db.fetch("""
                SELECT DISTINCT room, 
                       COUNT(*) as sensor_count,
                       MAX(timestamp) as last_activity
                FROM sensor_events
                WHERE room IS NOT NULL
                GROUP BY room
                ORDER BY room
            """)
            
            return {"rooms": [dict(room) for room in rooms]}
            
        except Exception as e:
            logger.error(f"Error listing rooms: {e}")
            raise HTTPException(status_code=500, detail="Failed to list rooms")
    
    # Manual training endpoint
    @app.post("/train")
    async def trigger_training(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        predictor = Depends(get_predictor)
    ):
        """Trigger manual training"""
        
        if not predictor:
            raise HTTPException(status_code=503, detail="Predictor not available")
        
        try:
            # Add training task to background
            background_tasks.add_task(
                predictor.manual_training,
                room_id=request.room,
                force_retrain=request.force_retrain,
                include_historical=request.include_historical
            )
            
            return {
                "message": "Training started",
                "room": request.room or "all",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Training trigger error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to trigger training: {str(e)}")
    
    # Metrics endpoint
    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics(
        start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
        end_time: Optional[datetime] = Query(None, description="End time for metrics"),
        monitor = Depends(get_monitor)
    ):
        """Get system metrics"""
        
        try:
            # Default to last hour if no time range specified
            if not start_time:
                start_time = datetime.now() - timedelta(hours=1)
            if not end_time:
                end_time = datetime.now()
            
            metrics = await monitor.get_metrics_range(start_time, end_time)
            
            return MetricsResponse(
                timestamp=datetime.now(),
                system_metrics=metrics.get('system', {}),
                prediction_metrics=metrics.get('predictions', {}),
                room_metrics=metrics.get('rooms', {}),
                performance_metrics=metrics.get('performance', {})
            )
            
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            raise HTTPException(status_code=500, detail="Failed to get metrics")
    
    # Prometheus metrics endpoint
    @app.get("/metrics/prometheus")
    async def prometheus_metrics(monitor = Depends(get_monitor)):
        """Get metrics in Prometheus format"""
        
        try:
            prometheus_metrics = await monitor.get_prometheus_metrics()
            return prometheus_metrics
            
        except Exception as e:
            logger.error(f"Prometheus metrics error: {e}")
            raise HTTPException(status_code=500, detail="Failed to get Prometheus metrics")
    
    # Pattern discovery endpoint
    @app.get("/patterns/{room_id}")
    async def get_room_patterns(
        room_id: str,
        limit: int = Query(10, ge=1, le=100),
        db = Depends(get_timeseries_db)
    ):
        """Get discovered patterns for a room"""
        
        try:
            patterns = await db.fetch("""
                SELECT pattern_type, pattern_data, significance_score, 
                       confidence, discovered_at
                FROM discovered_patterns
                WHERE room = $1
                ORDER BY significance_score DESC
                LIMIT $2
            """, room_id, limit)
            
            return {
                "room": room_id,
                "patterns": [dict(pattern) for pattern in patterns]
            }
            
        except Exception as e:
            logger.error(f"Pattern discovery error for {room_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to get patterns")
    
    # Anomaly detection endpoint
    @app.get("/anomalies")
    async def get_anomalies(
        start_time: Optional[datetime] = Query(None),
        end_time: Optional[datetime] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        db = Depends(get_timeseries_db)
    ):
        """Get detected anomalies"""
        
        try:
            # Default to last 24 hours
            if not start_time:
                start_time = datetime.now() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.now()
            
            anomalies = await db.fetch("""
                SELECT pattern_type, pattern_data, significance_score,
                       discovered_at, room
                FROM discovered_patterns
                WHERE pattern_type LIKE '%anomaly%'
                AND discovered_at BETWEEN $1 AND $2
                ORDER BY discovered_at DESC
                LIMIT $3
            """, start_time, end_time, limit)
            
            return {
                "anomalies": [dict(anomaly) for anomaly in anomalies],
                "count": len(anomalies),
                "time_range": {
                    "start": start_time,
                    "end": end_time
                }
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            raise HTTPException(status_code=500, detail="Failed to get anomalies")
    
    # Configuration endpoint
    @app.get("/config")
    async def get_configuration(system = Depends(get_system)):
        """Get system configuration"""
        
        try:
            # Return sanitized config (no secrets)
            config = system.config
            
            # Remove sensitive information
            sanitized_config = {
                'rooms': config.get_rooms(),
                'sensors': {
                    'count': len(config.get_sensor_list('presence')),
                    'types': list(config.get_sensors().keys())
                },
                'learning': config.get('learning', {}),
                'api': {
                    'host': config.get('api.host'),
                    'port': config.get('api.port')
                }
            }
            
            return {"configuration": sanitized_config}
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            raise HTTPException(status_code=500, detail="Failed to get configuration")
    
    # Occupancy history endpoint
    @app.get("/history/{room_id}")
    async def get_occupancy_history(
        room_id: str,
        start_time: Optional[datetime] = Query(None),
        end_time: Optional[datetime] = Query(None),
        limit: int = Query(1000, ge=1, le=10000),
        db = Depends(get_timeseries_db)
    ):
        """Get occupancy history for a room"""
        
        try:
            # Default to last 24 hours
            if not start_time:
                start_time = datetime.now() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.now()
            
            history = await db.fetch("""
                SELECT timestamp, 
                       CASE WHEN probability > 0.5 THEN true ELSE false END as occupied,
                       confidence, 
                       model_name as inference_method,
                       horizon_minutes as duration_minutes
                FROM occupancy_predictions
                WHERE room = $1
                AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp DESC
                LIMIT $4
            """, room_id, start_time, end_time, limit)
            
            return {
                "room": room_id,
                "history": [dict(record) for record in history],
                "count": len(history),
                "time_range": {
                    "start": start_time,
                    "end": end_time
                }
            }
            
        except Exception as e:
            logger.error(f"History error for {room_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to get occupancy history")
    
    # Error handler for 404
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        return JSONResponse(
            status_code=404,
            content={"detail": "Endpoint not found"}
        )
    
    # Error handler for 500
    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        logger.error(f"Internal server error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    return app


# Standalone server for development
if __name__ == "__main__":
    # Create a mock system for development
    class MockSystem:
        def __init__(self):
            self.running = True
            self.components = {}
    
    app = create_api_app(MockSystem())
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )