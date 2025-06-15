"""
Fraud Monitoring API for DOB-SOB System

This module provides REST API endpoints for real-time fraud detection,
monitoring, and alert management across all entity types.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..fraud_detection.algorithms.correlation import (
    CrossEntityCorrelationEngine, 
    EntityScore, 
    FraudAlert,
    CorrelationResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DOB-SOB Fraud Monitoring API",
    description="Real-time fraud detection and monitoring system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global correlation engine instance
correlation_engine = CrossEntityCorrelationEngine()

# Background monitoring state
monitoring_active = False
monitoring_interval = 300  # 5 minutes default
last_monitoring_run = None

# Pydantic models for API requests/responses

class EntityData(BaseModel):
    """Entity data for fraud analysis"""
    id: str
    entity_type: str = Field(..., description="Type: person, property, job, violation, inspection, financial")
    data: Dict[str, Any] = Field(..., description="Entity-specific data fields")

class FraudAnalysisRequest(BaseModel):
    """Request for fraud analysis"""
    entities: List[EntityData]
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    include_correlations: bool = Field(default=True, description="Include cross-entity correlations")

class EntityScoreResponse(BaseModel):
    """Response model for entity fraud score"""
    entity_id: str
    entity_type: str
    risk_score: float
    risk_level: str
    risk_factors: List[str]
    confidence: float
    last_updated: datetime

class CorrelationResponse(BaseModel):
    """Response model for correlation results"""
    primary_entity: str
    related_entities: List[str]
    correlation_strength: float
    correlation_type: str
    risk_amplification: float
    evidence: List[str]

class FraudAlertResponse(BaseModel):
    """Response model for fraud alerts"""
    alert_id: str
    severity: str
    entities_involved: List[str]
    correlation_score: float
    risk_factors: List[str]
    recommended_actions: List[str]
    created_at: datetime
    status: str

class FraudAnalysisResponse(BaseModel):
    """Response model for comprehensive fraud analysis"""
    analysis_id: str
    timestamp: datetime
    entities_analyzed: int
    composite_risk_score: float
    composite_risk_level: str
    individual_scores: List[EntityScoreResponse]
    correlations: List[CorrelationResponse]
    alerts_generated: int
    alerts: List[FraudAlertResponse]
    recommendations: List[str]

class MonitoringStatus(BaseModel):
    """Monitoring system status"""
    active: bool
    interval_seconds: int
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    total_entities_monitored: int
    active_alerts: int

class AlertUpdateRequest(BaseModel):
    """Request to update alert status"""
    status: str = Field(..., description="New status: ACTIVE, INVESTIGATING, RESOLVED, FALSE_POSITIVE")
    notes: Optional[str] = Field(None, description="Optional notes about the status change")

# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "service": "DOB-SOB Fraud Monitoring API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze/fraud", response_model=FraudAnalysisResponse, tags=["Fraud Analysis"])
async def analyze_fraud(request: FraudAnalysisRequest):
    """
    Perform comprehensive fraud analysis on provided entities
    """
    try:
        logger.info(f"Starting fraud analysis for {len(request.entities)} entities")
        
        # Convert request entities to format expected by correlation engine
        entities_data = []
        for entity in request.entities:
            entity_data = {
                "id": entity.id,
                "entity_id": entity.id,
                "entity_type": entity.entity_type,
                **entity.data
            }
            entities_data.append(entity_data)
        
        # Perform batch analysis
        analysis_result = await correlation_engine.batch_analyze_entities(entities_data)
        
        if "error" in analysis_result:
            raise HTTPException(status_code=400, detail=analysis_result["error"])
        
        # Convert to response format
        response = FraudAnalysisResponse(
            analysis_id=analysis_result["analysis_id"],
            timestamp=datetime.fromisoformat(analysis_result["timestamp"]),
            entities_analyzed=analysis_result["entities_analyzed"],
            composite_risk_score=analysis_result["composite_risk_score"],
            composite_risk_level=analysis_result["composite_risk_level"],
            individual_scores=[
                EntityScoreResponse(
                    entity_id=score["entity_id"],
                    entity_type=score["entity_type"],
                    risk_score=score["risk_score"],
                    risk_level=score["risk_level"],
                    risk_factors=score["risk_factors"],
                    confidence=score["confidence"],
                    last_updated=datetime.now()
                )
                for score in analysis_result["individual_scores"]
            ],
            correlations=[],  # Will be populated when correlation detection is implemented
            alerts_generated=analysis_result["alerts_generated"],
            alerts=[],  # Will be populated when alert generation is implemented
            recommendations=analysis_result["recommendations"]
        )
        
        logger.info(f"Fraud analysis complete. Risk level: {response.composite_risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"Error in fraud analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/entity", response_model=EntityScoreResponse, tags=["Fraud Analysis"])
async def analyze_single_entity(entity: EntityData):
    """
    Analyze fraud risk for a single entity
    """
    try:
        logger.info(f"Analyzing single entity: {entity.id} ({entity.entity_type})")
        
        # Prepare entity data
        entity_data = {
            "id": entity.id,
            "entity_id": entity.id,
            **entity.data
        }
        
        # Analyze entity
        score = await correlation_engine.analyze_entity_fraud_risk(entity_data, entity.entity_type)
        
        response = EntityScoreResponse(
            entity_id=score.entity_id,
            entity_type=score.entity_type,
            risk_score=score.risk_score,
            risk_level=score.risk_level,
            risk_factors=score.risk_factors,
            confidence=score.confidence,
            last_updated=score.last_updated
        )
        
        logger.info(f"Entity analysis complete. Risk: {response.risk_level} ({response.risk_score})")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing entity {entity.id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Entity analysis failed: {str(e)}")

@app.get("/alerts", response_model=List[FraudAlertResponse], tags=["Alerts"])
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: LOW, MEDIUM, HIGH, CRITICAL"),
    status: Optional[str] = Query("ACTIVE", description="Filter by status"),
    limit: int = Query(50, description="Maximum number of alerts to return")
):
    """
    Get fraud alerts with optional filtering
    """
    try:
        # Get alerts from correlation engine
        alerts = correlation_engine.get_active_alerts(severity_filter=severity)
        
        # Filter by status if specified
        if status:
            alerts = [alert for alert in alerts if alert.status == status]
        
        # Limit results
        alerts = alerts[:limit]
        
        # Convert to response format
        response = [
            FraudAlertResponse(
                alert_id=alert.alert_id,
                severity=alert.severity,
                entities_involved=alert.entities_involved,
                correlation_score=alert.correlation_score,
                risk_factors=alert.risk_factors,
                recommended_actions=alert.recommended_actions,
                created_at=alert.created_at,
                status=alert.status
            )
            for alert in alerts
        ]
        
        logger.info(f"Retrieved {len(response)} alerts")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alerts: {str(e)}")

@app.put("/alerts/{alert_id}", tags=["Alerts"])
async def update_alert_status(alert_id: str, update: AlertUpdateRequest):
    """
    Update the status of a fraud alert
    """
    try:
        # Find the alert
        alert = None
        for a in correlation_engine.alerts:
            if a.alert_id == alert_id:
                alert = a
                break
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Update status
        old_status = alert.status
        alert.status = update.status
        
        logger.info(f"Updated alert {alert_id} status from {old_status} to {update.status}")
        
        return {
            "alert_id": alert_id,
            "old_status": old_status,
            "new_status": update.status,
            "updated_at": datetime.now().isoformat(),
            "notes": update.notes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update alert: {str(e)}")

@app.get("/monitoring/status", response_model=MonitoringStatus, tags=["Monitoring"])
async def get_monitoring_status():
    """
    Get current monitoring system status
    """
    global monitoring_active, monitoring_interval, last_monitoring_run
    
    next_run = None
    if monitoring_active and last_monitoring_run:
        next_run = last_monitoring_run + timedelta(seconds=monitoring_interval)
    
    return MonitoringStatus(
        active=monitoring_active,
        interval_seconds=monitoring_interval,
        last_run=last_monitoring_run,
        next_run=next_run,
        total_entities_monitored=len(correlation_engine.entity_scores),
        active_alerts=len(correlation_engine.get_active_alerts())
    )

@app.post("/monitoring/start", tags=["Monitoring"])
async def start_monitoring(
    background_tasks: BackgroundTasks,
    interval_seconds: int = Query(300, description="Monitoring interval in seconds")
):
    """
    Start background fraud monitoring
    """
    global monitoring_active, monitoring_interval
    
    if monitoring_active:
        return {"message": "Monitoring already active", "interval": monitoring_interval}
    
    monitoring_active = True
    monitoring_interval = interval_seconds
    
    # Start background monitoring task
    background_tasks.add_task(background_monitoring_task)
    
    logger.info(f"Started fraud monitoring with {interval_seconds}s interval")
    
    return {
        "message": "Fraud monitoring started",
        "interval_seconds": interval_seconds,
        "started_at": datetime.now().isoformat()
    }

@app.post("/monitoring/stop", tags=["Monitoring"])
async def stop_monitoring():
    """
    Stop background fraud monitoring
    """
    global monitoring_active
    
    if not monitoring_active:
        return {"message": "Monitoring not active"}
    
    monitoring_active = False
    
    logger.info("Stopped fraud monitoring")
    
    return {
        "message": "Fraud monitoring stopped",
        "stopped_at": datetime.now().isoformat()
    }

@app.get("/monitoring/history", tags=["Monitoring"])
async def get_monitoring_history(
    hours: int = Query(24, description="Hours of history to retrieve"),
    limit: int = Query(100, description="Maximum number of records")
):
    """
    Get monitoring history and statistics
    """
    try:
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get recent alerts within time range
        recent_alerts = [
            alert for alert in correlation_engine.alerts
            if start_time <= alert.created_at <= end_time
        ]
        
        # Calculate statistics
        stats = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "alerts_generated": len(recent_alerts),
            "alerts_by_severity": {},
            "entities_analyzed": len(correlation_engine.entity_scores),
            "average_risk_score": 0.0,
            "high_risk_entities": 0
        }
        
        # Calculate alert statistics
        for alert in recent_alerts:
            severity = alert.severity
            stats["alerts_by_severity"][severity] = stats["alerts_by_severity"].get(severity, 0) + 1
        
        # Calculate entity statistics
        if correlation_engine.entity_scores:
            risk_scores = [score.risk_score for score in correlation_engine.entity_scores.values()]
            stats["average_risk_score"] = sum(risk_scores) / len(risk_scores)
            stats["high_risk_entities"] = len([score for score in risk_scores if score > 60])
        
        return {
            "monitoring_history": stats,
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity,
                    "created_at": alert.created_at.isoformat(),
                    "entities_count": len(alert.entities_involved)
                }
                for alert in recent_alerts[:limit]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error retrieving monitoring history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

# Background Tasks

async def background_monitoring_task():
    """
    Background task for continuous fraud monitoring
    """
    global monitoring_active, last_monitoring_run
    
    logger.info("Starting background fraud monitoring task")
    
    while monitoring_active:
        try:
            last_monitoring_run = datetime.now()
            
            # Perform monitoring checks
            await perform_monitoring_cycle()
            
            # Wait for next cycle
            await asyncio.sleep(monitoring_interval)
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    logger.info("Background fraud monitoring task stopped")

async def perform_monitoring_cycle():
    """
    Perform one cycle of fraud monitoring
    """
    logger.info("Performing fraud monitoring cycle")
    
    # In a real implementation, this would:
    # 1. Query database for new/updated entities
    # 2. Analyze entities for fraud risk
    # 3. Generate alerts for high-risk findings
    # 4. Update monitoring statistics
    
    # For now, we'll just log the cycle
    active_alerts = len(correlation_engine.get_active_alerts())
    total_entities = len(correlation_engine.entity_scores)
    
    logger.info(f"Monitoring cycle complete. {total_entities} entities monitored, {active_alerts} active alerts")

# Utility functions

def get_correlation_engine() -> CrossEntityCorrelationEngine:
    """Dependency to get correlation engine instance"""
    return correlation_engine

# Application startup/shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("DOB-SOB Fraud Monitoring API starting up")
    
    # Initialize correlation engine with any required setup
    # In production, this might load models, connect to databases, etc.
    
    logger.info("Fraud Monitoring API ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown"""
    global monitoring_active
    
    logger.info("DOB-SOB Fraud Monitoring API shutting down")
    
    # Stop monitoring
    monitoring_active = False
    
    logger.info("Fraud Monitoring API shutdown complete")

# Main entry point for development
if __name__ == "__main__":
    uvicorn.run(
        "fraud_monitoring:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 