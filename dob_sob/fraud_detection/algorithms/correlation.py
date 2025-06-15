# Cross-Entity Correlation Engine
"""
Cross-Entity Correlation Engine for DOB-SOB Fraud Detection

This module implements a comprehensive fraud detection system that correlates
patterns across all 6 entity categories:
1. Person/Professional entities
2. Property entities  
3. Job/Project entities
4. Violation/Enforcement entities
5. Regulatory/Inspection entities
6. Financial/Compliance entities
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EntityScore:
    """Individual entity fraud score with breakdown"""
    entity_id: str
    entity_type: str
    risk_score: float  # 0-100
    risk_level: str    # LOW, MEDIUM, HIGH, CRITICAL
    risk_factors: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0-1
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class CorrelationResult:
    """Result of cross-entity correlation analysis"""
    primary_entity: str
    related_entities: List[str]
    correlation_strength: float  # 0-1
    correlation_type: str
    risk_amplification: float  # Multiplier for combined risk
    evidence: List[str] = field(default_factory=list)

@dataclass
class FraudAlert:
    """Fraud alert with detailed information"""
    alert_id: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    entities_involved: List[str]
    correlation_score: float
    risk_factors: List[str]
    recommended_actions: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "ACTIVE"  # ACTIVE, INVESTIGATING, RESOLVED, FALSE_POSITIVE

class CrossEntityCorrelationEngine:
    """
    Advanced fraud detection engine that correlates patterns across all entity types
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/processed")
        self.entity_scores: Dict[str, EntityScore] = {}
        self.correlations: List[CorrelationResult] = []
        self.alerts: List[FraudAlert] = []
        
        # Risk thresholds
        self.risk_thresholds = {
            'LOW': (0, 25),
            'MEDIUM': (25, 50), 
            'HIGH': (50, 75),
            'CRITICAL': (75, 100)
        }
        
        # Correlation weights for different entity combinations
        self.correlation_weights = {
            ('person', 'property'): 0.8,
            ('person', 'job'): 0.9,
            ('person', 'violation'): 0.85,
            ('person', 'inspection'): 0.7,
            ('person', 'financial'): 0.95,
            ('property', 'job'): 0.9,
            ('property', 'violation'): 0.85,
            ('property', 'inspection'): 0.8,
            ('property', 'financial'): 0.75,
            ('job', 'violation'): 0.9,
            ('job', 'inspection'): 0.85,
            ('job', 'financial'): 0.9,
            ('violation', 'inspection'): 0.8,
            ('violation', 'financial'): 0.85,
            ('inspection', 'financial'): 0.7
        }

    async def analyze_entity_fraud_risk(self, entity_data: Dict[str, Any], entity_type: str) -> EntityScore:
        """
        Analyze fraud risk for a single entity
        """
        entity_id = str(entity_data.get('id', entity_data.get('entity_id', 'unknown')))
        
        # Calculate risk score based on entity type
        if entity_type == 'person':
            score = await self._analyze_person_risk(entity_data)
        elif entity_type == 'property':
            score = await self._analyze_property_risk(entity_data)
        elif entity_type == 'job':
            score = await self._analyze_job_risk(entity_data)
        elif entity_type == 'violation':
            score = await self._analyze_violation_risk(entity_data)
        elif entity_type == 'inspection':
            score = await self._analyze_inspection_risk(entity_data)
        elif entity_type == 'financial':
            score = await self._analyze_financial_risk(entity_data)
        else:
            logger.warning(f"Unknown entity type: {entity_type}")
            score = EntityScore(entity_id, entity_type, 0.0, 'LOW')
        
        self.entity_scores[entity_id] = score
        return score

    async def _analyze_person_risk(self, data: Dict[str, Any]) -> EntityScore:
        """Analyze fraud risk for person/professional entities"""
        entity_id = str(data.get('id', data.get('entity_id', 'unknown')))
        risk_score = 0.0
        risk_factors = []
        
        # License status issues
        if data.get('license_status') in ['EXPIRED', 'SUSPENDED', 'REVOKED']:
            risk_score += 25
            risk_factors.append(f"License status: {data.get('license_status')}")
        
        # Multiple violations
        violation_count = data.get('violation_count', 0)
        if violation_count > 5:
            risk_score += min(violation_count * 3, 30)
            risk_factors.append(f"High violation count: {violation_count}")
        
        # Rapid job submissions
        recent_jobs = data.get('recent_job_count', 0)
        if recent_jobs > 20:
            risk_score += 20
            risk_factors.append(f"Unusually high recent job activity: {recent_jobs}")
        
        # Address inconsistencies
        if data.get('address_inconsistencies', 0) > 2:
            risk_score += 15
            risk_factors.append("Multiple address inconsistencies")
        
        # Financial irregularities
        if data.get('payment_issues', False):
            risk_score += 20
            risk_factors.append("Payment processing issues")
        
        risk_level = self._get_risk_level(risk_score)
        confidence = min(len(risk_factors) * 0.2, 1.0)
        
        return EntityScore(entity_id, 'person', risk_score, risk_level, risk_factors, confidence)

    async def _analyze_property_risk(self, data: Dict[str, Any]) -> EntityScore:
        """Analyze fraud risk for property entities"""
        entity_id = str(data.get('id', data.get('entity_id', 'unknown')))
        risk_score = 0.0
        risk_factors = []
        
        # Excessive permit activity
        permit_count = data.get('permit_count_6months', 0)
        if permit_count > 10:
            risk_score += min(permit_count * 2, 25)
            risk_factors.append(f"High permit activity: {permit_count} in 6 months")
        
        # Violation density
        violation_density = data.get('violations_per_sqft', 0)
        if violation_density > 0.01:
            risk_score += 20
            risk_factors.append(f"High violation density: {violation_density}")
        
        # Ownership changes
        ownership_changes = data.get('ownership_changes_2years', 0)
        if ownership_changes > 3:
            risk_score += 15
            risk_factors.append(f"Frequent ownership changes: {ownership_changes}")
        
        # Zoning violations
        if data.get('zoning_violations', 0) > 2:
            risk_score += 20
            risk_factors.append("Multiple zoning violations")
        
        # Emergency orders
        if data.get('emergency_orders', 0) > 0:
            risk_score += 30
            risk_factors.append("Emergency orders issued")
        
        risk_level = self._get_risk_level(risk_score)
        confidence = min(len(risk_factors) * 0.25, 1.0)
        
        return EntityScore(entity_id, 'property', risk_score, risk_level, risk_factors, confidence)

    async def _analyze_job_risk(self, data: Dict[str, Any]) -> EntityScore:
        """Analyze fraud risk for job/project entities"""
        entity_id = str(data.get('id', data.get('entity_id', 'unknown')))
        risk_score = 0.0
        risk_factors = []
        
        # Permit value vs actual work discrepancies
        estimated_cost = data.get('estimated_cost', 0)
        actual_cost = data.get('actual_cost', 0)
        if estimated_cost > 0 and actual_cost > 0:
            cost_ratio = abs(actual_cost - estimated_cost) / estimated_cost
            if cost_ratio > 0.5:
                risk_score += 25
                risk_factors.append(f"Large cost discrepancy: {cost_ratio:.1%}")
        
        # Timeline irregularities
        if data.get('timeline_violations', 0) > 2:
            risk_score += 20
            risk_factors.append("Multiple timeline violations")
        
        # Inspection failures
        failed_inspections = data.get('failed_inspections', 0)
        if failed_inspections > 3:
            risk_score += min(failed_inspections * 5, 25)
            risk_factors.append(f"High inspection failure rate: {failed_inspections}")
        
        # Unlicensed work indicators
        if data.get('unlicensed_work_indicators', 0) > 1:
            risk_score += 30
            risk_factors.append("Indicators of unlicensed work")
        
        # Rapid permit cycling
        if data.get('rapid_permit_cycling', False):
            risk_score += 20
            risk_factors.append("Rapid permit cycling detected")
        
        risk_level = self._get_risk_level(risk_score)
        confidence = min(len(risk_factors) * 0.2, 1.0)
        
        return EntityScore(entity_id, 'job', risk_score, risk_level, risk_factors, confidence)

    async def _analyze_violation_risk(self, data: Dict[str, Any]) -> EntityScore:
        """Analyze fraud risk for violation/enforcement entities"""
        entity_id = str(data.get('id', data.get('entity_id', 'unknown')))
        risk_score = 0.0
        risk_factors = []
        
        # Severity and frequency
        severity = data.get('severity_score', 0)
        if severity > 7:
            risk_score += severity * 3
            risk_factors.append(f"High severity violations: {severity}")
        
        # Repeat violations
        repeat_count = data.get('repeat_violation_count', 0)
        if repeat_count > 2:
            risk_score += min(repeat_count * 8, 30)
            risk_factors.append(f"Repeat violations: {repeat_count}")
        
        # Non-compliance patterns
        if data.get('non_compliance_days', 0) > 90:
            risk_score += 25
            risk_factors.append("Extended non-compliance period")
        
        # Fine evasion
        if data.get('fine_evasion_indicators', False):
            risk_score += 20
            risk_factors.append("Fine evasion indicators")
        
        # Safety violations
        if data.get('safety_violations', 0) > 1:
            risk_score += 25
            risk_factors.append("Multiple safety violations")
        
        risk_level = self._get_risk_level(risk_score)
        confidence = min(len(risk_factors) * 0.25, 1.0)
        
        return EntityScore(entity_id, 'violation', risk_score, risk_level, risk_factors, confidence)

    async def _analyze_inspection_risk(self, data: Dict[str, Any]) -> EntityScore:
        """Analyze fraud risk for regulatory/inspection entities"""
        entity_id = str(data.get('id', data.get('entity_id', 'unknown')))
        risk_score = 0.0
        risk_factors = []
        
        # Failed inspection patterns
        failure_rate = data.get('inspection_failure_rate', 0)
        if failure_rate > 0.4:
            risk_score += failure_rate * 30
            risk_factors.append(f"High inspection failure rate: {failure_rate:.1%}")
        
        # Inspection avoidance
        if data.get('inspection_avoidance_indicators', 0) > 2:
            risk_score += 25
            risk_factors.append("Inspection avoidance patterns")
        
        # Fraudulent documentation
        if data.get('document_fraud_indicators', False):
            risk_score += 35
            risk_factors.append("Fraudulent documentation indicators")
        
        # Inspector concerns
        if data.get('inspector_red_flags', 0) > 1:
            risk_score += 20
            risk_factors.append("Inspector red flags raised")
        
        # Compliance gaming
        if data.get('compliance_gaming_score', 0) > 5:
            risk_score += 15
            risk_factors.append("Compliance gaming detected")
        
        risk_level = self._get_risk_level(risk_score)
        confidence = min(len(risk_factors) * 0.2, 1.0)
        
        return EntityScore(entity_id, 'inspection', risk_score, risk_level, risk_factors, confidence)

    async def _analyze_financial_risk(self, data: Dict[str, Any]) -> EntityScore:
        """Analyze fraud risk for financial/compliance entities"""
        entity_id = str(data.get('id', data.get('entity_id', 'unknown')))
        risk_score = 0.0
        risk_factors = []
        
        # Payment irregularities
        if data.get('payment_irregularities', 0) > 2:
            risk_score += 25
            risk_factors.append("Multiple payment irregularities")
        
        # Fee evasion
        unpaid_fees = data.get('unpaid_fees_amount', 0)
        if unpaid_fees > 10000:
            risk_score += min(unpaid_fees / 1000, 30)
            risk_factors.append(f"High unpaid fees: ${unpaid_fees:,.2f}")
        
        # Financial document fraud
        if data.get('financial_document_fraud', False):
            risk_score += 40
            risk_factors.append("Financial document fraud indicators")
        
        # Money laundering indicators
        if data.get('money_laundering_score', 0) > 6:
            risk_score += 35
            risk_factors.append("Money laundering indicators")
        
        # Tax evasion patterns
        if data.get('tax_evasion_indicators', 0) > 1:
            risk_score += 20
            risk_factors.append("Tax evasion patterns")
        
        risk_level = self._get_risk_level(risk_score)
        confidence = min(len(risk_factors) * 0.3, 1.0)
        
        return EntityScore(entity_id, 'financial', risk_score, risk_level, risk_factors, confidence)

    def _get_risk_level(self, score: float) -> str:
        """Convert numeric risk score to risk level"""
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= score < max_score:
                return level
        return 'CRITICAL' if score >= 75 else 'LOW'

    async def generate_fraud_analysis(self, entity_ids: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive fraud analysis for a set of entities
        """
        # Calculate composite risk score
        entity_scores = [self.entity_scores[eid] for eid in entity_ids if eid in self.entity_scores]
        if not entity_scores:
            return {"error": "No valid entity scores found"}
        
        # Base composite score (weighted average)
        base_score = sum(score.risk_score * score.confidence for score in entity_scores) / len(entity_scores)
        composite_score = min(base_score, 100.0)
        composite_risk_level = self._get_risk_level(composite_score)
        
        # Compile analysis
        analysis = {
            "analysis_id": f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "entities_analyzed": len(entity_ids),
            "composite_risk_score": round(composite_score, 2),
            "composite_risk_level": composite_risk_level,
            "individual_scores": [
                {
                    "entity_id": score.entity_id,
                    "entity_type": score.entity_type,
                    "risk_score": score.risk_score,
                    "risk_level": score.risk_level,
                    "confidence": score.confidence,
                    "risk_factors": score.risk_factors
                }
                for score in entity_scores
            ],
            "correlations": [],
            "alerts_generated": 0,
            "alerts": [],
            "recommendations": self._generate_recommendations(composite_score, [], entity_scores)
        }
        
        return analysis

    def _generate_recommendations(self, composite_score: float, correlations: List[CorrelationResult], 
                                entity_scores: List[EntityScore]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if composite_score > 75:
            recommendations.extend([
                "IMMEDIATE ACTION: Suspend all active permits and licenses",
                "Initiate comprehensive fraud investigation",
                "Coordinate with law enforcement agencies",
                "Freeze all financial transactions pending investigation"
            ])
        elif composite_score > 50:
            recommendations.extend([
                "Conduct priority investigation within 48 hours",
                "Implement enhanced monitoring protocols",
                "Require additional documentation for all transactions",
                "Schedule comprehensive audit of all related activities"
            ])
        elif composite_score > 25:
            recommendations.extend([
                "Schedule routine investigation within 2 weeks",
                "Increase inspection frequency",
                "Verify compliance with all regulations",
                "Monitor for pattern escalation"
            ])
        
        # Entity-specific recommendations
        high_risk_entities = [e for e in entity_scores if e.risk_score > 60]
        if high_risk_entities:
            recommendations.append(f"Focus investigation on high-risk entities: {', '.join([e.entity_id for e in high_risk_entities])}")
        
        return recommendations

    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[FraudAlert]:
        """Get active fraud alerts, optionally filtered by severity"""
        active_alerts = [alert for alert in self.alerts if alert.status == "ACTIVE"]
        
        if severity_filter:
            active_alerts = [alert for alert in active_alerts if alert.severity == severity_filter]
        
        return sorted(active_alerts, key=lambda x: x.created_at, reverse=True)

    async def batch_analyze_entities(self, entities_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze multiple entities in batch for efficiency
        """
        logger.info(f"Starting batch analysis of {len(entities_data)} entities")
        
        # Analyze individual entities
        tasks = []
        for entity_data in entities_data:
            entity_type = entity_data.get('entity_type', 'unknown')
            task = self.analyze_entity_fraud_risk(entity_data, entity_type)
            tasks.append(task)
        
        # Execute all analyses concurrently
        entity_scores = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_scores = [score for score in entity_scores if isinstance(score, EntityScore)]
        entity_ids = [score.entity_id for score in valid_scores]
        
        # Perform cross-entity analysis
        analysis = await self.generate_fraud_analysis(entity_ids)
        
        logger.info(f"Batch analysis complete. Analyzed {len(valid_scores)} entities, "
                   f"found {len(analysis.get('correlations', []))} correlations, "
                   f"generated {len(analysis.get('alerts', []))} alerts")
        
        return analysis
