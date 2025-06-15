"""
Comprehensive tests for the Cross-Entity Correlation Engine

Tests all aspects of fraud detection across the 6 entity categories:
- Person/Professional entities
- Property entities
- Job/Project entities
- Violation/Enforcement entities
- Regulatory/Inspection entities
- Financial/Compliance entities
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dob_sob.fraud_detection.algorithms.correlation import (
    CrossEntityCorrelationEngine,
    EntityScore,
    FraudAlert,
    CorrelationResult
)

class TestCrossEntityCorrelationEngine:
    """Test suite for the Cross-Entity Correlation Engine"""
    
    @pytest.fixture
    def correlation_engine(self):
        """Create a correlation engine instance for testing"""
        return CrossEntityCorrelationEngine()
    
    @pytest.fixture
    def sample_person_data(self):
        """Sample person/professional entity data for testing"""
        return {
            # High risk person
            "high_risk_person": {
                "id": "person_001",
                "entity_id": "person_001",
                "license_status": "SUSPENDED",
                "violation_count": 8,
                "recent_job_count": 25,
                "address_inconsistencies": 3,
                "payment_issues": True
            },
            # Medium risk person
            "medium_risk_person": {
                "id": "person_002", 
                "entity_id": "person_002",
                "license_status": "EXPIRED",
                "violation_count": 3,
                "recent_job_count": 15,
                "address_inconsistencies": 1,
                "payment_issues": False
            },
            # Low risk person
            "low_risk_person": {
                "id": "person_003",
                "entity_id": "person_003", 
                "license_status": "ACTIVE",
                "violation_count": 1,
                "recent_job_count": 5,
                "address_inconsistencies": 0,
                "payment_issues": False
            }
        }
    
    @pytest.fixture
    def sample_property_data(self):
        """Sample property entity data for testing"""
        return {
            # High risk property
            "high_risk_property": {
                "id": "property_001",
                "entity_id": "property_001",
                "permit_count_6months": 15,
                "violations_per_sqft": 0.02,
                "ownership_changes_2years": 5,
                "zoning_violations": 3,
                "emergency_orders": 2
            },
            # Medium risk property
            "medium_risk_property": {
                "id": "property_002",
                "entity_id": "property_002",
                "permit_count_6months": 8,
                "violations_per_sqft": 0.008,
                "ownership_changes_2years": 2,
                "zoning_violations": 1,
                "emergency_orders": 0
            },
            # Low risk property
            "low_risk_property": {
                "id": "property_003",
                "entity_id": "property_003",
                "permit_count_6months": 3,
                "violations_per_sqft": 0.002,
                "ownership_changes_2years": 0,
                "zoning_violations": 0,
                "emergency_orders": 0
            }
        }
    
    @pytest.fixture
    def sample_job_data(self):
        """Sample job/project entity data for testing"""
        return {
            # High risk job
            "high_risk_job": {
                "id": "job_001",
                "entity_id": "job_001",
                "estimated_cost": 50000,
                "actual_cost": 100000,  # 100% increase
                "timeline_violations": 4,
                "failed_inspections": 5,
                "unlicensed_work_indicators": 2,
                "rapid_permit_cycling": True
            },
            # Medium risk job
            "medium_risk_job": {
                "id": "job_002",
                "entity_id": "job_002",
                "estimated_cost": 30000,
                "actual_cost": 40000,  # 33% increase
                "timeline_violations": 1,
                "failed_inspections": 2,
                "unlicensed_work_indicators": 0,
                "rapid_permit_cycling": False
            },
            # Low risk job
            "low_risk_job": {
                "id": "job_003",
                "entity_id": "job_003",
                "estimated_cost": 25000,
                "actual_cost": 26000,  # 4% increase
                "timeline_violations": 0,
                "failed_inspections": 0,
                "unlicensed_work_indicators": 0,
                "rapid_permit_cycling": False
            }
        }
    
    @pytest.fixture
    def sample_violation_data(self):
        """Sample violation/enforcement entity data for testing"""
        return {
            # High risk violation
            "high_risk_violation": {
                "id": "violation_001",
                "entity_id": "violation_001",
                "severity_score": 9,
                "repeat_violation_count": 4,
                "non_compliance_days": 120,
                "fine_evasion_indicators": True,
                "safety_violations": 3
            },
            # Medium risk violation
            "medium_risk_violation": {
                "id": "violation_002",
                "entity_id": "violation_002",
                "severity_score": 6,
                "repeat_violation_count": 2,
                "non_compliance_days": 60,
                "fine_evasion_indicators": False,
                "safety_violations": 1
            },
            # Low risk violation
            "low_risk_violation": {
                "id": "violation_003",
                "entity_id": "violation_003",
                "severity_score": 3,
                "repeat_violation_count": 0,
                "non_compliance_days": 15,
                "fine_evasion_indicators": False,
                "safety_violations": 0
            }
        }
    
    @pytest.fixture
    def sample_inspection_data(self):
        """Sample regulatory/inspection entity data for testing"""
        return {
            # High risk inspection
            "high_risk_inspection": {
                "id": "inspection_001",
                "entity_id": "inspection_001",
                "inspection_failure_rate": 0.7,
                "inspection_avoidance_indicators": 3,
                "document_fraud_indicators": True,
                "inspector_red_flags": 2,
                "compliance_gaming_score": 8
            },
            # Medium risk inspection
            "medium_risk_inspection": {
                "id": "inspection_002",
                "entity_id": "inspection_002",
                "inspection_failure_rate": 0.3,
                "inspection_avoidance_indicators": 1,
                "document_fraud_indicators": False,
                "inspector_red_flags": 1,
                "compliance_gaming_score": 4
            },
            # Low risk inspection
            "low_risk_inspection": {
                "id": "inspection_003",
                "entity_id": "inspection_003",
                "inspection_failure_rate": 0.1,
                "inspection_avoidance_indicators": 0,
                "document_fraud_indicators": False,
                "inspector_red_flags": 0,
                "compliance_gaming_score": 1
            }
        }
    
    @pytest.fixture
    def sample_financial_data(self):
        """Sample financial/compliance entity data for testing"""
        return {
            # High risk financial
            "high_risk_financial": {
                "id": "financial_001",
                "entity_id": "financial_001",
                "payment_irregularities": 4,
                "unpaid_fees_amount": 25000,
                "financial_document_fraud": True,
                "money_laundering_score": 8,
                "tax_evasion_indicators": 2
            },
            # Medium risk financial
            "medium_risk_financial": {
                "id": "financial_002",
                "entity_id": "financial_002",
                "payment_irregularities": 2,
                "unpaid_fees_amount": 8000,
                "financial_document_fraud": False,
                "money_laundering_score": 4,
                "tax_evasion_indicators": 1
            },
            # Low risk financial
            "low_risk_financial": {
                "id": "financial_003",
                "entity_id": "financial_003",
                "payment_irregularities": 0,
                "unpaid_fees_amount": 500,
                "financial_document_fraud": False,
                "money_laundering_score": 1,
                "tax_evasion_indicators": 0
            }
        }

    # Individual Entity Analysis Tests
    
    @pytest.mark.asyncio
    async def test_person_risk_analysis_high_risk(self, correlation_engine, sample_person_data):
        """Test high-risk person analysis"""
        data = sample_person_data["high_risk_person"]
        score = await correlation_engine.analyze_entity_fraud_risk(data, "person")
        
        assert isinstance(score, EntityScore)
        assert score.entity_id == "person_001"
        assert score.entity_type == "person"
        assert score.risk_score >= 70  # Should be high risk
        assert score.risk_level in ["HIGH", "CRITICAL"]
        assert len(score.risk_factors) >= 4  # Should have multiple risk factors
        assert score.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_person_risk_analysis_low_risk(self, correlation_engine, sample_person_data):
        """Test low-risk person analysis"""
        data = sample_person_data["low_risk_person"]
        score = await correlation_engine.analyze_entity_fraud_risk(data, "person")
        
        assert score.risk_score <= 25  # Should be low risk
        assert score.risk_level == "LOW"
        assert len(score.risk_factors) <= 1
    
    @pytest.mark.asyncio
    async def test_property_risk_analysis_high_risk(self, correlation_engine, sample_property_data):
        """Test high-risk property analysis"""
        data = sample_property_data["high_risk_property"]
        score = await correlation_engine.analyze_entity_fraud_risk(data, "property")
        
        assert score.risk_score >= 70
        assert score.risk_level in ["HIGH", "CRITICAL"]
        assert "emergency_orders" in str(score.risk_factors).lower()
    
    @pytest.mark.asyncio
    async def test_job_risk_analysis_high_risk(self, correlation_engine, sample_job_data):
        """Test high-risk job analysis"""
        data = sample_job_data["high_risk_job"]
        score = await correlation_engine.analyze_entity_fraud_risk(data, "job")
        
        assert score.risk_score >= 70
        assert score.risk_level in ["HIGH", "CRITICAL"]
        assert any("cost discrepancy" in factor.lower() for factor in score.risk_factors)
    
    @pytest.mark.asyncio
    async def test_violation_risk_analysis_high_risk(self, correlation_engine, sample_violation_data):
        """Test high-risk violation analysis"""
        data = sample_violation_data["high_risk_violation"]
        score = await correlation_engine.analyze_entity_fraud_risk(data, "violation")
        
        assert score.risk_score >= 70
        assert score.risk_level in ["HIGH", "CRITICAL"]
        assert any("safety" in factor.lower() for factor in score.risk_factors)
    
    @pytest.mark.asyncio
    async def test_inspection_risk_analysis_high_risk(self, correlation_engine, sample_inspection_data):
        """Test high-risk inspection analysis"""
        data = sample_inspection_data["high_risk_inspection"]
        score = await correlation_engine.analyze_entity_fraud_risk(data, "inspection")
        
        assert score.risk_score >= 70
        assert score.risk_level in ["HIGH", "CRITICAL"]
        assert any("fraud" in factor.lower() for factor in score.risk_factors)
    
    @pytest.mark.asyncio
    async def test_financial_risk_analysis_high_risk(self, correlation_engine, sample_financial_data):
        """Test high-risk financial analysis"""
        data = sample_financial_data["high_risk_financial"]
        score = await correlation_engine.analyze_entity_fraud_risk(data, "financial")
        
        assert score.risk_score >= 70
        assert score.risk_level in ["HIGH", "CRITICAL"]
        assert any("laundering" in factor.lower() for factor in score.risk_factors)

    # Cross-Entity Correlation Tests
    
    @pytest.mark.asyncio
    async def test_cross_entity_correlation_detection(self, correlation_engine, sample_person_data, sample_property_data):
        """Test detection of correlations between different entity types"""
        # Analyze person and property entities
        person_data = sample_person_data["high_risk_person"]
        property_data = sample_property_data["high_risk_property"]
        
        person_score = await correlation_engine.analyze_entity_fraud_risk(person_data, "person")
        property_score = await correlation_engine.analyze_entity_fraud_risk(property_data, "property")
        
        # Both should be stored in the engine
        assert "person_001" in correlation_engine.entity_scores
        assert "property_001" in correlation_engine.entity_scores
        
        # Check correlation weights are defined
        assert ("person", "property") in correlation_engine.correlation_weights
        assert correlation_engine.correlation_weights[("person", "property")] == 0.8

    # Comprehensive Fraud Analysis Tests
    
    @pytest.mark.asyncio
    async def test_comprehensive_fraud_analysis(self, correlation_engine, sample_person_data, 
                                              sample_property_data, sample_job_data):
        """Test comprehensive fraud analysis across multiple entity types"""
        # Prepare mixed risk entities
        entities_data = [
            {**sample_person_data["high_risk_person"], "entity_type": "person"},
            {**sample_property_data["medium_risk_property"], "entity_type": "property"},
            {**sample_job_data["low_risk_job"], "entity_type": "job"}
        ]
        
        # Perform batch analysis
        analysis = await correlation_engine.batch_analyze_entities(entities_data)
        
        # Verify analysis structure
        assert "analysis_id" in analysis
        assert "timestamp" in analysis
        assert analysis["entities_analyzed"] == 3
        assert "composite_risk_score" in analysis
        assert "composite_risk_level" in analysis
        assert "individual_scores" in analysis
        assert "recommendations" in analysis
        
        # Check individual scores
        assert len(analysis["individual_scores"]) == 3
        
        # Verify recommendations are provided
        assert len(analysis["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_fraud_analysis_all_entity_types(self, correlation_engine, sample_person_data,
                                                  sample_property_data, sample_job_data,
                                                  sample_violation_data, sample_inspection_data,
                                                  sample_financial_data):
        """Test fraud analysis with all 6 entity types"""
        entities_data = [
            {**sample_person_data["high_risk_person"], "entity_type": "person"},
            {**sample_property_data["high_risk_property"], "entity_type": "property"},
            {**sample_job_data["high_risk_job"], "entity_type": "job"},
            {**sample_violation_data["high_risk_violation"], "entity_type": "violation"},
            {**sample_inspection_data["high_risk_inspection"], "entity_type": "inspection"},
            {**sample_financial_data["high_risk_financial"], "entity_type": "financial"}
        ]
        
        analysis = await correlation_engine.batch_analyze_entities(entities_data)
        
        # Should have high composite risk due to all high-risk entities
        assert analysis["composite_risk_score"] >= 70
        assert analysis["composite_risk_level"] in ["HIGH", "CRITICAL"]
        assert analysis["entities_analyzed"] == 6
        
        # Check that all entity types are represented
        entity_types = [score["entity_type"] for score in analysis["individual_scores"]]
        expected_types = ["person", "property", "job", "violation", "inspection", "financial"]
        for expected_type in expected_types:
            assert expected_type in entity_types

    # Fraud Network Detection Tests
    
    @pytest.mark.asyncio
    async def test_fraud_network_detection(self, correlation_engine):
        """Test detection of fraud networks across entities"""
        # Create entities that should form a network (same addresses, overlapping timeframes, etc.)
        network_entities = [
            {
                "id": "person_network_1",
                "entity_id": "person_network_1",
                "entity_type": "person",
                "license_status": "SUSPENDED",
                "violation_count": 10,
                "address": "123 Fraud Street",
                "recent_job_count": 30
            },
            {
                "id": "property_network_1", 
                "entity_id": "property_network_1",
                "entity_type": "property",
                "address": "123 Fraud Street",
                "permit_count_6months": 20,
                "emergency_orders": 3,
                "ownership_changes_2years": 6
            },
            {
                "id": "job_network_1",
                "entity_id": "job_network_1", 
                "entity_type": "job",
                "property_address": "123 Fraud Street",
                "contractor_id": "person_network_1",
                "estimated_cost": 10000,
                "actual_cost": 50000,
                "unlicensed_work_indicators": 3
            }
        ]
        
        analysis = await correlation_engine.batch_analyze_entities(network_entities)
        
        # Should detect high risk due to network connections
        assert analysis["composite_risk_score"] >= 60
        assert analysis["entities_analyzed"] == 3

    # Performance and Scalability Tests
    
    @pytest.mark.asyncio
    async def test_batch_analysis_performance(self, correlation_engine):
        """Test performance with larger datasets"""
        # Create 50 entities for performance testing
        large_dataset = []
        for i in range(50):
            entity = {
                "id": f"entity_{i}",
                "entity_id": f"entity_{i}",
                "entity_type": "person",
                "license_status": "ACTIVE" if i % 3 == 0 else "EXPIRED",
                "violation_count": i % 10,
                "recent_job_count": i % 25,
                "address_inconsistencies": i % 3,
                "payment_issues": i % 4 == 0
            }
            large_dataset.append(entity)
        
        # Measure analysis time
        start_time = datetime.now()
        analysis = await correlation_engine.batch_analyze_entities(large_dataset)
        end_time = datetime.now()
        
        analysis_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert analysis_time < 30  # 30 seconds max for 50 entities
        assert analysis["entities_analyzed"] == 50
        assert len(analysis["individual_scores"]) == 50

    # Error Handling Tests
    
    @pytest.mark.asyncio
    async def test_invalid_entity_type(self, correlation_engine):
        """Test handling of invalid entity types"""
        invalid_data = {
            "id": "invalid_001",
            "entity_id": "invalid_001",
            "some_field": "some_value"
        }
        
        score = await correlation_engine.analyze_entity_fraud_risk(invalid_data, "invalid_type")
        
        # Should return low-risk score for unknown entity type
        assert score.risk_score == 0.0
        assert score.risk_level == "LOW"
        assert score.entity_type == "invalid_type"

    @pytest.mark.asyncio
    async def test_empty_entity_data(self, correlation_engine):
        """Test handling of empty entity data"""
        empty_data = {"id": "empty_001", "entity_id": "empty_001"}
        
        score = await correlation_engine.analyze_entity_fraud_risk(empty_data, "person")
        
        # Should handle gracefully with minimal risk
        assert score.risk_score >= 0
        assert score.entity_id == "empty_001"

    @pytest.mark.asyncio
    async def test_missing_entity_ids(self, correlation_engine):
        """Test fraud analysis with missing entity IDs"""
        entities_data = [
            {"entity_type": "person", "license_status": "ACTIVE"},  # Missing ID
            {"id": "valid_001", "entity_id": "valid_001", "entity_type": "person", "license_status": "ACTIVE"}
        ]
        
        analysis = await correlation_engine.batch_analyze_entities(entities_data)
        
        # Should handle missing IDs gracefully
        assert "error" not in analysis
        assert analysis["entities_analyzed"] >= 1

    # Alert Generation Tests
    
    def test_get_active_alerts_empty(self, correlation_engine):
        """Test getting alerts when none exist"""
        alerts = correlation_engine.get_active_alerts()
        assert isinstance(alerts, list)
        assert len(alerts) == 0

    def test_get_active_alerts_with_filter(self, correlation_engine):
        """Test getting alerts with severity filter"""
        # Add a test alert
        test_alert = FraudAlert(
            alert_id="test_001",
            severity="HIGH",
            entities_involved=["entity_001"],
            correlation_score=0.8,
            risk_factors=["Test risk factor"],
            recommended_actions=["Test action"],
            status="ACTIVE"
        )
        correlation_engine.alerts.append(test_alert)
        
        # Test filtering
        high_alerts = correlation_engine.get_active_alerts(severity_filter="HIGH")
        assert len(high_alerts) == 1
        assert high_alerts[0].severity == "HIGH"
        
        # Test non-matching filter
        critical_alerts = correlation_engine.get_active_alerts(severity_filter="CRITICAL")
        assert len(critical_alerts) == 0

    # Risk Level Classification Tests
    
    def test_risk_level_classification(self, correlation_engine):
        """Test risk level classification thresholds"""
        assert correlation_engine._get_risk_level(10) == "LOW"
        assert correlation_engine._get_risk_level(35) == "MEDIUM"
        assert correlation_engine._get_risk_level(60) == "HIGH"
        assert correlation_engine._get_risk_level(85) == "CRITICAL"
        assert correlation_engine._get_risk_level(100) == "CRITICAL"

    # Recommendation Generation Tests
    
    def test_recommendation_generation_critical_risk(self, correlation_engine):
        """Test recommendation generation for critical risk scenarios"""
        # Create high-risk entity scores
        high_risk_scores = [
            EntityScore("entity_001", "person", 85, "CRITICAL", ["High violations"], 0.9),
            EntityScore("entity_002", "property", 80, "CRITICAL", ["Emergency orders"], 0.8)
        ]
        
        recommendations = correlation_engine._generate_recommendations(85, [], high_risk_scores)
        
        assert len(recommendations) > 0
        assert any("IMMEDIATE ACTION" in rec for rec in recommendations)
        assert any("investigation" in rec.lower() for rec in recommendations)
        assert any("entity_001" in rec for rec in recommendations)

    def test_recommendation_generation_medium_risk(self, correlation_engine):
        """Test recommendation generation for medium risk scenarios"""
        medium_risk_scores = [
            EntityScore("entity_001", "person", 40, "MEDIUM", ["Some violations"], 0.6)
        ]
        
        recommendations = correlation_engine._generate_recommendations(40, [], medium_risk_scores)
        
        assert len(recommendations) > 0
        assert any("priority investigation" in rec.lower() for rec in recommendations)
        assert not any("IMMEDIATE ACTION" in rec for rec in recommendations)

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 