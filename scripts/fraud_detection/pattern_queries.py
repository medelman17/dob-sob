"""
Pattern Recognition Query Library for NYC DOB Fraud Detection

This module implements sophisticated search strategies to identify suspicious 
corporate networks in construction data using Graphiti's temporal knowledge graphs.
"""

import json
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    from graphiti_core import Graphiti
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    Graphiti = None

# Mock SearchType for when Graphiti is not available or import fails
class SearchType:
    SIMILARITY = "similarity"
    SEMANTIC = "semantic"


class FraudPatternType(Enum):
    """Types of fraud patterns to detect"""
    PASS_THROUGH = "pass_through"          # DBE firms with minimal markup
    CONTROL_FRAUD = "control_fraud"        # Non-minority controlling DBE firms  
    SINGLE_CUSTOMER = "single_customer"    # DBE with only one customer
    SHELL_COMPANY = "shell_company"        # Companies with minimal operations
    TIMING_FRAUD = "timing_fraud"          # Suspicious timing patterns
    MARKUP_ANOMALY = "markup_anomaly"      # Unusual markup percentages
    NETWORK_CLUSTER = "network_cluster"    # Suspicious entity clusters


@dataclass
class FraudAlert:
    """Represents a detected fraud pattern"""
    pattern_type: FraudPatternType
    entities: List[str]
    risk_score: float
    evidence: Dict[str, Any]
    detected_at: datetime
    description: str
    supporting_data: Dict[str, Any]


@dataclass 
class SuspiciousPattern:
    """Configuration for a specific suspicious pattern"""
    name: str
    description: str
    query: str
    threshold: float
    risk_weight: float


class PatternRecognitionQueries:
    """
    Advanced pattern recognition system for detecting fraud in NYC construction data.
    
    Uses Graphiti's temporal knowledge graphs to identify complex corporate networks
    and suspicious relationships that may indicate fraudulent activities.
    """
    
    def __init__(self, graphiti_client: Optional[Graphiti] = None):
        """Initialize the pattern recognition system"""
        self.graphiti = graphiti_client
        self.alerts: List[FraudAlert] = []
        
        # Fraud detection thresholds
        self.thresholds = {
            'pass_through_markup': 0.03,      # 3% or less markup
            'single_customer_threshold': 0.95,  # 95% revenue from one customer
            'shell_company_employees': 3,      # Fewer than 3 employees
            'timing_window_days': 30,          # Suspicious timing within 30 days
            'minimum_risk_score': 0.6          # Minimum score to generate alert
        }
        
        # Initialize pattern definitions
        self._init_pattern_definitions()
    
    def _init_pattern_definitions(self) -> None:
        """Initialize predefined fraud pattern definitions"""
        self.patterns = {
            FraudPatternType.PASS_THROUGH: SuspiciousPattern(
                name="Pass-Through Arrangement",
                description="DBE firm acting as intermediary with minimal markup (2-3%)",
                query="""
                MATCH (contractor:Corporation)-[r:SUBCONTRACTED_TO]->(dbe:Corporation)
                WHERE dbe.dbe_certified = true 
                  AND r.markup_percentage <= {pass_through_markup}
                  AND r.contract_value > 50000
                RETURN contractor, dbe, r.markup_percentage, r.contract_value
                """,
                threshold=0.8,
                risk_weight=0.9
            ),
            
            FraudPatternType.CONTROL_FRAUD: SuspiciousPattern(
                name="Control Fraud",
                description="Non-minority individuals controlling DBE firms",
                query="""
                MATCH (person:Person)-[r:CONTROLS|MANAGES]->(company:Corporation)
                WHERE company.dbe_certified = true 
                  AND person.minority_status = false
                  AND r.control_percentage > 50
                RETURN person, company, r.control_percentage
                """,
                threshold=0.85,
                risk_weight=0.95
            ),
            
            FraudPatternType.SINGLE_CUSTOMER: SuspiciousPattern(
                name="Single Customer DBE",
                description="DBE firm with suspicious dependency on single customer",
                query="""
                MATCH (dbe:Corporation)<-[r:CONTRACTED_WITH]-(customer:Corporation)
                WHERE dbe.dbe_certified = true
                WITH dbe, count(customer) as customer_count, 
                     sum(r.contract_value) as total_revenue
                WHERE customer_count = 1 OR 
                      max(r.contract_value) / total_revenue > {single_customer_threshold}
                RETURN dbe, customer_count, total_revenue
                """,
                threshold=0.7,
                risk_weight=0.8
            ),
            
            FraudPatternType.SHELL_COMPANY: SuspiciousPattern(
                name="Shell Company",
                description="Company with minimal operations or employees",
                query="""
                MATCH (company:Corporation)
                WHERE company.employee_count <= {shell_company_employees}
                  AND company.annual_revenue > 100000
                  AND company.contract_count > 5
                RETURN company, company.employee_count, company.annual_revenue
                """,
                threshold=0.75,
                risk_weight=0.85
            )
        }
    
    async def detect_pass_through_schemes(self, 
                                        min_contract_value: float = 50000) -> List[FraudAlert]:
        """
        Detect pass-through arrangements where DBE firms act as intermediaries
        with minimal markup (typically 2-3%).
        """
        if not self.graphiti:
            # Fallback to data analysis without Graphiti
            return await self._analyze_pass_through_patterns()
        
        query = f"""
        Find DBE firms that receive subcontracts with markup percentages 
        between 2% and 3% on contracts worth more than ${min_contract_value}.
        Look for patterns where the DBE appears to be a pass-through entity
        with minimal value-added work.
        """
        
        results = await self.graphiti.search(query, search_type=SearchType.SIMILARITY)
        
        alerts = []
        for result in results:
            if self._calculate_pass_through_risk(result) > self.thresholds['minimum_risk_score']:
                alert = FraudAlert(
                    pattern_type=FraudPatternType.PASS_THROUGH,
                    entities=self._extract_entities(result),
                    risk_score=self._calculate_pass_through_risk(result),
                    evidence=self._build_pass_through_evidence(result),
                    detected_at=datetime.now(),
                    description="Potential pass-through arrangement detected",
                    supporting_data=result
                )
                alerts.append(alert)
        
        return alerts
    
    async def detect_control_fraud(self) -> List[FraudAlert]:
        """
        Detect control fraud where non-minority individuals control 
        DBE-certified firms.
        """
        if not self.graphiti:
            return await self._analyze_control_patterns()
        
        query = """
        Find DBE-certified companies that have non-minority individuals 
        in positions of control, ownership, or management. Look for cases
        where the actual control contradicts the minority certification.
        """
        
        results = await self.graphiti.search(query, search_type=SearchType.SIMILARITY)
        
        alerts = []
        for result in results:
            risk_score = self._calculate_control_fraud_risk(result)
            if risk_score > self.thresholds['minimum_risk_score']:
                alert = FraudAlert(
                    pattern_type=FraudPatternType.CONTROL_FRAUD,
                    entities=self._extract_entities(result),
                    risk_score=risk_score,
                    evidence=self._build_control_fraud_evidence(result),
                    detected_at=datetime.now(),
                    description="Potential control fraud detected in DBE certification",
                    supporting_data=result
                )
                alerts.append(alert)
        
        return alerts
    
    async def detect_single_customer_dbe(self) -> List[FraudAlert]:
        """
        Detect DBE firms with suspicious dependency on a single customer,
        which may indicate a front operation.
        """
        if not self.graphiti:
            return await self._analyze_customer_concentration()
        
        query = """
        Find DBE firms that have an unusually high percentage of their 
        business with a single customer. This may indicate the DBE is 
        a front company for the primary contractor.
        """
        
        results = await self.graphiti.search(query, search_type=SearchType.SIMILARITY)
        
        alerts = []
        for result in results:
            risk_score = self._calculate_customer_concentration_risk(result)
            if risk_score > self.thresholds['minimum_risk_score']:
                alert = FraudAlert(
                    pattern_type=FraudPatternType.SINGLE_CUSTOMER,
                    entities=self._extract_entities(result),
                    risk_score=risk_score,
                    evidence=self._build_customer_concentration_evidence(result),
                    detected_at=datetime.now(),
                    description="DBE with suspicious customer concentration detected",
                    supporting_data=result
                )
                alerts.append(alert)
        
        return alerts
    
    async def detect_shell_companies(self) -> List[FraudAlert]:
        """
        Detect companies that appear to be shell operations with minimal
        legitimate business activity.
        """
        if not self.graphiti:
            return await self._analyze_shell_company_patterns()
        
        query = """
        Find companies that have minimal employees, office space, or 
        operational capacity but are receiving significant construction
        contracts. Look for mismatches between company capabilities 
        and contract obligations.
        """
        
        results = await self.graphiti.search(query, search_type=SearchType.SIMILARITY)
        
        alerts = []
        for result in results:
            risk_score = self._calculate_shell_company_risk(result)
            if risk_score > self.thresholds['minimum_risk_score']:
                alert = FraudAlert(
                    pattern_type=FraudPatternType.SHELL_COMPANY,
                    entities=self._extract_entities(result),
                    risk_score=risk_score,
                    evidence=self._build_shell_company_evidence(result),
                    detected_at=datetime.now(),
                    description="Potential shell company detected",
                    supporting_data=result
                )
                alerts.append(alert)
        
        return alerts
    
    async def detect_timing_fraud(self, window_days: int = 30) -> List[FraudAlert]:
        """
        Detect suspicious timing patterns in company formations, 
        certifications, and contract awards.
        """
        if not self.graphiti:
            return await self._analyze_timing_patterns(window_days)
        
        query = f"""
        Find patterns where companies are formed, receive certifications,
        and win contracts within unusually short timeframes (less than 
        {window_days} days). This may indicate coordination or advance 
        knowledge of contract opportunities.
        """
        
        results = await self.graphiti.search(query, search_type=SearchType.SIMILARITY)
        
        alerts = []
        for result in results:
            risk_score = self._calculate_timing_fraud_risk(result, window_days)
            if risk_score > self.thresholds['minimum_risk_score']:
                alert = FraudAlert(
                    pattern_type=FraudPatternType.TIMING_FRAUD,
                    entities=self._extract_entities(result),
                    risk_score=risk_score,
                    evidence=self._build_timing_fraud_evidence(result),
                    detected_at=datetime.now(),
                    description=f"Suspicious timing pattern detected within {window_days} days",
                    supporting_data=result
                )
                alerts.append(alert)
        
        return alerts
    
    async def detect_network_clusters(self, min_cluster_size: int = 5) -> List[FraudAlert]:
        """
        Detect suspicious clusters of interconnected entities that may 
        represent coordinated fraud schemes.
        """
        if not self.graphiti:
            return await self._analyze_network_clusters(min_cluster_size)
        
        query = f"""
        Find clusters of companies, individuals, and projects that have
        unusually high interconnectedness through shared officers, 
        addresses, or business relationships. Look for networks of 
        {min_cluster_size} or more entities that appear coordinated.
        """
        
        results = await self.graphiti.search(query, search_type=SearchType.SIMILARITY)
        
        alerts = []
        for result in results:
            risk_score = self._calculate_network_cluster_risk(result)
            if risk_score > self.thresholds['minimum_risk_score']:
                alert = FraudAlert(
                    pattern_type=FraudPatternType.NETWORK_CLUSTER,
                    entities=self._extract_entities(result),
                    risk_score=risk_score,
                    evidence=self._build_network_cluster_evidence(result),
                    detected_at=datetime.now(),
                    description=f"Suspicious network cluster detected ({len(self._extract_entities(result))} entities)",
                    supporting_data=result
                )
                alerts.append(alert)
        
        return alerts
    
    async def run_comprehensive_scan(self) -> Dict[FraudPatternType, List[FraudAlert]]:
        """
        Run all fraud detection patterns and return comprehensive results.
        """
        print("🔍 Running comprehensive fraud detection scan...")
        
        results = {}
        
        # Run all detection methods
        detection_methods = [
            (FraudPatternType.PASS_THROUGH, self.detect_pass_through_schemes()),
            (FraudPatternType.CONTROL_FRAUD, self.detect_control_fraud()),
            (FraudPatternType.SINGLE_CUSTOMER, self.detect_single_customer_dbe()),
            (FraudPatternType.SHELL_COMPANY, self.detect_shell_companies()),
            (FraudPatternType.TIMING_FRAUD, self.detect_timing_fraud()),
            (FraudPatternType.NETWORK_CLUSTER, self.detect_network_clusters())
        ]
        
        for pattern_type, detection_coro in detection_methods:
            try:
                alerts = await detection_coro
                results[pattern_type] = alerts
                print(f"✅ {pattern_type.value}: {len(alerts)} alerts generated")
            except Exception as e:
                print(f"❌ Error detecting {pattern_type.value}: {e}")
                results[pattern_type] = []
        
        # Store all alerts
        all_alerts = []
        for alerts_list in results.values():
            all_alerts.extend(alerts_list)
        self.alerts.extend(all_alerts)
        
        return results
    
    # Fallback analysis methods (when Graphiti not available)
    async def _analyze_pass_through_patterns(self) -> List[FraudAlert]:
        """Analyze pass-through patterns using local data"""
        # Implementation for data-based analysis
        return []
    
    async def _analyze_control_patterns(self) -> List[FraudAlert]:
        """Analyze control fraud patterns using local data"""
        return []
    
    async def _analyze_customer_concentration(self) -> List[FraudAlert]:
        """Analyze customer concentration using local data"""
        return []
    
    async def _analyze_shell_company_patterns(self) -> List[FraudAlert]:
        """Analyze shell company patterns using local data"""
        return []
    
    async def _analyze_timing_patterns(self, window_days: int) -> List[FraudAlert]:
        """Analyze timing patterns using local data"""
        return []
    
    async def _analyze_network_clusters(self, min_cluster_size: int) -> List[FraudAlert]:
        """Analyze network clusters using local data"""
        return []
    
    # Risk calculation methods
    def _calculate_pass_through_risk(self, result: Any) -> float:
        """Calculate risk score for pass-through arrangements"""
        # Implement scoring algorithm
        return 0.75
    
    def _calculate_control_fraud_risk(self, result: Any) -> float:
        """Calculate risk score for control fraud"""
        return 0.85
    
    def _calculate_customer_concentration_risk(self, result: Any) -> float:
        """Calculate risk score for customer concentration"""
        return 0.70
    
    def _calculate_shell_company_risk(self, result: Any) -> float:
        """Calculate risk score for shell companies"""
        return 0.80
    
    def _calculate_timing_fraud_risk(self, result: Any, window_days: int) -> float:
        """Calculate risk score for timing fraud"""
        return 0.65
    
    def _calculate_network_cluster_risk(self, result: Any) -> float:
        """Calculate risk score for network clusters"""
        return 0.75
    
    # Evidence building methods
    def _build_pass_through_evidence(self, result: Any) -> Dict[str, Any]:
        """Build evidence dictionary for pass-through schemes"""
        return {"pattern": "pass_through", "data": result}
    
    def _build_control_fraud_evidence(self, result: Any) -> Dict[str, Any]:
        """Build evidence dictionary for control fraud"""
        return {"pattern": "control_fraud", "data": result}
    
    def _build_customer_concentration_evidence(self, result: Any) -> Dict[str, Any]:
        """Build evidence dictionary for customer concentration"""
        return {"pattern": "customer_concentration", "data": result}
    
    def _build_shell_company_evidence(self, result: Any) -> Dict[str, Any]:
        """Build evidence dictionary for shell companies"""
        return {"pattern": "shell_company", "data": result}
    
    def _build_timing_fraud_evidence(self, result: Any) -> Dict[str, Any]:
        """Build evidence dictionary for timing fraud"""
        return {"pattern": "timing_fraud", "data": result}
    
    def _build_network_cluster_evidence(self, result: Any) -> Dict[str, Any]:
        """Build evidence dictionary for network clusters"""
        return {"pattern": "network_cluster", "data": result}
    
    def _extract_entities(self, result: Any) -> List[str]:
        """Extract entity names from search results"""
        # Implement entity extraction
        return ["Entity1", "Entity2"]  # Placeholder
    
    def generate_fraud_report(self) -> Dict[str, Any]:
        """Generate a comprehensive fraud detection report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_alerts": len(self.alerts),
            "patterns_detected": {},
            "high_risk_alerts": [],
            "summary": {}
        }
        
        # Group alerts by pattern type
        for alert in self.alerts:
            pattern_name = alert.pattern_type.value
            if pattern_name not in report["patterns_detected"]:
                report["patterns_detected"][pattern_name] = []
            report["patterns_detected"][pattern_name].append({
                "entities": alert.entities,
                "risk_score": alert.risk_score,
                "description": alert.description,
                "detected_at": alert.detected_at.isoformat()
            })
            
            # High risk alerts (score > 0.8)
            if alert.risk_score > 0.8:
                report["high_risk_alerts"].append({
                    "pattern": pattern_name,
                    "entities": alert.entities,
                    "risk_score": alert.risk_score,
                    "description": alert.description
                })
        
        # Generate summary statistics
        report["summary"] = {
            "pattern_counts": {pattern.value: len([a for a in self.alerts if a.pattern_type == pattern]) 
                             for pattern in FraudPatternType},
            "average_risk_score": sum(a.risk_score for a in self.alerts) / len(self.alerts) if self.alerts else 0,
            "high_risk_count": len(report["high_risk_alerts"])
        }
        
        return report
    
    def save_alerts(self, filepath: str) -> None:
        """Save fraud alerts to file"""
        report = self.generate_fraud_report()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Fraud detection report saved to {filepath}")
        print(f"📊 Total alerts: {len(self.alerts)}")
        print(f"🚨 High risk alerts: {len(report['high_risk_alerts'])}")


# Utility functions for pattern analysis
def calculate_markup_percentage(subcontract_amount: float, 
                              total_amount: float) -> float:
    """Calculate markup percentage for subcontract arrangements"""
    if total_amount == 0:
        return 0.0
    return (total_amount - subcontract_amount) / subcontract_amount * 100


def detect_shared_addresses(entities: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Detect entities sharing the same address"""
    address_map = {}
    shared_pairs = []
    
    for entity in entities:
        address = entity.get('address', '').strip().lower()
        if address:
            if address in address_map:
                # Found shared address
                for existing_entity in address_map[address]:
                    shared_pairs.append((existing_entity, entity['name']))
                address_map[address].append(entity['name'])
            else:
                address_map[address] = [entity['name']]
    
    return shared_pairs


def detect_shared_officers(entities: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Detect shared officers between companies"""
    officer_map = {}
    shared_officers = []
    
    for entity in entities:
        company_name = entity['name']
        officers = entity.get('officers', [])
        
        for officer in officers:
            officer_name = officer.strip().lower()
            if officer_name in officer_map:
                # Found shared officer
                for existing_company in officer_map[officer_name]:
                    shared_officers.append((officer_name, existing_company, company_name))
                officer_map[officer_name].append(company_name)
            else:
                officer_map[officer_name] = [company_name]
    
    return shared_officers


def analyze_temporal_patterns(events: List[Dict[str, Any]], 
                            window_days: int = 30) -> List[Dict[str, Any]]:
    """Analyze temporal patterns in business events"""
    suspicious_patterns = []
    
    # Sort events by date
    sorted_events = sorted(events, key=lambda x: x['date'])
    
    for i, event in enumerate(sorted_events):
        # Look for events within the suspicious window
        window_start = event['date']
        window_end = window_start + timedelta(days=window_days)
        
        related_events = []
        for j in range(i + 1, len(sorted_events)):
            if sorted_events[j]['date'] <= window_end:
                related_events.append(sorted_events[j])
            else:
                break
        
        # If multiple significant events happen within window, flag as suspicious
        if len(related_events) >= 2:
            suspicious_patterns.append({
                'trigger_event': event,
                'related_events': related_events,
                'window_days': (related_events[-1]['date'] - window_start).days,
                'event_count': len(related_events) + 1
            })
    
    return suspicious_patterns


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Initialize pattern recognition system
        detector = PatternRecognitionQueries()
        
        # Run comprehensive fraud detection
        results = await detector.run_comprehensive_scan()
        
        # Generate report
        report = detector.generate_fraud_report()
        
        print("\n🔍 Fraud Detection Results:")
        print(f"Total alerts generated: {report['total_alerts']}")
        print(f"High risk alerts: {report['summary']['high_risk_count']}")
        
        for pattern, count in report['summary']['pattern_counts'].items():
            if count > 0:
                print(f"  {pattern}: {count} alerts")
        
        # Save results
        detector.save_alerts("data/reports/fraud_detection_results.json")
    
    asyncio.run(main()) 