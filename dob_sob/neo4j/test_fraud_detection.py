#!/usr/bin/env python3
"""
Test fraud detection queries on the expanded NYC dataset.
"""

import logging
from neo4j import GraphDatabase
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetectionTester:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize the fraud detection tester."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("✅ Connected to Neo4j for fraud detection testing")
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def run_query(self, query: str, description: str) -> List[Dict[str, Any]]:
        """Run a query and return results."""
        logger.info(f"🔍 {description}")
        
        with self.driver.session() as session:
            result = session.run(query)
            records = [record.data() for record in result]
            logger.info(f"  📊 Found {len(records)} results")
            return records
    
    def test_database_stats(self):
        """Test basic database statistics."""
        query = """
        MATCH (n)
        RETURN labels(n) as node_type, count(n) as count
        ORDER BY count DESC
        """
        
        results = self.run_query(query, "Database node statistics")
        for record in results:
            logger.info(f"  📈 {record['node_type']}: {record['count']}")
    
    def test_shell_company_detection(self):
        """Test shell company detection across all violation types."""
        query = """
        // Find properties with multiple violations from different sources
        MATCH (p:Property)
        OPTIONAL MATCH (p)<-[:VIOLATION_AT]-(dob:Violation)
        OPTIONAL MATCH (p)<-[:VIOLATION_AT]-(ecb:ECB_Violation)
        OPTIONAL MATCH (p)<-[:JOB_APPLICATION_AT]-(job:Job_Application)
        
        WITH p, 
             count(dob) as dob_violations,
             count(ecb) as ecb_violations,
             count(job) as job_applications,
             (count(dob) + count(ecb)) as total_violations
        
        WHERE total_violations >= 2
        
        RETURN p.bin as bin,
               p.address as address,
               p.borough as borough,
               dob_violations,
               ecb_violations,
               job_applications,
               total_violations
        ORDER BY total_violations DESC
        LIMIT 10
        """
        
        results = self.run_query(query, "Shell company detection - Properties with multiple violations")
        for i, record in enumerate(results[:5], 1):
            logger.info(f"  🏢 #{i}: BIN {record['bin']} - {record['address']} ({record['borough']})")
            logger.info(f"      DOB: {record['dob_violations']}, ECB: {record['ecb_violations']}, Jobs: {record['job_applications']}")
    
    def test_respondent_analysis(self):
        """Test analysis of ECB violation respondents."""
        query = """
        // Find respondents with multiple ECB violations
        MATCH (r:Respondent)<-[:ISSUED_TO]-(v:ECB_Violation)-[:VIOLATION_AT]->(p:Property)
        
        WITH r, 
             count(v) as violation_count,
             count(DISTINCT p) as property_count,
             collect(DISTINCT p.address) as addresses
        
        WHERE violation_count >= 2
        
        RETURN r.name as respondent_name,
               r.address as respondent_address,
               violation_count,
               property_count,
               addresses[0..3] as sample_addresses
        ORDER BY violation_count DESC
        LIMIT 10
        """
        
        results = self.run_query(query, "ECB Respondent analysis - Multiple violations")
        for i, record in enumerate(results[:5], 1):
            logger.info(f"  👤 #{i}: {record['respondent_name']}")
            logger.info(f"      Violations: {record['violation_count']}, Properties: {record['property_count']}")
    
    def test_professional_network_analysis(self):
        """Test professional network analysis from job applications."""
        query = """
        // Find professionals filing applications for multiple properties
        MATCH (prof:Professional)-[:FILED_APPLICATION]->(job:Job_Application)-[:JOB_APPLICATION_AT]->(p:Property)
        
        WITH prof,
             count(job) as application_count,
             count(DISTINCT p) as property_count,
             collect(DISTINCT p.address) as addresses
        
        WHERE application_count >= 2
        
        RETURN prof.name as professional_name,
               prof.title as professional_title,
               prof.license_number as license_number,
               application_count,
               property_count,
               addresses[0..3] as sample_addresses
        ORDER BY application_count DESC
        LIMIT 10
        """
        
        results = self.run_query(query, "Professional network analysis - Multiple applications")
        for i, record in enumerate(results[:5], 1):
            logger.info(f"  👨‍💼 #{i}: {record['professional_name']} ({record['professional_title']})")
            logger.info(f"      Applications: {record['application_count']}, Properties: {record['property_count']}")
    
    def test_cross_dataset_fraud_patterns(self):
        """Test fraud patterns across different datasets."""
        query = """
        // Find properties with violations AND job applications from same owner
        MATCH (o:Owner)-[:OWNS]->(p:Property)
        OPTIONAL MATCH (p)<-[:VIOLATION_AT]-(v:Violation)
        OPTIONAL MATCH (p)<-[:VIOLATION_AT]-(ecb:ECB_Violation)
        OPTIONAL MATCH (p)<-[:JOB_APPLICATION_AT]-(job:Job_Application)
        
        WITH o, p,
             count(v) as dob_violations,
             count(ecb) as ecb_violations,
             count(job) as job_applications,
             (count(v) + count(ecb)) as total_violations
        
        WHERE total_violations > 0 AND job_applications > 0
        
        RETURN o.name as owner_name,
               count(p) as properties_owned,
               sum(dob_violations) as total_dob_violations,
               sum(ecb_violations) as total_ecb_violations,
               sum(job_applications) as total_job_applications,
               sum(total_violations) as total_violations
        ORDER BY total_violations DESC
        LIMIT 10
        """
        
        results = self.run_query(query, "Cross-dataset fraud patterns - Owners with violations and job applications")
        for i, record in enumerate(results[:5], 1):
            logger.info(f"  🏗️ #{i}: {record['owner_name']}")
            logger.info(f"      Properties: {record['properties_owned']}, Total violations: {record['total_violations']}")
            logger.info(f"      Job applications: {record['total_job_applications']}")
    
    def test_geographic_clustering(self):
        """Test geographic clustering of violations."""
        query = """
        // Find boroughs with highest violation density
        MATCH (p:Property)
        OPTIONAL MATCH (p)<-[:VIOLATION_AT]-(dob:Violation)
        OPTIONAL MATCH (p)<-[:VIOLATION_AT]-(ecb:ECB_Violation)
        
        WITH p.borough as borough,
             count(DISTINCT p) as properties,
             count(dob) as dob_violations,
             count(ecb) as ecb_violations,
             (count(dob) + count(ecb)) as total_violations
        
        WHERE borough IS NOT NULL AND borough <> ''
        
        RETURN borough,
               properties,
               dob_violations,
               ecb_violations,
               total_violations,
               round(toFloat(total_violations) / properties, 2) as violations_per_property
        ORDER BY violations_per_property DESC
        """
        
        results = self.run_query(query, "Geographic clustering - Violation density by borough")
        for record in results:
            logger.info(f"  🗺️ {record['borough']}: {record['violations_per_property']} violations/property")
            logger.info(f"      Properties: {record['properties']}, Total violations: {record['total_violations']}")
    
    def test_penalty_analysis(self):
        """Test penalty analysis from ECB violations."""
        query = """
        // Analyze ECB violation penalties
        MATCH (ecb:ECB_Violation)
        WHERE ecb.penalty_imposed IS NOT NULL
        
        WITH ecb.violation_type as violation_type,
             count(ecb) as violation_count,
             avg(ecb.penalty_imposed) as avg_penalty,
             sum(ecb.penalty_imposed) as total_penalties,
             max(ecb.penalty_imposed) as max_penalty
        
        WHERE violation_count >= 5
        
        RETURN violation_type,
               violation_count,
               round(avg_penalty, 2) as avg_penalty,
               round(total_penalties, 2) as total_penalties,
               max_penalty
        ORDER BY total_penalties DESC
        LIMIT 10
        """
        
        results = self.run_query(query, "ECB Penalty analysis - By violation type")
        for i, record in enumerate(results[:5], 1):
            logger.info(f"  💰 #{i}: {record['violation_type']}")
            logger.info(f"      Count: {record['violation_count']}, Avg penalty: ${record['avg_penalty']}")
    
    def run_all_tests(self):
        """Run all fraud detection tests."""
        logger.info("🚀 Starting comprehensive fraud detection testing...")
        
        self.test_database_stats()
        print()
        
        self.test_shell_company_detection()
        print()
        
        self.test_respondent_analysis()
        print()
        
        self.test_professional_network_analysis()
        print()
        
        self.test_cross_dataset_fraud_patterns()
        print()
        
        self.test_geographic_clustering()
        print()
        
        self.test_penalty_analysis()
        print()
        
        logger.info("✅ All fraud detection tests completed!")

def main():
    """Main function."""
    # Connection parameters
    uri = "bolt://localhost:37687"
    user = "neo4j"
    password = "password"
    
    tester = FraudDetectionTester(uri, user, password)
    
    try:
        tester.run_all_tests()
    finally:
        tester.close()

if __name__ == "__main__":
    main() 