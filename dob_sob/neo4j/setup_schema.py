#!/usr/bin/env python3
"""
NYC DOB Fraud Detection - Neo4j Schema Setup Script

This script sets up the complete Neo4j schema for fraud detection including:
1. Constraints and indexes for performance
2. Data quality procedures
3. Fraud detection query validation
4. Sample data testing

Usage:
    python setup_schema.py [--reset] [--test-queries]
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, ClientError
except ImportError:
    print("Error: neo4j package not installed. Run: pip install neo4j")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Neo4jSchemaSetup:
    """Handles Neo4j schema setup and validation for fraud detection."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = "password"):
        """Initialize Neo4j connection."""
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.schema_dir = Path(__file__).parent / "schema"
        
    def connect(self) -> bool:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("✅ Successfully connected to Neo4j")
                    return True
        except ServiceUnavailable as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            logger.error("Make sure Neo4j is running: docker compose up neo4j -d")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting to Neo4j: {e}")
            return False
        
        return False
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_cypher_file(self, file_path: Path) -> bool:
        """Execute a Cypher file with proper error handling."""
        if not file_path.exists():
            logger.error(f"❌ Schema file not found: {file_path}")
            return False
        
        logger.info(f"📄 Executing schema file: {file_path.name}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Split content by semicolons and filter out comments/empty lines
            statements = []
            for statement in content.split(';'):
                statement = statement.strip()
                if statement and not statement.startswith('//') and not statement.startswith('/*'):
                    statements.append(statement)
            
            with self.driver.session() as session:
                success_count = 0
                for i, statement in enumerate(statements):
                    try:
                        result = session.run(statement)
                        # Consume result to ensure execution
                        summary = result.consume()
                        success_count += 1
                        
                        # Log progress for long operations
                        if i % 10 == 0 and len(statements) > 20:
                            logger.info(f"  Progress: {i+1}/{len(statements)} statements")
                            
                    except ClientError as e:
                        # Some errors are expected (e.g., constraint already exists)
                        if "already exists" in str(e) or "equivalent" in str(e):
                            logger.debug(f"  Skipping existing constraint/index: {e}")
                            success_count += 1
                        else:
                            logger.warning(f"  ⚠️ Error in statement {i+1}: {e}")
                            logger.debug(f"  Statement: {statement[:100]}...")
                
                logger.info(f"✅ Executed {success_count}/{len(statements)} statements successfully")
                return success_count > 0
                
        except Exception as e:
            logger.error(f"❌ Error executing {file_path.name}: {e}")
            return False
    
    def setup_core_schema(self) -> bool:
        """Set up core schema with constraints and indexes."""
        logger.info("🏗️ Setting up core schema...")
        return self.execute_cypher_file(self.schema_dir / "01_core_schema.cypher")
    
    def setup_data_quality(self) -> bool:
        """Set up data quality procedures."""
        logger.info("🧹 Setting up data quality procedures...")
        return self.execute_cypher_file(self.schema_dir / "03_data_quality_procedures.cypher")
    
    def validate_schema(self) -> Dict[str, Any]:
        """Validate that schema was created correctly."""
        logger.info("🔍 Validating schema setup...")
        
        validation_results = {
            "constraints": [],
            "indexes": [],
            "node_labels": [],
            "relationship_types": []
        }
        
        try:
            with self.driver.session() as session:
                # Check constraints
                result = session.run("SHOW CONSTRAINTS")
                constraints = [record["name"] for record in result]
                validation_results["constraints"] = constraints
                logger.info(f"  📋 Found {len(constraints)} constraints")
                
                # Check indexes
                result = session.run("SHOW INDEXES")
                indexes = [record["name"] for record in result]
                validation_results["indexes"] = indexes
                logger.info(f"  📊 Found {len(indexes)} indexes")
                
                # Check if we can create sample nodes (schema validation)
                test_queries = [
                    "CREATE (o:Owner {owner_id: 'test_001', name: 'Test Owner'}) RETURN o",
                    "CREATE (p:Property {bin: 999999, address: 'Test Address'}) RETURN p",
                    "CREATE (c:Contractor {license_sl_no: 999999, business_name: 'Test Contractor'}) RETURN c"
                ]
                
                for query in test_queries:
                    try:
                        result = session.run(query)
                        result.consume()
                        logger.debug("  ✅ Schema validation query passed")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Schema validation issue: {e}")
                
                # Clean up test data
                session.run("MATCH (n) WHERE n.owner_id = 'test_001' OR n.bin = 999999 OR n.license_sl_no = 999999 DELETE n")
                
        except Exception as e:
            logger.error(f"❌ Error validating schema: {e}")
        
        return validation_results
    
    def test_fraud_queries(self) -> bool:
        """Test that fraud detection queries can be parsed."""
        logger.info("🕵️ Testing fraud detection queries...")
        
        query_file = self.schema_dir / "02_fraud_detection_queries.cypher"
        if not query_file.exists():
            logger.warning("⚠️ Fraud detection queries file not found")
            return False
        
        try:
            with open(query_file, 'r') as f:
                content = f.read()
            
            # Extract individual queries (simplified parsing)
            queries = []
            current_query = []
            in_query = False
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('MATCH') or line.startswith('WITH') or line.startswith('RETURN'):
                    in_query = True
                    current_query.append(line)
                elif in_query and (line.startswith('//') or line == ''):
                    if current_query:
                        queries.append('\n'.join(current_query))
                        current_query = []
                    in_query = False
                elif in_query:
                    current_query.append(line)
            
            # Add last query if exists
            if current_query:
                queries.append('\n'.join(current_query))
            
            logger.info(f"  📝 Found {len(queries)} fraud detection queries")
            
            # Test query syntax (without executing)
            with self.driver.session() as session:
                valid_queries = 0
                for i, query in enumerate(queries[:3]):  # Test first 3 queries
                    try:
                        # Use EXPLAIN to validate syntax without execution
                        result = session.run(f"EXPLAIN {query}")
                        result.consume()
                        valid_queries += 1
                        logger.debug(f"  ✅ Query {i+1} syntax valid")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Query {i+1} syntax issue: {e}")
                
                logger.info(f"  ✅ {valid_queries}/{min(len(queries), 3)} test queries validated")
                return valid_queries > 0
                
        except Exception as e:
            logger.error(f"❌ Error testing fraud queries: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get current database statistics."""
        stats = {}
        
        try:
            with self.driver.session() as session:
                # Node counts
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                stats["total_nodes"] = result.single()["node_count"]
                
                # Relationship counts
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                stats["total_relationships"] = result.single()["rel_count"]
                
                # Label counts
                result = session.run("CALL db.labels()")
                stats["node_labels"] = len(list(result))
                
                # Relationship type counts
                result = session.run("CALL db.relationshipTypes()")
                stats["relationship_types"] = len(list(result))
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
        
        return stats
    
    def reset_database(self) -> bool:
        """Reset database by removing all nodes and relationships."""
        logger.warning("🗑️ Resetting database (removing all data)...")
        
        try:
            with self.driver.session() as session:
                # Remove all nodes and relationships
                result = session.run("MATCH (n) DETACH DELETE n")
                summary = result.consume()
                
                # Drop all constraints
                constraints_result = session.run("SHOW CONSTRAINTS")
                for record in constraints_result:
                    constraint_name = record["name"]
                    try:
                        session.run(f"DROP CONSTRAINT {constraint_name}")
                        logger.debug(f"  Dropped constraint: {constraint_name}")
                    except Exception as e:
                        logger.debug(f"  Could not drop constraint {constraint_name}: {e}")
                
                # Drop all indexes
                indexes_result = session.run("SHOW INDEXES")
                for record in indexes_result:
                    index_name = record["name"]
                    if not index_name.startswith("system"):  # Don't drop system indexes
                        try:
                            session.run(f"DROP INDEX {index_name}")
                            logger.debug(f"  Dropped index: {index_name}")
                        except Exception as e:
                            logger.debug(f"  Could not drop index {index_name}: {e}")
                
                logger.info("✅ Database reset completed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error resetting database: {e}")
            return False

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up Neo4j schema for fraud detection")
    parser.add_argument("--reset", action="store_true", help="Reset database before setup")
    parser.add_argument("--test-queries", action="store_true", help="Test fraud detection queries")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    
    args = parser.parse_args()
    
    # Initialize schema setup
    schema_setup = Neo4jSchemaSetup(args.uri, args.user, args.password)
    
    try:
        # Connect to Neo4j
        if not schema_setup.connect():
            sys.exit(1)
        
        # Reset database if requested
        if args.reset:
            if not schema_setup.reset_database():
                logger.error("Failed to reset database")
                sys.exit(1)
        
        # Get initial stats
        initial_stats = schema_setup.get_database_stats()
        logger.info(f"📊 Initial database stats: {initial_stats}")
        
        # Set up core schema
        if not schema_setup.setup_core_schema():
            logger.error("Failed to set up core schema")
            sys.exit(1)
        
        # Set up data quality procedures
        if not schema_setup.setup_data_quality():
            logger.error("Failed to set up data quality procedures")
            sys.exit(1)
        
        # Validate schema
        validation_results = schema_setup.validate_schema()
        logger.info("✅ Schema validation completed")
        
        # Test fraud queries if requested
        if args.test_queries:
            if not schema_setup.test_fraud_queries():
                logger.warning("Some fraud detection queries failed validation")
        
        # Get final stats
        final_stats = schema_setup.get_database_stats()
        logger.info(f"📊 Final database stats: {final_stats}")
        
        logger.info("🎉 Neo4j schema setup completed successfully!")
        logger.info("Next steps:")
        logger.info("  1. Load sample data to test the schema")
        logger.info("  2. Run fraud detection queries")
        logger.info("  3. Set up ETL pipeline for full dataset")
        
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        schema_setup.close()

if __name__ == "__main__":
    main() 