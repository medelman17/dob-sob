#!/usr/bin/env python3
"""
NYC DOB Fraud Detection - Data Loading Script

This script loads NYC DOB data into Neo4j for fraud detection analysis.
It follows the same pattern as setup_schema.py for consistency.

Usage:
    python load_data.py --dataset dob_violations --limit 1000
    python load_data.py --dataset all --limit 10000
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

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

class Neo4jDataLoader:
    """Loads NYC DOB data into Neo4j for fraud detection."""
    
    def __init__(self, uri: str = "bolt://localhost:37687", 
                 user: str = "neo4j", password: str = "password"):
        """Initialize Neo4j connection."""
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        
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
    
    def discover_datasets(self) -> List[str]:
        """Discover available datasets."""
        datasets = []
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return datasets
        
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                csv_files = list(subdir.glob("*.csv"))
                if csv_files:
                    datasets.append(subdir.name)
        
        return sorted(datasets)
    
    def get_dataset_files(self, dataset_name: str) -> List[Path]:
        """Get all CSV files for a dataset."""
        dataset_dir = self.data_dir / dataset_name
        if dataset_dir.exists() and dataset_dir.is_dir():
            return sorted(dataset_dir.glob("*.csv"))
        return []
    
    def load_dataset(self, dataset_name: str, limit: Optional[int] = None) -> bool:
        """Load a specific dataset into Neo4j."""
        logger.info(f"📋 Loading {dataset_name} data...")
        
        dataset_dir = self.data_dir / dataset_name
        if not dataset_dir.exists():
            logger.error(f"❌ Dataset directory not found: {dataset_dir}")
            return False
        
        csv_files = list(dataset_dir.glob("*.csv"))
        if not csv_files:
            logger.error(f"❌ No CSV files found in {dataset_dir}")
            return False
        
        total_loaded = 0
        
        for csv_file in csv_files:
            logger.info(f"📄 Processing file: {csv_file.name}")
            
            try:
                # Read CSV file info
                df = pd.read_csv(csv_file, low_memory=False)
                logger.info(f"  📊 Found {len(df)} records in {csv_file.name}")
                
                # Apply limit if specified
                if limit and total_loaded >= limit:
                    logger.info(f"  ⏹️ Reached limit of {limit} records")
                    break
                
                # Calculate how many records to process from this file
                remaining_limit = limit - total_loaded if limit else len(df)
                records_to_process = min(len(df), remaining_limit)
                df_to_process = df.head(records_to_process)
                
                # Process in batches
                batch_size = 1000
                for i in range(0, len(df_to_process), batch_size):
                    batch_df = df_to_process.iloc[i:i+batch_size]
                    
                    # Route to appropriate loader based on dataset
                    success = False
                    if dataset_name == 'dob_violations':
                        success = self._load_violations_batch(batch_df)
                    elif dataset_name == 'ecb_violations':
                        success = self._load_ecb_violations_batch(batch_df)
                    elif dataset_name == 'job_application_filings':
                        success = self._load_job_applications_batch(batch_df)
                    else:
                        logger.warning(f"⚠️ Unknown dataset type: {dataset_name}")
                        continue
                    
                    if success:
                        batch_count = len(batch_df)
                        total_loaded += batch_count
                        logger.info(f"  ✅ Loaded batch {(i//batch_size)+1}: {batch_count} records (total: {total_loaded})")
                    else:
                        logger.error(f"  ❌ Failed to load batch {(i//batch_size)+1}")
                        return False
                    
                    # Check if we've reached the limit
                    if limit and total_loaded >= limit:
                        break
                
            except Exception as e:
                logger.error(f"❌ Error processing {csv_file.name}: {e}")
                return False
        
        logger.info(f"🎉 Successfully loaded {total_loaded} {dataset_name}")
        return True
    
    def _load_violations_batch(self, batch_df: pd.DataFrame) -> bool:
        """Load a batch of violations into Neo4j."""
        try:
            # Preprocess dates in Python
            def format_date(date_val):
                if pd.isna(date_val) or date_val == '' or str(date_val) == 'nan':
                    return None
                try:
                    # Handle text dates like "JUNE 31" by returning None
                    date_str = str(date_val).strip().upper()
                    if any(month in date_str for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']):
                        return None  # Skip text dates
                    
                    # Try to convert to int for numeric dates
                    date_str = str(int(float(date_val)))  # Handle decimals
                    if len(date_str) == 8:  # YYYYMMDD format
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        
                        # Validate date components
                        if year < 1900 or year > 2030:
                            return None
                        if month < 1 or month > 12:
                            return None
                        if day < 1 or day > 31:
                            return None
                        
                        # Additional validation for days in month
                        if month in [4, 6, 9, 11] and day > 30:  # April, June, Sept, Nov
                            return None
                        if month == 2 and day > 29:  # February
                            return None
                            
                        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    return None
                except (ValueError, TypeError):
                    return None
            
            # Apply date formatting
            batch_df = batch_df.copy()
            batch_df['formatted_issue_date'] = batch_df['ISSUE_DATE'].apply(format_date)
            batch_df['formatted_disposition_date'] = batch_df['DISPOSITION_DATE'].apply(format_date)
            
            with self.driver.session() as session:
                # Convert DataFrame to list of dicts
                records = batch_df.to_dict('records')
                
                # Cypher query to create violations and related entities
                query = """
                UNWIND $records AS row
                
                // Create or merge Property using BIN
                MERGE (p:Property {bin: toString(row.BIN)})
                ON CREATE SET 
                    p.address = toString(row.HOUSE_NUMBER) + ' ' + toString(row.STREET),
                    p.borough = row.BORO,
                    p.block = toString(row.BLOCK),
                    p.lot = toString(row.LOT)
                
                // Create or merge Violation using unique identifier
                WITH p, row
                MERGE (v:Violation {isn_dob_bis_viol: toString(row.ISN_DOB_BIS_VIOL)})
                ON CREATE SET 
                    v.violation_number = toString(row.VIOLATION_NUMBER),
                    v.violation_category = row.VIOLATION_CATEGORY,
                    v.violation_type = row.VIOLATION_TYPE,
                    v.violation_type_code = row.VIOLATION_TYPE_CODE,
                    v.issue_date = CASE WHEN row.formatted_issue_date IS NOT NULL 
                               THEN date(row.formatted_issue_date) ELSE null END,
                    v.disposition_date = CASE WHEN row.formatted_disposition_date IS NOT NULL 
                                     THEN date(row.formatted_disposition_date) ELSE null END,
                    v.disposition_comments = row.DISPOSITION_COMMENTS,
                    v.device_number = toString(row.DEVICE_NUMBER),
                    v.description = row.DESCRIPTION,
                    v.ecb_number = toString(row.ECB_NUMBER),
                    v.number = toString(row.NUMBER),
                    v.street = row.STREET
                
                // Link violation to property (avoid duplicates)
                MERGE (v)-[:VIOLATION_AT]->(p)
                """
                
                result = session.run(query, records=records)
                summary = result.consume()
                
                return True
                
        except Exception as e:
            logger.warning(f"Date validation error in batch, skipping problematic records: {e}")
            # Try to process records individually to skip only problematic ones
            try:
                with self.driver.session() as session:
                    successful_records = 0
                    for record in records:
                        try:
                            result = session.run(query, records=[record])
                            summary = result.consume()
                            successful_records += 1
                        except Exception as record_error:
                            logger.debug(f"Skipping record due to error: {record_error}")
                            continue
                    
                    logger.info(f"  ✅ Processed {successful_records}/{len(records)} records from batch (skipped {len(records) - successful_records} problematic records)")
                    return True
            except Exception as fallback_error:
                logger.error(f"Error in fallback processing: {fallback_error}")
                return False
    
    def _load_ecb_violations_batch(self, batch_df: pd.DataFrame) -> bool:
        """Load a batch of ECB violations into Neo4j."""
        try:
            # Clean data - replace NaN with empty strings
            batch_df = batch_df.fillna('')
            
            # Preprocess dates in Python
            def format_date(date_val):
                if pd.isna(date_val) or date_val == '' or str(date_val) == 'nan':
                    return None
                try:
                    # Handle text dates like "JUNE 31" by returning None
                    date_str = str(date_val).strip().upper()
                    if any(month in date_str for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']):
                        return None  # Skip text dates
                    
                    # Try to convert to int for numeric dates
                    date_str = str(int(float(date_val)))  # Handle decimals
                    if len(date_str) == 8:  # YYYYMMDD format
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        
                        # Validate date components
                        if year < 1900 or year > 2030:
                            return None
                        if month < 1 or month > 12:
                            return None
                        if day < 1 or day > 31:
                            return None
                        
                        # Additional validation for days in month
                        if month in [4, 6, 9, 11] and day > 30:  # April, June, Sept, Nov
                            return None
                        if month == 2 and day > 29:  # February
                            return None
                            
                        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    return None
                except (ValueError, TypeError):
                    return None
            
            # Apply date formatting
            batch_df = batch_df.copy()
            batch_df['formatted_issue_date'] = batch_df['ISSUE_DATE'].apply(format_date)
            batch_df['formatted_served_date'] = batch_df['SERVED_DATE'].apply(format_date)
            batch_df['formatted_hearing_date'] = batch_df['HEARING_DATE'].apply(format_date)
            
            with self.driver.session() as session:
                # Convert DataFrame to list of dicts
                records = batch_df.to_dict('records')
                
                # Cypher query to create ECB violations and related entities
                query = """
                UNWIND $records AS row
                
                // Create or merge Property using BIN
                MERGE (p:Property {bin: toString(row.BIN)})
                ON CREATE SET 
                    p.borough = row.BORO,
                    p.block = toString(row.BLOCK),
                    p.lot = toString(row.LOT)
                
                // Create or merge ECB Violation using unique identifier
                WITH p, row
                MERGE (v:ECB_Violation {isn_dob_bis_extract: toString(row.ISN_DOB_BIS_EXTRACT)})
                ON CREATE SET
                    v.ecb_violation_number = toString(row.ECB_VIOLATION_NUMBER),
                    v.ecb_violation_status = row.ECB_VIOLATION_STATUS,
                    v.dob_violation_number = toString(row.DOB_VIOLATION_NUMBER),
                    v.issue_date = CASE WHEN row.formatted_issue_date IS NOT NULL 
                               THEN date(row.formatted_issue_date) ELSE null END,
                    v.served_date = CASE WHEN row.formatted_served_date IS NOT NULL 
                                THEN date(row.formatted_served_date) ELSE null END,
                    v.hearing_date = CASE WHEN row.formatted_hearing_date IS NOT NULL 
                                 THEN date(row.formatted_hearing_date) ELSE null END,
                    v.hearing_time = row.HEARING_TIME,
                    v.severity = row.SEVERITY,
                    v.violation_type = row.VIOLATION_TYPE,
                    v.violation_description = row.VIOLATION_DESCRIPTION,
                    v.penalty_imposed = CASE WHEN row.PENALITY_IMPOSED <> '' 
                                    THEN toFloat(row.PENALITY_IMPOSED) ELSE null END,
                    v.amount_paid = CASE WHEN row.AMOUNT_PAID <> '' 
                                THEN toFloat(row.AMOUNT_PAID) ELSE null END,
                    v.balance_due = CASE WHEN row.BALANCE_DUE <> '' 
                                THEN toFloat(row.BALANCE_DUE) ELSE null END,
                    v.hearing_status = row.HEARING_STATUS,
                    v.certification_status = row.CERTIFICATION_STATUS,
                    v.aggravated_level = row.AGGRAVATED_LEVEL
                
                // Create Respondent if available
                WITH v, p, row
                WHERE row.RESPONDENT_NAME <> '' AND row.RESPONDENT_NAME IS NOT NULL
                MERGE (r:Respondent {
                    name: row.RESPONDENT_NAME,
                    address: CASE WHEN row.RESPONDENT_HOUSE_NUMBER <> '' AND row.RESPONDENT_STREET <> ''
                            THEN toString(row.RESPONDENT_HOUSE_NUMBER) + ' ' + toString(row.RESPONDENT_STREET)
                            ELSE '' END,
                    city: CASE WHEN row.RESPONDENT_CITY <> '' THEN row.RESPONDENT_CITY ELSE '' END,
                    zip: CASE WHEN row.RESPONDENT_ZIP <> '' THEN toString(row.RESPONDENT_ZIP) ELSE '' END
                })
                MERGE (v)-[:ISSUED_TO]->(r)
                
                // Link violation to property
                WITH v, p
                MERGE (v)-[:VIOLATION_AT]->(p)
                """
                
                result = session.run(query, records=records)
                summary = result.consume()
                
                return True
                
        except Exception as e:
            logger.warning(f"Date validation error in ECB batch, skipping problematic records: {e}")
            # Try to process records individually to skip only problematic ones
            try:
                with self.driver.session() as session:
                    successful_records = 0
                    for record in records:
                        try:
                            result = session.run(query, records=[record])
                            summary = result.consume()
                            successful_records += 1
                        except Exception as record_error:
                            logger.debug(f"Skipping ECB record due to error: {record_error}")
                            continue
                    
                    logger.info(f"  ✅ Processed {successful_records}/{len(records)} ECB records from batch (skipped {len(records) - successful_records} problematic records)")
                    return True
            except Exception as fallback_error:
                logger.error(f"Error in ECB fallback processing: {fallback_error}")
                return False
    
    def _load_job_applications_batch(self, batch_df: pd.DataFrame) -> bool:
        """Load a batch of job applications into Neo4j."""
        try:
            # Clean data - replace NaN with empty strings
            batch_df = batch_df.fillna('')
            
            with self.driver.session() as session:
                # Convert DataFrame to list of dicts
                records = batch_df.to_dict('records')
                
                # Cypher query to create job applications and related entities
                query = """
                UNWIND $records AS row
                
                // Create or merge Property using BIN
                MERGE (p:Property {bin: toString(row.`Bin #`)})
                ON CREATE SET 
                    p.address = toString(row.`House #`) + ' ' + toString(row.`Street Name`),
                    p.borough = row.Borough,
                    p.block = toString(row.Block),
                    p.lot = toString(row.Lot)
                
                // Create or merge Owner if available
                WITH p, row
                WHERE row.`Owner's Business Name` <> '' OR (row.`Owner's First Name` <> '' AND row.`Owner's Last Name` <> '')
                MERGE (o:Owner {
                    name: CASE 
                        WHEN row.`Owner's Business Name` <> '' THEN row.`Owner's Business Name`
                        ELSE row.`Owner's First Name` + ' ' + row.`Owner's Last Name`
                    END,
                    address: CASE WHEN row.`Owner's House Number` <> '' AND row.`Owner'sHouse Street Name` <> ''
                            THEN toString(row.`Owner's House Number`) + ' ' + toString(row.`Owner'sHouse Street Name`)
                            ELSE '' END,
                    city: CASE WHEN row.`City ` <> '' THEN row.`City ` ELSE '' END,
                    state: CASE WHEN row.State <> '' THEN row.State ELSE '' END,
                    zip: CASE WHEN row.Zip <> '' THEN toString(row.Zip) ELSE '' END,
                    phone: CASE WHEN row.`Owner'sPhone #` <> '' THEN toString(row.`Owner'sPhone #`) ELSE '' END,
                    owner_type: row.`Owner Type`,
                    non_profit: row.`Non-Profit`
                })
                CREATE (o)-[:OWNS]->(p)
                
                // Create Job Application
                WITH p, row
                CREATE (j:Job_Application {
                    job_number: toString(row.`Job #`),
                    doc_number: toString(row.`Doc #`),
                    job_type: row.`Job Type`,
                    job_status: row.`Job Status`,
                    job_status_description: row.`Job Status Descrp`,
                    building_type: row.`Building Type`,
                    community_board: toString(row.`Community - Board`),
                    cluster: row.Cluster,
                    landmarked: row.Landmarked,
                    adult_establishment: row.`Adult Estab`,
                    loft_board: row.`Loft Board`,
                    city_owned: row.`City Owned`,
                    little_e: row.`Little e`,
                    pc_filed: row.`PC Filed`,
                    efiling_filed: row.`eFiling Filed`,
                    job_description: row.`Job Description`,
                    initial_cost: CASE WHEN row.`Initial Cost` <> '' 
                                 THEN toFloat(row.`Initial Cost`) ELSE null END,
                    total_estimated_fee: CASE WHEN row.`Total Est. Fee` <> '' 
                                        THEN toFloat(row.`Total Est. Fee`) ELSE null END,
                    fee_status: row.`Fee Status`,
                    existing_zoning_sqft: CASE WHEN row.`Existing Zoning Sqft` <> '' 
                                         THEN toFloat(row.`Existing Zoning Sqft`) ELSE null END,
                    proposed_zoning_sqft: CASE WHEN row.`Proposed Zoning Sqft` <> '' 
                                         THEN toFloat(row.`Proposed Zoning Sqft`) ELSE null END,
                    existing_dwelling_units: CASE WHEN row.`Existing Dwelling Units` <> '' 
                                           THEN toInteger(row.`Existing Dwelling Units`) ELSE null END,
                    proposed_dwelling_units: CASE WHEN row.`Proposed Dwelling Units` <> '' 
                                           THEN toInteger(row.`Proposed Dwelling Units`) ELSE null END,
                    existing_stories: CASE WHEN row.`ExistingNo. of Stories` <> '' 
                                     THEN toInteger(row.`ExistingNo. of Stories`) ELSE null END,
                    proposed_stories: CASE WHEN row.`Proposed No. of Stories` <> '' 
                                     THEN toInteger(row.`Proposed No. of Stories`) ELSE null END,
                    existing_height: CASE WHEN row.`Existing Height` <> '' 
                                    THEN toFloat(row.`Existing Height`) ELSE null END,
                    proposed_height: CASE WHEN row.`Proposed Height` <> '' 
                                    THEN toFloat(row.`Proposed Height`) ELSE null END,
                    zoning_district_1: row.`Zoning Dist1`,
                    zoning_district_2: row.`Zoning Dist2`,
                    zoning_district_3: row.`Zoning Dist3`,
                    special_district_1: row.`Special District 1`,
                    special_district_2: row.`Special District 2`,
                    building_class: row.BUILDING_CLASS,
                    latitude: CASE WHEN row.GIS_LATITUDE <> '' 
                             THEN toFloat(row.GIS_LATITUDE) ELSE null END,
                    longitude: CASE WHEN row.GIS_LONGITUDE <> '' 
                              THEN toFloat(row.GIS_LONGITUDE) ELSE null END,
                    council_district: CASE WHEN row.GIS_COUNCIL_DISTRICT <> '' 
                                     THEN toString(row.GIS_COUNCIL_DISTRICT) ELSE null END,
                    nta_name: row.GIS_NTA_NAME
                })
                
                // Link job application to property
                CREATE (j)-[:JOB_APPLICATION_AT]->(p)
                
                // Create or merge Applicant/Professional if available
                WITH j, p, row
                WHERE row.`Applicant's First Name` <> '' AND row.`Applicant's Last Name` <> ''
                MERGE (a:Professional {
                    name: row.`Applicant's First Name` + ' ' + row.`Applicant's Last Name`,
                    title: row.`Applicant Professional Title`,
                    license_number: toString(row.`Applicant License #`),
                    professional_cert: row.`Professional Cert`
                })
                CREATE (a)-[:FILED_APPLICATION]->(j)
                """
                
                result = session.run(query, records=records)
                summary = result.consume()
                
                return True
                
        except Exception as e:
            logger.error(f"Error loading job applications batch: {e}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        stats = {}
        try:
            with self.driver.session() as session:
                # Count nodes by label
                result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
                for record in result:
                    label = record["label"] or "Unknown"
                    stats[f"{label}_nodes"] = record["count"]
                
                # Count relationships
                result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
                for record in result:
                    stats[f"{record['rel_type']}_relationships"] = record["count"]
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
        
        return stats

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Load NYC DOB data into Neo4j")
    parser.add_argument(
        "dataset",
        choices=["dob_violations", "ecb_violations", "job_application_filings", "all"],
        help="Dataset to load"
    )
    parser.add_argument("--limit", type=int, help="Limit number of records to load")
    parser.add_argument("--uri", default="bolt://localhost:37687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = Neo4jDataLoader(args.uri, args.user, args.password)
    
    try:
        # Connect to Neo4j
        if not loader.connect():
            sys.exit(1)
        
        # Show initial stats
        initial_stats = loader.get_stats()
        logger.info(f"📊 Initial database stats: {initial_stats}")
        
        # Load data based on dataset selection
        if args.dataset == "all":
            logger.info("🚀 Loading all datasets...")
            datasets = ["dob_violations", "ecb_violations", "job_application_filings"]
            success = True
            for dataset in datasets:
                if not loader.load_dataset(dataset, args.limit):
                    success = False
                    break
        else:
            success = loader.load_dataset(args.dataset, args.limit)
        
        if success:
            logger.info("✅ Data loading completed successfully!")
        else:
            logger.error("❌ Data loading failed!")
            sys.exit(1)
    
    finally:
        loader.close()

if __name__ == "__main__":
    main() 