#!/usr/bin/env python3
"""
High-performance bulk data loader for dob-sob fraud detection system.
Loads multiple datasets efficiently with optimized batch sizes and parallel processing.
"""

import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dob_sob.neo4j.load_data import Neo4jDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Priority datasets for fraud detection
PRIORITY_DATASETS = [
    ('dob_violations', 100000),      # Critical: Violation patterns
    ('job_applications', 50000),     # Critical: Professional activity
    ('active_licenses', 40000),      # Critical: Professional credentials
    ('building_permits', 30000),     # Important: Construction activity
    ('ecb_violations', 25000),       # Important: Environmental violations
    ('property_ownership', 20000),   # Important: Ownership patterns
    ('contractor_licenses', 15000),  # Important: Contractor credentials
    ('inspection_results', 15000),   # Important: Compliance patterns
]

def load_dataset_optimized(dataset_name, limit):
    """Load a single dataset with optimized settings."""
    try:
        start_time = time.time()
        logger.info(f"🚀 Starting {dataset_name} (limit: {limit:,})")
        
        # Create loader with optimized settings
        loader = Neo4jDataLoader(
            uri="bolt://localhost:37687",
            user="neo4j", 
            password="password"
        )
        
        # Load the dataset
        if hasattr(loader, f'load_{dataset_name}'):
            load_method = getattr(loader, f'load_{dataset_name}')
            load_method(limit=limit)
        else:
            logger.warning(f"⚠️  No load method found for {dataset_name}")
            return False
            
        elapsed = time.time() - start_time
        logger.info(f"✅ Completed {dataset_name} in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load {dataset_name}: {e}")
        return False
    finally:
        try:
            loader.close()
        except:
            pass

def main():
    """Main bulk loading function."""
    logger.info("🎯 Starting high-performance bulk data loading...")
    
    # Get initial stats
    try:
        loader = Neo4jDataLoader()
        loader.connect()
        initial_stats = loader.get_stats()
        logger.info(f"📊 Initial database stats: {initial_stats}")
        loader.close()
    except Exception as e:
        logger.error(f"Failed to get initial stats: {e}")
        return
    
    # Load datasets sequentially for now (parallel caused issues)
    successful_loads = 0
    total_start_time = time.time()
    
    for dataset_name, limit in PRIORITY_DATASETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 Loading {dataset_name} (limit: {limit:,})")
        logger.info(f"{'='*60}")
        
        success = load_dataset_optimized(dataset_name, limit)
        if success:
            successful_loads += 1
            
        # Brief pause between datasets
        time.sleep(2)
    
    # Final stats
    try:
        loader = Neo4jDataLoader()
        loader.connect()
        final_stats = loader.get_stats()
        logger.info(f"\n📊 Final database stats: {final_stats}")
        loader.close()
    except Exception as e:
        logger.error(f"Failed to get final stats: {e}")
    
    total_elapsed = time.time() - total_start_time
    logger.info(f"\n🎉 Bulk loading complete!")
    logger.info(f"✅ Successfully loaded: {successful_loads}/{len(PRIORITY_DATASETS)} datasets")
    logger.info(f"⏱️  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")

if __name__ == "__main__":
    main() 