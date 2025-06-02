#!/usr/bin/env python3
"""
Test script for episode design functionality

This script validates the episode transformation pipeline with actual NYC DOB data.
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.fraud_detection.episode_design import EpisodeDesigner, NYCDOBEntityTypes


def test_housing_litigation_episodes():
    """Test housing litigation episode transformation"""
    print("🏠 Testing Housing Litigation Episode Design...")
    
    designer = EpisodeDesigner()
    
    # Find housing litigation files
    housing_files = list(Path("data/raw/housing_litigations").glob("*.csv"))
    if not housing_files:
        print("❌ No housing litigation files found")
        return False
    
    # Use the most recent file
    file_path = max(housing_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Processing: {file_path.name}")
    
    try:
        episodes = designer.prepare_housing_litigation_episodes(str(file_path))
        print(f"✅ Generated {len(episodes)} housing litigation episodes")
        
        # Show sample episode structure
        if episodes:
            sample_episode = episodes[0]
            print("\n📄 Sample Episode Structure:")
            print(json.dumps(sample_episode, indent=2, default=str)[:500] + "...")
            
        return True
        
    except Exception as e:
        print(f"❌ Error processing housing litigation data: {e}")
        return False


def test_complaint_episodes():
    """Test complaint episode transformation"""
    print("\n📢 Testing Complaint Episode Design...")
    
    designer = EpisodeDesigner()
    
    # Find complaint files
    complaint_files = list(Path("data/raw/complaints_received").glob("*.csv"))
    if not complaint_files:
        print("❌ No complaint files found")
        return False
    
    file_path = max(complaint_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Processing: {file_path.name}")
    
    try:
        # Process only first 100 rows for testing (large file)
        import polars as pl
        df = pl.read_csv(str(file_path), n_rows=100)
        temp_file = "temp_complaints_sample.csv"
        df.write_csv(temp_file)
        
        episodes = designer.prepare_complaint_episodes(temp_file)
        print(f"✅ Generated {len(episodes)} complaint episodes (sample)")
        
        # Show sample episode structure
        if episodes:
            sample_episode = episodes[0]
            print("\n📄 Sample Episode Structure:")
            print(json.dumps(sample_episode, indent=2, default=str)[:500] + "...")
        
        # Clean up temp file
        Path(temp_file).unlink()
        return True
        
    except Exception as e:
        print(f"❌ Error processing complaint data: {e}")
        return False


def test_entity_types():
    """Test custom entity type definitions"""
    print("\n🏢 Testing Custom Entity Types...")
    
    try:
        # Test Building entity
        building = NYCDOBEntityTypes.Building(
            bin="1234567",
            bbl="1001234567",
            address="123 Main St",
            borough="Manhattan",
            zip_code="10001"
        )
        print(f"✅ Building entity: {building.address} in {building.borough}")
        
        # Test Corporation entity
        corp = NYCDOBEntityTypes.Corporation(
            name="Test Construction LLC",
            entity_type="LLC",
            certification="MBE",
            business_address="456 Business Ave"
        )
        print(f"✅ Corporation entity: {corp.name} ({corp.certification})")
        
        # Test Person entity
        person = NYCDOBEntityTypes.Person(
            name="John Smith",
            role="General Contractor",
            license_type="GC License"
        )
        print(f"✅ Person entity: {person.name} - {person.role}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating entity types: {e}")
        return False


def analyze_data_relationships():
    """Analyze potential relationships in the data"""
    print("\n🔗 Analyzing Data Relationships...")
    
    try:
        # Check housing litigation data for entity patterns
        housing_files = list(Path("data/raw/housing_litigations").glob("*.csv"))
        if housing_files:
            import polars as pl
            df = pl.read_csv(str(housing_files[0]), n_rows=1000)
            
            # Analyze respondent patterns
            respondents = df.select("Respondent").unique().drop_nulls()
            respondent_list = respondents["Respondent"].to_list()
            print(f"📊 Found {len(respondent_list)} unique respondents in housing litigation")
            
            # Look for patterns that suggest corporate relationships
            complex_respondents = [r for r in respondent_list if "," in str(r) and len(str(r)) > 50]
            print(f"🏢 Found {len(complex_respondents)} complex respondent entries (likely multiple entities)")
            
            if complex_respondents:
                print("📝 Sample complex respondent:")
                print(f"   {complex_respondents[0][:100]}...")
            
            # Analyze building patterns
            bins = df.select("BIN").unique().drop_nulls()
            bin_list = bins["BIN"].to_list()
            print(f"🏢 Found {len(bin_list)} unique buildings in litigation data")
            
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing relationships: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Episode Design Tests\n")
    
    tests = [
        test_entity_types,
        test_housing_litigation_episodes,
        test_complaint_episodes,
        analyze_data_relationships
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All episode design tests passed!")
        return True
    else:
        print("⚠️  Some tests failed - check output above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 