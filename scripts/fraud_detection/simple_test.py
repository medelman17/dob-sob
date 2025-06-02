#!/usr/bin/env python3
"""
Simple test script for episode design functionality
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_data_access():
    """Test basic data access"""
    print("🔍 Testing Data Access...")
    
    # Check housing litigation data
    housing_files = list(Path("../../data/raw/housing_litigations").glob("*.csv"))
    if housing_files:
        print(f"✅ Found {len(housing_files)} housing litigation files")
        file_path = housing_files[0]
        print(f"📁 Sample file: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("❌ No housing litigation files found")
        return False
    
    # Check complaints data
    complaint_files = list(Path("../../data/raw/complaints_received").glob("*.csv"))
    if complaint_files:
        print(f"✅ Found {len(complaint_files)} complaint files")
        file_path = complaint_files[0]
        print(f"📁 Sample file: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("❌ No complaint files found")
        return False
    
    return True

def test_polars_import():
    """Test polars import and basic functionality"""
    print("\n📊 Testing Polars...")
    
    try:
        import polars as pl
        print("✅ Polars imported successfully")
        
        # Test with housing litigation data
        housing_files = list(Path("../../data/raw/housing_litigations").glob("*.csv"))
        if housing_files:
            df = pl.read_csv(str(housing_files[0]), n_rows=10)
            print(f"✅ Read {len(df)} sample rows")
            print(f"📋 Columns: {df.columns}")
            
            # Test respondent analysis
            if "Respondent" in df.columns:
                respondents = df.select("Respondent").unique().drop_nulls()
                print(f"🏢 Found {len(respondents)} unique respondents in sample")
                
        return True
        
    except Exception as e:
        print(f"❌ Error with polars: {e}")
        return False

def test_pydantic_import():
    """Test pydantic import"""
    print("\n🔧 Testing Pydantic...")
    
    try:
        from pydantic import BaseModel, Field
        print("✅ Pydantic imported successfully")
        
        # Test simple model
        class TestModel(BaseModel):
            name: str = Field(..., description="Test name")
            value: int = Field(0, description="Test value")
        
        test_obj = TestModel(name="test", value=42)
        print(f"✅ Created test model: {test_obj.name} = {test_obj.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with pydantic: {e}")
        return False

def analyze_respondent_patterns():
    """Analyze respondent patterns for fraud detection insights"""
    print("\n🕵️ Analyzing Respondent Patterns...")
    
    try:
        import polars as pl
        
        housing_files = list(Path("../../data/raw/housing_litigations").glob("*.csv"))
        if not housing_files:
            print("❌ No housing litigation files found")
            return False
            
        df = pl.read_csv(str(housing_files[0]), n_rows=1000)
        
        # Analyze respondent complexity
        respondents = df.select("Respondent").drop_nulls()["Respondent"].to_list()
        
        # Find multi-entity respondents (likely corporate networks)
        multi_entity = [r for r in respondents if "," in str(r)]
        print(f"🏢 Multi-entity respondents: {len(multi_entity)}/{len(respondents)} ({len(multi_entity)/len(respondents)*100:.1f}%)")
        
        # Find LLC patterns
        llc_respondents = [r for r in respondents if "LLC" in str(r).upper()]
        print(f"🏢 LLC entities: {len(llc_respondents)}/{len(respondents)} ({len(llc_respondents)/len(respondents)*100:.1f}%)")
        
        # Find complex names (potential shell companies)
        complex_names = [r for r in respondents if len(str(r)) > 50]
        print(f"🔍 Complex names (>50 chars): {len(complex_names)}/{len(respondents)} ({len(complex_names)/len(respondents)*100:.1f}%)")
        
        if complex_names:
            print("📝 Sample complex respondent:")
            print(f"   {complex_names[0][:100]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing patterns: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Simple Episode Design Tests\n")
    
    tests = [
        test_data_access,
        test_polars_import,
        test_pydantic_import,
        analyze_respondent_patterns
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
        print("🎉 All tests passed! Ready for Graphiti integration.")
        return True
    else:
        print("⚠️  Some tests failed - check output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 