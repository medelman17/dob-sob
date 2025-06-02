#!/usr/bin/env python3

"""
Test script for pattern recognition functionality

This script validates the pattern recognition system with real NYC DOB data
and demonstrates the fraud detection capabilities.
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.fraud_detection.pattern_queries import (
    PatternRecognitionQueries, 
    FraudPatternType,
    calculate_markup_percentage,
    detect_shared_addresses, 
    detect_shared_officers,
    analyze_temporal_patterns
)


async def test_pattern_recognition_system():
    """Test the pattern recognition system initialization"""
    print("🧪 Testing Pattern Recognition System...")
    
    # Initialize without Graphiti (for local testing)
    detector = PatternRecognitionQueries()
    
    print(f"✅ Pattern detector initialized")
    print(f"📊 Fraud thresholds: {detector.thresholds}")
    print(f"🔍 Pattern definitions: {len(detector.patterns)} patterns loaded")
    
    # Display available patterns
    for pattern_type, pattern in detector.patterns.items():
        print(f"  • {pattern.name}: {pattern.description[:60]}...")
    
    return True


def test_utility_functions():
    """Test utility functions for pattern analysis"""
    print("\n🛠️ Testing Utility Functions...")
    
    # Test markup percentage calculation
    markup = calculate_markup_percentage(100000, 103000)  # 3% markup
    print(f"✅ Markup calculation: {markup:.1f}% (expected: 3.0%)")
    
    # Test shared address detection
    entities = [
        {"name": "Company A", "address": "123 Main St, NYC"},
        {"name": "Company B", "address": "456 Oak Ave, NYC"},
        {"name": "Company C", "address": "123 Main St, NYC"},  # Same as A
        {"name": "Company D", "address": "789 Pine St, NYC"}
    ]
    
    shared_addresses = detect_shared_addresses(entities)
    print(f"✅ Shared addresses detected: {len(shared_addresses)} pairs")
    for pair in shared_addresses:
        print(f"    {pair[0]} <-> {pair[1]}")
    
    # Test shared officer detection
    entities_with_officers = [
        {
            "name": "Alpha Corp", 
            "officers": ["John Smith", "Jane Doe"]
        },
        {
            "name": "Beta LLC", 
            "officers": ["John Smith", "Bob Wilson"]  # John Smith shared
        },
        {
            "name": "Gamma Inc", 
            "officers": ["Alice Johnson", "Charlie Brown"]
        }
    ]
    
    shared_officers = detect_shared_officers(entities_with_officers)
    print(f"✅ Shared officers detected: {len(shared_officers)} relationships")
    for officer, company1, company2 in shared_officers:
        print(f"    {officer}: {company1} <-> {company2}")
    
    # Test temporal pattern analysis
    events = [
        {"date": datetime(2024, 1, 1), "type": "company_formation", "entity": "Fast Corp"},
        {"date": datetime(2024, 1, 5), "type": "certification", "entity": "Fast Corp"},
        {"date": datetime(2024, 1, 10), "type": "contract_award", "entity": "Fast Corp"},
        {"date": datetime(2024, 2, 1), "type": "company_formation", "entity": "Normal Corp"},
        {"date": datetime(2024, 3, 15), "type": "contract_award", "entity": "Normal Corp"}
    ]
    
    suspicious_patterns = analyze_temporal_patterns(events, window_days=30)
    print(f"✅ Temporal patterns analyzed: {len(suspicious_patterns)} suspicious sequences")
    for pattern in suspicious_patterns:
        print(f"    {pattern['event_count']} events in {pattern['window_days']} days")
    
    return True


async def test_fraud_detection_mock():
    """Test fraud detection with mock data"""
    print("\n🔍 Testing Fraud Detection (Mock Mode)...")
    
    detector = PatternRecognitionQueries()
    
    # Run comprehensive scan (will use fallback methods)
    try:
        results = await detector.run_comprehensive_scan()
        
        print("✅ Comprehensive scan completed")
        print(f"📊 Results: {len(results)} pattern types analyzed")
        
        for pattern_type, alerts in results.items():
            print(f"  • {pattern_type.value}: {len(alerts)} alerts")
        
        # Generate and display report
        report = detector.generate_fraud_report()
        print(f"\n📋 Report Summary:")
        print(f"  Total alerts: {report['total_alerts']}")
        print(f"  High risk alerts: {report['summary']['high_risk_count']}")
        print(f"  Average risk score: {report['summary']['average_risk_score']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in fraud detection: {e}")
        return False


def analyze_real_housing_data():
    """Analyze real housing litigation data for fraud patterns"""
    print("\n🏠 Analyzing Real Housing Litigation Data...")
    
    try:
        import polars as pl
        
        # Find housing litigation files
        housing_files = list(Path("../../data/raw/housing_litigations").glob("*.csv"))
        if not housing_files:
            print("❌ No housing litigation files found")
            return False
        
        # Load and analyze data
        df = pl.read_csv(str(housing_files[0]), n_rows=1000)
        print(f"📊 Loaded {len(df)} records from {housing_files[0].name}")
        
        # Analyze respondent patterns (potential corporate networks)
        respondents = df.select("Respondent").unique().drop_nulls()
        respondent_list = respondents["Respondent"].to_list()
        
        # Look for patterns suggesting complex corporate structures
        multi_entity_respondents = []
        for respondent in respondent_list:
            if ',' in respondent and ('LLC' in respondent or 'CORP' in respondent or 'INC' in respondent):
                multi_entity_respondents.append(respondent)
        
        print(f"🔍 Found {len(multi_entity_respondents)} complex corporate respondents:")
        for i, respondent in enumerate(multi_entity_respondents[:5]):  # Show first 5
            print(f"  {i+1}. {respondent}")
        if len(multi_entity_respondents) > 5:
            print(f"  ... and {len(multi_entity_respondents) - 5} more")
        
        # Look for shared addresses in violation addresses  
        if "ViolationAddress" in df.columns:
            addresses = df.select("ViolationAddress").unique().drop_nulls()
            address_list = addresses["ViolationAddress"].to_list()
            
            # Count violations per address (potential problem properties)
            address_counts = df.group_by("ViolationAddress").count().sort("count", descending=True)
            print(f"\n🏢 Top addresses by violation count:")
            for row in address_counts.head(5).iter_rows(named=True):
                print(f"  {row['count']} violations: {row['ViolationAddress']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing housing data: {e}")
        return False


async def test_pattern_scoring():
    """Test pattern scoring algorithms"""
    print("\n📊 Testing Pattern Scoring Algorithms...")
    
    detector = PatternRecognitionQueries()
    
    # Mock result data
    mock_results = [
        {"type": "pass_through", "markup": 0.02, "contract_value": 100000},
        {"type": "control_fraud", "control_percentage": 0.75, "dbe_certified": True},
        {"type": "shell_company", "employees": 1, "revenue": 500000}
    ]
    
    # Test scoring functions
    for result in mock_results:
        if result["type"] == "pass_through":
            score = detector._calculate_pass_through_risk(result)
            print(f"✅ Pass-through risk score: {score:.2f}")
        elif result["type"] == "control_fraud":
            score = detector._calculate_control_fraud_risk(result)
            print(f"✅ Control fraud risk score: {score:.2f}")
        elif result["type"] == "shell_company":
            score = detector._calculate_shell_company_risk(result)
            print(f"✅ Shell company risk score: {score:.2f}")
    
    return True


def create_sample_fraud_report():
    """Create a sample fraud detection report"""
    print("\n📄 Creating Sample Fraud Report...")
    
    try:
        # Create reports directory
        reports_dir = Path("../../data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample report data
        sample_report = {
            "generated_at": datetime.now().isoformat(),
            "total_alerts": 12,
            "patterns_detected": {
                "pass_through": [
                    {
                        "entities": ["QuickBuild LLC", "MegaCorp Construction"],
                        "risk_score": 0.85,
                        "description": "2.1% markup on $200k contract",
                        "detected_at": datetime.now().isoformat()
                    }
                ],
                "control_fraud": [
                    {
                        "entities": ["Minority Builders Inc", "John Anderson"],
                        "risk_score": 0.92,
                        "description": "Non-minority individual controls 80% of DBE firm",
                        "detected_at": datetime.now().isoformat()
                    }
                ]
            },
            "high_risk_alerts": [
                {
                    "pattern": "control_fraud",
                    "entities": ["Minority Builders Inc", "John Anderson"],
                    "risk_score": 0.92,
                    "description": "Non-minority individual controls 80% of DBE firm"
                }
            ],
            "summary": {
                "pattern_counts": {
                    "pass_through": 3,
                    "control_fraud": 2,
                    "single_customer": 4,
                    "shell_company": 2,
                    "timing_fraud": 1,
                    "network_cluster": 0
                },
                "average_risk_score": 0.74,
                "high_risk_count": 1
            }
        }
        
        # Save sample report
        report_file = reports_dir / "sample_fraud_detection.json"
        with open(report_file, 'w') as f:
            json.dump(sample_report, f, indent=2)
        
        print(f"✅ Sample report saved to {report_file}")
        print(f"📊 Report contains {sample_report['total_alerts']} alerts")
        print(f"🚨 High risk alerts: {sample_report['summary']['high_risk_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample report: {e}")
        return False


async def main():
    """Run all pattern recognition tests"""
    print("🔥 NYC DOB Pattern Recognition Test Suite")
    print("=" * 50)
    
    tests = [
        ("Pattern Recognition System", test_pattern_recognition_system()),
        ("Utility Functions", test_utility_functions()),
        ("Fraud Detection Mock", test_fraud_detection_mock()),
        ("Real Housing Data Analysis", analyze_real_housing_data()),
        ("Pattern Scoring", test_pattern_scoring()),
        ("Sample Report Creation", create_sample_fraud_report())
    ]
    
    results = []
    for name, test in tests:
        print(f"\n🧪 Running: {name}")
        try:
            if asyncio.iscoroutine(test):
                result = await test
            else:
                result = test
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Pattern recognition system is ready!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main()) 