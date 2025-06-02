#!/usr/bin/env python3
"""
Test Graphiti integration with uv-managed dependencies
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_graphiti_imports():
    """Test that Graphiti imports work properly"""
    print("🧪 Testing Graphiti Integration...")
    
    try:
        from graphiti_core.nodes import EpisodeType
        from graphiti_core import Graphiti
        print("✅ Successfully imported Graphiti core components")
        
        # Test EpisodeType
        json_type = EpisodeType.json
        text_type = EpisodeType.text
        print(f"✅ EpisodeType imports: json={json_type}, text={text_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing Graphiti: {e}")
        return False

def test_episode_design_with_real_graphiti():
    """Test our episode design module with real Graphiti"""
    print("\n🏗️ Testing Episode Design with Real Graphiti...")
    
    try:
        from scripts.fraud_detection.episode_design import EpisodeDesigner, NYCDOBEntityTypes
        print("✅ Successfully imported episode design module")
        
        # Test designer initialization
        designer = EpisodeDesigner()
        print("✅ EpisodeDesigner initialized successfully")
        
        # Test entity types
        building = NYCDOBEntityTypes.Building(
            bin="1234567",
            address="123 Test St",
            borough="Manhattan"
        )
        print(f"✅ Created building entity: {building.address}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing episode design: {e}")
        return False

def main():
    """Run all Graphiti integration tests"""
    print("🚀 Starting Graphiti Integration Tests\n")
    
    tests = [
        test_graphiti_imports,
        test_episode_design_with_real_graphiti
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
        print("🎉 All Graphiti integration tests passed! Ready for implementation.")
        return True
    else:
        print("⚠️  Some tests failed - check output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 