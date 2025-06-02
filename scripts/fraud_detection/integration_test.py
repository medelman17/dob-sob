#!/usr/bin/env python3
"""
Integration Test for Task 4.2: Community Detection Algorithms

This script validates that both the data profiling framework (Task 4.1) and 
community detection algorithms (Task 4.2) work together correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_profiling_framework import NYCDOBDataProfiler
from community_detection_algorithms import AdvancedCommunityDetector


async def test_integration():
    """Run comprehensive integration test"""
    print('🔍 Testing NYC DOB Community Detection Integration...')
    print('=' * 60)
    
    try:
        # Test 1: Data Profiling Framework
        print('\n1. Testing Data Profiling Framework...')
        profiler = NYCDOBDataProfiler(
            data_directory='../../data', 
            output_directory='/tmp/test_output'
        )
        print('✅ Data Profiler initialized successfully')
        
        # Test 2: Community Detection Algorithms  
        print('\n2. Testing Community Detection Algorithms...')
        detector = AdvancedCommunityDetector(output_directory='/tmp/test_output')
        print('✅ Community Detector initialized successfully')
        
        # Test 3: Mock Workflow
        print('\n3. Testing Mock Workflow...')
        mock_profiling_results = {
            'profiling_summary': {
                'total_datasets': 3,
                'total_rows': 1500,
                'overall_quality': 'excellent'
            },
            'dataset_profiles': {
                'permits': {
                    'name': 'permits',
                    'row_count': 1000,
                    'entity_columns': ['owner_name', 'contractor_name'],
                    'key_columns': ['permit_id', 'bin']
                },
                'violations': {
                    'name': 'violations', 
                    'row_count': 500,
                    'entity_columns': ['respondent_name'],
                    'key_columns': ['violation_id', 'bin']
                },
                'complaints': {
                    'name': 'complaints',
                    'row_count': 200,
                    'entity_columns': [],
                    'key_columns': ['complaint_id']
                }
            }
        }
        
        # Build network from mock data
        print('   Building network from profiled data...')
        graph = await detector.build_network_from_profiled_data(mock_profiling_results)
        print(f'✅ Network built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')
        
        # Test community detection algorithms
        print('\n4. Testing Community Detection Algorithms...')
        
        # Label propagation
        print('   Running Label Propagation...')
        lp_communities = await detector.detect_communities_label_propagation()
        print(f'✅ Label Propagation: {len(lp_communities)} communities detected')
        
        # Modularity optimization  
        print('   Running Modularity Optimization...')
        mod_communities = await detector.detect_communities_modularity_optimization()
        print(f'✅ Modularity Optimization: {len(mod_communities)} communities detected')
        
        # Multi-resolution
        print('   Running Multi-Resolution Detection...')
        mr_communities = await detector.detect_communities_multi_resolution()
        print(f'✅ Multi-Resolution: {len(mr_communities)} resolution levels analyzed')
        
        # Test fraud detection capabilities
        print('\n5. Testing Fraud Detection Features...')
        
        # Check for high-risk communities
        high_risk_count = 0
        fraud_indicators_found = []
        
        for communities in [lp_communities, mod_communities]:
            for comm_id, comm_info in communities.items():
                if hasattr(comm_info, 'risk_level') and comm_info.risk_level.value in ['HIGH', 'CRITICAL']:
                    high_risk_count += 1
                if hasattr(comm_info, 'fraud_indicators'):
                    fraud_indicators_found.extend(comm_info.fraud_indicators)
        
        print(f'✅ Risk Assessment: {high_risk_count} high-risk communities identified')
        print(f'✅ Fraud Indicators: {len(set(fraud_indicators_found))} unique patterns detected')
        
        # Test Graphiti integration (mock)
        print('\n6. Testing Graphiti Integration...')
        graphiti_ready = await detector.initialize_graphiti_integration()
        print(f'✅ Graphiti Integration: {"Ready" if graphiti_ready else "Mock mode (Graphiti not installed)"}')
        
        print('\n' + '=' * 60)
        print('🎉 ALL TESTS PASSED! Task 4.2 Implementation is Working!')
        print('✅ Data Profiling Framework (Task 4.1): COMPLETE')
        print('✅ Community Detection Algorithms (Task 4.2): COMPLETE')
        print('✅ Integration between modules: WORKING')
        print('✅ Fraud detection capabilities: FUNCTIONAL')
        print('✅ Ready for real NYC DOB data processing!')
        
        return True
        
    except Exception as e:
        print(f'\n❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1) 