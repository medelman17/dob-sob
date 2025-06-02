"""
Comprehensive tests for Advanced Community Detection Algorithms

Tests the community detection functionality for fraud detection in NYC DOB data,
validating all algorithms and their integration with the data profiling framework.
"""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import networkx as nx
import numpy as np

from community_detection_algorithms import (
    AdvancedCommunityDetector,
    CommunityDetectionAlgorithm,
    CommunityRiskLevel,
    CommunityInfo,
    CommunityEvolution,
    run_community_detection_analysis
)


class TestAdvancedCommunityDetector:
    """Test the core community detection functionality"""
    
    @pytest.fixture
    def mock_profiling_results(self):
        """Create mock profiling results for testing"""
        return {
            'profiling_summary': {
                'total_datasets': 3,
                'total_rows': 1700,
                'overall_quality': 'excellent'
            },
            'dataset_profiles': {
                'permits': {
                    'name': 'permits',
                    'row_count': 1000,
                    'column_count': 8,
                    'entity_columns': ['owner_name', 'contractor_name', 'address'],
                    'key_columns': ['permit_id', 'bin'],
                    'temporal_columns': ['filing_date']
                },
                'violations': {
                    'name': 'violations',
                    'row_count': 500,
                    'column_count': 6,
                    'entity_columns': ['respondent_name', 'respondent_address'],
                    'key_columns': ['violation_id', 'bin'],
                    'temporal_columns': ['violation_date']
                },
                'complaints': {
                    'name': 'complaints',
                    'row_count': 200,
                    'column_count': 5,
                    'entity_columns': [],
                    'key_columns': ['complaint_id', 'bin'],
                    'temporal_columns': ['date_received']
                }
            }
        }
    
    @pytest.fixture
    def sample_output_dir(self):
        """Create temporary output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def detector(self, sample_output_dir):
        """Create detector instance"""
        return AdvancedCommunityDetector(output_directory=sample_output_dir)
    
    def test_detector_initialization(self, detector, sample_output_dir):
        """Test detector initializes correctly"""
        assert detector.output_dir == Path(sample_output_dir)
        assert detector.output_dir.exists()
        assert detector.network_graph is None
        assert len(detector.detected_communities) == 0
        assert len(detector.community_evolution) == 0
        
        # Check algorithm configurations
        assert CommunityDetectionAlgorithm.LABEL_PROPAGATION in detector.algorithm_configs
        assert CommunityDetectionAlgorithm.MODULARITY_OPTIMIZATION in detector.algorithm_configs
        assert CommunityDetectionAlgorithm.DYNAMIC_EVOLUTION in detector.algorithm_configs
        assert CommunityDetectionAlgorithm.MULTI_RESOLUTION in detector.algorithm_configs
    
    @pytest.mark.asyncio
    async def test_graphiti_integration_initialization(self, detector):
        """Test Graphiti integration initialization"""
        # Test without Graphiti available
        with patch('community_detection_algorithms.GRAPHITI_AVAILABLE', False):
            result = await detector.initialize_graphiti_integration()
            assert result is False
        
        # Test with Graphiti available but no client
        with patch('community_detection_algorithms.GRAPHITI_AVAILABLE', True):
            result = await detector.initialize_graphiti_integration()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_network_building_from_profiled_data(self, detector, mock_profiling_results):
        """Test building network graph from profiled data"""
        graph = await detector.build_network_from_profiled_data(mock_profiling_results)
        
        # Validate graph structure
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        assert detector.network_graph is graph
        
        # Check node types
        node_types = set()
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type')
            if node_type:
                node_types.add(node_type)
        
        expected_types = {'permit', 'applicant', 'contractor', 'property', 'violation', 'respondent', 'complaint'}
        assert node_types.issubset(expected_types)
        
        # Check edge relationships
        relationship_types = set()
        for _, _, data in graph.edges(data=True):
            relationship = data.get('relationship')
            if relationship:
                relationship_types.add(relationship)
        
        assert len(relationship_types) > 0
        assert 'applied_by' in relationship_types or 'contractor' in relationship_types
    
    @pytest.mark.asyncio
    async def test_permit_relationship_creation(self, detector):
        """Test permit relationship creation"""
        graph = nx.Graph()
        profile = {'row_count': 10}
        
        await detector._create_permit_relationships(graph, profile)
        
        # Check nodes were created
        permit_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'permit']
        applicant_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'applicant']
        contractor_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'contractor']
        property_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'property']
        
        assert len(permit_nodes) == 10
        assert len(applicant_nodes) > 0
        assert len(contractor_nodes) > 0
        assert len(property_nodes) > 0
        
        # Check relationships were created
        assert graph.number_of_edges() > 0
        
        # Validate specific relationship types
        relationships = set()
        for _, _, data in graph.edges(data=True):
            relationships.add(data.get('relationship'))
        
        expected_relationships = {'applied_by', 'contractor', 'for_property', 'owns'}
        assert relationships.intersection(expected_relationships)
    
    @pytest.mark.asyncio
    async def test_temporal_attributes_addition(self, detector):
        """Test adding temporal attributes to graph"""
        graph = nx.Graph()
        graph.add_edge('node1', 'node2')
        graph.add_edge('node2', 'node3')
        
        await detector._add_temporal_attributes(graph, {})
        
        # Check temporal attributes were added
        for _, _, data in graph.edges(data=True):
            assert 'timestamp' in data
            assert 'temporal_weight' in data
            assert isinstance(data['timestamp'], datetime)
            assert 0 <= data['temporal_weight'] <= 1
    
    @pytest.mark.asyncio
    async def test_network_metrics_calculation(self, detector):
        """Test network metrics calculation"""
        graph = nx.Graph()
        graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')])
        
        await detector._calculate_network_metrics(graph)
        
        # Check basic metrics
        assert graph.graph['num_nodes'] == 4
        assert graph.graph['num_edges'] == 4
        assert 'density' in graph.graph
        assert 'largest_component_size' in graph.graph
        assert 'num_components' in graph.graph
    
    @pytest.mark.asyncio
    async def test_label_propagation_community_detection(self, detector, mock_profiling_results):
        """Test label propagation algorithm"""
        # Build network first
        await detector.build_network_from_profiled_data(mock_profiling_results)
        
        # Run label propagation
        communities = await detector.detect_communities_label_propagation(temporal_constraints=True)
        
        # Validate results
        assert isinstance(communities, dict)
        assert len(communities) > 0
        
        # Check community structure
        for comm_id, comm_info in communities.items():
            assert isinstance(comm_info, CommunityInfo)
            assert comm_info.algorithm == CommunityDetectionAlgorithm.LABEL_PROPAGATION
            assert len(comm_info.nodes) > 0
            assert comm_info.size == len(comm_info.nodes)
            assert isinstance(comm_info.risk_level, CommunityRiskLevel)
            assert 0 <= comm_info.modularity_score <= 1
    
    @pytest.mark.asyncio
    async def test_modularity_optimization_community_detection(self, detector, mock_profiling_results):
        """Test modularity optimization algorithm"""
        # Build network first
        await detector.build_network_from_profiled_data(mock_profiling_results)
        
        # Run modularity optimization
        communities = await detector.detect_communities_modularity_optimization()
        
        # Validate results
        assert isinstance(communities, dict)
        assert len(communities) > 0
        
        # Check community structure
        for comm_id, comm_info in communities.items():
            assert isinstance(comm_info, CommunityInfo)
            assert comm_info.algorithm == CommunityDetectionAlgorithm.MODULARITY_OPTIMIZATION
            assert len(comm_info.nodes) > 0
            assert comm_info.size == len(comm_info.nodes)
    
    @pytest.mark.asyncio
    async def test_multi_resolution_community_detection(self, detector, mock_profiling_results):
        """Test multi-resolution community detection"""
        # Build network first
        await detector.build_network_from_profiled_data(mock_profiling_results)
        
        # Run multi-resolution detection
        multi_resolution_communities = await detector.detect_communities_multi_resolution()
        
        # Validate results
        assert isinstance(multi_resolution_communities, dict)
        assert len(multi_resolution_communities) > 0
        
        # Check that we have multiple resolutions
        resolution_keys = list(multi_resolution_communities.keys())
        assert len(resolution_keys) >= 5  # Should have multiple resolution levels
        
        # Check each resolution has communities
        for resolution_key, communities in multi_resolution_communities.items():
            assert isinstance(communities, list)
            if communities:  # Some resolutions might have no communities
                for comm_info in communities:
                    assert isinstance(comm_info, CommunityInfo)
                    assert comm_info.algorithm == CommunityDetectionAlgorithm.MULTI_RESOLUTION
    
    @pytest.mark.asyncio
    async def test_dynamic_evolution_tracking(self, detector, mock_profiling_results):
        """Test dynamic community evolution tracking"""
        # Build network first
        await detector.build_network_from_profiled_data(mock_profiling_results)
        
        # Generate time windows
        time_windows = detector._generate_time_windows()
        assert len(time_windows) > 0
        
        # Run dynamic evolution tracking
        evolution_history = await detector.detect_communities_dynamic_evolution(time_windows)
        
        # Validate results
        assert isinstance(evolution_history, list)
        # Evolution history might be empty if no temporal patterns found
        
        if evolution_history:
            for evolution in evolution_history:
                assert isinstance(evolution, CommunityEvolution)
                assert evolution.community_id
                assert isinstance(evolution.timestamp, datetime)
                assert evolution.change_type in ['created', 'merged', 'split', 'dissolved', 'stable']
    
    def test_temporal_subgraph_creation(self, detector):
        """Test creating temporal subgraphs"""
        # Create graph with temporal edges
        graph = nx.Graph()
        base_date = datetime(2023, 1, 1)
        
        graph.add_edge('a', 'b', timestamp=base_date)
        graph.add_edge('b', 'c', timestamp=base_date + timedelta(days=10))
        graph.add_edge('c', 'd', timestamp=base_date + timedelta(days=20))
        graph.add_edge('d', 'e', timestamp=base_date + timedelta(days=40))
        
        detector.network_graph = graph
        
        # Create subgraph for specific time window
        start_time = base_date + timedelta(days=5)
        end_time = base_date + timedelta(days=25)
        
        temporal_graph = detector._create_temporal_subgraph(start_time, end_time)
        
        # Should include edges within time window
        assert temporal_graph.number_of_edges() == 2  # b-c and c-d edges
        assert temporal_graph.has_edge('b', 'c')
        assert temporal_graph.has_edge('c', 'd')
        assert not temporal_graph.has_edge('a', 'b')  # Too early
        assert not temporal_graph.has_edge('d', 'e')  # Too late
    
    @pytest.mark.asyncio
    async def test_fraud_indicator_detection(self, detector):
        """Test fraud indicator detection"""
        # Create graph with fraud patterns
        graph = nx.Graph()
        
        # Pattern 1: Single contractor with multiple permits
        contractor_node = 'contractor_1'
        graph.add_node(contractor_node, node_type='contractor')
        
        for i in range(6):  # 6 permits for same contractor
            permit_node = f'permit_{i}'
            graph.add_node(permit_node, node_type='permit')
            graph.add_edge(contractor_node, permit_node)
        
        # Pattern 2: Permits and violations
        violation_node = 'violation_1'
        graph.add_node(violation_node, node_type='violation')
        
        nodes = [contractor_node] + [f'permit_{i}' for i in range(6)] + [violation_node]
        
        fraud_indicators = await detector._detect_fraud_indicators(nodes, graph)
        
        # Should detect fraud patterns
        assert 'single_contractor_multiple_permits' in fraud_indicators
    
    def test_risk_level_calculation(self, detector):
        """Test risk level calculation"""
        # Test different scenarios
        
        # Low risk: no indicators, small community
        risk_level = detector._calculate_risk_level([], 3)
        assert risk_level == CommunityRiskLevel.LOW
        
        # Medium risk: some indicators
        risk_level = detector._calculate_risk_level(['pattern1'], 5)
        assert risk_level == CommunityRiskLevel.MEDIUM
        
        # High risk: multiple indicators
        risk_level = detector._calculate_risk_level(['pattern1', 'pattern2', 'pattern3'], 8)
        assert risk_level == CommunityRiskLevel.HIGH
        
        # Critical risk: many indicators + large community
        risk_level = detector._calculate_risk_level(['p1', 'p2', 'p3', 'p4'], 25)
        assert risk_level == CommunityRiskLevel.CRITICAL
    
    def test_modularity_calculation(self, detector):
        """Test modularity calculation for communities"""
        # Create simple graph
        graph = nx.Graph()
        graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')])
        
        # Test modularity for different community sizes
        modularity_1 = detector._calculate_community_modularity(['a'], graph)
        modularity_2 = detector._calculate_community_modularity(['a', 'b'], graph)
        modularity_4 = detector._calculate_community_modularity(['a', 'b', 'c', 'd'], graph)
        
        assert modularity_1 == 0.0  # Single node
        assert isinstance(modularity_2, float)
        assert isinstance(modularity_4, float)
    
    def test_temporal_span_calculation(self, detector):
        """Test temporal span calculation"""
        # Create graph with temporal edges
        graph = nx.Graph()
        base_date = datetime(2023, 1, 1)
        
        edges = [('a', 'b'), ('b', 'c')]
        graph.add_edge('a', 'b', timestamp=base_date)
        graph.add_edge('b', 'c', timestamp=base_date + timedelta(days=30))
        
        temporal_span = detector._get_community_temporal_span(edges, graph)
        
        assert temporal_span is not None
        start_time, end_time = temporal_span
        assert start_time == base_date
        assert end_time == base_date + timedelta(days=30)
        assert (end_time - start_time).days == 30
    
    @pytest.mark.asyncio
    async def test_community_info_creation(self, detector, mock_profiling_results):
        """Test CommunityInfo object creation"""
        # Build network first
        await detector.build_network_from_profiled_data(mock_profiling_results)
        
        # Get some nodes for testing
        nodes = list(detector.network_graph.nodes())[:5]
        
        community_info = await detector._create_community_info(
            'test_community',
            CommunityDetectionAlgorithm.LABEL_PROPAGATION,
            nodes
        )
        
        # Validate CommunityInfo structure
        assert community_info.id == 'test_community'
        assert community_info.algorithm == CommunityDetectionAlgorithm.LABEL_PROPAGATION
        assert community_info.nodes == nodes
        assert community_info.size == len(nodes)
        assert isinstance(community_info.risk_level, CommunityRiskLevel)
        assert isinstance(community_info.fraud_indicators, list)
        assert 0 <= community_info.density <= 1
        assert 0 <= community_info.clustering_coefficient <= 1
        assert isinstance(community_info.centrality_measures, dict)


class TestComprehensiveWorkflow:
    """Test the complete community detection workflow"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_community_detection_workflow(self, sample_output_dir):
        """Test the complete workflow from profiling results to final analysis"""
        
        # Create mock profiling results
        mock_profiling_results = {
            'profiling_summary': {
                'total_datasets': 3,
                'total_rows': 1700,
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
                    'key_columns': ['complaint_id', 'bin']
                }
            }
        }
        
        detector = AdvancedCommunityDetector(output_directory=sample_output_dir)
        
        # Run comprehensive analysis
        results = await detector.run_comprehensive_community_detection(mock_profiling_results)
        
        # Validate comprehensive results structure
        assert 'label_propagation' in results
        assert 'modularity_optimization' in results
        assert 'multi_resolution' in results
        assert 'analysis_report' in results
        
        # Validate analysis report
        report = results['analysis_report']
        assert 'timestamp' in report
        assert 'summary' in report
        assert 'risk_assessment' in report
        assert 'recommendations' in report
        
        # Validate summary
        summary = report['summary']
        assert 'total_communities_detected' in summary
        assert 'high_risk_communities' in summary
        assert 'algorithms_used' in summary
        assert 'network_nodes' in summary
        assert 'network_edges' in summary
        
        # Validate risk assessment
        risk_assessment = report['risk_assessment']
        assert 'overall_risk_level' in risk_assessment
        assert 'fraud_indicators_found' in risk_assessment
        assert 'requires_investigation' in risk_assessment
        
        # Check that results were saved
        output_path = Path(sample_output_dir)
        result_files = list(output_path.glob("community_detection_results_*.json"))
        assert len(result_files) > 0
        
        # Validate saved results
        with open(result_files[0], 'r') as f:
            saved_results = json.load(f)
            assert 'analysis_report' in saved_results
    
    @pytest.mark.asyncio
    async def test_convenience_function(self, sample_output_dir):
        """Test the convenience function for running analysis"""
        
        mock_profiling_results = {
            'profiling_summary': {'total_datasets': 2},
            'dataset_profiles': {
                'permits': {'row_count': 100, 'entity_columns': [], 'key_columns': []},
                'violations': {'row_count': 50, 'entity_columns': [], 'key_columns': []}
            }
        }
        
        results = await run_community_detection_analysis(
            mock_profiling_results,
            output_directory=sample_output_dir
        )
        
        # Validate results
        assert isinstance(results, dict)
        assert 'analysis_report' in results
        assert results['analysis_report']['summary']['total_communities_detected'] >= 0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_empty_profiling_results(self, sample_output_dir):
        """Test handling of empty profiling results"""
        detector = AdvancedCommunityDetector(output_directory=sample_output_dir)
        
        empty_results = {
            'profiling_summary': {'total_datasets': 0},
            'dataset_profiles': {}
        }
        
        graph = await detector.build_network_from_profiled_data(empty_results)
        
        # Should handle empty data gracefully
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
    
    @pytest.mark.asyncio
    async def test_community_detection_without_network(self, sample_output_dir):
        """Test community detection when network is not built"""
        detector = AdvancedCommunityDetector(output_directory=sample_output_dir)
        
        # Should raise error when trying to detect communities without network
        with pytest.raises(ValueError, match="Network graph not initialized"):
            await detector.detect_communities_label_propagation()
        
        with pytest.raises(ValueError, match="Network graph not initialized"):
            await detector.detect_communities_modularity_optimization()
    
    def test_time_window_generation(self, sample_output_dir):
        """Test time window generation for dynamic analysis"""
        detector = AdvancedCommunityDetector(output_directory=sample_output_dir)
        
        time_windows = detector._generate_time_windows()
        
        # Should generate 12 monthly windows
        assert len(time_windows) == 12
        
        # Windows should be in chronological order
        for i in range(len(time_windows) - 1):
            assert time_windows[i][1] <= time_windows[i + 1][0]  # No overlap
        
        # Each window should be approximately 30 days
        for start, end in time_windows:
            duration = (end - start).days
            assert 28 <= duration <= 32  # Account for month variations


if __name__ == "__main__":
    # Run tests manually
    async def run_tests():
        print("🧪 Running Community Detection Algorithm Tests")
        
        # Create temporary test environment
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Test detector initialization
            detector = AdvancedCommunityDetector(output_directory=temp_dir)
            print("✅ Detector initialization successful")
            
            # Test network building
            mock_profiling_results = {
                'profiling_summary': {'total_datasets': 3, 'overall_quality': 'excellent'},
                'dataset_profiles': {
                    'permits': {
                        'row_count': 100,
                        'entity_columns': ['owner_name', 'contractor_name'],
                        'key_columns': ['permit_id', 'bin']
                    },
                    'violations': {
                        'row_count': 50,
                        'entity_columns': ['respondent_name'],
                        'key_columns': ['violation_id', 'bin']
                    }
                }
            }
            
            graph = await detector.build_network_from_profiled_data(mock_profiling_results)
            print(f"✅ Network building successful: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
            # Test label propagation
            lp_communities = await detector.detect_communities_label_propagation()
            print(f"✅ Label propagation: {len(lp_communities)} communities detected")
            
            # Test modularity optimization
            mod_communities = await detector.detect_communities_modularity_optimization()
            print(f"✅ Modularity optimization: {len(mod_communities)} communities detected")
            
            # Test comprehensive workflow
            results = await detector.run_comprehensive_community_detection(mock_profiling_results)
            report = results['analysis_report']
            print(f"✅ Comprehensive analysis completed")
            print(f"   - Total communities: {report['summary']['total_communities_detected']}")
            print(f"   - High-risk communities: {report['summary']['high_risk_communities']}")
            print(f"   - Overall risk: {report['risk_assessment']['overall_risk_level']}")
            
            # Test time window generation
            time_windows = detector._generate_time_windows()
            print(f"✅ Time window generation: {len(time_windows)} windows created")
            
            print("\n🎉 All community detection algorithm tests passed!")
            print("🔍 Advanced fraud detection capabilities validated")
    
    asyncio.run(run_tests()) 