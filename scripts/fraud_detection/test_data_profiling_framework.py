"""
Comprehensive tests for NYC DOB Data Profiling Framework

Tests the solid foundations we've built for community detection and fraud analysis.
Validates data profiling, quality assessment, and performance capabilities.
"""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import polars as pl

from data_profiling_framework import (
    NYCDOBDataProfiler,
    DataCategory,
    DataQualityLevel,
    DatasetProfile,
    EntityProfile,
    profile_nyc_dob_data,
    get_graph_construction_recommendations
)


class TestNYCDOBDataProfiler:
    """Test the core data profiling functionality"""
    
    @pytest.fixture
    def sample_data_dir(self):
        """Create temporary directory with sample data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            
            # Create sample CSV files
            self._create_sample_permit_data(data_dir / "permit_data.csv")
            self._create_sample_violation_data(data_dir / "violation_data.csv")
            self._create_sample_complaint_data(data_dir / "complaint_data.csv")
            
            # Create subdirectories
            raw_dir = data_dir / "raw" / "permits"
            raw_dir.mkdir(parents=True)
            self._create_sample_permit_data(raw_dir / "historical_permits.csv")
            
            yield data_dir
    
    def _create_sample_permit_data(self, filepath: Path):
        """Create sample permit dataset"""
        data = {
            'permit_id': [f'P{i:06d}' for i in range(1000)],
            'bin': [f'{1000000 + i}' for i in range(1000)],
            'owner_name': [f'Company {i % 100}' if i % 10 != 0 else None for i in range(1000)],
            'contractor_name': [f'Contractor {i % 50}' for i in range(1000)],
            'work_type': [['ALT1', 'NB', 'DM', 'ALT2'][i % 4] for i in range(1000)],
            'estimated_cost': [10000 + (i * 1000) for i in range(1000)],
            'filing_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'address': [f'{i + 1} Main St, New York, NY' for i in range(1000)]
        }
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def _create_sample_violation_data(self, filepath: Path):
        """Create sample violation dataset"""
        data = {
            'violation_id': [f'V{i:08d}' for i in range(500)],
            'bin': [f'{1000000 + (i % 800)}' for i in range(500)],  # Some overlap with permits
            'respondent_name': [f'Company {i % 80}' for i in range(500)],
            'violation_type': [['SAFETY', 'ZONING', 'CONSTRUCTION'][i % 3] for i in range(500)],
            'penalty_amount': [500 + (i * 100) for i in range(500)],
            'violation_date': pd.date_range('2020-01-01', periods=500, freq='2D'),
            'respondent_address': [f'{i + 1} Main St, New York, NY' if i % 5 != 0 else None for i in range(500)]
        }
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def _create_sample_complaint_data(self, filepath: Path):
        """Create sample complaint dataset"""
        data = {
            'complaint_id': [f'C{i:07d}' for i in range(200)],
            'bin': [f'{1000000 + (i % 900)}' for i in range(200)],  # Overlap with permits and violations
            'complaint_category': [['SAFETY', 'NOISE', 'ILLEGAL_WORK'][i % 3] for i in range(200)],
            'status': [['OPEN', 'CLOSED', 'PENDING'][i % 3] for i in range(200)],
            'date_received': pd.date_range('2020-01-01', periods=200, freq='5D')
        }
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    @pytest.fixture
    def profiler(self, sample_data_dir):
        """Create profiler instance with sample data"""
        return NYCDOBDataProfiler(
            data_directory=str(sample_data_dir),
            output_directory=str(sample_data_dir / "metadata"),
            max_memory_gb=2.0,
            use_parallel=False  # Disable for testing
        )
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initializes correctly"""
        assert profiler.data_dir.exists()
        assert profiler.output_dir.exists()
        assert profiler.max_memory_bytes == 2.0 * 1024**3
        assert not profiler.use_parallel
        assert len(profiler.category_mappings) > 0
        assert len(profiler.entity_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_dataset_discovery(self, profiler):
        """Test automatic dataset discovery"""
        datasets = await profiler._discover_datasets()
        
        assert len(datasets) >= 3  # permit, violation, complaint data
        
        # Check dataset structure
        for dataset in datasets:
            assert 'name' in dataset
            assert 'path' in dataset
            assert 'size_mb' in dataset
            assert 'category' in dataset
            assert dataset['size_mb'] > 0
    
    def test_dataset_categorization(self, profiler):
        """Test dataset categorization logic"""
        assert profiler._categorize_dataset('permit_data.csv') == DataCategory.PERMITS
        assert profiler._categorize_dataset('violation_report.csv') == DataCategory.VIOLATIONS
        assert profiler._categorize_dataset('complaint_log.csv') == DataCategory.COMPLAINTS
        assert profiler._categorize_dataset('unknown_file.csv') == DataCategory.PROPERTIES
    
    @pytest.mark.asyncio
    async def test_single_dataset_profiling(self, profiler, sample_data_dir):
        """Test profiling of a single dataset"""
        dataset = {
            'name': 'test_permit_data',
            'path': str(sample_data_dir / 'permit_data.csv'),
            'size_mb': 0.1,
            'category': DataCategory.PERMITS
        }
        
        profile = await profiler._profile_single_dataset(dataset)
        
        # Validate profile structure
        assert isinstance(profile, DatasetProfile)
        assert profile.name == 'test_permit_data'
        assert profile.category == DataCategory.PERMITS
        assert profile.row_count == 1000
        assert profile.column_count == 8
        
        # Check quality metrics
        assert 0 <= profile.completeness_score <= 100
        assert isinstance(profile.quality_level, DataQualityLevel)
        assert profile.missing_data_percentage >= 0
        
        # Check schema analysis
        assert len(profile.column_types) == 8
        assert len(profile.key_columns) > 0  # Should detect permit_id, bin
        assert len(profile.entity_columns) > 0  # Should detect owner_name, contractor_name
        assert len(profile.temporal_columns) > 0  # Should detect filing_date
        
        # Check performance metrics
        assert profile.load_time_seconds > 0
        assert profile.processing_time_seconds > 0
    
    @pytest.mark.asyncio
    async def test_data_quality_analysis(self, profiler, sample_data_dir):
        """Test data quality analysis"""
        # Load sample data
        df = pd.read_csv(sample_data_dir / 'permit_data.csv')
        
        quality_metrics = await profiler._analyze_data_quality(df, is_polars=False)
        
        assert 'completeness_score' in quality_metrics
        assert 'quality_level' in quality_metrics
        assert 'missing_percentage' in quality_metrics
        assert 'duplicate_rows' in quality_metrics
        assert 'issues' in quality_metrics
        
        # Check score bounds
        assert 0 <= quality_metrics['completeness_score'] <= 100
        assert isinstance(quality_metrics['quality_level'], DataQualityLevel)
        assert quality_metrics['missing_percentage'] >= 0
        
        # Since we have some nulls in owner_name, completeness should be < 100%
        assert quality_metrics['completeness_score'] < 100
    
    @pytest.mark.asyncio
    async def test_schema_analysis(self, profiler, sample_data_dir):
        """Test schema analysis functionality"""
        df = pd.read_csv(sample_data_dir / 'permit_data.csv')
        
        schema_analysis = await profiler._analyze_schema(df, is_polars=False)
        
        assert 'column_types' in schema_analysis
        assert 'key_columns' in schema_analysis
        assert 'entity_columns' in schema_analysis
        assert 'temporal_columns' in schema_analysis
        
        # Check key column detection
        key_columns = schema_analysis['key_columns']
        assert 'permit_id' in key_columns
        assert 'bin' in key_columns
        
        # Check entity column detection
        entity_columns = schema_analysis['entity_columns']
        assert 'owner_name' in entity_columns
        assert 'contractor_name' in entity_columns
        
        # Check temporal column detection
        temporal_columns = schema_analysis['temporal_columns']
        assert 'filing_date' in temporal_columns
    
    @pytest.mark.asyncio
    async def test_statistical_summary(self, profiler, sample_data_dir):
        """Test statistical summary generation"""
        df = pd.read_csv(sample_data_dir / 'permit_data.csv')
        
        stats_summary = await profiler._generate_statistical_summary(df, is_polars=False)
        
        assert 'numerical' in stats_summary
        assert 'categorical' in stats_summary
        assert 'temporal_range' in stats_summary
        
        # Check numerical summary
        numerical = stats_summary['numerical']
        assert 'estimated_cost' in numerical
        cost_stats = numerical['estimated_cost']
        assert 'count' in cost_stats
        assert 'mean' in cost_stats
        assert 'std' in cost_stats
        assert 'min' in cost_stats
        assert 'max' in cost_stats
        
        # Check categorical summary
        categorical = stats_summary['categorical']
        assert 'work_type' in categorical
        work_type_stats = categorical['work_type']
        assert 'unique_count' in work_type_stats
        assert 'most_common' in work_type_stats
    
    @pytest.mark.asyncio
    async def test_relationship_analysis(self, profiler, sample_data_dir):
        """Test potential relationship analysis"""
        df = pd.read_csv(sample_data_dir / 'permit_data.csv')
        
        relationships = await profiler._analyze_potential_relationships(df, 'permit_data', is_polars=False)
        
        assert 'potential_joins' in relationships
        assert 'entity_overlap' in relationships
        
        potential_joins = relationships['potential_joins']
        assert 'permit_id' in potential_joins
        assert 'bin' in potential_joins
    
    @pytest.mark.asyncio
    async def test_polars_integration(self, profiler, sample_data_dir):
        """Test Polars DataFrame integration"""
        df = pl.read_csv(sample_data_dir / 'permit_data.csv')
        
        # Test quality analysis with Polars
        quality_metrics = await profiler._analyze_data_quality(df, is_polars=True)
        assert 'completeness_score' in quality_metrics
        
        # Test schema analysis with Polars
        schema_analysis = await profiler._analyze_schema(df, is_polars=True)
        assert len(schema_analysis['column_types']) == 8
        
        # Test statistical summary with Polars
        stats_summary = await profiler._generate_statistical_summary(df, is_polars=True)
        assert 'numerical' in stats_summary
        assert 'categorical' in stats_summary


class TestIntegrationAndPerformance:
    """Test integration capabilities and performance"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_profiling_workflow(self, sample_data_dir):
        """Test the complete profiling workflow"""
        profiler = NYCDOBDataProfiler(
            data_directory=str(sample_data_dir),
            output_directory=str(sample_data_dir / "metadata"),
            use_parallel=False
        )
        
        # Run comprehensive profiling
        results = await profiler.run_comprehensive_profiling()
        
        # Validate results structure
        assert 'profiling_summary' in results
        assert 'dataset_profiles' in results
        assert 'entity_profiles' in results
        assert 'cross_dataset_analysis' in results
        assert 'recommendations' in results
        
        # Check profiling summary
        summary = results['profiling_summary']
        assert summary['total_datasets'] >= 3
        assert summary['total_size_gb'] > 0
        assert summary['total_rows'] > 0
        assert 'overall_quality' in summary
        assert 'profiling_timestamp' in summary
        
        # Check dataset profiles
        dataset_profiles = results['dataset_profiles']
        assert len(dataset_profiles) >= 3
        
        # Validate individual profiles
        for name, profile in dataset_profiles.items():
            assert 'name' in profile
            assert 'category' in profile
            assert 'row_count' in profile
            assert 'completeness_score' in profile
            assert 'quality_level' in profile
    
    @pytest.mark.asyncio
    async def test_memory_management(self, sample_data_dir):
        """Test memory management with large datasets"""
        profiler = NYCDOBDataProfiler(
            data_directory=str(sample_data_dir),
            max_memory_gb=1.0  # Low memory limit
        )
        
        # Mock a large dataset
        large_dataset = {
            'name': 'large_dataset',
            'path': str(sample_data_dir / 'permit_data.csv'),
            'size_mb': 2000,  # Larger than memory limit
            'category': DataCategory.PERMITS
        }
        
        # Should trigger chunked processing
        with patch.object(profiler, '_profile_dataset_chunked') as mock_chunked:
            mock_chunked.return_value = Mock(spec=DatasetProfile)
            await profiler._profile_single_dataset(large_dataset)
            mock_chunked.assert_called_once()
    
    def test_error_handling(self, sample_data_dir):
        """Test error handling in profiling"""
        profiler = NYCDOBDataProfiler(data_directory=str(sample_data_dir))
        
        # Test with non-existent file
        bad_dataset = {
            'name': 'nonexistent',
            'path': '/nonexistent/file.csv',
            'size_mb': 1.0,
            'category': DataCategory.PERMITS
        }
        
        # Should handle errors gracefully
        with pytest.raises(Exception):
            asyncio.run(profiler._profile_single_dataset(bad_dataset))


class TestUtilityFunctions:
    """Test utility functions and helper methods"""
    
    @pytest.mark.asyncio
    async def test_profile_nyc_dob_data_function(self, sample_data_dir):
        """Test the convenience function for profiling"""
        results = await profile_nyc_dob_data(
            data_directory=str(sample_data_dir),
            output_directory=str(sample_data_dir / "metadata"),
            max_memory_gb=2.0
        )
        
        assert 'profiling_summary' in results
        assert results['profiling_summary']['total_datasets'] >= 3
    
    def test_graph_construction_recommendations(self):
        """Test graph construction recommendation generation"""
        # Mock profiling results
        mock_results = {
            'profiling_summary': {
                'total_datasets': 3,
                'overall_quality': 'good'
            },
            'dataset_profiles': {
                'permits': {
                    'temporal_columns': ['filing_date', 'issued_date'],
                    'entity_columns': ['owner_name', 'contractor_name'],
                    'key_columns': ['permit_id', 'bin']
                },
                'violations': {
                    'temporal_columns': ['violation_date'],
                    'entity_columns': ['respondent_name'],
                    'key_columns': ['violation_id', 'bin']
                }
            }
        }
        
        recommendations = get_graph_construction_recommendations(mock_results)
        
        assert len(recommendations) >= 2
        assert any(rec['type'] == 'entity_resolution' for rec in recommendations)
        assert any(rec['type'] == 'temporal_indexing' for rec in recommendations)


class TestRealDataCompatibility:
    """Test compatibility with real NYC DOB data structures"""
    
    def test_category_mapping_coverage(self):
        """Test that category mappings cover real DOB data types"""
        profiler = NYCDOBDataProfiler()
        
        # Real DOB dataset patterns
        real_dataset_names = [
            'DOB_NOW__Build___Job_Application_Filings_20250602.csv',
            'DOB_ECB_Violations_20250602.csv',
            'DOB_Complaints_Received_20250602.csv',
            'DOB_License_Info_20250602.csv',
            'Historical_DOB_Permit_Issuance_20250602.csv',
            'Housing_Litigations_20250602.csv'
        ]
        
        for dataset_name in real_dataset_names:
            category = profiler._categorize_dataset(dataset_name)
            assert isinstance(category, DataCategory)
    
    def test_entity_pattern_coverage(self):
        """Test entity pattern recognition for real data"""
        profiler = NYCDOBDataProfiler()
        
        # Real column names from NYC DOB data
        real_columns = [
            'Applicant_Name', 'Owner_Name', 'Contractor_Name',
            'Respondent_Name', 'Principal_Name', 'Contact_Person',
            'House_Number', 'Street_Name', 'Property_Address',
            'BIN', 'BBL', 'Permit_Number', 'License_ID',
            'Estimated_Cost', 'Penalty_Amount', 'Filing_Fee'
        ]
        
        # Test pattern matching
        key_columns = []
        entity_columns = []
        financial_columns = []
        
        for col in real_columns:
            col_lower = col.lower()
            
            if any(pattern in col_lower for pattern in ['id', 'number', 'bin', 'bbl']):
                key_columns.append(col)
            
            if any(pattern in col_lower for pattern_list in profiler.entity_patterns.values() 
                   for pattern in pattern_list):
                entity_columns.append(col)
            
            if any(pattern in col_lower for pattern in profiler.entity_patterns['financial']):
                financial_columns.append(col)
        
        # Should detect key columns
        assert len(key_columns) >= 3
        assert 'BIN' in key_columns
        assert 'BBL' in key_columns
        
        # Should detect entity columns
        assert len(entity_columns) >= 5
        
        # Should detect financial columns
        assert len(financial_columns) >= 2


if __name__ == "__main__":
    # Run tests manually
    async def run_tests():
        print("🧪 Running Data Profiling Framework Tests")
        
        # Create temporary test data
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            
            # Create sample data
            test_instance = TestNYCDOBDataProfiler()
            test_instance._create_sample_permit_data(data_dir / "permit_data.csv")
            test_instance._create_sample_violation_data(data_dir / "violation_data.csv")
            test_instance._create_sample_complaint_data(data_dir / "complaint_data.csv")
            
            # Test profiler initialization
            profiler = NYCDOBDataProfiler(
                data_directory=str(data_dir),
                output_directory=str(data_dir / "metadata"),
                use_parallel=False
            )
            
            print("✅ Profiler initialization successful")
            
            # Test dataset discovery
            datasets = await profiler._discover_datasets()
            print(f"✅ Discovered {len(datasets)} datasets")
            
            # Test single dataset profiling
            if datasets:
                profile = await profiler._profile_single_dataset(datasets[0])
                print(f"✅ Dataset profiling successful: {profile.name}")
                print(f"   - Rows: {profile.row_count:,}")
                print(f"   - Columns: {profile.column_count}")
                print(f"   - Quality: {profile.quality_level.value}")
                print(f"   - Completeness: {profile.completeness_score:.1f}%")
            
            # Test comprehensive profiling
            results = await profiler.run_comprehensive_profiling()
            print(f"✅ Comprehensive profiling completed")
            print(f"   - Total datasets: {results['profiling_summary']['total_datasets']}")
            print(f"   - Total rows: {results['profiling_summary']['total_rows']:,}")
            print(f"   - Overall quality: {results['profiling_summary']['overall_quality']}")
            
            # Test graph construction recommendations
            recommendations = get_graph_construction_recommendations(results)
            print(f"✅ Generated {len(recommendations)} graph construction recommendations")
            
            print("\n🎉 All foundation tests passed! Ready for advanced community detection.")
    
    asyncio.run(run_tests()) 