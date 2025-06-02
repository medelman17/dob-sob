"""
NYC DOB Data Profiling Framework

Comprehensive foundation framework for NYC Department of Buildings data analysis,
providing solid foundations for advanced community detection and fraud analysis.

This module handles large-scale data profiling, quality assessment, entity resolution,
and preparation for Graphiti temporal knowledge graph construction.
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import gc

import polars as pl
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

try:
    from graphiti_core import Graphiti
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    Graphiti = None


class DataCategory(Enum):
    """NYC DOB data categories"""
    PERMITS = "permits"
    VIOLATIONS = "violations"
    COMPLAINTS = "complaints"
    LICENSES = "licenses"
    PROPERTIES = "properties"
    REGISTRATIONS = "registrations"
    CERTIFICATES = "certificates"
    APPLICATIONS = "applications"
    FACADES = "facades"
    LITIGATION = "litigation"
    MAINTENANCE = "maintenance"
    CONSTRUCTION = "construction"


class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"     # >95% complete, minimal issues
    GOOD = "good"              # 85-95% complete, minor issues
    FAIR = "fair"              # 70-85% complete, moderate issues
    POOR = "poor"              # <70% complete, significant issues


@dataclass
class DatasetProfile:
    """Comprehensive profile of a dataset"""
    name: str
    category: DataCategory
    file_path: str
    size_mb: float
    row_count: int
    column_count: int
    memory_usage_mb: float
    
    # Data quality metrics
    completeness_score: float
    quality_level: DataQualityLevel
    missing_data_percentage: float
    duplicate_rows: int
    
    # Schema analysis
    column_types: Dict[str, str]
    key_columns: List[str]
    entity_columns: List[str]
    temporal_columns: List[str]
    
    # Statistical summary
    numerical_summary: Dict[str, Dict[str, float]]
    categorical_summary: Dict[str, Dict[str, Any]]
    temporal_range: Optional[Tuple[datetime, datetime]]
    
    # Relationships and connections
    potential_joins: List[str]
    entity_overlap: Dict[str, float]
    
    # Performance metrics
    load_time_seconds: float
    processing_time_seconds: float
    
    # Data anomalies
    anomalies: List[Dict[str, Any]]
    quality_issues: List[str]


@dataclass
class EntityProfile:
    """Profile of a specific entity type in the data"""
    entity_type: str
    total_count: int
    unique_count: int
    duplicate_rate: float
    completeness_score: float
    
    # Entity characteristics
    name_patterns: List[str]
    id_formats: List[str]
    address_standardization: float
    
    # Cross-dataset presence
    dataset_presence: Dict[str, int]
    join_success_rates: Dict[str, float]
    
    # Potential fraud indicators
    suspicious_patterns: List[Dict[str, Any]]
    risk_indicators: List[str]


@dataclass
class CrossDatasetAnalysis:
    """Analysis of relationships across multiple datasets"""
    dataset_pairs: List[Tuple[str, str]]
    join_statistics: Dict[str, Dict[str, Any]]
    entity_overlap_matrix: Dict[str, Dict[str, float]]
    temporal_alignment: Dict[str, float]
    relationship_strength: Dict[str, float]
    
    # Network analysis readiness
    graph_construction_readiness: float
    recommended_join_strategies: List[Dict[str, Any]]
    entity_resolution_needs: List[str]


class NYCDOBDataProfiler:
    """
    Comprehensive data profiling system for NYC DOB datasets.
    
    Provides solid foundations for advanced community detection and fraud analysis
    by thoroughly analyzing data quality, structure, and relationships.
    """
    
    def __init__(self, 
                 data_directory: str = "../../data",
                 output_directory: str = "../../data/metadata",
                 max_memory_gb: float = 8.0,
                 use_parallel: bool = True):
        """Initialize the data profiler"""
        self.data_dir = Path(data_directory)
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance configuration
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.use_parallel = use_parallel
        self.max_workers = min(psutil.cpu_count(), 8)
        
        # Results storage
        self.dataset_profiles: Dict[str, DatasetProfile] = {}
        self.entity_profiles: Dict[str, EntityProfile] = {}
        self.cross_dataset_analysis: Optional[CrossDatasetAnalysis] = None
        
        # Data category mappings
        self.category_mappings = {
            'permit': DataCategory.PERMITS,
            'violation': DataCategory.VIOLATIONS,
            'complaint': DataCategory.COMPLAINTS,
            'license': DataCategory.LICENSES,
            'property': DataCategory.PROPERTIES,
            'registration': DataCategory.REGISTRATIONS,
            'certificate': DataCategory.CERTIFICATES,
            'application': DataCategory.APPLICATIONS,
            'facade': DataCategory.FACADES,
            'litigation': DataCategory.LITIGATION,
            'maintenance': DataCategory.MAINTENANCE,
            'construction': DataCategory.CONSTRUCTION
        }
        
        # Configure logging
        self._setup_logging()
        
        # Entity extraction patterns
        self.entity_patterns = {
            'company_name': [
                'applicant', 'owner', 'contractor', 'respondent',
                'company', 'corporation', 'firm', 'business'
            ],
            'person_name': [
                'name', 'contact', 'principal', 'officer', 'manager'
            ],
            'address': [
                'address', 'street', 'location', 'site', 'property'
            ],
            'identifier': [
                'id', 'number', 'permit', 'license', 'registration',
                'bin', 'bbl', 'ein', 'registration_id'
            ],
            'financial': [
                'amount', 'cost', 'fee', 'value', 'estimated_cost'
            ]
        }
    
    def _setup_logging(self) -> None:
        """Configure logging for the profiler"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'profiling.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def run_comprehensive_profiling(self) -> Dict[str, Any]:
        """
        Run comprehensive profiling of all NYC DOB datasets.
        
        Returns complete profiling results including dataset profiles,
        entity analysis, and cross-dataset relationships.
        """
        self.logger.info("Starting comprehensive NYC DOB data profiling")
        start_time = datetime.now()
        
        # Step 1: Discover and catalog datasets
        datasets = await self._discover_datasets()
        self.logger.info(f"Discovered {len(datasets)} datasets")
        
        # Step 2: Profile individual datasets
        if self.use_parallel:
            await self._profile_datasets_parallel(datasets)
        else:
            await self._profile_datasets_sequential(datasets)
        
        # Step 3: Analyze entities across datasets
        await self._analyze_entities()
        
        # Step 4: Cross-dataset relationship analysis
        await self._analyze_cross_dataset_relationships()
        
        # Step 5: Generate comprehensive report
        results = await self._generate_comprehensive_report()
        
        # Step 6: Save results
        await self._save_profiling_results(results)
        
        duration = datetime.now() - start_time
        self.logger.info(f"Comprehensive profiling completed in {duration}")
        
        return results
    
    async def _discover_datasets(self) -> List[Dict[str, Any]]:
        """Discover all available datasets in the data directory"""
        datasets = []
        
        # Check main data directory
        for file_path in self.data_dir.glob("*.csv"):
            if file_path.stat().st_size > 0:  # Non-empty files only
                datasets.append({
                    'name': file_path.stem,
                    'path': str(file_path),
                    'size_mb': file_path.stat().st_size / (1024**2),
                    'category': self._categorize_dataset(file_path.name)
                })
        
        # Check subdirectories
        for category_dir in ['raw', 'incremental']:
            category_path = self.data_dir / category_dir
            if category_path.exists():
                for subdir in category_path.iterdir():
                    if subdir.is_dir():
                        for file_path in subdir.glob("*.csv"):
                            if file_path.stat().st_size > 0:
                                datasets.append({
                                    'name': f"{subdir.name}_{file_path.stem}",
                                    'path': str(file_path),
                                    'size_mb': file_path.stat().st_size / (1024**2),
                                    'category': self._categorize_dataset(subdir.name)
                                })
        
        # Sort by size for efficient processing
        datasets.sort(key=lambda x: x['size_mb'])
        return datasets
    
    def _categorize_dataset(self, filename: str) -> DataCategory:
        """Categorize dataset based on filename patterns"""
        filename_lower = filename.lower()
        
        for keyword, category in self.category_mappings.items():
            if keyword in filename_lower:
                return category
        
        return DataCategory.PROPERTIES  # Default category
    
    async def _profile_datasets_parallel(self, datasets: List[Dict[str, Any]]) -> None:
        """Profile datasets in parallel for better performance"""
        self.logger.info(f"Profiling {len(datasets)} datasets in parallel")
        
        # Group datasets by size for balanced processing
        small_datasets = [d for d in datasets if d['size_mb'] < 50]
        large_datasets = [d for d in datasets if d['size_mb'] >= 50]
        
        # Process small datasets in parallel
        if small_datasets:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                tasks = [
                    executor.submit(self._profile_single_dataset_sync, dataset)
                    for dataset in small_datasets
                ]
                
                for task in as_completed(tasks):
                    try:
                        profile = task.result()
                        self.dataset_profiles[profile.name] = profile
                    except Exception as e:
                        self.logger.error(f"Error profiling dataset: {e}")
        
        # Process large datasets sequentially to avoid memory issues
        for dataset in large_datasets:
            try:
                profile = await self._profile_single_dataset(dataset)
                self.dataset_profiles[profile.name] = profile
                gc.collect()  # Force garbage collection after large datasets
            except Exception as e:
                self.logger.error(f"Error profiling large dataset {dataset['name']}: {e}")
    
    async def _profile_datasets_sequential(self, datasets: List[Dict[str, Any]]) -> None:
        """Profile datasets sequentially"""
        self.logger.info(f"Profiling {len(datasets)} datasets sequentially")
        
        for dataset in datasets:
            try:
                profile = await self._profile_single_dataset(dataset)
                self.dataset_profiles[profile.name] = profile
                gc.collect()
            except Exception as e:
                self.logger.error(f"Error profiling dataset {dataset['name']}: {e}")
    
    def _profile_single_dataset_sync(self, dataset: Dict[str, Any]) -> DatasetProfile:
        """Synchronous version for parallel processing"""
        return asyncio.run(self._profile_single_dataset(dataset))
    
    async def _profile_single_dataset(self, dataset: Dict[str, Any]) -> DatasetProfile:
        """Profile a single dataset comprehensively"""
        self.logger.info(f"Profiling dataset: {dataset['name']}")
        start_time = datetime.now()
        
        # Check memory before loading
        available_memory = psutil.virtual_memory().available
        if dataset['size_mb'] * 3 > available_memory / (1024**2):  # 3x safety factor
            self.logger.warning(f"Low memory for dataset {dataset['name']}, using chunked processing")
            return await self._profile_dataset_chunked(dataset)
        
        try:
            # Load data efficiently
            load_start = datetime.now()
            if dataset['size_mb'] > 100:  # Use Polars for large datasets
                df = pl.read_csv(dataset['path'])
                is_polars = True
            else:
                df = pd.read_csv(dataset['path'])
                is_polars = False
            
            load_time = (datetime.now() - load_start).total_seconds()
            
            # Calculate basic metrics
            if is_polars:
                row_count = df.height
                column_count = df.width
                memory_usage = df.estimated_size("mb")
            else:
                row_count = len(df)
                column_count = len(df.columns)
                memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
            
            # Analyze data quality
            quality_metrics = await self._analyze_data_quality(df, is_polars)
            
            # Analyze schema
            schema_analysis = await self._analyze_schema(df, is_polars)
            
            # Generate statistical summary
            stats_summary = await self._generate_statistical_summary(df, is_polars)
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(df, is_polars)
            
            # Analyze potential relationships
            relationships = await self._analyze_potential_relationships(df, dataset['name'], is_polars)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            profile = DatasetProfile(
                name=dataset['name'],
                category=dataset['category'],
                file_path=dataset['path'],
                size_mb=dataset['size_mb'],
                row_count=row_count,
                column_count=column_count,
                memory_usage_mb=memory_usage,
                
                completeness_score=quality_metrics['completeness_score'],
                quality_level=quality_metrics['quality_level'],
                missing_data_percentage=quality_metrics['missing_percentage'],
                duplicate_rows=quality_metrics['duplicate_rows'],
                
                column_types=schema_analysis['column_types'],
                key_columns=schema_analysis['key_columns'],
                entity_columns=schema_analysis['entity_columns'],
                temporal_columns=schema_analysis['temporal_columns'],
                
                numerical_summary=stats_summary['numerical'],
                categorical_summary=stats_summary['categorical'],
                temporal_range=stats_summary['temporal_range'],
                
                potential_joins=relationships['potential_joins'],
                entity_overlap=relationships['entity_overlap'],
                
                load_time_seconds=load_time,
                processing_time_seconds=processing_time,
                
                anomalies=anomalies,
                quality_issues=quality_metrics['issues']
            )
            
            self.logger.info(f"Completed profiling {dataset['name']}: "
                           f"{row_count:,} rows, {column_count} columns, "
                           f"Quality: {quality_metrics['quality_level'].value}")
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error profiling dataset {dataset['name']}: {e}")
            raise
    
    async def _profile_dataset_chunked(self, dataset: Dict[str, Any]) -> DatasetProfile:
        """Profile large datasets using chunked processing"""
        self.logger.info(f"Using chunked processing for {dataset['name']}")
        
        chunk_size = 10000
        profiles = []
        
        # Process in chunks
        reader = pd.read_csv(dataset['path'], chunksize=chunk_size)
        for i, chunk in enumerate(reader):
            if i > 100:  # Limit chunks for very large files
                break
                
            # Create mini-profile for chunk
            chunk_profile = await self._profile_single_dataset({
                'name': f"{dataset['name']}_chunk_{i}",
                'path': dataset['path'],
                'size_mb': 0,  # Calculated separately
                'category': dataset['category']
            })
            profiles.append(chunk_profile)
        
        # Aggregate chunk profiles
        return await self._aggregate_chunk_profiles(profiles, dataset)
    
    async def _analyze_data_quality(self, df: Union[pl.DataFrame, pd.DataFrame], 
                                  is_polars: bool) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        if is_polars:
            total_cells = df.height * df.width
            null_count = df.null_count().sum(axis=1)[0]
            duplicate_rows = df.height - df.n_unique()
        else:
            total_cells = df.size
            null_count = df.isnull().sum().sum()
            duplicate_rows = len(df) - len(df.drop_duplicates())
        
        missing_percentage = (null_count / total_cells) * 100
        completeness_score = (1 - (null_count / total_cells)) * 100
        
        # Determine quality level
        if completeness_score >= 95:
            quality_level = DataQualityLevel.EXCELLENT
        elif completeness_score >= 85:
            quality_level = DataQualityLevel.GOOD
        elif completeness_score >= 70:
            quality_level = DataQualityLevel.FAIR
        else:
            quality_level = DataQualityLevel.POOR
        
        # Identify specific quality issues
        issues = []
        if missing_percentage > 20:
            issues.append(f"High missing data: {missing_percentage:.1f}%")
        if duplicate_rows > 0:
            issues.append(f"Duplicate rows found: {duplicate_rows:,}")
        
        return {
            'completeness_score': completeness_score,
            'quality_level': quality_level,
            'missing_percentage': missing_percentage,
            'duplicate_rows': duplicate_rows,
            'issues': issues
        }
    
    async def _analyze_schema(self, df: Union[pl.DataFrame, pd.DataFrame], 
                            is_polars: bool) -> Dict[str, Any]:
        """Analyze dataset schema and identify key columns"""
        if is_polars:
            columns = df.columns
            dtypes = {col: str(df[col].dtype) for col in columns}
        else:
            columns = df.columns.tolist()
            dtypes = {col: str(df[col].dtype) for col in columns}
        
        # Identify column types by pattern matching
        key_columns = []
        entity_columns = []
        temporal_columns = []
        
        for col in columns:
            col_lower = col.lower()
            
            # Key columns (likely join keys)
            if any(pattern in col_lower for pattern in ['id', 'number', 'bin', 'bbl']):
                key_columns.append(col)
            
            # Entity columns (companies, people, addresses)
            if any(pattern in col_lower for pattern_list in self.entity_patterns.values() 
                   for pattern in pattern_list):
                entity_columns.append(col)
            
            # Temporal columns
            if any(pattern in col_lower for pattern in ['date', 'time', 'created', 'issued', 'filed']):
                temporal_columns.append(col)
        
        return {
            'column_types': dtypes,
            'key_columns': key_columns,
            'entity_columns': entity_columns,
            'temporal_columns': temporal_columns
        }
    
    async def _generate_statistical_summary(self, df: Union[pl.DataFrame, pd.DataFrame], 
                                          is_polars: bool) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        numerical_summary = {}
        categorical_summary = {}
        temporal_range = None
        
        if is_polars:
            # Numerical columns
            numeric_cols = [col for col, dtype in df.schema.items() 
                          if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]
            
            for col in numeric_cols:
                stats = df[col].describe()
                numerical_summary[col] = {
                    'count': df[col].count(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                }
            
            # Categorical columns
            categorical_cols = [col for col, dtype in df.schema.items() 
                              if dtype == pl.Utf8][:10]  # Limit for performance
            
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                categorical_summary[col] = {
                    'unique_count': df[col].n_unique(),
                    'most_common': value_counts.head(5).to_dict() if len(value_counts) > 0 else {},
                    'null_count': df[col].null_count()
                }
        
        else:
            # Pandas processing
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                numerical_summary[col] = {
                    'count': df[col].count(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                }
            
            categorical_cols = df.select_dtypes(include=['object']).columns[:10]
            for col in categorical_cols:
                try:
                    value_counts = df[col].value_counts().head(5)
                    categorical_summary[col] = {
                        'unique_count': df[col].nunique(),
                        'most_common': value_counts.to_dict(),
                        'null_count': df[col].isnull().sum()
                    }
                except:
                    categorical_summary[col] = {
                        'unique_count': 0,
                        'most_common': {},
                        'null_count': len(df)
                    }
        
        return {
            'numerical': numerical_summary,
            'categorical': categorical_summary,
            'temporal_range': temporal_range
        }
    
    async def _detect_anomalies(self, df: Union[pl.DataFrame, pd.DataFrame], 
                              is_polars: bool) -> List[Dict[str, Any]]:
        """Detect data anomalies and quality issues"""
        anomalies = []
        
        # For now, return placeholder - can be extended with specific anomaly detection
        return anomalies
    
    async def _analyze_potential_relationships(self, df: Union[pl.DataFrame, pd.DataFrame], 
                                             dataset_name: str, is_polars: bool) -> Dict[str, Any]:
        """Analyze potential relationships with other datasets"""
        potential_joins = []
        entity_overlap = {}
        
        # Identify potential join columns
        columns = df.columns if is_polars else df.columns.tolist()
        
        for col in columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['id', 'number', 'bin', 'bbl', 'permit']):
                potential_joins.append(col)
        
        return {
            'potential_joins': potential_joins,
            'entity_overlap': entity_overlap
        }
    
    async def _aggregate_chunk_profiles(self, profiles: List[DatasetProfile], 
                                      dataset: Dict[str, Any]) -> DatasetProfile:
        """Aggregate multiple chunk profiles into a single profile"""
        # Placeholder implementation - aggregate statistics from chunks
        if not profiles:
            raise ValueError("No profiles to aggregate")
        
        # Use first profile as template and aggregate others
        base_profile = profiles[0]
        base_profile.name = dataset['name']
        base_profile.size_mb = dataset['size_mb']
        
        return base_profile
    
    async def _analyze_entities(self) -> None:
        """Analyze entity patterns across all datasets"""
        self.logger.info("Analyzing entity patterns across datasets")
        
        # For each entity type, analyze across datasets
        entity_types = ['company', 'person', 'address', 'property']
        
        for entity_type in entity_types:
            entity_profile = await self._analyze_entity_type(entity_type)
            self.entity_profiles[entity_type] = entity_profile
    
    async def _analyze_entity_type(self, entity_type: str) -> EntityProfile:
        """Analyze a specific entity type across datasets"""
        # Placeholder implementation
        return EntityProfile(
            entity_type=entity_type,
            total_count=0,
            unique_count=0,
            duplicate_rate=0.0,
            completeness_score=0.0,
            name_patterns=[],
            id_formats=[],
            address_standardization=0.0,
            dataset_presence={},
            join_success_rates={},
            suspicious_patterns=[],
            risk_indicators=[]
        )
    
    async def _analyze_cross_dataset_relationships(self) -> None:
        """Analyze relationships across multiple datasets"""
        self.logger.info("Analyzing cross-dataset relationships")
        
        # Placeholder implementation
        self.cross_dataset_analysis = CrossDatasetAnalysis(
            dataset_pairs=[],
            join_statistics={},
            entity_overlap_matrix={},
            temporal_alignment={},
            relationship_strength={},
            graph_construction_readiness=0.0,
            recommended_join_strategies=[],
            entity_resolution_needs=[]
        )
    
    async def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiling report"""
        return {
            'profiling_summary': {
                'total_datasets': len(self.dataset_profiles),
                'total_size_gb': sum(p.size_mb for p in self.dataset_profiles.values()) / 1024,
                'total_rows': sum(p.row_count for p in self.dataset_profiles.values()),
                'overall_quality': self._calculate_overall_quality(),
                'profiling_timestamp': datetime.now().isoformat()
            },
            'dataset_profiles': {name: asdict(profile) for name, profile in self.dataset_profiles.items()},
            'entity_profiles': {name: asdict(profile) for name, profile in self.entity_profiles.items()},
            'cross_dataset_analysis': asdict(self.cross_dataset_analysis) if self.cross_dataset_analysis else {},
            'recommendations': await self._generate_recommendations()
        }
    
    def _calculate_overall_quality(self) -> str:
        """Calculate overall data quality across all datasets"""
        if not self.dataset_profiles:
            return "unknown"
        
        avg_completeness = sum(p.completeness_score for p in self.dataset_profiles.values()) / len(self.dataset_profiles)
        
        if avg_completeness >= 95:
            return "excellent"
        elif avg_completeness >= 85:
            return "good"
        elif avg_completeness >= 70:
            return "fair"
        else:
            return "poor"
    
    async def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for data processing and analysis"""
        recommendations = []
        
        # Memory optimization recommendations
        large_datasets = [p for p in self.dataset_profiles.values() if p.size_mb > 500]
        if large_datasets:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'Memory Optimization Needed',
                'description': f'Found {len(large_datasets)} large datasets that may require chunked processing',
                'datasets': [p.name for p in large_datasets],
                'action': 'Implement lazy loading and chunked processing for these datasets'
            })
        
        # Data quality recommendations
        poor_quality = [p for p in self.dataset_profiles.values() if p.quality_level == DataQualityLevel.POOR]
        if poor_quality:
            recommendations.append({
                'type': 'quality',
                'priority': 'medium',
                'title': 'Data Quality Issues',
                'description': f'Found {len(poor_quality)} datasets with poor data quality',
                'datasets': [p.name for p in poor_quality],
                'action': 'Implement data cleaning and validation procedures'
            })
        
        return recommendations
    
    async def _save_profiling_results(self, results: Dict[str, Any]) -> None:
        """Save profiling results to files"""
        # Save main results
        results_file = self.output_dir / f"profiling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / "profiling_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results['profiling_summary'], f, indent=2, default=str)
        
        self.logger.info(f"Profiling results saved to {results_file}")


# Utility functions for advanced data profiling

async def profile_nyc_dob_data(data_directory: str = "../../data",
                             output_directory: str = "../../data/metadata",
                             max_memory_gb: float = 8.0) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive NYC DOB data profiling.
    
    Args:
        data_directory: Path to NYC DOB data files
        output_directory: Path to save profiling results
        max_memory_gb: Maximum memory to use for processing
    
    Returns:
        Comprehensive profiling results dictionary
    """
    profiler = NYCDOBDataProfiler(
        data_directory=data_directory,
        output_directory=output_directory,
        max_memory_gb=max_memory_gb
    )
    
    return await profiler.run_comprehensive_profiling()


def get_graph_construction_recommendations(profiling_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate specific recommendations for Graphiti graph construction
    based on profiling results.
    """
    recommendations = []
    
    # Analyze join capabilities
    dataset_profiles = profiling_results.get('dataset_profiles', {})
    
    # Recommend entity resolution strategies
    recommendations.append({
        'type': 'entity_resolution',
        'title': 'Entity Resolution Strategy',
        'description': 'Implement fuzzy matching for company names and addresses',
        'priority': 'high',
        'implementation': 'Use string similarity algorithms for entity deduplication'
    })
    
    # Recommend temporal indexing
    temporal_datasets = [name for name, profile in dataset_profiles.items() 
                        if profile.get('temporal_columns')]
    if temporal_datasets:
        recommendations.append({
            'type': 'temporal_indexing',
            'title': 'Temporal Graph Construction',
            'description': 'Build time-aware graph structures for fraud detection',
            'datasets': temporal_datasets,
            'implementation': 'Use Graphiti temporal capabilities for time-series analysis'
        })
    
    return recommendations


if __name__ == "__main__":
    # Example usage
    async def main():
        print("Starting NYC DOB Data Profiling Framework")
        results = await profile_nyc_dob_data()
        print(f"Profiling completed: {results['profiling_summary']['total_datasets']} datasets analyzed")
        
        # Generate graph construction recommendations
        recommendations = get_graph_construction_recommendations(results)
        print(f"Generated {len(recommendations)} recommendations for graph construction")
    
    asyncio.run(main()) 