#!/usr/bin/env python3
"""
NYC DOB Dataset Metadata Downloader for dob-sob Platform

This module downloads and processes dataset metadata/data dictionaries from NYC Open Data
using the Socrata API. It extracts comprehensive schema information including column 
definitions, data types, constraints, and field descriptions for all target datasets.

Key Features:
- Downloads metadata via /api/views/{dataset_id} endpoints
- Extracts complete schema and column definitions
- Stores structured metadata in JSON/YAML format
- Validates data quality and completeness
- Provides fraud detection optimized field mappings
- Handles errors and missing datasets gracefully

Author: dob-sob Team
Version: 1.0
"""

import json
import yaml
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ColumnMetadata:
    """Represents metadata for a single dataset column"""
    field_name: str
    display_name: str
    data_type: str
    description: Optional[str] = None
    format: Optional[str] = None
    width: Optional[int] = None
    position: Optional[int] = None
    default_value: Optional[str] = None
    cached_contents: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass 
class DatasetMetadata:
    """Represents complete metadata for a dataset"""
    dataset_id: str
    name: str
    description: str
    agency: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    update_frequency: Optional[str] = None
    attribution: Optional[str] = None
    license: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    download_count: Optional[int] = None
    view_count: Optional[int] = None
    columns: List[ColumnMetadata] = None
    metadata_retrieved_at: datetime = None
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = []
        if self.metadata_retrieved_at is None:
            self.metadata_retrieved_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        if data.get('created_at'):
            data['created_at'] = self.created_at.isoformat()
        if data.get('updated_at'):
            data['updated_at'] = self.updated_at.isoformat()
        if data.get('metadata_retrieved_at'):
            data['metadata_retrieved_at'] = self.metadata_retrieved_at.isoformat()
        # Convert column objects
        data['columns'] = [col.to_dict() for col in self.columns]
        return {k: v for k, v in data.items() if v is not None}


class NYCMetadataDownloader:
    """
    Downloads and processes metadata for NYC DOB datasets
    """
    
    BASE_METADATA_URL = "https://data.cityofnewyork.us/api/views"
    
    def __init__(self, output_dir: str = "data/metadata"):
        """
        Initialize metadata downloader
        
        Args:
            output_dir: Directory to store metadata files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'dob-sob-metadata-downloader/1.0'
        })
        
        # Load dataset configuration
        self.datasets_config = self._load_datasets_config()
        
    def _load_datasets_config(self) -> Dict[str, Any]:
        """Load dataset configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / "config" / "datasets.yaml"
        
        if not config_path.exists():
            logger.error(f"Dataset configuration not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('datasets', {})
        except Exception as e:
            logger.error(f"Error loading dataset configuration: {e}")
            return {}
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            self.session.close()
    
    def _fetch_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch raw metadata for a dataset from NYC Open Data API
        
        Args:
            dataset_id: Dataset identifier (e.g., '59kj-x8nc')
            
        Returns:
            Raw metadata dictionary or None if error
        """
        url = f"{self.BASE_METADATA_URL}/{dataset_id}"
        
        try:
            response = self.session.get(url, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch metadata for {dataset_id}: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching metadata for {dataset_id}: {e}")
            return None
    
    def _parse_column_metadata(self, column_data: Dict[str, Any]) -> ColumnMetadata:
        """
        Parse column metadata from API response
        
        Args:
            column_data: Raw column data from API
            
        Returns:
            Parsed column metadata
        """
        return ColumnMetadata(
            field_name=column_data.get('fieldName', ''),
            display_name=column_data.get('name', ''),
            data_type=column_data.get('dataTypeName', ''),
            description=column_data.get('description'),
            format=column_data.get('format', {}).get('precisionStyle') if column_data.get('format') else None,
            width=column_data.get('width'),
            position=column_data.get('position'),
            default_value=column_data.get('format', {}).get('noCommas') if column_data.get('format') else None,
            cached_contents=column_data.get('cachedContents')
        )
    
    def _parse_dataset_metadata(self, dataset_key: str, raw_metadata: Dict[str, Any]) -> DatasetMetadata:
        """
        Parse dataset metadata from API response
        
        Args:
            dataset_key: Dataset key from configuration
            raw_metadata: Raw metadata from API
            
        Returns:
            Parsed dataset metadata
        """
        # Parse basic information
        metadata = DatasetMetadata(
            dataset_id=raw_metadata.get('id', ''),
            name=raw_metadata.get('name', ''),
            description=raw_metadata.get('description', ''),
            agency=raw_metadata.get('attribution', ''),
            category=raw_metadata.get('category', ''),
            tags=raw_metadata.get('tags', []),
            update_frequency=raw_metadata.get('metadata', {}).get('rdfSubject') if raw_metadata.get('metadata') else None,
            attribution=raw_metadata.get('attribution'),
            license=raw_metadata.get('license', {}).get('name') if raw_metadata.get('license') else None,
            row_count=raw_metadata.get('rowsUpdatedAt'),
            column_count=len(raw_metadata.get('columns', [])),
            download_count=raw_metadata.get('downloadCount'),
            view_count=raw_metadata.get('viewCount')
        )
        
        # Parse timestamps
        if raw_metadata.get('createdAt'):
            metadata.created_at = datetime.fromtimestamp(raw_metadata['createdAt'])
        
        if raw_metadata.get('rowsUpdatedAt'):
            metadata.updated_at = datetime.fromtimestamp(raw_metadata['rowsUpdatedAt'])
        
        # Parse columns
        columns_data = raw_metadata.get('columns', [])
        metadata.columns = [
            self._parse_column_metadata(col_data) 
            for col_data in columns_data
        ]
        
        return metadata
    
    def download_dataset_metadata(self, dataset_key: str, dataset_id: str) -> Optional[DatasetMetadata]:
        """
        Download metadata for a specific dataset
        
        Args:
            dataset_key: Dataset key from configuration
            dataset_id: Dataset ID for API call
            
        Returns:
            Dataset metadata or None if error
        """
        logger.info(f"Downloading metadata for {dataset_key} (ID: {dataset_id})")
        
        raw_metadata = self._fetch_dataset_metadata(dataset_id)
        if not raw_metadata:
            return None
        
        try:
            parsed_metadata = self._parse_dataset_metadata(dataset_key, raw_metadata)
            logger.info(f"Successfully parsed metadata for {dataset_key}: {len(parsed_metadata.columns)} columns")
            return parsed_metadata
        except Exception as e:
            logger.error(f"Error parsing metadata for {dataset_key}: {e}")
            return None
    
    def save_metadata(self, dataset_key: str, metadata: DatasetMetadata, format_type: str = "json"):
        """
        Save metadata to file
        
        Args:
            dataset_key: Dataset key for filename
            metadata: Metadata to save
            format_type: File format ('json' or 'yaml')
        """
        filename = f"{dataset_key}_metadata.{format_type}"
        file_path = self.output_dir / filename
        
        try:
            metadata_dict = metadata.to_dict()
            
            with open(file_path, 'w') as f:
                if format_type == "json":
                    f.write(json.dumps(metadata_dict, indent=2, default=str))
                elif format_type == "yaml":
                    f.write(yaml.dump(metadata_dict, default_flow_style=False, sort_keys=False))
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Saved metadata for {dataset_key} to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving metadata for {dataset_key}: {e}")
    
    def download_all_metadata(self, format_type: str = "json") -> Dict[str, DatasetMetadata]:
        """
        Download metadata for all configured datasets
        
        Args:
            format_type: Output format ('json' or 'yaml')
            
        Returns:
            Dictionary of successfully downloaded metadata
        """
        logger.info(f"Starting metadata download for {len(self.datasets_config)} datasets")
        
        results = {}
        errors = {}
        
        for dataset_key, config in self.datasets_config.items():
            dataset_id = config.get('id')
            if not dataset_id:
                logger.warning(f"No ID found for dataset: {dataset_key}")
                continue
            
            try:
                metadata = self.download_dataset_metadata(dataset_key, dataset_id)
                if metadata:
                    self.save_metadata(dataset_key, metadata, format_type)
                    results[dataset_key] = metadata
                else:
                    errors[dataset_key] = "Failed to download metadata"
                    
            except Exception as e:
                logger.error(f"Error processing {dataset_key}: {e}")
                errors[dataset_key] = str(e)
        
        # Generate summary report
        self._generate_summary_report(results, errors, format_type)
        
        logger.info(f"Metadata download complete: {len(results)} successful, {len(errors)} failed")
        return results
    
    def _generate_summary_report(self, 
                                     results: Dict[str, DatasetMetadata], 
                                     errors: Dict[str, str],
                                     format_type: str):
        """Generate a summary report of the metadata download"""
        
        report = {
            "summary": {
                "total_datasets": len(self.datasets_config),
                "successful_downloads": len(results),
                "failed_downloads": len(errors),
                "success_rate": f"{len(results) / len(self.datasets_config) * 100:.1f}%",
                "generated_at": datetime.now().isoformat()
            },
            "successful_datasets": {},
            "failed_datasets": errors,
            "schema_summary": {
                "total_columns": 0,
                "common_field_types": {},
                "datasets_by_column_count": {}
            }
        }
        
        # Populate successful datasets and schema analysis
        total_columns = 0
        field_type_counts = {}
        
        for dataset_key, metadata in results.items():
            report["successful_datasets"][dataset_key] = {
                "name": metadata.name,
                "column_count": len(metadata.columns),
                "row_count": metadata.row_count,
                "agency": metadata.agency,
                "last_updated": metadata.updated_at.isoformat() if metadata.updated_at else None
            }
            
            # Column analysis
            total_columns += len(metadata.columns)
            report["schema_summary"]["datasets_by_column_count"][dataset_key] = len(metadata.columns)
            
            for column in metadata.columns:
                field_type = column.data_type
                field_type_counts[field_type] = field_type_counts.get(field_type, 0) + 1
        
        report["schema_summary"]["total_columns"] = total_columns
        report["schema_summary"]["common_field_types"] = dict(
            sorted(field_type_counts.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Save summary report
        summary_path = self.output_dir / f"metadata_summary.{format_type}"
        with open(summary_path, 'w') as f:
            if format_type == "json":
                f.write(json.dumps(report, indent=2, default=str))
            else:
                f.write(yaml.dump(report, default_flow_style=False, sort_keys=False))
        
        logger.info(f"Summary report saved to {summary_path}")
    
    def generate_fraud_detection_schema(self, results: Dict[str, DatasetMetadata]) -> Dict[str, Dict[str, str]]:
        """
        Generate optimized field mappings for fraud detection
        
        Args:
            results: Downloaded metadata results
            
        Returns:
            Dictionary mapping datasets to their key fraud detection fields
        """
        fraud_fields = {}
        
        # Key fraud detection field patterns
        key_patterns = {
            'bin': ['bin', 'building_id', 'buildingid'],
            'owner': ['owner', 'ownername', 'respondent', 'licensee'],
            'address': ['address', 'housenumber', 'streetname'],
            'dates': ['date', 'issued', 'filed', 'updated'],
            'violations': ['violation', 'complaint', 'penalty'],
            'permits': ['permit', 'job', 'application', 'certificate'],
            'amounts': ['amount', 'fee', 'penalty', 'cost']
        }
        
        for dataset_key, metadata in results.items():
            dataset_fields = {}
            
            for column in metadata.columns:
                field_name = column.field_name.lower()
                
                for category, patterns in key_patterns.items():
                    if any(pattern in field_name for pattern in patterns):
                        if category not in dataset_fields:
                            dataset_fields[category] = []
                        dataset_fields[category].append({
                            'field_name': column.field_name,
                            'display_name': column.display_name,
                            'data_type': column.data_type,
                            'description': column.description
                        })
            
            if dataset_fields:
                fraud_fields[dataset_key] = dataset_fields
        
        return fraud_fields


def main():
    """CLI interface for metadata downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download NYC DOB dataset metadata and data dictionaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all metadata as JSON
  python metadata_downloader.py --all
  
  # Download specific datasets as YAML  
  python metadata_downloader.py --datasets housing_litigations,dob_violations --format yaml
  
  # Generate fraud detection schema mapping
  python metadata_downloader.py --all --fraud-schema
        """
    )
    
    parser.add_argument("--all", action="store_true",
                       help="Download metadata for all configured datasets")
    parser.add_argument("--datasets", type=str,
                       help="Comma-separated list of specific datasets")
    parser.add_argument("--format", type=str, default="json",
                       choices=["json", "yaml"],
                       help="Output format (default: json)")
    parser.add_argument("--output-dir", type=str, default="data/metadata",
                       help="Output directory (default: data/metadata)")
    parser.add_argument("--fraud-schema", action="store_true",
                       help="Generate fraud detection schema mapping")
    
    args = parser.parse_args()
    
    if not args.all and not args.datasets:
        parser.print_help()
        print("\nPlease specify --all or --datasets")
        return
    
    with NYCMetadataDownloader(output_dir=args.output_dir) as downloader:
        
        if args.datasets:
            # Filter to specific datasets
            dataset_keys = [key.strip() for key in args.datasets.split(",")]
            filtered_config = {
                key: config for key, config in downloader.datasets_config.items()
                if key in dataset_keys
            }
            downloader.datasets_config = filtered_config
        
        # Download metadata
        results = downloader.download_all_metadata(format_type=args.format)
        
        # Generate fraud detection schema if requested
        if args.fraud_schema:
            fraud_schema = downloader.generate_fraud_detection_schema(results)
            
            fraud_schema_path = Path(args.output_dir) / f"fraud_detection_schema.{args.format}"
            with open(fraud_schema_path, 'w') as f:
                if args.format == "json":
                    f.write(json.dumps(fraud_schema, indent=2))
                else:
                    f.write(yaml.dump(fraud_schema, default_flow_style=False))
            
            logger.info(f"Fraud detection schema saved to {fraud_schema_path}")
        
        # Print summary
        print(f"\n=== Metadata Download Summary ===")
        print(f"Total datasets: {len(downloader.datasets_config)}")
        print(f"Successful: {len(results)}")
        print(f"Failed: {len(downloader.datasets_config) - len(results)}")
        print(f"Output directory: {args.output_dir}")
        print(f"Format: {args.format}")
        
        if results:
            print(f"\n✅ Successfully downloaded metadata for:")
            for dataset_key, metadata in results.items():
                print(f"  - {dataset_key}: {metadata.name} ({len(metadata.columns)} columns)")


if __name__ == "__main__":
    main()