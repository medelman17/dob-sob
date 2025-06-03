"""
Dataset registry and configuration management.

Centralizes dataset definitions and provides utilities for
dataset discovery and configuration.
"""

from typing import Dict, List, Optional
from pathlib import Path
import yaml

from ...core.config import get_settings, DatasetConfig
from ...core.exceptions import DatasetNotFoundError


class DatasetRegistry:
    """Registry for NYC DOB dataset configurations."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize dataset registry."""
        self.config_path = config_path or Path("config/datasets.yaml")
        self._datasets: Dict[str, DatasetConfig] = {}
        self._load_datasets()
    
    def _load_datasets(self) -> None:
        """Load dataset configurations from YAML file or use defaults."""
        # Try to load from YAML first
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            for name, config_data in data.get('datasets', {}).items():
                self._datasets[name] = DatasetConfig(**config_data)
        else:
            # Use hardcoded defaults if no config file
            self._load_default_datasets()
    
    def _load_default_datasets(self) -> None:
        """Load default dataset configurations."""
        default_datasets = {
            "housing_litigations": {
                "id": "59kj-x8nc",
                "name": "Housing Litigations",
                "description": "Housing litigation cases and status",
                "update_frequency": "daily",
                "estimated_size_mb": 50,
                "date_field": "case_open_date",
                "supports_incremental": True,
                "priority": "high"
            },
            "dob_violations": {
                "id": "3h2n-5cm9",
                "name": "DOB Violations",
                "description": "Department of Buildings violations",
                "update_frequency": "daily",
                "estimated_size_mb": 250,
                "date_field": "issue_date",
                "supports_incremental": True,
                "priority": "high"
            },
            "ecb_violations": {
                "id": "6bgk-3dad",
                "name": "ECB Violations",
                "description": "Environmental Control Board violations",
                "update_frequency": "daily",
                "estimated_size_mb": 300,
                "date_field": "issue_date",
                "supports_incremental": True,
                "priority": "high"
            },
            "complaints_received": {
                "id": "eabe-havv",
                "name": "Complaints Received",
                "description": "DOB complaints received",
                "update_frequency": "daily",
                "estimated_size_mb": 317,
                "date_field": "date_entered",
                "supports_incremental": True,
                "priority": "high"
            },
            "maintenance_code_violations": {
                "id": "wvxf-dwi5",
                "name": "Maintenance Code Violations",
                "description": "Housing maintenance code violations",
                "update_frequency": "daily",
                "estimated_size_mb": 400,
                "date_field": "approved_date",
                "supports_incremental": True,
                "priority": "high"
            },
            "certificates_of_occupancy": {
                "id": "bs8b-p36w",
                "name": "Certificates of Occupancy",
                "description": "Certificate of occupancy records",
                "update_frequency": "daily",
                "estimated_size_mb": 200,
                "date_field": "issue_date",
                "supports_incremental": True,
                "priority": "high"
            },
            "active_projects_under_construction": {
                "id": "w9ak-ipjd",
                "name": "Active Projects Under Construction",
                "description": "Currently active construction projects",
                "update_frequency": "daily",
                "estimated_size_mb": 150,
                "date_field": "latest_action_date",
                "supports_incremental": True,
                "priority": "high"
            },
            "historical_permit_issuance": {
                "id": "ipu4-2q9a",
                "name": "Historical Permit Issuance",
                "description": "Historical DOB permit issuance data",
                "update_frequency": "daily",
                "estimated_size_mb": 1200,
                "date_field": "issuance_date",
                "supports_incremental": True,
                "priority": "medium"
            },
            "job_application_filings": {
                "id": "ic3t-wcy2",
                "name": "Job Application Filings",
                "description": "DOB job application filings",
                "update_frequency": "daily",
                "estimated_size_mb": 800,
                "date_field": "filing_date",
                "supports_incremental": True,
                "priority": "medium"
            },
            "multiple_dwelling_registrations": {
                "id": "tesw-yqqr",
                "name": "Multiple Dwelling Registrations",
                "description": "Multiple dwelling registration data",
                "update_frequency": "daily",
                "estimated_size_mb": 100,
                "date_field": "registration_date",
                "supports_incremental": True,
                "priority": "medium"
            }
        }
        
        for name, config_data in default_datasets.items():
            self._datasets[name] = DatasetConfig(**config_data)
    
    def get_dataset(self, name: str) -> DatasetConfig:
        """Get dataset configuration by name."""
        if name not in self._datasets:
            raise DatasetNotFoundError(name)
        return self._datasets[name]
    
    def list_datasets(self) -> List[str]:
        """List all available dataset names."""
        return list(self._datasets.keys())
    
    def list_by_priority(self, priority: str) -> List[str]:
        """List datasets by priority level."""
        return [
            name for name, config in self._datasets.items()
            if config.priority == priority
        ]
    
    def list_primary_datasets(self) -> List[str]:
        """List high-priority datasets for primary processing."""
        return self.list_by_priority("high")
    
    def list_secondary_datasets(self) -> List[str]:
        """List medium-priority datasets for secondary processing."""
        return self.list_by_priority("medium")
    
    def get_total_estimated_size(self, dataset_names: List[str] = None) -> int:
        """Get total estimated size in MB for specified datasets."""
        if dataset_names is None:
            dataset_names = self.list_datasets()
        
        total_size = 0
        for name in dataset_names:
            if name in self._datasets:
                total_size += self._datasets[name].estimated_size_mb
        
        return total_size
    
    def filter_incremental_datasets(self, dataset_names: List[str] = None) -> List[str]:
        """Filter datasets that support incremental updates."""
        if dataset_names is None:
            dataset_names = self.list_datasets()
        
        return [
            name for name in dataset_names
            if name in self._datasets and self._datasets[name].supports_incremental
        ]
    
    def get_update_schedule(self) -> Dict[str, List[str]]:
        """Group datasets by update frequency."""
        schedule = {}
        for name, config in self._datasets.items():
            frequency = config.update_frequency
            if frequency not in schedule:
                schedule[frequency] = []
            schedule[frequency].append(name)
        
        return schedule
    
    def export_config(self, output_path: Path) -> None:
        """Export current dataset configurations to YAML file."""
        datasets_dict = {}
        for name, config in self._datasets.items():
            datasets_dict[name] = config.dict()
        
        config_data = {"datasets": datasets_dict}
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=True)


# Global dataset registry instance
dataset_registry = DatasetRegistry()