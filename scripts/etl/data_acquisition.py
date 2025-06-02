#!/usr/bin/env python3
"""
NYC DOB Data Acquisition Module

This module handles downloading and managing NYC Department of Buildings datasets
from the NYC Open Data portal. Supports both CSV downloads and OData API integration
for 18 target datasets with advanced incremental download capabilities.

Target Datasets:
1. Housing Litigations
2. License Info  
3. Historical Permit Issuance
4. Multiple Dwelling Registrations
5. Certificates of Occupancy
6. Property Data
7. Job Application Filings
8. ECB Violations
9. Active Projects Under Construction
10. Complaints Received
11. Maintenance Code Violations
12. Approved Permits
13. DOB Violations
14. Permit Issuance
15. DOB Job Application Filings
16. Order to Repair/Vacate
17. Facades Compliance Filings
18. Registration Contacts
"""

import os
import json
import csv
import hashlib
import logging
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from urllib.parse import urljoin, quote_plus
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class DataAcquisitionConfig:
    """Configuration management for data acquisition"""
    
    # NYC Open Data base URLs
    BASE_URL = "https://data.cityofnewyork.us"
    API_BASE_URL = "https://data.cityofnewyork.us/resource"
    ODATA_BASE_URL = "https://data.cityofnewyork.us/OData.svc"
    
    # Dataset identifiers and metadata with update frequency information
    DATASETS = {
        "housing_litigations": {
            "id": "59kj-x8nc",  # Confirmed - Housing Litigations
            "name": "Housing Litigations",
            "description": "Housing litigation cases and status",
            "update_frequency": "daily",
            "estimated_size_mb": 50,
            "date_field": "case_open_date",  # Field to use for incremental updates
            "supports_incremental": True
        },
        "license_info": {
            "id": "ipu4-2q9a",  # DOB Permit Issuance (contains license info)
            "name": "License Information", 
            "description": "DOB license information and status",
            "update_frequency": "daily",
            "estimated_size_mb": 15,
            "date_field": "issuance_date",
            "supports_incremental": True
        },
        "historical_permit_issuance": {
            "id": "ipu4-2q9a",  # Confirmed - DOB Permit Issuance
            "name": "Historical Permit Issuance",
            "description": "Historical DOB permit issuance data",
            "update_frequency": "daily", 
            "estimated_size_mb": 1200,
            "date_field": "issuance_date",
            "supports_incremental": True
        },
        "multiple_dwelling_registrations": {
            "id": "tesw-yqqr",  # Need to verify - Multiple Dwelling Registrations
            "name": "Multiple Dwelling Registrations",
            "description": "Multiple dwelling registration data",
            "update_frequency": "daily",
            "estimated_size_mb": 100,
            "date_field": "registration_date",
            "supports_incremental": True
        },
        "certificates_of_occupancy": {
            "id": "bs8b-p36w",  # Confirmed - DOB Certificate Of Occupancy
            "name": "Certificates of Occupancy",
            "description": "Certificate of occupancy records",
            "update_frequency": "daily",
            "estimated_size_mb": 200,
            "date_field": "issue_date",
            "supports_incremental": True
        },
        "property_data": {
            "id": "rgy2-tti8",  # Confirmed - Property Valuation and Assessment Data
            "name": "Property Data",
            "description": "Property valuation and assessment data",
            "update_frequency": "quarterly",
            "estimated_size_mb": 500,
            "date_field": "last_modified",
            "supports_incremental": False  # Property data is typically full refresh
        },
        "job_application_filings": {
            "id": "ic3t-wcy2",  # Confirmed - DOB Job Application Filings
            "name": "Job Application Filings",
            "description": "DOB job application filings",
            "update_frequency": "daily",
            "estimated_size_mb": 800,
            "date_field": "filing_date",
            "supports_incremental": True
        },
        "ecb_violations": {
            "id": "6bgk-3dad",  # Confirmed - DOB ECB Violations
            "name": "ECB Violations",
            "description": "Environmental Control Board violations",
            "update_frequency": "daily",
            "estimated_size_mb": 300,
            "date_field": "issue_date",
            "supports_incremental": True
        },
        "active_projects_under_construction": {
            "id": "w9ak-ipjd",  # Confirmed - DOB NOW: Build
            "name": "Active Projects Under Construction",
            "description": "Currently active construction projects",
            "update_frequency": "daily",
            "estimated_size_mb": 150,
            "date_field": "latest_action_date",
            "supports_incremental": True
        },
        "complaints_received": {
            "id": "eabe-havv",  # Need to verify - DOB Complaints
            "name": "Complaints Received",
            "description": "DOB complaints received",
            "update_frequency": "daily",
            "estimated_size_mb": 317,
            "date_field": "date_entered",
            "supports_incremental": True
        },
        "maintenance_code_violations": {
            "id": "wvxf-dwi5",  # Confirmed - Housing Maintenance Code Violations
            "name": "Maintenance Code Violations",
            "description": "Housing maintenance code violations",
            "update_frequency": "daily",
            "estimated_size_mb": 400,
            "date_field": "issue_date",
            "supports_incremental": True
        },
        "approved_permits": {
            "id": "ipu4-2q9a",  # DOB Permit Issuance (approved subset)
            "name": "Approved Permits",
            "description": "Approved DOB permits",
            "update_frequency": "daily",
            "estimated_size_mb": 600,
            "date_field": "issuance_date",
            "supports_incremental": True
        },
        "dob_violations": {
            "id": "3h2n-5cm9",  # Confirmed - DOB Violations
            "name": "DOB Violations",
            "description": "Department of Buildings violations",
            "update_frequency": "daily",
            "estimated_size_mb": 250,
            "date_field": "issue_date",
            "supports_incremental": True
        },
        "permit_issuance": {
            "id": "ipu4-2q9a",  # Confirmed - DOB Permit Issuance
            "name": "Permit Issuance", 
            "description": "DOB permit issuance records",
            "update_frequency": "daily",
            "estimated_size_mb": 700,
            "date_field": "issuance_date",
            "supports_incremental": True
        },
        "dob_job_application_filings": {
            "id": "ic3t-wcy2",  # Confirmed - DOB Job Application Filings
            "name": "DOB Job Application Filings",
            "description": "DOB job application filings",
            "update_frequency": "daily",
            "estimated_size_mb": 850,
            "date_field": "filing_date",
            "supports_incremental": True
        },
        "order_to_repair_vacate": {
            "id": "6bgk-3dad",  # Need to verify - may be same as ECB Violations
            "name": "Order to Repair/Vacate",
            "description": "Orders to repair or vacate",
            "update_frequency": "daily",
            "estimated_size_mb": 75,
            "date_field": "issue_date",
            "supports_incremental": True
        },
        "facades_compliance_filings": {
            "id": "qsjy-6dfz",  # Need to verify - Facades Compliance
            "name": "Facades Compliance Filings",
            "description": "Building facades compliance filings",
            "update_frequency": "daily",
            "estimated_size_mb": 25,
            "date_field": "filing_date",
            "supports_incremental": True
        },
        "registration_contacts": {
            "id": "feu5-w2e2",  # Need to verify - Registration Contacts
            "name": "Registration Contacts",
            "description": "Building registration contact information",
            "update_frequency": "daily",
            "estimated_size_mb": 30,
            "date_field": "registration_date",
            "supports_incremental": True
        }
    }
    
    # Request configuration
    REQUEST_TIMEOUT = 300  # 5 minutes for large datasets
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 2
    CHUNK_SIZE = 8192  # 8KB chunks for streaming downloads
    
    # Incremental download configuration
    INCREMENTAL_CHUNK_SIZE = 1048576  # 1MB chunks for resumable downloads
    MAX_INCREMENTAL_DAYS = 30  # Maximum days to look back for incremental updates
    MIN_FILE_SIZE_FOR_RESUME = 10485760  # 10MB minimum for resumable downloads
    
    # File paths
    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    METADATA_DIR = DATA_DIR / "metadata"
    INCREMENTAL_DIR = DATA_DIR / "incremental"
    LOGS_DIR = Path("logs")

class DataAcquisitionLogger:
    """Centralized logging for data acquisition operations"""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("data_acquisition")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create logs directory if it doesn't exist
        DataAcquisitionConfig.LOGS_DIR.mkdir(exist_ok=True)
        
        # File handler
        log_file = DataAcquisitionConfig.LOGS_DIR / f"data_acquisition_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger

class HTTPClient:
    """HTTP client with retry logic and robust error handling"""
    
    def __init__(self, timeout: int = DataAcquisitionConfig.REQUEST_TIMEOUT):
        self.session = requests.Session()
        self.timeout = timeout
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=DataAcquisitionConfig.MAX_RETRIES,
            backoff_factor=DataAcquisitionConfig.RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set user agent
        self.session.headers.update({
            'User-Agent': 'NYC-DOB-Data-Explorer/1.0'
        })
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Make GET request with error handling"""
        try:
            response = self.session.get(url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise DataAcquisitionError(f"HTTP request failed: {e}")
    
    def head(self, url: str, **kwargs) -> requests.Response:
        """Make HEAD request with error handling"""
        try:
            response = self.session.head(url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise DataAcquisitionError(f"HTTP HEAD request failed: {e}")
    
    def stream_download(self, url: str, file_path: Path, progress_callback=None, resume: bool = False) -> None:
        """Stream download large files with progress tracking and resume capability"""
        headers = {}
        initial_pos = 0
        
        # Support resumable downloads
        if resume and file_path.exists():
            initial_pos = file_path.stat().st_size
            headers['Range'] = f'bytes={initial_pos}-'
        
        try:
            response = self.session.get(url, stream=True, timeout=self.timeout, headers=headers)
            
            # Handle partial content response for resume
            if response.status_code == 206:  # Partial content
                mode = 'ab'  # Append mode for resume
            elif response.status_code == 200:
                mode = 'wb'  # Write mode for new download
                initial_pos = 0  # Reset position for new download
            else:
                response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0)) + initial_pos
            downloaded_size = initial_pos
            
            with open(file_path, mode) as f:
                for chunk in response.iter_content(chunk_size=DataAcquisitionConfig.INCREMENTAL_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded_size, total_size)
        
        except requests.exceptions.RequestException as e:
            if file_path.exists() and not resume:
                file_path.unlink()  # Clean up partial download
            raise DataAcquisitionError(f"Download failed: {e}")

class ChecksumValidator:
    """File integrity validation using checksums"""
    
    @staticmethod
    def calculate_md5(file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @staticmethod
    def verify_checksum(file_path: Path, expected_checksum: str, algorithm: str = "md5") -> bool:
        """Verify file checksum"""
        if algorithm.lower() == "md5":
            actual_checksum = ChecksumValidator.calculate_md5(file_path)
        elif algorithm.lower() == "sha256":
            actual_checksum = ChecksumValidator.calculate_sha256(file_path)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return actual_checksum.lower() == expected_checksum.lower()

class IncrementalDownloadManager:
    """Manages incremental downloads and smart update detection"""
    
    def __init__(self, metadata_manager, logger):
        self.metadata_manager = metadata_manager
        self.logger = logger
    
    def should_use_incremental(self, dataset_key: str, dataset_config: Dict) -> bool:
        """Determine if incremental download should be used"""
        if not dataset_config.get('supports_incremental', False):
            return False
        
        # Check if we have previous download metadata
        local_metadata = self.metadata_manager.get_dataset_metadata(dataset_key)
        if not local_metadata.get('last_download_timestamp'):
            self.logger.info(f"No previous download found for {dataset_key}, using full download")
            return False
        
        # Check if last download was recent enough for incremental
        last_download = datetime.fromisoformat(local_metadata['last_download_timestamp'])
        days_since_download = (datetime.now() - last_download).days
        
        if days_since_download > DataAcquisitionConfig.MAX_INCREMENTAL_DAYS:
            self.logger.info(f"Last download for {dataset_key} was {days_since_download} days ago, using full download")
            return False
        
        return True
    
    def build_incremental_url(self, dataset_id: str, date_field: str, since_date: datetime, format_type: str = "csv") -> str:
        """Build URL for incremental download using date filtering"""
        since_date_str = since_date.strftime('%Y-%m-%dT%H:%M:%S')
        
        if format_type.lower() == "csv":
            # Use OData-style filtering for CSV exports
            filter_clause = f"{date_field} >= '{since_date_str}'"
            return f"{DataAcquisitionConfig.BASE_URL}/api/views/{dataset_id}/rows.csv?$where={quote_plus(filter_clause)}"
        elif format_type.lower() == "json":
            # Use SoQL query for JSON API
            filter_clause = f"{date_field} >= '{since_date_str}'"
            return f"{DataAcquisitionConfig.API_BASE_URL}/{dataset_id}.json?$where={quote_plus(filter_clause)}"
        else:
            raise ValueError(f"Incremental downloads not supported for format: {format_type}")
    
    def get_incremental_since_date(self, dataset_key: str) -> datetime:
        """Get the date to use for incremental downloads"""
        local_metadata = self.metadata_manager.get_dataset_metadata(dataset_key)
        last_download = datetime.fromisoformat(local_metadata['last_download_timestamp'])
        
        # Go back 1 day from last download to ensure we don't miss any records
        return last_download - timedelta(days=1)
    
    def merge_incremental_data(self, dataset_key: str, incremental_file: Path, base_file: Path) -> Path:
        """Merge incremental data with existing base file"""
        merged_file = base_file.parent / f"{base_file.stem}_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            # Simple merge for CSV files - append new data to existing file
            with open(merged_file, 'w', newline='', encoding='utf-8') as outfile:
                # Copy base file
                with open(base_file, 'r', encoding='utf-8') as base:
                    outfile.write(base.read())
                
                # Append incremental data (skip header)
                with open(incremental_file, 'r', encoding='utf-8') as inc:
                    lines = inc.readlines()
                    if len(lines) > 1:  # Skip header if present
                        outfile.writelines(lines[1:])
            
            self.logger.info(f"Merged incremental data: {merged_file}")
            return merged_file
            
        except Exception as e:
            self.logger.error(f"Failed to merge incremental data: {e}")
            raise DataAcquisitionError(f"Incremental merge failed: {e}")

class MetadataManager:
    """Manage download metadata and tracking with enhanced incremental support"""
    
    def __init__(self):
        self.metadata_file = DataAcquisitionConfig.METADATA_DIR / "download_metadata.json"
        self.incremental_metadata_file = DataAcquisitionConfig.METADATA_DIR / "incremental_metadata.json"
        self._ensure_metadata_dir()
        self.metadata = self._load_metadata()
        self.incremental_metadata = self._load_incremental_metadata()
    
    def _ensure_metadata_dir(self):
        """Ensure metadata directory exists"""
        DataAcquisitionConfig.METADATA_DIR.mkdir(parents=True, exist_ok=True)
        DataAcquisitionConfig.INCREMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _load_incremental_metadata(self) -> Dict:
        """Load incremental download metadata"""
        if self.incremental_metadata_file.exists():
            try:
                with open(self.incremental_metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def save_incremental_metadata(self):
        """Save incremental metadata to file"""
        with open(self.incremental_metadata_file, 'w') as f:
            json.dump(self.incremental_metadata, f, indent=2, default=str)
    
    def update_dataset_metadata(self, dataset_key: str, **kwargs):
        """Update metadata for a specific dataset"""
        if dataset_key not in self.metadata:
            self.metadata[dataset_key] = {}
        
        self.metadata[dataset_key].update(kwargs)
        self.metadata[dataset_key]['last_updated'] = datetime.now().isoformat()
        self.save_metadata()
    
    def update_incremental_metadata(self, dataset_key: str, **kwargs):
        """Update incremental download metadata"""
        if dataset_key not in self.incremental_metadata:
            self.incremental_metadata[dataset_key] = {}
        
        self.incremental_metadata[dataset_key].update(kwargs)
        self.incremental_metadata[dataset_key]['last_updated'] = datetime.now().isoformat()
        self.save_incremental_metadata()
    
    def get_dataset_metadata(self, dataset_key: str) -> Dict:
        """Get metadata for a specific dataset"""
        return self.metadata.get(dataset_key, {})
    
    def get_incremental_metadata(self, dataset_key: str) -> Dict:
        """Get incremental metadata for a specific dataset"""
        return self.incremental_metadata.get(dataset_key, {})

class DataAcquisitionError(Exception):
    """Custom exception for data acquisition errors"""
    pass

class ProgressTracker:
    """Progress tracking for downloads with enhanced incremental support"""
    
    def __init__(self, dataset_name: str, logger: logging.Logger, is_incremental: bool = False):
        self.dataset_name = dataset_name
        self.logger = logger
        self.is_incremental = is_incremental
        self.start_time = None
        self.last_update = None
    
    def start(self):
        """Start progress tracking"""
        self.start_time = time.time()
        self.last_update = self.start_time
        download_type = "incremental" if self.is_incremental else "full"
        self.logger.info(f"Starting {download_type} download: {self.dataset_name}")
    
    def update(self, downloaded: int, total: int):
        """Update progress"""
        current_time = time.time()
        
        # Only log progress every 5 seconds to avoid spam
        if current_time - self.last_update >= 5:
            percentage = (downloaded / total) * 100 if total > 0 else 0
            elapsed = current_time - self.start_time
            speed = downloaded / elapsed if elapsed > 0 else 0
            
            download_type = "incremental" if self.is_incremental else "full"
            self.logger.info(
                f"{self.dataset_name} ({download_type}): {percentage:.1f}% "
                f"({downloaded:,}/{total:,} bytes) "
                f"Speed: {speed/1024/1024:.1f} MB/s"
            )
            self.last_update = current_time
    
    def complete(self, file_size: int):
        """Mark download as complete"""
        elapsed = time.time() - self.start_time
        speed = file_size / elapsed if elapsed > 0 else 0
        
        download_type = "incremental" if self.is_incremental else "full"
        self.logger.info(
            f"Completed {download_type} download: {self.dataset_name} "
            f"({file_size:,} bytes in {elapsed:.1f}s, "
            f"avg speed: {speed/1024/1024:.1f} MB/s)"
        )

def setup_directory_structure():
    """Create necessary directory structure for data acquisition"""
    directories = [
        DataAcquisitionConfig.DATA_DIR,
        DataAcquisitionConfig.RAW_DATA_DIR,
        DataAcquisitionConfig.METADATA_DIR,
        DataAcquisitionConfig.INCREMENTAL_DIR,
        DataAcquisitionConfig.LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each dataset
    for dataset_key in DataAcquisitionConfig.DATASETS.keys():
        dataset_dir = DataAcquisitionConfig.RAW_DATA_DIR / dataset_key
        dataset_dir.mkdir(exist_ok=True)
        
        # Create incremental subdirectories
        incremental_dir = DataAcquisitionConfig.INCREMENTAL_DIR / dataset_key
        incremental_dir.mkdir(exist_ok=True)

def get_dataset_url(dataset_id: str, format_type: str = "csv") -> str:
    """Generate dataset URL for different formats"""
    if format_type.lower() == "csv":
        return f"{DataAcquisitionConfig.BASE_URL}/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD"
    elif format_type.lower() == "json":
        return f"{DataAcquisitionConfig.API_BASE_URL}/{dataset_id}.json"
    elif format_type.lower() == "odata":
        return f"{DataAcquisitionConfig.ODATA_BASE_URL}/{dataset_id}"
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def validate_dataset_config():
    """Validate dataset configuration"""
    logger = DataAcquisitionLogger().get_logger()
    
    missing_ids = []
    incremental_supported = 0
    
    for key, config in DataAcquisitionConfig.DATASETS.items():
        if not config.get('id'):
            missing_ids.append(key)
        if config.get('supports_incremental', False):
            incremental_supported += 1
    
    if missing_ids:
        logger.warning(f"Missing dataset IDs for: {', '.join(missing_ids)}")
        logger.warning("These datasets will need to be manually configured with correct NYC Open Data IDs")
    
    logger.info(f"Validated configuration for {len(DataAcquisitionConfig.DATASETS)} datasets")
    logger.info(f"Incremental download support: {incremental_supported}/{len(DataAcquisitionConfig.DATASETS)} datasets")

class NYCDataDownloader:
    """Main class for downloading NYC DOB datasets with advanced incremental capabilities"""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = DataAcquisitionLogger(log_level).get_logger()
        self.http_client = HTTPClient()
        self.metadata_manager = MetadataManager()
        self.incremental_manager = IncrementalDownloadManager(self.metadata_manager, self.logger)
        self.setup_directories()
    
    def setup_directories(self):
        """Setup directory structure"""
        setup_directory_structure()
        self.logger.info("Directory structure initialized")
    
    def download_dataset(self, dataset_key: str, format_type: str = "csv", 
                        force_download: bool = False, use_incremental: bool = True) -> bool:
        """
        Download a specific dataset with intelligent incremental download support
        
        Args:
            dataset_key: Key for the dataset in DATASETS config
            format_type: Format to download (csv, json, odata)
            force_download: Force download even if file exists
            use_incremental: Enable incremental downloads when possible
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if dataset_key not in DataAcquisitionConfig.DATASETS:
            self.logger.error(f"Unknown dataset: {dataset_key}")
            return False
        
        dataset_config = DataAcquisitionConfig.DATASETS[dataset_key]
        dataset_id = dataset_config["id"]
        dataset_name = dataset_config["name"]
        
        try:
            # Check if incremental download is appropriate
            is_incremental = False
            if use_incremental and not force_download:
                is_incremental = self.incremental_manager.should_use_incremental(dataset_key, dataset_config)
            
            # Create progress tracker
            progress_tracker = ProgressTracker(dataset_name, self.logger, is_incremental)
            
            # Determine download approach
            if is_incremental:
                return self._download_incremental(dataset_key, dataset_config, format_type, progress_tracker)
            else:
                return self._download_full(dataset_key, dataset_config, format_type, force_download, progress_tracker)
                
        except Exception as e:
            self.logger.error(f"Failed to download {dataset_name}: {e}")
            return False
    
    def _download_full(self, dataset_key: str, dataset_config: Dict, format_type: str, 
                      force_download: bool, progress_tracker: ProgressTracker) -> bool:
        """Download complete dataset"""
        dataset_id = dataset_config["id"]
        dataset_name = dataset_config["name"]
        
        # Get download URL
        download_url = get_dataset_url(dataset_id, format_type)
        
        # Determine output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = format_type.lower()
        filename = f"{dataset_key}_{timestamp}.{file_extension}"
        output_path = DataAcquisitionConfig.RAW_DATA_DIR / dataset_key / filename
        
        # Check if we should skip download
        if output_path.exists() and not force_download:
            self.logger.info(f"File already exists: {output_path}")
            return True
        
        self.logger.info(f"Starting full download: {dataset_name}")
        self.logger.info(f"URL: {download_url}")
        self.logger.info(f"Output: {output_path}")
        
        # Start progress tracking
        progress_tracker.start()
        
        # Check if resume is possible for large files
        resume_download = False
        if output_path.exists() and output_path.stat().st_size >= DataAcquisitionConfig.MIN_FILE_SIZE_FOR_RESUME:
            resume_download = True
            self.logger.info(f"Resuming download from {output_path.stat().st_size:,} bytes")
        
        # Download file
        self.http_client.stream_download(
            download_url, 
            output_path, 
            progress_callback=progress_tracker.update,
            resume=resume_download
        )
        
        # Calculate file size and complete progress tracking
        file_size = output_path.stat().st_size
        progress_tracker.complete(file_size)
        
        # Calculate and store checksum
        checksum = ChecksumValidator.calculate_md5(output_path)
        
        # Update metadata
        self.metadata_manager.update_dataset_metadata(
            dataset_key,
            last_download_timestamp=datetime.now().isoformat(),
            file_path=str(output_path),
            file_size=file_size,
            checksum_md5=checksum,
            download_url=download_url,
            format_type=format_type,
            download_type="full"
        )
        
        self.logger.info(f"Successfully downloaded {dataset_name}")
        self.logger.info(f"File size: {file_size:,} bytes")
        self.logger.info(f"MD5 checksum: {checksum}")
        
        return True
    
    def _download_incremental(self, dataset_key: str, dataset_config: Dict, format_type: str,
                             progress_tracker: ProgressTracker) -> bool:
        """Download only new/updated records since last download"""
        dataset_id = dataset_config["id"]
        dataset_name = dataset_config["name"]
        date_field = dataset_config.get("date_field")
        
        if not date_field:
            self.logger.warning(f"No date field configured for {dataset_key}, falling back to full download")
            return self._download_full(dataset_key, dataset_config, format_type, False, progress_tracker)
        
        # Get incremental download parameters
        since_date = self.incremental_manager.get_incremental_since_date(dataset_key)
        
        # Build incremental download URL
        try:
            download_url = self.incremental_manager.build_incremental_url(
                dataset_id, date_field, since_date, format_type
            )
        except ValueError as e:
            self.logger.warning(f"Incremental download not supported for format {format_type}: {e}")
            return self._download_full(dataset_key, dataset_config, format_type, False, progress_tracker)
        
        # Determine output file path for incremental data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = format_type.lower()
        filename = f"{dataset_key}_incremental_{timestamp}.{file_extension}"
        incremental_path = DataAcquisitionConfig.INCREMENTAL_DIR / dataset_key / filename
        
        self.logger.info(f"Starting incremental download: {dataset_name}")
        self.logger.info(f"Since date: {since_date.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"URL: {download_url}")
        self.logger.info(f"Output: {incremental_path}")
        
        # Start progress tracking
        progress_tracker.start()
        
        # Download incremental data
        self.http_client.stream_download(
            download_url,
            incremental_path,
            progress_callback=progress_tracker.update
        )
        
        # Calculate file size and complete progress tracking
        file_size = incremental_path.stat().st_size
        progress_tracker.complete(file_size)
        
        # Check if incremental file has meaningful data
        if file_size < 100:  # Less than 100 bytes likely means no new data
            self.logger.info(f"No new data found for {dataset_name} (file size: {file_size} bytes)")
            incremental_path.unlink()  # Remove empty file
            
            # Update incremental metadata anyway to track the attempt
            self.metadata_manager.update_incremental_metadata(
                dataset_key,
                last_incremental_check=datetime.now().isoformat(),
                since_date=since_date.isoformat(),
                records_found=0,
                file_size=0
            )
            return True
        
        # Calculate checksum for incremental data
        checksum = ChecksumValidator.calculate_md5(incremental_path)
        
        # Find the most recent base file for merging
        base_metadata = self.metadata_manager.get_dataset_metadata(dataset_key)
        base_file_path = base_metadata.get('file_path')
        
        if base_file_path and Path(base_file_path).exists():
            # Merge incremental data with base file
            try:
                merged_file = self.incremental_manager.merge_incremental_data(
                    dataset_key, incremental_path, Path(base_file_path)
                )
                
                # Update main metadata to point to merged file
                merged_checksum = ChecksumValidator.calculate_md5(merged_file)
                self.metadata_manager.update_dataset_metadata(
                    dataset_key,
                    last_download_timestamp=datetime.now().isoformat(),
                    file_path=str(merged_file),
                    file_size=merged_file.stat().st_size,
                    checksum_md5=merged_checksum,
                    download_url=download_url,
                    format_type=format_type,
                    download_type="incremental_merged"
                )
                
                self.logger.info(f"Successfully merged incremental data for {dataset_name}")
                self.logger.info(f"Merged file: {merged_file}")
                self.logger.info(f"Total size: {merged_file.stat().st_size:,} bytes")
                
            except Exception as e:
                self.logger.error(f"Failed to merge incremental data for {dataset_name}: {e}")
                # Keep incremental file as-is and update metadata
                self.metadata_manager.update_dataset_metadata(
                    dataset_key,
                    last_download_timestamp=datetime.now().isoformat(),
                    file_path=str(incremental_path),
                    file_size=file_size,
                    checksum_md5=checksum,
                    download_url=download_url,
                    format_type=format_type,
                    download_type="incremental_only"
                )
        
        # Update incremental metadata
        self.metadata_manager.update_incremental_metadata(
            dataset_key,
            last_incremental_download=datetime.now().isoformat(),
            incremental_file_path=str(incremental_path),
            incremental_file_size=file_size,
            incremental_checksum_md5=checksum,
            since_date=since_date.isoformat(),
            records_found="unknown"  # We'd need to parse CSV to get exact count
        )
        
        self.logger.info(f"Successfully completed incremental download for {dataset_name}")
        self.logger.info(f"Incremental file size: {file_size:,} bytes")
        self.logger.info(f"MD5 checksum: {checksum}")
        
        return True
    
    def download_primary_datasets(self, format_type: str = "csv", 
                                 force_download: bool = False,
                                 use_incremental: bool = True) -> Dict[str, bool]:
        """
        Download the first 9 primary datasets
        
        Returns:
            Dict[str, bool]: Results for each dataset download
        """
        primary_datasets = [
            "housing_litigations",
            "license_info", 
            "historical_permit_issuance",
            "multiple_dwelling_registrations",
            "certificates_of_occupancy",
            "property_data",
            "job_application_filings", 
            "ecb_violations",
            "active_projects_under_construction"
        ]
        
        results = {}
        self.logger.info(f"Starting download of {len(primary_datasets)} primary datasets")
        
        for dataset_key in primary_datasets:
            self.logger.info(f"Processing dataset: {dataset_key}")
            results[dataset_key] = self.download_dataset(
                dataset_key, 
                format_type, 
                force_download,
                use_incremental
            )
            
            # Brief pause between downloads to be respectful
            time.sleep(2)
        
        # Log summary
        successful = sum(1 for success in results.values() if success)
        self.logger.info(f"Primary dataset downloads completed: {successful}/{len(primary_datasets)} successful")
        
        return results
    
    def download_secondary_datasets(self, format_type: str = "csv", 
                                   force_download: bool = False,
                                   use_incremental: bool = True) -> Dict[str, bool]:
        """
        Download the remaining 9 secondary datasets
        
        Returns:
            Dict[str, bool]: Results for each dataset download
        """
        secondary_datasets = [
            "complaints_received",
            "maintenance_code_violations", 
            "approved_permits",
            "dob_violations",
            "permit_issuance",
            "dob_job_application_filings",
            "order_to_repair_vacate",
            "facades_compliance_filings",
            "registration_contacts"
        ]
        
        results = {}
        self.logger.info(f"Starting download of {len(secondary_datasets)} secondary datasets")
        
        for dataset_key in secondary_datasets:
            self.logger.info(f"Processing dataset: {dataset_key}")
            results[dataset_key] = self.download_dataset(
                dataset_key, 
                format_type, 
                force_download,
                use_incremental
            )
            
            # Brief pause between downloads
            time.sleep(2)
        
        # Log summary
        successful = sum(1 for success in results.values() if success)
        self.logger.info(f"Secondary dataset downloads completed: {successful}/{len(secondary_datasets)} successful")
        
        return results
    
    def download_all_datasets(self, format_type: str = "csv", 
                             force_download: bool = False,
                             use_incremental: bool = True) -> Dict[str, bool]:
        """
        Download all 18 target datasets
        
        Returns:
            Dict[str, bool]: Results for each dataset download
        """
        self.logger.info("Starting download of all 18 target datasets")
        
        # Download primary datasets first
        primary_results = self.download_primary_datasets(format_type, force_download, use_incremental)
        
        # Download secondary datasets
        secondary_results = self.download_secondary_datasets(format_type, force_download, use_incremental)
        
        # Combine results
        all_results = {**primary_results, **secondary_results}
        
        # Log final summary
        successful = sum(1 for success in all_results.values() if success)
        total = len(all_results)
        self.logger.info(f"All dataset downloads completed: {successful}/{total} successful")
        
        # Log any failures
        failures = [key for key, success in all_results.items() if not success]
        if failures:
            self.logger.warning(f"Failed downloads: {', '.join(failures)}")
        
        return all_results
    
    def check_dataset_updates(self, dataset_key: str) -> bool:
        """
        Check if a dataset has been updated since last download
        
        Args:
            dataset_key: Key for the dataset to check
            
        Returns:
            bool: True if update is available, False otherwise
        """
        if dataset_key not in DataAcquisitionConfig.DATASETS:
            self.logger.error(f"Unknown dataset: {dataset_key}")
            return False
        
        dataset_config = DataAcquisitionConfig.DATASETS[dataset_key]
        dataset_id = dataset_config["id"]
        
        try:
            # Get dataset metadata URL to check last modified
            metadata_url = f"{DataAcquisitionConfig.BASE_URL}/api/views/{dataset_id}.json"
            
            response = self.http_client.get(metadata_url)
            metadata = response.json()
            
            # Get last update time from API
            remote_updated = metadata.get('rowsUpdatedAt', metadata.get('dataUpdatedAt'))
            if not remote_updated:
                self.logger.warning(f"Could not determine update time for {dataset_key}")
                return True  # Assume update available if uncertain
            
            # Get local metadata
            local_metadata = self.metadata_manager.get_dataset_metadata(dataset_key)
            local_updated = local_metadata.get('last_download_timestamp')
            
            if not local_updated:
                self.logger.info(f"No local download recorded for {dataset_key}")
                return True  # Need initial download
            
            # Compare timestamps
            remote_dt = datetime.fromisoformat(remote_updated.replace('Z', '+00:00'))
            local_dt = datetime.fromisoformat(local_updated)
            
            if remote_dt > local_dt:
                self.logger.info(f"Update available for {dataset_key}")
                return True
            else:
                self.logger.info(f"No update needed for {dataset_key}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking updates for {dataset_key}: {e}")
            return True  # Assume update available if check fails
    
    def get_download_summary(self) -> Dict[str, Any]:
        """
        Get summary of all downloaded datasets with incremental info
        
        Returns:
            Dict with summary information
        """
        summary = {
            "total_datasets": len(DataAcquisitionConfig.DATASETS),
            "downloaded_datasets": 0,
            "total_size_bytes": 0,
            "incremental_supported": 0,
            "incremental_enabled": 0,
            "datasets": {}
        }
        
        for dataset_key, dataset_config in DataAcquisitionConfig.DATASETS.items():
            metadata = self.metadata_manager.get_dataset_metadata(dataset_key)
            incremental_metadata = self.metadata_manager.get_incremental_metadata(dataset_key)
            
            if dataset_config.get('supports_incremental', False):
                summary["incremental_supported"] += 1
            
            if metadata.get('file_path'):
                summary["downloaded_datasets"] += 1
                file_size = metadata.get('file_size', 0)
                summary["total_size_bytes"] += file_size
                
                download_type = metadata.get('download_type', 'unknown')
                if 'incremental' in download_type:
                    summary["incremental_enabled"] += 1
                
                summary["datasets"][dataset_key] = {
                    "downloaded": True,
                    "file_size": file_size,
                    "last_download": metadata.get('last_download_timestamp'),
                    "checksum": metadata.get('checksum_md5'),
                    "download_type": download_type,
                    "supports_incremental": dataset_config.get('supports_incremental', False),
                    "incremental_downloads": incremental_metadata.get('last_incremental_download') is not None
                }
            else:
                summary["datasets"][dataset_key] = {
                    "downloaded": False,
                    "supports_incremental": dataset_config.get('supports_incremental', False)
                }
        
        return summary

def create_download_cli():
    """Create command-line interface for downloads with incremental support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NYC DOB Data Acquisition Tool with Incremental Downloads")
    parser.add_argument("--dataset", type=str, help="Specific dataset to download")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "json", "odata"], 
                       help="Download format")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--primary", action="store_true", help="Download primary datasets")
    parser.add_argument("--secondary", action="store_true", help="Download secondary datasets")
    parser.add_argument("--force", action="store_true", help="Force download even if file exists")
    parser.add_argument("--no-incremental", action="store_true", help="Disable incremental downloads")
    parser.add_argument("--check-updates", action="store_true", help="Check for dataset updates")
    parser.add_argument("--summary", action="store_true", help="Show download summary")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser

if __name__ == "__main__":
    # Parse command line arguments
    parser = create_download_cli()
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = NYCDataDownloader(log_level=args.log_level)
    
    # Determine incremental setting
    use_incremental = not args.no_incremental
    
    try:
        if args.summary:
            # Show download summary
            summary = downloader.get_download_summary()
            print(f"\n=== NYC DOB Data Download Summary ===")
            print(f"Total datasets configured: {summary['total_datasets']}")
            print(f"Downloaded datasets: {summary['downloaded_datasets']}")
            print(f"Total size: {summary['total_size_bytes']:,} bytes ({summary['total_size_bytes']/1024/1024:.1f} MB)")
            print(f"Incremental support: {summary['incremental_supported']}/{summary['total_datasets']} datasets")
            print(f"Using incremental: {summary['incremental_enabled']}/{summary['downloaded_datasets']} downloads")
            print("\nDataset Status:")
            for dataset_key, info in summary['datasets'].items():
                if info['downloaded']:
                    status = "✅ Downloaded"
                    download_type = info.get('download_type', 'unknown')
                    incremental_info = " (incremental)" if 'incremental' in download_type else " (full)"
                    supports_inc = " 📈" if info['supports_incremental'] else ""
                    print(f"  {dataset_key}: {status}{incremental_info}{supports_inc}")
                    file_size_mb = info['file_size'] / 1024 / 1024
                    print(f"    Size: {file_size_mb:.1f} MB, Last: {info['last_download'][:10]}")
                else:
                    status = "❌ Not downloaded"
                    supports_inc = " (supports incremental)" if info['supports_incremental'] else ""
                    print(f"  {dataset_key}: {status}{supports_inc}")
        
        elif args.check_updates:
            # Check for updates
            print("\n=== Checking for Dataset Updates ===")
            if args.dataset:
                # Check specific dataset
                has_update = downloader.check_dataset_updates(args.dataset)
                status = "Update available" if has_update else "Up to date"
                print(f"{args.dataset}: {status}")
            else:
                # Check all datasets
                for dataset_key in DataAcquisitionConfig.DATASETS.keys():
                    has_update = downloader.check_dataset_updates(dataset_key)
                    status = "📄 Update available" if has_update else "✅ Up to date"
                    print(f"  {dataset_key}: {status}")
        
        elif args.dataset:
            # Download specific dataset
            inc_status = "with incremental" if use_incremental else "full download"
            print(f"\n=== Downloading Dataset: {args.dataset} ({inc_status}) ===")
            success = downloader.download_dataset(
                args.dataset, 
                args.format, 
                args.force,
                use_incremental
            )
            if success:
                print(f"✅ Successfully downloaded {args.dataset}")
            else:
                print(f"❌ Failed to download {args.dataset}")
                exit(1)
        
        elif args.all:
            # Download all datasets
            inc_status = "with incremental" if use_incremental else "full downloads"
            print(f"\n=== Downloading All 18 Datasets ({inc_status}, Format: {args.format}) ===")
            results = downloader.download_all_datasets(args.format, args.force, use_incremental)
            successful = sum(1 for success in results.values() if success)
            print(f"\n✅ Download Summary: {successful}/{len(results)} datasets successful")
            
            # Show failures if any
            failures = [key for key, success in results.items() if not success]
            if failures:
                print(f"❌ Failed: {', '.join(failures)}")
        
        elif args.primary:
            # Download primary datasets
            inc_status = "with incremental" if use_incremental else "full downloads"
            print(f"\n=== Downloading Primary Datasets ({inc_status}, Format: {args.format}) ===")
            results = downloader.download_primary_datasets(args.format, args.force, use_incremental)
            successful = sum(1 for success in results.values() if success)
            print(f"\n✅ Primary Download Summary: {successful}/{len(results)} datasets successful")
        
        elif args.secondary:
            # Download secondary datasets
            inc_status = "with incremental" if use_incremental else "full downloads"
            print(f"\n=== Downloading Secondary Datasets ({inc_status}, Format: {args.format}) ===")
            results = downloader.download_secondary_datasets(args.format, args.force, use_incremental)
            successful = sum(1 for success in results.values() if success)
            print(f"\n✅ Secondary Download Summary: {successful}/{len(results)} datasets successful")
        
        else:
            # Default behavior - show help and initialize
            parser.print_help()
            print(f"\n=== NYC DOB Data Acquisition Module with Incremental Downloads ===")
            print("Module initialized successfully")
            print(f"Configured datasets: {len(DataAcquisitionConfig.DATASETS)}")
            
            # Show incremental support info
            summary = downloader.get_download_summary()
            print(f"Incremental download support: {summary['incremental_supported']}/{summary['total_datasets']} datasets")
            print("\nNew Features:")
            print("  📈 Smart incremental downloads (only new/updated records)")
            print("  🔄 Resumable downloads for large files")
            print("  📅 Date-range filtering using OData queries")
            print("  🔗 Automatic data merging for incremental updates")
            print("  📊 Enhanced progress tracking and metadata")
            print("\nUse --help to see available options")
            print("Example: python data_acquisition.py --primary --format csv")
            print("Example: python data_acquisition.py --dataset housing_litigations --no-incremental")
            
    except KeyboardInterrupt:
        downloader.logger.info("Download interrupted by user")
        print("\n❌ Download interrupted")
    except Exception as e:
        downloader.logger.error(f"Unexpected error: {e}")
        print(f"❌ Error: {e}")
        exit(1) 