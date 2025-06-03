"""
Centralized configuration management for dob-sob platform.

Provides environment-based configuration with validation using Pydantic.
Supports loading from environment variables, YAML files, and defaults.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import yaml


class DatasetConfig(BaseSettings):
    """Configuration for individual NYC DOB datasets."""
    
    id: str = Field(..., description="NYC Open Data dataset identifier")
    name: str = Field(..., description="Human-readable dataset name")
    description: str = Field(..., description="Dataset description")
    update_frequency: str = Field(default="daily", description="How often dataset is updated")
    estimated_size_mb: int = Field(..., description="Estimated dataset size in MB")
    date_field: Optional[str] = Field(None, description="Field to use for incremental updates")
    supports_incremental: bool = Field(default=True, description="Whether dataset supports incremental updates")
    priority: str = Field(default="medium", description="Dataset processing priority")
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ['low', 'medium', 'high']
        if v not in valid_priorities:
            raise ValueError(f'Priority must be one of {valid_priorities}')
        return v
    
    @validator('update_frequency')
    def validate_update_frequency(cls, v):
        valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly']
        if v not in valid_frequencies:
            raise ValueError(f'Update frequency must be one of {valid_frequencies}')
        return v


class DatabaseConfig(BaseSettings):
    """Neo4j database configuration."""
    
    uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    user: str = Field(default="neo4j", env="NEO4J_USER")
    password: str = Field(default="password", env="NEO4J_PASSWORD")
    database: str = Field(default="neo4j", env="NEO4J_DATABASE")
    
    # Connection settings
    max_connection_lifetime: int = Field(default=3600)  # 1 hour
    max_connection_pool_size: int = Field(default=50)
    connection_acquisition_timeout: int = Field(default=60)


class APIConfig(BaseSettings):
    """NYC Open Data API configuration."""
    
    base_url: str = Field(default="https://data.cityofnewyork.us")
    api_base_url: str = Field(default="https://data.cityofnewyork.us/resource")
    odata_base_url: str = Field(default="https://data.cityofnewyork.us/OData.svc")
    
    # Rate limiting and timeouts
    requests_per_second: int = Field(default=10, description="API rate limit")
    timeout_seconds: int = Field(default=30, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    backoff_factor: float = Field(default=1.0, description="Backoff factor for retries")
    
    # Concurrent processing
    max_concurrent_downloads: int = Field(default=8, env="MAX_CONCURRENT_DOWNLOADS")
    chunk_size: int = Field(default=8192, description="Download chunk size in bytes")


class FraudDetectionConfig(BaseSettings):
    """Fraud detection algorithm configuration."""
    
    # Community detection parameters
    min_community_size: int = Field(default=3, description="Minimum community size for detection")
    max_communities: int = Field(default=100, description="Maximum communities to process")
    
    # Risk scoring thresholds
    high_risk_threshold: float = Field(default=0.8, description="High risk score threshold")
    medium_risk_threshold: float = Field(default=0.5, description="Medium risk score threshold")
    
    # Algorithm selection
    default_algorithms: List[str] = Field(
        default=["louvain", "label_propagation", "edge_betweenness"],
        description="Default community detection algorithms to run"
    )
    
    @validator('default_algorithms')
    def validate_algorithms(cls, v):
        valid_algorithms = [
            "louvain", "label_propagation", "edge_betweenness", 
            "clique_percolation", "spectral_clustering", "infomap"
        ]
        for algo in v:
            if algo not in valid_algorithms:
                raise ValueError(f'Algorithm {algo} not in valid algorithms: {valid_algorithms}')
        return v


class WebConfig(BaseSettings):
    """Web interface configuration."""
    
    # Streamlit settings
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    streamlit_host: str = Field(default="0.0.0.0", env="STREAMLIT_HOST")
    
    # Jupyter settings  
    jupyter_port: int = Field(default=8889, env="JUPYTER_PORT")
    jupyter_host: str = Field(default="0.0.0.0", env="JUPYTER_HOST")
    
    # Future API settings
    api_port: int = Field(default=8000, env="API_PORT")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    # File logging
    enable_file_logging: bool = Field(default=True)
    log_file: str = Field(default="logs/dob_sob.log")
    max_file_size_mb: int = Field(default=10)
    backup_count: int = Field(default=5)
    
    @validator('level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()


class Settings(BaseSettings):
    """Main configuration class for dob-sob platform."""
    
    # Environment
    environment: str = Field(default="development", env="DOB_SOB_ENV")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    fraud_detection: FraudDetectionConfig = Field(default_factory=FraudDetectionConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Data storage paths
    data_dir: Path = Field(default="./data", env="DATA_DIR")
    raw_data_dir: Path = Field(default="./data/raw")
    metadata_dir: Path = Field(default="./data/metadata") 
    incremental_dir: Path = Field(default="./data/incremental")
    reports_dir: Path = Field(default="./data/reports")
    
    # Dataset configurations (loaded from YAML)
    datasets: Dict[str, DatasetConfig] = Field(default_factory=dict)
    
    class Config:
        env_prefix = "DOB_SOB_"
        case_sensitive = False
        env_nested_delimiter = "__"
        
    @validator('data_dir', 'raw_data_dir', 'metadata_dir', 'incremental_dir', 'reports_dir')
    def ensure_path_objects(cls, v):
        return Path(v) if isinstance(v, str) else v
    
    def load_datasets_config(self, config_path: Optional[Path] = None) -> None:
        """Load dataset configurations from YAML file."""
        if config_path is None:
            config_path = Path("config/datasets.yaml")
            
        if config_path.exists():
            with open(config_path, 'r') as f:
                datasets_data = yaml.safe_load(f)
                
            for name, config_data in datasets_data.get('datasets', {}).items():
                self.datasets[name] = DatasetConfig(**config_data)
    
    def get_dataset_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """Get configuration for a specific dataset."""
        return self.datasets.get(dataset_name)
    
    def list_datasets_by_priority(self, priority: str) -> List[str]:
        """List datasets filtered by priority level."""
        return [
            name for name, config in self.datasets.items() 
            if config.priority == priority
        ]
    
    def create_directories(self) -> None:
        """Create necessary data directories."""
        directories = [
            self.data_dir,
            self.raw_data_dir, 
            self.metadata_dir,
            self.incremental_dir,
            self.reports_dir,
        ]
        
        # Add log directory if logging is properly configured
        if hasattr(self.logging, 'log_file') and self.logging.log_file:
            directories.append(Path(self.logging.log_file).parent)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def load_config(
    environment: Optional[str] = None,
    config_file: Optional[Path] = None
) -> Settings:
    """
    Load configuration based on environment.
    
    Args:
        environment: Environment name (development, production, test)
        config_file: Path to specific config file
        
    Returns:
        Configured Settings instance
    """
    if environment is None:
        environment = os.getenv('DOB_SOB_ENV', 'development')
    
    # Load base configuration
    settings = Settings(environment=environment)
    
    # Load environment-specific configuration if available
    if config_file is None:
        config_file = Path(f"config/environments/{environment}.yaml")
    
    if config_file and config_file.exists():
        with open(config_file, 'r') as f:
            env_config = yaml.safe_load(f)
            
        # Update settings with environment-specific values
        for key, value in env_config.items():
            if hasattr(settings, key):
                if key in ['database', 'api', 'fraud_detection', 'web', 'logging']:
                    # For nested config objects, update the existing object
                    existing_config = getattr(settings, key)
                    if hasattr(existing_config, '__dict__'):
                        for sub_key, sub_value in value.items():
                            if hasattr(existing_config, sub_key):
                                setattr(existing_config, sub_key, sub_value)
                else:
                    setattr(settings, key, value)
    
    # Load dataset configurations
    settings.load_datasets_config()
    
    # Don't create directories by default - let commands do it as needed
    # settings.create_directories()
    
    return settings


# Global settings instance - will be initialized when first accessed
settings = None

def get_settings():
    """Get global settings instance, initializing if needed."""
    global settings
    if settings is None:
        settings = load_config()
    return settings