"""
Pydantic data models for dob-sob platform.

Provides strongly-typed data models for API responses, database entities,
and internal data structures with validation.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from pathlib import Path


class Priority(str, Enum):
    """Priority levels for various operations."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"


class DatasetStatus(str, Enum):
    """Status of dataset processing."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskLevel(str, Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlgorithmType(str, Enum):
    """Community detection algorithm types."""
    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "label_propagation"
    EDGE_BETWEENNESS = "edge_betweenness"
    CLIQUE_PERCOLATION = "clique_percolation"
    SPECTRAL_CLUSTERING = "spectral_clustering"
    INFOMAP = "infomap"


class DatasetMetadata(BaseModel):
    """Metadata for NYC DOB datasets."""
    
    id: str = Field(..., description="NYC Open Data dataset identifier")
    name: str = Field(..., description="Human-readable dataset name")
    description: str = Field(..., description="Dataset description")
    update_frequency: str = Field(..., description="Update frequency")
    estimated_size_mb: int = Field(..., description="Estimated size in MB")
    date_field: Optional[str] = Field(None, description="Date field for incremental updates")
    supports_incremental: bool = Field(default=True)
    priority: Priority = Field(default=Priority.MEDIUM)
    
    # Processing metadata
    last_downloaded: Optional[datetime] = None
    last_processed: Optional[datetime] = None
    download_count: int = Field(default=0)
    status: DatasetStatus = Field(default=DatasetStatus.PENDING)
    
    class Config:
        use_enum_values = True


class DownloadResult(BaseModel):
    """Result of a dataset download operation."""
    
    dataset_name: str
    dataset_id: str
    success: bool
    file_path: Optional[Path] = None
    file_size_bytes: Optional[int] = None
    download_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    checksum: Optional[str] = None
    record_count: Optional[int] = None
    
    # Incremental download info
    is_incremental: bool = Field(default=False)
    incremental_filter: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class BuildingInfo(BaseModel):
    """Building information for fraud analysis."""
    
    bin: str = Field(..., description="Building Identification Number")
    address: Optional[str] = None
    borough: Optional[str] = None
    block: Optional[str] = None
    lot: Optional[str] = None
    
    # Risk assessment
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Counts for quick assessment
    violation_count: int = Field(default=0)
    complaint_count: int = Field(default=0)
    permit_count: int = Field(default=0)
    
    # Temporal data
    first_permit_date: Optional[datetime] = None
    last_activity_date: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class ContractorInfo(BaseModel):
    """Contractor information for fraud analysis."""
    
    business_name: str
    license_number: Optional[str] = None
    license_type: Optional[str] = None
    
    # Risk assessment
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Activity metrics
    total_permits: int = Field(default=0)
    active_permits: int = Field(default=0)
    violation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Network analysis
    associated_buildings: List[str] = Field(default_factory=list)
    community_id: Optional[str] = None
    
    class Config:
        use_enum_values = True


class CommunityDetectionResult(BaseModel):
    """Result of community detection analysis."""
    
    algorithm: AlgorithmType
    dataset_name: str
    execution_time_seconds: float
    
    # Community statistics
    total_communities: int
    largest_community_size: int
    modularity_score: Optional[float] = None
    
    # Communities data
    communities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map of community_id to list of entity_ids"
    )
    
    # Fraud patterns detected
    suspicious_communities: List[str] = Field(
        default_factory=list,
        description="Community IDs flagged as suspicious"
    )
    
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class FraudPattern(BaseModel):
    """Identified fraud pattern."""
    
    pattern_id: str
    pattern_type: str = Field(..., description="Type of fraud pattern")
    description: str
    
    # Risk assessment
    risk_level: RiskLevel
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    
    # Entities involved
    buildings: List[str] = Field(default_factory=list)
    contractors: List[str] = Field(default_factory=list)
    permits: List[str] = Field(default_factory=list)
    
    # Evidence
    evidence: Dict[str, Any] = Field(default_factory=dict)
    
    # Detection metadata
    algorithm_used: AlgorithmType
    detected_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool
    data: Optional[Union[Dict[str, Any], List[Any]]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    
    # Pagination info
    page: Optional[int] = None
    per_page: Optional[int] = None
    total: Optional[int] = None
    
    # Request metadata
    request_id: Optional[str] = None
    execution_time_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ProcessingJob(BaseModel):
    """Background processing job information."""
    
    job_id: str
    job_type: str = Field(..., description="Type of processing job")
    status: DatasetStatus
    
    # Progress tracking
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True


class DataQualityReport(BaseModel):
    """Data quality assessment report."""
    
    dataset_name: str
    assessed_at: datetime = Field(default_factory=datetime.now)
    
    # Basic statistics
    total_records: int
    null_percentage: float = Field(ge=0.0, le=100.0)
    duplicate_percentage: float = Field(ge=0.0, le=100.0)
    
    # Data quality scores
    completeness_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    validity_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    
    # Issues found
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Field-level statistics
    field_statistics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)