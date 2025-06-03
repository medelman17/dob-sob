# Graphiti Custom Entity Types for NYC DOB Fraud Detection

## Overview

This document defines the custom entity types for the dob-sob NYC DOB fraud detection platform using Zep's Graphiti knowledge graph framework. These entity types are designed specifically for **relationship discovery** rather than pre-computed fraud detection results.

## Design Philosophy

### Core Principles
1. **Raw Data First**: Entity types model actual data from NYC DOB datasets, not derived analytical results
2. **Relationship-Centric**: Every entity is designed to maximize connection points with other entities
3. **Natural Discovery**: Let Graphiti discover fraud patterns through entity relationships rather than pre-defining them
4. **Dot Connecting**: Entities serve as "dots" that can be connected through various relationship paths

### Anti-Patterns Avoided
- ❌ Derived analytical entities (FraudPattern, ShellCompanyNetwork)
- ❌ Pre-computed clusters (AddressCluster, TimelineCluster) 
- ❌ Statistical aggregations (NetworkBridge, RiskScore)
- ❌ Inferred relationships (computed connections)

## Custom Entity Type Definitions

### Core Hub Entities

#### Property
Central connecting entity - all other entities connect through properties.

```python
class Property(BaseModel):
    """Central hub entity - everything connects through properties"""
    bin: str = Field(..., description="Building Identification Number - primary connector")
    bbl: str = Field(..., description="Borough-Block-Lot - alternative connector") 
    
    # Multiple address formats for fuzzy matching
    full_address: str = Field(..., description="Complete standardized address")
    house_number: str = Field(..., description="House number")
    street_name: str = Field(..., description="Street name")
    
    # Geographic clustering
    borough: str = Field(..., description="Borough identifier")
    block: str = Field(..., description="Tax block")
    lot: str = Field(..., description="Tax lot")
    zip_code: Optional[str] = Field(None, description="ZIP code")
    
    # Geospatial for proximity analysis
    latitude: Optional[float] = Field(None, description="Latitude")
    longitude: Optional[float] = Field(None, description="Longitude")
```

**Key Relationships**: Connected to all event types via BIN/BBL, enables property portfolio analysis

#### Person  
Represents individuals in various roles (owners, architects, contractors).

```python
class Person(BaseModel):
    """People entity - connects owners, architects, contractors"""
    name_raw: str = Field(..., description="Original name as appears in data")
    name_normalized: str = Field(..., description="Cleaned name for matching")
    
    # Role tracking from raw data
    roles: List[str] = Field(default_factory=list, description="owner, architect, contractor, etc.")
    
    # Business connections from raw data
    business_names: List[str] = Field(default_factory=list, description="Associated business entities")
    license_numbers: List[str] = Field(default_factory=list, description="Professional licenses")
```

**Key Relationships**: Links to Organization via business names, to events via respondent/owner fields

#### Organization
Business entities including LLCs, corporations, management companies.

```python
class Organization(BaseModel):
    """Business entity - LLCs, corporations, management companies"""
    business_name_raw: str = Field(..., description="Original business name")
    business_name_normalized: str = Field(..., description="Cleaned business name")
    
    # Entity type from raw data
    entity_type: str = Field(..., description="LLC, CORP, PARTNERSHIP, etc.")
    
    # Address from raw data
    business_addresses: List[str] = Field(default_factory=list, description="Business addresses")
```

**Key Relationships**: Connected to Person entities, enables shell company network discovery

#### Inspector
Inspector entities from device numbers in violation records.

```python
class Inspector(BaseModel):
    """Inspector entity from device numbers in violation data"""
    device_number: str = Field(..., description="Inspector device/ID number")
    
    # Raw activity data
    inspection_dates: List[str] = Field(default_factory=list, description="Inspection timeline")
    active_areas: List[str] = Field(default_factory=list, description="Areas of activity")
```

**Key Relationships**: Links to ViolationEvent, enables corruption pattern analysis

### Event Entities

#### ViolationEvent
Individual violations connecting properties, respondents, and inspectors.

```python
class ViolationEvent(BaseModel):
    """Violation as a connecting event between property, person, inspector"""
    violation_id: str = Field(..., description="Unique violation identifier")
    
    # Primary connections from raw data
    property_bin: str = Field(..., description="Property where violation occurred")
    respondent_name: Optional[str] = Field(None, description="Person/entity responsible")
    inspector_device: Optional[str] = Field(None, description="Issuing inspector")
    
    # Raw violation data
    violation_code: str = Field(..., description="Violation type code")
    violation_class: str = Field(..., description="Severity class")
    issue_date: str = Field(..., description="When violation was issued")
    disposition_date: Optional[str] = Field(None, description="Resolution date")
    disposition_type: str = Field(..., description="How resolved")
```

**Key Relationships**: Property→Person→Inspector triangle, enables corruption and pattern analysis

#### LitigationEvent
Legal cases from housing litigation dataset.

```python
class LitigationEvent(BaseModel):
    """Legal case from housing litigation data"""
    case_id: str = Field(..., description="Case identifier")
    
    # Connections from raw data
    property_bin: Optional[str] = Field(None, description="Property involved")
    property_address: str = Field(..., description="Property address")
    respondent_name: str = Field(..., description="Defendant")
    
    # Raw case data
    case_type: str = Field(..., description="Type of legal action")
    case_status: str = Field(..., description="Current status")
    open_date: Optional[str] = Field(None, description="Case opened")
    close_date: Optional[str] = Field(None, description="Case closed")
    
    # Raw outcomes
    harassment_finding: Optional[str] = Field(None, description="Harassment finding")
    penalty_amount: Optional[str] = Field(None, description="Financial penalty")
    open_judgement: str = Field(..., description="YES/NO from raw data")
```

**Key Relationships**: Property→Person escalation patterns, enforcement effectiveness analysis

#### PermitEvent
Construction permits and jobs from active projects dataset.

```python
class PermitEvent(BaseModel):
    """Permit/construction from active projects data"""
    job_number: str = Field(..., description="DOB job number")
    
    # Connections from raw data
    property_bin: Optional[str] = Field(None, description="Property BIN")
    owner_name: Optional[str] = Field(None, description="Property owner")
    owner_business_name: Optional[str] = Field(None, description="Owner business")
    architect_first_name: Optional[str] = Field(None, description="Architect first name")
    architect_last_name: Optional[str] = Field(None, description="Architect last name")
    architect_license: Optional[str] = Field(None, description="Architect license")
    
    # Raw permit data
    work_type: str = Field(..., description="Type of work")
    permit_status: str = Field(..., description="Current status")
    
    # Raw timeline data
    pre_filing_date: Optional[str] = Field(None, description="Pre-filing date")
    paid_date: Optional[str] = Field(None, description="Payment date")
    approved_date: Optional[str] = Field(None, description="Approval date")
    latest_action_date: Optional[str] = Field(None, description="Latest action")
```

**Key Relationships**: Property→Person→Professional networks, timeline analysis capabilities

#### ComplaintEvent
Citizen complaints from complaints received dataset.

```python
class ComplaintEvent(BaseModel):
    """Complaint from complaints received data"""
    complaint_id: str = Field(..., description="Complaint number")
    
    # Connections from raw data
    property_bin: Optional[str] = Field(None, description="Property complained about")
    house_number: Optional[str] = Field(None, description="House number")
    house_street: Optional[str] = Field(None, description="Street name")
    
    # Raw complaint data
    complaint_category: str = Field(..., description="Type of complaint")
    date_entered: str = Field(..., description="When complaint made")
    disposition_date: Optional[str] = Field(None, description="When resolved")
    disposition_code: Optional[str] = Field(None, description="Resolution code")
    inspection_date: Optional[str] = Field(None, description="Inspection date")
```

**Key Relationships**: Property-centric complaint patterns, response time analysis

#### MaintenanceViolationEvent
HPD maintenance code violations.

```python
class MaintenanceViolationEvent(BaseModel):
    """HPD maintenance violation from maintenance code violations data"""
    violation_id: str = Field(..., description="Violation identifier")
    
    # Connections from raw data
    property_bin: Optional[str] = Field(None, description="Building BIN")
    building_id: str = Field(..., description="HPD building ID")
    
    # Raw violation data
    class_violation: str = Field(..., description="Violation class A/B/C")
    inspection_date: str = Field(..., description="Inspection date")
    approved_date: Optional[str] = Field(None, description="Approval date")
    current_status_id: str = Field(..., description="Current status")
    nov_description: str = Field(..., description="Violation description")
    nov_issued_date: str = Field(..., description="NOV issued date")
```

**Key Relationships**: Parallel violation tracking system, cross-agency pattern analysis

#### CertificateEvent
Certificates of Occupancy and related approvals.

```python
class CertificateEvent(BaseModel):
    """Certificate of Occupancy from CO data"""
    co_number: str = Field(..., description="Certificate number")
    
    # Connections from raw data
    property_bin: Optional[str] = Field(None, description="Building BIN")
    job_number: Optional[str] = Field(None, description="Associated job number")
    
    # Raw certificate data
    co_issue_date: str = Field(..., description="Date CO issued")
    co_type: str = Field(..., description="Type of certificate")
    occupancy_type: str = Field(..., description="Permitted occupancy")
    co_latest_status: str = Field(..., description="Latest status")
    co_latest_status_date: str = Field(..., description="Status date")
```

**Key Relationships**: PermitEvent→CertificateEvent workflows, completion pattern analysis

## Implementation

### Entity Type Registration

```python
# Complete entity type mapping for Graphiti
raw_data_entities = {
    "Property": Property,
    "Person": Person, 
    "Organization": Organization,
    "Inspector": Inspector,
    "ViolationEvent": ViolationEvent,
    "LitigationEvent": LitigationEvent,
    "PermitEvent": PermitEvent,
    "ComplaintEvent": ComplaintEvent,
    "MaintenanceViolationEvent": MaintenanceViolationEvent,
    "CertificateEvent": CertificateEvent
}

# Usage with data pipeline
await client.add_episode(
    name='NYC DOB Raw Data Ingestion',
    episode_body=processed_data,
    entity_types=raw_data_entities
)
```

### Data Processing Pipeline

1. **Extract**: Load raw CSV data from downloaded NYC DOB datasets
2. **Transform**: Clean and normalize names, addresses, identifiers
3. **Load**: Create entity instances and add to Graphiti knowledge graph
4. **Link**: Let Graphiti discover relationships through entity connections

## Fraud Detection Patterns Enabled

### Shell Company Networks
- **Query**: Find Organizations sharing addresses with multiple Properties
- **Mechanism**: Organization.business_addresses → Property address matching
- **Discovery**: Person→Organization→Property ownership webs

### Inspector Corruption  
- **Query**: Find Inspectors with unusual violation dismissal patterns at Properties owned by same Person/Organization
- **Mechanism**: Inspector→ViolationEvent→Property→Person relationship analysis
- **Discovery**: Systematic bias in enforcement

### Permit Acceleration
- **Query**: Find Properties where PermitEvents process unusually fast after ViolationEvents
- **Mechanism**: Temporal relationship analysis between event types
- **Discovery**: Influence peddling or corruption in permit processing

### Slumlord Identification
- **Query**: Find Person/Organization entities connected to multiple Properties with high ViolationEvent density
- **Mechanism**: Person→Property portfolio analysis with violation aggregation
- **Discovery**: Systematic neglect patterns

### Cross-Agency Coordination
- **Query**: Find Properties with both DOB ViolationEvents and HPD MaintenanceViolationEvents showing coordination gaps
- **Mechanism**: Cross-dataset event correlation at Property level
- **Discovery**: Enforcement coordination failures

## Benefits of This Approach

1. **Natural Discovery**: Relationships emerge organically from data connections
2. **Flexible Queries**: Can discover new fraud patterns not pre-anticipated
3. **Evidence Trails**: Clear audit trails from raw data to discovered patterns
4. **Temporal Analysis**: Built-in timeline capabilities for pattern evolution
5. **Multi-Modal Relationships**: Name, address, geographic, and temporal connection types
6. **Scalable**: Works with full 25GB dataset without pre-computation overhead

## Usage Examples

### Basic Relationship Queries
```python
# Find all properties owned by a specific person
await client.search_nodes(
    query="Properties owned by VED PARKASH",
    entity_types=["Property", "Person", "ViolationEvent"]
)

# Find inspector patterns at properties with multiple violations
await client.search_nodes(
    query="Inspector device numbers at properties with repeated violations",
    entity_types=["Inspector", "ViolationEvent", "Property"]
)
```

### Advanced Pattern Discovery
```python
# Complex fraud pattern discovery
await client.search_nodes(
    query="Properties where complaints led to violations then quick permit approvals",
    entity_types=["Property", "ComplaintEvent", "ViolationEvent", "PermitEvent"]
)
```

This entity type design provides the foundation for sophisticated fraud detection through natural relationship discovery in the Graphiti knowledge graph.

## Dataset Scale

**Current NYC DOB Dataset Coverage:**
- **Total Raw Data**: 25GB across 94 dataset types
- **Total Files**: 119 CSV files (including multiple downloads/versions)
- **Major Datasets by Size**:
  - Maintenance Code Violations: 4.9GB
  - ECB Violations: 2.8GB
  - DOB Violations: 2.0GB
  - Job Application Filings: 1.5GB
  - Historical Permit Issuance: 1.5GB
  - DOB Job Application Filings: 1.4GB
  - Historical DOB Permit Issuance: 1.2GB
  - Complaints Received: 963MB
  - Plus 86 additional specialized datasets

This represents **comprehensive coverage** of NYC's entire construction, housing, and building enforcement ecosystem - providing an unprecedented 25GB data foundation for sophisticated fraud pattern detection through Graphiti's relationship discovery capabilities.