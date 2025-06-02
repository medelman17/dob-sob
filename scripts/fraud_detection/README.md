# NYC DOB Fraud Detection System

A Graphiti-powered pattern recognition system for detecting complex fraud schemes in NYC construction data, with focus on corporate network analysis and temporal relationship tracking.

## 🎯 Project Overview

This system leverages **Graphiti's temporal knowledge graphs** to automatically discover hidden relationships and patterns in NYC Department of Buildings data that may indicate fraudulent activities, particularly in minority/women-owned business enterprise (MWBE) programs.

### Key Capabilities

- **Automatic Entity Resolution**: Graphiti's LLM-powered extraction identifies businesses, individuals, and projects without manual relationship definition
- **Temporal Network Analysis**: Track changes in corporate structures over time to detect shell company patterns
- **Community Detection**: Find hidden connections between seemingly unrelated entities
- **Pattern Recognition**: Identify suspicious corporate networks and financial flows

## 📊 Data Analysis Results

Our initial analysis of NYC DOB data reveals **excellent patterns for fraud detection**:

### Housing Litigation Data (84.9 MB, 1000+ records analyzed)

- **49% multi-entity respondents** - Complex corporate structures involving multiple parties
- **40.5% LLC entities** - High percentage of limited liability companies
- **16.2% complex names (>50 chars)** - Potential shell company indicators

### Sample Pattern Identified

```
"DAVID BILDIRICI,EAST 14TH DELAWARE REALTY, LLC,YUSUF Y. BILDIRICI"
```

This shows exactly what we're looking for: multiple individuals and a company linked together in a single litigation case.

### Complaints Data (316.8 MB)

- Comprehensive building-level complaint tracking
- Links to specific addresses and Building Identification Numbers (BIN)
- Temporal patterns of complaint filing and resolution

## 🏗️ System Architecture

### Core Components

1. **Episode Design Module** (`episode_design.py`)

   - Custom entity types for NYC DOB domain
   - Data transformation pipelines for Graphiti ingestion
   - Polars-based processing for large datasets

2. **Pattern Recognition Queries** (`pattern_queries.py`) - _Coming Next_

   - Fraud detection algorithms
   - Suspicious network identification
   - Scoring and ranking systems

3. **Community Analysis** (`community_analysis.py`) - _Coming Next_

   - Corporate network clustering
   - Hidden relationship discovery
   - Central entity identification

4. **Temporal Analysis** (`temporal_analysis.py`) - _Coming Next_

   - Time-aware pattern detection
   - Corporate succession tracking
   - Shell company lifecycle analysis

5. **Interactive Explorer** (`explorer.py`) - _Coming Next_
   - Graph visualization interface
   - Investigation workflows
   - Report generation

## 🏢 Entity Types

### Building

- Building Identification Number (BIN)
- Borough, Block, Lot (BBL) identifier
- Address and geographic coordinates
- Community district and census tract

### Corporation

- Legal business name and entity type
- MBE/WBE/DBE certification status
- DOB license numbers
- Business addresses and status

### Person

- Individual names and roles
- Professional licenses
- Address associations
- Corporate relationships

### Permit

- Permit numbers and types
- Work descriptions and status
- Filing and issuance dates
- Cost estimates

### Violation

- Violation numbers and categories
- Severity levels and penalties
- Issue and disposition dates
- Resolution status

### Project

- Construction project details
- Work descriptions and timelines
- Cost estimates and completion dates
- Multi-permit coordination

### FinancialFlow

- Transaction amounts and types
- Markup percentages
- Payment flows between entities
- Temporal transaction patterns

## 🔍 Fraud Detection Patterns

### Pass-Through Arrangements

- DBE firms with minimal markup (2-3%) acting as intermediaries
- Single-customer DBE relationships
- Disproportionate profit distributions

### Control Fraud

- Non-minority individuals controlling DBE firms
- Shared officers across multiple companies
- Common business addresses

### Shell Company Networks

- Complex ownership structures designed to obscure control
- Rapid company formation and dissolution patterns
- Suspicious timing of contract awards relative to company changes

### Operational Incapability

- DBEs lacking necessary resources for contracted work
- Subcontracting patterns that bypass DBE requirements
- Geographic mismatches between company location and work sites

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Neo4j database (already running via Task 2)
- uv for dependency management
- Graphiti for knowledge graph management

### Installation

```bash
# Ensure all dependencies are synced using uv
uv sync

# Test the episode design module
cd scripts/fraud_detection
uv run python simple_test.py
```

### Data Processing

```python
from scripts.fraud_detection.episode_design import EpisodeDesigner

# Initialize the designer
designer = EpisodeDesigner()

# Process housing litigation data
episodes = designer.prepare_housing_litigation_episodes(
    "data/raw/housing_litigations/housing_litigations_20250602_130230.csv"
)

# Process complaints data
complaint_episodes = designer.prepare_complaint_episodes(
    "data/raw/complaints_received/complaints_received_20250602_130901.csv"
)
```

## 📈 Next Steps

1. **Install Graphiti** and integrate with existing Neo4j instance
2. **Implement Pattern Recognition Queries** for fraud detection
3. **Build Community Analysis** capabilities for network discovery
4. **Create Temporal Analysis** tools for time-based pattern detection
5. **Develop Interactive Explorer** for investigation workflows

## 🎯 Expected Outcomes

With the rich patterns already identified in our data:

- **Automatic discovery** of corporate networks without manual relationship definition
- **Temporal tracking** of shell company lifecycles and succession schemes
- **Community detection** to find hidden connections between entities
- **Scoring algorithms** to rank suspicious patterns by fraud likelihood
- **Investigation tools** to explore and document findings

## 📝 Data Sources

Currently processing 3 of 18 available NYC DOB datasets:

- Housing Litigations (84.9 MB)
- Complaints Received (316.8 MB)
- Additional datasets available for expansion

The system is designed to scale to all 18 NYC DOB datasets for comprehensive fraud detection coverage.

---

_This system represents a significant advancement in automated fraud detection for construction industry oversight, leveraging cutting-edge knowledge graph technology to uncover complex schemes that would be difficult to detect through traditional analysis methods._
