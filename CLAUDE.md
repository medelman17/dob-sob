# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dob-sob** is a NYC DOB fraud detection platform that uses graph analytics, machine learning, and real-time data processing to identify corrupt contractors and permit fraud in NYC's construction industry. The codebase integrates 18 NYC DOB datasets, implements 6 community detection algorithms, and provides real-time fraud assessment capabilities.

## Development Environment

### Python Package Management
- **Use `uv` for all Python dependency management** (not pip/poetry)
- Project uses Python 3.12+ with comprehensive type hints
- Dependencies defined in `pyproject.toml` with locked versions in `uv.lock`

### Key Commands
```bash
# Setup environment
uv venv && source .venv/bin/activate
uv sync

# Install development dependencies
uv sync --dev

# Use the unified CLI
uv run dob-sob --help
uv run dob-sob data fetch --primary
uv run dob-sob data fetch --list
uv run dob-sob fraud detect --all-algorithms

# Legacy test commands (until migration complete)
uv run python scripts/fraud_detection/test_community_detection_algorithms.py
uv run python scripts/etl/test_orchestration_system.py
uv run python scripts/etl/test_odata_client.py

# Code quality (when available)
black dob_sob/ --check
isort dob_sob/ --check-only
mypy dob_sob/
```

### Docker Services
Use `docker compose` (not `docker-compose`) for all container operations:
```bash
# Start all services (Neo4j, Streamlit, Jupyter)
docker compose up -d

# Start specific services
docker compose up neo4j -d

# View logs
docker compose logs -f neo4j

# Stop and cleanup
docker compose down
```

## Architecture

### Core Components
- **Data Layer** (`dob_sob/data/`): Unified data acquisition, storage, and validation
- **Fraud Detection** (`dob_sob/fraud_detection/`): Community detection algorithms and analysis
- **CLI Interface** (`dob_sob/cli/`): Unified command-line interface for all operations
- **Core Infrastructure** (`dob_sob/core/`): Configuration, models, and exceptions
- **Web Interface**: Streamlit dashboard and Jupyter notebooks

### Data Architecture
- **Raw Data**: 18 NYC DOB datasets (~6.1GB) stored in `data/raw/`
- **Metadata**: Download tracking and checksums in `data/metadata/`
- **Incremental**: Smart update management in `data/incremental/`
- **Graph Storage**: Neo4j database with APOC plugins for advanced analytics

### Key Data Flows
1. **Data Acquisition**: `dob-sob data fetch` → `ModernBulkPipeline` → `data/raw/`
2. **Real-time Queries**: `dob-sob data query` → `NYCODataClient` → instant fraud assessment
3. **Community Detection**: `dob-sob fraud detect` → `CommunityDetector` → fraud patterns
4. **Graph Analysis**: Processed data → Neo4j → relationship discovery

## Important File Locations

### Entry Points
- `dob-sob` command: Unified CLI for all operations
- `dob_sob/cli/main.py`: CLI implementation
- `main.py`: Streamlit dashboard application (legacy)
- Legacy scripts in `scripts/` directory (being phased out)

### Configuration
- `pyproject.toml`: Python project configuration and dependencies
- `config/datasets.yaml`: External dataset configurations
- `config/environments/`: Environment-specific settings (dev, prod, test)
- `docker-compose.yml`: Multi-service container orchestration
- `.env`: Environment variables (copy from `docker.env.example`)

### Testing Structure
All test files follow pattern `test_*.py` and are located alongside their source modules:
- `scripts/etl/test_*.py`: Data pipeline and API client tests
- `scripts/fraud_detection/test_*.py`: Fraud detection algorithm tests

## Development Patterns

### Data Processing
- Use **Polars for large datasets** (preferred over Pandas for performance)
- **Async operations** for API calls and concurrent data processing
- **Type hints required** for all functions and classes
- **Comprehensive error handling** with detailed logging

### Fraud Detection Workflows
1. **Community Detection**: Use all 6 algorithms (Louvain, Label Propagation, etc.)
2. **Graph Analysis**: Leverage Neo4j for relationship discovery
3. **Pattern Matching**: Implement in `pattern_queries.py` for suspicious behavior
4. **Risk Scoring**: ML models for contractor and property assessment

### Testing Requirements
- **Test data pipelines** before modifying acquisition logic
- **Validate API connectivity** when changing OData client
- **Community detection accuracy** when updating algorithms
- **Integration tests** for end-to-end workflows

## NYC DOB Dataset Context

### Primary Datasets (Daily Updates)
- Housing Litigations, DOB Violations, ECB Violations
- Complaints Received, Maintenance Code Violations
- Active Projects, Certificates of Occupancy

### API Integration
- Real-time queries via `nyc_odata_client.py`
- Smart caching and rate limiting built-in
- BIN (Building Identification Number) instant analysis
- Contractor investigation by business name

### Data Quality
- Automated profiling via `data_profiling_framework.py`
- Checksum verification for download integrity
- Incremental update detection to minimize bandwidth
- Comprehensive logging for audit trails

## Performance Considerations

### Resource Management
- Neo4j configured for 2G heap, 1G page cache
- Parallel data downloads (max 8 concurrent by default)
- Memory-efficient processing with streaming where possible
- Docker resource limits configured in compose file

### Optimization Targets
- Real-time queries: <2 seconds for BIN analysis
- OData API response: <500ms with caching
- Community detection: <2 minutes for full dataset analysis
- Data downloads: Up to 8.5 MB/s with parallel processing

## Fraud Detection Specifics

### Community Detection Algorithms
1. **Louvain**: Fast modularity optimization
2. **Label Propagation**: Network-based community discovery
3. **Edge Betweenness**: Bridge-based community detection
4. **Clique Percolation**: Overlapping community identification
5. **Spectral Clustering**: Eigenvalue-based partitioning
6. **InfoMap**: Information-theoretic community detection

### Pattern Recognition Areas
- **Multi-entity fraud** (49% of detected patterns)
- **LLC shell company networks** (40.5% of cases)
- **Accelerated permit processing** chains
- **Suspicious inspector-contractor relationships**
- **Temporal correlation** between permits and complaints

## Docker Infrastructure

### Service Dependencies
- **Neo4j**: Must be running before app/jupyter services
- **Network**: Uses `dob-sob-network` bridge network
- **Volumes**: Persistent Neo4j data, shared app/data directories
- **Health Checks**: Neo4j readiness verification before service start

### Environment Variables
Key configurations in `.env` file:
- `NEO4J_PASSWORD`: Database password (change from default)
- `STREAMLIT_PORT`: Dashboard port (default 8501)
- `JUPYTER_PORT`: Notebook port (default 8889)
- `NEO4J_HTTP_PORT`: Database browser port (default 7474)

## Common Development Tasks

### Adding New Fraud Detection Algorithms
1. Implement in `community_detection_algorithms.py`
2. Add tests in `test_community_detection_algorithms.py`
3. Update algorithm selection logic
4. Document performance characteristics

### Extending Dataset Coverage
1. Add dataset configuration to `data_acquisition.py`
2. Update orchestration logic in `orchestrate_data_acquisition.py`
3. Test connectivity via `nyc_odata_client.py`
4. Add data profiling support

### Neo4j Schema Changes
1. Update graph model design
2. Modify ETL pipeline for new relationships
3. Update Cypher queries in analysis scripts
4. Test with sample data before full processing