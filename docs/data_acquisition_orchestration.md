# dob-sob: Data Acquisition Orchestration Guide

## Overview

The **dob-sob** Data Acquisition Orchestration system provides advanced capabilities for managing the download and processing of all 18 target datasets from the NYC Department of Buildings Open Data portal. This system builds upon the base data acquisition module to provide parallel processing, intelligent scheduling, resource management, and comprehensive reporting.

_Because catching dob-sobs requires comprehensive data - and we get ALL of it._

## Architecture

### Core Components

1. **DataAcquisitionOrchestrator**: Main coordination class
2. **DatasetJob**: Individual download job representation
3. **OrchestrationConfig**: Configuration management
4. **JobPriority/JobStatus**: Job state management enums

### Key Features

- **Parallel Downloads**: Intelligent concurrent processing with resource limits
- **Intelligent Scheduling**: Priority-based job ordering and dependency management
- **Resource Management**: Memory and bandwidth throttling
- **Error Handling**: Automatic retries with exponential backoff
- **Progress Tracking**: Real-time monitoring and comprehensive reporting
- **Cron Integration**: Built-in scheduling support for automation
- **Graceful Shutdown**: Signal handling for clean termination

## Dataset Catalog

### Primary Datasets (High Priority, Daily Updates)

| Dataset                            | Size (MB) | Update Freq | NYC Open Data ID |
| ---------------------------------- | --------- | ----------- | ---------------- |
| Housing Litigations                | 50        | Daily       | 59kj-x8nc        |
| DOB Violations                     | 250       | Daily       | 3h2n-5cm9        |
| ECB Violations                     | 300       | Daily       | 6bgk-3dad        |
| Complaints Received                | 317       | Daily       | eabe-havv        |
| Maintenance Code Violations        | 400       | Daily       | wvxf-dwi5        |
| Active Projects Under Construction | 150       | Daily       | w9ak-ipjd        |
| Certificates of Occupancy          | 200       | Daily       | bs8b-p36w        |

### Secondary Datasets (Medium Priority, Varied Updates)

| Dataset                         | Size (MB) | Update Freq | NYC Open Data ID |
| ------------------------------- | --------- | ----------- | ---------------- |
| Historical Permit Issuance      | 1200      | Daily       | ipu4-2q9a        |
| Job Application Filings         | 800       | Daily       | ic3t-wcy2        |
| DOB Job Application Filings     | 850       | Daily       | ic3t-wcy2        |
| Approved Permits                | 600       | Daily       | ipu4-2q9a        |
| Permit Issuance                 | 700       | Daily       | ipu4-2q9a        |
| Multiple Dwelling Registrations | 100       | Daily       | tesw-yqqr        |
| License Information             | 15        | Daily       | ipu4-2q9a        |
| Order to Repair/Vacate          | 75        | Daily       | 6bgk-3dad        |
| Facades Compliance Filings      | 25        | Daily       | qsjy-6dfz        |
| Registration Contacts           | 30        | Daily       | feu5-w2e2        |

### Tertiary Datasets (Lower Priority, Infrequent Updates)

| Dataset       | Size (MB) | Update Freq | NYC Open Data ID |
| ------------- | --------- | ----------- | ---------------- |
| Property Data | 500       | Quarterly   | rgy2-tti8        |

**Total Data Volume**: ~6.1 GB across all datasets _(That's a lot of dob-sob data to analyze!)_

## Installation & Setup

### Prerequisites

```bash
# Python 3.8+ with required packages
pip install requests urllib3 asyncio

# Or using uv (recommended)
uv add requests urllib3 asyncio
```

### Directory Structure

```
project/
├── scripts/etl/
│   ├── data_acquisition.py           # Base acquisition module
│   ├── orchestrate_data_acquisition.py  # Orchestration script
├── data/
│   ├── raw/                         # Downloaded datasets
│   ├── metadata/                    # Download metadata
│   ├── incremental/                 # Incremental updates
│   └── reports/                     # Execution reports
├── logs/                            # Orchestration logs
└── docs/                            # Documentation
```

## Usage Guide

### Basic Usage

#### Download All Datasets

```bash
# Download all 18 datasets with default settings
python scripts/etl/orchestrate_data_acquisition.py --all

# Force download all datasets (ignore existing files)
python scripts/etl/orchestrate_data_acquisition.py --all --force
```

#### Download Specific Dataset Groups

```bash
# Download high-priority datasets only
python scripts/etl/orchestrate_data_acquisition.py --primary

# Download secondary datasets
python scripts/etl/orchestrate_data_acquisition.py --secondary

# Download specific datasets
python scripts/etl/orchestrate_data_acquisition.py --datasets housing_litigations,dob_violations
```

#### Incremental Updates

```bash
# Run incremental updates only (skip full downloads)
python scripts/etl/orchestrate_data_acquisition.py --incremental-only

# Disable incremental downloads (force full downloads)
python scripts/etl/orchestrate_data_acquisition.py --all --no-incremental
```

### Advanced Configuration

#### Performance Tuning

```bash
# Adjust concurrent downloads for resource-constrained environments
python scripts/etl/orchestrate_data_acquisition.py --all --max-concurrent 2

# Limit large file downloads to prevent bandwidth saturation
python scripts/etl/orchestrate_data_acquisition.py --all --max-large-files 1

# Add throttling between job starts
python scripts/etl/orchestrate_data_acquisition.py --all --throttle 3.0

# Increase retry attempts for unreliable connections
python scripts/etl/orchestrate_data_acquisition.py --all --max-retries 5
```

#### Format Options

```bash
# Download in JSON format
python scripts/etl/orchestrate_data_acquisition.py --all --format json

# Use OData API (for filtered downloads)
python scripts/etl/orchestrate_data_acquisition.py --all --format odata
```

#### Testing & Validation

```bash
# Dry run to see what would be downloaded
python scripts/etl/orchestrate_data_acquisition.py --all --dry-run

# Custom report output location
python scripts/etl/orchestrate_data_acquisition.py --all --report-path ./monthly_reports/

# Quiet mode (minimal output)
python scripts/etl/orchestrate_data_acquisition.py --all --quiet
```

### Scheduling & Automation

#### Generate Cron Configuration

```bash
python scripts/etl/orchestrate_data_acquisition.py --generate-cron
```

**Output:**

```cron
# NYC DOB Data Acquisition - Daily Updates (2 AM)
0 2 * * * cd /path/to/project && python scripts/etl/orchestrate_data_acquisition.py --schedule-check --incremental-only

# NYC DOB Data Acquisition - Weekly Full Update (Sunday 3 AM)
0 3 * * 0 cd /path/to/project && python scripts/etl/orchestrate_data_acquisition.py --all --force

# NYC DOB Data Acquisition - Monthly Report (1st of month, 4 AM)
0 4 1 * * cd /path/to/project && python scripts/etl/orchestrate_data_acquisition.py --all --report-path /path/to/monthly_reports/
```

#### Schedule-Aware Downloads

```bash
# Only download datasets that need updates based on their schedule
python scripts/etl/orchestrate_data_acquisition.py --schedule-check
```

## Configuration Reference

### OrchestrationConfig Class

```python
class OrchestrationConfig:
    # Parallel processing settings
    MAX_CONCURRENT_DOWNLOADS = 4      # Maximum simultaneous downloads
    MAX_CONCURRENT_LARGE_FILES = 2    # Max large file downloads (>100MB)
    LARGE_FILE_THRESHOLD_MB = 100     # Large file size threshold

    # Resource management
    MAX_MEMORY_USAGE_GB = 4           # Maximum memory usage
    THROTTLE_DELAY_SECONDS = 1        # Delay between job starts

    # Retry settings
    DEFAULT_MAX_RETRIES = 3           # Default retry attempts
    RETRY_DELAY_BASE = 30             # Base delay for exponential backoff
    RETRY_DELAY_MAX = 300             # Maximum retry delay

    # Scheduling settings
    DAILY_SCHEDULE_HOUR = 2           # 2 AM for daily updates
    WEEKLY_SCHEDULE_DAY = 0           # Monday (0=Monday, 6=Sunday)
    MONTHLY_SCHEDULE_DAY = 1          # 1st of month
    QUARTERLY_SCHEDULE_MONTHS = [1,4,7,10]  # Jan, Apr, Jul, Oct

    # Logging
    LOG_LEVEL = "INFO"
    LOG_RETENTION_DAYS = 30
```

### Job Priority Calculation

Jobs are automatically prioritized based on:

1. **Update Frequency**: Daily > Weekly > Monthly > Quarterly
2. **Data Size**: Smaller files get higher priority for quick wins
3. **Critical Datasets**: Housing, violations, and complaints prioritized

Priority levels:

- **CRITICAL**: Essential for daily operations
- **HIGH**: Frequently updated, small datasets
- **MEDIUM**: Daily updates, larger datasets
- **LOW**: Infrequent updates, archival data

## Monitoring & Reporting

### Execution Reports

Each orchestration run generates a comprehensive JSON report containing:

```json
{
  "execution_summary": {
    "start_time": "2024-01-15T02:00:00",
    "end_time": "2024-01-15T02:45:23",
    "total_duration_seconds": 2723.0,
    "total_jobs": 18,
    "completed_jobs": 17,
    "failed_jobs": 1,
    "success_rate_percent": 94.44,
    "total_data_downloaded_mb": 5234.7,
    "average_speed_mbps": 1.924
  },
  "completed_jobs": {
    "housing_litigations": {
      "dataset_name": "Housing Litigations",
      "duration_seconds": 45.2,
      "file_size_mb": 52.3,
      "file_path": "data/raw/housing_litigations/housing_litigations_20240115.csv",
      "retry_count": 0
    }
  },
  "failed_jobs": {
    "property_data": {
      "dataset_name": "Property Data",
      "error_message": "Connection timeout after 300 seconds",
      "retry_count": 3,
      "final_status": "failed"
    }
  },
  "performance_metrics": {
    "peak_concurrent_jobs": 4,
    "throttle_delay_seconds": 1.0,
    "retry_attempts": 7,
    "large_files_processed": 3
  }
}
```

### Log Files

Orchestration logs are saved to `logs/orchestrator_YYYYMMDD.log` with detailed information:

- Job start/completion times
- Error messages and retry attempts
- Resource usage and throttling
- Download speeds and file sizes
- Signal handling and graceful shutdowns

### Real-time Monitoring

Monitor running orchestrations:

```bash
# View real-time logs
tail -f logs/orchestrator_$(date +%Y%m%d).log

# Monitor disk space during large downloads
watch -n 5 'df -h data/'

# Check process resource usage
top -p $(pgrep -f orchestrate_data_acquisition)
```

## Troubleshooting Guide

### Common Issues

#### 1. Connection Timeouts

**Symptoms**: Jobs failing with "Connection timeout" errors
**Solutions**:

```bash
# Reduce concurrent downloads
python orchestrate_data_acquisition.py --all --max-concurrent 2

# Increase throttling
python orchestrate_data_acquisition.py --all --throttle 5

# Use base data_acquisition.py for individual problematic datasets
python data_acquisition.py --dataset property_data --force
```

#### 2. Disk Space Issues

**Symptoms**: Jobs failing with "No space left on device"
**Solutions**:

```bash
# Check available space
df -h data/

# Clean up old incremental files
find data/incremental -name "*.csv" -mtime +30 -delete

# Download datasets individually to smaller drives
python orchestrate_data_acquisition.py --datasets housing_litigations --report-path /tmp/
```

#### 3. Memory Issues

**Symptoms**: System becomes unresponsive during large downloads
**Solutions**:

```bash
# Limit large file concurrency
python orchestrate_data_acquisition.py --all --max-large-files 1

# Process datasets sequentially
python orchestrate_data_acquisition.py --all --max-concurrent 1

# Monitor memory usage
watch -n 2 'free -h && ps aux | grep orchestrate'
```

#### 4. Network Interruptions

**Symptoms**: Frequent retry attempts, inconsistent download speeds
**Solutions**:

```bash
# Increase retry attempts and delays
python orchestrate_data_acquisition.py --all --max-retries 5 --throttle 10

# Use incremental mode to resume partial downloads
python orchestrate_data_acquisition.py --all --incremental-only

# Check network stability
ping -c 10 data.cityofnewyork.us
```

### Error Recovery

#### Resume Failed Downloads

```bash
# Retry only failed datasets from last run
python orchestrate_data_acquisition.py --datasets $(grep '"failed"' data/reports/orchestration_report_*.json | grep -o '"[^"]*":' | tr -d '":' | tr '\n' ',')
```

#### Manual Intervention

```bash
# Download specific dataset with base module
python data_acquisition.py --dataset property_data --force --log-level DEBUG

# Check dataset metadata
python data_acquisition.py --summary

# Validate downloaded files
python data_acquisition.py --check-updates
```

### Performance Optimization

#### For Fast Networks

```bash
python orchestrate_data_acquisition.py --all --max-concurrent 8 --max-large-files 4 --throttle 0.5
```

#### For Slow/Unreliable Networks

```bash
python orchestrate_data_acquisition.py --all --max-concurrent 1 --throttle 5 --max-retries 10
```

#### For Resource-Constrained Systems

```bash
python orchestrate_data_acquisition.py --primary --max-concurrent 2 --max-large-files 1 --throttle 3
```

## Integration Examples

### With Task Master AI

```python
# Update Task 3.5 status after successful orchestration
from orchestrate_data_acquisition import DataAcquisitionOrchestrator

async def run_orchestrated_download():
    orchestrator = DataAcquisitionOrchestrator()
    jobs = orchestrator.create_jobs_from_datasets()
    report = await orchestrator.run_parallel_orchestration(jobs)

    if report["execution_summary"]["success_rate_percent"] > 90:
        # Mark task as complete in Task Master
        return True
    return False
```

### With External Monitoring

```python
import requests

def send_completion_notification(report):
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

    message = {
        "text": f"NYC DOB Data Download Complete: {report['execution_summary']['success_rate_percent']}% success rate"
    }

    requests.post(webhook_url, json=message)
```

### With Data Processing Pipeline

```bash
#!/bin/bash
# Complete data pipeline script

# 1. Download latest data
python scripts/etl/orchestrate_data_acquisition.py --schedule-check --incremental-only

# 2. Run data quality checks
python scripts/data_profiling/data_profiling_framework.py

# 3. Execute fraud detection analysis
python scripts/fraud_detection/community_detection_algorithms.py

# 4. Generate reports
python scripts/analysis/generate_summary_report.py
```

## Advanced Features

### Custom Job Configuration

```python
from orchestrate_data_acquisition import DataAcquisitionOrchestrator, DatasetJob, JobPriority

# Create custom jobs
orchestrator = DataAcquisitionOrchestrator()
custom_jobs = [
    DatasetJob(
        dataset_key="housing_litigations",
        dataset_name="Housing Litigations",
        priority=JobPriority.CRITICAL,
        estimated_size_mb=50,
        update_frequency="daily",
        force_download=True
    )
]

report = await orchestrator.run_parallel_orchestration(custom_jobs)
```

### Resource Monitoring

```python
import psutil

class ResourceMonitoringConfig(OrchestrationConfig):
    def __init__(self):
        super().__init__()
        # Adjust based on system resources
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.MAX_CONCURRENT_DOWNLOADS = min(4, int(total_memory_gb // 2))
```

### Notification Integration

```python
def setup_notifications(orchestrator):
    # Email notifications
    orchestrator.config.ENABLE_EMAIL_NOTIFICATIONS = True

    # Slack notifications
    orchestrator.config.ENABLE_SLACK_NOTIFICATIONS = True
    orchestrator.config.SLACK_WEBHOOK_URL = "https://hooks.slack.com/..."

    # Custom webhook
    orchestrator.config.ENABLE_WEBHOOK_NOTIFICATIONS = True
    orchestrator.config.WEBHOOK_URL = "https://api.example.com/notifications"
```

## Security Considerations

### API Rate Limiting

- The NYC Open Data portal has rate limits
- Orchestrator includes built-in throttling (1 second default)
- Increase throttling for high-volume usage

### Data Privacy

- Downloaded data may contain PII
- Ensure appropriate access controls on data directories
- Consider encryption for sensitive datasets

### Network Security

- All downloads use HTTPS
- SSL certificate validation enabled by default
- Consider using VPN for automated downloads

## Future Enhancements

### Planned Features

1. **Machine Learning-Based Scheduling**: Predict optimal download times
2. **Data Quality Integration**: Automatic validation post-download
3. **Cloud Storage Support**: Direct upload to S3/GCS/Azure
4. **Real-time Streaming**: WebSocket-based incremental updates
5. **Dashboard Interface**: Web UI for monitoring and control

### Extension Points

- Custom notification handlers
- Alternative storage backends
- Data transformation pipelines
- External monitoring integration

---

**Documentation Version**: 1.0  
**Last Updated**: January 2024  
**Maintainer**: Task Master AI System
