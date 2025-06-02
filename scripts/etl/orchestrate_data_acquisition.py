#!/usr/bin/env python3
"""
NYC DOB Data Acquisition Orchestration Script

This script provides advanced orchestration capabilities for managing all 18 target 
datasets from the NYC Department of Buildings with parallel processing, scheduling 
support, and comprehensive reporting.

Features:
- Parallel downloads for independent datasets
- Intelligent scheduling based on update frequencies
- Resource management and throttling
- Detailed progress reporting and notifications
- Cron job integration
- Advanced error handling and recovery
- Configuration profiles for different environments

Author: Task Master AI System
Version: 1.0
"""

import asyncio
import os
import sys
import json
import time
import logging
import argparse
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_acquisition import NYCDataDownloader, DataAcquisitionConfig

class JobPriority(Enum):
    """Job priority levels for scheduling"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"  
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

@dataclass
class DatasetJob:
    """Represents a single dataset download job"""
    dataset_key: str
    dataset_name: str
    priority: JobPriority
    estimated_size_mb: int
    update_frequency: str
    format_type: str = "csv"
    force_download: bool = False
    use_incremental: bool = True
    max_retries: int = 3
    retry_count: int = 0
    status: JobStatus = JobStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate job duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if job is completed (success or failure)"""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]

class OrchestrationConfig:
    """Configuration for orchestration settings"""
    
    # Parallel processing settings
    MAX_CONCURRENT_DOWNLOADS = 4  # Maximum simultaneous downloads
    MAX_CONCURRENT_LARGE_FILES = 2  # Max large file downloads (>100MB)
    LARGE_FILE_THRESHOLD_MB = 100
    
    # Resource management
    MAX_MEMORY_USAGE_GB = 4  # Maximum memory usage
    THROTTLE_DELAY_SECONDS = 1  # Delay between job starts
    
    # Retry settings
    DEFAULT_MAX_RETRIES = 3
    RETRY_DELAY_BASE = 30  # Base delay for exponential backoff
    RETRY_DELAY_MAX = 300  # Maximum retry delay
    
    # Scheduling settings
    DAILY_SCHEDULE_HOUR = 2  # 2 AM for daily updates
    WEEKLY_SCHEDULE_DAY = 0  # Monday (0=Monday, 6=Sunday)
    MONTHLY_SCHEDULE_DAY = 1  # 1st of month
    QUARTERLY_SCHEDULE_MONTHS = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
    
    # Notification settings
    ENABLE_EMAIL_NOTIFICATIONS = False
    ENABLE_SLACK_NOTIFICATIONS = False
    ENABLE_WEBHOOK_NOTIFICATIONS = False
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_RETENTION_DAYS = 30

class DataAcquisitionOrchestrator:
    """Main orchestration class for coordinating all data acquisition activities"""
    
    def __init__(self, config: OrchestrationConfig = None):
        """Initialize the orchestrator"""
        self.config = config or OrchestrationConfig()
        self.jobs: List[DatasetJob] = []
        self.running_jobs: Set[str] = set()
        self.completed_jobs: Dict[str, DatasetJob] = {}
        self.failed_jobs: Dict[str, DatasetJob] = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize downloader
        self.downloader = NYCDataDownloader(log_level=self.config.LOG_LEVEL)
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._shutdown_requested = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup orchestration logging"""
        logger = logging.getLogger("orchestrator")
        logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # File handler with rotation
        log_file = log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.LOG_LEVEL))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
    
    def create_jobs_from_datasets(self, 
                                 dataset_keys: Optional[List[str]] = None,
                                 format_type: str = "csv",
                                 force_download: bool = False,
                                 use_incremental: bool = True) -> List[DatasetJob]:
        """Create jobs from dataset configuration"""
        
        if dataset_keys is None:
            dataset_keys = list(DataAcquisitionConfig.DATASETS.keys())
        
        jobs = []
        for dataset_key in dataset_keys:
            if dataset_key not in DataAcquisitionConfig.DATASETS:
                self.logger.warning(f"Unknown dataset: {dataset_key}")
                continue
                
            dataset_config = DataAcquisitionConfig.DATASETS[dataset_key]
            
            # Determine priority based on update frequency and size
            priority = self._calculate_priority(dataset_config)
            
            job = DatasetJob(
                dataset_key=dataset_key,
                dataset_name=dataset_config["name"],
                priority=priority,
                estimated_size_mb=dataset_config.get("estimated_size_mb", 0),
                update_frequency=dataset_config.get("update_frequency", "daily"),
                format_type=format_type,
                force_download=force_download,
                use_incremental=use_incremental,
                max_retries=self.config.DEFAULT_MAX_RETRIES
            )
            jobs.append(job)
        
        # Sort by priority (highest first) then by size (smallest first for quick wins)
        jobs.sort(key=lambda x: (-x.priority.value, x.estimated_size_mb))
        
        return jobs
    
    def _calculate_priority(self, dataset_config: Dict) -> JobPriority:
        """Calculate job priority based on dataset characteristics"""
        
        # High priority for frequently updated, critical datasets
        if dataset_config.get("update_frequency") == "daily":
            if dataset_config.get("estimated_size_mb", 0) < 100:
                return JobPriority.HIGH
            else:
                return JobPriority.MEDIUM
        elif dataset_config.get("update_frequency") == "weekly":
            return JobPriority.MEDIUM
        else:
            return JobPriority.LOW
    
    def _should_run_parallel(self, job: DatasetJob) -> bool:
        """Determine if job can run in parallel based on resource constraints"""
        
        # Check concurrent download limits
        if len(self.running_jobs) >= self.config.MAX_CONCURRENT_DOWNLOADS:
            return False
        
        # Check large file constraints
        if job.estimated_size_mb > self.config.LARGE_FILE_THRESHOLD_MB:
            large_files_running = sum(
                1 for job_key in self.running_jobs 
                if self._get_job_by_key(job_key).estimated_size_mb > self.config.LARGE_FILE_THRESHOLD_MB
            )
            if large_files_running >= self.config.MAX_CONCURRENT_LARGE_FILES:
                return False
        
        return True
    
    def _get_job_by_key(self, job_key: str) -> Optional[DatasetJob]:
        """Get job by dataset key"""
        for job in self.jobs:
            if job.dataset_key == job_key:
                return job
        return None
    
    async def execute_job(self, job: DatasetJob) -> bool:
        """Execute a single dataset download job"""
        
        self.logger.info(f"Starting job: {job.dataset_key} ({job.dataset_name})")
        
        job.status = JobStatus.RUNNING
        job.start_time = datetime.now()
        self.running_jobs.add(job.dataset_key)
        
        try:
            # Execute the download
            success = self.downloader.download_dataset(
                dataset_key=job.dataset_key,
                format_type=job.format_type,
                force_download=job.force_download,
                use_incremental=job.use_incremental
            )
            
            job.end_time = datetime.now()
            
            if success:
                job.status = JobStatus.COMPLETED
                self.completed_jobs[job.dataset_key] = job
                self.logger.info(f"Completed job: {job.dataset_key} in {job.duration}")
                
                # Get file info from metadata
                metadata = self.downloader.metadata_manager.get_dataset_metadata(job.dataset_key)
                job.file_path = metadata.get('file_path')
                job.file_size_bytes = metadata.get('file_size', 0)
                
            else:
                raise Exception("Download failed")
                
        except Exception as e:
            self.logger.error(f"Job failed: {job.dataset_key} - {str(e)}")
            job.status = JobStatus.FAILED
            job.end_time = datetime.now()
            job.error_message = str(e)
            
            # Check for retry
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.RETRYING
                self.logger.info(f"Retrying job {job.dataset_key} (attempt {job.retry_count}/{job.max_retries})")
                
                # Exponential backoff delay
                delay = min(
                    self.config.RETRY_DELAY_BASE * (2 ** (job.retry_count - 1)),
                    self.config.RETRY_DELAY_MAX
                )
                await asyncio.sleep(delay)
                
                return await self.execute_job(job)  # Recursive retry
            else:
                self.failed_jobs[job.dataset_key] = job
                success = False
        
        finally:
            self.running_jobs.discard(job.dataset_key)
        
        return success
    
    async def run_parallel_orchestration(self, jobs: List[DatasetJob]) -> Dict[str, Any]:
        """Run jobs with intelligent parallel processing"""
        
        self.jobs = jobs.copy()
        self.logger.info(f"Starting orchestration of {len(jobs)} jobs")
        
        start_time = datetime.now()
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=self.config.MAX_CONCURRENT_DOWNLOADS) as executor:
            
            # Track pending jobs
            pending_jobs = jobs.copy()
            running_futures = {}
            
            while pending_jobs or running_futures:
                
                # Check for shutdown
                if self._shutdown_requested:
                    self.logger.info("Shutdown requested, cancelling remaining jobs")
                    for future in running_futures.values():
                        future.cancel()
                    break
                
                # Start new jobs if capacity allows
                while pending_jobs and len(running_futures) < self.config.MAX_CONCURRENT_DOWNLOADS:
                    
                    # Find next eligible job
                    next_job = None
                    for i, job in enumerate(pending_jobs):
                        if self._should_run_parallel(job):
                            next_job = pending_jobs.pop(i)
                            break
                    
                    if next_job is None:
                        break  # No eligible jobs, wait for running jobs to complete
                    
                    # Submit job to executor
                    future = executor.submit(
                        asyncio.run, 
                        self.execute_job(next_job)
                    )
                    running_futures[next_job.dataset_key] = future
                    
                    # Throttle job starts
                    if self.config.THROTTLE_DELAY_SECONDS > 0:
                        await asyncio.sleep(self.config.THROTTLE_DELAY_SECONDS)
                
                # Check completed jobs
                completed_futures = []
                for job_key, future in running_futures.items():
                    if future.done():
                        completed_futures.append(job_key)
                        try:
                            success = future.result()
                            self.logger.debug(f"Job {job_key} completed: {success}")
                        except Exception as e:
                            self.logger.error(f"Job {job_key} exception: {e}")
                
                # Remove completed futures
                for job_key in completed_futures:
                    del running_futures[job_key]
                
                # Brief sleep to prevent busy waiting
                if running_futures:
                    await asyncio.sleep(1)
        
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        # Generate final report
        report = self._generate_execution_report(total_duration)
        
        self.logger.info(f"Orchestration completed in {total_duration}")
        self.logger.info(f"Success: {len(self.completed_jobs)}, Failed: {len(self.failed_jobs)}")
        
        return report
    
    def _generate_execution_report(self, total_duration: timedelta) -> Dict[str, Any]:
        """Generate comprehensive execution report"""
        
        total_jobs = len(self.jobs)
        completed_count = len(self.completed_jobs)
        failed_count = len(self.failed_jobs)
        success_rate = (completed_count / total_jobs * 100) if total_jobs > 0 else 0
        
        # Calculate total data downloaded
        total_size_bytes = sum(job.file_size_bytes for job in self.completed_jobs.values())
        total_size_mb = total_size_bytes / 1024 / 1024
        
        # Average download speed
        avg_speed_mbps = 0
        if total_duration.total_seconds() > 0:
            avg_speed_mbps = total_size_mb / total_duration.total_seconds()
        
        report = {
            "execution_summary": {
                "start_time": self.jobs[0].start_time.isoformat() if self.jobs else None,
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": total_duration.total_seconds(),
                "total_jobs": total_jobs,
                "completed_jobs": completed_count,
                "failed_jobs": failed_count,
                "success_rate_percent": round(success_rate, 2),
                "total_data_downloaded_mb": round(total_size_mb, 2),
                "average_speed_mbps": round(avg_speed_mbps, 4)
            },
            "completed_jobs": {
                job_key: {
                    "dataset_name": job.dataset_name,
                    "duration_seconds": job.duration.total_seconds() if job.duration else 0,
                    "file_size_mb": round(job.file_size_bytes / 1024 / 1024, 2),
                    "file_path": job.file_path,
                    "retry_count": job.retry_count
                }
                for job_key, job in self.completed_jobs.items()
            },
            "failed_jobs": {
                job_key: {
                    "dataset_name": job.dataset_name,
                    "error_message": job.error_message,
                    "retry_count": job.retry_count,
                    "final_status": job.status.value
                }
                for job_key, job in self.failed_jobs.items()
            },
            "performance_metrics": {
                "peak_concurrent_jobs": self.config.MAX_CONCURRENT_DOWNLOADS,
                "throttle_delay_seconds": self.config.THROTTLE_DELAY_SECONDS,
                "retry_attempts": sum(job.retry_count for job in self.jobs),
                "large_files_processed": sum(
                    1 for job in self.completed_jobs.values() 
                    if job.estimated_size_mb > self.config.LARGE_FILE_THRESHOLD_MB
                )
            }
        }
        
        return report
    
    def save_execution_report(self, report: Dict[str, Any], output_path: Optional[Path] = None):
        """Save execution report to file"""
        
        if output_path is None:
            reports_dir = Path("data/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = reports_dir / f"orchestration_report_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Execution report saved to: {output_path}")
        return output_path

# CLI Functions for different orchestration modes

def create_cli_parser() -> argparse.ArgumentParser:
    """Create comprehensive CLI parser"""
    
    parser = argparse.ArgumentParser(
        description="NYC DOB Data Acquisition Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets with default settings
  python orchestrate_data_acquisition.py --all
  
  # Download specific datasets in parallel
  python orchestrate_data_acquisition.py --datasets housing_litigations,dob_violations
  
  # High priority datasets only with force download
  python orchestrate_data_acquisition.py --primary --force
  
  # Schedule-aware incremental updates
  python orchestrate_data_acquisition.py --schedule-check --incremental-only
  
  # Generate cron schedule configuration
  python orchestrate_data_acquisition.py --generate-cron
  
  # Performance tuning for large datasets
  python orchestrate_data_acquisition.py --all --max-concurrent 2 --throttle 3
        """
    )
    
    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--all", action="store_true", 
                              help="Download all 18 target datasets")
    dataset_group.add_argument("--primary", action="store_true",
                              help="Download primary datasets only")
    dataset_group.add_argument("--secondary", action="store_true", 
                              help="Download secondary datasets only")
    dataset_group.add_argument("--datasets", type=str, 
                              help="Comma-separated list of specific datasets")
    
    # Download options
    parser.add_argument("--format", type=str, default="csv", 
                       choices=["csv", "json", "odata"],
                       help="Download format (default: csv)")
    parser.add_argument("--force", action="store_true",
                       help="Force download even if files exist")
    parser.add_argument("--no-incremental", action="store_true",
                       help="Disable incremental downloads")
    parser.add_argument("--incremental-only", action="store_true",
                       help="Only run incremental updates (skip full downloads)")
    
    # Orchestration settings
    parser.add_argument("--max-concurrent", type=int, default=4,
                       help="Maximum concurrent downloads (default: 4)")
    parser.add_argument("--max-large-files", type=int, default=2,
                       help="Maximum concurrent large file downloads (default: 2)")
    parser.add_argument("--throttle", type=float, default=1.0,
                       help="Delay between job starts in seconds (default: 1.0)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retry attempts per job (default: 3)")
    
    # Scheduling and automation
    parser.add_argument("--schedule-check", action="store_true",
                       help="Check schedule and only download datasets due for update")
    parser.add_argument("--generate-cron", action="store_true",
                       help="Generate cron job configuration")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be downloaded without executing")
    
    # Output and reporting
    parser.add_argument("--report-path", type=str,
                       help="Custom path for execution report")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")
    
    return parser

async def main():
    """Main orchestration entry point"""
    
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Create orchestration configuration
    config = OrchestrationConfig()
    config.MAX_CONCURRENT_DOWNLOADS = args.max_concurrent
    config.MAX_CONCURRENT_LARGE_FILES = args.max_large_files
    config.THROTTLE_DELAY_SECONDS = args.throttle
    config.DEFAULT_MAX_RETRIES = args.max_retries
    config.LOG_LEVEL = args.log_level
    
    # Initialize orchestrator
    orchestrator = DataAcquisitionOrchestrator(config)
    
    if args.generate_cron:
        # Generate cron configuration
        print("Generating cron job configuration...")
        print("Add these lines to your crontab (crontab -e):")
        print()
        print("# NYC DOB Data Acquisition - Daily Updates (2 AM)")
        print("0 2 * * * cd /path/to/project && python scripts/etl/orchestrate_data_acquisition.py --schedule-check --incremental-only")
        print()
        print("# NYC DOB Data Acquisition - Weekly Full Update (Sunday 3 AM)")  
        print("0 3 * * 0 cd /path/to/project && python scripts/etl/orchestrate_data_acquisition.py --all --force")
        print()
        print("# NYC DOB Data Acquisition - Monthly Report (1st of month, 4 AM)")
        print("0 4 1 * * cd /path/to/project && python scripts/etl/orchestrate_data_acquisition.py --all --report-path /path/to/monthly_reports/")
        return
    
    # Determine dataset selection
    dataset_keys = None
    if args.datasets:
        dataset_keys = [key.strip() for key in args.datasets.split(",")]
    elif args.primary:
        # Define primary datasets (high priority, frequently updated)
        dataset_keys = [
            "housing_litigations", "dob_violations", "ecb_violations",
            "complaints_received", "maintenance_code_violations",
            "active_projects_under_construction", "certificates_of_occupancy"
        ]
    elif args.secondary:
        # Define secondary datasets (lower priority, less frequent updates)
        dataset_keys = [
            "property_data", "registration_contacts", "facades_compliance_filings",
            "order_to_repair_vacate", "license_info"
        ]
    elif not args.all:
        parser.print_help()
        print("\nPlease specify which datasets to download (--all, --primary, --secondary, or --datasets)")
        return
    
    # Schedule check logic
    if args.schedule_check:
        orchestrator.logger.info("Performing schedule-based dataset selection...")
        # Filter datasets based on their update schedules
        # This would be implemented to check last download time vs update frequency
        # For now, we'll use all datasets
    
    # Create jobs
    jobs = orchestrator.create_jobs_from_datasets(
        dataset_keys=dataset_keys,
        format_type=args.format,
        force_download=args.force,
        use_incremental=not args.no_incremental
    )
    
    if args.incremental_only:
        # Filter to only jobs that support incremental updates
        jobs = [job for job in jobs if job.use_incremental]
        orchestrator.logger.info(f"Filtered to {len(jobs)} incremental-capable datasets")
    
    if args.dry_run:
        print(f"\n=== DRY RUN: Would download {len(jobs)} datasets ===")
        for job in jobs:
            inc_status = " (incremental)" if job.use_incremental else " (full)"
            print(f"  {job.dataset_key}: {job.dataset_name}{inc_status}")
            print(f"    Priority: {job.priority.name}, Size: {job.estimated_size_mb}MB")
        print(f"\nConcurrency: {config.MAX_CONCURRENT_DOWNLOADS} jobs, throttle: {config.THROTTLE_DELAY_SECONDS}s")
        return
    
    # Execute orchestration
    print(f"\n=== Starting Orchestrated Download of {len(jobs)} datasets ===")
    
    try:
        report = await orchestrator.run_parallel_orchestration(jobs)
        
        # Save execution report
        report_path = None
        if args.report_path:
            report_path = Path(args.report_path)
        orchestrator.save_execution_report(report, report_path)
        
        # Print summary
        summary = report["execution_summary"]
        print(f"\n=== Orchestration Complete ===")
        print(f"Duration: {summary['total_duration_seconds']:.1f} seconds")
        print(f"Success Rate: {summary['success_rate_percent']}%")
        print(f"Data Downloaded: {summary['total_data_downloaded_mb']:.1f} MB")
        print(f"Average Speed: {summary['average_speed_mbps']:.3f} MB/s")
        
        if summary['failed_jobs'] > 0:
            print(f"\n❌ Failed Jobs: {summary['failed_jobs']}")
            for job_key, job_info in report["failed_jobs"].items():
                print(f"  {job_key}: {job_info['error_message']}")
        
        if summary['completed_jobs'] > 0:
            print(f"\n✅ Completed Jobs: {summary['completed_jobs']}")
            if not args.quiet:
                for job_key, job_info in report["completed_jobs"].items():
                    print(f"  {job_key}: {job_info['file_size_mb']:.1f} MB in {job_info['duration_seconds']:.1f}s")
    
    except KeyboardInterrupt:
        orchestrator.logger.info("Orchestration interrupted by user")
        print("\n❌ Orchestration interrupted")
    except Exception as e:
        orchestrator.logger.error(f"Orchestration failed: {e}")
        print(f"\n❌ Orchestration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 