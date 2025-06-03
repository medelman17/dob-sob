"""
Modern bulk data acquisition pipeline using latest Python features and packages.

Uses Python 3.11+ features, modern async patterns, and contemporary libraries
for enterprise-grade data downloading with excellent observability.
"""

import asyncio
import sys
from collections.abc import Mapping, Sequence, AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import httpx
import structlog
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeRemainingColumn,
    TransferSpeedColumn,
    TaskID
)
from rich.table import Table
from rich.live import Live
import aiofiles
import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum, auto

from ...core.config import get_settings
from ...core.exceptions import DownloadError, DataValidationError
from ...core.models import DownloadResult, DatasetStatus
from .datasets import dataset_registry

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
console = Console()


class DownloadStage(Enum):
    """Download pipeline stages using auto() for cleaner enum definition."""
    PREPARING = auto()
    DOWNLOADING = auto()
    VALIDATING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()


@runtime_checkable
class ProgressReporter(Protocol):
    """Protocol for progress reporting - more flexible than inheritance."""
    
    async def report_progress(
        self, 
        dataset_name: str,
        stage: DownloadStage,
        bytes_downloaded: int,
        total_bytes: int,
        **kwargs: Any
    ) -> None:
        """Report download progress."""
        ...


@dataclass(frozen=True, slots=True)  # Frozen and slots for better performance
class DownloadMetrics:
    """Immutable download metrics using modern dataclass features."""
    dataset_name: str
    stage: DownloadStage
    bytes_downloaded: int = 0
    total_bytes: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    retry_count: int = 0
    
    @property
    def progress_percentage(self) -> float:
        """Calculate download progress percentage."""
        return (self.bytes_downloaded / self.total_bytes * 100) if self.total_bytes > 0 else 0.0
    
    @property
    def download_speed_bps(self) -> float:
        """Calculate current download speed in bytes per second."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.bytes_downloaded / elapsed if elapsed > 0 else 0.0
    
    @property
    def eta_seconds(self) -> float | None:
        """Estimate time remaining in seconds using modern union syntax."""
        if self.total_bytes == 0 or self.bytes_downloaded == 0:
            return None
        
        remaining_bytes = self.total_bytes - self.bytes_downloaded
        return remaining_bytes / self.download_speed_bps if self.download_speed_bps > 0 else None


class RichProgressReporter:
    """Modern progress reporting using Rich library."""
    
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[dataset]}", justify="left"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
            expand=True
        )
        self.tasks: dict[str, TaskID] = {}
        self.live: Live | None = None
    
    async def __aenter__(self):
        """Async context manager for Rich display."""
        self.live = Live(self.progress, console=console, refresh_per_second=10)
        self.live.__enter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup Rich display."""
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)
    
    async def report_progress(
        self,
        dataset_name: str,
        stage: DownloadStage,
        bytes_downloaded: int,
        total_bytes: int,
        **kwargs: Any
    ) -> None:
        """Report progress using Rich progress bars."""
        if dataset_name not in self.tasks:
            self.tasks[dataset_name] = self.progress.add_task(
                description="",
                dataset=dataset_name,
                total=total_bytes
            )
        
        task_id = self.tasks[dataset_name]
        
        # Update task based on stage
        if stage == DownloadStage.DOWNLOADING:
            self.progress.update(task_id, completed=bytes_downloaded, total=total_bytes)
        elif stage == DownloadStage.VALIDATING:
            self.progress.update(task_id, description="[yellow]Validating...")
        elif stage == DownloadStage.COMPLETED:
            self.progress.update(task_id, description="[green]✓ Completed", completed=total_bytes)
        elif stage == DownloadStage.FAILED:
            self.progress.update(task_id, description="[red]✗ Failed")


class ModernBulkPipeline:
    """
    Modern bulk download pipeline using latest Python features and packages.
    
    Features Python 3.11+ improvements:
    - TaskGroup for better async error handling
    - Modern type annotations with built-in generics
    - httpx instead of aiohttp for better async HTTP
    - Rich for beautiful progress reporting
    - Structured logging with contextvars
    - Protocol-based interfaces for flexibility
    """
    
    def __init__(
        self,
        *,
        max_retries: int = 5,
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        timeout_seconds: float = 600.0,
        chunk_size: int = 1024 * 1024,
        progress_reporter: ProgressReporter | None = None
    ):
        """Initialize modern pipeline with keyword-only arguments."""
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        self.timeout_seconds = timeout_seconds
        self.chunk_size = chunk_size
        self.progress_reporter = progress_reporter
        
        # State tracking with modern type hints
        self.active_downloads: dict[str, DownloadMetrics] = {}
        self.completed_downloads: list[DownloadResult] = []
        self.failed_downloads: list[DownloadResult] = []
        
        # HTTP client configuration
        self.client_config = {
            "timeout": httpx.Timeout(timeout_seconds),
            "limits": httpx.Limits(
                max_keepalive_connections=50,
                max_connections=100,
                keepalive_expiry=30.0
            ),
            "headers": {
                "User-Agent": "dob-sob-fraud-detection/2.0 (+https://github.com/medelman17/dob-sob)"
            }
        }
    
    @asynccontextmanager
    async def http_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Modern context manager for HTTP client lifecycle."""
        async with httpx.AsyncClient(**self.client_config) as client:
            yield client
    
    async def _stream_download(
        self,
        client: httpx.AsyncClient,
        url: str,
        file_path: Path,
        metrics: DownloadMetrics
    ) -> DownloadMetrics:
        """Stream download with modern async patterns and error handling."""
        # Check for resumable download
        resume_pos = 0
        if file_path.exists():
            resume_pos = file_path.stat().st_size
            logger.info("resuming_download", dataset=metrics.dataset_name, resume_pos=resume_pos)
        
        headers = {}
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"
        
        try:
            # Use modern httpx streaming
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                
                # Handle partial content for resume
                file_mode = "ab" if response.status_code == 206 else "wb"
                if response.status_code != 206:
                    resume_pos = 0  # Server doesn't support resume
                
                # Get total size
                content_length = response.headers.get("content-length")
                total_bytes = int(content_length) + resume_pos if content_length else 0
                
                # Update metrics with total size
                metrics = replace(
                    metrics,
                    total_bytes=total_bytes,
                    bytes_downloaded=resume_pos,
                    stage=DownloadStage.DOWNLOADING
                )
                
                # Stream to file with progress reporting
                async with aiofiles.open(file_path, file_mode) as f:
                    async for chunk in response.aiter_bytes(chunk_size=self.chunk_size):
                        if chunk:
                            await f.write(chunk)
                            
                            # Update metrics (immutable pattern)
                            metrics = replace(
                                metrics,
                                bytes_downloaded=metrics.bytes_downloaded + len(chunk),
                                last_update=datetime.now()
                            )
                            
                            # Report progress if reporter available
                            if self.progress_reporter:
                                await self.progress_reporter.report_progress(
                                    dataset_name=metrics.dataset_name,
                                    stage=metrics.stage,
                                    bytes_downloaded=metrics.bytes_downloaded,
                                    total_bytes=metrics.total_bytes
                                )
                
                logger.info(
                    "download_completed",
                    dataset=metrics.dataset_name,
                    bytes_downloaded=metrics.bytes_downloaded,
                    duration_seconds=(datetime.now() - metrics.start_time).total_seconds()
                )
                
                return replace(metrics, stage=DownloadStage.COMPLETED)
                
        except httpx.RequestError as e:
            logger.error("download_network_error", dataset=metrics.dataset_name, error=str(e))
            raise DownloadError(f"Network error downloading {metrics.dataset_name}") from e
        except httpx.HTTPStatusError as e:
            logger.error("download_http_error", dataset=metrics.dataset_name, status=e.response.status_code)
            raise DownloadError(f"HTTP {e.response.status_code} downloading {metrics.dataset_name}") from e
    
    async def _calculate_checksum_async(self, file_path: Path) -> str:
        """Calculate checksum using async file I/O."""
        hash_md5 = hashlib.md5()
        
        async with aiofiles.open(file_path, "rb") as f:
            async for chunk in self._read_chunks(f, 8192):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    async def _read_chunks(self, file_handle, chunk_size: int) -> AsyncIterator[bytes]:
        """Async generator for reading file chunks."""
        while chunk := await file_handle.read(chunk_size):
            yield chunk
    
    async def _download_with_exponential_backoff(
        self,
        dataset_name: str,
        client: httpx.AsyncClient
    ) -> DownloadResult:
        """Download with modern exponential backoff using asyncio.sleep."""
        try:
            config = dataset_registry.get_dataset(dataset_name)
        except Exception as e:
            logger.error("dataset_config_error", dataset=dataset_name, error=str(e))
            return DownloadResult(
                dataset_name=dataset_name,
                dataset_id="unknown",
                success=False,
                error_message=f"Dataset configuration error: {e!s}"
            )
        
        settings = get_settings()
        settings.create_directories()
        
        download_url = f"{settings.api.base_url}/api/views/{config.id}/rows.csv?accessType=DOWNLOAD"
        output_path = settings.raw_data_dir / dataset_name / f"{dataset_name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        metrics = DownloadMetrics(
            dataset_name=dataset_name,
            stage=DownloadStage.PREPARING
        )
        
        self.active_downloads[dataset_name] = metrics
        
        # Retry loop with exponential backoff
        last_exception: Exception | None = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(
                    "download_attempt",
                    dataset=dataset_name,
                    attempt=attempt + 1,
                    max_attempts=self.max_retries + 1
                )
                
                # Perform download
                final_metrics = await self._stream_download(client, download_url, output_path, metrics)
                
                # Validation stage
                final_metrics = replace(final_metrics, stage=DownloadStage.VALIDATING)
                if self.progress_reporter:
                    await self.progress_reporter.report_progress(
                        dataset_name=dataset_name,
                        stage=DownloadStage.VALIDATING,
                        bytes_downloaded=final_metrics.bytes_downloaded,
                        total_bytes=final_metrics.total_bytes
                    )
                
                # Calculate checksum
                checksum = await self._calculate_checksum_async(output_path)
                
                # Success!
                result = DownloadResult(
                    dataset_name=dataset_name,
                    dataset_id=config.id,
                    success=True,
                    file_path=output_path,
                    file_size_bytes=final_metrics.bytes_downloaded,
                    download_time_seconds=(datetime.now() - final_metrics.start_time).total_seconds(),
                    checksum=checksum,
                    record_count=None  # Could be calculated if needed
                )
                
                # Report completion
                if self.progress_reporter:
                    await self.progress_reporter.report_progress(
                        dataset_name=dataset_name,
                        stage=DownloadStage.COMPLETED,
                        bytes_downloaded=final_metrics.bytes_downloaded,
                        total_bytes=final_metrics.total_bytes
                    )
                
                self.completed_downloads.append(result)
                del self.active_downloads[dataset_name]
                
                return result
                
            except Exception as e:
                last_exception = e
                metrics = replace(
                    metrics,
                    error_count=metrics.error_count + 1,
                    retry_count=attempt
                )
                
                logger.warning(
                    "download_attempt_failed",
                    dataset=dataset_name,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                # Clean up partial file on final failure
                if attempt == self.max_retries:
                    if output_path.exists():
                        output_path.unlink()
                    break
                
                # Exponential backoff with jitter
                delay = min(
                    self.base_retry_delay * (2 ** attempt),
                    self.max_retry_delay
                )
                
                logger.info("retry_delay", dataset=dataset_name, delay_seconds=delay)
                await asyncio.sleep(delay)
        
        # All retries failed
        if self.progress_reporter:
            await self.progress_reporter.report_progress(
                dataset_name=dataset_name,
                stage=DownloadStage.FAILED,
                bytes_downloaded=metrics.bytes_downloaded,
                total_bytes=metrics.total_bytes
            )
        
        result = DownloadResult(
            dataset_name=dataset_name,
            dataset_id=config.id,
            success=False,
            error_message=f"Download failed after {self.max_retries + 1} attempts: {last_exception!s}"
        )
        
        self.failed_downloads.append(result)
        del self.active_downloads[dataset_name]
        
        return result
    
    async def download_datasets(
        self,
        dataset_names: Sequence[str],  # Modern sequence type
        max_concurrent: int = 4
    ) -> list[DownloadResult]:
        """
        Download datasets using modern TaskGroup for better error handling.
        
        Uses Python 3.11+ asyncio.TaskGroup for automatic cleanup and
        better exception propagation.
        """
        logger.info(
            "bulk_download_starting",
            dataset_count=len(dataset_names),
            max_concurrent=max_concurrent
        )
        
        async with self.http_client() as client:
            # Use semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def download_with_limit(dataset_name: str) -> DownloadResult:
                async with semaphore:
                    return await self._download_with_exponential_backoff(dataset_name, client)
            
            # For Python 3.11+, use TaskGroup for better error handling
            if sys.version_info >= (3, 11):
                results = []
                try:
                    async with asyncio.TaskGroup() as tg:
                        tasks = [
                            tg.create_task(download_with_limit(name))
                            for name in dataset_names
                        ]
                    
                    results = [task.result() for task in tasks]
                    
                except* Exception as eg:  # Exception groups (Python 3.11+)
                    # Handle exception group
                    logger.error("bulk_download_exceptions", exception_count=len(eg.exceptions))
                    
                    # Log each individual exception
                    for i, exc in enumerate(eg.exceptions):
                        logger.error(f"exception_{i}", error_type=type(exc).__name__, error_message=str(exc))
                    
                    # Still try to get partial results
                    results = []
                    for task in tasks:
                        if task.done() and not task.cancelled():
                            try:
                                results.append(task.result())
                            except Exception as task_exc:
                                logger.error("task_result_error", error=str(task_exc))
                                results.append(DownloadResult(
                                    dataset_name="unknown",
                                    dataset_id="unknown", 
                                    success=False,
                                    error_message=f"Task error: {task_exc}"
                                ))
                        else:
                            # Create failure result for incomplete tasks
                            results.append(DownloadResult(
                                dataset_name="unknown",
                                dataset_id="unknown", 
                                success=False,
                                error_message="Task cancelled or incomplete"
                            ))
            else:
                # Fallback for older Python versions
                tasks = [download_with_limit(name) for name in dataset_names]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Convert exceptions to DownloadResult objects
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        processed_results.append(DownloadResult(
                            dataset_name=dataset_names[i],
                            dataset_id="unknown",
                            success=False,
                            error_message=f"Unexpected error: {result!s}"
                        ))
                    else:
                        processed_results.append(result)
                
                results = processed_results
        
        # Log final summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_mb = sum(r.file_size_bytes or 0 for r in results if r.success) / (1024 * 1024)
        
        logger.info(
            "bulk_download_completed",
            successful=successful,
            failed=failed,
            total_mb=total_mb
        )
        
        return results