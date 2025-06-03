# Modern Python Features & Best Practices

This document outlines the modern Python approaches and latest packages used in the dob-sob project, demonstrating current best practices for 2024.

## 🐍 Python Version Requirements

- **Minimum**: Python 3.12+ (for latest type annotations and performance)
- **Recommended**: Python 3.11+ (for TaskGroup and exception groups)
- **Optimal**: Python 3.12+ (for improved performance and newest features)

## 📦 Modern Package Choices

### HTTP Client: httpx → aiohttp
```python
# ❌ Old: aiohttp (older, more complex)
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.read()

# ✅ Modern: httpx (requests-like API, better async)
async with httpx.AsyncClient() as client:
    response = await client.get(url)
    response.raise_for_status()  # Built-in error handling
```

### UI/Progress: Rich Library
```python
# ❌ Old: Basic click progress
click.echo(f"[{'█' * filled}{'░' * remaining}] {progress:.1f}%")

# ✅ Modern: Rich progress bars with beautiful formatting
from rich.progress import Progress, SpinnerColumn, BarColumn
with Progress() as progress:
    task = progress.add_task("Downloading...", total=total_bytes)
    progress.update(task, advance=chunk_size)
```

### Logging: Structured Logging
```python
# ❌ Old: Basic string logging
logger.info(f"Downloading {dataset_name} size {size_mb}MB")

# ✅ Modern: Structured logging with context
import structlog
logger.info("download_started", dataset=dataset_name, size_mb=size_mb)
```

## 🔧 Modern Language Features

### 1. Built-in Generic Types (Python 3.9+)
```python
# ❌ Old: typing module imports
from typing import Dict, List, Optional
def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:

# ✅ Modern: Built-in generics
def process_data(data: list[dict[str, any]]) -> dict[str, any]:
```

### 2. Union Types with | (Python 3.10+)
```python
# ❌ Old: Union syntax
from typing import Union, Optional
def get_size(path: Union[str, Path]) -> Optional[int]:

# ✅ Modern: | syntax
def get_size(path: str | Path) -> int | None:
```

### 3. Dataclasses with Modern Features
```python
# ❌ Old: Basic dataclass
@dataclass
class DownloadMetrics:
    dataset_name: str
    bytes_downloaded: int = 0

# ✅ Modern: frozen, slots for performance
@dataclass(frozen=True, slots=True)
class DownloadMetrics:
    dataset_name: str
    bytes_downloaded: int = 0
```

### 4. Protocol-Based Interfaces
```python
# ❌ Old: ABC inheritance
from abc import ABC, abstractmethod
class ProgressReporter(ABC):
    @abstractmethod
    def report_progress(self): pass

# ✅ Modern: Protocol for duck typing
from typing import Protocol, runtime_checkable

@runtime_checkable
class ProgressReporter(Protocol):
    async def report_progress(self, **kwargs) -> None: ...
```

### 5. TaskGroup for Better Async (Python 3.11+)
```python
# ❌ Old: asyncio.gather with manual exception handling
try:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Manual exception processing...
except Exception:
    # Handle individually

# ✅ Modern: TaskGroup with automatic cleanup
async with asyncio.TaskGroup() as tg:
    tasks = [tg.create_task(download(name)) for name in names]
# Automatic exception handling and resource cleanup
```

### 6. Exception Groups (Python 3.11+)
```python
# ❌ Old: Individual exception handling
for task in tasks:
    try:
        result = await task
    except Exception as e:
        handle_single_error(e)

# ✅ Modern: Exception groups
try:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(work()) for work in items]
except* NetworkError as eg:
    for error in eg.exceptions:
        handle_network_error(error)
except* ValidationError as eg:
    for error in eg.exceptions:
        handle_validation_error(error)
```

### 7. Context Variables for Request Context
```python
# ❌ Old: Manual context passing
async def download(dataset_name: str, request_id: str):
    logger.info(f"[{request_id}] Downloading {dataset_name}")

# ✅ Modern: Context variables
import contextvars
request_id = contextvars.ContextVar('request_id')

async def download(dataset_name: str):
    logger.info("download_started", dataset=dataset_name)  # request_id automatically included
```

## 🏗️ Architectural Patterns

### 1. Async Context Managers
```python
# ✅ Modern: Proper resource management
@asynccontextmanager
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(**config) as client:
        yield client
```

### 2. Async Generators
```python
# ✅ Modern: Memory-efficient streaming
async def read_chunks(file_handle, chunk_size: int) -> AsyncIterator[bytes]:
    while chunk := await file_handle.read(chunk_size):
        yield chunk
```

### 3. Immutable Data Patterns
```python
# ✅ Modern: Immutable updates with dataclasses
@dataclass(frozen=True)
class Metrics:
    bytes_downloaded: int = 0
    
# Update immutably
new_metrics = dataclass.replace(metrics, bytes_downloaded=metrics.bytes_downloaded + 1024)
```

### 4. Modern Error Handling
```python
# ✅ Modern: Specific exception chaining
try:
    response = await client.get(url)
except httpx.RequestError as e:
    raise DownloadError(f"Network error: {e}") from e
except httpx.HTTPStatusError as e:
    raise DownloadError(f"HTTP {e.response.status_code}") from e
```

## 📊 Performance Optimizations

### 1. Slots for Memory Efficiency
```python
@dataclass(slots=True)  # Reduces memory usage by 20-30%
class DownloadMetrics:
    dataset_name: str
    bytes_downloaded: int
```

### 2. Modern HTTP Client Configuration
```python
# ✅ Modern: Optimized HTTP configuration
client_config = {
    "timeout": httpx.Timeout(600),
    "limits": httpx.Limits(
        max_keepalive_connections=50,
        max_connections=100,
        keepalive_expiry=30.0
    )
}
```

### 3. Async File I/O
```python
# ✅ Modern: Non-blocking file operations
async with aiofiles.open(file_path, 'wb') as f:
    async for chunk in response.aiter_bytes():
        await f.write(chunk)
```

## 🎨 Modern CLI and UI

### 1. Rich Console Output
```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Beautiful tables
table = Table(title="Datasets")
table.add_column("Name", style="cyan")
table.add_row("Housing Violations", "50MB")
console.print(table)

# Panels and styling
console.print(Panel("Success!", style="green"))
```

### 2. Structured CLI Commands
```python
@click.group()
def data():
    """Data commands with modern Click patterns"""

@data.command()
@click.option('--timeout', type=int, default=600)
@click.pass_context
def download(ctx, timeout):
    """Modern command with proper context passing"""
```

## 🔍 Type Safety and Validation

### 1. Runtime Type Checking
```python
from typing import runtime_checkable, Protocol

@runtime_checkable
class DownloadProtocol(Protocol):
    async def download(self) -> bool: ...

# Runtime checking
if isinstance(downloader, DownloadProtocol):
    await downloader.download()
```

### 2. Pydantic v2 Integration
```python
from pydantic import BaseModel, Field, field_validator

class DownloadConfig(BaseModel):
    timeout: int = Field(gt=0, le=3600)
    max_retries: int = Field(ge=1, le=10)
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v < 30:
            raise ValueError('Timeout too short for large files')
        return v
```

## 📈 Observability and Monitoring

### 1. Structured Logging with Context
```python
import structlog

logger = structlog.get_logger()

# Automatic context injection
logger.info("download_completed", 
           dataset=name, 
           bytes=size, 
           duration_ms=elapsed)
```

### 2. Rich Error Display
```python
from rich.console import Console

console = Console()

try:
    await download()
except Exception:
    console.print_exception()  # Beautiful exception formatting
```

## 🚀 Getting Started with Modern Features

1. **Update Python**: Use Python 3.12+ for best experience
2. **Install modern packages**: `uv add httpx rich structlog`
3. **Enable type checking**: Use `mypy` with strict mode
4. **Use modern patterns**: Start with async context managers and protocols
5. **Add observability**: Implement structured logging from the start

## 📚 Key Benefits

- **Performance**: 20-30% memory reduction with slots, faster HTTP with httpx
- **Developer Experience**: Rich UI, better error messages, type safety
- **Maintainability**: Protocol-based interfaces, immutable data patterns
- **Observability**: Structured logging, beautiful error display
- **Future-Proof**: Uses latest Python features and modern async patterns

This modern approach makes the codebase more maintainable, performant, and enjoyable to work with while following current Python best practices.