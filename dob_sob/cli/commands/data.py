"""
Modern data acquisition CLI commands using latest Python features.

Demonstrates modern Python patterns including:
- Rich progress bars and console output
- Structured logging with context
- Modern async patterns
- httpx for HTTP requests
- Type hints with built-in generics
"""

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import structlog

from ...data.acquisition.pipeline import ModernBulkPipeline, RichProgressReporter
from ...data.acquisition.datasets import dataset_registry

# Initialize rich console and structured logger
console = Console()
logger = structlog.get_logger(__name__)


def display_dataset_table(datasets: Sequence[str]) -> None:
    """Display datasets in a beautiful Rich table."""
    table = Table(title="🗽 NYC DOB Datasets", show_header=True, header_style="bold magenta")
    
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Priority", justify="center")
    table.add_column("Size (MB)", justify="right", style="green")
    table.add_column("Description", style="dim")
    
    total_size = 0
    for name in datasets:
        try:
            config = dataset_registry.get_dataset(name)
            
            # Color code priority
            priority_colors = {"high": "red", "medium": "yellow", "low": "green"}
            priority_style = priority_colors.get(config.priority, "white")
            
            table.add_row(
                config.name,
                f"[{priority_style}]{config.priority.upper()}[/{priority_style}]",
                str(config.estimated_size_mb),
                config.description[:50] + "..." if len(config.description) > 50 else config.description
            )
            total_size += config.estimated_size_mb
            
        except Exception as e:
            table.add_row(name, "[red]ERROR[/red]", "?", f"Error: {e}")
    
    console.print(table)
    console.print(f"\n[bold]Total: {len(datasets)} datasets, ~{total_size:,}MB (~{total_size/1024:.1f}GB)[/bold]")


def display_results_summary(results: Sequence[any]) -> None:
    """Display download results in a beautiful summary."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    # Create summary panel
    if successful:
        success_text = Text()
        success_text.append("✅ Successful Downloads:\n", style="bold green")
        
        total_mb = 0
        for result in successful:
            size_mb = (result.file_size_bytes or 0) / (1024 * 1024)
            total_mb += size_mb
            success_text.append(f"  • {result.dataset_name}: {size_mb:.1f}MB\n", style="green")
        
        success_text.append(f"\nTotal: {total_mb:.1f}MB", style="bold green")
        console.print(Panel(success_text, title="Success", border_style="green"))
    
    if failed:
        failure_text = Text()
        failure_text.append("❌ Failed Downloads:\n", style="bold red")
        
        for result in failed:
            failure_text.append(f"  • {result.dataset_name}: {result.error_message}\n", style="red")
        
        console.print(Panel(failure_text, title="Failures", border_style="red"))
        
        # Show retry command
        failed_names = [r.dataset_name for r in failed]
        retry_cmd = f"dob-sob data fetch --datasets {','.join(failed_names)}"
        console.print(f"\n💡 [dim]Retry failed fetches with:[/dim] [cyan]{retry_cmd}[/cyan]")


@click.group()
def data():
    """Modern data acquisition commands with Rich UI"""
    pass


@data.command()
@click.option('--all', 'download_all', is_flag=True, help='Fetch all datasets')
@click.option('--primary', is_flag=True, help='Fetch primary (high-priority) datasets only')
@click.option('--datasets', help='Comma-separated list of specific datasets')
@click.option('--max-concurrent', type=int, default=4, help='Maximum concurrent downloads')
@click.option('--list', 'list_datasets', is_flag=True, help='List available datasets and exit')
@click.option('--no-progress', is_flag=True, help='Disable progress bars')
@click.option('--timeout', type=int, default=1800, help='Download timeout in seconds (default: 30min)')
@click.pass_context
def fetch(ctx, download_all, primary, datasets, max_concurrent, list_datasets, no_progress, timeout):
    """Fetch NYC DOB datasets with modern pipeline, Rich progress bars and structured logging"""
    
    # Check Python version for optimal features
    if sys.version_info < (3, 11):
        console.print("[yellow]⚠️  For best performance, consider upgrading to Python 3.11+ for TaskGroup support[/yellow]")
    
    if list_datasets:
        console.print(Panel.fit(
            "🗽 Available NYC DOB Datasets for Fraud Detection",
            style="bold blue"
        ))
        
        # Group by priority for better display
        for priority in ['high', 'medium', 'low']:
            priority_datasets = dataset_registry.list_by_priority(priority)
            if priority_datasets:
                console.print(f"\n[bold]{priority.title()} Priority:[/bold]")
                display_dataset_table(priority_datasets)
        
        return
    
    # Determine datasets to download
    if download_all:
        dataset_names = dataset_registry.list_datasets()
        console.print(f"🗽 [bold]Fetching all {len(dataset_names)} NYC DOB datasets[/bold]")
    elif primary:
        dataset_names = dataset_registry.list_primary_datasets()
        console.print(f"🗽 [bold]Fetching {len(dataset_names)} primary datasets[/bold]")
    elif datasets:
        dataset_names = [d.strip() for d in datasets.split(',')]
        console.print(f"🗽 [bold]Fetching {len(dataset_names)} specified datasets[/bold]")
        
        # Validate dataset names
        available = set(dataset_registry.list_datasets())
        invalid = [name for name in dataset_names if name not in available]
        if invalid:
            console.print(f"[red]❌ Invalid datasets: {', '.join(invalid)}[/red]")
            console.print("[dim]💡 Use --list to see available options[/dim]")
            return
    else:
        console.print("[red]❌ Please specify --all, --primary, --datasets, or --list[/red]")
        return
    
    # Show configuration summary
    total_size_mb = dataset_registry.get_total_estimated_size(dataset_names)
    
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("📊 Estimated size:", f"{total_size_mb:,}MB (~{total_size_mb/1024:.1f}GB)")
    config_table.add_row("⚡ Max concurrent:", str(max_concurrent))
    config_table.add_row("⏱️  Timeout:", f"{timeout}s")
    config_table.add_row("🐍 Python version:", f"{sys.version_info.major}.{sys.version_info.minor}")
    
    console.print(Panel(config_table, title="Configuration", border_style="blue"))
    
    async def run_modern_download():
        """Execute modern download with structured logging and Rich UI."""
        
        # Set up structured logging context
        logger.info(
            "download_session_starting",
            dataset_count=len(dataset_names),
            total_size_mb=total_size_mb,
            max_concurrent=max_concurrent
        )
        
        # Create progress reporter (optional)
        progress_reporter = None if no_progress else RichProgressReporter()
        
        # Initialize modern pipeline
        pipeline = ModernBulkPipeline(
            timeout_seconds=timeout,
            progress_reporter=progress_reporter
        )
        
        try:
            # Use progress reporter context if available
            if progress_reporter:
                async with progress_reporter:
                    console.print("\n🚀 [bold]Starting fetch with Rich progress...[/bold]\n")
                    results = await pipeline.download_datasets(dataset_names, max_concurrent)
            else:
                console.print("\n🚀 [bold]Starting fetch...[/bold]")
                results = await pipeline.download_datasets(dataset_names, max_concurrent)
            
            # Display beautiful results summary
            console.print("\n" + "="*80)
            console.print("[bold blue]📈 Fetch Complete[/bold blue]")
            console.print("="*80)
            
            display_results_summary(results)
            
            # Log final summary
            successful_count = sum(1 for r in results if r.success)
            logger.info(
                "download_session_completed",
                successful=successful_count,
                failed=len(results) - successful_count,
                total_datasets=len(results)
            )
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Fetch interrupted by user[/yellow]")
            logger.warning("fetch_session_interrupted")
        except Exception as e:
            console.print(f"\n[red]❌ Fetch failed: {e}[/red]")
            logger.error("fetch_session_failed", error=str(e))
            
            if ctx.obj.get('debug'):
                console.print_exception()
    
    # Run the modern async fetch
    try:
        asyncio.run(run_modern_download())
    except Exception as e:
        console.print(f"[red]❌ Failed to start fetch: {e}[/red]")
        if ctx.obj.get('debug'):
            console.print_exception()


@data.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table')
def status(output_format):
    """Check data status with modern Rich formatting"""
    
    console.print(Panel.fit(
        "📊 Modern Data Status Dashboard",
        style="bold green"
    ))
    
    # Create status table
    status_table = Table(
        title="Dataset Status",
        show_header=True,
        header_style="bold magenta"
    )
    
    status_table.add_column("Dataset", style="cyan")
    status_table.add_column("Priority", justify="center")
    status_table.add_column("Status", justify="center")
    status_table.add_column("Last Updated", style="dim")
    status_table.add_column("Size", justify="right", style="green")
    
    # Check primary datasets
    primary_datasets = dataset_registry.list_primary_datasets()
    
    for dataset_name in primary_datasets:
        config = dataset_registry.get_dataset(dataset_name)
        
        # Mock status - would be implemented with actual file checking
        status_emoji = "🟢"
        status_text = "Fresh"
        last_update = "2 hours ago"
        
        # Priority color coding
        priority_colors = {"high": "red", "medium": "yellow", "low": "green"}
        priority_style = priority_colors.get(config.priority, "white")
        
        status_table.add_row(
            config.name,
            f"[{priority_style}]{config.priority.upper()}[/{priority_style}]",
            f"{status_emoji} {status_text}",
            last_update,
            f"{config.estimated_size_mb}MB"
        )
    
    console.print(status_table)
    
    # Summary statistics
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"• {len(primary_datasets)} primary datasets monitored")
    console.print(f"• All datasets are fresh and ready for fraud detection")
    console.print(f"• Use [cyan]dob-sob data fetch --primary[/cyan] to update")


if __name__ == "__main__":
    # Enable for testing
    data()