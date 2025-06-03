"""
Main CLI entry point for dob-sob platform.

Provides unified command interface for all platform operations.
"""

import click
import asyncio
from pathlib import Path
from typing import List, Optional

from ..core.config import load_config
from ..data.acquisition import NYCODataClient
from ..fraud_detection.algorithms import CommunityDetector
from ..core.models import AlgorithmType, Priority
from .commands.data import data


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.pass_context
def cli(ctx, debug, config):
    """dob-sob: NYC DOB Fraud Detection Platform
    
    Think you can slip fraudulent permits past NYC? Think again, dob-sob.
    """
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        ctx.obj['config'] = load_config(config_file=Path(config))
    else:
        ctx.obj['config'] = load_config()
    
    ctx.obj['debug'] = debug


# Register data command group
cli.add_command(data)


@cli.group()
def fraud():
    """Fraud detection commands"""
    pass


@fraud.command()
@click.option('--algorithm', type=click.Choice(['louvain', 'label_propagation', 'edge_betweenness', 'spectral_clustering']), 
              help='Specific algorithm to run')
@click.option('--all-algorithms', is_flag=True, help='Run all available algorithms')
@click.option('--min-community-size', type=int, default=3, help='Minimum community size')
@click.option('--dataset', help='Dataset name to analyze')
@click.pass_context
def detect(ctx, algorithm, all_algorithms, min_community_size, dataset):
    """Run community detection algorithms for fraud pattern identification"""
    
    if not (algorithm or all_algorithms):
        click.echo("❌ Please specify --algorithm or --all-algorithms")
        return
    
    # This is a simplified version - in practice would load actual data
    click.echo("🕵️ Starting fraud detection analysis...")
    click.echo("📊 Loading NYC DOB data...")
    
    # Placeholder for actual implementation
    click.echo("⚠️  Fraud detection implementation in progress")
    click.echo("💡 Use the legacy scripts for now:")
    click.echo("   python scripts/fraud_detection/community_detection_algorithms.py")


@cli.command()
@click.option('--port', type=int, default=8501, help='Dashboard port')
@click.pass_context
def dashboard(ctx, port):
    """Launch Streamlit dashboard"""
    import subprocess
    
    click.echo(f"🚀 Starting dob-sob dashboard on port {port}...")
    
    try:
        subprocess.run([
            "streamlit", "run", "main.py", 
            "--server.port", str(port),
            "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Failed to start dashboard: {e}")
    except FileNotFoundError:
        click.echo("❌ Streamlit not found. Please install: pip install streamlit")


@cli.command()
def version():
    """Show version information"""
    from .. import __version__
    click.echo(f"dob-sob version {__version__}")
    click.echo("NYC DOB Fraud Detection Platform")
    click.echo("Think you can slip fraudulent permits past NYC? Think again, dob-sob.")


if __name__ == '__main__':
    cli()