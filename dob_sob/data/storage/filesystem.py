"""
Filesystem storage operations for dob-sob platform.

Provides file system operations for managing downloaded datasets
and processed data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class FileSystemStorage:
    """Filesystem storage operations."""
    
    def __init__(self, base_path: Path = None):
        """Initialize filesystem storage."""
        self.base_path = base_path or Path("data")
        logger.info(f"Filesystem storage initialized at {self.base_path}")
    
    def ensure_directories(self):
        """Create necessary directories."""
        directories = [
            self.base_path / "raw",
            self.base_path / "processed", 
            self.base_path / "metadata",
            self.base_path / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)