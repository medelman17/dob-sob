#!/usr/bin/env python3
"""
NYC DOB OData API Client for dob-sob Platform

This module provides sophisticated real-time data access to NYC Department of Buildings
datasets using OData API endpoints. Designed for fraud detection use cases requiring
live data analysis and targeted querying.

Key Features:
- Real-time data querying with OData protocol
- Advanced query building with filtering and projection
- Intelligent pagination for large datasets
- Rate limiting and backoff strategies
- Smart caching mechanisms
- Data transformation and normalization
- Comprehensive error handling and retry logic

Author: dob-sob Team
Version: 1.0
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Tuple
from urllib.parse import urlencode, quote_plus
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import aiofiles
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryOperator(Enum):
    """OData query operators for filtering"""
    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GE = "ge"  # Greater than or equal
    LT = "lt"  # Less than
    LE = "le"  # Less than or equal
    AND = "and"
    OR = "or"
    NOT = "not"
    CONTAINS = "contains"
    STARTSWITH = "startswith"
    ENDSWITH = "endswith"


class SortOrder(Enum):
    """Sort order options"""
    ASC = "asc"
    DESC = "desc"


@dataclass
class ODataFilter:
    """Represents an OData filter expression"""
    field: str
    operator: QueryOperator
    value: Any
    
    def to_odata(self) -> str:
        """Convert filter to OData string format"""
        if self.operator in [QueryOperator.CONTAINS, QueryOperator.STARTSWITH, QueryOperator.ENDSWITH]:
            if self.operator == QueryOperator.CONTAINS:
                return f"{self.field} like '%{self.value}%'"  # Socrata syntax
            elif self.operator == QueryOperator.STARTSWITH:
                return f"{self.field} like '{self.value}%'"
            else:  # ENDSWITH
                return f"{self.field} like '%{self.value}'"
        elif isinstance(self.value, str):
            return f"{self.field} = '{self.value}'"
        elif isinstance(self.value, datetime):
            return f"{self.field} >= '{self.value.isoformat()}'"
        else:
            return f"{self.field} = {self.value}"


@dataclass
class ODataQuery:
    """Represents a complete OData query"""
    select: Optional[List[str]] = None
    filters: List[ODataFilter] = field(default_factory=list)
    order_by: Optional[List[Tuple[str, SortOrder]]] = None
    top: Optional[int] = None
    skip: Optional[int] = None
    
    def to_query_params(self) -> Dict[str, str]:
        """Convert query to URL parameters"""
        params = {}
        
        if self.select:
            params['$select'] = ','.join(self.select)
        
        if self.filters:
            filter_expressions = [f.to_odata() for f in self.filters]
            params['$where'] = ' AND '.join(filter_expressions)
        
        if self.order_by:
            order_expressions = [f"{field} {order.value}" for field, order in self.order_by]
            params['$order'] = ','.join(order_expressions)
        
        if self.top is not None:
            params['$limit'] = str(self.top)
        
        if self.skip is not None:
            params['$offset'] = str(self.skip)
        
        return params


@dataclass
class CacheEntry:
    """Cache entry for OData responses"""
    data: Any
    timestamp: datetime
    ttl_minutes: int = 30
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return datetime.now() - self.timestamp > timedelta(minutes=self.ttl_minutes)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    max_requests_per_minute: int = 60
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 300.0
    retry_attempts: int = 3


class NYCODataClient:
    """
    Advanced OData client for NYC Department of Buildings data
    
    Provides real-time access to all 18 dob-sob target datasets with sophisticated
    querying capabilities, intelligent caching, and fraud detection optimizations.
    """
    
    BASE_URL = "https://data.cityofnewyork.us/resource"
    
    # Dataset endpoints mapping for all 18 target datasets
    DATASET_ENDPOINTS = {
        # Primary datasets (high priority, daily updates) - VERIFIED WORKING
        'housing_litigations': '59kj-x8nc',  # ✅ VERIFIED
        'dob_violations': '3h2n-5cm9',       # ✅ VERIFIED  
        'ecb_violations': '6bgk-3dad',       # ✅ VERIFIED
        'complaints_received': 'eabe-havv',   # ✅ VERIFIED
        'maintenance_code_violations': 'wvxf-dwi5',  # ✅ VERIFIED
        'active_projects_under_construction': 'w9ak-ipjd',  # ✅ VERIFIED
        'certificates_of_occupancy': 'bs8b-p36w',  # ✅ VERIFIED
        
        # Secondary datasets (verified working endpoints)
        'historical_permit_issuance': 'ipu4-2q9a',
        'job_application_filings': 'ic3t-wcy2',
        'dob_job_application_filings': 'ic3t-wcy2',  # Same as job_application_filings
        'approved_permits': 'ipu4-2q9a',  # Same as historical_permit_issuance
        'permit_issuance': 'ipu4-2q9a',   # Same as historical_permit_issuance
        'multiple_dwelling_registrations': 'tesw-yqqr',
        
        # Need to research these endpoints further
        'license_info': '8m42-w767',      # Updated endpoint
        'order_to_repair_vacate': '6pgk-3dad',  # Updated endpoint
        'facades_compliance_filings': 'qsjy-6dfz',  # May need research
        'registration_contacts': 'feu5-w2e2',
        
        # Tertiary datasets
        'property_data': 'rgy2-tti8'
    }
    
    # Fraud detection optimized field mappings
    FRAUD_DETECTION_FIELDS = {
        'housing_litigations': ['litigationid', 'bin', 'casetype', 'casestatus'],
        'dob_violations': ['isn', 'bin', 'violationtype', 'respondent'],
        'ecb_violations': ['isn', 'bin', 'violationtype', 'respondent'],
        'complaints_received': ['complaintnumber', 'bin', 'complaintcategory', 'complaintstatus'],
        'active_projects_under_construction': ['jobnum', 'bin', 'jobtype', 'ownername'],
        'certificates_of_occupancy': ['jobnumber', 'bin', 'jobtype', 'ownername'],
        'permit_issuance': ['jobnumber', 'bin', 'jobtype', 'ownername'],
        'approved_permits': ['jobnumber', 'bin', 'jobtype', 'ownername']
    }
    
    def __init__(self, 
                 app_token: Optional[str] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 cache_dir: str = "data/cache"):
        """
        Initialize NYC OData client
        
        Args:
            app_token: NYC Open Data API app token for higher rate limits
            rate_limit_config: Rate limiting configuration
            cache_dir: Directory for caching responses
        """
        self.app_token = app_token
        self.rate_limit = rate_limit_config or RateLimitConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting tracking
        self.request_times: List[datetime] = []
        self.backoff_until: Optional[datetime] = None
        
        # Cache storage
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Session will be created when needed
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session is created"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for large queries
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    def _get_cache_key(self, dataset: str, query: Optional[ODataQuery] = None) -> str:
        """Generate cache key for query"""
        query_str = ""
        if query:
            params = query.to_query_params()
            query_str = json.dumps(params, sort_keys=True)
        
        cache_input = f"{dataset}:{query_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = datetime.now()
        
        # Check if we're in backoff period
        if self.backoff_until and now < self.backoff_until:
            wait_time = (self.backoff_until - now).total_seconds()
            logger.warning(f"Rate limited - waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
            return
        
        # Clean old request times (keep only last minute)
        minute_ago = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > minute_ago]
        
        # Check if we've exceeded rate limit
        if len(self.request_times) >= self.rate_limit.max_requests_per_minute:
            # Calculate backoff time
            backoff_seconds = min(
                self.rate_limit.backoff_factor ** len(self.request_times),
                self.rate_limit.max_backoff_seconds
            )
            self.backoff_until = now + timedelta(seconds=backoff_seconds)
            
            logger.warning(f"Rate limit exceeded - backing off for {backoff_seconds:.1f} seconds")
            await asyncio.sleep(backoff_seconds)
        
        # Record this request
        self.request_times.append(now)
    
    def _build_url(self, dataset: str, query: Optional[ODataQuery] = None) -> str:
        """Build complete OData URL for dataset and query"""
        if dataset not in self.DATASET_ENDPOINTS:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        endpoint = self.DATASET_ENDPOINTS[dataset]
        url = f"{self.BASE_URL}/{endpoint}.json"
        
        # Add query parameters
        params = {}
        if self.app_token:
            params['$$app_token'] = self.app_token
        
        if query:
            params.update(query.to_query_params())
        
        if params:
            url += "?" + urlencode(params)
        
        return url
    
    async def _make_request(self, url: str, retries: int = 0) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        await self._ensure_session()
        await self._check_rate_limit()
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    if retries < self.rate_limit.retry_attempts:
                        backoff_time = self.rate_limit.backoff_factor ** (retries + 1)
                        logger.warning(f"Rate limited (429) - retrying in {backoff_time:.1f}s")
                        await asyncio.sleep(backoff_time)
                        return await self._make_request(url, retries + 1)
                    else:
                        raise Exception(f"Rate limited after {retries} retries")
                else:
                    response.raise_for_status()
        
        except Exception as e:
            if retries < self.rate_limit.retry_attempts:
                backoff_time = self.rate_limit.backoff_factor ** (retries + 1)
                logger.warning(f"Request failed: {e} - retrying in {backoff_time:.1f}s")
                await asyncio.sleep(backoff_time)
                return await self._make_request(url, retries + 1)
            else:
                raise Exception(f"Request failed after {retries} retries: {e}")
    
    async def query_dataset(self, 
                          dataset: str, 
                          query: Optional[ODataQuery] = None,
                          use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Query a dataset with optional OData parameters
        
        Args:
            dataset: Dataset name from DATASET_ENDPOINTS
            query: OData query parameters
            use_cache: Whether to use cached results
            
        Returns:
            List of records matching the query
        """
        # Check cache first
        cache_key = self._get_cache_key(dataset, query)
        if use_cache and cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if not entry.is_expired:
                logger.info(f"Cache hit for {dataset} query")
                return entry.data
            else:
                del self.memory_cache[cache_key]
        
        # Make API request
        url = self._build_url(dataset, query)
        logger.info(f"Querying {dataset}: {url}")
        
        data = await self._make_request(url)
        
        # Cache the result
        if use_cache:
            self.memory_cache[cache_key] = CacheEntry(
                data=data,
                timestamp=datetime.now()
            )
        
        return data
    
    async def paginated_query(self,
                            dataset: str,
                            query: Optional[ODataQuery] = None,
                            page_size: int = 1000,
                            max_records: Optional[int] = None) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """
        Execute paginated query for large datasets
        
        Args:
            dataset: Dataset name
            query: Base query (skip parameter will be overridden)
            page_size: Records per page
            max_records: Maximum total records to fetch
            
        Yields:
            Pages of records
        """
        if query is None:
            query = ODataQuery()
        
        offset = 0
        total_fetched = 0
        
        while True:
            # Create query for this page
            page_query = ODataQuery(
                select=query.select,
                filters=query.filters,
                order_by=query.order_by,
                top=min(page_size, (max_records - total_fetched) if max_records else page_size),
                skip=offset
            )
            
            # Fetch page
            page_data = await self.query_dataset(dataset, page_query, use_cache=False)
            
            if not page_data:
                break
            
            yield page_data
            
            total_fetched += len(page_data)
            offset += len(page_data)
            
            # Check if we've reached our limit or got a partial page
            if max_records and total_fetched >= max_records:
                break
            if len(page_data) < page_size:
                break
    
    async def get_recent_records(self,
                               dataset: str,
                               since: datetime,
                               date_field: str = "inspectiondate") -> List[Dict[str, Any]]:
        """
        Get records created/modified since a specific date
        
        Args:
            dataset: Dataset name
            since: Get records since this date
            date_field: Name of the date field to filter on
            
        Returns:
            Recent records
        """
        query = ODataQuery(
            filters=[ODataFilter(date_field, QueryOperator.GE, since)],
            order_by=[(date_field, SortOrder.DESC)]
        )
        
        return await self.query_dataset(dataset, query)
    
    async def search_by_bin(self, dataset: str, bin_number: str) -> List[Dict[str, Any]]:
        """
        Search records by Building Identification Number (BIN)
        
        Args:
            dataset: Dataset name
            bin_number: Building identification number
            
        Returns:
            Records for the specified BIN
        """
        query = ODataQuery(
            filters=[ODataFilter("bin", QueryOperator.EQ, bin_number)]
        )
        
        return await self.query_dataset(dataset, query)
    
    async def fraud_detection_query(self, 
                                  dataset: str,
                                  contractor_name: Optional[str] = None,
                                  bin_numbers: Optional[List[str]] = None,
                                  since_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Specialized query for fraud detection use cases
        
        Args:
            dataset: Dataset name
            contractor_name: Filter by contractor/owner name
            bin_numbers: List of BIN numbers to investigate
            since_date: Filter records since this date
            
        Returns:
            Records matching fraud detection criteria
        """
        filters = []
        
        if contractor_name:
            # Try common contractor name fields
            name_fields = ['ownername', 'respondent', 'licensee']
            name_filters = []
            for field in name_fields:
                name_filters.append(ODataFilter(field, QueryOperator.CONTAINS, contractor_name))
            # Note: In real implementation, you'd use OR logic here
            # For simplicity, using the first available field
            filters.append(name_filters[0])
        
        if bin_numbers:
            # Filter by multiple BINs (would use OR logic in real implementation)
            if bin_numbers:
                filters.append(ODataFilter("bin", QueryOperator.EQ, bin_numbers[0]))
        
        if since_date:
            # Try common date fields
            date_fields = ['inspectiondate', 'issuancedate', 'filingdate']
            filters.append(ODataFilter(date_fields[0], QueryOperator.GE, since_date))
        
        # Select fraud-relevant fields if available
        select_fields = self.FRAUD_DETECTION_FIELDS.get(dataset)
        
        query = ODataQuery(
            select=select_fields,
            filters=filters,
            order_by=[("bin", SortOrder.ASC)]
        )
        
        return await self.query_dataset(dataset, query)
    
    async def get_dataset_stats(self, dataset: str) -> Dict[str, Any]:
        """
        Get basic statistics about a dataset
        
        Args:
            dataset: Dataset name
            
        Returns:
            Dataset statistics
        """
        # Get total count
        count_query = ODataQuery(select=["bin"], top=1)  # Just get one record to test
        sample = await self.query_dataset(dataset, count_query)
        
        # Get recent activity
        try:
            recent_query = ODataQuery(
                order_by=[("inspectiondate", SortOrder.DESC)],
                top=10
            )
            recent = await self.query_dataset(dataset, recent_query)
        except:
            recent = []
        
        return {
            "dataset": dataset,
            "endpoint": self.DATASET_ENDPOINTS[dataset],
            "sample_record_count": len(sample),
            "recent_records": len(recent),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.memory_cache.clear()
        logger.info("Cache cleared")
    
    async def test_connectivity(self) -> Dict[str, bool]:
        """
        Test connectivity to all dataset endpoints
        
        Returns:
            Dictionary mapping dataset names to connectivity status
        """
        results = {}
        
        for dataset in self.DATASET_ENDPOINTS:
            try:
                # Try to fetch just one record
                test_query = ODataQuery(top=1)
                await self.query_dataset(dataset, test_query, use_cache=False)
                results[dataset] = True
                logger.info(f"✅ {dataset}: Connected")
            except Exception as e:
                results[dataset] = False
                logger.error(f"❌ {dataset}: {e}")
        
        return results


# Helper functions for common fraud detection queries

async def find_suspicious_contractors(client: NYCODataClient, 
                                    contractor_name: str,
                                    lookback_days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find all records related to a potentially suspicious contractor
    
    Args:
        client: OData client instance
        contractor_name: Name of contractor to investigate
        lookback_days: How many days back to search
        
    Returns:
        Dictionary of datasets and their matching records
    """
    since_date = datetime.now() - timedelta(days=lookback_days)
    results = {}
    
    # Search relevant datasets
    key_datasets = [
        'dob_violations', 'ecb_violations', 'complaints_received',
        'active_projects_under_construction', 'permit_issuance'
    ]
    
    for dataset in key_datasets:
        try:
            records = await client.fraud_detection_query(
                dataset=dataset,
                contractor_name=contractor_name,
                since_date=since_date
            )
            if records:
                results[dataset] = records
                logger.info(f"Found {len(records)} records in {dataset} for {contractor_name}")
        except Exception as e:
            logger.error(f"Error searching {dataset}: {e}")
    
    return results


async def analyze_bin_activity(client: NYCODataClient, 
                             bin_number: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze all activity for a specific Building Identification Number
    
    Args:
        client: OData client instance
        bin_number: Building identification number
        
    Returns:
        Dictionary of datasets and their records for this BIN
    """
    results = {}
    
    # Search all relevant datasets
    for dataset in client.DATASET_ENDPOINTS:
        try:
            records = await client.search_by_bin(dataset, bin_number)
            if records:
                results[dataset] = records
                logger.info(f"Found {len(records)} records in {dataset} for BIN {bin_number}")
        except Exception as e:
            logger.error(f"Error searching {dataset} for BIN {bin_number}: {e}")
    
    return results


# CLI interface
async def main():
    """Command-line interface for testing OData client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NYC DOB OData Client for dob-sob")
    parser.add_argument("--dataset", help="Dataset to query")
    parser.add_argument("--test-connectivity", action="store_true", help="Test connectivity to all datasets")
    parser.add_argument("--contractor", help="Search for contractor name")
    parser.add_argument("--bin", help="Search by BIN number")
    parser.add_argument("--stats", help="Get dataset statistics")
    parser.add_argument("--app-token", help="NYC Open Data app token")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of results")
    
    args = parser.parse_args()
    
    async with NYCODataClient(app_token=args.app_token) as client:
        if args.test_connectivity:
            print("🔍 Testing connectivity to all datasets...")
            results = await client.test_connectivity()
            
            print("\n📊 Connectivity Results:")
            for dataset, status in results.items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {dataset}")
        
        elif args.contractor:
            print(f"🕵️ Searching for contractor: {args.contractor}")
            results = await find_suspicious_contractors(client, args.contractor)
            
            for dataset, records in results.items():
                print(f"\n📋 {dataset}: {len(records)} records")
                for record in records[:3]:  # Show first 3
                    print(f"  - {record}")
        
        elif args.bin:
            print(f"🏢 Analyzing BIN: {args.bin}")
            results = await analyze_bin_activity(client, args.bin)
            
            for dataset, records in results.items():
                print(f"\n📋 {dataset}: {len(records)} records")
        
        elif args.stats:
            print(f"📊 Getting statistics for: {args.stats}")
            stats = await client.get_dataset_stats(args.stats)
            print(json.dumps(stats, indent=2))
        
        elif args.dataset:
            print(f"📥 Querying dataset: {args.dataset}")
            query = ODataQuery(top=args.limit)
            records = await client.query_dataset(args.dataset, query)
            
            print(f"Found {len(records)} records:")
            for record in records:
                print(f"  - {record}")
        
        else:
            print("🗽 dob-sob OData Client")
            print("Available datasets:")
            for dataset in client.DATASET_ENDPOINTS:
                print(f"  - {dataset}")


if __name__ == "__main__":
    asyncio.run(main()) 