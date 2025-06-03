"""
NYC DOB OData API Client

Provides real-time data access to NYC Department of Buildings datasets
using OData API endpoints with intelligent caching and rate limiting.
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
import httpx
import aiofiles
from pathlib import Path

from ...core.config import get_settings
from ...core.exceptions import APIConnectionError, DataValidationError
from ...core.models import APIResponse, BuildingInfo, RiskLevel

logger = logging.getLogger(__name__)


class QueryOperator(Enum):
    """OData query operators for filtering."""
    EQ = "eq"
    NE = "ne" 
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    AND = "and"
    OR = "or"
    NOT = "not"
    CONTAINS = "contains"
    STARTSWITH = "startswith"
    ENDSWITH = "endswith"


class SortOrder(Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


@dataclass
class ODataFilter:
    """Represents an OData filter expression."""
    field: str
    operator: QueryOperator
    value: Any
    
    def to_odata(self) -> str:
        """Convert filter to OData string format."""
        if self.operator in [QueryOperator.CONTAINS, QueryOperator.STARTSWITH, QueryOperator.ENDSWITH]:
            if self.operator == QueryOperator.CONTAINS:
                return f"{self.field} like '%{self.value}%'"
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
    """Represents a complete OData query."""
    select: Optional[List[str]] = None
    filters: List[ODataFilter] = field(default_factory=list)
    order_by: Optional[List[Tuple[str, SortOrder]]] = None
    top: Optional[int] = None
    skip: Optional[int] = None
    
    def to_query_params(self) -> Dict[str, str]:
        """Convert query to URL parameters."""
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
    """Cache entry for OData responses."""
    data: Any
    timestamp: datetime
    ttl_minutes: int = 30
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() - self.timestamp > timedelta(minutes=self.ttl_minutes)


class NYCODataClient:
    """
    Asynchronous client for NYC Open Data OData API.
    
    Provides intelligent querying, caching, and rate limiting for
    real-time fraud detection use cases.
    """
    
    def __init__(self, base_url: str = None):
        """Initialize the OData client."""
        settings = get_settings()
        self.base_url = base_url or settings.api.odata_base_url
        self.client: Optional[httpx.AsyncClient] = None
        self.cache: Dict[str, CacheEntry] = {}
        self.request_times: List[datetime] = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self.client is None:
            settings = get_settings()
            timeout = httpx.Timeout(settings.api.timeout_seconds)
            self.client = httpx.AsyncClient(timeout=timeout)
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    def _get_cache_key(self, dataset_id: str, query: ODataQuery) -> str:
        """Generate cache key for query."""
        query_str = json.dumps(query.to_query_params(), sort_keys=True)
        return hashlib.md5(f"{dataset_id}:{query_str}".encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check if cached result is available and valid."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired:
                logger.debug(f"Cache hit for key: {cache_key}")
                return entry.data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, data: Any, ttl_minutes: int = 30) -> None:
        """Cache query result."""
        self.cache[cache_key] = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl_minutes=ttl_minutes
        )
        logger.debug(f"Cached result for key: {cache_key}")
    
    async def _rate_limit(self) -> None:
        """Implement rate limiting."""
        now = datetime.now()
        
        # Remove old request times (older than 1 minute)
        self.request_times = [
            req_time for req_time in self.request_times 
            if now - req_time < timedelta(minutes=1)
        ]
        
        # Check if we're hitting rate limit
        settings = get_settings()
        if len(self.request_times) >= settings.api.requests_per_second * 60:
            sleep_time = 60 - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(now)
    
    async def query_dataset(
        self,
        dataset_id: str,
        query: ODataQuery,
        use_cache: bool = True,
        cache_ttl_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Query a dataset with OData parameters.
        
        Args:
            dataset_id: NYC Open Data dataset identifier
            query: OData query object
            use_cache: Whether to use caching
            cache_ttl_minutes: Cache TTL in minutes
            
        Returns:
            List of records from the dataset
            
        Raises:
            APIConnectionError: If API request fails
        """
        if not self.client:
            await self.connect()
        
        # Check cache first
        cache_key = self._get_cache_key(dataset_id, query) if use_cache else None
        if use_cache and cache_key:
            cached_result = self._check_cache(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Apply rate limiting
        await self._rate_limit()
        
        # Build URL and query parameters
        url = f"{self.base_url}/{dataset_id}.json"
        params = query.to_query_params()
        
        logger.info(f"Querying dataset {dataset_id} with params: {params}")
        
        try:
            response = await self.client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # Cache the result
                if use_cache and cache_key:
                    self._cache_result(cache_key, data, cache_ttl_minutes)
                
                return data
            else:
                raise APIConnectionError(
                    url=url,
                    status_code=response.status_code,
                    message=f"API request failed: {response.text}"
                )
                    
        except httpx.RequestError as e:
            raise APIConnectionError(url=url, message=f"HTTP client error: {str(e)}")
    
    async def analyze_building_by_bin(self, bin_number: str) -> BuildingInfo:
        """
        Perform instant fraud risk analysis for a building by BIN.
        
        Args:
            bin_number: Building Identification Number
            
        Returns:
            BuildingInfo with risk assessment
        """
        logger.info(f"Analyzing building BIN: {bin_number}")
        
        # Query multiple datasets for comprehensive analysis
        tasks = []
        
        # DOB Violations
        violations_query = ODataQuery(
            filters=[ODataFilter("bin", QueryOperator.EQ, bin_number)],
            top=1000
        )
        tasks.append(self.query_dataset("3h2n-5cm9", violations_query))
        
        # Complaints Received
        complaints_query = ODataQuery(
            filters=[ODataFilter("bin", QueryOperator.EQ, bin_number)],
            top=1000
        )
        tasks.append(self.query_dataset("eabe-havv", complaints_query))
        
        # Execute queries concurrently
        try:
            violations_data, complaints_data = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error querying building data for BIN {bin_number}: {e}")
            violations_data, complaints_data = [], []
        
        # Calculate risk metrics
        violation_count = len(violations_data)
        complaint_count = len(complaints_data)
        
        # Simple risk scoring algorithm
        risk_score = min(1.0, (violation_count * 0.1 + complaint_count * 0.05))
        
        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.2:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Extract building info from first violation/complaint if available
        address = None
        borough = None
        if violations_data:
            first_violation = violations_data[0]
            address = first_violation.get('house_number', '') + ' ' + first_violation.get('street_name', '')
            borough = first_violation.get('boro', '')
        elif complaints_data:
            first_complaint = complaints_data[0]
            address = first_complaint.get('house_number', '') + ' ' + first_complaint.get('street_name', '')
            borough = first_complaint.get('borough', '')
        
        return BuildingInfo(
            bin=bin_number,
            address=address.strip() if address else None,
            borough=borough,
            risk_level=risk_level,
            risk_score=risk_score,
            violation_count=violation_count,
            complaint_count=complaint_count,
            last_activity_date=datetime.now()
        )
    
    async def search_contractors(self, business_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search for contractors by business name.
        
        Args:
            business_name: Contractor business name to search
            limit: Maximum number of results
            
        Returns:
            List of contractor records
        """
        logger.info(f"Searching contractors for: {business_name}")
        
        query = ODataQuery(
            filters=[ODataFilter("applicant_business_name", QueryOperator.CONTAINS, business_name)],
            top=limit,
            order_by=[("filing_date", SortOrder.DESC)]
        )
        
        return await self.query_dataset("ic3t-wcy2", query)  # Job Application Filings
    
    async def get_recent_activity(
        self,
        dataset_id: str,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent activity from a dataset.
        
        Args:
            dataset_id: Dataset to query
            hours: Hours of recent activity to fetch
            limit: Maximum number of records
            
        Returns:
            List of recent records
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Note: Date field varies by dataset, this is a simplified example
        query = ODataQuery(
            filters=[ODataFilter("filing_date", QueryOperator.GE, cutoff_time)],
            top=limit,
            order_by=[("filing_date", SortOrder.DESC)]
        )
        
        return await self.query_dataset(dataset_id, query)
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired)
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_entries,
            "expired_entries": expired_entries
        }