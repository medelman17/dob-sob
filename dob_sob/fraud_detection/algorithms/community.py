"""
Community detection algorithms for NYC DOB fraud detection.

Implements multiple community detection algorithms optimized for
identifying fraud patterns in building permit and violation networks.
"""

import logging
import asyncio
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading

from ...core.config import get_settings
from ...core.models import CommunityDetectionResult, AlgorithmType, RiskLevel
from ...core.exceptions import CommunityDetectionError

logger = logging.getLogger(__name__)


class CommunityDetector:
    """
    Advanced community detection for fraud pattern identification.
    
    Implements multiple algorithms to identify suspicious communities
    in NYC DOB data networks.
    """
    
    def __init__(self):
        """Initialize community detector."""
        self.execution_stats: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def _build_graph_from_data(self, data: List[Dict[str, Any]], relationship_fields: List[str]) -> nx.Graph:
        """
        Build NetworkX graph from tabular data.
        
        Args:
            data: List of records
            relationship_fields: Fields to use for creating edges
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        for record in data:
            # Create nodes for each entity mentioned in relationship fields
            entities = []
            for field in relationship_fields:
                if field in record and record[field]:
                    entity_id = str(record[field])
                    entities.append(entity_id)
                    
                    # Add node with attributes
                    if not G.has_node(entity_id):
                        G.add_node(entity_id, **{
                            f"{field}_data": record.get(field),
                            "record_count": 1
                        })
                    else:
                        # Increment record count
                        G.nodes[entity_id]["record_count"] = G.nodes[entity_id].get("record_count", 0) + 1
            
            # Create edges between entities in the same record
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    if G.has_edge(entity1, entity2):
                        G[entity1][entity2]["weight"] += 1
                    else:
                        G.add_edge(entity1, entity2, weight=1)
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _louvain_algorithm(self, G: nx.Graph) -> Dict[str, List[str]]:
        """
        Louvain community detection algorithm.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping community_id to list of node_ids
        """
        try:
            import community as community_louvain
        except ImportError:
            logger.warning("python-louvain not available, using greedy modularity")
            return self._greedy_modularity(G)
        
        partition = community_louvain.best_partition(G)
        
        # Convert to community -> nodes mapping
        communities = {}
        for node, community_id in partition.items():
            community_key = f"louvain_{community_id}"
            if community_key not in communities:
                communities[community_key] = []
            communities[community_key].append(node)
        
        return communities
    
    def _label_propagation(self, G: nx.Graph) -> Dict[str, List[str]]:
        """
        Label propagation algorithm.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping community_id to list of node_ids
        """
        communities_generator = nx.algorithms.community.label_propagation_communities(G)
        
        communities = {}
        for i, community in enumerate(communities_generator):
            community_key = f"label_prop_{i}"
            communities[community_key] = list(community)
        
        return communities
    
    def _edge_betweenness(self, G: nx.Graph) -> Dict[str, List[str]]:
        """
        Edge betweenness centrality algorithm.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping community_id to list of node_ids
        """
        # Use Girvan-Newman algorithm
        communities_generator = nx.algorithms.community.girvan_newman(G)
        
        try:
            # Get the first level of division
            first_level = next(communities_generator)
            
            communities = {}
            for i, community in enumerate(first_level):
                community_key = f"edge_betweenness_{i}"
                communities[community_key] = list(community)
            
            return communities
            
        except StopIteration:
            # No communities found
            return {"edge_betweenness_0": list(G.nodes())}
    
    def _greedy_modularity(self, G: nx.Graph) -> Dict[str, List[str]]:
        """
        Greedy modularity maximization algorithm.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping community_id to list of node_ids
        """
        communities_generator = nx.algorithms.community.greedy_modularity_communities(G)
        
        communities = {}
        for i, community in enumerate(communities_generator):
            community_key = f"greedy_mod_{i}"
            communities[community_key] = list(community)
        
        return communities
    
    def _spectral_clustering(self, G: nx.Graph, n_clusters: int = None) -> Dict[str, List[str]]:
        """
        Spectral clustering algorithm.
        
        Args:
            G: NetworkX graph
            n_clusters: Number of clusters (auto-detect if None)
            
        Returns:
            Dictionary mapping community_id to list of node_ids
        """
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            logger.warning("scikit-learn not available, falling back to greedy modularity")
            return self._greedy_modularity(G)
        
        if G.number_of_nodes() < 2:
            return {"spectral_0": list(G.nodes())}
        
        # Convert to adjacency matrix
        adj_matrix = nx.adjacency_matrix(G)
        
        # Auto-detect number of clusters if not specified
        if n_clusters is None:
            n_clusters = min(10, max(2, G.number_of_nodes() // 10))
        
        # Apply spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        try:
            labels = clustering.fit_predict(adj_matrix)
        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}, falling back to greedy modularity")
            return self._greedy_modularity(G)
        
        # Convert to community mapping
        communities = {}
        nodes = list(G.nodes())
        
        for i, label in enumerate(labels):
            community_key = f"spectral_{label}"
            if community_key not in communities:
                communities[community_key] = []
            communities[community_key].append(nodes[i])
        
        return communities
    
    def _infomap_algorithm(self, G: nx.Graph) -> Dict[str, List[str]]:
        """
        InfoMap algorithm (placeholder - requires infomap package).
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping community_id to list of node_ids
        """
        # InfoMap requires special installation, fall back to other methods
        logger.warning("InfoMap not implemented, using label propagation instead")
        return self._label_propagation(G)
    
    def detect_communities(
        self,
        data: List[Dict[str, Any]],
        algorithm: AlgorithmType,
        relationship_fields: List[str],
        min_community_size: int = None
    ) -> CommunityDetectionResult:
        """
        Detect communities using specified algorithm.
        
        Args:
            data: Dataset records
            algorithm: Algorithm to use
            relationship_fields: Fields that define relationships
            min_community_size: Minimum size for a valid community
            
        Returns:
            CommunityDetectionResult with detected communities
        """
        if min_community_size is None:
            settings = get_settings()
            min_community_size = settings.fraud_detection.min_community_size
        
        start_time = datetime.now()
        
        try:
            # Build graph from data
            G = self._build_graph_from_data(data, relationship_fields)
            
            if G.number_of_nodes() == 0:
                logger.warning("Empty graph, no communities to detect")
                return CommunityDetectionResult(
                    algorithm=algorithm,
                    dataset_name="unknown",
                    execution_time_seconds=0,
                    total_communities=0,
                    largest_community_size=0
                )
            
            # Apply selected algorithm
            if algorithm == AlgorithmType.LOUVAIN:
                communities = self._louvain_algorithm(G)
            elif algorithm == AlgorithmType.LABEL_PROPAGATION:
                communities = self._label_propagation(G)
            elif algorithm == AlgorithmType.EDGE_BETWEENNESS:
                communities = self._edge_betweenness(G)
            elif algorithm == AlgorithmType.SPECTRAL_CLUSTERING:
                communities = self._spectral_clustering(G)
            elif algorithm == AlgorithmType.INFOMAP:
                communities = self._infomap_algorithm(G)
            else:
                raise CommunityDetectionError(
                    algorithm.value,
                    "unknown",
                    f"Unsupported algorithm: {algorithm.value}"
                )
            
            # Filter by minimum size
            filtered_communities = {
                comm_id: nodes for comm_id, nodes in communities.items()
                if len(nodes) >= min_community_size
            }
            
            # Calculate statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            total_communities = len(filtered_communities)
            largest_community_size = max((len(nodes) for nodes in filtered_communities.values()), default=0)
            
            # Calculate modularity if possible
            modularity_score = None
            try:
                if total_communities > 1:
                    # Create partition for modularity calculation
                    partition = {}
                    for comm_id, nodes in filtered_communities.items():
                        for node in nodes:
                            partition[node] = comm_id
                    
                    modularity_score = nx.algorithms.community.modularity(G, filtered_communities.values())
            except Exception as e:
                logger.warning(f"Could not calculate modularity: {e}")
            
            # Identify suspicious communities (placeholder logic)
            suspicious_communities = []
            for comm_id, nodes in filtered_communities.items():
                # Simple heuristic: large communities are more suspicious
                if len(nodes) > largest_community_size * 0.8:
                    suspicious_communities.append(comm_id)
            
            logger.info(
                f"Community detection completed: {total_communities} communities found "
                f"in {execution_time:.2f}s using {algorithm.value}"
            )
            
            return CommunityDetectionResult(
                algorithm=algorithm,
                dataset_name="unknown",  # Should be passed from caller
                execution_time_seconds=execution_time,
                total_communities=total_communities,
                largest_community_size=largest_community_size,
                modularity_score=modularity_score,
                communities=filtered_communities,
                suspicious_communities=suspicious_communities
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Community detection failed: {str(e)}")
            raise CommunityDetectionError(algorithm.value, "unknown", str(e))
    
    async def detect_communities_async(
        self,
        data: List[Dict[str, Any]],
        algorithm: AlgorithmType,
        relationship_fields: List[str],
        min_community_size: int = None
    ) -> CommunityDetectionResult:
        """
        Async wrapper for community detection.
        
        Args:
            data: Dataset records
            algorithm: Algorithm to use
            relationship_fields: Fields that define relationships
            min_community_size: Minimum size for a valid community
            
        Returns:
            CommunityDetectionResult with detected communities
        """
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self.detect_communities,
                data,
                algorithm,
                relationship_fields,
                min_community_size
            )
        
        return result
    
    async def run_all_algorithms(
        self,
        data: List[Dict[str, Any]],
        relationship_fields: List[str],
        min_community_size: int = None
    ) -> List[CommunityDetectionResult]:
        """
        Run all available community detection algorithms.
        
        Args:
            data: Dataset records
            relationship_fields: Fields that define relationships
            min_community_size: Minimum size for a valid community
            
        Returns:
            List of CommunityDetectionResult for each algorithm
        """
        algorithms = [
            AlgorithmType.LOUVAIN,
            AlgorithmType.LABEL_PROPAGATION,
            AlgorithmType.EDGE_BETWEENNESS,
            AlgorithmType.SPECTRAL_CLUSTERING
        ]
        
        tasks = [
            self.detect_communities_async(data, algorithm, relationship_fields, min_community_size)
            for algorithm in algorithms
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Algorithm {algorithms[i].value} failed: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            return self.execution_stats.copy()
    
    def clear_stats(self) -> None:
        """Clear execution statistics."""
        with self._lock:
            self.execution_stats.clear()