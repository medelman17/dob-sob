# LLM Response Caching & Backup Strategy for Graphiti NYC DOB Data

## Executive Summary

Given the potential $561-$2,835 cost for full 2.2GB data ingestion into Graphiti, implementing robust caching and backup strategies is **mission-critical**. This document outlines multiple defensive layers to protect against re-processing costs during development.

## 🎯 Core Problem Statement

**The Challenge**: At scale, Graphiti ingestion costs make it economically unfeasible to reprocess data multiple times during development, debugging, or database changes.

**The Solution**: Multi-layered caching and backup architecture to ensure we only pay once for LLM processing.

## 🏗️ Architecture Overview

```
Raw Data (2.2GB CSV)
    ↓
┌─────────────────────────┐
│ LLM Response Cache      │ ← Primary Protection Layer
│ (Hot Storage)           │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Processed Response      │ ← Backup & Version Control
│ Archive (Cold Storage)  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Neo4j Database          │ ← Final Destination
│ (Can be rebuilt)        │
└─────────────────────────┘
```

## 🛡️ Layer 1: LLM Response Caching

### 1.1 Request-Level Caching

**Strategy**: Cache individual LLM requests and responses before they reach Graphiti

```python
# Pseudo-implementation
@cache_llm_response(ttl=None)  # Never expire
async def get_llm_extraction(text_chunk: str, schema: dict) -> dict:
    cache_key = hash(text_chunk + str(schema) + model_version)

    if cached_response := cache.get(cache_key):
        return cached_response

    response = await llm_client.extract_entities(text_chunk, schema)
    cache.set(cache_key, response, ttl=None)
    return response
```

**Cache Key Strategy**:

- `hash(text_content + schema_version + model_name + prompt_version)`
- Ensures exact same input always returns cached result
- Invalidates cache only when inputs change

**Storage Options**:

1. **Redis** (Recommended for development)
   - Fast access
   - Persistent storage with RDB snapshots
   - Easy to backup/restore
2. **File-based cache** (Fallback)
   - JSON files organized by date/batch
   - Easy to inspect and debug
   - Works without additional infrastructure

### 1.2 Graphiti-Specific Caching

**Strategy**: Intercept and cache Graphiti's internal LLM calls

```python
# Custom Graphiti extension
class CachedGraphitiClient(GraphitiClient):
    def __init__(self, cache_backend='redis'):
        super().__init__()
        self.cache = self._setup_cache(cache_backend)

    async def add_nodes(self, nodes_data):
        # Cache the LLM processing step
        cache_key = self._generate_cache_key(nodes_data)

        if cached_result := self.cache.get(cache_key):
            return self._restore_from_cache(cached_result)

        result = await super().add_nodes(nodes_data)
        self.cache.set(cache_key, self._serialize_result(result))
        return result
```

**What to Cache**:

- Entity extraction results
- Relationship mapping outputs
- Schema normalization decisions
- Embedding computations (if not using external service)

## 🗄️ Layer 2: Structured Response Archives

### 2.1 Incremental Processing Archives

**Strategy**: Save processed chunks in structured format for replay

```
data_cache/
├── 2025-01-09_batch_001/
│   ├── metadata.json
│   ├── dob_violations/
│   │   ├── chunk_000001.json
│   │   ├── chunk_000002.json
│   │   └── ...
│   └── job_applications/
│       ├── chunk_000001.json
│       └── ...
└── 2025-01-09_batch_002/
    └── ...
```

**Chunk Format**:

```json
{
  "input": {
    "text": "original text chunk",
    "schema": "schema used",
    "model": "gpt-4",
    "prompt_version": "v1.2"
  },
  "output": {
    "entities": [...],
    "relationships": [...],
    "confidence_scores": {...}
  },
  "metadata": {
    "timestamp": "2025-01-09T10:30:00Z",
    "processing_time_ms": 1250,
    "token_count": {"input": 800, "output": 300},
    "cost_usd": 0.045
  }
}
```

### 2.2 Database State Snapshots

**Strategy**: Regular Neo4j database backups at key milestones

```bash
# Automated backup script
backup_neo4j() {
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_name="graphiti_backup_${timestamp}"

    # Neo4j admin backup
    neo4j-admin database backup neo4j \
        --to-path=/backups/${backup_name} \
        --verbose

    # Compress and store
    tar -czf /backups/${backup_name}.tar.gz /backups/${backup_name}/

    # Upload to cloud storage
    gsutil cp /backups/${backup_name}.tar.gz gs://nycdob-graphiti-backups/
}
```

**Backup Triggers**:

- Before major processing runs
- After successful dataset completion
- Before schema changes
- Daily during active development

## 💾 Layer 3: Cloud Storage Integration

### 3.1 Multi-Cloud Backup Strategy

**Primary Storage**: Google Cloud Storage (project location)
**Secondary Storage**: AWS S3 (different provider)
**Tertiary Storage**: Local NAS/external drives

```python
class MultiCloudBackup:
    def __init__(self):
        self.gcs_client = storage.Client()
        self.s3_client = boto3.client('s3')

    async def backup_cache(self, cache_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Compress cache
        archive_name = f"graphiti_cache_{timestamp}.tar.gz"
        subprocess.run(["tar", "-czf", archive_name, cache_dir])

        # Upload to multiple locations
        await asyncio.gather(
            self._upload_to_gcs(archive_name),
            self._upload_to_s3(archive_name),
            self._copy_to_local_backup(archive_name)
        )
```

### 3.2 Version Control for Configurations

**Strategy**: Git-based tracking of all processing configurations

```
graphiti_configs/
├── schemas/
│   ├── v1.0_initial.json
│   ├── v1.1_refined.json
│   └── v2.0_production.json
├── prompts/
│   ├── entity_extraction_v1.txt
│   └── relationship_mapping_v1.txt
├── processing_batches/
│   ├── 2025-01-09_initial_run.yaml
│   └── 2025-01-10_refined_run.yaml
└── cache_manifests/
    ├── batch_001_manifest.json
    └── batch_002_manifest.json
```

## 🔄 Layer 4: Incremental Processing Framework

### 4.1 Resumable Processing

**Strategy**: Design processing to be resumable from any point

```python
class ResumableGraphitiProcessor:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.processed_chunks = self._load_checkpoint()

    async def process_dataset(self, dataset_path: str):
        chunks = self._chunk_dataset(dataset_path)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{dataset_path}_{i:06d}"

            if chunk_id in self.processed_chunks:
                continue  # Skip already processed

            result = await self._process_chunk_with_cache(chunk)
            self._save_checkpoint(chunk_id, result)

            # Periodic full checkpoint save
            if i % 100 == 0:
                self._save_full_checkpoint()

    def _save_checkpoint(self, chunk_id: str, result: dict):
        checkpoint_file = f"{self.checkpoint_dir}/{chunk_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(result, f)
        self.processed_chunks.add(chunk_id)
```

### 4.2 Smart Dataset Partitioning

**Strategy**: Process data in logical, independent batches

```python
# Partition strategy
DATASET_PARTITIONS = {
    "multiple_dwelling_registrations": {
        "size_mb": 20,
        "partitions": 1,
        "priority": "high"  # Start with smallest
    },
    "dob_violations": {
        "size_mb": 668,
        "partitions": 10,  # 67MB chunks
        "priority": "medium"
    },
    "job_application_filings": {
        "size_mb": 1500,
        "partitions": 20,  # 75MB chunks
        "priority": "low"
    }
}
```

## 🚀 Implementation Roadmap

### Phase 1: Basic Caching (Week 1)

- [ ] Implement Redis-based LLM response cache
- [ ] Create cache key generation strategy
- [ ] Build file-based fallback cache
- [ ] Add cache hit/miss metrics

### Phase 2: Response Archives (Week 2)

- [ ] Design structured response format
- [ ] Implement incremental chunk processing
- [ ] Create backup/restore scripts
- [ ] Set up cloud storage integration

### Phase 3: Database Protection (Week 3)

- [ ] Automate Neo4j backup process
- [ ] Implement resumable processing
- [ ] Create restore verification tests
- [ ] Add monitoring and alerting

### Phase 4: Advanced Features (Week 4)

- [ ] Multi-cloud backup strategy
- [ ] Configuration version control
- [ ] Performance optimization
- [ ] Cost monitoring dashboard

## 💰 Cost Savings Analysis

### Scenario: Development Iteration

**Without Caching**:

- 5 development iterations × $561 = $2,805
- 5 iterations × $2,835 (GPT-4) = $14,175

**With Full Caching**:

- First run: $561 - $2,835
- Subsequent runs: $0 (cache hits)
- **Total savings: 80-95%**

### Scenario: Partial Dataset Changes

**Without Caching**:

- Reprocess entire 2.2GB for small changes

**With Incremental Caching**:

- Only reprocess changed portions
- Cache hit rate: 70-90%
- **Savings: $400-2,500 per iteration**

## 🔧 Technical Implementation Details

### Cache Storage Sizing

```
Estimated cache size per full run:
- Compressed LLM responses: ~500MB
- Structured archives: ~200MB
- Metadata & indices: ~50MB
- Total per run: ~750MB

For 10 development iterations: ~7.5GB storage needed
```

### Performance Optimization

```python
# Async batch processing with caching
async def process_with_cache_pool(chunks: List[str], pool_size: int = 10):
    semaphore = asyncio.Semaphore(pool_size)

    async def process_chunk_throttled(chunk):
        async with semaphore:
            return await process_chunk_with_cache(chunk)

    results = await asyncio.gather(
        *[process_chunk_throttled(chunk) for chunk in chunks]
    )
    return results
```

## 🚨 Risk Mitigation

### Cache Corruption

- **Solution**: Multiple backup copies, checksums
- **Recovery**: Rebuild from archived responses

### API Key Limits

- **Solution**: Rotate between multiple keys
- **Fallback**: Queue requests for retry

### Storage Failures

- **Solution**: Multi-cloud redundancy
- **Recovery**: Automated failover

### Version Conflicts

- **Solution**: Semantic versioning for all configs
- **Recovery**: Git-based rollback procedures

## 📊 Monitoring & Metrics

### Key Metrics to Track

```python
CACHE_METRICS = {
    "cache_hit_rate": "percentage",
    "cost_saved_usd": "cumulative",
    "processing_time_saved": "seconds",
    "storage_used_gb": "current",
    "backup_status": "boolean",
    "last_backup_timestamp": "datetime"
}
```

### Alerting Thresholds

- Cache hit rate < 60% (investigate caching logic)
- Storage usage > 80% (cleanup old archives)
- Backup age > 24 hours (trigger new backup)
- Processing failure rate > 5% (review error handling)

## 🎯 Success Criteria

### Development Phase

- [ ] Zero re-processing costs after initial run
- [ ] < 5 minute restore time from cache
- [ ] 90%+ cache hit rate for repeated processing
- [ ] Automatic recovery from common failures

### Production Readiness

- [ ] Multi-cloud backup redundancy
- [ ] Automated monitoring and alerting
- [ ] Complete disaster recovery procedures
- [ ] Performance optimization for large datasets

## 📞 Emergency Procedures

### Cache Loss Recovery

1. **Stop all processing immediately**
2. **Assess backup integrity**
3. **Restore from most recent valid backup**
4. **Resume from last checkpoint**
5. **Document lessons learned**

### Budget Overrun Protection

1. **Set up billing alerts** at 50%, 75%, 90% of budget
2. **Implement circuit breakers** to halt processing
3. **Require manual approval** for high-cost operations
4. **Regular cost monitoring** and projection updates

---

## Conclusion

This multi-layered caching and backup strategy ensures that the significant investment in LLM processing for Graphiti ingestion is protected against common development scenarios. By implementing comprehensive caching at multiple levels, we can iterate rapidly without fear of cost multiplication.

The key is **defense in depth** - if one layer fails, others provide protection. With proper implementation, we should only need to pay the full ingestion cost once, regardless of how many development iterations are required.

**Estimated Total Protection**: 80-95% cost savings on subsequent runs
**Implementation Time**: 2-4 weeks  
**Infrastructure Cost**: <2% of LLM processing costs
**Risk Reduction**: Near-elimination of re-processing costs
