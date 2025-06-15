// ============================================================================
// NYC DOB Fraud Detection - Core Query Library
// ============================================================================
// These queries implement the three primary fraud detection patterns:
// 1. Shell company detection via address/entity matching
// 2. Cross-property work analysis via ownership networks
// 3. Contractor community detection via collaboration patterns
// ============================================================================

// ============================================================================
// 1. SHELL COMPANY DETECTION QUERIES
// ============================================================================

// Query 1A: Find owners with identical addresses (exact matches)
// Purpose: Detect obvious shell companies using same address
MATCH (o1:Owner)-[:OWNS]->(p1:Property)
MATCH (o2:Owner)-[:OWNS]->(p2:Property)
WHERE o1 <> o2 
  AND o1.normalized_address = o2.normalized_address
  AND o1.normalized_address IS NOT NULL
RETURN o1.name as owner1, 
       o2.name as owner2,
       o1.address as shared_address,
       collect(DISTINCT p1.address) as properties1,
       collect(DISTINCT p2.address) as properties2,
       count(DISTINCT p1) + count(DISTINCT p2) as total_properties
ORDER BY total_properties DESC;

// Query 1B: Find owners with similar addresses (fuzzy matching)
// Purpose: Detect shell companies with slight address variations
MATCH (o1:Owner)-[:OWNS]->(p1:Property)
MATCH (o2:Owner)-[:OWNS]->(p2:Property)
WHERE o1 <> o2 
  AND o1.normalized_address <> o2.normalized_address
  AND apoc.text.distance(o1.normalized_address, o2.normalized_address) <= 3
  AND o1.normalized_address IS NOT NULL
  AND o2.normalized_address IS NOT NULL
WITH o1, o2, p1, p2, 
     apoc.text.distance(o1.normalized_address, o2.normalized_address) as edit_distance
RETURN o1.name as owner1,
       o2.name as owner2, 
       o1.address as address1,
       o2.address as address2,
       edit_distance,
       collect(DISTINCT p1.address) as properties1,
       collect(DISTINCT p2.address) as properties2
ORDER BY edit_distance ASC, owner1;

// Query 1C: Find owners with similar names and any address overlap
// Purpose: Detect shell companies with name variations
MATCH (o1:Owner)-[:OWNS]->(p1:Property)
MATCH (o2:Owner)-[:OWNS]->(p2:Property)
WHERE o1 <> o2
  AND (apoc.text.distance(o1.normalized_name, o2.normalized_name) <= 2
       OR o1.normalized_address = o2.normalized_address)
  AND o1.normalized_name IS NOT NULL
  AND o2.normalized_name IS NOT NULL
RETURN o1.name as owner1,
       o2.name as owner2,
       apoc.text.distance(o1.normalized_name, o2.normalized_name) as name_distance,
       o1.address as address1,
       o2.address as address2,
       collect(DISTINCT p1.bin) as bins1,
       collect(DISTINCT p2.bin) as bins2
ORDER BY name_distance ASC;

// ============================================================================
// 2. CROSS-PROPERTY WORK ANALYSIS QUERIES  
// ============================================================================

// Query 2A: Find all work and violations across commonly-owned properties
// Purpose: Analyze patterns across properties with shared ownership
MATCH (owner:Owner)-[:OWNS]->(prop:Property)
WITH owner, collect(prop) as properties
WHERE size(properties) >= 2  // Only owners with multiple properties
UNWIND properties as prop
OPTIONAL MATCH (prop)<-[:PERFORMED_AT]-(job:Job)<-[:WORKED_ON]-(contractor:Contractor)
OPTIONAL MATCH (prop)<-[:ISSUED_FOR]-(violation:Violation)
RETURN owner.name as owner_name,
       owner.address as owner_address,
       size(properties) as property_count,
       collect(DISTINCT prop.address) as property_addresses,
       collect(DISTINCT contractor.business_name) as contractors_used,
       count(DISTINCT job) as total_jobs,
       count(DISTINCT violation) as total_violations,
       collect(DISTINCT violation.violation_type) as violation_types
ORDER BY property_count DESC, total_jobs DESC;

// Query 2B: Find suspicious work patterns across shell company networks
// Purpose: Combine shell company detection with work pattern analysis
MATCH (o1:Owner)-[:OWNS]->(p1:Property)
MATCH (o2:Owner)-[:OWNS]->(p2:Property)
WHERE o1 <> o2 
  AND (o1.normalized_address = o2.normalized_address 
       OR apoc.text.distance(o1.normalized_address, o2.normalized_address) <= 2)
WITH [o1, o2] as related_owners, [p1, p2] as related_properties
UNWIND related_owners as owner
UNWIND related_properties as prop
MATCH (owner)-[:OWNS]->(prop)
OPTIONAL MATCH (prop)<-[:PERFORMED_AT]-(job:Job)<-[:WORKED_ON]-(contractor:Contractor)
OPTIONAL MATCH (prop)<-[:ISSUED_FOR]-(violation:Violation)
RETURN related_owners[0].name as owner1,
       related_owners[1].name as owner2,
       collect(DISTINCT prop.address) as all_properties,
       collect(DISTINCT contractor.business_name) as all_contractors,
       count(DISTINCT job) as total_jobs,
       count(DISTINCT violation) as total_violations
ORDER BY total_jobs DESC;

// Query 2C: Timeline analysis of work across related properties
// Purpose: Detect coordinated timing patterns across ownership networks
MATCH (owner:Owner)-[:OWNS]->(prop:Property)
WITH owner, collect(prop) as properties
WHERE size(properties) >= 2
UNWIND properties as prop
MATCH (prop)<-[:PERFORMED_AT]-(job:Job)
WHERE job.filing_date IS NOT NULL
RETURN owner.name as owner_name,
       collect(DISTINCT prop.address) as properties,
       collect({
         property: prop.address,
         job_type: job.job_type,
         filing_date: job.filing_date,
         status: job.status
       }) as job_timeline
ORDER BY owner.name;

// ============================================================================
// 3. CONTRACTOR NETWORK ANALYSIS QUERIES
// ============================================================================

// Query 3A: Find contractor communities working on related properties
// Purpose: Detect networks of contractors collaborating across properties
MATCH (c1:Contractor)-[:WORKED_ON]->(j1:Job)-[:PERFORMED_AT]->(p:Property)<-[:OWNS]-(owner:Owner)
MATCH (c2:Contractor)-[:WORKED_ON]->(j2:Job)-[:PERFORMED_AT]->(p)
WHERE c1 <> c2
WITH c1, c2, count(*) as collaboration_count, collect(DISTINCT p.address) as shared_properties
WHERE collaboration_count >= 2  // Minimum collaboration threshold
RETURN c1.business_name as contractor1,
       c2.business_name as contractor2,
       collaboration_count,
       shared_properties,
       size(shared_properties) as property_count
ORDER BY collaboration_count DESC, property_count DESC;

// Query 3B: Find contractor networks across shell company properties
// Purpose: Detect contractors working across suspected shell company networks
MATCH (o1:Owner)-[:OWNS]->(p1:Property)
MATCH (o2:Owner)-[:OWNS]->(p2:Property)
WHERE o1 <> o2 
  AND o1.normalized_address = o2.normalized_address
WITH [o1, o2] as shell_owners, [p1, p2] as shell_properties
UNWIND shell_properties as prop
MATCH (prop)<-[:PERFORMED_AT]-(job:Job)<-[:WORKED_ON]-(contractor:Contractor)
RETURN shell_owners[0].name as owner1,
       shell_owners[1].name as owner2,
       collect(DISTINCT contractor.business_name) as contractors,
       collect(DISTINCT prop.address) as properties,
       count(DISTINCT job) as total_jobs
ORDER BY total_jobs DESC;

// Query 3C: Contractor specialization and geographic patterns
// Purpose: Analyze contractor work patterns for anomaly detection
MATCH (contractor:Contractor)-[:WORKED_ON]->(job:Job)-[:PERFORMED_AT]->(prop:Property)
WITH contractor, 
     collect(DISTINCT job.job_type) as job_types,
     collect(DISTINCT prop.borough) as boroughs,
     collect(DISTINCT prop.zipcode) as zipcodes,
     count(job) as total_jobs
RETURN contractor.business_name as contractor_name,
       contractor.license_type as license_type,
       total_jobs,
       job_types,
       size(job_types) as job_type_diversity,
       boroughs,
       size(boroughs) as borough_diversity,
       zipcodes
ORDER BY total_jobs DESC;

// ============================================================================
// 4. COMBINED FRAUD PATTERN QUERIES
// ============================================================================

// Query 4A: Comprehensive fraud risk assessment
// Purpose: Score entities based on multiple fraud indicators
MATCH (owner:Owner)-[:OWNS]->(prop:Property)
OPTIONAL MATCH (owner2:Owner)-[:OWNS]->(prop2:Property)
WHERE owner <> owner2 
  AND owner.normalized_address = owner2.normalized_address
WITH owner, prop, count(owner2) as address_sharing_count
OPTIONAL MATCH (prop)<-[:PERFORMED_AT]-(job:Job)<-[:WORKED_ON]-(contractor:Contractor)
OPTIONAL MATCH (prop)<-[:ISSUED_FOR]-(violation:Violation)
WITH owner, 
     count(DISTINCT prop) as property_count,
     address_sharing_count,
     count(DISTINCT contractor) as contractor_count,
     count(DISTINCT job) as job_count,
     count(DISTINCT violation) as violation_count
RETURN owner.name as owner_name,
       owner.address as owner_address,
       property_count,
       address_sharing_count,
       contractor_count,
       job_count,
       violation_count,
       // Simple fraud risk score
       (address_sharing_count * 3 + 
        CASE WHEN property_count > 5 THEN 2 ELSE 0 END +
        CASE WHEN violation_count > property_count THEN 2 ELSE 0 END) as fraud_risk_score
ORDER BY fraud_risk_score DESC, property_count DESC;

// Query 4B: Network analysis - Find connected fraud rings
// Purpose: Detect larger networks of connected fraudulent entities
MATCH path = (o1:Owner)-[:OWNS]->(p1:Property)<-[:PERFORMED_AT]-(j:Job)<-[:WORKED_ON]-(c:Contractor)
             -[:WORKED_ON]->(j2:Job)-[:PERFORMED_AT]->(p2:Property)<-[:OWNS]-(o2:Owner)
WHERE o1 <> o2
  AND (o1.normalized_address = o2.normalized_address 
       OR apoc.text.distance(o1.normalized_address, o2.normalized_address) <= 2)
RETURN o1.name as owner1,
       o2.name as owner2,
       c.business_name as connecting_contractor,
       collect(DISTINCT p1.address) as properties1,
       collect(DISTINCT p2.address) as properties2
ORDER BY owner1, owner2; 