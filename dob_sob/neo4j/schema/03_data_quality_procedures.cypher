// ============================================================================
// NYC DOB Fraud Detection - Data Quality & Normalization Procedures
// ============================================================================
// These procedures handle data quality issues common in NYC datasets:
// 1. Address normalization and standardization
// 2. Entity name normalization and deduplication
// 3. Data validation and cleanup
// 4. Relationship inference and creation
// ============================================================================

// ============================================================================
// 1. ADDRESS NORMALIZATION PROCEDURES
// ============================================================================

// Procedure 1A: Normalize all owner addresses
// Purpose: Standardize address formats for matching
MATCH (o:Owner) 
WHERE o.address IS NOT NULL AND o.normalized_address IS NULL
SET o.normalized_address = apoc.text.clean(
  upper(
    apoc.text.replace(
      apoc.text.replace(
        apoc.text.replace(o.address, '\\bSTREET\\b', 'ST'),
        '\\bAVENUE\\b', 'AVE'
      ),
      '\\bBOULEVARD\\b', 'BLVD'
    )
  )
),
o.address_hash = apoc.util.md5([o.normalized_address]);

// Procedure 1B: Normalize all property addresses  
// Purpose: Standardize property address formats
MATCH (p:Property)
WHERE p.address IS NOT NULL AND p.normalized_address IS NULL
SET p.normalized_address = apoc.text.clean(
  upper(
    apoc.text.replace(
      apoc.text.replace(
        apoc.text.replace(p.address, '\\bSTREET\\b', 'ST'),
        '\\bAVENUE\\b', 'AVE'
      ),
      '\\bBOULEVARD\\b', 'BLVD'
    )
  )
);

// Procedure 1C: Normalize contractor business addresses
// Purpose: Standardize contractor address formats
MATCH (c:Contractor)
WHERE c.business_address IS NOT NULL AND c.normalized_business_address IS NULL
SET c.normalized_business_address = apoc.text.clean(
  upper(
    apoc.text.replace(
      apoc.text.replace(
        apoc.text.replace(c.business_address, '\\bSTREET\\b', 'ST'),
        '\\bAVENUE\\b', 'AVE'
      ),
      '\\bBOULEVARD\\b', 'BLVD'
    )
  )
);

// ============================================================================
// 2. ENTITY NAME NORMALIZATION PROCEDURES
// ============================================================================

// Procedure 2A: Normalize owner names
// Purpose: Standardize owner names for entity matching
MATCH (o:Owner)
WHERE o.name IS NOT NULL AND o.normalized_name IS NULL
SET o.normalized_name = apoc.text.clean(
  upper(
    apoc.text.replace(
      apoc.text.replace(
        apoc.text.replace(
          apoc.text.replace(o.name, '\\bLLC\\b', ''),
          '\\bINC\\b', ''
        ),
        '\\bCORP\\b', ''
      ),
      '\\bLTD\\b', ''
    )
  )
);

// Procedure 2B: Normalize contractor business names
// Purpose: Standardize contractor names for matching
MATCH (c:Contractor)
WHERE c.business_name IS NOT NULL AND c.normalized_business_name IS NULL
SET c.normalized_business_name = apoc.text.clean(
  upper(
    apoc.text.replace(
      apoc.text.replace(
        apoc.text.replace(
          apoc.text.replace(c.business_name, '\\bLLC\\b', ''),
          '\\bINC\\b', ''
        ),
        '\\bCORP\\b', ''
      ),
      '\\bLTD\\b', ''
    )
  )
);

// ============================================================================
// 3. SHELL COMPANY RELATIONSHIP CREATION
// ============================================================================

// Procedure 3A: Create SHARES_ADDRESS relationships for exact matches
// Purpose: Link entities with identical addresses
MATCH (o1:Owner), (o2:Owner)
WHERE o1 <> o2 
  AND o1.normalized_address = o2.normalized_address
  AND o1.normalized_address IS NOT NULL
  AND NOT EXISTS((o1)-[:SHARES_ADDRESS]-(o2))
CREATE (o1)-[:SHARES_ADDRESS {
  similarity: 1.0,
  match_type: 'exact_address',
  created_date: date()
}]->(o2);

// Procedure 3B: Create SHARES_ADDRESS relationships for fuzzy matches
// Purpose: Link entities with similar addresses (edit distance <= 2)
MATCH (o1:Owner), (o2:Owner)
WHERE o1 <> o2 
  AND o1.normalized_address <> o2.normalized_address
  AND apoc.text.distance(o1.normalized_address, o2.normalized_address) <= 2
  AND o1.normalized_address IS NOT NULL
  AND o2.normalized_address IS NOT NULL
  AND NOT EXISTS((o1)-[:SHARES_ADDRESS]-(o2))
WITH o1, o2, apoc.text.distance(o1.normalized_address, o2.normalized_address) as distance
CREATE (o1)-[:SHARES_ADDRESS {
  similarity: 1.0 - (distance / 10.0),
  match_type: 'fuzzy_address',
  edit_distance: distance,
  created_date: date()
}]->(o2);

// Procedure 3C: Create POTENTIAL_SHELL relationships
// Purpose: Flag potential shell companies based on multiple criteria
MATCH (o1:Owner)-[:SHARES_ADDRESS]-(o2:Owner)
WHERE NOT EXISTS((o1)-[:POTENTIAL_SHELL]-(o2))
WITH o1, o2
MATCH (o1)-[:OWNS]->(p1:Property)
MATCH (o2)-[:OWNS]->(p2:Property)
WITH o1, o2, count(DISTINCT p1) as props1, count(DISTINCT p2) as props2
WHERE props1 + props2 >= 3  // Minimum property threshold
CREATE (o1)-[:POTENTIAL_SHELL {
  confidence: CASE 
    WHEN props1 + props2 >= 10 THEN 0.9
    WHEN props1 + props2 >= 5 THEN 0.7
    ELSE 0.5
  END,
  property_count: props1 + props2,
  created_date: date()
}]->(o2);

// ============================================================================
// 4. CONTRACTOR COLLABORATION RELATIONSHIPS
// ============================================================================

// Procedure 4A: Create COLLABORATED_WITH relationships
// Purpose: Link contractors who work on the same properties
MATCH (c1:Contractor)-[:WORKED_ON]->(j1:Job)-[:PERFORMED_AT]->(p:Property)
MATCH (c2:Contractor)-[:WORKED_ON]->(j2:Job)-[:PERFORMED_AT]->(p)
WHERE c1 <> c2 
  AND NOT EXISTS((c1)-[:COLLABORATED_WITH]-(c2))
WITH c1, c2, count(*) as collaboration_count, collect(DISTINCT p.address) as shared_properties
WHERE collaboration_count >= 2
CREATE (c1)-[:COLLABORATED_WITH {
  job_count: collaboration_count,
  shared_properties: shared_properties,
  property_count: size(shared_properties),
  created_date: date()
}]->(c2);

// ============================================================================
// 5. DATA VALIDATION PROCEDURES
// ============================================================================

// Procedure 5A: Identify properties with missing BIN numbers
// Purpose: Flag data quality issues
MATCH (p:Property)
WHERE p.bin IS NULL OR p.bin = '' OR p.bin = 0
SET p:DataQualityIssue, p.issue_type = 'missing_bin';

// Procedure 5B: Identify owners with suspicious address patterns
// Purpose: Flag potential data entry errors
MATCH (o:Owner)
WHERE o.address IS NOT NULL 
  AND (o.address =~ '.*P\\.?O\\.? BOX.*' 
       OR o.address =~ '.*GENERAL DELIVERY.*'
       OR size(o.address) < 10)
SET o:DataQualityIssue, o.issue_type = 'suspicious_address';

// Procedure 5C: Identify contractors with invalid license numbers
// Purpose: Flag licensing issues
MATCH (c:Contractor)
WHERE c.license_sl_no IS NULL 
   OR c.license_sl_no = 0 
   OR c.license_sl_no = ''
SET c:DataQualityIssue, c.issue_type = 'invalid_license';

// ============================================================================
// 6. GEOGRAPHIC ENHANCEMENT PROCEDURES
// ============================================================================

// Procedure 6A: Extract borough from property addresses
// Purpose: Add geographic classification
MATCH (p:Property)
WHERE p.borough IS NULL AND p.address IS NOT NULL
SET p.borough = CASE
  WHEN p.address CONTAINS 'MANHATTAN' OR p.address CONTAINS 'NEW YORK' THEN 'Manhattan'
  WHEN p.address CONTAINS 'BROOKLYN' THEN 'Brooklyn'
  WHEN p.address CONTAINS 'QUEENS' THEN 'Queens'
  WHEN p.address CONTAINS 'BRONX' THEN 'Bronx'
  WHEN p.address CONTAINS 'STATEN ISLAND' THEN 'Staten Island'
  ELSE 'Unknown'
END;

// Procedure 6B: Extract ZIP codes from addresses
// Purpose: Add postal code classification
MATCH (p:Property)
WHERE p.zipcode IS NULL AND p.address IS NOT NULL
WITH p, apoc.text.regexGroups(p.address, '(\\d{5})')[0] as zip_match
WHERE zip_match IS NOT NULL AND size(zip_match) > 0
SET p.zipcode = zip_match[0];

// ============================================================================
// 7. BATCH PROCESSING PROCEDURES
// ============================================================================

// Procedure 7A: Run all normalization procedures
// Purpose: Execute complete data normalization pipeline
CALL {
  // Address normalization
  MATCH (o:Owner) 
  WHERE o.address IS NOT NULL AND o.normalized_address IS NULL
  WITH o LIMIT 1000
  SET o.normalized_address = apoc.text.clean(upper(o.address)),
      o.address_hash = apoc.util.md5([o.normalized_address])
} IN TRANSACTIONS OF 1000 ROWS;

CALL {
  // Name normalization  
  MATCH (o:Owner)
  WHERE o.name IS NOT NULL AND o.normalized_name IS NULL
  WITH o LIMIT 1000
  SET o.normalized_name = apoc.text.clean(upper(o.name))
} IN TRANSACTIONS OF 1000 ROWS;

// Procedure 7B: Create all fraud detection relationships
// Purpose: Build relationship network for fraud detection
CALL {
  // Shell company relationships
  MATCH (o1:Owner), (o2:Owner)
  WHERE o1 <> o2 
    AND o1.normalized_address = o2.normalized_address
    AND o1.normalized_address IS NOT NULL
    AND NOT EXISTS((o1)-[:SHARES_ADDRESS]-(o2))
  WITH o1, o2 LIMIT 1000
  CREATE (o1)-[:SHARES_ADDRESS {similarity: 1.0, match_type: 'exact'}]->(o2)
} IN TRANSACTIONS OF 1000 ROWS;

// ============================================================================
// 8. PERFORMANCE MONITORING QUERIES
// ============================================================================

// Query 8A: Data quality summary
// Purpose: Monitor data completeness and quality
RETURN 
  'Owners' as entity_type,
  count(*) as total_count,
  count(CASE WHEN normalized_address IS NOT NULL THEN 1 END) as normalized_addresses,
  count(CASE WHEN normalized_name IS NOT NULL THEN 1 END) as normalized_names
FROM (MATCH (o:Owner) RETURN o)
UNION ALL
RETURN 
  'Properties' as entity_type,
  count(*) as total_count,
  count(CASE WHEN normalized_address IS NOT NULL THEN 1 END) as normalized_addresses,
  count(CASE WHEN borough IS NOT NULL THEN 1 END) as with_borough
FROM (MATCH (p:Property) RETURN p);

// Query 8B: Relationship summary
// Purpose: Monitor relationship creation progress
MATCH ()-[r]->()
RETURN type(r) as relationship_type, count(r) as count
ORDER BY count DESC; 