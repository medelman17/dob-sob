// ============================================================================
// NYC DOB Fraud Detection - Core Neo4j Schema
// ============================================================================
// This schema is optimized for detecting fraud patterns through:
// 1. Shell company detection via address matching
// 2. Cross-property work analysis via ownership networks  
// 3. Contractor community detection via collaboration patterns
// ============================================================================

// ============================================================================
// CONSTRAINTS - Data Integrity & Performance
// ============================================================================

// Professional/Contractor Constraints
CREATE CONSTRAINT professional_license_unique IF NOT EXISTS 
FOR (p:Professional) REQUIRE p.license_sl_no IS UNIQUE;

CREATE CONSTRAINT contractor_license_unique IF NOT EXISTS 
FOR (c:Contractor) REQUIRE c.license_sl_no IS UNIQUE;

// Property Constraints
CREATE CONSTRAINT property_bin_unique IF NOT EXISTS 
FOR (p:Property) REQUIRE p.bin IS UNIQUE;

CREATE CONSTRAINT property_bbl_unique IF NOT EXISTS 
FOR (p:Property) REQUIRE p.bbl IS UNIQUE;

// Owner Constraints
CREATE CONSTRAINT owner_id_unique IF NOT EXISTS 
FOR (o:Owner) REQUIRE o.owner_id IS UNIQUE;

// Job/Project Constraints
CREATE CONSTRAINT job_number_unique IF NOT EXISTS 
FOR (j:Job) REQUIRE j.job_number IS UNIQUE;

// Violation Constraints
CREATE CONSTRAINT violation_id_unique IF NOT EXISTS 
FOR (v:Violation) REQUIRE v.violation_id IS UNIQUE;

// ============================================================================
// INDEXES - Optimized for Fraud Detection Queries
// ============================================================================

// Address-based fraud detection (shell companies)
CREATE INDEX owner_normalized_address IF NOT EXISTS 
FOR (o:Owner) ON (o.normalized_address);

CREATE INDEX property_normalized_address IF NOT EXISTS 
FOR (p:Property) ON (p.normalized_address);

CREATE INDEX contractor_business_address IF NOT EXISTS 
FOR (c:Contractor) ON (c.normalized_business_address);

// Name-based entity matching
CREATE INDEX owner_normalized_name IF NOT EXISTS 
FOR (o:Owner) ON (o.normalized_name);

CREATE INDEX contractor_normalized_name IF NOT EXISTS 
FOR (c:Contractor) ON (c.normalized_business_name);

// License and business type analysis
CREATE INDEX professional_license_type IF NOT EXISTS 
FOR (p:Professional) ON (p.license_type);

CREATE INDEX contractor_license_type IF NOT EXISTS 
FOR (c:Contractor) ON (c.license_type);

// Temporal analysis indexes
CREATE INDEX job_filing_date IF NOT EXISTS 
FOR (j:Job) ON (j.filing_date);

CREATE INDEX violation_issue_date IF NOT EXISTS 
FOR (v:Violation) ON (v.issue_date);

// Geographic analysis
CREATE INDEX property_borough IF NOT EXISTS 
FOR (p:Property) ON (p.borough);

CREATE INDEX property_zipcode IF NOT EXISTS 
FOR (p:Property) ON (p.zipcode);

// Composite indexes for complex fraud queries
CREATE INDEX owner_address_name_composite IF NOT EXISTS 
FOR (o:Owner) ON (o.normalized_address, o.normalized_name);

CREATE INDEX contractor_address_license_composite IF NOT EXISTS 
FOR (c:Contractor) ON (c.normalized_business_address, c.license_type);

// ============================================================================
// NODE LABELS & PROPERTIES DOCUMENTATION
// ============================================================================

/*
CORE ENTITY TYPES:

1. (:Professional) - Individual license holders
   Properties: license_sl_no, license_type, first_name, last_name, 
              business_name, license_status, issue_date, expiration_date

2. (:Contractor) - Business entities performing work
   Properties: license_sl_no, business_name, normalized_business_name,
              license_type, business_address, normalized_business_address,
              phone, registration_date

3. (:Owner) - Property ownership entities (individuals/companies)
   Properties: owner_id, name, normalized_name, address, normalized_address,
              address_hash, phone, entity_type, registration_date

4. (:Property) - Buildings and real estate
   Properties: bin, bbl, address, normalized_address, borough, zipcode,
              building_class, year_built, owner_name, lot_area

5. (:Job) - Construction projects and permits
   Properties: job_number, job_type, work_type, status, filing_date,
              approval_date, completion_date, permit_type, description

6. (:Violation) - Code violations and enforcement actions
   Properties: violation_id, violation_type, issue_date, status,
              description, penalty_amount, respondent

7. (:Inspection) - Regulatory inspections
   Properties: inspection_id, inspection_type, inspection_date, result,
              inspector_id, follow_up_required

8. (:License) - Professional licenses and certifications
   Properties: license_number, license_type, issue_date, expiration_date,
              status, issuing_agency

RELATIONSHIP TYPES:

Core Ownership & Work Relationships:
- (:Owner)-[:OWNS]->(:Property)
- (:Contractor)-[:WORKED_ON]->(:Job)
- (:Job)-[:PERFORMED_AT]->(:Property)
- (:Professional)-[:FILED]->(:Job)

Fraud Detection Relationships:
- (:Owner)-[:SHARES_ADDRESS {similarity: float}]->(:Owner)
- (:Contractor)-[:COLLABORATED_WITH {job_count: int}]->(:Contractor)
- (:Owner)-[:POTENTIAL_SHELL {confidence: float}]->(:Owner)

Violation & Enforcement:
- (:Violation)-[:ISSUED_FOR]->(:Property)
- (:Violation)-[:ISSUED_TO]->(:Owner)
- (:Inspection)-[:CONDUCTED_AT]->(:Property)

Professional Relationships:
- (:Professional)-[:WORKS_FOR]->(:Contractor)
- (:Professional)-[:HOLDS]->(:License)
*/ 