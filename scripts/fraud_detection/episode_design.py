"""
Episode Design Module for NYC DOB Fraud Detection

This module defines custom entity types and data transformation pipelines
optimized for Graphiti's automatic entity/relationship extraction from NYC DOB datasets.
"""

import json
import polars as pl
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field

# Import real Graphiti now that it's installed
from graphiti_core.nodes import EpisodeType

class NYCDOBEntityTypes:
    """Custom entity types for NYC DOB domain modeling"""
    
    class Building(BaseModel):
        """NYC Building entity"""
        bin: Optional[str] = Field(None, description="Building Identification Number")
        bbl: Optional[str] = Field(None, description="Borough, Block, Lot identifier")
        address: Optional[str] = Field(None, description="Full street address")
        house_number: Optional[str] = Field(None, description="House number")
        street_name: Optional[str] = Field(None, description="Street name")
        borough: Optional[str] = Field(None, description="NYC borough")
        zip_code: Optional[str] = Field(None, description="ZIP code")
        latitude: Optional[float] = Field(None, description="Geographic latitude")
        longitude: Optional[float] = Field(None, description="Geographic longitude")
    
    class Corporation(BaseModel):
        """Business entity in NYC construction ecosystem"""
        name: str = Field(..., description="Legal business name")
        entity_type: Optional[str] = Field(None, description="LLC, Corp, Partnership, etc.")
        certification: Optional[str] = Field(None, description="MBE/WBE/DBE certification type")
        license_number: Optional[str] = Field(None, description="DOB license number")
        business_address: Optional[str] = Field(None, description="Registered business address")
        status: Optional[str] = Field(None, description="Active, inactive, suspended, etc.")
    
    class Person(BaseModel):
        """Individual involved in construction projects"""
        name: str = Field(..., description="Person's full name")
        role: Optional[str] = Field(None, description="Owner, Officer, Director, Contractor, etc.")
        address: Optional[str] = Field(None, description="Personal or business address")
        license_type: Optional[str] = Field(None, description="Professional license type")
        
    class Permit(BaseModel):
        """DOB permit or application"""
        permit_number: str = Field(..., description="Unique permit identifier")
        permit_type: str = Field(..., description="Type of permit or filing")
        status: Optional[str] = Field(None, description="Current permit status")
        work_type: Optional[str] = Field(None, description="Type of work being performed")
        filing_date: Optional[str] = Field(None, description="Date permit was filed")
        issue_date: Optional[str] = Field(None, description="Date permit was issued")
        expiration_date: Optional[str] = Field(None, description="Permit expiration date")
        
    class Violation(BaseModel):
        """DOB or ECB violation"""
        violation_number: str = Field(..., description="Unique violation identifier")
        violation_type: str = Field(..., description="Type of violation")
        violation_category: Optional[str] = Field(None, description="Category of violation")
        severity: Optional[str] = Field(None, description="Violation severity level")
        issue_date: Optional[str] = Field(None, description="Date violation was issued")
        disposition: Optional[str] = Field(None, description="How violation was resolved")
        penalty_amount: Optional[float] = Field(None, description="Monetary penalty amount")
        
    class Project(BaseModel):
        """Construction project or development"""
        project_id: str = Field(..., description="Unique project identifier")
        project_type: Optional[str] = Field(None, description="Type of construction project")
        work_description: Optional[str] = Field(None, description="Description of work")
        estimated_cost: Optional[float] = Field(None, description="Estimated project cost")
        start_date: Optional[str] = Field(None, description="Project start date")
        completion_date: Optional[str] = Field(None, description="Project completion date")
        
    class FinancialFlow(BaseModel):
        """Financial transaction or payment flow"""
        amount: Optional[float] = Field(None, description="Transaction amount")
        markup_percentage: Optional[float] = Field(None, description="Markup percentage applied")
        payment_type: Optional[str] = Field(None, description="Type of payment or transaction")
        transaction_date: Optional[str] = Field(None, description="Date of transaction")


class EpisodeDesigner:
    """
    Designs and transforms NYC DOB data into optimal episode structures
    for Graphiti's automatic entity/relationship extraction.
    """
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.entity_types = {
            "Building": NYCDOBEntityTypes.Building,
            "Corporation": NYCDOBEntityTypes.Corporation,
            "Person": NYCDOBEntityTypes.Person,
            "Permit": NYCDOBEntityTypes.Permit,
            "Violation": NYCDOBEntityTypes.Violation,
            "Project": NYCDOBEntityTypes.Project,
            "FinancialFlow": NYCDOBEntityTypes.FinancialFlow
        }
    
    def prepare_housing_litigation_episodes(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Transform housing litigation data into episodes for Graphiti ingestion.
        
        Args:
            file_path: Path to housing litigation CSV file
            
        Returns:
            List of episode dictionaries ready for Graphiti
        """
        df = pl.read_csv(file_path)
        episodes = []
        
        for row in df.iter_rows(named=True):
            # Create episode for each litigation case
            episode_content = {
                "litigation_id": row.get("LitigationID"),
                "case_type": row.get("CaseType"),
                "case_status": row.get("CaseStatus"),
                "open_date": row.get("CaseOpenDate"),
                "respondent": row.get("Respondent"),
                "finding_of_harassment": row.get("FindingOfHarassment"),
                "penalty": row.get("Penalty"),
                "building": {
                    "bin": row.get("BIN"),
                    "bbl": row.get("BBL"),
                    "address": f"{row.get('HouseNumber', '')} {row.get('StreetName', '')}".strip(),
                    "house_number": row.get("HouseNumber"),
                    "street_name": row.get("StreetName"),
                    "borough": self._map_borough_code(row.get("Boro")),
                    "zip_code": row.get("Zip"),
                    "latitude": row.get("Latitude"),
                    "longitude": row.get("Longitude")
                }
            }
            
            # Parse reference time from case open date
            reference_time = self._parse_date(row.get("CaseOpenDate"))
            
            episode = {
                "name": f"Housing_Litigation_{row.get('LitigationID')}",
                "content": episode_content,
                "type": EpisodeType.json,
                "description": f"Housing litigation case {row.get('CaseType')}",
                "reference_time": reference_time
            }
            
            episodes.append(episode)
            
        return episodes
    
    def prepare_complaint_episodes(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Transform complaint data into episodes for Graphiti ingestion.
        
        Args:
            file_path: Path to complaints CSV file
            
        Returns:
            List of episode dictionaries ready for Graphiti
        """
        # Use polars for efficient processing of large files
        df = pl.read_csv(file_path)
        episodes = []
        
        for row in df.iter_rows(named=True):
            episode_content = {
                "complaint_number": row.get("Complaint Number"),
                "status": row.get("Status"),
                "complaint_category": row.get("Complaint Category"),
                "unit": row.get("Unit"),
                "disposition_code": row.get("Disposition Code"),
                "date_entered": row.get("Date Entered"),
                "disposition_date": row.get("Disposition Date"),
                "inspection_date": row.get("Inspection Date"),
                "building": {
                    "bin": row.get("BIN"),
                    "address": f"{row.get('House Number', '')} {row.get('House Street', '')}".strip(),
                    "house_number": row.get("House Number"),
                    "street_name": row.get("House Street"),
                    "zip_code": row.get("ZIP Code"),
                    "community_board": row.get("Community Board")
                }
            }
            
            reference_time = self._parse_date(row.get("Date Entered"))
            
            episode = {
                "name": f"Complaint_{row.get('Complaint Number')}",
                "content": episode_content,
                "type": EpisodeType.json,
                "description": f"DOB complaint {row.get('Complaint Category')}",
                "reference_time": reference_time
            }
            
            episodes.append(episode)
            
        return episodes
    
    def prepare_permit_episodes(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Transform permit data into episodes for Graphiti ingestion.
        
        Args:
            file_path: Path to permit CSV file
            
        Returns:
            List of episode dictionaries ready for Graphiti
        """
        df = pl.read_csv(file_path)
        episodes = []
        
        # Group permits by building and permit holder for more complex episodes
        for row in df.iter_rows(named=True):
            episode_content = {
                "permit_details": {
                    "permit_number": row.get("Job #") or row.get("Job"),
                    "permit_type": row.get("Permit Type") or row.get("Job Type"),
                    "work_type": row.get("Work Type"),
                    "permit_status": row.get("Current Status"),
                    "filing_date": row.get("Filing Date") or row.get("Pre- Filing Date"),
                    "issue_date": row.get("Issuance Date"),
                    "expiration_date": row.get("Expiration Date"),
                    "job_description": row.get("Job Description")
                },
                "building": {
                    "bin": row.get("BIN"),
                    "bbl": row.get("BBL"),
                    "address": f"{row.get('House #', '')} {row.get('Street Name', '')}".strip(),
                    "house_number": row.get("House #"),
                    "street_name": row.get("Street Name"),
                    "borough": self._map_borough_code(row.get("Borough")),
                    "zip_code": row.get("Zip Code")
                },
                "applicant": {
                    "owner_name": row.get("Owner's First Name", "") + " " + row.get("Owner's Last Name", ""),
                    "owner_business_name": row.get("Owner's Business Name"),
                    "owner_phone": row.get("Owner's Phone #"),
                    "owner_business_type": row.get("Owner's Business Type")
                }
            }
            
            reference_time = self._parse_date(row.get("Filing Date") or row.get("Pre- Filing Date"))
            
            episode = {
                "name": f"Permit_{row.get('Job #') or row.get('Job')}",
                "content": episode_content,
                "type": EpisodeType.json,
                "description": f"DOB permit filing {row.get('Permit Type') or row.get('Job Type')}",
                "reference_time": reference_time
            }
            
            episodes.append(episode)
            
        return episodes
    
    def create_complex_project_episodes(self, permit_data: List[Dict], 
                                      violation_data: List[Dict],
                                      complaint_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Create complex episodes that combine related permits, violations, and complaints
        for the same building or project to enable richer relationship discovery.
        
        Args:
            permit_data: List of permit episodes
            violation_data: List of violation episodes  
            complaint_data: List of complaint episodes
            
        Returns:
            List of complex project episodes
        """
        # Group by BIN (Building Identification Number)
        project_groups = {}
        
        # Group permits by BIN
        for permit in permit_data:
            bin_num = permit["content"]["building"]["bin"]
            if bin_num and bin_num not in project_groups:
                project_groups[bin_num] = {
                    "permits": [],
                    "violations": [],
                    "complaints": [],
                    "building": permit["content"]["building"]
                }
            if bin_num:
                project_groups[bin_num]["permits"].append(permit["content"]["permit_details"])
        
        # Add violations and complaints to groups
        for violation in violation_data:
            bin_num = violation["content"]["building"]["bin"]
            if bin_num in project_groups:
                project_groups[bin_num]["violations"].append(violation["content"])
                
        for complaint in complaint_data:
            bin_num = complaint["content"]["building"]["bin"]
            if bin_num in project_groups:
                project_groups[bin_num]["complaints"].append(complaint["content"])
        
        # Create complex episodes for buildings with multiple activities
        complex_episodes = []
        for bin_num, project_data in project_groups.items():
            if (len(project_data["permits"]) > 1 or 
                len(project_data["violations"]) > 0 or 
                len(project_data["complaints"]) > 0):
                
                episode_content = {
                    "project_summary": f"Building {bin_num} with {len(project_data['permits'])} permits, "
                                     f"{len(project_data['violations'])} violations, "
                                     f"{len(project_data['complaints'])} complaints",
                    "building": project_data["building"],
                    "permits": project_data["permits"],
                    "violations": project_data["violations"],
                    "complaints": project_data["complaints"],
                    "activity_timeline": self._create_timeline(project_data)
                }
                
                episode = {
                    "name": f"Building_Project_{bin_num}",
                    "content": episode_content,
                    "type": EpisodeType.json,
                    "description": f"Complex building project with multiple activities",
                    "reference_time": datetime.now(timezone.utc)
                }
                
                complex_episodes.append(episode)
        
        return complex_episodes
    
    def _map_borough_code(self, boro_code: Union[str, int, None]) -> Optional[str]:
        """Map borough code to borough name"""
        if boro_code is None:
            return None
            
        borough_map = {
            "1": "Manhattan",
            "2": "Bronx", 
            "3": "Brooklyn",
            "4": "Queens",
            "5": "Staten Island",
            1: "Manhattan",
            2: "Bronx",
            3: "Brooklyn", 
            4: "Queens",
            5: "Staten Island"
        }
        return borough_map.get(str(boro_code), str(boro_code))
    
    def _parse_date(self, date_str: Union[str, None]) -> datetime:
        """Parse date string into datetime object"""
        if not date_str:
            return datetime.now(timezone.utc)
            
        try:
            # Try common date formats
            for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"]:
                try:
                    return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        except:
            pass
            
        return datetime.now(timezone.utc)
    
    def _create_timeline(self, project_data: Dict) -> List[Dict[str, Any]]:
        """Create a timeline of events for a building project"""
        timeline = []
        
        # Add permit events
        for permit in project_data.get("permits", []):
            if permit.get("filing_date"):
                timeline.append({
                    "date": permit["filing_date"],
                    "event": "permit_filed",
                    "details": f"Permit {permit.get('permit_number')} filed for {permit.get('permit_type')}"
                })
        
        # Add violation events
        for violation in project_data.get("violations", []):
            if violation.get("issue_date"):
                timeline.append({
                    "date": violation["issue_date"],
                    "event": "violation_issued",
                    "details": f"Violation {violation.get('violation_number')} for {violation.get('violation_type')}"
                })
        
        # Add complaint events  
        for complaint in project_data.get("complaints", []):
            if complaint.get("date_entered"):
                timeline.append({
                    "date": complaint["date_entered"],
                    "event": "complaint_filed",
                    "details": f"Complaint {complaint.get('complaint_number')} for {complaint.get('complaint_category')}"
                })
        
        # Sort by date
        timeline.sort(key=lambda x: self._parse_date(x["date"]))
        
        return timeline 