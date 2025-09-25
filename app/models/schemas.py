"""Pydantic models for request/response schemas."""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, HttpUrl
from .enums import InfoCategory

class CompanyInfo(BaseModel):
    """Company information model."""
    name: str = Field(..., description="Name of the company")
    website: Optional[str] = Field(None, description="Company website URL")

class CompetitorChoice(BaseModel):
    """Competitor selection model."""
    competitors: List[str] = Field(..., description="List of competitor names")

class Preferences(BaseModel):
    """User preferences for the analysis."""
    export_format: Optional[str] = Field(
        None, 
        description="Export format: PDF, Slide Deck, or None",
        example="PDF"
    )

class UserPayload(BaseModel):
    """Request payload from the user."""
    business_category: str = Field(..., description="Business category/industry")
    company_info: CompanyInfo = Field(..., description="Information about the company")
    competitor_choice: Optional[CompetitorChoice] = Field(
        None,
        description="Competitor selection (required for manual competitor selection)"
    )
    insight_selection: List[str] = Field(
        ...,
        description="List of insights to include in the report"
    )
    deep_dive: Optional[List[str]] = Field(
        None,
        description="List of areas for deeper analysis"
    )
    preferences: Optional[Preferences] = Field(
        None,
        description="User preferences for the analysis"
    )

class CompanyData(BaseModel):
    """Data structure for company information."""
    name: str
    website: Optional[str] = None
    description: Optional[str] = None
    metrics: Dict[str, Any] = {}
    notes: List[str] = []
    sources: List[Dict[str, Any]] = []

class CompetitorInsight(BaseModel):
    """Detailed insights for a single competitor."""
    company: CompanyData
    summary: str
    confidence: str
    category_breakdown: Dict[str, str] = {}
    sources: List[Dict[str, Any]] = []

class ReportResponse(BaseModel):
    """Response model for the analysis report."""
    request_id: str
    executive_summary: str
    top_insights: List[str]
    detailed: Dict[str, CompetitorInsight]
    comparison_table: List[Dict[str, Any]]
    generated_at: datetime
    sources: List[Dict[str, Any]]
    pdf_url: Optional[str] = None