"""
FastAPI application for RivalLens - Competitor Intelligence API
"""
import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Configure logging before other imports to ensure all modules use it
from logging_config import configure_logging
logger = configure_logging()

# Now import other modules
from app.models.schemas import UserPayload, ReportResponse, CompanyData, CompetitorInsight
from app.services.llm_client import llm
from app.services.search import search_adapter
from app.utils import pdf_generator, charts
from app.models.enums import InfoCategory
from app.config import settings

# Log configuration status
logger.info(f"Starting {settings.APP_NAME}")
logger.debug(f"Debug mode: {settings.DEBUG}")
logger.debug(f"Using API URL: {settings.DEEPSEEK_API_URL}")
logger.debug(f"API Key configured: {'Yes' if settings.DEEPSEEK_API_KEY else 'No'}")

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="API for generating competitive intelligence reports",
    version="1.0.0",
    debug=settings.DEBUG
)

# Log application startup
logger.info(f"{settings.APP_NAME} v1.0.0 starting up...")
logger.info(f"Environment: {'development' if settings.DEBUG else 'production'}")
logger.info(f"API Key: {'Configured' if settings.DEEPSEEK_API_KEY else 'Not configured'}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
async def build_system_prompt(company_name: str, insight_selection: List[str], deep_dive: Optional[List[str]] = None) -> str:
    """Build a robust system prompt for the LLM based on user payload."""
    categories = ", ".join(insight_selection)
    deep_dive_text = ""
    if deep_dive:
        deep_dive_text = f"\nFor a deeper analysis, focus on: {', '.join(deep_dive)}."
    
    return (
        f"You are a competitive intelligence analyst for business strategy. "
        f"Analyze the company '{company_name}' and its competitors. "
        f"Focus specifically on these categories: {categories}.{deep_dive_text}\n"
        f"Provide actionable insights, highlight trends, gaps, and opportunities. "
        f"Be concise, professional, and data-driven. "
        f"Include an executive summary, detailed competitor insights, and a side-by-side comparison where possible. "
        f"Use real metrics and notes to synthesize meaningful analysis. "
        f"Do not use generic phrases like 'leading company' or 'based on available data'."
    )


async def generate_insights(company: CompanyData, categories: list, business_name: str = "your business") -> CompetitorInsight:
    """Generate insights for a single company using the LLM.
    
    Args:
        company: The company data to analyze
        categories: List of categories to focus the analysis on
        business_name: Name of the business being analyzed (for context in the prompt)
        
    Returns:
        CompetitorInsight: Detailed insights about the company
        
    Raises:
        ValueError: If the API key is not configured
        httpx.HTTPStatusError: If the API request fails
        Exception: For any other unexpected errors
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Build a detailed prompt for the LLM
        system_prompt = (
            f"You are a competitive intelligence analyst for business strategy. "
            f"Analyze the company '{company.name}' as a competitor to '{business_name}'. "
            f"Focus specifically on these categories: {', '.join(categories)}. "
            f"Provide actionable insights, highlight trends, gaps, and opportunities. "
            f"Be concise, professional, and data-driven. "
            f"Use the company's metrics and notes to provide specific, meaningful analysis. "
            f"Do not use generic phrases like 'leading company' or 'based on available data'."
        )
        
        user_prompt = (
            f"Company: {company.name}\n"
            f"Description: {company.description or 'No description available'}\n"
            f"Metrics: {company.metrics or 'No metrics available'}\n"
            f"Notes: {', '.join(company.notes) if company.notes else 'No notes available'}\n\n"
            f"Please provide a detailed analysis including: "
            f"1. A comprehensive summary of {company.name}'s competitive position\n"
            f"2. Analysis for each of these categories: {', '.join(categories)}\n"
            f"3. Key strengths and weaknesses compared to {business_name}"
        )
        
        logger.info(f"Generating insights for {company.name}...")
        llm_response = await llm.summarize(system_prompt, user_prompt)
        
        # Process the LLM response
        if not llm_response:
            raise ValueError("Empty response received from LLM")
            
        # Use the full response as the summary
        summary = llm_response
        
        # Create a category breakdown that includes the full analysis for each category
        category_breakdown = {}
        lines = [line.strip() for line in llm_response.split('\n') if line.strip()]
        
        # If we have categories, try to find sections for each one
        if categories:
            for category in categories:
                # Find all lines that start with the category name or a heading marker
                category_lines = []
                in_category = False
                
                for line in lines:
                    # Check if this line starts a new category section
                    if (line.lower().startswith(f"{category.lower()}:") or 
                        line.lower().startswith(f"**{category.lower()}**") or
                        line.lower().startswith(f"### {category}")):
                        in_category = True
                        category_lines.append(line)
                    # If we're in a category section, add lines until we hit another category
                    elif in_category and any(line.lower().startswith(f"{cat.lower()}: ") for cat in categories if cat != category):
                        in_category = False
                        break
                    elif in_category:
                        category_lines.append(line)
                
                # If we found lines for this category, join them. Otherwise, use the full response.
                if category_lines:
                    category_breakdown[category] = '\n'.join(category_lines)
                else:
                    category_breakdown[category] = llm_response
        else:
            # If no specific categories, include the full response for a default category
            category_breakdown["analysis"] = llm_response
        
        logger.info(f"Successfully generated insights for {company.name}")
        return CompetitorInsight(
            company=company,
            summary=summary,
            confidence="high",  # Since we're using real LLM now
            category_breakdown=category_breakdown,
            sources=company.sources[:3]  # Limit to top 3 sources
        )
        
    except Exception as e:
        error_msg = f"Failed to generate insights for {company.name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "app": settings.APP_NAME,
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/api/v1/analyze", response_model=ReportResponse)
async def analyze_competitors(
    payload: UserPayload,
    background_tasks: BackgroundTasks
):
    """
    Main endpoint for competitor analysis.
    """
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    
    # Get company info
    company_name = payload.company_info.name
    company_website = payload.company_info.website or ""
    
    # Get competitors
    competitors = []
    if payload.competitor_choice and payload.competitor_choice.competitors:
        competitors = payload.competitor_choice.competitors
    else:
        # Auto-discover competitors if none provided
        competitors = await search_adapter.discover_competitors(
            company_name,
            payload.business_category,
            None,  # geography can be added later
            settings.MAX_COMPETITORS
        )
    
    # Check if we have competitors to analyze
    if not competitors:
        raise HTTPException(
            status_code=400,
            detail="No competitors found or provided for analysis"
        )
    
    # Generate insights for each competitor
    tasks = []
    for competitor in competitors:
        company_data = await search_adapter.enrich_company(
            competitor,
            citation_depth=3,  # Default citation depth
            geography=None  # Can be updated if needed
        )
        task = generate_insights(
            company=company_data,
            categories=payload.insight_selection,
            business_name=company_name
        )
        tasks.append(task)
    
    # Run all tasks concurrently
    insights = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any errors
    valid_insights = []
    for i, insight in enumerate(insights):
        if isinstance(insight, Exception):
            logger.error(f"Error processing {competitors[i]}: {str(insight)}")
        else:
            valid_insights.append(insight)
    
    if not valid_insights:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate insights for any competitors"
        )
    
    # Prepare response with valid insights
    report = ReportResponse(
        request_id=request_id,
        executive_summary=f"Analysis of {len(valid_insights)} competitors for {company_name}.",
        top_insights=[
            f"{insight.company.name}: {insight.summary.split('\n')[0] if insight.summary else 'No summary available'}"
            for insight in valid_insights
        ],
        detailed={
            insight.company.name: {
                "company": insight.company.dict(),
                "summary": insight.summary,
                "confidence": insight.confidence,
                "category_breakdown": insight.category_breakdown,
                "sources": [
                    source.dict() if hasattr(source, 'dict') else source 
                    for source in insight.sources
                ] if insight.sources else []
            }
            for insight in valid_insights
        },
        comparison_table=[],  # This would be populated in a real implementation
        generated_at=datetime.utcnow(),
        sources=[],  # This would aggregate sources in a real implementation
    )
    
    # Handle export if requested
    if payload.preferences and payload.preferences.export_format:
        export_format = payload.preferences.export_format.lower()
        chart_tasks = []
        
        # For now, we'll just log the export request
        # In a real implementation, you would generate the appropriate export
        logger.info(f"Export requested in format: {export_format}")
        
        if export_format == 'pdf':
            # In a real implementation, you would generate charts and PDF here
            # For now, we'll just add a placeholder
            report.pdf_url = f"/api/v1/exports/{request_id}.pdf"
    
    return report

async def generate_pdf_export(request_id: str, report_data: dict, charts: list):
    """Background task to generate and store PDF report."""
    # In a real implementation, you would:
    # 1. Generate the PDF
    # 2. Store it in a persistent storage (S3, filesystem, etc.)
    # 3. Update the report status in your database
    pass

# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
