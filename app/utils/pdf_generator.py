"""PDF generation utilities for reports."""
import io
from datetime import datetime
from typing import List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

async def generate_pdf_report(
    request_id: str,
    report_data: dict,
    charts: Optional[List[bytes]] = None
) -> bytes:
    """Generate a PDF report from the analysis results."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title = Paragraph("Competitor Analysis Report", styles['Title'])
    elements.append(title)
    
    # Report metadata
    elements.append(Paragraph(f"Report ID: {request_id}", styles['Normal']))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                           styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", styles['Heading1']))
    elements.append(Paragraph(report_data.get('executive_summary', ''), styles['Normal']))
    
    # Add charts if available
    if charts:
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Key Metrics", styles['Heading2']))
        for chart in charts:
            try:
                img = Image(io.BytesIO(chart), width=400, height=300)
                elements.append(img)
                elements.append(Spacer(1, 10))
            except:
                continue
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
