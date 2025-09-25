"""Chart generation utilities."""
import io
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from app.models.schemas import CompanyData

def generate_bar_chart(
    data: Dict[str, float],
    title: str,
    x_label: str,
    y_label: str
) -> bytes:
    """Generate a bar chart and return as PNG bytes."""
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    labels = list(data.keys())
    values = list(data.values())
    
    # Create bar chart
    bars = plt.bar(labels, values, color='skyblue')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom')
    
    # Customize the chart
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    buffer.seek(0)
    return buffer.getvalue()

async def generate_company_metrics_charts(
    companies: List[CompanyData],
    metrics: List[str]
) -> List[bytes]:
    """Generate charts for company metrics."""
    charts = []
    
    # Example: Employee count comparison
    employee_data = {}
    revenue_data = {}
    
    for company in companies:
        if 'employees' in company.metrics:
            try:
                employee_data[company.name] = float(company.metrics['employees'])
            except (ValueError, TypeError):
                pass
                
        if 'revenue' in company.metrics and isinstance(company.metrics['revenue'], str):
            # Simple revenue parsing (in a real app, use a proper currency parser)
            rev_str = company.metrics['revenue'].replace('$', '').replace(',', '').replace(' ', '')
            if '-' in rev_str:
                rev_avg = sum(float(x) for x in rev_str.split('-')) / 2
                revenue_data[company.name] = rev_avg
            else:
                try:
                    revenue_data[company.name] = float(rev_str)
                except (ValueError, TypeError):
                    pass
    
    # Generate employee chart if we have data
    if employee_data:
        charts.append(
            generate_bar_chart(
                employee_data,
                "Employee Count Comparison",
                "Company",
                "Number of Employees"
            )
        )
    
    # Generate revenue chart if we have data
    if revenue_data:
        # Convert to millions for better readability
        revenue_millions = {k: v / 1_000_000 for k, v in revenue_data.items()}
        charts.append(
            generate_bar_chart(
                revenue_millions,
                "Estimated Annual Revenue (Millions)",
                "Company",
                "Revenue ($M)"
            )
        )
    
    return charts
