"""Enums and constants for the RivalLens application."""
from enum import Enum, auto

class InfoCategory(str, Enum):
    """Allowed parent categories for competitor information."""
    MARKET_PRESENCE = "Market Presence"
    FINANCIAL_HEALTH = "Financial Health"
    PRODUCTS = "Products & Offerings"
    MARKETING = "Marketing & Branding"
    TECH = "Technology & Innovation"
    CUSTOMER_SENTIMENT = "Customer Sentiment"
    HIRING = "Hiring & Organization"

# Set of all allowed categories for validation
ALLOWED_CATEGORIES = {category.value for category in InfoCategory}
