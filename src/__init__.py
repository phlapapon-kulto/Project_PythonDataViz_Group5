"""
Thai Agriculture & Climate Intelligence Dashboard - Source Package
Contains modules for data loading, data cleaning, feature engineering,
analytics, visualizations, and filter management.
"""

from __future__ import annotations

__version__ = "2.0.0"

from src import (
    data_loader,
    data_cleaning,
    feature_engineering,
    analytics,
    visualizations,
    filters
)

__all__ = [
    "data_loader",
    "data_cleaning",
    "feature_engineering",
    "analytics",
    "visualizations",
    "filters"
]
