"""
Filter management and state synchronization module for Streamlit.
Provides robust multi-dimensional filtering and prevents inadvertent row multiplication.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Tuple


def init_session_state(default_year: int = 2562):
    """Initialize Streamlit session state keys if not already present."""
    if "selected_year" not in st.session_state:
        st.session_state.selected_year = default_year
    if "selected_province" not in st.session_state:
        st.session_state.selected_province = "ทั้งหมด"
    if "selected_product" not in st.session_state:
        st.session_state.selected_product = "ทั้งหมด"
    if "selected_metric" not in st.session_state:
        st.session_state.selected_metric = "production_ton"
    if "harvest_threshold" not in st.session_state:
        st.session_state.harvest_threshold = 90.0


def reset_filters(default_year: int = 2562):
    """Callback to reset all interactive filters to their default state."""
    st.session_state.selected_year = default_year
    st.session_state.selected_province = "ทั้งหมด"
    st.session_state.selected_product = "ทั้งหมด"
    st.session_state.selected_metric = "production_ton"
    st.session_state.harvest_threshold = 90.0


def apply_ag_filters(
    df: pd.DataFrame,
    year: Optional[int] = None,
    province: Optional[str] = None,
    product: Optional[str] = None,
    geo_level: str = "province"
) -> pd.DataFrame:
    """
    Apply global filters to the agricultural dataset.
    
    Parameters:
    - df: Agricultural DataFrame
    - year: Specific calendar year (BE), or None for all years
    - province: Specific province name, 'ทั้งหมด' for all, or None
    - product: Specific crop name, 'ทั้งหมด' for all, or None
    - geo_level: 'province', 'region', 'country', or 'all'
    """
    filtered = df.copy()

    # Geo level filtering
    if geo_level != "all":
        filtered = filtered[filtered["geo_level"] == geo_level]

    # Year filter
    if year is not None and year != "ทั้งหมด":
        filtered = filtered[filtered["year"] == int(year)]

    # Province filter
    if province is not None and province != "ทั้งหมด":
        filtered = filtered[filtered["province"] == province]

    # Product filter
    if product is not None and product != "ทั้งหมด":
        filtered = filtered[filtered["product"] == product]

    return filtered.reset_index(drop=True)
