"""
Data loading module for Thai Agriculture & Climate Intelligence Dashboard.
Loads agricultural, climate, and geographic datasets with robust fallback and caching.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def load_agriculture_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the primary agricultural statistics Excel file.
    Default path: 'สถิติการผลิตสินค้าเกษตรรายปี 2550-2562.xlsx' in project root.
    """
    if filepath is None:
        root = get_project_root()
        filepath = os.path.join(root, "สถิติการผลิตสินค้าเกษตรรายปี 2550-2562.xlsx")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agriculture data file not found at: {filepath}")

    df = pd.read_excel(filepath, sheet_name=0)
    return df


def load_suphan_climate_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Suphan Buri meteorological station Excel file.
    Default path: 'สถิติปริมาณฝนและอุณหภูมิ.xlsx' in project root.
    """
    if filepath is None:
        root = get_project_root()
        filepath = os.path.join(root, "สถิติปริมาณฝนและอุณหภูมิ.xlsx")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Suphan Buri climate data file not found at: {filepath}")

    df = pd.read_excel(filepath, sheet_name=0)
    return df


def load_spatial_rain_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load HII provincial spatial rainfall data (2018-2026).
    """
    if filepath is None:
        root = get_project_root()
        filepath = os.path.join(root, "data", "raw", "spatial_rain_hii.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Spatial rain file not found at: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    return df


def load_national_rain_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load HII national monthly rainfall data (1970-2025).
    """
    if filepath is None:
        root = get_project_root()
        filepath = os.path.join(root, "data", "raw", "national_rain_1970_current.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"National rain file not found at: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    return df


def load_provincial_temp_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load HII provincial monthly temperature data for overlapping years (2018-2019).
    """
    if filepath is None:
        root = get_project_root()
        filepath = os.path.join(root, "data", "raw", "provincial_temp_2018_2019.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Provincial temp file not found at: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    return df


def load_thailand_geojson(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Load Thailand provincial GeoJSON boundary file.
    """
    if filepath is None:
        root = get_project_root()
        filepath = os.path.join(root, "data", "thai_provinces.geojson")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Thailand GeoJSON file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    return geojson_data
