"""
Data loading module for Thai Agriculture & Climate Intelligence Dashboard.
Loads agricultural statistics, nationwide open climate datasets, OAE market prices,
TMD agromet observations, and geographic boundary datasets.

NOTE: The retired single-station Suphan Buri dataset has been completely removed
in accordance with project governance requirements.
"""

import os
import json
import requests
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


def load_oae_farmgate_prices(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load official farmgate prices dataset from the Office of Agricultural Economics (OAE).
    Default path: 'data/raw/oae_farmgate_prices_2561_2562.csv'
    """
    if filepath is None:
        root = get_project_root()
        filepath = os.path.join(root, "data", "raw", "oae_farmgate_prices_2561_2562.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OAE farmgate price file not found at: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    return df


def load_tmd_daily_agromet(
    filepath: Optional[str] = None,
    use_live_api: bool = True,
    timeout: int = 5
) -> Dict[str, Any]:
    """
    Load daily agricultural meteorology observations from the Thai Meteorological Department (TMD).
    Attempts live API call (Catalog c1-1-3-7) with automatic fallback to cached sample.
    """
    root = get_project_root()
    if filepath is None:
        filepath = os.path.join(root, "data", "raw", "tmd_agromet_sample.json")

    if use_live_api:
        api_url = "https://data.tmd.go.th/api/WeatherTodayByAgro/v1/?uid=api&ukey=api12345&format=json"
        try:
            r = requests.get(api_url, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if "Stations" in data:
                    return data
        except Exception:
            pass  # Fall back to local cache if network/API fails

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


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
