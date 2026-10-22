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
    Reads all relevant sheets and combines them if multiple sheets exist.
    Default path: 'สถิติการผลิตสินค้าเกษตรรายปี 2550-2562.xlsx' in project root.
    """
    if filepath is None:
        root = get_project_root()
        filepath = os.path.join(root, "สถิติการผลิตสินค้าเกษตรรายปี 2550-2562.xlsx")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agriculture data file not found at: {filepath}")

    # Inspect all sheets and load all valid data sheets
    excel_file = pd.ExcelFile(filepath)
    sheet_dfs = []
    for sheet_name in excel_file.sheet_names:
        try:
            sheet_df = pd.read_excel(excel_file, sheet_name=sheet_name)
            if not sheet_df.empty and "product" in sheet_df.columns:
                sheet_dfs.append(sheet_df)
        except Exception:
            continue

    if sheet_dfs:
        df = pd.concat(sheet_dfs, ignore_index=True)
    else:
        df = pd.read_excel(filepath, sheet_name=0)

    return df


def query_gistda_check_crop(
    lat: float,
    lon: float,
    api_key: Optional[str] = None,
    timeout: int = 8
) -> Dict[str, Any]:
    """
    Query GISTDA Open Data Gateway (Catalog agriculture-07: check-crop API)
    to detect crop type from satellite observation at specified lat/lon.

    Strictly complies with Data Governance:
    - Zero mock/fabricated data
    - Real HTTP request with timeout handling
    - Explicit detection of missing key, HTTP 401/403, and network timeouts
    """
    import urllib.request
    import urllib.error

    # Coordinate boundary validation for Thailand
    if not (5.0 <= lat <= 21.5 and 96.5 <= lon <= 106.5):
        return {
            "status": "invalid_coordinates",
            "message": f"พิกัด ({lat:.4f}, {lon:.4f}) อยู่นอกขอบเขตราชอาณาจักรไทย (ละติจูด 5.0–21.5°N, ลองจิจูด 96.5–106.5°E)",
            "data": None
        }

    # Resolve API Key
    resolved_key = (api_key or "").strip()
    if not resolved_key:
        resolved_key = os.getenv("GISTDA_API_KEY", "").strip()

    if not resolved_key:
        try:
            import streamlit as st
            resolved_key = st.secrets.get("GISTDA_API_KEY", "").strip()
        except Exception:
            pass

    if not resolved_key:
        return {
            "status": "no_key",
            "message": "ต้องระบุ GISTDA API Key ที่ลงทะเบียนผ่าน opendata.gistda.or.th (ระบบไม่สร้างข้อมูลจำลองทดแทนตามหลักธรรมาภิบาลข้อมูล)",
            "data": None
        }

    url = f"https://api-gateway.gistda.or.th/api/2.0/resources/gi-service/v2.2/agriculture/check-crop?lat={lat}&lon={lon}&api_key={resolved_key}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Thai-Agri-Climate-Intelligence/2.0",
            "Accept": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            resp_code = response.status
            content = response.read().decode("utf-8")
            data = json.loads(content)

            # Check if crop information was identified
            crop_info = None
            if isinstance(data, dict):
                crop_info = data.get("crop") or data.get("data") or data.get("result") or data.get("crop_name")
            elif isinstance(data, list) and len(data) > 0:
                crop_info = data[0]

            if crop_info:
                return {
                    "status": "success",
                    "code": resp_code,
                    "message": "ตรวจพบข้อมูลพื้นที่ปลูกพืชเศรษฐกิจจากดาวเทียมสำเร็จ",
                    "crop_data": crop_info,
                    "raw": data
                }
            else:
                return {
                    "status": "not_found",
                    "code": resp_code,
                    "message": f"ไม่พบข้อมูลพื้นที่ปลูกพืชเศรษฐกิจ ณ พิกัด ({lat:.4f}, {lon:.4f}) ในฐานข้อมูลดาวเทียมล่าสุด",
                    "crop_data": None,
                    "raw": data
                }

    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            msg = f"การเข้าถึงถูกปฏิเสธ (HTTP {e.code}): API Key ไม่ถูกต้อง หมดอายุ หรือไม่มีสิทธิ์เข้าถึงบริการนี้ กรุณาตรวจสอบคีย์ที่ opendata.gistda.or.th"
        elif e.code == 404:
            msg = f"ไม่พบ Endpoint บริการบนเซิร์ฟเวอร์ GISTDA (HTTP 404): {url.split('?')[0]}"
        elif e.code >= 500:
            msg = f"เซิร์ฟเวอร์ GISTDA ขัดข้องชั่วคราว (HTTP {e.code}) กรุณาลองใหม่อีกครั้งในภายหลัง"
        else:
            msg = f"ข้อผิดพลาด HTTP {e.code}: {e.reason}"
        return {
            "status": "http_error",
            "code": e.code,
            "message": msg,
            "data": None
        }

    except urllib.error.URLError as e:
        reason_str = str(e.reason).lower()
        if "timed out" in reason_str or "timeout" in reason_str:
            msg = f"หมดเวลาการเชื่อมต่อ (Timeout เกิน {timeout} วินาที) เครือข่ายอาจมีความล่าช้า กรุณากดลองใหม่อีกครั้ง"
        else:
            msg = f"ไม่สามารถเชื่อมต่อกับ GISTDA API Gateway ได้: {e.reason}"
        return {
            "status": "network_error",
            "message": msg,
            "data": None
        }

    except json.JSONDecodeError:
        return {
            "status": "parse_error",
            "message": "เซิร์ฟเวอร์ GISTDA ส่งคืนข้อมูลที่ไม่ใช่รูปแบบ JSON ที่ถูกต้อง",
            "data": None
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"เกิดข้อผิดพลาดในการตรวจสอบ: {str(e)}",
            "data": None
        }


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
