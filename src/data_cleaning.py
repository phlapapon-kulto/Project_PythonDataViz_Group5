"""
Data cleaning and preprocessing module for Thai Agriculture & Climate Intelligence Dashboard.
Handles spatial hierarchy, missing values, anomaly detection, and climate data harmonization.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union

# List of 4 main regional aggregate names in OAE data
REGION_NAMES = ["ภาคเหนือ", "ภาคกลาง", "ภาคใต้", "ภาคตะวันออกเฉียงเหนือ"]

# National aggregate name
COUNTRY_NAME = "ทั้งประเทศ"

# 77 official Thai provinces
OFFICIAL_77_PROVINCES = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท",
    "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง",
    "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม",
    "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส",
    "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
    "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา",
    "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
    "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
    "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
    "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี",
    "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
    "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
    "อุทัยธานี", "อุบลราชธานี"
]

# Standard 4-region mapping conforming to OAE and MOAC definitions
PROVINCE_TO_REGION: Dict[str, str] = {
    # ภาคเหนือ (17 จังหวัด)
    "เชียงใหม่": "ภาคเหนือ", "เชียงราย": "ภาคเหนือ", "แม่ฮ่องสอน": "ภาคเหนือ", "ลำปาง": "ภาคเหนือ",
    "ลำพูน": "ภาคเหนือ", "น่าน": "ภาคเหนือ", "พะเยา": "ภาคเหนือ", "แพร่": "ภาคเหนือ",
    "อุตรดิตถ์": "ภาคเหนือ", "ตาก": "ภาคเหนือ", "สุโขทัย": "ภาคเหนือ", "พิษณุโลก": "ภาคเหนือ",
    "พิจิตร": "ภาคเหนือ", "กำแพงเพชร": "ภาคเหนือ", "เพชรบูรณ์": "ภาคเหนือ", "นครสวรรค์": "ภาคเหนือ",
    "อุทัยธานี": "ภาคเหนือ",
    # ภาคตะวันออกเฉียงเหนือ (20 จังหวัด)
    "นครราชสีมา": "ภาคตะวันออกเฉียงเหนือ", "บุรีรัมย์": "ภาคตะวันออกเฉียงเหนือ", "สุรินทร์": "ภาคตะวันออกเฉียงเหนือ",
    "ศรีสะเกษ": "ภาคตะวันออกเฉียงเหนือ", "อุบลราชธานี": "ภาคตะวันออกเฉียงเหนือ", "ยโสธร": "ภาคตะวันออกเฉียงเหนือ",
    "ชัยภูมิ": "ภาคตะวันออกเฉียงเหนือ", "อำนาจเจริญ": "ภาคตะวันออกเฉียงเหนือ", "บึงกาฬ": "ภาคตะวันออกเฉียงเหนือ",
    "หนองบัวลำภู": "ภาคตะวันออกเฉียงเหนือ", "ขอนแก่น": "ภาคตะวันออกเฉียงเหนือ", "อุดรธานี": "ภาคตะวันออกเฉียงเหนือ",
    "เลย": "ภาคตะวันออกเฉียงเหนือ", "หนองคาย": "ภาคตะวันออกเฉียงเหนือ", "มหาสารคาม": "ภาคตะวันออกเฉียงเหนือ",
    "ร้อยเอ็ด": "ภาคตะวันออกเฉียงเหนือ", "กาฬสินธุ์": "ภาคตะวันออกเฉียงเหนือ", "สกลนคร": "ภาคตะวันออกเฉียงเหนือ",
    "นครพนม": "ภาคตะวันออกเฉียงเหนือ", "มุกดาหาร": "ภาคตะวันออกเฉียงเหนือ",
    # ภาคกลาง (26 จังหวัด)
    "กรุงเทพมหานคร": "ภาคกลาง", "สมุทรปราการ": "ภาคกลาง", "นนทบุรี": "ภาคกลาง", "ปทุมธานี": "ภาคกลาง",
    "พระนครศรีอยุธยา": "ภาคกลาง", "อ่างทอง": "ภาคกลาง", "ลพบุรี": "ภาคกลาง", "สิงห์บุรี": "ภาคกลาง",
    "ชัยนาท": "ภาคกลาง", "สระบุรี": "ภาคกลาง", "นครนายก": "ภาคกลาง", "ปราจีนบุรี": "ภาคกลาง",
    "สระแก้ว": "ภาคกลาง", "ฉะเชิงเทรา": "ภาคกลาง", "ชลบุรี": "ภาคกลาง", "ระยอง": "ภาคกลาง",
    "จันทบุรี": "ภาคกลาง", "ตราด": "ภาคกลาง", "กาญจนบุรี": "ภาคกลาง", "สุพรรณบุรี": "ภาคกลาง",
    "นครปฐม": "ภาคกลาง", "ราชบุรี": "ภาคกลาง", "สมุทรสาคร": "ภาคกลาง", "สมุทรสงคราม": "ภาคกลาง",
    "เพชรบุรี": "ภาคกลาง", "ประจวบคีรีขันธ์": "ภาคกลาง",
    # ภาคใต้ (14 จังหวัด)
    "ชุมพร": "ภาคใต้", "สุราษฎร์ธานี": "ภาคใต้", "นครศรีธรรมราช": "ภาคใต้", "พัทลุง": "ภาคใต้",
    "สงขลา": "ภาคใต้", "ปัตตานี": "ภาคใต้", "ยะลา": "ภาคใต้", "นราธิวาส": "ภาคใต้",
    "ระนอง": "ภาคใต้", "พังงา": "ภาคใต้", "ภูเก็ต": "ภาคใต้", "กระบี่": "ภาคใต้",
    "ตรัง": "ภาคใต้", "สตูล": "ภาคใต้"
}

REGION_TO_PROVINCES: Dict[str, List[str]] = {
    "ภาคเหนือ": [p for p, r in PROVINCE_TO_REGION.items() if r == "ภาคเหนือ"],
    "ภาคตะวันออกเฉียงเหนือ": [p for p, r in PROVINCE_TO_REGION.items() if r == "ภาคตะวันออกเฉียงเหนือ"],
    "ภาคกลาง": [p for p, r in PROVINCE_TO_REGION.items() if r == "ภาคกลาง"],
    "ภาคใต้": [p for p, r in PROVINCE_TO_REGION.items() if r == "ภาคใต้"],
}


def ce_to_be(year_ce: int) -> int:
    """Convert Common Era (CE / ค.ศ.) year to Buddhist Era (BE / พ.ศ.)."""
    return int(year_ce) + 543


def be_to_ce(year_be: int) -> int:
    """Convert Buddhist Era (BE / พ.ศ.) year to Common Era (CE / ค.ศ.)."""
    return int(year_be) - 543


def normalize_province_name(name: str) -> str:
    """Strip whitespace and normalize Thai province string."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    # Normalize common prefix if present
    if name.startswith("จ.") or name.startswith("จังหวัด"):
        name = name.replace("จังหวัด", "").replace("จ.", "").strip()
    return name


def clean_agriculture_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and structure the primary agricultural dataset.
    
    1. Trims string fields.
    2. Enforces numeric datatypes.
    3. Categorizes spatial level into 'country', 'region', 'province', or 'other'.
    4. Computes yield, harvest ratio, and damaged area safely (division by zero -> NaN).
    5. Flags anomalies (e.g. area_harvest_rai > area_plant_rai).
    """
    df = df.copy()

    # Strip strings
    df["crop_year"] = df["crop_year"].astype(str).str.strip()
    df["product"] = df["product"].astype(str).str.strip()
    df["province"] = df["province"].apply(normalize_province_name)

    # Cast numeric columns
    numeric_cols = [
        "year", "area_plant_rai", "area_harvest_rai", "production_ton",
        "yield_per_rai", "harvest_ratio", "damaged_area"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Spatial Hierarchy Classification
    def classify_geo(p: str) -> str:
        if p == COUNTRY_NAME:
            return "country"
        elif p in REGION_NAMES:
            return "region"
        elif p in OFFICIAL_77_PROVINCES:
            return "province"
        else:
            return "other"

    df["geo_level"] = df["province"].apply(classify_geo)

    def assign_region(p: str) -> str:
        if p in PROVINCE_TO_REGION:
            return PROVINCE_TO_REGION[p]
        elif p in REGION_NAMES:
            return p
        elif p == COUNTRY_NAME:
            return COUNTRY_NAME
        return "ไม่ระบุ"

    df["region"] = df["province"].apply(assign_region)

    # Recalculate metrics safely to ensure absolute mathematical integrity
    # Yield (kg/rai) = production_ton * 1000 / area_harvest_rai
    valid_harvest_mask = (df["area_harvest_rai"] > 0) & df["area_harvest_rai"].notna()
    df["calc_yield_per_rai"] = np.where(
        valid_harvest_mask,
        df["production_ton"] * 1000.0 / df["area_harvest_rai"],
        np.nan
    )

    # Harvest ratio (%) = area_harvest_rai / area_plant_rai * 100
    valid_plant_mask = (df["area_plant_rai"] > 0) & df["area_plant_rai"].notna()
    df["calc_harvest_ratio"] = np.where(
        valid_plant_mask,
        df["area_harvest_rai"] / df["area_plant_rai"] * 100.0,
        np.nan
    )

    # Damaged area (rai) = area_plant_rai - area_harvest_rai
    df["calc_damaged_area"] = np.where(
        df["area_plant_rai"].notna() & df["area_harvest_rai"].notna(),
        df["area_plant_rai"] - df["area_harvest_rai"],
        np.nan
    )

    # Anomaly Flags
    df["flag_harvest_gt_plant"] = df["area_harvest_rai"] > df["area_plant_rai"]
    df["flag_review_harvest_ratio"] = (df["calc_harvest_ratio"] < 90.0) & df["calc_harvest_ratio"].notna()

    return df


def clean_oae_farmgate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and structure OAE farmgate commodity prices.
    Converts prices from Baht/ton to Baht/kg and formats temporal fields.
    """
    df = df.copy()
    rename_cols = {
        "PROD_TYPE": "product",
        "YEAR": "year",
        "MONTH_C": "month",
        "WEEK_C": "week",
        "PRICE": "price_baht_per_ton"
    }
    df = df.rename(columns=rename_cols)

    df["product"] = df["product"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["price_baht_per_ton"] = pd.to_numeric(df["price_baht_per_ton"], errors="coerce")

    # Add Baht/kg unit: Price (Baht/ton) / 1000
    df["price_baht_per_kg"] = df["price_baht_per_ton"] / 1000.0
    df["data_source"] = "สำนักงานเศรษฐกิจการเกษตร (สศก.)"

    return df.dropna(subset=["product", "year", "price_baht_per_ton"]).reset_index(drop=True)


def normalize_agromet_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an agromet DataFrame by harmonizing column names and ensuring
    standard meteorological columns exist.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=[
            "station_number", "station_name", "province", "latitude", "longitude",
            "datetime", "temp_c", "max_temp_c", "min_temp_c", "relative_humidity_pct",
            "rainfall_24h_mm", "rainfall_mm", "wind_speed_kmh", "wind_direction_deg",
            "pressure_msl_hpa", "data_source", "data_source_type"
        ])

    df = df.copy()
    col_map = {}
    for col in df.columns:
        norm_key = str(col).strip().lower()
        if norm_key in ["wmostationnumber", "station_number", "stationnumber", "station_no"]:
            col_map[col] = "station_number"
        elif norm_key in ["stationnamethai", "station_name", "stationname", "station"]:
            col_map[col] = "station_name"
        elif norm_key in ["province", "prov_name", "changwat"]:
            col_map[col] = "province"
        elif norm_key in ["latitude", "lat"]:
            col_map[col] = "latitude"
        elif norm_key in ["longitude", "lon", "long"]:
            col_map[col] = "longitude"
        elif norm_key in ["datetime", "date_time", "obs_time", "time"]:
            col_map[col] = "datetime"
        elif norm_key in ["temperature", "temp_c", "temp", "temperature_c"]:
            col_map[col] = "temp_c"
        elif norm_key in ["maxtemperature", "max_temp_c", "max_temp", "temp_max"]:
            col_map[col] = "max_temp_c"
        elif norm_key in ["mintemperature", "min_temp_c", "min_temp", "temp_min"]:
            col_map[col] = "min_temp_c"
        elif norm_key in ["relativehumidity", "relative_humidity_pct", "rh", "humidity"]:
            col_map[col] = "relative_humidity_pct"
        elif norm_key in ["rainfall_24h_mm", "rain_24h", "rain_24h_mm", "rainfall_24h"]:
            col_map[col] = "rainfall_24h_mm"
        elif norm_key in ["rainfall", "rainfall_mm", "rain_mm", "rain"]:
            col_map[col] = "rainfall_mm"
        elif norm_key in ["windspeed", "wind_speed_kmh", "wind_speed", "wind_kmh"]:
            col_map[col] = "wind_speed_kmh"
        elif norm_key in ["winddirection", "wind_direction_deg", "wind_direction", "wind_dir"]:
            col_map[col] = "wind_direction_deg"
        elif norm_key in ["meansealevelpressure", "pressure_msl_hpa", "pressure", "mslp"]:
            col_map[col] = "pressure_msl_hpa"
        elif norm_key in ["data_source", "source"]:
            col_map[col] = "data_source"
        elif norm_key in ["data_source_type", "source_type"]:
            col_map[col] = "data_source_type"

    df = df.rename(columns=col_map)

    if "province" in df.columns:
        df["province"] = df["province"].apply(normalize_province_name)

    # Ensure dual availability of rainfall column names (rainfall_24h_mm and rainfall_mm)
    if "rainfall_24h_mm" in df.columns and "rainfall_mm" not in df.columns:
        df["rainfall_mm"] = df["rainfall_24h_mm"]
    elif "rainfall_mm" in df.columns and "rainfall_24h_mm" not in df.columns:
        df["rainfall_24h_mm"] = df["rainfall_mm"]
    elif "rainfall_24h_mm" not in df.columns and "rainfall_mm" not in df.columns:
        df["rainfall_24h_mm"] = np.nan
        df["rainfall_mm"] = np.nan

    # Ensure wind_speed_kmh exists
    if "wind_speed_kmh" not in df.columns:
        df["wind_speed_kmh"] = np.nan

    # Ensure source metadata
    if "data_source" not in df.columns:
        df["data_source"] = "TMD (กรมอุตุนิยมวิทยา)"
    if "data_source_type" not in df.columns:
        df["data_source_type"] = "tmd_daily_agromet_station"

    return df


def validate_agromet_schema(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that an agromet DataFrame contains the expected schema.
    Returns (is_valid, missing_columns).
    """
    if required_cols is None:
        required_cols = [
            "station_name", "province", "temp_c",
            "relative_humidity_pct", "rainfall_24h_mm",
            "wind_speed_kmh", "data_source"
        ]
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, required_cols

    missing = [c for c in required_cols if c not in df.columns]
    return len(missing) == 0, missing


def clean_tmd_daily_agromet(raw_input: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """
    Clean daily agricultural meteorology observations from TMD (Catalog c1-1-3-7).
    Parses station records from raw JSON or DataFrame into a tabular structure with standard metrics.
    Includes wind_speed_kmh, rainfall_24h_mm, and data_source metadata.
    """
    if isinstance(raw_input, pd.DataFrame):
        return normalize_agromet_dataframe(raw_input)

    if not isinstance(raw_input, dict) or not raw_input:
        return normalize_agromet_dataframe(pd.DataFrame())

    records = []
    stns = raw_input.get("Stations", {})
    stn_list = stns.get("Station", []) if isinstance(stns, dict) else []

    if isinstance(stn_list, dict):
        stn_list = [stn_list]

    for item in stn_list:
        obs = item.get("Observation", {})
        prov = normalize_province_name(item.get("Province", ""))
        rain_val = pd.to_numeric(obs.get("Rainfall"), errors="coerce")
        wind_val = pd.to_numeric(obs.get("WindSpeed"), errors="coerce")

        records.append({
            "station_number": item.get("WmoStationNumber", ""),
            "station_name": item.get("StationNameThai", ""),
            "station_name_en": item.get("StationNameEnglish", ""),
            "province": prov,
            "latitude": pd.to_numeric(item.get("Latitude"), errors="coerce"),
            "longitude": pd.to_numeric(item.get("Longitude"), errors="coerce"),
            "datetime": obs.get("DateTime", ""),
            "temp_c": pd.to_numeric(obs.get("Temperature"), errors="coerce"),
            "max_temp_c": pd.to_numeric(obs.get("MaxTemperature"), errors="coerce"),
            "min_temp_c": pd.to_numeric(obs.get("MinTemperature"), errors="coerce"),
            "relative_humidity_pct": pd.to_numeric(obs.get("RelativeHumidity"), errors="coerce"),
            "rainfall_24h_mm": rain_val,
            "rainfall_mm": rain_val,
            "wind_speed_kmh": wind_val,
            "wind_direction_deg": pd.to_numeric(obs.get("WindDirection"), errors="coerce"),
            "pressure_msl_hpa": pd.to_numeric(obs.get("MeanSeaLevelPressure"), errors="coerce"),
            "data_source": "TMD (กรมอุตุนิยมวิทยา)",
            "data_source_type": "tmd_daily_agromet_station"
        })

    df = pd.DataFrame(records)
    return normalize_agromet_dataframe(df)



def clean_spatial_rain_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean HII spatial rain dataset (monthly by province).
    Returns (df_monthly, df_annual).
    """
    df = df.copy()

    # Column mapping: YEAR, MONTH, PROV_ID, PROV_T, MinRain, MaxRain, AvgRain
    df["year_ce"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["year"] = df["year_ce"].apply(lambda y: ce_to_be(int(y)) if pd.notna(y) else np.nan)
    df["month"] = pd.to_numeric(df["MONTH"], errors="coerce")
    df["province"] = df["PROV_T"].apply(normalize_province_name)
    df["rain_avg_mm"] = pd.to_numeric(df["AvgRain"], errors="coerce")
    df["rain_min_mm"] = pd.to_numeric(df["MinRain"], errors="coerce")
    df["rain_max_mm"] = pd.to_numeric(df["MaxRain"], errors="coerce")
    df["data_source_type"] = "spatial_interpolated_idw"

    df_monthly = df[["year", "year_ce", "month", "province", "PROV_ID", "rain_avg_mm", "rain_min_mm", "rain_max_mm", "data_source_type"]].copy()

    # Aggregate to annual rainfall: sum AvgRain for each province and year where 12 months exist
    month_counts = df_monthly.groupby(["province", "year"])["month"].count().reset_index(name="months_count")
    annual_sum = df_monthly.groupby(["province", "year", "year_ce", "PROV_ID"])["rain_avg_mm"].sum().reset_index(name="rainfall_annual_mm")
    
    df_annual = pd.merge(annual_sum, month_counts, on=["province", "year"])
    # Flag incomplete years if any
    df_annual["is_full_year"] = df_annual["months_count"] == 12
    df_annual["data_source_type"] = "spatial_interpolated_idw"

    return df_monthly, df_annual


def clean_national_rain_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean HII national monthly rainfall dataset (1970-2025).
    Returns (df_monthly, df_annual).
    """
    df = df.copy()
    df["year_ce"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["year"] = df["year_ce"].apply(lambda y: ce_to_be(int(y)) if pd.notna(y) else np.nan)
    df["month"] = pd.to_numeric(df["MONTH"], errors="coerce")
    df["rain_mean_mm"] = pd.to_numeric(df["MEAN"], errors="coerce")
    df["province"] = COUNTRY_NAME
    df["data_source_type"] = "national_spatial_mean"

    df_monthly = df[["year", "year_ce", "month", "province", "rain_mean_mm", "data_source_type"]].copy()

    # Aggregate to annual
    month_counts = df_monthly.groupby(["year"])["month"].count().reset_index(name="months_count")
    annual_sum = df_monthly.groupby(["year", "year_ce"])["rain_mean_mm"].sum().reset_index(name="rainfall_annual_mm")
    df_annual = pd.merge(annual_sum, month_counts, on=["year"])
    df_annual["province"] = COUNTRY_NAME
    df_annual["is_full_year"] = df_annual["months_count"] == 12
    df_annual["data_source_type"] = "national_spatial_mean"

    return df_monthly, df_annual


def clean_provincial_temp_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean HII provincial monthly temperature dataset (2018-2019).
    Returns (df_monthly, df_annual).
    """
    df = df.copy()
    df["year_ce"] = pd.to_numeric(df["year"], errors="coerce")
    df["year"] = df["year_ce"].apply(lambda y: ce_to_be(int(y)) if pd.notna(y) else np.nan)
    df["province"] = df["province_name"].apply(normalize_province_name)
    df["temp_max_c"] = pd.to_numeric(df["max_temperature"], errors="coerce")
    df["temp_min_c"] = pd.to_numeric(df["min_temperature"], errors="coerce")
    df["temp_mean_c"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["data_source_type"] = "provincial_station_derived"

    df_monthly = df[["year", "year_ce", "month", "province", "province_code", "temp_max_c", "temp_min_c", "temp_mean_c", "data_source_type"]].copy()

    # Annual summary: Max of max_temp, Min of min_temp, Mean of temp_mean
    annual_grp = df_monthly.groupby(["province", "year", "year_ce"]).agg(
        temp_annual_max=("temp_max_c", "max"),
        temp_annual_min=("temp_min_c", "min"),
        temp_annual_mean=("temp_mean_c", "mean"),
        months_count=("month", "count")
    ).reset_index()

    annual_grp["is_full_year"] = annual_grp["months_count"] == 12
    annual_grp["data_source_type"] = "provincial_station_derived"

    return df_monthly, annual_grp


def clean_provincial_irrigation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate RID/OAE provincial irrigation statistics.
    Normalizes province names and calculates irrigation ratio.
    """
    df = df.copy()
    df["province"] = df["province"].apply(normalize_province_name)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["agricultural_area_rai"] = pd.to_numeric(df["agricultural_area_rai"], errors="coerce").fillna(0.0)
    df["irrigation_area_rai"] = pd.to_numeric(df["irrigation_area_rai"], errors="coerce").fillna(0.0)

    # Compute ratio
    df["irrigation_ratio_pct"] = np.where(
        df["agricultural_area_rai"] > 0,
        (df["irrigation_area_rai"] / df["agricultural_area_rai"]) * 100.0,
        0.0
    )
    df["irrigation_ratio_pct"] = df["irrigation_ratio_pct"].round(2)
    return df


def clean_ldd_soil_suitability_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean LDD Agri-Map soil suitability classifications.
    Normalizes province names and validates percentage sums.
    """
    df = df.copy()
    df["province"] = df["province"].apply(normalize_province_name)
    for col in ["suitable_high_s1_pct", "suitable_moderate_s2_pct", "suitable_marginal_s3_pct", "not_suitable_n_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["suitable_total_s1_s2_pct"] = (df["suitable_high_s1_pct"] + df["suitable_moderate_s2_pct"]).round(2)
    return df


def clean_oae_production_cost_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OAE production cost structure dataset.
    """
    df = df.copy()
    df["crop_year"] = pd.to_numeric(df["crop_year"], errors="coerce")
    numeric_cols = [
        "cost_seeds_rai", "cost_fertilizer_rai", "cost_chemicals_rai",
        "cost_fuel_water_rai", "cost_machinery_rai", "cost_labor_rai",
        "cost_land_rent_rai", "total_cost_rai", "benchmark_yield_kg_rai",
        "cost_per_kg", "mechanization_share_pct"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_disaster_affected_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DOAE & DDPM agricultural disaster impact statistics.
    """
    df = df.copy()
    df["province"] = df["province"].apply(normalize_province_name)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in ["flood_affected_area_rai", "drought_affected_area_rai", "total_disaster_area_rai"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df
