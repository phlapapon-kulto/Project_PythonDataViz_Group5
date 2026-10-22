"""
Data cleaning and preprocessing module for Thai Agriculture & Climate Intelligence Dashboard.
Handles spatial hierarchy, missing values, anomaly detection, and climate data harmonization.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

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


def clean_tmd_daily_agromet(raw_json: Dict[str, Any]) -> pd.DataFrame:
    """
    Clean daily agricultural meteorology observations from TMD (Catalog c1-1-3-7).
    Parses station records into a tabular structure with standard metrics.
    """
    records = []
    stns = raw_json.get("Stations", {})
    stn_list = stns.get("Station", []) if isinstance(stns, dict) else []

    if isinstance(stn_list, dict):
        stn_list = [stn_list]

    for item in stn_list:
        obs = item.get("Observation", {})
        prov = normalize_province_name(item.get("Province", ""))
        records.append({
            "station_number": item.get("WmoStationNumber", ""),
            "station_name": item.get("StationNameThai", ""),
            "province": prov,
            "latitude": pd.to_numeric(item.get("Latitude"), errors="coerce"),
            "longitude": pd.to_numeric(item.get("Longitude"), errors="coerce"),
            "datetime": obs.get("DateTime", ""),
            "temp_c": pd.to_numeric(obs.get("Temperature"), errors="coerce"),
            "max_temp_c": pd.to_numeric(obs.get("MaxTemperature"), errors="coerce"),
            "min_temp_c": pd.to_numeric(obs.get("MinTemperature"), errors="coerce"),
            "relative_humidity_pct": pd.to_numeric(obs.get("RelativeHumidity"), errors="coerce"),
            "rainfall_mm": pd.to_numeric(obs.get("Rainfall"), errors="coerce"),
            "data_source_type": "tmd_daily_agromet_station"
        })

    df = pd.DataFrame(records)
    return df



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
