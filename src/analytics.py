"""
Analytics module for Thai Agriculture & Climate Intelligence Dashboard.
Implements correlation analysis, statistical tests, quadrant classification,
ranking algorithms, and automated data-driven interpretation generation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple


def join_climate_agriculture_suphan(
    df_ag: pd.DataFrame,
    df_climate_suphan: pd.DataFrame,
    product: Optional[str] = None
) -> pd.DataFrame:
    """
    Join Suphan Buri station climate data with agricultural statistics for Suphan Buri province.
    Only joins on identical common years (2550-2562).
    Never propagates Suphan Buri station data to any other province.
    """
    ag_suphan = df_ag[(df_ag["province"] == "สุพรรณบุรี") & (df_ag["geo_level"] == "province")].copy()
    if product and product != "ทั้งหมด":
        ag_suphan = ag_suphan[ag_suphan["product"] == product]
    
    # Overlapping years
    joined = pd.merge(
        ag_suphan,
        df_climate_suphan,
        on=["province", "year"],
        how="inner"
    )
    
    return joined.sort_values(by="year").reset_index(drop=True)


def join_climate_agriculture_nationwide(
    df_ag: pd.DataFrame,
    df_rain_annual: pd.DataFrame,
    df_temp_annual: Optional[pd.DataFrame] = None,
    year: Optional[int] = None,
    product: Optional[str] = None
) -> pd.DataFrame:
    """
    Join HII provincial annual rainfall (and temperature) with provincial agricultural records.
    Filters exclusively for provincial level records to prevent double counting.
    """
    ag_prov = df_ag[df_ag["geo_level"] == "province"].copy()
    if year:
        ag_prov = ag_prov[ag_prov["year"] == year]
    if product and product != "ทั้งหมด":
        ag_prov = ag_prov[ag_prov["product"] == product]
        
    joined = pd.merge(
        ag_prov,
        df_rain_annual[["province", "year", "rainfall_annual_mm", "data_source_type"]],
        on=["province", "year"],
        how="inner"
    )
    
    if df_temp_annual is not None and not df_temp_annual.empty:
        joined = pd.merge(
            joined,
            df_temp_annual[["province", "year", "temp_annual_max", "temp_annual_min", "temp_annual_mean"]],
            on=["province", "year"],
            how="left"
        )
        
    return joined


def calculate_bivariate_stats(
    x: pd.Series,
    y: pd.Series
) -> Dict[str, Any]:
    """
    Calculate Pearson and Spearman correlation coefficients and p-values.
    Safely handles missing values, low sample size, and zero-variance.
    """
    valid_df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(valid_df)
    
    if n < 3:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "is_valid": False,
            "message": f"จำนวนคู่ข้อมูลมีเพียง {n} จุด ซึ่งไม่เพียงพอสำหรับการวิเคราะห์สหสัมพันธ์ (ต้องการอย่างน้อย 3 จุด)"
        }
        
    x_clean = valid_df["x"]
    y_clean = valid_df["y"]
    
    if x_clean.std() == 0 or y_clean.std() == 0:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "is_valid": False,
            "message": "ตัวแปรตัวใดตัวหนึ่งมีค่าคงที่ (ความแปรปรวนเป็นศูนย์) ไม่สามารถคำนวณสหสัมพันธ์ได้"
        }
        
    pr, pp = stats.pearsonr(x_clean, y_clean)
    sr, sp = stats.spearmanr(x_clean, y_clean)
    
    return {
        "n": n,
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "is_valid": True,
        "message": "คำนวณสำเร็จ"
    }


def generate_climate_interpretation(
    bivariate_result: Dict[str, Any],
    crop_name: str,
    climate_var_name: str,
    ag_var_name: str,
    geo_scope: str = "จังหวัดสุพรรณบุรี",
    year_range: str = "2550-2562"
) -> str:
    """
    Generate an academic, data-driven interpretation of the climate-agriculture relationship.
    Avoids unfounded causal assertions and explicitly mentions statistical caveats.
    """
    if not bivariate_result.get("is_valid", False):
        return f"**ข้อจำกัดของข้อมูล:** {bivariate_result.get('message', 'ข้อมูลไม่เพียงพอสำหรับการสรุปผล')}"
        
    r = bivariate_result["pearson_r"]
    p = bivariate_result["pearson_p"]
    n = bivariate_result["n"]
    
    # Strength categorization
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = "ระดับสูง (Strong)"
    elif abs_r >= 0.4:
        strength = "ระดับปานกลาง (Moderate)"
    elif abs_r >= 0.2:
        strength = "ระดับต่ำ (Weak)"
    else:
        strength = "น้อยมากหรือแทบไม่มีความสัมพันธ์ (Negligible)"
        
    direction = "เชิงบวก (Positive)" if r > 0 else "เชิงลบ (Negative)"
    significance = "มีนัยสำคัญทางสถิติที่ระดับ 0.05 (p < 0.05)" if p < 0.05 else "ยังไม่มีนัยสำคัญทางสถิติที่ระดับ 0.05 (p ≥ 0.05)"
    
    text = (
        f"**ผลการวิเคราะห์ทางสถิติ:**\n"
        f"- จากการวิเคราะห์ข้อมูลจำนวน **{n} ปีคู่ข้อมูล** ในช่วงปี **พ.ศ. {year_range}** ({geo_scope}) สำหรับ **{crop_name}**\n"
        f"- ค่าสัมประสิทธิ์สหสัมพันธ์แบบเพียร์สัน ($r$) ระหว่าง **{climate_var_name}** กับ **{ag_var_name}** เท่ากับ **{r:.3f}** (p-value = **{p:.4f}**)\n"
        f"- บ่งชี้ความสัมพันธ์{direction}ใน{strength} โดยความสัมพันธ์นี้{significance}\n\n"
        f"**ข้อควรระวังทางวิชาการและข้อจำกัด:**\n"
        f"1. **Correlation ไม่ใช่ Causation:** ความสัมพันธ์ทางสถิตินี้ไม่ได้พิสูจน์ว่า {climate_var_name} เป็นสาเหตุโดยตรงของการเปลี่ยนแปลง {ag_var_name} เนื่องจากผลผลิตทางการเกษตรยังขึ้นอยู่กับระบบชลประทาน เทคโนโลยีการเพาะปลูก ปุ๋ย โรคแมลง และปัจจัยทางเศรษฐกิจ\n"
        f"2. **ขนาดกลุ่มตัวอย่าง:** ข้อมูลมีจำนวน {n} จุด การสรุปผลควรใช้เป็นสมมติฐานเบื้องต้นเพื่อการศึกษาต่อยอดร่วมกับผู้เชี่ยวชาญด้านปฐพีวิทยาและพืชศาสตร์ต่อไป"
    )
    return text


def classify_area_yield_quadrants(
    df: pd.DataFrame,
    area_col: str = "area_plant_rai",
    yield_col: str = "calc_yield_per_rai",
    label_col: str = "product"
) -> Tuple[pd.DataFrame, float, float]:
    """
    Classify items into 4 quadrants based on median Planted Area and median Yield:
    - Q1 (High Area, High Yield): พื้นที่ปลูกมาก & ผลผลิตต่อไร่สูง (เสาหลักการผลิต)
    - Q2 (Low Area, High Yield): พื้นที่ปลูกน้อย & ผลผลิตต่อไร่สูง (พืชผลตอบแทนต่อไร่สูง / พืชเฉพาะทาง)
    - Q3 (Low Area, Low Yield): พื้นที่ปลูกน้อย & ผลผลิตต่อไร่ต่ำ (พืชขนาดเล็ก / ความเสี่ยงผลตอบแทนต่ำ)
    - Q4 (High Area, Low Yield): พื้นที่ปลูกมาก & ผลผลิตต่อไร่ต่ำ (พื้นที่กว้างแต่ Yield ต่ำ / มีโอกาสพัฒนาเพิ่มประสิทธิภาพ)
    """
    clean_df = df.dropna(subset=[area_col, yield_col]).copy()
    if clean_df.empty:
        return clean_df, 0.0, 0.0
        
    area_median = float(clean_df[area_col].median())
    yield_median = float(clean_df[yield_col].median())
    
    def assign_quad(row):
        is_high_area = row[area_col] >= area_median
        is_high_yield = row[yield_col] >= yield_median
        if is_high_area and is_high_yield:
            return "Q1: พื้นที่มาก - Yield สูง"
        elif not is_high_area and is_high_yield:
            return "Q2: พื้นที่น้อย - Yield สูง"
        elif not is_high_area and not is_high_yield:
            return "Q3: พื้นที่น้อย - Yield ต่ำ"
        else:
            return "Q4: พื้นที่มาก - Yield ต่ำ"
            
    clean_df["quadrant"] = clean_df.apply(assign_quad, axis=1)
    return clean_df, area_median, yield_median


def get_top_provinces(
    df: pd.DataFrame,
    metric: str = "production_ton",
    top_n: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Get top N provinces by specified agricultural metric, strictly excluding aggregates.
    """
    prov_df = df[df["geo_level"] == "province"].copy()
    if prov_df.empty or metric not in prov_df.columns:
        return pd.DataFrame()
        
    # If grouped by province
    grouped = prov_df.groupby("province").agg(
        production_ton=("production_ton", "sum"),
        area_plant_rai=("area_plant_rai", "sum"),
        area_harvest_rai=("area_harvest_rai", "sum")
    ).reset_index()
    
    # Calculate weighted yield and harvest ratio
    grouped["calc_yield_per_rai"] = np.where(
        grouped["area_harvest_rai"] > 0,
        grouped["production_ton"] * 1000.0 / grouped["area_harvest_rai"],
        np.nan
    )
    grouped["calc_harvest_ratio"] = np.where(
        grouped["area_plant_rai"] > 0,
        grouped["area_harvest_rai"] / grouped["area_plant_rai"] * 100.0,
        np.nan
    )
    
    res = grouped.sort_values(by=metric, ascending=ascending).head(top_n).reset_index(drop=True)
    return res


def get_top_crops(
    df: pd.DataFrame,
    metric: str = "production_ton",
    top_n: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Get top N crops by specified agricultural metric, strictly avoiding double counting.
    """
    # Use provincial records sum to ensure consistency
    prov_df = df[df["geo_level"] == "province"].copy()
    if prov_df.empty or metric not in prov_df.columns:
        return pd.DataFrame()
        
    grouped = prov_df.groupby("product").agg(
        production_ton=("production_ton", "sum"),
        area_plant_rai=("area_plant_rai", "sum"),
        area_harvest_rai=("area_harvest_rai", "sum")
    ).reset_index()
    
    grouped["calc_yield_per_rai"] = np.where(
        grouped["area_harvest_rai"] > 0,
        grouped["production_ton"] * 1000.0 / grouped["area_harvest_rai"],
        np.nan
    )
    grouped["calc_harvest_ratio"] = np.where(
        grouped["area_plant_rai"] > 0,
        grouped["area_harvest_rai"] / grouped["area_plant_rai"] * 100.0,
        np.nan
    )
    
    res = grouped.sort_values(by=metric, ascending=ascending).head(top_n).reset_index(drop=True)
    return res
