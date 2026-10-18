"""
Feature engineering and metrics calculation module.
Implements standardized academic and business definitions for agricultural KPIs,
ensuring division by zero produces NaN, and preventing incorrect arithmetic averages.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union


def calculate_weighted_yield(
    df: pd.DataFrame,
    prod_col: str = "production_ton",
    harvest_col: str = "area_harvest_rai"
) -> float:
    """
    Calculate weighted average yield across multiple records.
    Formula: Total Production (tons) * 1,000 / Total Harvested Area (rai)
    
    Returns np.nan if total harvested area <= 0 or missing.
    Never computes a simple unweighted arithmetic average of yields.
    """
    if df.empty:
        return np.nan
    
    total_prod = df[prod_col].sum(skipna=True)
    total_harvest = df[harvest_col].sum(skipna=True)
    
    if pd.isna(total_harvest) or total_harvest <= 0:
        return np.nan
    
    return float(total_prod * 1000.0 / total_harvest)


def calculate_aggregated_harvest_ratio(
    df: pd.DataFrame,
    harvest_col: str = "area_harvest_rai",
    plant_col: str = "area_plant_rai"
) -> float:
    """
    Calculate aggregated harvest ratio across multiple records.
    Formula: Total Harvested Area (rai) / Total Planted Area (rai) * 100
    
    Returns np.nan if total planted area <= 0 or missing.
    """
    if df.empty:
        return np.nan
    
    total_harvest = df[harvest_col].sum(skipna=True)
    total_plant = df[plant_col].sum(skipna=True)
    
    if pd.isna(total_plant) or total_plant <= 0:
        return np.nan
    
    return float(total_harvest / total_plant * 100.0)


def calculate_unharvested_area(
    df: pd.DataFrame,
    plant_col: str = "area_plant_rai",
    harvest_col: str = "area_harvest_rai"
) -> float:
    """
    Calculate total unharvested (damaged/abandoned) area.
    Formula: Total Planted Area - Total Harvested Area
    """
    if df.empty:
        return np.nan
    
    total_plant = df[plant_col].sum(skipna=True)
    total_harvest = df[harvest_col].sum(skipna=True)
    
    return float(total_plant - total_harvest)


def calculate_yoy_growth(series: pd.Series) -> pd.Series:
    """
    Calculate Year-over-Year (YoY) percentage change for a time-series.
    Formula: (Current - Previous) / Previous * 100
    
    If previous value is 0, negative, or NaN, returns NaN.
    """
    prev = series.shift(1)
    valid_mask = (prev > 0) & prev.notna() & series.notna()
    yoy = np.where(valid_mask, (series - prev) / prev * 100.0, np.nan)
    return pd.Series(yoy, index=series.index)


def calculate_cv(series: pd.Series, min_periods: int = 3) -> float:
    """
    Calculate Coefficient of Variation (CV) percentage.
    Formula: Standard Deviation / Mean * 100
    
    Requires at least `min_periods` non-null observations.
    Returns np.nan if mean <= 0 or count < min_periods.
    """
    clean_series = series.dropna()
    if len(clean_series) < min_periods:
        return np.nan
    
    mean_val = clean_series.mean()
    if mean_val <= 0:
        return np.nan
    
    std_val = clean_series.std(ddof=1)
    return float((std_val / mean_val) * 100.0)


def aggregate_agricultural_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary KPIs for a given agricultural subset DataFrame.
    """
    if df.empty:
        return {
            "total_production_ton": 0.0,
            "total_plant_rai": 0.0,
            "total_harvest_rai": 0.0,
            "weighted_yield_kg_rai": np.nan,
            "harvest_ratio_pct": np.nan,
            "unharvested_area_rai": 0.0,
            "records_count": 0
        }
    
    total_prod = float(df["production_ton"].sum(skipna=True))
    total_plant = float(df["area_plant_rai"].sum(skipna=True))
    total_harvest = float(df["area_harvest_rai"].sum(skipna=True))
    weighted_yield = calculate_weighted_yield(df)
    harvest_ratio = calculate_aggregated_harvest_ratio(df)
    unharvested = total_plant - total_harvest
    
    return {
        "total_production_ton": total_prod,
        "total_plant_rai": total_plant,
        "total_harvest_rai": total_harvest,
        "weighted_yield_kg_rai": weighted_yield,
        "harvest_ratio_pct": harvest_ratio,
        "unharvested_area_rai": unharvested,
        "records_count": len(df)
    }


def compute_yoy_dataframe(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute YoY changes across continuous years for specified metrics and group columns.
    Ensures years are continuous (delta = 1 year) before assigning YoY change.
    """
    if metric_cols is None:
        metric_cols = ["production_ton", "area_plant_rai", "area_harvest_rai", "calc_yield_per_rai"]
    
    df_sorted = df.sort_values(by=group_cols + ["year"]).copy()
    
    for metric in metric_cols:
        if metric in df_sorted.columns:
            # Shift within group
            prev_val = df_sorted.groupby(group_cols)[metric].shift(1)
            prev_yr = df_sorted.groupby(group_cols)["year"].shift(1)
            
            # Continuous year check: year - prev_yr == 1
            is_continuous = (df_sorted["year"] - prev_yr) == 1
            is_valid_denom = (prev_val > 0) & prev_val.notna() & df_sorted[metric].notna()
            
            yoy_col_name = f"{metric}_yoy_pct"
            df_sorted[yoy_col_name] = np.where(
                is_continuous & is_valid_denom,
                (df_sorted[metric] - prev_val) / prev_val * 100.0,
                np.nan
            )
            
    return df_sorted


def compute_cv_summary(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str = "production_ton",
    min_years: int = 3
) -> pd.DataFrame:
    """
    Compute Coefficient of Variation (CV) summary table grouped by `group_col` (e.g., 'product' or 'province').
    """
    records = []
    for grp_val, sub_df in df.groupby(group_col):
        vals = sub_df[metric_col].dropna()
        n_years = len(vals)
        if n_years >= min_years:
            mean_v = vals.mean()
            std_v = vals.std(ddof=1)
            cv_v = (std_v / mean_v * 100.0) if mean_v > 0 else np.nan
            min_yr = int(sub_df["year"].min())
            max_yr = int(sub_df["year"].max())
            records.append({
                group_col: grp_val,
                "metric": metric_col,
                "n_years": n_years,
                "year_range": f"{min_yr}-{max_yr}",
                "mean": mean_v,
                "std": std_v,
                "cv_pct": cv_v
            })
            
    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df = res_df.sort_values(by="cv_pct", ascending=False).reset_index(drop=True)
    return res_df
