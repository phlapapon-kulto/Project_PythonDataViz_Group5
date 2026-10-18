"""
Comprehensive unit test suite for Thai Agriculture & Climate Intelligence Dashboard.
Tests mathematical formulas, data cleaning, spatial hierarchies, double-count prevention,
and climate data harmonization.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_cleaning import (
    ce_to_be, be_to_ce, normalize_province_name, clean_agriculture_data,
    clean_suphan_climate_data, clean_spatial_rain_data,
    OFFICIAL_77_PROVINCES, REGION_NAMES, COUNTRY_NAME
)
from src.feature_engineering import (
    calculate_weighted_yield, calculate_aggregated_harvest_ratio,
    calculate_unharvested_area, calculate_yoy_growth, calculate_cv,
    aggregate_agricultural_kpis, compute_yoy_dataframe, compute_cv_summary
)
from src.analytics import (
    calculate_bivariate_stats, classify_area_yield_quadrants,
    join_climate_agriculture_suphan, get_top_provinces, get_top_crops,
    generate_climate_interpretation
)
from src.filters import apply_ag_filters


class TestDataCleaning:
    def test_year_conversions(self):
        assert ce_to_be(2018) == 2561
        assert ce_to_be(2019) == 2562
        assert be_to_ce(2562) == 2019
        assert be_to_ce(2550) == 2007

    def test_normalize_province_name(self):
        assert normalize_province_name("  เชียงใหม่  ") == "เชียงใหม่"
        assert normalize_province_name("จ.สุพรรณบุรี") == "สุพรรณบุรี"
        assert normalize_province_name("จังหวัดขอนแก่น") == "ขอนแก่น"

    def test_spatial_hierarchy_classification(self):
        raw_df = pd.DataFrame({
            "crop_year": ["2549/50", "2549/50", "2549/50", "2549/50"],
            "year": [2550, 2550, 2550, 2550],
            "product": ["ข้าวนาปี", "ข้าวนาปี", "ข้าวนาปี", "ข้าวนาปี"],
            "province": ["ทั้งประเทศ", "ภาคเหนือ", "เชียงใหม่", "อื่นๆ"],
            "area_plant_rai": [1000.0, 500.0, 100.0, 10.0],
            "area_harvest_rai": [900.0, 450.0, 95.0, 8.0],
            "production_ton": [450.0, 225.0, 47.5, 4.0],
            "yield_per_rai": [500.0, 500.0, 500.0, 500.0],
            "harvest_ratio": [90.0, 90.0, 95.0, 80.0],
            "damaged_area": [100.0, 50.0, 5.0, 2.0]
        })
        cleaned = clean_agriculture_data(raw_df)
        assert cleaned.loc[0, "geo_level"] == "country"
        assert cleaned.loc[1, "geo_level"] == "region"
        assert cleaned.loc[2, "geo_level"] == "province"
        assert cleaned.loc[3, "geo_level"] == "other"


class TestMathematicalFormulas:
    def test_weighted_yield_calculation(self):
        # 2 provinces:
        # P1: 100 tons, 200 rai -> yield = 500 kg/rai
        # P2: 900 tons, 1000 rai -> yield = 900 kg/rai
        # Simple average = (500 + 900) / 2 = 700 kg/rai
        # Weighted average = (1000 tons * 1000) / 1200 rai = 833.33 kg/rai
        df = pd.DataFrame({
            "production_ton": [100.0, 900.0],
            "area_harvest_rai": [200.0, 1000.0]
        })
        wy = calculate_weighted_yield(df)
        assert pytest.approx(wy, 0.01) == 833.33

    def test_weighted_yield_zero_division(self):
        df_zero = pd.DataFrame({
            "production_ton": [0.0],
            "area_harvest_rai": [0.0]
        })
        assert np.isnan(calculate_weighted_yield(df_zero))

    def test_harvest_ratio_calculation(self):
        df = pd.DataFrame({
            "area_harvest_rai": [850.0],
            "area_plant_rai": [1000.0]
        })
        hr = calculate_aggregated_harvest_ratio(df)
        assert hr == 85.0

    def test_harvest_ratio_zero_plant_area(self):
        df = pd.DataFrame({
            "area_harvest_rai": [0.0],
            "area_plant_rai": [0.0]
        })
        assert np.isnan(calculate_aggregated_harvest_ratio(df))

    def test_unharvested_area(self):
        df = pd.DataFrame({
            "area_plant_rai": [1000.0, 500.0],
            "area_harvest_rai": [850.0, 450.0]
        })
        assert calculate_unharvested_area(df) == 200.0

    def test_yoy_growth(self):
        s = pd.Series([100.0, 120.0, 150.0, 0.0, 50.0])
        yoy = calculate_yoy_growth(s)
        assert np.isnan(yoy.iloc[0])
        assert pytest.approx(yoy.iloc[1], 0.01) == 20.0
        assert pytest.approx(yoy.iloc[2], 0.01) == 25.0
        assert yoy.iloc[3] == -100.0
        # When previous was 0, YoY should be NaN
        assert np.isnan(yoy.iloc[4])

    def test_cv_calculation(self):
        # Stable series: [100, 100, 100] -> CV = 0%
        s_stable = pd.Series([100.0, 100.0, 100.0])
        assert calculate_cv(s_stable) == 0.0

        # Sample with std = 10, mean = 100 -> CV = 10%
        s_var = pd.Series([90.0, 100.0, 110.0])
        assert pytest.approx(calculate_cv(s_var), 0.01) == 10.0

        # Too few points (< 3)
        s_few = pd.Series([100.0, 110.0])
        assert np.isnan(calculate_cv(s_few, min_periods=3))


class TestDoubleCountingPrevention:
    def test_aggregate_double_count_prevention(self):
        # Ensure country or regional rows are NOT summed with provincial rows
        df = pd.DataFrame({
            "province": ["ทั้งประเทศ", "ภาคกลาง", "สุพรรณบุรี", "นครปฐม"],
            "geo_level": ["country", "region", "province", "province"],
            "year": [2560, 2560, 2560, 2560],
            "product": ["ข้าวนาปี", "ข้าวนาปี", "ข้าวนาปี", "ข้าวนาปี"],
            "production_ton": [1000.0, 500.0, 300.0, 200.0],
            "area_plant_rai": [2000.0, 1000.0, 600.0, 400.0],
            "area_harvest_rai": [1900.0, 950.0, 570.0, 380.0]
        })
        # Filtering strictly for province level
        prov_only = apply_ag_filters(df, geo_level="province")
        kpis = aggregate_agricultural_kpis(prov_only)
        # Sum of provinces: 300 + 200 = 500 tons (NOT 1000 + 500 + 500 = 2000 tons)
        assert kpis["total_production_ton"] == 500.0
        assert kpis["total_plant_rai"] == 1000.0
        assert kpis["total_harvest_rai"] == 950.0


class TestAnalyticsAndJoins:
    def test_bivariate_stats(self):
        x = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        y = pd.Series([15.0, 25.0, 35.0, 45.0, 55.0])
        res = calculate_bivariate_stats(x, y)
        assert res["is_valid"] is True
        assert res["n"] == 5
        assert pytest.approx(res["pearson_r"], 0.001) == 1.0
        assert res["pearson_p"] < 0.001

    def test_quadrant_classification(self):
        df = pd.DataFrame({
            "product": ["Crop A", "Crop B", "Crop C", "Crop D"],
            "area_plant_rai": [1000.0, 100.0, 50.0, 800.0],
            "calc_yield_per_rai": [500.0, 600.0, 200.0, 150.0]
        })
        q_df, a_med, y_med = classify_area_yield_quadrants(df)
        assert not q_df.empty
        assert "quadrant" in q_df.columns
        assert len(q_df) == 4

    def test_suphan_climate_join_integrity(self):
        df_ag = pd.DataFrame({
            "province": ["สุพรรณบุรี", "สุพรรณบุรี", "เชียงใหม่"],
            "geo_level": ["province", "province", "province"],
            "year": [2550, 2551, 2550],
            "product": ["ข้าวนาปี", "ข้าวนาปี", "ข้าวนาปี"],
            "production_ton": [500.0, 520.0, 300.0],
            "area_harvest_rai": [1000.0, 1040.0, 600.0]
        })
        df_clim = pd.DataFrame({
            "province": ["สุพรรณบุรี", "สุพรรณบุรี"],
            "year": [2550, 2551],
            "rainfall_mm": [800.0, 950.0],
            "temp_avg_c": [28.5, 28.0]
        })
        joined = join_climate_agriculture_suphan(df_ag, df_clim, product="ข้าวนาปี")
        assert len(joined) == 2
        assert "เชียงใหม่" not in joined["province"].values
        assert set(joined["year"]) == {2550, 2551}

    def test_top_rankings(self):
        df = pd.DataFrame({
            "province": ["เชียงใหม่", "เชียงราย", "ลำปาง"],
            "geo_level": ["province", "province", "province"],
            "product": ["ข้าวนาปี", "ข้าวนาปี", "ข้าวนาปี"],
            "production_ton": [500.0, 1000.0, 250.0],
            "area_plant_rai": [1000.0, 2000.0, 500.0],
            "area_harvest_rai": [900.0, 1900.0, 480.0]
        })
        top_p = get_top_provinces(df, metric="production_ton", top_n=2)
        assert len(top_p) == 2
        assert top_p.iloc[0]["province"] == "เชียงราย"
        assert top_p.iloc[1]["province"] == "เชียงใหม่"

    def test_climate_interpretation_academic_caveats(self):
        biv = {
            "n": 10,
            "pearson_r": 0.65,
            "pearson_p": 0.04,
            "is_valid": True
        }
        text = generate_climate_interpretation(biv, "ข้าวนาปี", "ปริมาณฝน", "Yield")
        assert "Correlation ไม่ใช่ Causation" in text
        assert "0.650" in text
        assert "มีนัยสำคัญทางสถิติ" in text

    def test_harvest_flags(self):
        df = pd.DataFrame({
            "crop_year": ["2560/61", "2560/61"],
            "year": [2560, 2560],
            "product": ["พืช A", "พืช B"],
            "province": ["สุพรรณบุรี", "สุพรรณบุรี"],
            "area_plant_rai": [100.0, 100.0],
            "area_harvest_rai": [110.0, 80.0],  # A: harvest > plant, B: harvest ratio = 80% (<90%)
            "production_ton": [50.0, 40.0],
            "yield_per_rai": [454.5, 500.0],
            "harvest_ratio": [110.0, 80.0],
            "damaged_area": [-10.0, 20.0]
        })
        cleaned = clean_agriculture_data(df)
        assert cleaned.loc[0, "flag_harvest_gt_plant"] == True
        assert cleaned.loc[1, "flag_harvest_gt_plant"] == False
        assert cleaned.loc[1, "flag_review_harvest_ratio"] == True


class TestDataLoaderAndFiles:
    def test_load_real_agriculture_data(self):
        from src.data_loader import load_agriculture_data, load_suphan_climate_data, load_thailand_geojson
        df_ag = load_agriculture_data()
        assert len(df_ag) == 11914
        assert "crop_year" in df_ag.columns
        assert "production_ton" in df_ag.columns

        df_sup = load_suphan_climate_data()
        assert len(df_sup) == 19
        assert "ปริมาณน้ำฝน" in df_sup.columns

        gj = load_thailand_geojson()
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 77

