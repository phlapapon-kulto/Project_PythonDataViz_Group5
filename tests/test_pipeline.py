"""
Comprehensive unit test suite for Thai Agriculture & Climate Intelligence Dashboard.
Tests mathematical formulas, data cleaning, spatial hierarchies, double-count prevention,
farm budget calculations, crop calendars, and verifies that the retired Suphan Buri
dataset is never loaded or referenced.
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
    clean_spatial_rain_data, clean_national_rain_data, clean_provincial_temp_data,
    clean_oae_farmgate_prices, clean_tmd_daily_agromet,
    OFFICIAL_77_PROVINCES, REGION_NAMES, COUNTRY_NAME
)
from src.feature_engineering import (
    calculate_weighted_yield, calculate_aggregated_harvest_ratio,
    calculate_unharvested_area, calculate_yoy_growth, calculate_cv,
    aggregate_agricultural_kpis, compute_yoy_dataframe, compute_cv_summary
)
from src.analytics import (
    calculate_bivariate_stats, classify_area_yield_quadrants,
    join_climate_agriculture_nationwide, get_top_provinces, get_top_crops,
    generate_climate_interpretation, calculate_farm_budget, get_crop_calendar_guide
)
from src.filters import apply_ag_filters
from src.data_loader import (
    load_agriculture_data, load_spatial_rain_data, load_national_rain_data,
    load_provincial_temp_data, load_oae_farmgate_prices, load_tmd_daily_agromet,
    load_thailand_geojson, get_project_root
)


class TestMandatoryDataRemoval:
    def test_retired_suphan_dataset_not_used(self):
        """
        Verify that the retired Suphan Buri climate Excel file does NOT exist
        anywhere in the repository and is never read.
        """
        root = get_project_root()
        retired_filenames = [
            "สถิติปริมาณฝนและอุณหภูมิ.xlsx",
            "suphan_climate.xlsx"
        ]
        for root_dir, dirs, files in os.walk(root):
            # Ignore .git directory
            if ".git" in root_dir:
                continue
            for f in files:
                assert f not in retired_filenames, f"Retired file found: {os.path.join(root_dir, f)}"

    def test_no_suphan_legacy_loader_exported(self):
        """
        Verify that data_loader and data_cleaning do NOT expose retired Suphan functions.
        """
        import src.data_loader as dl
        import src.data_cleaning as dc
        import src.analytics as an

        assert not hasattr(dl, "load_suphan_climate_data")
        assert not hasattr(dc, "clean_suphan_climate_data")
        assert not hasattr(an, "join_climate_agriculture_suphan")


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

    def test_clean_oae_farmgate_prices(self):
        raw_df = pd.DataFrame({
            "PROD_TYPE": ["ข้าวเปลือกเจ้านาปี 15%"],
            "YEAR": [2561],
            "MONTH_C": [1],
            "WEEK_C": [1],
            "PRICE": [7500.0]  # Baht/ton
        })
        cleaned = clean_oae_farmgate_prices(raw_df)
        assert len(cleaned) == 1
        assert cleaned.loc[0, "product"] == "ข้าวเปลือกเจ้านาปี 15%"
        assert cleaned.loc[0, "price_baht_per_ton"] == 7500.0
        assert cleaned.loc[0, "price_baht_per_kg"] == 7.5

    def test_clean_tmd_daily_agromet(self):
        sample_json = {
            "Stations": {
                "Station": [
                    {
                        "WmoStationNumber": "48381",
                        "StationNameThai": "สถานีเกษตรกาญจนบุรี",
                        "Province": "กาญจนบุรี",
                        "Latitude": "14.0",
                        "Longitude": "99.5",
                        "Observation": {
                            "DateTime": "2026-04-20 07:00:00",
                            "Temperature": "28.5",
                            "MaxTemperature": "36.0",
                            "MinTemperature": "24.0",
                            "RelativeHumidity": "75.0",
                            "Rainfall": "0.0"
                        }
                    }
                ]
            }
        }
        df = clean_tmd_daily_agromet(sample_json)
        assert len(df) == 1
        assert df.loc[0, "province"] == "กาญจนบุรี"
        assert df.loc[0, "max_temp_c"] == 36.0
        assert df.loc[0, "rainfall_mm"] == 0.0


class TestMathematicalFormulas:
    def test_weighted_yield_calculation(self):
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
        assert np.isnan(yoy.iloc[4])

    def test_cv_calculation(self):
        s_stable = pd.Series([100.0, 100.0, 100.0])
        assert calculate_cv(s_stable) == 0.0

        s_var = pd.Series([90.0, 100.0, 110.0])
        assert pytest.approx(calculate_cv(s_var), 0.01) == 10.0

        s_few = pd.Series([100.0, 110.0])
        assert np.isnan(calculate_cv(s_few, min_periods=3))


class TestDoubleCountingPrevention:
    def test_aggregate_double_count_prevention(self):
        df = pd.DataFrame({
            "province": ["ทั้งประเทศ", "ภาคกลาง", "สุพรรณบุรี", "นครปฐม"],
            "geo_level": ["country", "region", "province", "province"],
            "year": [2560, 2560, 2560, 2560],
            "product": ["ข้าวนาปี", "ข้าวนาปี", "ข้าวนาปี", "ข้าวนาปี"],
            "production_ton": [1000.0, 500.0, 300.0, 200.0],
            "area_plant_rai": [2000.0, 1000.0, 600.0, 400.0],
            "area_harvest_rai": [1900.0, 950.0, 570.0, 380.0]
        })
        prov_only = apply_ag_filters(df, geo_level="province")
        kpis = aggregate_agricultural_kpis(prov_only)
        # Sum of provinces: 300 + 200 = 500 tons
        assert kpis["total_production_ton"] == 500.0
        assert kpis["total_plant_rai"] == 1000.0
        assert kpis["total_harvest_rai"] == 950.0


class TestAnalyticsAndPlanning:
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

    def test_calculate_farm_budget(self):
        # 10 rai, 600 kg/rai = 6000 kg total
        # Selling price: 10 Baht/kg -> Revenue: 60,000 Baht
        # Total cost: 5,000 (seeds) + 10,000 (fert) + 5,000 (labor) = 20,000 Baht
        # Net profit: 40,000 Baht
        # Cost per kg: 20,000 / 6,000 = 3.33 Baht/kg
        # Break-even price: 3.33 Baht/kg
        # Break-even yield: 20,000 / (10 * 10) = 200 kg/rai
        budget = calculate_farm_budget(
            planted_area_rai=10.0,
            expected_yield_kg_rai=600.0,
            selling_price_baht_kg=10.0,
            seeds_cost=5000.0,
            fertilizer_cost=10000.0,
            labor_cost=5000.0
        )
        assert budget["is_valid"] is True
        assert budget["total_yield_kg"] == 6000.0
        assert budget["gross_revenue"] == 60000.0
        assert budget["total_cost"] == 20000.0
        assert budget["net_profit"] == 40000.0
        assert pytest.approx(budget["cost_per_kg"], 0.01) == 3.33
        assert pytest.approx(budget["breakeven_price_kg"], 0.01) == 3.33
        assert budget["breakeven_yield_kg_rai"] == 200.0
        assert len(budget["scenarios"]) == 12

    def test_calculate_farm_budget_invalid(self):
        bad_budget = calculate_farm_budget(
            planted_area_rai=0.0,
            expected_yield_kg_rai=600.0,
            selling_price_baht_kg=10.0
        )
        assert bad_budget["is_valid"] is False

    def test_crop_calendar_guide(self):
        guide_rice = get_crop_calendar_guide("ข้าวนาปี")
        assert "ข้าว" in guide_rice["crop_group"]
        assert len(guide_rice["pre_planting_checklist"]) >= 4

        guide_cane = get_crop_calendar_guide("อ้อยโรงงาน")
        assert "อ้อย" in guide_cane["crop_group"]

        guide_cassava = get_crop_calendar_guide("มันสำปะหลังโรงงาน")
        assert "มันสำปะหลัง" in guide_cassava["crop_group"]

        guide_corn = get_crop_calendar_guide("ข้าวโพดเลี้ยงสัตว์")
        assert "ข้าวโพด" in guide_corn["crop_group"]

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


class TestDataLoaderAndFiles:
    def test_load_real_agriculture_data(self):
        df_ag = load_agriculture_data()
        assert len(df_ag) == 11914
        assert "crop_year" in df_ag.columns
        assert "production_ton" in df_ag.columns

    def test_load_oae_prices(self):
        df_p = load_oae_farmgate_prices()
        assert len(df_p) > 1000
        assert "PROD_TYPE" in df_p.columns
        assert "PRICE" in df_p.columns

    def test_load_tmd_agromet(self):
        data = load_tmd_daily_agromet(use_live_api=False)
        assert "Stations" in data
        assert len(data["Stations"]["Station"]) > 10

    def test_load_thailand_geojson(self):
        gj = load_thailand_geojson()
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 77


class TestEnhancedFeatures:
    def test_dynamic_26_crops_loaded(self):
        from src.data_loader import load_agriculture_data
        from src.data_cleaning import clean_agriculture_data
        df_ag = clean_agriculture_data(load_agriculture_data())
        unique_crops = df_ag["product"].unique()
        assert len(unique_crops) == 26
        assert "มันสำปะหลัง" in unique_crops
        assert "ข้าวนาปี" in unique_crops
        assert "ข้าวนาปรัง" in unique_crops
        assert "ข้าวโพดเลี้ยงสัตว์" in unique_crops

    def test_bottom_provinces_ranking(self):
        df_mock = pd.DataFrame({
            "province": ["เชียงใหม่", "ลำพูน", "ลำปาง", "แพร่", "น่าน"],
            "geo_level": ["province", "province", "province", "province", "province"],
            "production_ton": [500.0, 100.0, 20.0, 0.0, np.nan],
            "area_plant_rai": [1000.0, 200.0, 50.0, 0.0, 0.0],
            "area_harvest_rai": [900.0, 180.0, 40.0, 0.0, 0.0]
        })
        # Top 2
        top_res = get_top_provinces(df_mock, metric="production_ton", top_n=2, ascending=False)
        assert len(top_res) == 2
        assert top_res.iloc[0]["province"] == "เชียงใหม่"
        assert top_res.iloc[1]["province"] == "ลำพูน"

        # Bottom 2 (should filter 0 and NaN and return 20.0 and 100.0)
        bot_res = get_top_provinces(df_mock, metric="production_ton", top_n=2, ascending=True)
        assert len(bot_res) == 2
        assert bot_res.iloc[0]["province"] == "ลำปาง"
        assert bot_res.iloc[0]["production_ton"] == 20.0
        assert bot_res.iloc[1]["province"] == "ลำพูน"
        assert bot_res.iloc[1]["production_ton"] == 100.0

    def test_gistda_check_crop_validation(self):
        from src.data_loader import query_gistda_check_crop
        # Outside Thailand bounds
        res_out = query_gistda_check_crop(lat=45.0, lon=2.0)
        assert res_out["status"] == "invalid_coordinates"

        # Inside Thailand but without key
        res_no_key = query_gistda_check_crop(lat=14.5, lon=100.5, api_key="")
        assert res_no_key["status"] == "no_key"
        assert "opendata.gistda.or.th" in res_no_key["message"]

    def test_create_ranking_bar_chart_bottom(self):
        from src.visualizations import create_ranking_bar_chart
        df = pd.DataFrame({"province": ["ก", "ข"], "production_ton": [10.0, 20.0]})
        fig_top = create_ranking_bar_chart(df, x_col="production_ton", y_col="province", title="Top", x_label="ตัน", unit="ตัน", is_bottom=False)
        assert fig_top is not None
        fig_bot = create_ranking_bar_chart(df, x_col="production_ton", y_col="province", title="Bot", x_label="ตัน", unit="ตัน", is_bottom=True)
        assert fig_bot is not None

    def test_create_bivariate_scatter_chart_uncluttered(self):
        from src.visualizations import create_bivariate_scatter_chart
        df = pd.DataFrame({
            "province": ["ก", "ข", "ค", "ง"],
            "rain": [1000.0, 1200.0, 1400.0, 1600.0],
            "yield": [400.0, 450.0, 500.0, 550.0]
        })
        fig = create_bivariate_scatter_chart(
            df,
            x_col="rain",
            y_col="yield",
            title="Rain vs Yield",
            x_label="Rain",
            y_label="Yield",
            x_unit="mm",
            y_unit="kg/rai",
            text_col="province",
            correlation_val=0.99,
            p_val=0.01,
            n_samples=4
        )
        assert fig is not None
        # Scatter traces should not have text on dots (mode is markers)
        scatter_trace = fig.data[0]
        assert getattr(scatter_trace, "text", None) is None
        assert "Pearson r = +0.990" in fig.layout.title.text

    def test_farm_budget_detailed_formulas(self):
        # 20 rai, 800 kg/rai = 16,000 kg total
        # Selling price: 12.0 Baht/kg -> Revenue: 192,000 Baht
        # Costs: 8,000 (seed) + 20,000 (fert) + 6,000 (chem) + 16,000 (labor) + 8,000 (energy) + 6,000 (rent) = 64,000 Baht
        # Cost per rai: 64,000 / 20 = 3,200 Baht/rai
        # Cost per kg: 64,000 / 16,000 = 4.0 Baht/kg
        # Break-even price: 4.0 Baht/kg
        # Break-even yield: 64,000 / (12.0 * 20) = 266.67 kg/rai
        # Net profit: 192,000 - 64,000 = 128,000 Baht
        res = calculate_farm_budget(
            planted_area_rai=20.0,
            expected_yield_kg_rai=800.0,
            selling_price_baht_kg=12.0,
            seeds_cost=8000.0,
            fertilizer_cost=20000.0,
            water_energy_cost=8000.0,
            labor_cost=16000.0,
            land_rent_cost=6000.0,
            other_cost=6000.0
        )
        assert res["is_valid"] is True
        assert res["total_yield_kg"] == 16000.0
        assert res["total_yield_ton"] == 16.0
        assert res["gross_revenue"] == 192000.0
        assert res["total_cost"] == 64000.0
        assert res["net_profit"] == 128000.0
        assert res["cost_per_rai"] == 3200.0
        assert res["cost_per_kg"] == 4.0
        assert res["breakeven_price_kg"] == 4.0
        assert pytest.approx(res["breakeven_yield_kg_rai"], 0.01) == 266.67


    def test_src_package_structure(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        assert os.path.exists(os.path.join(root, "src", "__init__.py")), "src/__init__.py must exist for Streamlit Cloud"

    def test_data_loader_all_symbols(self):
        from src.data_loader import (
            load_agriculture_data,
            load_spatial_rain_data,
            load_national_rain_data,
            load_provincial_temp_data,
            load_oae_farmgate_prices,
            load_tmd_daily_agromet,
            load_thailand_geojson,
            query_gistda_check_crop
        )
        assert callable(load_agriculture_data)
        assert callable(query_gistda_check_crop)

    def test_streamlit_cloud_import_resilience(self):
        import importlib
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        src_dir = os.path.join(root, "src")
        # Ensure root and src are present on sys.path
        assert root in sys.path or any(os.path.samefile(p, root) for p in sys.path if os.path.exists(p))
        mod = importlib.import_module("src.data_loader")
        assert hasattr(mod, "load_agriculture_data")
        assert hasattr(mod, "query_gistda_check_crop")
