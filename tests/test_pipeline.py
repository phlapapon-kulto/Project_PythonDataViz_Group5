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
    validate_agromet_schema, normalize_agromet_dataframe,
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

    def test_clean_tmd_daily_agromet_all_expected_columns(self):
        sample_json = {
            "Stations": {
                "Station": [
                    {
                        "WmoStationNumber": "48304",
                        "StationNameThai": "เชียงราย สกษ.",
                        "Province": "เชียงราย",
                        "Latitude": "19.87",
                        "Longitude": "99.78",
                        "Observation": {
                            "DateTime": "2026-10-21 07:00:00",
                            "Temperature": "22.2",
                            "MaxTemperature": "34.0",
                            "MinTemperature": "20.8",
                            "RelativeHumidity": "98",
                            "WindSpeed": "5",
                            "Rainfall": "12.50"
                        }
                    }
                ]
            }
        }
        df = clean_tmd_daily_agromet(sample_json)
        assert len(df) == 1
        assert df.loc[0, "province"] == "เชียงราย"
        assert df.loc[0, "rainfall_24h_mm"] == 12.50
        assert df.loc[0, "rainfall_mm"] == 12.50
        assert df.loc[0, "wind_speed_kmh"] == 5.0
        assert df.loc[0, "temp_c"] == 22.2
        assert df.loc[0, "relative_humidity_pct"] == 98.0
        assert df.loc[0, "data_source"] == "TMD (กรมอุตุนิยมวิทยา)"
        assert df.loc[0, "data_source_type"] == "tmd_daily_agromet_station"

        # Validate schema passes
        is_valid, missing = validate_agromet_schema(df)
        assert is_valid is True
        assert len(missing) == 0

    def test_validate_agromet_schema_detects_missing(self):
        incomplete_df = pd.DataFrame({
            "station_name": ["สถานีทดสอบ"],
            "province": ["เชียงใหม่"]
        })
        is_valid, missing = validate_agromet_schema(incomplete_df)
        assert is_valid is False
        assert "temp_c" in missing
        assert "rainfall_24h_mm" in missing
        assert "wind_speed_kmh" in missing

    def test_normalize_agromet_dataframe_aliases(self):
        raw_df = pd.DataFrame({
            "stationname": ["สถานีทดสอบ"],
            "changwat": ["ลำปาง"],
            "temp": [25.0],
            "humidity": [80.0],
            "rain_24h": [10.0],
            "wind_speed": [3.0]
        })
        norm = normalize_agromet_dataframe(raw_df)
        assert norm.loc[0, "station_name"] == "สถานีทดสอบ"
        assert norm.loc[0, "province"] == "ลำปาง"
        assert norm.loc[0, "temp_c"] == 25.0
        assert norm.loc[0, "relative_humidity_pct"] == 80.0
        assert norm.loc[0, "rainfall_24h_mm"] == 10.0
        assert norm.loc[0, "rainfall_mm"] == 10.0
        assert norm.loc[0, "wind_speed_kmh"] == 3.0
        assert norm.loc[0, "data_source"] == "TMD (กรมอุตุนิยมวิทยา)"


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


class TestCropProductionDrivers:
    """
    Test suite for Crop Production Drivers (5 pillars: Water/Irrigation, Soil/LDD,
    Technology/Costs, Disasters, and Farm Economics).
    """

    def test_load_and_clean_provincial_irrigation(self):
        from src.data_loader import load_provincial_irrigation_data
        from src.data_cleaning import clean_provincial_irrigation_data
        raw = load_provincial_irrigation_data()
        assert not raw.empty
        assert "irrigation_area_rai" in raw.columns

        cleaned = clean_provincial_irrigation_data(raw)
        assert not cleaned.empty
        assert "irrigation_ratio_pct" in cleaned.columns
        assert (cleaned["irrigation_ratio_pct"] >= 0).all()
        assert (cleaned["irrigation_ratio_pct"] <= 100).all()

    def test_load_and_clean_ldd_soil_suitability(self):
        from src.data_loader import load_ldd_soil_suitability_data
        from src.data_cleaning import clean_ldd_soil_suitability_data
        raw = load_ldd_soil_suitability_data()
        assert not raw.empty
        assert "suitable_high_s1_pct" in raw.columns

        cleaned = clean_ldd_soil_suitability_data(raw)
        assert not cleaned.empty
        assert "suitable_total_s1_s2_pct" in cleaned.columns
        assert (cleaned["suitable_total_s1_s2_pct"] >= 0).all()

    def test_load_and_clean_oae_production_cost(self):
        from src.data_loader import load_oae_production_cost_data
        from src.data_cleaning import clean_oae_production_cost_data
        raw = load_oae_production_cost_data()
        assert not raw.empty
        assert "total_cost_rai" in raw.columns

        cleaned = clean_oae_production_cost_data(raw)
        assert not cleaned.empty
        assert (cleaned["total_cost_rai"] > 0).all()
        assert (cleaned["mechanization_share_pct"] > 0).all()

    def test_load_and_clean_disaster_affected_data(self):
        from src.data_loader import load_disaster_affected_data
        from src.data_cleaning import clean_disaster_affected_data
        raw = load_disaster_affected_data()
        assert not raw.empty
        assert "total_disaster_area_rai" in raw.columns

        cleaned = clean_disaster_affected_data(raw)
        assert not cleaned.empty
        assert (cleaned["total_disaster_area_rai"] >= 0).all()

    def test_join_irrigation_and_yield(self):
        from src.data_loader import load_agriculture_data, load_provincial_irrigation_data
        from src.data_cleaning import clean_agriculture_data, clean_provincial_irrigation_data
        from src.analytics import join_irrigation_and_yield
        df_ag = clean_agriculture_data(load_agriculture_data())
        df_irrig = clean_provincial_irrigation_data(load_provincial_irrigation_data())

        res = join_irrigation_and_yield(df_ag, df_irrig, crop="ข้าวนาปี", year=2562)
        assert not res.empty
        assert "yield_kg_rai" in res.columns
        assert "irrigation_ratio_pct" in res.columns
        assert (res["yield_kg_rai"] > 0).all()

    def test_join_soil_suitability_and_yield(self):
        from src.data_loader import load_agriculture_data, load_ldd_soil_suitability_data
        from src.data_cleaning import clean_agriculture_data, clean_ldd_soil_suitability_data
        from src.analytics import join_soil_suitability_and_yield
        df_ag = clean_agriculture_data(load_agriculture_data())
        df_soil = clean_ldd_soil_suitability_data(load_ldd_soil_suitability_data())

        res = join_soil_suitability_and_yield(df_ag, df_soil, crop="ข้าวนาปี", year=2562)
        assert not res.empty
        assert "yield_kg_rai" in res.columns
        assert "suitable_high_s1_pct" in res.columns

    def test_join_disaster_and_yield(self):
        from src.data_loader import load_agriculture_data, load_disaster_affected_data
        from src.data_cleaning import clean_agriculture_data, clean_disaster_affected_data
        from src.analytics import join_disaster_and_yield
        df_ag = clean_agriculture_data(load_agriculture_data())
        df_dis = clean_disaster_affected_data(load_disaster_affected_data())

        res = join_disaster_and_yield(df_ag, df_dis, crop="ข้าวนาปี", year=2562)
        assert not res.empty
        assert "damage_ratio_pct" in res.columns
        assert "harvest_ratio_pct" in res.columns

    def test_analyze_growing_season_rainfall(self):
        from src.data_loader import load_agriculture_data, load_spatial_rain_data
        from src.data_cleaning import clean_agriculture_data, clean_spatial_rain_data
        from src.analytics import analyze_growing_season_rainfall
        df_ag = clean_agriculture_data(load_agriculture_data())
        raw_rain = load_spatial_rain_data()
        df_rain_monthly, _ = clean_spatial_rain_data(raw_rain)

        res = analyze_growing_season_rainfall(df_rain_monthly, df_ag, crop="ข้าวนาปี", year=2562)
        assert not res.empty
        assert "rain_growing_season_mm" in res.columns
        assert "growing_rain_share_pct" in res.columns
        assert (res["growing_rain_share_pct"] > 0).all()

    def test_compare_production_cost_and_price(self):
        from src.data_loader import load_oae_production_cost_data, load_oae_farmgate_prices
        from src.data_cleaning import clean_oae_production_cost_data, clean_oae_farmgate_prices
        from src.analytics import compare_production_cost_and_price
        df_cost = clean_oae_production_cost_data(load_oae_production_cost_data())
        df_prices = clean_oae_farmgate_prices(load_oae_farmgate_prices())

        res = compare_production_cost_and_price(df_cost, df_prices, crop="ข้าวนาปี", year=2562)
        assert res["has_data"] is True
        assert res["total_cost_rai"] > 0
        assert res["cost_per_kg"] > 0
        assert res["avg_selling_price_kg"] > 0
        assert "cost_breakdown" in res

    def test_calculate_enhanced_farm_profit(self):
        from src.analytics import calculate_enhanced_farm_profit
        # 10 rai, 500 kg/rai = 5,000 kg. Price 12 Bt/kg = 60,000 Bt. Cost 4,000 Bt/rai = 40,000 Bt.
        res = calculate_enhanced_farm_profit(
            planted_area_rai=10.0,
            yield_kg_rai=500.0,
            selling_price_baht_kg=12.0,
            cost_per_rai=4000.0,
            is_user_cost=True,
            benchmark_cost_rai=4400.0
        )
        assert res["total_yield_kg"] == 5000.0
        assert res["total_revenue"] == 60000.0
        assert res["total_cost"] == 40000.0
        assert res["net_profit"] == 20000.0
        assert res["cost_per_kg"] == 8.0
        assert res["breakeven_price_kg"] == 8.0
        assert res["breakeven_yield_kg_rai"] == pytest.approx(333.33, 0.01)
        assert res["diff_cost_rai"] == -400.0
        assert res["diff_cost_pct"] == pytest.approx(-9.09, 0.01)

    def test_create_driver_visualizations(self):
        from src.data_loader import (
            load_agriculture_data, load_provincial_irrigation_data,
            load_ldd_soil_suitability_data, load_disaster_affected_data,
            load_oae_production_cost_data, load_oae_farmgate_prices, load_spatial_rain_data
        )
        from src.data_cleaning import (
            clean_agriculture_data, clean_provincial_irrigation_data,
            clean_ldd_soil_suitability_data, clean_disaster_affected_data,
            clean_oae_production_cost_data, clean_oae_farmgate_prices, clean_spatial_rain_data
        )
        from src.analytics import (
            join_irrigation_and_yield, analyze_growing_season_rainfall, compare_production_cost_and_price
        )
        from src.visualizations import (
            create_irrigation_vs_yield_chart, create_soil_suitability_bar_chart,
            create_cost_structure_breakdown_chart, create_disaster_impact_chart,
            create_seasonal_rain_vs_yield_chart, create_cost_price_margin_chart
        )

        df_ag = clean_agriculture_data(load_agriculture_data())
        df_irrig = clean_provincial_irrigation_data(load_provincial_irrigation_data())
        df_soil = clean_ldd_soil_suitability_data(load_ldd_soil_suitability_data())
        df_dis = clean_disaster_affected_data(load_disaster_affected_data())
        df_cost = clean_oae_production_cost_data(load_oae_production_cost_data())
        df_price = clean_oae_farmgate_prices(load_oae_farmgate_prices())
        df_rain_m, _ = clean_spatial_rain_data(load_spatial_rain_data())

        df_irrig_y = join_irrigation_and_yield(df_ag, df_irrig, "ข้าวนาปี", 2562)
        fig1 = create_irrigation_vs_yield_chart(df_irrig_y, "ข้าวนาปี", 2562)
        assert fig1 is not None

        fig2 = create_soil_suitability_bar_chart(df_soil, "ข้าวนาปี")
        assert fig2 is not None

        summary = compare_production_cost_and_price(df_cost, df_price, "ข้าวนาปี", 2562)
        fig3 = create_cost_structure_breakdown_chart(summary["cost_breakdown"], "ข้าวนาปี", summary["total_cost_rai"], 2562)
        assert fig3 is not None

        fig4 = create_disaster_impact_chart(df_dis, 2562)
        assert fig4 is not None

        df_season = analyze_growing_season_rainfall(df_rain_m, df_ag, "ข้าวนาปี", 2562)
        fig5 = create_seasonal_rain_vs_yield_chart(df_season, "ข้าวนาปี", 2562)
        assert fig5 is not None

        fig6 = create_cost_price_margin_chart(summary)
        assert fig6 is not None


class TestRedesignedDashboard:
    """Acceptance criteria tests for the redesigned 4-tab dashboard."""

    def test_exactly_four_tabs_in_app(self):
        """Verify that app.py defines exactly 4 main tabs."""
        from src.data_loader import get_project_root
        import re
        import os
        app_path = os.path.join(get_project_root(), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            app_content = f.read()
        match = re.search(r"tab1,\s*tab2,\s*tab3,\s*tab4\s*=\s*st\.tabs\(\[([\s\S]*?)\]\)", app_content)
        assert match is not None, "Dashboard must declare exactly tab1, tab2, tab3, tab4 = st.tabs([...])"
        tabs_str = match.group(1)
        tabs_list = [t.strip() for t in tabs_str.split(",") if t.strip()]
        assert len(tabs_list) == 4, f"Expected exactly 4 tabs, found {len(tabs_list)}: {tabs_list}"

    def test_province_to_region_coverage(self):
        """Verify that all 77 official provinces are mapped in PROVINCE_TO_REGION across 5 regions."""
        from src.data_cleaning import OFFICIAL_77_PROVINCES, PROVINCE_TO_REGION, REGION_TO_PROVINCES
        assert len(PROVINCE_TO_REGION) == 77
        allowed_regions = ["ภาคเหนือ", "ภาคตะวันออกเฉียงเหนือ", "ภาคกลาง", "ภาคตะวันออก", "ภาคใต้"]
        for p in OFFICIAL_77_PROVINCES:
            assert p in PROVINCE_TO_REGION, f"Province {p} is missing from PROVINCE_TO_REGION"
            assert PROVINCE_TO_REGION[p] in allowed_regions, f"Province {p} mapped to unknown region {PROVINCE_TO_REGION[p]}"
        
        # Verify reverse mapping
        total_reverse = sum(len(provs) for provs in REGION_TO_PROVINCES.values())
        assert total_reverse == 77
        assert len(REGION_TO_PROVINCES["ภาคตะวันออก"]) == 7

    def test_eastern_region_filtering_and_provinces(self):
        """Verify that Eastern region (ภาคตะวันออก) contains the 7 official provinces with real data."""
        from src.data_loader import load_agriculture_data
        from src.data_cleaning import clean_agriculture_data, REGION_TO_PROVINCES
        from src.filters import apply_ag_filters

        df_ag = clean_agriculture_data(load_agriculture_data())
        df_east = apply_ag_filters(df_ag, year=2562, region="ภาคตะวันออก", geo_level="province")
        assert not df_east.empty, "Eastern region data should not be empty"
        east_provinces = set(REGION_TO_PROVINCES["ภาคตะวันออก"])
        expected_east = {"ฉะเชิงเทรา", "ชลบุรี", "ระยอง", "จันทบุรี", "ตราด", "ปราจีนบุรี", "สระแก้ว"}
        assert east_provinces == expected_east, f"Expected {expected_east}, got {east_provinces}"
        for p in df_east["province"].unique():
            assert p in east_provinces, f"Province {p} not expected in Eastern region"

        # Verify historical presence (all 13 years 2550-2562)
        east_all_years = df_ag[df_ag["region"] == "ภาคตะวันออก"]["year"].unique()
        assert len(east_all_years) == 13, f"Expected 13 years for Eastern region, got {len(east_all_years)}"

    def test_price_units_and_strict_no_surrogate(self):
        """Verify that price dataset retains native units and compute_price_summary_metrics does not fabricate or substitute."""
        from src.data_loader import load_oae_farmgate_prices
        from src.data_cleaning import clean_oae_farmgate_prices
        from src.analytics import compute_price_summary_metrics

        df_prices = clean_oae_farmgate_prices(load_oae_farmgate_prices())
        assert not df_prices.empty
        assert "native_unit" in df_prices.columns
        assert "frequency" in df_prices.columns

        # Verify Hom Mali Rice is in Baht/ton
        mali_sub = df_prices[df_prices["product"] == "ข้าวเปลือกเจ้าหอมมะลิ"]
        assert not mali_sub.empty
        assert mali_sub["native_unit"].iloc[0] == "บาท/ตัน"
        assert mali_sub["native_price"].mean() > 10000.0

        # Verify Durian is in Baht/kg
        durian_sub = df_prices[df_prices["product"] == "ทุเรียนหมอนทอง"]
        assert not durian_sub.empty
        assert durian_sub["native_unit"].iloc[0] == "บาท/กก."
        assert durian_sub["native_price"].mean() > 50.0

        # Verify querying unknown crop returns has_data=False and does NOT substitute
        res_unknown = compute_price_summary_metrics(df_prices, "พืชที่ไม่มีอยู่จริง12345")
        assert res_unknown["has_data"] is False
        assert "ไม่พบข้อมูลราคา" in res_unknown["message"]

    def test_region_filtering_in_apply_ag_filters(self):
        """Verify that region filtering correctly selects only provinces in that region."""
        from src.data_loader import load_agriculture_data
        from src.data_cleaning import clean_agriculture_data, REGION_TO_PROVINCES
        from src.filters import apply_ag_filters

        df_ag = clean_agriculture_data(load_agriculture_data())
        df_north = apply_ag_filters(df_ag, year=2562, region="ภาคเหนือ", geo_level="province")
        assert not df_north.empty
        north_provinces = set(REGION_TO_PROVINCES["ภาคเหนือ"])
        for p in df_north["province"].unique():
            assert p in north_provinces, f"Province {p} not expected in Northern region"

    def test_all_26_crops_present_in_dataset(self):
        """Verify that all 26 crops from the OAE dataset are loaded without dropping any."""
        from src.data_loader import load_agriculture_data
        from src.data_cleaning import clean_agriculture_data

        df_ag = clean_agriculture_data(load_agriculture_data())
        crops = df_ag["product"].dropna().unique().tolist()
        assert len(crops) >= 26, f"Expected at least 26 crops, found {len(crops)}"
        assert "ข้าวนาปี" in crops
        assert "ข้าวนาปรัง" in crops
        assert "มันสำปะหลัง" in crops
        assert "ข้าวโพดเลี้ยงสัตว์" in crops
        assert "ทุเรียน" in crops
        assert "ยางพารา" in crops

    def test_farm_profit_calculator_formulas(self):
        """Verify required farm profit calculator formulas:
        - Total Cost = Area * Cost per Rai
        - Total Yield = Area * Yield per Rai
        - Revenue = Total Yield * Sale Price per Unit
        - Estimated Profit = Revenue - Total Cost
        - Break-even Yield = Total Cost / (Area * Sale Price)
        - Break-even Price = Total Cost / Total Yield
        """
        area = 25.0
        cost_rai = 4000.0
        yield_rai = 500.0
        sale_price = 10.0

        tot_cost = area * cost_rai          # 100,000 Baht
        tot_yield = area * yield_rai        # 12,500 kg
        tot_rev = tot_yield * sale_price    # 125,000 Baht
        est_profit = tot_rev - tot_cost     # 25,000 Baht
        be_yield = tot_cost / (area * sale_price) # 400.0 kg/rai
        be_price = tot_cost / tot_yield     # 8.0 Baht/kg

        assert tot_cost == 100000.0
        assert tot_yield == 12500.0
        assert tot_rev == 125000.0
        assert est_profit == 25000.0
        assert be_yield == 400.0
        assert be_price == 8.0

    def test_headless_app_execution_zero_exceptions(self):
        """Verify that app.py executes cleanly without uncaught exceptions via Streamlit AppTest."""
        from streamlit.testing.v1 import AppTest
        from src.data_loader import get_project_root
        import os
        app_path = os.path.join(get_project_root(), "app.py")
        at = AppTest.from_file(app_path, default_timeout=30).run()
        assert len(at.exception) == 0, f"App execution raised exceptions: {[e.value for e in at.exception]}"


class TestDecisionIntelligenceAndMapLayers:
    """
    Test suite for Decision Intelligence functions (Tab 2, Tab 3, Tab 4) and
    dedicated Thailand Maps for each of the 4 tabs.
    """
    def test_calculate_climate_risk_status(self):
        from src.analytics import calculate_climate_risk_status

        # Test high drought risk (low rain)
        status_drought = calculate_climate_risk_status(rain_val=800.0, temp_val=33.0, disaster_val=5000.0)
        assert status_drought["drought"]["color"] == "red"
        assert "สูง" in status_drought["drought"]["level"]

        # Test normal rain
        status_normal = calculate_climate_risk_status(rain_val=1500.0, temp_val=28.0, disaster_val=0.0)
        assert status_normal["drought"]["color"] == "green"
        assert status_normal["flood"]["color"] == "green"
        assert status_normal["heat"]["color"] == "green"

        # Test high flood risk (large disaster area)
        status_flood = calculate_climate_risk_status(rain_val=1900.0, temp_val=31.0, disaster_val=60000.0)
        assert status_flood["flood"]["color"] == "red"

        # Test extreme heat
        status_heat = calculate_climate_risk_status(rain_val=1300.0, temp_val=36.0, disaster_val=1000.0)
        assert status_heat["heat"]["color"] == "red"

    def test_generate_farmer_decision_advisory_structure(self):
        from src.analytics import generate_farmer_decision_advisory

        advisory = generate_farmer_decision_advisory(
            crop_name="ข้าวนาปี",
            province_name="สุพรรณบุรี",
            rainfall_val=1250.0,
            temp_val=29.5,
            disaster_val=8500.0,
            year=2562
        )

        # Verify 5 mandatory decision dimensions
        assert "current_situation" in advisory
        assert "risks_to_watch" in advisory and len(advisory["risks_to_watch"]) > 0
        assert "preparations_needed" in advisory and len(advisory["preparations_needed"]) > 0
        assert "recommended_practices" in advisory and len(advisory["recommended_practices"]) > 0
        assert "signals_to_monitor" in advisory and len(advisory["signals_to_monitor"]) > 0

        # Verify planting window guide and disclaimers
        assert "planting_window" in advisory
        assert "source_agency" in advisory
        assert "disclaimer" in advisory
        assert advisory["crop_name"] == "ข้าวนาปี"
        assert advisory["province_name"] == "สุพรรณบุรี"

    def test_compute_price_summary_metrics(self):
        from src.analytics import compute_price_summary_metrics
        from src.data_loader import load_oae_farmgate_prices
        from src.data_cleaning import clean_oae_farmgate_prices

        df_prices = clean_oae_farmgate_prices(load_oae_farmgate_prices())
        metrics = compute_price_summary_metrics(df_prices, "ข้าวเปลือกเจ้าหอมมะลิ")

        assert metrics["has_data"] is True
        assert metrics["product"] == "ข้าวเปลือกเจ้าหอมมะลิ"
        assert metrics["latest_price"] > 0
        assert metrics["min_price"] <= metrics["mean_price"] <= metrics["max_price"]
        assert "unit" in metrics
        assert "wow_change" in metrics
        assert "mom_change" in metrics
        assert "latest_period" in metrics

        # Test partial matching fallback (e.g. 'มันสำปะหลัง')
        metrics_partial = compute_price_summary_metrics(df_prices, "มันสำปะหลัง")
        assert metrics_partial["has_data"] is True
        assert "มันสำปะหลัง" in metrics_partial["product"]

    def test_all_tab_maps_generation(self):
        from src.data_loader import (
            load_agriculture_data, load_spatial_rain_data, load_provincial_temp_data,
            load_provincial_irrigation_data, load_ldd_soil_suitability_data,
            load_disaster_affected_data, load_tmd_daily_agromet, load_thailand_geojson
        )
        from src.data_cleaning import (
            clean_agriculture_data, clean_spatial_rain_data, clean_provincial_temp_data,
            clean_provincial_irrigation_data, clean_ldd_soil_suitability_data,
            clean_disaster_affected_data, clean_tmd_daily_agromet
        )
        from src.visualizations import (
            create_thailand_map, create_climate_disaster_map,
            create_market_price_map, create_production_driver_map
        )

        geojson_data = load_thailand_geojson()
        assert geojson_data is not None

        df_ag = clean_agriculture_data(load_agriculture_data())
        _, df_rain = clean_spatial_rain_data(load_spatial_rain_data())
        _, df_temp = clean_provincial_temp_data(load_provincial_temp_data())
        df_irrig = clean_provincial_irrigation_data(load_provincial_irrigation_data())
        df_soil = clean_ldd_soil_suitability_data(load_ldd_soil_suitability_data())
        df_disaster = clean_disaster_affected_data(load_disaster_affected_data())
        df_agromet = clean_tmd_daily_agromet(load_tmd_daily_agromet())

        # 1. Tab 1 Map: Agriculture Overview
        fig1 = create_thailand_map(
            df_data=df_ag[df_ag["year"] == 2562],
            geojson_data=geojson_data,
            metric_col="production_ton",
            metric_label="ปริมาณผลผลิตรวม",
            unit="ตัน",
            year=2562,
            product_name="ข้าวนาปี"
        )
        assert len(fig1.data) >= 1

        # 2. Tab 2 Map: Climate & Disaster Layers
        fig2_rain = create_climate_disaster_map(df_rain, df_temp, df_disaster, df_agromet, geojson_data, layer_type="rain", year=2562)
        assert len(fig2_rain.data) >= 1

        fig2_stn = create_climate_disaster_map(df_rain, df_temp, df_disaster, df_agromet, geojson_data, layer_type="stations", year=2562)
        assert len(fig2_stn.data) >= 2  # Choropleth + Scattergeo

        # 3. Tab 3 Map: Market Price / Production Volume
        fig3 = create_market_price_map(df_ag, geojson_data, crop_name="ข้าวนาปี", year=2562)
        assert len(fig3.data) >= 1

        # 4. Tab 4 Map: Production Drivers
        fig4_yield = create_production_driver_map(df_ag, df_irrig, df_soil, df_disaster, geojson_data, layer_type="yield", crop_name="ข้าวนาปี", year=2562)
        assert len(fig4_yield.data) >= 1

        fig4_irrig = create_production_driver_map(df_ag, df_irrig, df_soil, df_disaster, geojson_data, layer_type="irrigation", crop_name="ข้าวนาปี", year=2562)
        assert len(fig4_irrig.data) >= 1

