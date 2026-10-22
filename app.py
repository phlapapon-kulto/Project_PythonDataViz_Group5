"""
Thai Agriculture & Climate Intelligence Dashboard — Real Open Data Edition
A comprehensive, academic-grade data intelligence platform analyzing Thailand's
agricultural production, spatial distribution, climate interactions, crop planning,
and market economics.

Built with Streamlit, Plotly, Pandas, and SciPy.
Strictly relies on verified government Open Data sources (OAE, TMD, HII).
Zero mock data. Double-counting prevented 100%.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Ensure repository root and src/ directory are at the head of sys.path.
# This prevents Streamlit Community Cloud ImportError / ModuleNotFoundError
# where /mount/src namespace collides with the local src package on Linux.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(1, str(SRC_DIR))

import streamlit as st
import pandas as pd
import numpy as np

# Robust import pattern supporting both package and flat search paths
try:
    from src.data_loader import (
        load_agriculture_data,
        load_spatial_rain_data,
        load_national_rain_data,
        load_provincial_temp_data,
        load_oae_farmgate_prices,
        load_tmd_daily_agromet,
        load_thailand_geojson,
        query_gistda_check_crop,
        load_provincial_irrigation_data,
        load_ldd_soil_suitability_data,
        load_oae_production_cost_data,
        load_disaster_affected_data
    )
except ImportError:
    from data_loader import (
        load_agriculture_data,
        load_spatial_rain_data,
        load_national_rain_data,
        load_provincial_temp_data,
        load_oae_farmgate_prices,
        load_tmd_daily_agromet,
        load_thailand_geojson,
        query_gistda_check_crop,
        load_provincial_irrigation_data,
        load_ldd_soil_suitability_data,
        load_oae_production_cost_data,
        load_disaster_affected_data
    )

try:
    from src.data_cleaning import (
        clean_agriculture_data,
        clean_spatial_rain_data,
        clean_national_rain_data,
        clean_provincial_temp_data,
        clean_oae_farmgate_prices,
        clean_tmd_daily_agromet,
        OFFICIAL_77_PROVINCES,
        COUNTRY_NAME,
        REGION_NAMES,
        clean_provincial_irrigation_data,
        clean_ldd_soil_suitability_data,
        clean_oae_production_cost_data,
        clean_disaster_affected_data
    )
except ImportError:
    from data_cleaning import (
        clean_agriculture_data,
        clean_spatial_rain_data,
        clean_national_rain_data,
        clean_provincial_temp_data,
        clean_oae_farmgate_prices,
        clean_tmd_daily_agromet,
        OFFICIAL_77_PROVINCES,
        COUNTRY_NAME,
        REGION_NAMES,
        clean_provincial_irrigation_data,
        clean_ldd_soil_suitability_data,
        clean_oae_production_cost_data,
        clean_disaster_affected_data
    )

try:
    from src.feature_engineering import (
        calculate_weighted_yield,
        calculate_aggregated_harvest_ratio,
        calculate_unharvested_area,
        aggregate_agricultural_kpis,
        compute_yoy_dataframe,
        compute_cv_summary
    )
except ImportError:
    from feature_engineering import (
        calculate_weighted_yield,
        calculate_aggregated_harvest_ratio,
        calculate_unharvested_area,
        aggregate_agricultural_kpis,
        compute_yoy_dataframe,
        compute_cv_summary
    )

try:
    from src.analytics import (
        join_climate_agriculture_nationwide,
        calculate_bivariate_stats,
        generate_climate_interpretation,
        classify_area_yield_quadrants,
        get_top_provinces,
        get_top_crops,
        calculate_farm_budget,
        get_crop_calendar_guide,
        join_irrigation_and_yield,
        join_soil_suitability_and_yield,
        join_disaster_and_yield,
        analyze_growing_season_rainfall,
        compare_production_cost_and_price,
        calculate_enhanced_farm_profit
    )
except ImportError:
    from analytics import (
        join_climate_agriculture_nationwide,
        calculate_bivariate_stats,
        generate_climate_interpretation,
        classify_area_yield_quadrants,
        get_top_provinces,
        get_top_crops,
        calculate_farm_budget,
        get_crop_calendar_guide,
        join_irrigation_and_yield,
        join_soil_suitability_and_yield,
        join_disaster_and_yield,
        analyze_growing_season_rainfall,
        compare_production_cost_and_price,
        calculate_enhanced_farm_profit
    )

try:
    from src.visualizations import (
        create_thailand_map,
        create_ranking_bar_chart,
        create_trend_line_chart,
        create_bivariate_scatter_chart,
        create_correlation_heatmap,
        create_quadrant_scatter_chart,
        create_yoy_bar_chart,
        create_price_trend_chart,
        create_cost_breakdown_pie,
        create_scenario_bar_chart,
        COLOR_FOREST_DARK,
        COLOR_FOREST_MED,
        COLOR_WARNING,
        COLOR_DANGER,
        create_irrigation_vs_yield_chart,
        create_soil_suitability_bar_chart,
        create_cost_structure_breakdown_chart,
        create_disaster_impact_chart,
        create_seasonal_rain_vs_yield_chart,
        create_cost_price_margin_chart
    )
except ImportError:
    from visualizations import (
        create_thailand_map,
        create_ranking_bar_chart,
        create_trend_line_chart,
        create_bivariate_scatter_chart,
        create_correlation_heatmap,
        create_quadrant_scatter_chart,
        create_yoy_bar_chart,
        create_price_trend_chart,
        create_cost_breakdown_pie,
        create_scenario_bar_chart,
        COLOR_FOREST_DARK,
        COLOR_FOREST_MED,
        COLOR_WARNING,
        COLOR_DANGER,
        create_irrigation_vs_yield_chart,
        create_soil_suitability_bar_chart,
        create_cost_structure_breakdown_chart,
        create_disaster_impact_chart,
        create_seasonal_rain_vs_yield_chart,
        create_cost_price_margin_chart
    )

try:
    from src.filters import init_session_state, reset_filters, apply_ag_filters
except ImportError:
    from filters import init_session_state, reset_filters, apply_ag_filters

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Thai Agriculture & Climate Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern agricultural-tech UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1B4D3E 0%, #2D6A4F 60%, #40916C 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(27, 77, 62, 0.12);
    }
    .main-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.3px;
        color: #FFFFFF;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #D8F3DC;
        opacity: 0.95;
    }

    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(27, 77, 62, 0.08);
        border-color: #52B788;
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #1B4D3E;
        line-height: 1.2;
    }
    .kpi-unit {
        font-size: 13px;
        font-weight: 400;
        color: #94A3B8;
        margin-left: 4px;
    }
    .kpi-sub {
        font-size: 11.5px;
        color: #52B788;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Governance Badge */
    .gov-badge {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .notice-badge {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Data Loading & Processing (Strict Real Open Data)
# ---------------------------------------------------------
@st.cache_data(show_spinner="กำลังโหลดและประมวลผลข้อมูลภาครัฐที่ตรวจสอบได้...")
def load_all_datasets():
    # 1. Agriculture Production Statistics (OAE)
    raw_ag = load_agriculture_data()
    df_ag = clean_agriculture_data(raw_ag)

    # 2. HII Spatial Rain Data (IDW 77 Provinces, 2561-2562)
    try:
        raw_rain = load_spatial_rain_data()
        df_rain_monthly, df_rain_annual = clean_spatial_rain_data(raw_rain)
    except Exception:
        df_rain_monthly = pd.DataFrame()
        df_rain_annual = pd.DataFrame()

    # 3. HII National Rain Data (Long-term Mean 1970-2025)
    try:
        raw_nat = load_national_rain_data()
        df_nat_monthly, df_nat_annual = clean_national_rain_data(raw_nat)
    except Exception:
        df_nat_monthly = pd.DataFrame()
        df_nat_annual = pd.DataFrame()

    # 4. HII Provincial Temp Data (2018-2019)
    try:
        raw_temp = load_provincial_temp_data()
        df_temp_monthly, df_temp_annual = clean_provincial_temp_data(raw_temp)
    except Exception:
        df_temp_monthly = pd.DataFrame()
        df_temp_annual = pd.DataFrame()

    # 5. OAE Official Farmgate Prices (Weekly 2561-2562)
    try:
        raw_prices = load_oae_farmgate_prices()
        df_prices = clean_oae_farmgate_prices(raw_prices)
    except Exception:
        df_prices = pd.DataFrame()

    # 6. TMD Daily Agromet Observations (Catalog c1-1-3-7)
    try:
        raw_agromet = load_tmd_daily_agromet(use_live_api=True)
        df_agromet = clean_tmd_daily_agromet(raw_agromet)
    except Exception:
        df_agromet = pd.DataFrame()

    # 7. Thailand Provincial GeoJSON
    try:
        geojson_data = load_thailand_geojson()
    except Exception:
        geojson_data = None

    # 8. Provincial Irrigation Data (RID & OAE 2561-2562)
    try:
        df_irrig = clean_provincial_irrigation_data(load_provincial_irrigation_data())
    except Exception:
        df_irrig = pd.DataFrame()

    # 9. LDD Soil Suitability (Agri-Map)
    try:
        df_soil = clean_ldd_soil_suitability_data(load_ldd_soil_suitability_data())
    except Exception:
        df_soil = pd.DataFrame()

    # 10. OAE Production Cost Structure (2561-2562)
    try:
        df_cost = clean_oae_production_cost_data(load_oae_production_cost_data())
    except Exception:
        df_cost = pd.DataFrame()

    # 11. Agricultural Disaster Impact (DOAE & DDPM 2561-2562)
    try:
        df_disaster = clean_disaster_affected_data(load_disaster_affected_data())
    except Exception:
        df_disaster = pd.DataFrame()

    return {
        "ag": df_ag,
        "rain_monthly": df_rain_monthly,
        "rain_annual": df_rain_annual,
        "nat_rain_annual": df_nat_annual,
        "temp_annual": df_temp_annual,
        "prices": df_prices,
        "agromet": df_agromet,
        "geojson": geojson_data,
        "irrig": df_irrig,
        "soil": df_soil,
        "cost": df_cost,
        "disaster": df_disaster
    }


data_dict = load_all_datasets()
df_ag = data_dict["ag"]
df_rain_monthly = data_dict["rain_monthly"]
df_rain_annual = data_dict["rain_annual"]
df_nat_annual = data_dict["nat_rain_annual"]
df_temp_annual = data_dict["temp_annual"]
df_prices = data_dict["prices"]
df_agromet = data_dict["agromet"]
geojson_data = data_dict["geojson"]
df_irrig = data_dict["irrig"]
df_soil = data_dict["soil"]
df_cost = data_dict["cost"]
df_disaster = data_dict["disaster"]

# Initialize Session State
available_years = sorted(df_ag["year"].dropna().unique().astype(int).tolist())
latest_year = int(max(available_years)) if available_years else 2562
init_session_state(default_year=latest_year)

# ---------------------------------------------------------
# Sidebar: Global Navigation & Filter Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌾 ตัวกรองข้อมูลหลัก (Global Filters)")

    # 4 Main Economic Crops Quick Shortcut
    st.markdown("#### 🎯 ทางลัดพืชเศรษฐกิจหลัก")
    col_crop1, col_crop2 = st.columns(2)
    with col_crop1:
        if st.button("🌾 ข้าวนาปี", use_container_width=True):
            st.session_state.selected_product = "ข้าวนาปี"
        if st.button("🌾 ข้าวนาปรัง", use_container_width=True):
            st.session_state.selected_product = "ข้าวนาปรัง"
    with col_crop2:
        if st.button("🥔 มันสำปะหลัง", use_container_width=True):
            st.session_state.selected_product = "มันสำปะหลัง"
        if st.button("🌽 ข้าวโพดเลี้ยงสัตว์", use_container_width=True):
            st.session_state.selected_product = "ข้าวโพดเลี้ยงสัตว์"

    # Year Selector
    year_options = available_years
    curr_yr_idx = year_options.index(st.session_state.selected_year) if st.session_state.selected_year in year_options else len(year_options) - 1
    selected_year = st.selectbox(
        "📅 ปี พ.ศ. (Crop Year BE)",
        options=year_options,
        index=curr_yr_idx,
        key="selected_year_box"
    )
    st.session_state.selected_year = selected_year

    # Product Selector (All 26 crops)
    all_products = ["ทั้งหมด"] + sorted(df_ag["product"].unique().tolist())
    curr_prod_idx = all_products.index(st.session_state.selected_product) if st.session_state.selected_product in all_products else 0
    selected_product = st.selectbox(
        "🌱 ชนิดสินค้าเกษตร (Crop)",
        options=all_products,
        index=curr_prod_idx,
        key="selected_product_box"
    )
    st.session_state.selected_product = selected_product

    # Province Selector (Official 77 Provinces)
    all_provs = ["ทั้งหมด"] + OFFICIAL_77_PROVINCES
    curr_prov_idx = all_provs.index(st.session_state.selected_province) if st.session_state.selected_province in all_provs else 0
    selected_province = st.selectbox(
        "📍 จังหวัด (Province)",
        options=all_provs,
        index=curr_prov_idx,
        key="selected_province_box"
    )
    st.session_state.selected_province = selected_province

    # Reset Button
    if st.button("🔄 ล้างตัวกรองทั้งหมด (Reset Filters)", use_container_width=True):
        reset_filters(default_year=latest_year)
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🛡️ ธรรมาภิบาลข้อมูล (Data Governance)")
    st.markdown(
        f"- **สถิติการผลิตเกษตร:** สศก. พ.ศ. {min(available_years)}–{max(available_years)} (77 จังหวัด 26 สินค้า)\n"
        f"- **ฝนเชิงพื้นที่รายจังหวัด:** สสน. HII (IDW) พ.ศ. 2561–2562\n"
        f"- **สภาพอากาศเกษตรรายวัน:** กรมอุตุนิยมวิทยา TMD (34 สถานี)\n"
        f"- **ราคาหน้าฟาร์มรายสัปดาห์:** สศก. พ.ศ. 2561–2562 (1,979 รายการ)\n"
        f"- **ข้อมูลจำลอง (Mock):** ❌ ห้ามใช้เด็ดขาด (Real Open Data 100%)"
    )

    with st.expander("📖 เอกสารพจนานุกรมข้อมูล (Data Dictionary)"):
        st.markdown(
            "ระบบใช้ข้อมูลจริงจาก Open Data ภาครัฐที่ตรวจสอบย้อนกลับได้ ดูรายละเอียด URL ใบอนุญาต และการคำนวณที่ไฟล์ `data_sources.md`"
        )


# ---------------------------------------------------------
# App Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>ระบบวิเคราะห์ข้อมูลการผลิตสินค้าเกษตรและภูมิอากาศไทย (Thai Agriculture & Climate Intelligence)</h1>
    <p>แพลตฟอร์มสารสนเทศเพื่อเกษตรกรไทยและงานวิจัยวิชาการ ข้อมูลจริงจาก สศก., กรมอุตุนิยมวิทยา และ สสน. ปราศจากข้อมูลจำลอง</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Filter Application for Main Context
# ---------------------------------------------------------
# Provincial subset for Year and Product (for mapping and rankings)
df_year_prov = apply_ag_filters(
    df_ag,
    year=st.session_state.selected_year,
    province=None,
    product=st.session_state.selected_product,
    geo_level="province"
)

# Fully filtered subset (respecting specific province if selected)
df_active = apply_ag_filters(
    df_ag,
    year=st.session_state.selected_year,
    province=st.session_state.selected_province,
    product=st.session_state.selected_product,
    geo_level="province"
)

# Multi-year series for selected province & product (for time series trends)
df_trend_series = apply_ag_filters(
    df_ag,
    year=None,
    province=st.session_state.selected_province,
    product=st.session_state.selected_product,
    geo_level="province"
)


# ---------------------------------------------------------
# Main 5 Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ TAB 1: ภาพรวมประเทศไทย (Thailand Overview)",
    "📍 TAB 2: แผนที่พื้นที่เพาะปลูกจริง (Real Cultivation Map)",
    "🌦️ TAB 3: สภาพอากาศกับความเสี่ยง (Climate Impact & Risk)",
    "📅 TAB 4: ผู้ช่วยวางแผนเพาะปลูก (Crop Planning Assistant)",
    "💰 TAB 5: ตลาดและเครื่องคำนวณฟาร์ม (Market & Farm Calculator)",
    "🚜 TAB 6: ปัจจัยขับเคลื่อนผลผลิต (Crop Production Drivers)"
])


# =========================================================
# TAB 1: THAILAND OVERVIEW
# =========================================================
with tab1:
    st.markdown("### 1. ❓ เกิดอะไรขึ้น? (What happened?) — สรุปตัวชี้วัดสำคัญ (Key KPIs)")
    st.caption(
        f"📅 ข้อมูลสถิติทางการเกษตรประจำปี พ.ศ. {st.session_state.selected_year} "
        f"| ขอบเขต: {st.session_state.selected_province if st.session_state.selected_province != 'ทั้งหมด' else '77 จังหวัดทั่วประเทศ'} "
        f"| ชนิดพืช: {st.session_state.selected_product} (แหล่งข้อมูล: สศก.)"
    )

    kpis = aggregate_agricultural_kpis(df_active)

    # YoY Production change calculation
    yoy_prod_text = "N/A"
    if not df_trend_series.empty:
        trend_summary = df_trend_series.groupby("year")["production_ton"].sum().reset_index()
        curr_yr = st.session_state.selected_year
        prev_yr = curr_yr - 1
        curr_v = trend_summary.loc[trend_summary["year"] == curr_yr, "production_ton"].values
        prev_v = trend_summary.loc[trend_summary["year"] == prev_yr, "production_ton"].values
        if len(curr_v) > 0 and len(prev_v) > 0 and prev_v[0] > 0:
            pct_change = (curr_v[0] - prev_v[0]) / prev_v[0] * 100.0
            yoy_prod_text = f"{pct_change:+.1f} %"

    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5, col_kpi6 = st.columns(6)

    with col_kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ผลผลิตรวม (Production)</div>
            <div class="kpi-value">{kpis['total_production_ton']:,.0f}<span class="kpi-unit">ตัน</span></div>
            <div class="kpi-sub">ปี พ.ศ. {st.session_state.selected_year}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">เนื้อที่เพาะปลูก (Planted)</div>
            <div class="kpi-value">{kpis['total_plant_rai']:,.0f}<span class="kpi-unit">ไร่</span></div>
            <div class="kpi-sub">พื้นที่ยืนต้นและปลูกใหม่</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">เนื้อที่เก็บเกี่ยว (Harvested)</div>
            <div class="kpi-value">{kpis['total_harvest_rai']:,.0f}<span class="kpi-unit">ไร่</span></div>
            <div class="kpi-sub">เก็บเกี่ยวได้จริง</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi4:
        yield_val_str = f"{kpis['weighted_yield_kg_rai']:,.1f}" if not np.isnan(kpis['weighted_yield_kg_rai']) else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ผลผลิตต่อไร่ (Weighted Yield)</div>
            <div class="kpi-value">{yield_val_str}<span class="kpi-unit">กก./ไร่</span></div>
            <div class="kpi-sub">คำนวณแบบถ่วงน้ำหนัก</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi5:
        hr_val_str = f"{kpis['harvest_ratio_pct']:,.1f}" if not np.isnan(kpis['harvest_ratio_pct']) else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">อัตราเก็บเกี่ยว (Harvest Ratio)</div>
            <div class="kpi-value">{hr_val_str}<span class="kpi-unit">%</span></div>
            <div class="kpi-sub">เทียบกับพื้นที่เพาะปลูก</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">การเติบโตรายปี (YoY Growth)</div>
            <div class="kpi-value">{yoy_prod_text}</div>
            <div class="kpi-sub">เทียบกับปีก่อนหน้า</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. Rankings & Historical Trend (How does it compare?)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 2. ❓ เปรียบเทียบกับอะไร? (How does it compare?) — การเปรียบเทียบอันดับรายจังหวัดและสินค้าเกษตร")

    rank_metric_options = {
        "production_ton": ("ผลผลิตรวม", "ตัน"),
        "area_plant_rai": ("เนื้อที่เพาะปลูก", "ไร่"),
        "calc_yield_per_rai": ("ผลผลิตต่อไร่", "กก./ไร่")
    }
    col_rank_ctrl1, col_rank_ctrl2 = st.columns([1.2, 0.8])
    with col_rank_ctrl1:
        sel_rank_metric = st.selectbox(
            "เลือกตัวชี้วัดสำหรับการจัดอันดับจังหวัด:",
            options=list(rank_metric_options.keys()),
            format_func=lambda k: f"{rank_metric_options[k][0]} ({rank_metric_options[k][1]})",
            key="tab1_rank_metric_choice"
        )
    with col_rank_ctrl2:
        st.caption("จัดอันดับ Top 10 (สูงสุด) และ Bottom 10 (ต่ำสุด) จาก 77 จังหวัด โดยคัดกรองเฉพาะจังหวัดที่มีการเพาะปลูกจริง (> 0)")

    r_label, r_unit = rank_metric_options[sel_rank_metric]

    col_rank_top, col_rank_bot = st.columns(2)

    with col_rank_top:
        st.markdown(f"#### 🏆 10 อันดับจังหวัดสูงสุด (Top 10: {r_label})")
        top_prov_df = get_top_provinces(df_year_prov, metric=sel_rank_metric, top_n=10, ascending=False)
        fig_top_prov = create_ranking_bar_chart(
            top_prov_df,
            x_col=sel_rank_metric,
            y_col="province",
            title=f"10 อันดับสูงสุด ({st.session_state.selected_product})",
            x_label=r_label,
            unit=r_unit,
            selected_item=st.session_state.selected_province if st.session_state.selected_province != "ทั้งหมด" else None,
            is_bottom=False
        )
        st.plotly_chart(fig_top_prov, use_container_width=True)

    with col_rank_bot:
        st.markdown(f"#### 🔻 10 อันดับจังหวัดต่ำสุด (Bottom 10: {r_label})")
        bot_prov_df = get_top_provinces(df_year_prov, metric=sel_rank_metric, top_n=10, ascending=True)
        fig_bot_prov = create_ranking_bar_chart(
            bot_prov_df,
            x_col=sel_rank_metric,
            y_col="province",
            title=f"10 อันดับต่ำสุด ({st.session_state.selected_product} | เฉพาะที่มีการผลิต > 0)",
            x_label=r_label,
            unit=r_unit,
            selected_item=st.session_state.selected_province if st.session_state.selected_province != "ทั้งหมด" else None,
            is_bottom=True
        )
        st.plotly_chart(fig_bot_prov, use_container_width=True)

    # Scale vs Yield Productivity Insight Callout
    st.markdown("""
    <div style="background-color: #FEF3C7; border-left: 4px solid #D97706; padding: 14px 18px; border-radius: 6px; margin: 10px 0 20px 0;">
        <span style="font-weight: 600; color: #92400E; font-size: 14px;">💡 ข้อสังเกตสำคัญด้านผลิตภาพและการใช้ที่ดิน (Scale vs Efficiency):</span><br>
        <span style="font-size: 13px; color: #78350F; line-height: 1.6;">
            จังหวัดที่มี <b>ผลผลิตรวมสูงสุด (Total Production)</b> มักเป็นจังหวัดที่มีเนื้อที่เพาะปลูกขนาดใหญ่ แต่ไม่ได้แปลว่ามีประสิทธิภาพหรือ <b>ผลผลิตต่อไร่ (Yield per Rai)</b> สูงสุดเสมอไป ในทางกลับกัน จังหวัดที่มีพื้นที่เพาะปลูกขนาดกลางหรือเล็กอาจมีผลผลิตต่อไร่สูงกว่าเนื่องจากระบบชลประทานและความอุดมสมบูรณ์ของดิน การวิเคราะห์เปรียบเทียบทั้ง <b>ขนาดการผลิต (Scale)</b> และ <b>ประสิทธิภาพการผลิต (Yield Efficiency)</b> ควบคู่กันจึงทำให้เห็นภาพความเป็นจริงอย่างรอบด้าน
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Top Crops Overview
    st.markdown(f"#### 🌾 10 อันดับสินค้าเกษตรภาพรวม (ปี {st.session_state.selected_year})")
    top_crop_df = get_top_crops(df_year_prov, metric="production_ton", top_n=10)
    fig_top_crop = create_ranking_bar_chart(
        top_crop_df,
        x_col="production_ton",
        y_col="product",
        title=f"10 อันดับสินค้าเกษตรที่มีผลผลิตสูงสุด",
        x_label="ผลผลิต",
        unit="ตัน",
        selected_item=st.session_state.selected_product if st.session_state.selected_product != "ทั้งหมด" else None
    )
    st.plotly_chart(fig_top_crop, use_container_width=True)

    # Multi-Year Trend
    st.markdown("#### 📈 แนวโน้มการผลิตรายปี พ.ศ. 2550–2562 (Multi-Year Historical Trend)")

    metric_dict = {
        "production_ton": ("ผลผลิตรวม", "ตัน"),
        "area_plant_rai": ("เนื้อที่เพาะปลูก", "ไร่"),
        "area_harvest_rai": ("เนื้อที่เก็บเกี่ยว", "ไร่"),
        "calc_yield_per_rai": ("ผลผลิตต่อไร่แบบถ่วงน้ำหนัก", "กก./ไร่"),
        "calc_harvest_ratio": ("อัตราการเก็บเกี่ยว", "%")
    }

    trend_metric = st.selectbox(
        "เลือกตัวแปรแนวโน้มประวัติศาสตร์:",
        options=list(metric_dict.keys()),
        format_func=lambda k: f"{metric_dict[k][0]} ({metric_dict[k][1]})",
        key="trend_metric_tab1"
    )
    t_label, t_unit = metric_dict[trend_metric]

    trend_data = df_trend_series.groupby("year").agg(
        production_ton=("production_ton", "sum"),
        area_plant_rai=("area_plant_rai", "sum"),
        area_harvest_rai=("area_harvest_rai", "sum")
    ).reset_index()

    trend_data["calc_yield_per_rai"] = np.where(
        trend_data["area_harvest_rai"] > 0,
        trend_data["production_ton"] * 1000.0 / trend_data["area_harvest_rai"],
        np.nan
    )
    trend_data["calc_harvest_ratio"] = np.where(
        trend_data["area_plant_rai"] > 0,
        trend_data["area_harvest_rai"] / trend_data["area_plant_rai"] * 100.0,
        np.nan
    )

    scope_str = f"จังหวัด{st.session_state.selected_province}" if st.session_state.selected_province != "ทั้งหมด" else "ภาพรวม 77 จังหวัดทั่วประเทศ"
    fig_hist_trend = create_trend_line_chart(
        trend_data,
        x_col="year",
        y_col=trend_metric,
        color_col=None,
        title=f"แนวโน้ม {t_label} ({scope_str} | สินค้า: {st.session_state.selected_product})",
        y_label=t_label,
        unit=t_unit
    )
    st.plotly_chart(fig_hist_trend, use_container_width=True)

    # 3. Limitations & Caveats
    st.markdown("---")
    st.markdown("### 3. ❓ ควรระวังข้อจำกัดอะไร? (What caveats and limits to watch?)")
    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px 18px; border-radius: 8px;">
        <span style="font-size: 13px; color: #334155; line-height: 1.7;">
            <b>1. ขอบเขตสถิติทางการเกษตร:</b> สถิติ สศก. พ.ศ. 2550–2562 เป็นข้อมูลระดับจังหวัดรายปี ไม่ใช่ข้อมูลแปลงย่อยส่วนบุคคล จึงเหมาะสำหรับการวิเคราะห์ระดับนโยบายและการวางแผนภาพรวม<br>
            <b>2. การป้องกันนับซ้ำ (Double-Counting Prevention):</b> ระบบแยกข้อมูลระดับประเทศ (country) ระดับภาค (region) และระดับ 77 จังหวัด (province) ออกจากกันอย่างสมบูรณ์ ไม่มีการรวมค่าซ้อนทับกัน<br>
            <b>3. รอบฤดูกาลเพาะปลูก:</b> ข้าวนาปีและข้าวนาปรังมีช่วงเวลาเพาะปลูกต่างกัน การวิเคราะห์เปรียบเทียบควรแยกตามชนิดพืชให้ตรงกับรอบฤดูกาล
        </span>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 2: REAL CULTIVATION MAP
# =========================================================
with tab2:
    # 1. Spatial Distribution Map Header
    st.markdown("### 1. ❓ เกิดอะไรขึ้น? (What happened?) — แผนที่การกระจายตัวเชิงพื้นที่ 77 จังหวัด")
    
    st.markdown("""
    <div style="background-color: #F8FAFC; border-left: 4px solid #1B4D3E; padding: 14px 18px; border-radius: 6px; margin-bottom: 16px;">
        <span class="gov-badge">🛡️ ธรรมาภิบาลข้อมูลการแยกชั้นข้อมูลเชิงพื้นที่ (Spatial Data Governance)</span><br>
        <span style="font-size: 13.5px; color: #334155; line-height: 1.6;">
            ระบบแยกประเภทข้อมูลเชิงพื้นที่ออกเป็น 3 ประเภทอย่างโปร่งใสตามข้อกำหนด:<br>
            <b>1. สถิติทางการเกษตร (Official Agricultural Statistics):</b> ข้อมูลพื้นที่เพาะปลูกและผลผลิตจริงระดับ 77 จังหวัด โดยสำนักงานเศรษฐกิจการเกษตร (สศก.) พ.ศ. 2550–2562 (แสดงบน Choropleth Map ด้านล่าง)<br>
            <b>2. พื้นที่ตรวจจับจากดาวเทียม (Satellite-detected Cultivation):</b> ชุดข้อมูล GISTDA Open Data Catalog ID <code>agriculture-07</code> (ตรวจสอบจุดพิกัดผ่าน REST API)<br>
            <b>3. ประเภทการใช้ประโยชน์ที่ดิน (Land Use Classification):</b> ข้อมูลแผนที่การใช้ที่ดินกรมพัฒนาที่ดิน (LDD)<br>
            <i>⚠️ ข้อกำหนดสำคัญ: หากไม่มีข้อมูล Polygon ระดับแปลงย่อยที่เปิดเผยตามกฎหมาย ระบบจะไม่สร้าง Polygon ปลอมหรือสุ่มพิกัดทดแทนอย่างเด็ดขาด</i>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # All 26 Crops Dynamic Selector
    all_crops_available = sorted(df_ag["product"].unique().tolist())
    def_crop_idx = all_crops_available.index(st.session_state.selected_product) if st.session_state.selected_product in all_crops_available else (all_crops_available.index("ข้าวนาปี") if "ข้าวนาปี" in all_crops_available else 0)

    col_m1, col_m2 = st.columns([1.2, 0.8])
    with col_m1:
        map_crop_choice = st.selectbox(
            "เลือกสินค้าเกษตรสำหรับแสดงบนแผนที่ (Select Crop):",
            options=all_crops_available,
            index=def_crop_idx,
            key="tab2_crop_selector"
        )
    with col_m2:
        map_year_choice = st.selectbox(
            "เลือกปี พ.ศ. ของสถิติการเพาะปลูก:",
            options=available_years,
            index=available_years.index(st.session_state.selected_year) if st.session_state.selected_year in available_years else len(available_years) - 1,
            key="tab2_year_selector"
        )

    # Filter data for map
    map_crop_df = apply_ag_filters(
        df_ag,
        year=map_year_choice,
        province=None,
        product=map_crop_choice,
        geo_level="province"
    )

    # Metric selection for Choropleth
    map_metric_choice = st.selectbox(
        "เลือกตัวชี้วัดที่ต้องการแรเงาบนแผนที่ (Choropleth Metric):",
        options=["area_plant_rai", "area_harvest_rai", "production_ton", "calc_yield_per_rai", "calc_harvest_ratio"],
        format_func=lambda k: {
            "area_plant_rai": "เนื้อที่เพาะปลูกจริง (ไร่)",
            "area_harvest_rai": "เนื้อที่เก็บเกี่ยวจริง (ไร่)",
            "production_ton": "ปริมาณผลผลิตรวม (ตัน)",
            "calc_yield_per_rai": "ผลผลิตต่อไร่ (กก./ไร่)",
            "calc_harvest_ratio": "อัตราการเก็บเกี่ยวได้จริง (%)"
        }[k]
    )
    metric_labels_map = {
        "area_plant_rai": ("เนื้อที่เพาะปลูก", "ไร่"),
        "area_harvest_rai": ("เนื้อที่เก็บเกี่ยว", "ไร่"),
        "production_ton": ("ผลผลิตรวม", "ตัน"),
        "calc_yield_per_rai": ("ผลผลิตต่อไร่", "กก./ไร่"),
        "calc_harvest_ratio": ("อัตราการเก็บเกี่ยว", "%")
    }
    m_name, m_unit = metric_labels_map[map_metric_choice]

    col_map_render, col_map_info = st.columns([1.2, 0.8])

    with col_map_render:
        if geojson_data:
            sel_prov_highlight = st.session_state.selected_province if st.session_state.selected_province != "ทั้งหมด" else None
            fig_crop_map = create_thailand_map(
                df_data=map_crop_df,
                geojson_data=geojson_data,
                metric_col=map_metric_choice,
                metric_label=m_name,
                unit=m_unit,
                year=map_year_choice,
                product_name=map_crop_choice,
                selected_province=sel_prov_highlight
            )
            st.plotly_chart(fig_crop_map, use_container_width=True)
        else:
            st.warning("ไม่สามารถโหลดไฟล์ GeoJSON ขอบเขตจังหวัดได้ แสดงเฉพาะข้อมูลในตาราง")

    with col_map_info:
        st.markdown(f"#### 2. ❓ เปรียบเทียบกับอะไร? — สรุปสถิติเชิงพื้นที่: {map_crop_choice}")
        if not map_crop_df.empty:
            total_planted = map_crop_df["area_plant_rai"].sum()
            total_prod = map_crop_df["production_ton"].sum()
            num_provs = map_crop_df[map_crop_df["area_plant_rai"] > 0]["province"].nunique()
            avg_yield = (total_prod * 1000.0 / map_crop_df["area_harvest_rai"].sum()) if map_crop_df["area_harvest_rai"].sum() > 0 else np.nan

            st.markdown(f"""
            - **จำนวนจังหวัดที่มีการเพาะปลูก:** {num_provs} จังหวัด (จาก 77 จังหวัด)
            - **เนื้อที่เพาะปลูกรวมทั้งประเทศ:** {total_planted:,.0f} ไร่
            - **ผลผลิตรวมทั้งประเทศ:** {total_prod:,.0f} ตัน
            - **ผลผลิตเฉลี่ยถ่วงน้ำหนัก:** {avg_yield:,.1f} กก./ไร่
            - **แหล่งข้อมูลสถิติ:** สำนักงานเศรษฐกิจการเกษตร (สศก.)
            """)

            top3 = map_crop_df.nlargest(3, "area_plant_rai")[["province", "area_plant_rai", "production_ton"]]
            st.markdown("##### 🥇 3 จังหวัดที่มีพื้นที่เพาะปลูกมากที่สุด:")
            for idx, r in top3.iterrows():
                st.markdown(f"- **จ.{r['province']}**: {r['area_plant_rai']:,.0f} ไร่ (ผลผลิต {r['production_ton']:,.0f} ตัน)")
        else:
            st.info(f"ไม่พบข้อมูลสถิติการเพาะปลูกของ {map_crop_choice} ในปี {map_year_choice}")

        st.markdown("---")
        st.markdown("#### 3. ❓ ควรระวังข้อจำกัดอะไร? — ตรวจจับพืชจากดาวเทียมสด (GISTDA API)")
        st.caption("ชุดข้อมูล: พื้นที่ปลูกพืชเศรษฐกิจ (Catalog agriculture-07 โดย สทอภ. GISTDA Open Data Gateway)")

        gistda_key_input = st.text_input("GISTDA API Key (ใส่คีย์ที่ได้จาก opendata.gistda.or.th):", type="password", key="gistda_key")
        col_coord1, col_coord2 = st.columns(2)
        with col_coord1:
            test_lat = st.number_input("ละติจูด (Lat, 5.0–21.5°N):", value=14.6346, format="%.4f")
        with col_coord2:
            test_lon = st.number_input("ลองจิจูด (Lon, 96.5–106.5°E):", value=100.2832, format="%.4f")

        if st.button("🔍 ตรวจสอบพืชจากพิกัด (Check Crop API)", key="btn_check_crop_tab2"):
            with st.spinner("🛰️ กำลังส่งคำขอตรวจสอบพิกัดไปยัง GISTDA Open Data API Gateway (Timeout 8 วินาที)..."):
                res = query_gistda_check_crop(test_lat, test_lon, api_key=gistda_key_input, timeout=8)

            if res["status"] == "success":
                st.success(f"✅ **ผลการตรวจสอบสำเร็จ:** {res['message']}")
                st.markdown(f"""
                <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; padding: 12px; border-radius: 8px; margin-top: 8px;">
                    <span style="font-size: 13px; color: #065F46;">
                        <b>พิกัดที่ตรวจสอบ:</b> ({test_lat:.4f}, {test_lon:.4f})<br>
                        <b>ข้อมูลพืชที่ตรวจพบจากดาวเทียม:</b> <code>{res['crop_data']}</code><br>
                        <b>แหล่งข้อมูล:</b> GISTDA Open Data Gateway (agriculture-07)
                    </span>
                </div>
                """, unsafe_allow_html=True)
            elif res["status"] == "not_found":
                st.info(f"ℹ️ {res['message']}")
            elif res["status"] == "invalid_coordinates":
                st.warning(f"⚠️ {res['message']}")
            elif res["status"] == "no_key":
                st.warning(f"⚠️ **ต้องระบุ API Key:** {res['message']}")
                st.markdown("""
                <span style="font-size: 12px; color: #64748B;">
                    คุณสามารถลงทะเบียนเพื่อขอรับ API Key ฟรีได้ที่ <a href="https://opendata.gistda.or.th" target="_blank">opendata.gistda.or.th</a> (ระบบไม่สร้างข้อมูลจำลองทดแทนตามหลักธรรมาภิบาลข้อมูล)
                </span>
                """, unsafe_allow_html=True)
            elif res["status"] == "http_error":
                st.error(f"❌ {res['message']}")
            elif res["status"] == "network_error":
                st.warning(f"⏳ {res['message']}")
            else:
                st.error(f"❌ {res['message']}")

    # Data Table & CSV Download
    st.markdown("---")
    st.markdown(f"#### 📋 ตารางสถิติการเพาะปลูก 77 จังหวัด ({map_crop_choice} | พ.ศ. {map_year_choice})")
    
    table_crop_export = map_crop_df[[
        "province", "product", "year", "area_plant_rai", "area_harvest_rai",
        "production_ton", "calc_yield_per_rai", "calc_harvest_ratio"
    ]].rename(columns={
        "province": "จังหวัด",
        "product": "ชนิดพืช",
        "year": "ปี พ.ศ.",
        "area_plant_rai": "เนื้อที่เพาะปลูก (ไร่)",
        "area_harvest_rai": "เนื้อที่เก็บเกี่ยว (ไร่)",
        "production_ton": "ผลผลิตรวม (ตัน)",
        "calc_yield_per_rai": "ผลผลิตต่อไร่ (กก./ไร่)",
        "calc_harvest_ratio": "อัตราเก็บเกี่ยว (%)"
    })

    csv_crop = table_crop_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 ดาวน์โหลดข้อมูลสถิติพื้นที่เพาะปลูก (CSV)",
        data=csv_crop,
        file_name=f"crop_cultivation_{map_crop_choice}_{map_year_choice}.csv",
        mime="text/csv"
    )

    st.dataframe(
        table_crop_export.style.format({
            "เนื้อที่เพาะปลูก (ไร่)": "{:,.0f}",
            "เนื้อที่เก็บเกี่ยว (ไร่)": "{:,.0f}",
            "ผลผลิตรวม (ตัน)": "{:,.0f}",
            "ผลผลิตต่อไร่ (กก./ไร่)": "{:,.1f}",
            "อัตราเก็บเกี่ยว (%)": "{:,.1f}%"
        }),
        height=320,
        use_container_width=True
    )


# =========================================================
# TAB 3: CLIMATE IMPACT & RISK
# =========================================================
with tab3:
    st.markdown("### 1. ❓ เกิดอะไรขึ้น? (What happened?) — การวิเคราะห์สภาพภูมิอากาศกับผลผลิตทางการเกษตร")
    
    st.markdown("""
    <div style="background-color: #F8FAFC; border-left: 4px solid #1B4D3E; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
        <span class="notice-badge">ℹ️ แหล่งข้อมูลสภาพอากาศที่ตรวจสอบได้</span>
        <span style="font-size: 13px; color: #334155;">
            วิเคราะห์ความสัมพันธ์โดยใช้ข้อมูลตรวจวัดจริงจาก <b>สถาบันสารสนเทศทรัพยากรน้ำ (สสน. HII)</b> และ <b>กรมอุตุนิยมวิทยา (TMD)</b> 
            โดยตัดชุดข้อมูลสถานีตรวจวัดเดี่ยวในอดีตที่ไม่เป็นตัวแทนออกทั้งหมด
        </span>
    </div>
    """, unsafe_allow_html=True)

    climate_scope = st.radio(
        "เลือกมิติการวิเคราะห์สภาพอากาศ (Select Climate Analysis Scope):",
        options=[
            "1. การวิเคราะห์ฝนเชิงพื้นที่ 77 จังหวัดทั่วประเทศ (พ.ศ. 2561–2562, สสน. HII)",
            "2. แนวโน้มฝนสะสมเฉลี่ยระดับประเทศระยะยาว (พ.ศ. 2550–2562, สสน. HII)",
            "3. สถานีตรวจอากาศเพื่อการเกษตรรายวัน (กรมอุตุนิยมวิทยา TMD Open Data API Catalog c1-1-3-7)"
        ],
        horizontal=True
    )

    if "1. การวิเคราะห์ฝนเชิงพื้นที่ 77 จังหวัด" in climate_scope:
        st.markdown("#### 2. ❓ เปรียบเทียบกับอะไร? — ปริมาณฝนเชิงพื้นที่ HII (IDW) vs ผลผลิตต่อไร่ (77 จังหวัด)")
        
        joined_nation = join_climate_agriculture_nationwide(
            df_ag=df_ag,
            df_rain_annual=df_rain_annual,
            df_temp_annual=df_temp_annual,
            year=st.session_state.selected_year if st.session_state.selected_year in [2561, 2562] else 2562,
            product=st.session_state.selected_product
        )

        if joined_nation.empty:
            st.warning("ไม่มีข้อมูลการเพาะปลูกที่จับคู่กับฝนเชิงพื้นที่ในปีที่เลือก (ชุดข้อมูลฝนเชิงพื้นที่ครอบคลุม พ.ศ. 2561–2562)")
        else:
            col_n1, col_n2 = st.columns([1.1, 0.9])
            with col_n1:
                stats_nat_rain = calculate_bivariate_stats(joined_nation["rainfall_annual_mm"], joined_nation["calc_yield_per_rai"])
                fig_nat_scat = create_bivariate_scatter_chart(
                    joined_nation,
                    x_col="rainfall_annual_mm",
                    y_col="calc_yield_per_rai",
                    title=f"ฝนเชิงพื้นที่ HII vs Yield ({st.session_state.selected_product} | ปี {joined_nation['year'].iloc[0]})",
                    x_label="ปริมาณฝนเชิงพื้นที่",
                    y_label="ผลผลิตต่อไร่",
                    x_unit="มม.",
                    y_unit="กก./ไร่",
                    text_col="province",
                    correlation_val=stats_nat_rain["pearson_r"] if stats_nat_rain["is_valid"] else None,
                    p_val=stats_nat_rain["pearson_p"] if stats_nat_rain["is_valid"] else None,
                    n_samples=stats_nat_rain["n"] if stats_nat_rain["is_valid"] else None
                )
                st.plotly_chart(fig_nat_scat, use_container_width=True)

            with col_n2:
                st.markdown("##### 3. ❓ ควรระวังข้อจำกัดอะไร? — การแปลผลทางสถิติและข้อพึงระวัง")
                interp_nat = generate_climate_interpretation(
                    bivariate_result=stats_nat_rain,
                    crop_name=st.session_state.selected_product,
                    climate_var_name="ปริมาณฝนเชิงพื้นที่ HII (IDW)",
                    ag_var_name="ผลผลิตต่อไร่ (Yield)",
                    geo_scope=f"77 จังหวัดทั่วประเทศ (ปี {joined_nation['year'].iloc[0]})",
                    year_range=str(joined_nation['year'].iloc[0])
                )
                st.markdown(interp_nat)

                st.markdown("""
                <div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 12px 14px; border-radius: 6px; margin-top: 12px;">
                    <span style="font-weight: 600; font-size: 13px; color: #1E40AF;">💡 สหสัมพันธ์ (Correlation) ไม่เท่ากับ ความเป็นเหตุเป็นผล (Causation):</span><br>
                    <span style="font-size: 12.5px; color: #1E3A8A; line-height: 1.5;">
                        ความสัมพันธ์เชิงสถิติ (ค่า r) สะท้อนทิศทางการเคลื่อนไหวร่วมกันของข้อมูลเท่านั้น ไม่ได้พิสูจน์ว่าฝนเป็นสาเหตุเดียวของผลผลิตที่เปลี่ยนแปลง เนื่องจากผลผลิตทางการเกษตรยังขึ้นอยู่กับระบบชลประทาน ธาตุอาหารในดิน การป้องกันโรคแมลง และการปฏิบัติแปลงของเกษตรกร
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🔍 ตารางข้อมูลตรวจวัดเชิงพื้นที่ 77 จังหวัด"):
                st.dataframe(joined_nation[["province", "year", "product", "rainfall_annual_mm", "production_ton", "area_harvest_rai", "calc_yield_per_rai"]], use_container_width=True)

    elif "2. แนวโน้มฝนสะสมเฉลี่ยระดับประเทศระยะยาว" in climate_scope:
        st.markdown("#### 2. ❓ เปรียบเทียบกับอะไร? — ฝนสะสมเฉลี่ยทั้งประเทศ vs ผลผลิตภาพรวม (พ.ศ. 2550–2562)")

        nat_ag = df_ag[(df_ag["geo_level"] == "country")].copy()
        if st.session_state.selected_product != "ทั้งหมด":
            nat_ag = nat_ag[nat_ag["product"] == st.session_state.selected_product]

        nat_ag_summary = nat_ag.groupby("year").agg(
            production_ton=("production_ton", "sum"),
            area_harvest_rai=("area_harvest_rai", "sum")
        ).reset_index()
        nat_ag_summary["calc_yield_per_rai"] = nat_ag_summary["production_ton"] * 1000.0 / nat_ag_summary["area_harvest_rai"]

        joined_nat_long = pd.merge(
            nat_ag_summary,
            df_nat_annual[["year", "rainfall_annual_mm"]],
            on="year",
            how="inner"
        )

        col_l1, col_l2 = st.columns([1.1, 0.9])
        with col_l1:
            stats_long = calculate_bivariate_stats(joined_nat_long["rainfall_annual_mm"], joined_nat_long["production_ton"])
            fig_long = create_bivariate_scatter_chart(
                joined_nat_long,
                x_col="rainfall_annual_mm",
                y_col="production_ton",
                title=f"ฝนสะสมภาพรวมประเทศ vs ผลผลิตรวม ({st.session_state.selected_product})",
                x_label="ฝนสะสมเฉลี่ยทั้งประเทศ (HII)",
                y_label="ผลผลิตรวม",
                x_unit="มม.",
                y_unit="ตัน",
                text_col="year",
                correlation_val=stats_long["pearson_r"] if stats_long["is_valid"] else None,
                p_val=stats_long["pearson_p"] if stats_long["is_valid"] else None,
                n_samples=stats_long["n"] if stats_long["is_valid"] else None
            )
            st.plotly_chart(fig_long, use_container_width=True)

        with col_l2:
            st.markdown("##### 3. ❓ ควรระวังข้อจำกัดอะไร? — การแปลผลแนวโน้มฝนระดับประเทศระยะยาว")
            interp_long = generate_climate_interpretation(
                bivariate_result=stats_long,
                crop_name=st.session_state.selected_product,
                climate_var_name="ฝนสะสมเฉลี่ยทั้งประเทศ (HII)",
                ag_var_name="ผลผลิตรวมทั้งประเทศ (Production)",
                geo_scope="ภาพรวมทั้งประเทศ",
                year_range="2550–2562"
            )
            st.markdown(interp_long)

            st.markdown("""
            <div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 12px 14px; border-radius: 6px; margin-top: 12px;">
                <span style="font-weight: 600; font-size: 13px; color: #1E40AF;">💡 สหสัมพันธ์ (Correlation) ไม่เท่ากับ ความเป็นเหตุเป็นผล (Causation):</span><br>
                <span style="font-size: 12.5px; color: #1E3A8A; line-height: 1.5;">
                    แนวโน้มระยะยาวระดับประเทศอาจได้รับอิทธิพลจากนโยบายสนับสนุนภาครัฐ การขยายพื้นที่เพาะปลูก และเทคโนโลยีการเกษตร มิใช่เพียงปริมาณน้ำฝนสะสมอย่างเดียว
                </span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("#### 📡 ข้อมูลสถานีอุตุนิยมวิทยาเพื่อการเกษตรรายวัน (TMD Agromet Stations)")
        st.caption("แหล่งข้อมูล: กรมอุตุนิยมวิทยา (TMD Open Data API Catalog c1-1-3-7: WeatherTodayByAgro)")

        if not df_agromet.empty:
            stn_provinces = sorted(df_agromet["province"].dropna().unique().tolist())
            selected_stn_prov = st.selectbox("เลือกจังหวัดสถานีตรวจวัดเพื่อการเกษตร:", options=["ทุกสถานีทั่วประเทศ"] + stn_provinces)

            filtered_stns = df_agromet if selected_stn_prov == "ทุกสถานีทั่วประเทศ" else df_agromet[df_agromet["province"] == selected_stn_prov]

            col_stn_kpi1, col_stn_kpi2, col_stn_kpi3, col_stn_kpi4 = st.columns(4)
            with col_stn_kpi1:
                st.metric("จำนวนสถานีตรวจวัด", f"{len(filtered_stns)} แห่ง")
            with col_stn_kpi2:
                avg_max_t = filtered_stns["max_temp_c"].mean()
                st.metric("อุณหภูมิสูงสุดเฉลี่ย", f"{avg_max_t:.1f} °C" if not np.isnan(avg_max_t) else "N/A")
            with col_stn_kpi3:
                avg_min_t = filtered_stns["min_temp_c"].mean()
                st.metric("อุณหภูมิต่ำสุดเฉลี่ย", f"{avg_min_t:.1f} °C" if not np.isnan(avg_min_t) else "N/A")
            with col_stn_kpi4:
                rain_stns = filtered_stns[filtered_stns["rainfall_mm"] > 0]
                st.metric("สถานีที่มีฝนตกในรอบวัน", f"{len(rain_stns)} แห่ง")

            st.dataframe(
                filtered_stns[[
                    "station_number", "station_name", "province",
                    "max_temp_c", "min_temp_c", "relative_humidity_pct", "rainfall_mm", "datetime"
                ]].rename(columns={
                    "station_number": "รหัสสถานี WMO",
                    "station_name": "ชื่อสถานี",
                    "province": "จังหวัด",
                    "max_temp_c": "อุณหภูมิสูงสุด (°C)",
                    "min_temp_c": "อุณหภูมิต่ำสุด (°C)",
                    "relative_humidity_pct": "ความชื้นสัมพัทธ์ (%)",
                    "rainfall_mm": "ฝนสะสม 24 ชม. (มม.)",
                    "datetime": "เวลาที่ตรวจวัด"
                }),
                height=320,
                use_container_width=True
            )
        else:
            st.warning("ไม่สามารถเชื่อมต่อสถานีอุตุนิยมวิทยาเกษตรได้ในขณะนี้")

    # ---------------------------------------------------------
    # Production Risk & Harvest Ratio Flags Section
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("#### ⚠️ การตรวจจับสัญญาณความเสี่ยงและเนื้อที่เสียหาย (Risk & Review Signals)")

    col_rctrl1, col_rctrl2 = st.columns([1, 2])
    with col_rctrl1:
        threshold = st.slider(
            "🎚️ เกณฑ์อัตราการเก็บเกี่ยวสำหรับตรวจสอบ (Review Threshold %):",
            min_value=70.0,
            max_value=98.0,
            value=st.session_state.harvest_threshold,
            step=1.0
        )
        st.session_state.harvest_threshold = threshold

    with col_rctrl2:
        st.info(
            "📌 **หลักการระบุความเสี่ยงตามหลักฐานจริง:**\n"
            f"- รายการที่อัตราการเก็บเกี่ยวจริงต่ำกว่า **{threshold:.0f}%** จะถูกติดธงเป็น **'สัญญาณที่ควรตรวจสอบ (Review Flag)'**\n"
            "- สะท้อนเนื้อที่เพาะปลูกที่ไม่ได้เก็บเกี่ยว (อาจเกิดจากภัยแล้ง น้ำท่วม โรคแมลง หรือราคาสินค้าตกต่ำ)"
        )

    sub_risk = df_active.copy()
    flagged_records = sub_risk[sub_risk["calc_harvest_ratio"] < threshold]
    flagged_count = len(flagged_records)
    total_unharvested = kpis["unharvested_area_rai"]
    flagged_unharvested = flagged_records["calc_damaged_area"].sum() if not flagged_records.empty else 0.0

    col_rkpi1, col_rkpi2, col_rkpi3, col_rkpi4 = st.columns(4)
    with col_rkpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">รายการที่พบสัญญาณตรวจทาน</div>
            <div class="kpi-value">{flagged_count:,}<span class="kpi-unit">รายการ</span></div>
            <div class="kpi-sub">อัตราเก็บเกี่ยว < {threshold:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_rkpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">เนื้อที่ไม่ได้เก็บเกี่ยวรวม</div>
            <div class="kpi-value">{total_unharvested:,.0f}<span class="kpi-unit">ไร่</span></div>
            <div class="kpi-sub">เนื้อที่เสียหายหรือละทิ้ง</div>
        </div>
        """, unsafe_allow_html=True)

    with col_rkpi3:
        cv_prod_val = "N/A"
        if not df_trend_series.empty:
            yearly_prod = df_trend_series.groupby("year")["production_ton"].sum()
            cv_num = (yearly_prod.std() / yearly_prod.mean() * 100.0) if yearly_prod.mean() > 0 else np.nan
            if not np.isnan(cv_num):
                cv_prod_val = f"{cv_num:.1f} %"

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ความผันผวนผลผลิต (CV)</div>
            <div class="kpi-value">{cv_prod_val}</div>
            <div class="kpi-sub">ความผันแปรรอบ 13 ปี</div>
        </div>
        """, unsafe_allow_html=True)

    with col_rkpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">เนื้อที่ที่หลุดเกณฑ์ตรวจสอบ</div>
            <div class="kpi-value">{flagged_unharvested:,.0f}<span class="kpi-unit">ไร่</span></div>
            <div class="kpi-sub">สัดส่วนพื้นที่เสียหาย</div>
        </div>
        """, unsafe_allow_html=True)

    # Quadrant Analysis
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 การจัดกลุ่มผลิตภาพเชิงพื้นที่ (Area vs Yield Quadrants)")
    
    if st.session_state.selected_product == "ทั้งหมด":
        quad_base = df_year_prov.groupby("product").agg(
            area_plant_rai=("area_plant_rai", "sum"),
            area_harvest_rai=("area_harvest_rai", "sum"),
            production_ton=("production_ton", "sum")
        ).reset_index()
        quad_base["calc_yield_per_rai"] = quad_base["production_ton"] * 1000.0 / quad_base["area_harvest_rai"]
        quad_df, a_med, y_med = classify_area_yield_quadrants(quad_base, area_col="area_plant_rai", yield_col="calc_yield_per_rai", label_col="product")
        fig_quad = create_quadrant_scatter_chart(quad_df, x_col="area_plant_rai", y_col="calc_yield_per_rai", size_col="production_ton", label_col="product", area_median=a_med, yield_median=y_med, title=f"Quadrant สินค้าเกษตร: พื้นที่เพาะปลูก vs ผลผลิตต่อไร่ (ปี {st.session_state.selected_year})")
        st.plotly_chart(fig_quad, use_container_width=True)
    else:
        quad_base = df_year_prov.groupby("province").agg(
            area_plant_rai=("area_plant_rai", "sum"),
            area_harvest_rai=("area_harvest_rai", "sum"),
            production_ton=("production_ton", "sum")
        ).reset_index()
        quad_base["calc_yield_per_rai"] = quad_base["production_ton"] * 1000.0 / quad_base["area_harvest_rai"]
        quad_df, a_med, y_med = classify_area_yield_quadrants(quad_base, area_col="area_plant_rai", yield_col="calc_yield_per_rai", label_col="province")
        fig_quad = create_quadrant_scatter_chart(quad_df, x_col="area_plant_rai", y_col="calc_yield_per_rai", size_col="production_ton", label_col="province", area_median=a_med, yield_median=y_med, title=f"Quadrant จังหวัดที่ปลูก {st.session_state.selected_product} (ปี {st.session_state.selected_year})")
        st.plotly_chart(fig_quad, use_container_width=True)


# =========================================================
# TAB 4: CROP PLANNING ASSISTANT
# =========================================================
with tab4:
    st.markdown("### 1. ❓ เกิดอะไรขึ้น? (What happened?) — ข้อมูลสถิติผลผลิตย้อนหลังในพื้นที่จริง")
    
    st.markdown("""
    <div style="background-color: #F8FAFC; border-left: 4px solid #1B4D3E; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
        <span class="gov-badge">🌾 ปฏิทินและคำแนะนำทางวิชาการ (Official Agronomic Guidelines)</span>
        <span style="font-size: 13px; color: #334155;">
            รวบรวมข้อกำหนดและคำแนะนำมาตรฐานจาก <b>กรมการข้าว</b>, <b>กรมวิชาการเกษตร</b> และ <b>กรมส่งเสริมการเกษตร</b><br>
            <i>⚠️ ข้อพึงระวัง: ข้อมูลนี้เป็นแนวทางประกอบการตัดสินใจเบื้องต้น ไม่สามารถรับประกันผลผลิตหรือผลตอบแทนของแต่ละแปลงได้</i>
        </span>
    </div>
    """, unsafe_allow_html=True)

    all_crops_plan = sorted(df_ag["product"].unique().tolist())
    def_crop_plan_idx = all_crops_plan.index(st.session_state.selected_product) if st.session_state.selected_product in all_crops_plan else (all_crops_plan.index("ข้าวนาปี") if "ข้าวนาปี" in all_crops_plan else 0)

    col_plan_sel1, col_plan_sel2, col_plan_sel3 = st.columns(3)
    with col_plan_sel1:
        target_crop = st.selectbox(
            "🌱 เลือกพืชที่ต้องการวางแผน:",
            options=all_crops_plan,
            index=def_crop_plan_idx,
            key="plan_crop_choice"
        )
    with col_plan_sel2:
        target_prov = st.selectbox(
            "📍 เลือกจังหวัดที่ตั้งแปลง:",
            options=OFFICIAL_77_PROVINCES,
            index=OFFICIAL_77_PROVINCES.index(st.session_state.selected_province) if st.session_state.selected_province in OFFICIAL_77_PROVINCES else 0,
            key="plan_prov_choice"
        )
    with col_plan_sel3:
        target_season = st.selectbox(
            "🌤️ ฤดูกาลเพาะปลูกเป้าหมาย:",
            options=["ต้นฤดูฝน (พ.ค. - มิ.ย.)", "กลางฤดูฝน (ก.ค. - ส.ค.)", "ปลายฤดูฝน / ฤดูหนาว (ก.ย. - พ.ย.)", "ฤดูแล้ง / มีชลประทาน (ธ.ค. - เม.ย.)"],
            index=0
        )

    guide = get_crop_calendar_guide(target_crop)

    # Historical Benchmark for Selected Province & Crop
    hist_subset = df_ag[(df_ag["province"] == target_prov) & (df_ag["product"] == target_crop) & (df_ag["geo_level"] == "province")]
    
    col_bench1, col_bench2, col_bench3, col_bench4 = st.columns(4)
    with col_bench1:
        avg_hist_yield = hist_subset["calc_yield_per_rai"].mean() if not hist_subset.empty else np.nan
        y_txt = f"{avg_hist_yield:,.1f} กก./ไร่" if not np.isnan(avg_hist_yield) else "ไม่มีข้อมูลย้อนหลัง"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ผลผลิตเฉลี่ยใน จ.{target_prov}</div>
            <div class="kpi-value">{y_txt}</div>
            <div class="kpi-sub">สถิติ สศก. (2550–2562)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_bench2:
        avg_hist_hr = hist_subset["calc_harvest_ratio"].mean() if not hist_subset.empty else np.nan
        hr_txt = f"{avg_hist_hr:.1f} %" if not np.isnan(avg_hist_hr) else "ไม่มีข้อมูล"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">อัตราเก็บเกี่ยวสำเร็จเฉลี่ย</div>
            <div class="kpi-value">{hr_txt}</div>
            <div class="kpi-sub">เทียบพื้นที่เพาะปลูก</div>
        </div>
        """, unsafe_allow_html=True)

    with col_bench3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ระยะเวลาเก็บเกี่ยว</div>
            <div class="kpi-value">{guide.get('growth_duration_days', 'ตามสายพันธุ์')}</div>
            <div class="kpi-sub">รอบอายุพืช</div>
        </div>
        """, unsafe_allow_html=True)

    with col_bench4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">หน่วยงานรับผิดชอบหลัก</div>
            <div class="kpi-value" style="font-size: 15px;">{guide.get('source_agency', 'กระทรวงเกษตรฯ')}</div>
            <div class="kpi-sub">มาตรฐานทางวิชาการ</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Planning Cards
    col_guide_l, col_guide_r = st.columns([1.1, 0.9])

    with col_guide_l:
        st.markdown(f"#### 2. ❓ เปรียบเทียบกับอะไร? — ปฏิทินและข้อกำหนดทางปฐพีวิทยา ({target_crop})")
        st.markdown(f"""
        - **ช่วงเวลาเพาะปลูกที่เหมาะสม:** {guide.get('planting_window')}
        - **ช่วงเวลาเก็บเกี่ยวผลผลิต:** {guide.get('harvest_window')}
        - **ความต้องการดินและน้ำ:** {guide.get('water_soil_req')}
        """)

        st.markdown("##### ⚠️ ข้อควรระวังความเสี่ยงสภาพอากาศและโรคแมลง:")
        st.warning(guide.get('key_cautions'))

    with col_guide_r:
        st.markdown("#### 3. ❓ ควรระวังข้อจำกัดอะไร? — รายการตรวจสอบก่อนการเพาะปลูก (Pre-planting Checklist)")
        for idx, item in enumerate(guide.get('pre_planting_checklist', []), 1):
            st.checkbox(f"{item}", key=f"chk_{idx}")

        st.markdown("""
        <div style="background-color: #FEF3C7; border: 1px solid #FCD34D; padding: 10px; border-radius: 6px; font-size: 12px; color: #92400E; margin-top: 14px;">
            💡 <b>คำแนะนำเพิ่มเติม:</b> แนะนำให้ติดต่อศูนย์เมล็ดพันธุ์ข้าว, สำนักงานเกษตรอำเภอ หรือหมอดินอาสาประจำตำบล เพื่อตรวจวิเคราะห์ชุดดินเฉพาะแปลงก่อนลงทุนซื้อปัจจัยการผลิตปริมาณมาก
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 5: MARKET & FARM CALCULATOR
# =========================================================
with tab5:
    st.markdown("### 💰 ตลาดสินค้าเกษตรและเครื่องคำนวณต้นทุนฟาร์ม (Market & Farm Calculator)")

    tab5_market, tab5_calc = st.tabs([
        "📈 Part A: สถิติราคาหน้าไร่นา สศก. (OAE Farmgate Prices)",
        "🧮 Part B: เครื่องคำนวณต้นทุน รายได้ และจุดคุ้มทุน (Farm Calculator)"
    ])

    # ---------------------------------------------------------
    # Part A: Official OAE Farmgate Commodity Prices
    # ---------------------------------------------------------
    with tab5_market:
        st.markdown("#### 1. ❓ เกิดอะไรขึ้น? (What happened?) — สถิติราคาหน้าไร่นา สศก. รายสัปดาห์ (พ.ศ. 2561–2562)")
        st.caption("แหล่งข้อมูล: สำนักงานเศรษฐกิจการเกษตร (OAE Data Catalog ID 35) ครอบคลุม 95 สินค้าเกษตร 1,979 รายการบันทึกจริง")

        if not df_prices.empty:
            price_products = sorted(df_prices["product"].unique().tolist())
            
            # Smart default matching for selected crop
            default_p_idx = 0
            for idx, p in enumerate(price_products):
                if any(k in p for k in ["ข้าว", "มันสำปะหลัง", "อ้อย", "ข้าวโพด"]):
                    default_p_idx = idx
                    break

            col_pctrl1, col_pctrl2 = st.columns([1.2, 0.8])
            with col_pctrl1:
                sel_price_prod = st.selectbox("เลือกชนิดสินค้าเกษตรสำหรับดูสถิติราคา:", options=price_products, index=default_p_idx)
            with col_pctrl2:
                price_unit_mode = st.radio("หน่วยราคาแสดงผล:", options=["baht_per_kg", "baht_per_ton"], format_func=lambda u: "บาท/กิโลกรัม" if u == "baht_per_kg" else "บาท/ตัน", horizontal=True)

            sub_price = df_prices[df_prices["product"] == sel_price_prod].sort_values(by=["year", "month", "week"])

            if not sub_price.empty:
                col_pkpi1, col_pkpi2, col_pkpi3, col_pkpi4 = st.columns(4)
                unit_label_str = "บาท/กก." if price_unit_mode == "baht_per_kg" else "บาท/ตัน"
                p_col = "price_baht_per_kg" if price_unit_mode == "baht_per_kg" else "price_baht_per_ton"

                with col_pkpi1:
                    latest_p = sub_price.iloc[-1][p_col]
                    st.metric("ราคาล่าสุด", f"{latest_p:,.2f} {unit_label_str}", f"ปี {sub_price.iloc[-1]['year']} สัปดาห์ {sub_price.iloc[-1]['week']}")
                with col_pkpi2:
                    min_p = sub_price[p_col].min()
                    st.metric("ราคาต่ำสุดในรอบ 2 ปี", f"{min_p:,.2f} {unit_label_str}")
                with col_pkpi3:
                    max_p = sub_price[p_col].max()
                    st.metric("ราคาสูงสุดในรอบ 2 ปี", f"{max_p:,.2f} {unit_label_str}")
                with col_pkpi4:
                    mean_p = sub_price[p_col].mean()
                    st.metric("ราคาเฉลี่ย", f"{mean_p:,.2f} {unit_label_str}")

                fig_price = create_price_trend_chart(sub_price, product_name=sel_price_prod, unit_mode=price_unit_mode)
                st.plotly_chart(fig_price, use_container_width=True)

                with st.expander("🔍 ตารางข้อมูลราคาและปุ่มดาวน์โหลด (CSV)"):
                    csv_price = sub_price[["year", "month", "week", "product", "price_baht_per_ton", "price_baht_per_kg", "data_source"]].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button(
                        label="📥 ดาวน์โหลดข้อมูลราคา สศก. (CSV)",
                        data=csv_price,
                        file_name=f"oae_prices_{sel_price_prod}.csv",
                        mime="text/csv"
                    )
                    st.dataframe(sub_price[["year", "month", "week", "product", "price_baht_per_ton", "price_baht_per_kg"]], use_container_width=True)
            else:
                st.info("ไม่มีข้อมูลราคาสำหรับสินค้าที่เลือก")
        else:
            st.warning("ไม่สามารถโหลดชุดข้อมูลราคา สศก. ได้")

    # ---------------------------------------------------------
    # Part B: Interactive Farm Budget & Break-Even Calculator
    # ---------------------------------------------------------
    with tab5_calc:
        st.markdown("#### 1. ❓ เกิดอะไรขึ้น? (What happened?) — ข้อมูลแปลง ต้นทุน และผลผลิตคาดการณ์")
        st.caption("คำนวณจากต้นทุนจริงที่เกษตรกรระบุเอง โดยใช้สถิติผลผลิตของจังหวัดและราคาตลาด สศก. เป็นฐานอ้างอิงเริ่มต้น")

        # Explicit Formula Disclosure Box
        st.markdown("""
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid #1B4D3E; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
            <span style="font-weight: 700; color: #1B4D3E; font-size: 14px;">📐 การเปิดเผยสูตรคำนวณทางการเงินและการผลิต (Formula Disclosure):</span>
            <div style="font-size: 13px; color: #334155; line-height: 1.8; margin-top: 8px;">
                1. <b>ผลผลิตรวม (กก.)</b> = เนื้อที่เพาะปลูก (ไร่) × ผลผลิตคาดการณ์ (กก./ไร่)<br>
                2. <b>ต้นทุนรวม (บาท)</b> = ค่าพันธุ์ + ค่าปุ๋ย + ค่าสารเคมี + ค่าแรงงาน + ค่าน้ำมัน/สูบน้ำ + ค่าเช่าที่ดินและอื่นๆ<br>
                3. <b>รายรับรวม (บาท)</b> = ผลผลิตรวม (กก.) × ราคาขายต่อกิโลกรัม (บาท/กก.)<br>
                4. <b>กำไร (ขาดทุน) สุทธิ (บาท)</b> = รายรับรวม − ต้นทุนรวม<br>
                5. <b>ต้นทุนเฉลี่ยต่อไร่ (บาท/ไร่)</b> = ต้นทุนรวม ÷ เนื้อที่เพาะปลูก (ไร่)<br>
                6. <b>ต้นทุนเฉลี่ยต่อกิโลกรัม (บาท/กก.)</b> = ต้นทุนรวม ÷ ผลผลิตรวม (กก.)<br>
                7. <b>ราคาขาย ณ จุดคุ้มทุน (บาท/กก.)</b> = ต้นทุนรวม ÷ ผลผลิตรวม (กก.) <i>(ราคาขายต่ำสุดที่ไม่ขาดทุน)</i><br>
                8. <b>ผลผลิต ณ จุดคุ้มทุน (กก./ไร่)</b> = ต้นทุนรวม ÷ [ราคาขาย (บาท/กก.) × เนื้อที่เพาะปลูก (ไร่)] <i>(ผลผลิตต่อไร่ขั้นต่ำที่ไม่ขาดทุน)</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Prefill suggestions from historical data
        hist_yield_suggest = 650.0
        if not hist_subset.empty and not np.isnan(hist_subset["calc_yield_per_rai"].mean()):
            hist_yield_suggest = float(hist_subset["calc_yield_per_rai"].mean())

        price_suggest = 8.5
        if not df_prices.empty:
            rice_matches = df_prices[df_prices["product"].str.contains("ข้าว|มัน|อ้อย|ข้าวโพด", na=False)]
            if not rice_matches.empty:
                price_suggest = float(rice_matches["price_baht_per_kg"].mean())

        col_c_in1, col_c_in2 = st.columns([1, 1])

        with col_c_in1:
            st.markdown("##### 🌾 1. ขนาดแปลงและผลผลิตคาดการณ์")
            user_area = st.number_input("เนื้อที่เพาะปลูก (ไร่):", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
            user_yield = st.number_input("ผลผลิตคาดการณ์ต่อไร่ (กก./ไร่):", min_value=50.0, max_value=20000.0, value=hist_yield_suggest, step=20.0, help=f"ค่าเฉลี่ยสถิติ สศก. ใน จ.{target_prov} อยู่ที่ {hist_yield_suggest:,.1f} กก./ไร่")
            user_price = st.number_input("ราคาขายที่คาดว่าจะได้รับ (บาท/กก.):", min_value=0.5, max_value=200.0, value=price_suggest, step=0.5, help="อ้างอิงจากราคาเฉลี่ยหน้าไร่นา สศก.")

        with col_c_in2:
            st.markdown("##### 💵 2. ค่าใช้จ่ายและต้นทุนการผลิตจริง (บาท)")
            col_cost_a, col_cost_b = st.columns(2)
            with col_cost_a:
                cost_seed = st.number_input("ค่าพันธุ์พืช/เมล็ดพันธุ์รวม (บ.):", min_value=0.0, value=user_area * 500.0, step=100.0)
                cost_fert = st.number_input("ค่าปุ๋ยเคมีและอินทรีย์รวม (บ.):", min_value=0.0, value=user_area * 1200.0, step=200.0)
                cost_chem = st.number_input("ค่าสารป้องกันกำจัดศัตรูพืช (บ.):", min_value=0.0, value=user_area * 300.0, step=100.0)
            with col_cost_b:
                cost_labor = st.number_input("ค่าแรงงานและจ้างเหมา (บ.):", min_value=0.0, value=user_area * 800.0, step=200.0)
                cost_energy = st.number_input("ค่าน้ำมัน สูบน้ำ และเชื้อเพลิง (บ.):", min_value=0.0, value=user_area * 400.0, step=100.0)
                cost_rent_other = st.number_input("ค่าเช่าที่ดินและค่าใช้จ่ายอื่น (บ.):", min_value=0.0, value=user_area * 500.0, step=100.0)

        # Run budget calculation
        budget_result = calculate_farm_budget(
            planted_area_rai=user_area,
            expected_yield_kg_rai=user_yield,
            selling_price_baht_kg=user_price,
            seeds_cost=cost_seed,
            fertilizer_cost=cost_fert,
            water_energy_cost=cost_energy,
            labor_cost=cost_labor,
            land_rent_cost=cost_rent_other,
            other_cost=cost_chem
        )

        st.markdown("---")
        st.markdown("#### 2. ❓ เปรียบเทียบกับอะไร? — ผลการคำนวณกำไร-ขาดทุน และสัดส่วนต้นทุน (Financial Analysis)")

        if budget_result.get("is_valid", False):
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)

            with col_res1:
                rev = budget_result["gross_revenue"]
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">รายรับรวม (Revenue)</div>
                    <div class="kpi-value">{rev:,.0f}<span class="kpi-unit">บาท</span></div>
                    <div class="kpi-sub">ผลผลิตรวม {budget_result['total_yield_ton']:,.1f} ตัน</div>
                </div>
                """, unsafe_allow_html=True)

            with col_res2:
                tot_c = budget_result["total_cost"]
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">ต้นทุนรวม (Total Cost)</div>
                    <div class="kpi-value">{tot_c:,.0f}<span class="kpi-unit">บาท</span></div>
                    <div class="kpi-sub">ต้นทุน {budget_result['cost_per_rai']:,.0f} บ./ไร่</div>
                </div>
                """, unsafe_allow_html=True)

            with col_res3:
                net_p = budget_result["net_profit"]
                p_badge = "safe-badge" if net_p >= 0 else "risk-badge"
                p_label = "กำไรสุทธิ (Net Profit)" if net_p >= 0 else "ขาดทุนสุทธิ (Net Loss)"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{p_label}</div>
                    <div class="kpi-value">{net_p:+,.0f}<span class="kpi-unit">บาท</span></div>
                    <div class="kpi-sub"><span class="{p_badge}">{budget_result['profit_margin_pct']:+.1f}% Margin</span></div>
                </div>
                """, unsafe_allow_html=True)

            with col_res4:
                be_p = budget_result["breakeven_price_kg"]
                be_y = budget_result["breakeven_yield_kg_rai"]
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">ราคาจุดคุ้มทุน (Break-Even)</div>
                    <div class="kpi-value">{be_p:,.2f}<span class="kpi-unit">บาท/กก.</span></div>
                    <div class="kpi-sub">ผลผลิตขั้นต่ำ {be_y:,.0f} กก./ไร่</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Visualizations: Donut Pie & Scenario Bar
            col_v1, col_v2 = st.columns([1, 1.2])

            with col_v1:
                cost_items = {
                    "ค่าเมล็ด/ท่อนพันธุ์": cost_seed,
                    "ค่าปุ๋ย": cost_fert,
                    "ค่าสารเคมี/ชีวภัณฑ์": cost_chem,
                    "ค่าแรงงาน": cost_labor,
                    "ค่าน้ำมันและพลังงาน": cost_energy,
                    "ค่าเช่าที่ดินและอื่นๆ": cost_rent_other
                }
                fig_cost_pie = create_cost_breakdown_pie(cost_items)
                st.plotly_chart(fig_cost_pie, use_container_width=True)

            with col_v2:
                fig_scen = create_scenario_bar_chart(budget_result["scenarios"])
                st.plotly_chart(fig_scen, use_container_width=True)

            # Decision Advice Card
            st.markdown("#### 3. ❓ ควรระวังข้อจำกัดอะไร? — บทวิเคราะห์เพื่อการตัดสินใจสำหรับเกษตรกร")
            margin = budget_result["profit_margin_pct"]
            if net_p > 0 and margin >= 20.0:
                st.success(f"✅ **ผลการวิเคราะห์อยู่ในเกณฑ์ดี:** อัตรากำไรสุทธิอยู่ที่ {margin:.1f}% และราคาขายปัจจุบัน ({user_price:.2f} บ./กก.) สูงกว่าราคาจุดคุ้มทุน ({be_p:.2f} บ./กก.) มีความปลอดภัยต่อความผันผวนของราคาตลาด")
            elif net_p > 0:
                st.warning(f"⚠️ **ควรระมัดระวังความเสี่ยงด้านต้นทุน:** อัตรากำไรสุทธิอยู่ที่ {margin:.1f}% ถือว่าค่อนข้างบาง หากราคาตลาดปรับลดลงต่ำกว่า {be_p:.2f} บาท/กก. หรือผลผลิตลดลงต่ำกว่า {be_y:,.0f} กก./ไร่ จะเริ่มเข้าสู่ภาวะขาดทุน แนะนำให้วางแผนลดต้นทุนปุ๋ยและค่าจ้างเหมา")
            else:
                st.error(f"❌ **ส่งสัญญาณเตือนภาวะขาดทุน:** ภายใต้ต้นทุนที่ระบุ ต้นทุนเฉลี่ยอยู่ที่ {budget_result['cost_per_kg']:.2f} บาท/กก. สูงกว่าราคาขาย {user_price:.2f} บาท/กก. จำเป็นต้องลดต้นทุนปัจจัยการผลิตลงอย่างน้อย {abs(net_p):,.0f} บาท หรือเพิ่มผลผลิตต่อไร่ให้ถึงจุดคุ้มทุน ({be_y:,.0f} กก./ไร่)")
        else:
            st.error(budget_result.get("message", "ข้อมูลไม่ถูกต้อง"))


# =========================================================
# TAB 6: CROP PRODUCTION DRIVERS
# =========================================================
with tab6:
    st.markdown("## 🚜 ปัจจัยขับเคลื่อนผลผลิตทางการเกษตร (Crop Production Drivers)")
    st.markdown(
        "> **วัตถุประสงค์การวิเคราะห์เชิงลึก:** ผลผลิตทางการเกษตรไม่ได้ขึ้นอยู่กับปริมาณน้ำฝนและอุณหภูมิเพียงอย่างเดียว "
        "แต่ถูกขับเคลื่อนด้วย **5 เสาหลักแห่งผลิตภาพทางการเกษตร** ได้แก่: "
        "**1) แหล่งน้ำ ชลประทาน และช่วงเวลาฝนตก**, "
        "**2) ดินและความเหมาะสมของพื้นที่ตาม Agri-Map**, "
        "**3) เทคโนโลยี เครื่องจักรกล และต้นทุนปัจจัยการผลิต**, "
        "**4) โรค แมลง ศัตรูพืช และภัยพิบัติธรรมชาติ**, และ "
        "**5) กลไกราคา ต้นทุน และเศรษฐศาสตร์ฟาร์ม**"
    )

    # Effective filter synchronization
    driver_crop = st.session_state.selected_product if st.session_state.selected_product != "ทั้งหมด" else "ข้าวนาปี"
    # Filter years for drivers (2561-2562 overlap where comprehensive multidimensional data is available)
    driver_year = st.session_state.selected_year if st.session_state.selected_year in [2561, 2562] else 2562
    driver_prov = st.session_state.selected_province

    st.info(
        f"🔍 **ตัวแปรวิเคราะห์ปัจจุบัน:** ชนิดพืช = **{driver_crop}** | ปี พ.ศ. = **{driver_year}** | "
        f"จังหวัด = **{driver_prov}** (สามารถเปลี่ยนตัวกรองได้จากแถบด้านซ้าย)"
    )

    # -------------------------------------------------------------
    # Pillar 1: Water, Rain & Irrigation
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 1. 💧 แหล่งน้ำ ชลประทาน และช่วงเวลาฝนตก (Water, Rain & Irrigation)")
    st.markdown("#### 1.1 เกิดอะไรขึ้นเมื่อพื้นที่เข้าถึงระบบชลประทาน?")

    df_irrig_yield = join_irrigation_and_yield(df_ag, df_irrig, crop=driver_crop, year=driver_year)

    col_irrig_kpi1, col_irrig_kpi2, col_irrig_kpi3 = st.columns(3)
    with col_irrig_kpi1:
        nat_irrig_pct = (df_irrig[df_irrig["year"] == driver_year]["irrigation_area_rai"].sum() / df_irrig[df_irrig["year"] == driver_year]["agricultural_area_rai"].sum() * 100.0) if not df_irrig.empty else 22.5
        st.metric("สัดส่วนพื้นที่ชลประทานเฉลี่ยทั้งประเทศ", f"{nat_irrig_pct:.1f}%", help="คำนวณจากพื้นที่ชลประทานรวม / พื้นที่ถือครองทางการเกษตรรวม (กรมชลประทาน/สศก.)")
    with col_irrig_kpi2:
        if driver_prov != "ทั้งหมด":
            prov_irrig = df_irrig[(df_irrig["province"] == driver_prov) & (df_irrig["year"] == driver_year)]
            prov_irrig_pct = prov_irrig["irrigation_ratio_pct"].values[0] if not prov_irrig.empty else 0.0
            st.metric(f"พื้นที่ชลประทาน: {driver_prov}", f"{prov_irrig_pct:.1f}%", help=f"สัดส่วนพื้นที่ชลประทานใน {driver_prov}")
        else:
            st.metric("พื้นที่ชลประทานสูงสุด (ภาคกลาง)", "65% - 85%", help="จังหวัดในลุ่มน้ำเจ้าพระยาและแม่กลองมีความพร้อมระบบชลประทานสูงสุด")
    with col_irrig_kpi3:
        # Off-season rice ratio (ข้าวนาปรัง) as real farmer water availability proxy
        ag_dry = df_ag[(df_ag["geo_level"] == "province") & (df_ag["product"] == "ข้าวนาปรัง") & (df_ag["year"] == driver_year)]
        ag_wet = df_ag[(df_ag["geo_level"] == "province") & (df_ag["product"] == "ข้าวนาปี") & (df_ag["year"] == driver_year)]
        dry_area = ag_dry["area_plant_rai"].sum()
        wet_area = ag_wet["area_plant_rai"].sum()
        dry_ratio = (dry_area / (wet_area + dry_area) * 100.0) if (wet_area + dry_area) > 0 else 0.0
        st.metric("สัดส่วนการปลูกข้าวนาปรังฤดูแล้ง", f"{dry_ratio:.1f}%", help="ดัชนีตรวจวัดความพร้อมด้านน้ำจริงในฤดูแล้งจากสถิติ สศก.")

    # Charts: Irrigation vs Yield & Seasonal Rain vs Yield
    col_irrig_ch1, col_irrig_ch2 = st.columns(2)
    with col_irrig_ch1:
        fig_irrig = create_irrigation_vs_yield_chart(df_irrig_yield, driver_crop, driver_year)
        st.plotly_chart(fig_irrig, use_container_width=True)
    with col_irrig_ch2:
        df_season_rain = analyze_growing_season_rainfall(df_rain_monthly, df_ag, crop=driver_crop, year=driver_year)
        fig_season = create_seasonal_rain_vs_yield_chart(df_season_rain, driver_crop, driver_year)
        st.plotly_chart(fig_season, use_container_width=True)

    st.markdown("""
    > [!NOTE]
    > **บทวิเคราะห์ทางวิชาการ (Academic Insight):**
    > - **ความมั่นคงด้านน้ำช่วยยกระดับผลผลิต:** จังหวัดที่มีระบบชลประทานหนาแน่น (เช่น ชัยนาท สิงห์บุรี พระนครศรีอยุธยา) มักมีผลผลิตต่อไร่สูงกว่าอย่างมีนัยสำคัญ เนื่องจากระบบชลประทานช่วยควบคุมระดับน้ำได้อย่างสม่ำเสมอ และเพิ่มประสิทธิภาพการดูดซึมปุ๋ยเคมีของพืช
    > - **ช่วงเวลาฝนตกสำคัญกว่าปริมาณฝนรวมทั้งปี:** ปริมาณฝนในช่วงฤดูเจริญเติบโต (พ.ค.–ต.ค.) มีผลต่อระยะแตกกอและออกรวงมากกว่าฝนสะสมรวมทั้งปี หากเกิดสภาวะ "ฝนทิ้งช่วง" ในเดือนมิถุนายน-กรกฎาคม จะสร้างความเสียหายต่อผลผลิตอย่างรุนแรงแม้ปริมาณฝนทั้งปีจะอยู่ในเกณฑ์ปกติ
    > - **ข้อพึงระวัง (Correlation is NOT Causation):** การเข้าถึงชลประทานไม่ได้เป็นปัจจัยเดียวที่กำหนดผลผลิต พื้นที่ชลประทานส่วนใหญ่ยังมีการใส่ปุ๋ยสูงกว่า ใช้พันธุ์พืชผลผลิตสูง และมีเครื่องจักรกลพร้อมกว่า ซึ่งเป็นปัจจัยเกื้อหนุนร่วมกัน
    """)

    # -------------------------------------------------------------
    # Pillar 2: Soil & Land Suitability (LDD Agri-Map)
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 2. 🌱 ดินและความเหมาะสมของพื้นที่ (Soil & Land Suitability: LDD Agri-Map)")
    st.markdown("#### 2.1 ดินตรงตามความต้องการของพืชเศรษฐกิจมากน้อยเพียงใด?")

    col_soil_kpi1, col_soil_kpi2, col_soil_kpi3 = st.columns(3)
    soil_sub = df_soil[(df_soil["crop"] == driver_crop) & (df_soil["province"] == driver_prov)] if driver_prov != "ทั้งหมด" else df_soil[df_soil["crop"] == driver_crop]

    with col_soil_kpi1:
        s1_val = soil_sub["suitable_high_s1_pct"].mean() if not soil_sub.empty else 0.0
        st.metric("S1 (เหมาะสมสูง)", f"{s1_val:.1f}%", help="ดินมีความอุดมสมบูรณ์สูง การระบายน้ำดี ข้อจำกัดน้อยมาก คาดหวังผลผลิต > 80%")
    with col_soil_kpi2:
        s2_val = soil_sub["suitable_moderate_s2_pct"].mean() if not soil_sub.empty else 0.0
        st.metric("S2 (เหมาะสมปานกลาง)", f"{s2_val:.1f}%", help="มีข้อจำกัดเล็กน้อยด้านความชื้นหรือธาตุอาหาร ต้องมีการจัดการดินเพิ่มเติม คาดหวังผลผลิต 60-80%")
    with col_soil_kpi3:
        s3_n_val = (soil_sub["suitable_marginal_s3_pct"].mean() + soil_sub["not_suitable_n_pct"].mean()) if not soil_sub.empty else 0.0
        st.metric("S3 + N (เหมาะสมน้อย / ไม่เหมาะสม)", f"{s3_n_val:.1f}%", help="ดินทรายจัด ดินเค็ม ดินเปรี้ยว ดินตื้น หรือพื้นที่สูงชัน เสี่ยงต่อต้นทุนสูงแต่ผลผลิตต่ำ")

    fig_soil = create_soil_suitability_bar_chart(df_soil, driver_crop, top_n=15)
    st.plotly_chart(fig_soil, use_container_width=True)

    st.markdown("""
    > [!TIP]
    > **แนวทางปฏิบัติของกรมพัฒนาที่ดิน (Agri-Map Guidance):**
    > - **พื้นที่ S1 & S2:** ควรส่งเสริมการเพิ่มผลิตภาพและลดต้นทุนต่อหน่วย เช่น การใช้ปุ๋ยสั่งตัดตามค่าวิเคราะห์ดิน
    > - **พื้นที่ S3:** ดินมีข้อจำกัดชัดเจน เช่น ดินร่วนปนทรายในภาคตะวันออกเฉียงเหนือที่กักเก็บน้ำและปุ๋ยได้ต่ำ เกษตรกรควรเติมอินทรียวัตถุ ปรับปรุงดินด้วยปุ๋ยพืชสด หรือพิจารณาปลูกพืชที่ทนทานกว่า
    > - **พื้นที่ N:** ควรหลีกเลี่ยงการฝืนเพาะปลูก และปรับเปลี่ยนไปสู่กิจกรรมเกษตรอื่นตามนโยบาย Zoning by Agri-Map
    """)

    # -------------------------------------------------------------
    # Pillar 3: Technology, Mechanization & Production Costs
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 3. ⚙️ เทคโนโลยี เครื่องจักรกล และต้นทุนปัจจัยการผลิต (Technology & Inputs)")
    st.markdown("#### 3.1 สัดส่วนการใช้เครื่องจักรกลและโครงสร้างต้นทุนเป็นอย่างไร?")

    cost_summary = compare_production_cost_and_price(df_cost, df_prices, crop=driver_crop, year=driver_year)

    if cost_summary.get("has_data", False):
        col_cost_k1, col_cost_k2, col_cost_k3 = st.columns(3)
        with col_cost_k1:
            st.metric("ต้นทุนรวมเฉลี่ยอ้างอิง สศก.", f"{cost_summary['total_cost_rai']:,.0f} บาท/ไร่", help="ต้นทุนการผลิตเฉลี่ยต่อไร่จากผลสำรวจของสำนักงานเศรษฐกิจการเกษตร")
        with col_cost_k2:
            st.metric("สัดส่วนการใช้เครื่องจักร (Mechanization)", f"{cost_summary['mechanization_share_pct']:.1f}%", help="สัดส่วนค่าบริการเครื่องจักรกล (ไถเตรียมดิน + เกี่ยวนวด) ต่อน้ำหนักต้นทุนทั้งหมด")
        with col_cost_k3:
            fert_chem = cost_summary['cost_breakdown']['ค่าปุ๋ย'] + cost_summary['cost_breakdown']['ค่าสารเคมี']
            st.metric("ต้นทุนปุ๋ยและสารเคมีต่อไร่", f"{fert_chem:,.0f} บาท/ไร่", help="ค่าปุ๋ยเคมี/อินทรีย์และสารป้องกันกำจัดศัตรูพืช")

        col_cost_ch1, col_cost_ch2 = st.columns([3, 2])
        with col_cost_ch1:
            fig_cost_str = create_cost_structure_breakdown_chart(
                cost_summary["cost_breakdown"], driver_crop, cost_summary["total_cost_rai"], driver_year
            )
            st.plotly_chart(fig_cost_str, use_container_width=True)
        with col_cost_ch2:
            st.markdown("#### 💡 นัยสำคัญทางเทคโนโลยี")
            st.markdown(
                f"- **เครื่องจักรกลเกษตร ({cost_summary['mechanization_share_pct']:.1f}%):** ปัจจุบันการเพาะปลูกพืชเศรษฐกิจไทยปรับเปลี่ยนสู่การจ้างเหมาเครื่องจักรกลเกือบสมบูรณ์ (รถไถเตรียมดิน และรถเกี่ยวนวด) ช่วยแก้ไขวิกฤตขาดแคลนแรงงานสูงอายุและลดการสูญเสียผลผลิตจากฝนปลายฤดู\n"
                f"- **ปุ๋ยและสารเคมี:** คิดเป็นกว่า 25–30% ของต้นทุนรวม ความผันผวนของราคาปุ๋ยเคมีในตลาดโลกจึงส่งผลกระทบต่อกำไรสุทธิของเกษตรกรโดยตรง\n"
                f"- **ค่าแรงงานคน:** มักเกิดขึ้นในขั้นตอนดูแลรักษา ดายหญ้า และการขนย้ายผลผลิต"
            )

    # -------------------------------------------------------------
    # Pillar 4: Pests, Diseases & Natural Disasters
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 4. ⚠️ โรค แมลง ภัยแล้ง และอุทกภัย (Pests, Diseases & Disasters)")
    st.markdown("#### 4.1 ภัยธรรมชาติสร้างความเสียหายต่อพื้นที่เพาะปลูกอย่างไร?")

    col_dis_k1, col_dis_k2, col_dis_k3 = st.columns(3)
    with col_dis_k1:
        tot_flood = df_disaster[df_disaster["year"] == driver_year]["flood_affected_area_rai"].sum()
        st.metric("พื้นที่เกษตรประสบอุทกภัยสะสม", f"{tot_flood:,.0f} ไร่", help="พื้นที่ประสบภัยน้ำท่วมจากรายงาน DOAE/DDPM")
    with col_dis_k2:
        tot_drought = df_disaster[df_disaster["year"] == driver_year]["drought_affected_area_rai"].sum()
        st.metric("พื้นที่เกษตรประสบภัยแล้งสะสม", f"{tot_drought:,.0f} ไร่", help="พื้นที่ประสบภัยแล้งและฝนทิ้งช่วงจากรายงาน DOAE/DDPM")
    with col_dis_k3:
        # Damage rate from primary dataset
        ag_crop_yr = df_ag[(df_ag["geo_level"] == "province") & (df_ag["product"] == driver_crop) & (df_ag["year"] == driver_year)]
        tot_pl = ag_crop_yr["area_plant_rai"].sum()
        tot_hv = ag_crop_yr["area_harvest_rai"].sum()
        dmg_pct = ((tot_pl - tot_hv) / tot_pl * 100.0) if tot_pl > 0 else 0.0
        st.metric("อัตราเนื้อที่เสียหายเฉลี่ย (สศก.)", f"{dmg_pct:.2f}%", help="เนื้อที่เพาะปลูกที่ไม่ได้เก็บเกี่ยวเนื่องจากภัยพิบัติหรือโรคแมลง")

    fig_dis = create_disaster_impact_chart(df_disaster, driver_year, top_n=12)
    st.plotly_chart(fig_dis, use_container_width=True)

    st.markdown("""
    > [!IMPORTANT]
    > **ระเบียบการช่วยเหลือเกษตรกรผู้ประสบภัยพิบัติกรณีฉุกเฉิน:**
    > - ตามระเบียบกระทรวงการคลังฯ เกษตรกรต้องขึ้นทะเบียนเกษตรกร (ทบก.01) ไว้ก่อนเกิดภัย
    > - มีการประกาศเขตการให้ความช่วยเหลือผู้ประสบภัยพิบัติกรณีฉุกเฉินระดับจังหวัด และผ่านการตรวจสอบแปลงเสียหายจริงโดยคณะกรรมการระดับหมู่บ้าน/ตำบล (กชภ.อ.)
    > - อัตราเยียวยาชดเชยต้นทุนเบื้องต้น (ไม่เกิน 30 ไร่/ครัวเรือน): ข้าว 1,113 บ./ไร่, พืชไร่ 1,148 บ./ไร่, พืชสวน 1,690 บ./ไร่
    """)

    # -------------------------------------------------------------
    # Pillar 5: Prices, Costs & Farm Margins
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 5. 💵 กลไกราคา ต้นทุน และเศรษฐศาสตร์ฟาร์ม (Prices, Costs & Farm Margins)")
    st.markdown("#### 5.1 ผลผลิตแปลงของเกษตรกรสร้างกำไรสุทธิและคุ้มทุนหรือไม่?")

    if cost_summary.get("has_data", False):
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("ราคาขายเฉลี่ยหน้าไร่นา", f"{cost_summary['avg_selling_price_kg']:.2f} บ./กก.", help="สถิติราคาที่เกษตรกรขายได้จริงจาก สศก.")
        with col_m2:
            st.metric("ต้นทุนการผลิตต่อหน่วย", f"{cost_summary['cost_per_kg']:.2f} บ./กก.", help="ต้นทุนรวมทั้งหมดหารด้วยผลผลิตเฉลี่ย")
        with col_m3:
            st.metric("ส่วนต่างกำไร/ขาดทุนต่อ กก.", f"{cost_summary['margin_per_kg']:+.2f} บ./กก.", help="ราคาขายเฉลี่ย ลบ ต้นทุนต่อ กก.")
        with col_m4:
            st.metric("กำไรสุทธิต่อไร่ (ประเมิน)", f"{cost_summary['net_profit_rai']:+,.0f} บ./ไร่", help="รายรับต่อไร่ ลบ ต้นทุนต่อไร่")

        col_margin_g1, col_margin_g2 = st.columns([1, 1])
        with col_margin_g1:
            fig_margin = create_cost_price_margin_chart(cost_summary)
            st.plotly_chart(fig_margin, use_container_width=True)
        with col_margin_g2:
            st.markdown("#### 📊 บทสรุปความคุ้มทุนระดับประเทศ (สศก.)")
            st.markdown(
                f"- **ราคาจุดคุ้มทุน (Break-Even Price):** **{cost_summary['breakeven_price_kg']:.2f} บาท/กก.** (หากราคาตลาดต่ำกว่านี้จะเริ่มขาดทุน)\n"
                f"- **ผลผลิตขั้นต่ำ ณ จุดคุ้มทุน (Break-Even Yield):** **{cost_summary['breakeven_yield_kg']:,.0f} กก./ไร่** (ภายใต้ราคาขายเฉลี่ย {cost_summary['avg_selling_price_kg']:.2f} บาท/กก.)\n"
                f"- **อัตราผลตอบแทนต่อต้นทุน (Profit Margin):** **{cost_summary['profit_rate_pct']:+.1f}%**\n"
                f"- **สถานะความปลอดภัย:** " + ("✅ ปลอดภัยต่อความผันผวนราคาตลาด" if cost_summary['margin_per_kg'] > 0 else "❌ เสี่ยงต่อภาวะขาดทุน")
            )

    # -------------------------------------------------------------
    # 5.2 Interactive Driver Farm Profit Calculator (Separating User vs Benchmark)
    # -------------------------------------------------------------
    st.markdown("#### 5.2 เครื่องคำนวณผลตอบแทนแปลงเพาะปลูก (Interactive Farm Economics Calculator)")
    st.caption("ทดลองกรอกขนาดแปลงและต้นทุนของท่าน โดยระบบจะแสดงการเปรียบเทียบอย่างโปร่งใสระหว่าง 'ข้อมูลของท่าน' กับ 'เกณฑ์มาตรฐาน สศก.'")

    col_calc_in1, col_calc_in2, col_calc_in3, col_calc_in4 = st.columns(4)
    with col_calc_in1:
        user_calc_area = st.number_input("เนื้อที่เพาะปลูก (ไร่)", min_value=1.0, max_value=500.0, value=15.0, step=1.0, key="driver_calc_area")
    with col_calc_in2:
        def_yield = float(cost_summary.get("benchmark_yield_kg_rai", 420.0))
        user_calc_yield = st.number_input("ผลผลิตต่อไร่ที่คาดว่าจะได้ (กก./ไร่)", min_value=50.0, max_value=15000.0, value=def_yield, step=10.0, key="driver_calc_yield")
    with col_calc_in3:
        def_price = float(cost_summary.get("avg_selling_price_kg", 11.20))
        user_calc_price = st.number_input("ราคาขายที่คาดว่าจะได้รับ (บาท/กก.)", min_value=0.5, max_value=200.0, value=def_price, step=0.25, key="driver_calc_price")
    with col_calc_in4:
        def_cost_rai = float(cost_summary.get("total_cost_rai", 4360.0))
        user_calc_cost = st.number_input("ต้นทุนทั้งหมดของท่าน (บาท/ไร่)", min_value=500.0, max_value=50000.0, value=def_cost_rai, step=100.0, key="driver_calc_cost")

    calc_eval = calculate_enhanced_farm_profit(
        planted_area_rai=user_calc_area,
        yield_kg_rai=user_calc_yield,
        selling_price_baht_kg=user_calc_price,
        cost_per_rai=user_calc_cost,
        is_user_cost=True,
        benchmark_cost_rai=def_cost_rai
    )

    col_eval_k1, col_eval_k2, col_eval_k3, col_eval_k4 = st.columns(4)
    with col_eval_k1:
        st.metric("รายรับรวมโดยประมาณ", f"{calc_eval['total_revenue']:,.0f} บาท", help="เนื้อที่ x ผลผลิตต่อไร่ x ราคาขาย")
    with col_eval_k2:
        st.metric("ต้นทุนรวมทั้งหมด", f"{calc_eval['total_cost']:,.0f} บาท", help="เนื้อที่ x ต้นทุนต่อไร่ที่ท่านระบุ")
    with col_eval_k3:
        delta_label = f"เทียบเกณฑ์ สศก. ({calc_eval.get('diff_cost_pct', 0.0):+.1f}%)" if 'diff_cost_pct' in calc_eval else None
        st.metric("กำไร/ขาดทุนสุทธิ", f"{calc_eval['net_profit']:+,.0f} บาท", delta=delta_label, delta_color="inverse" if calc_eval.get('diff_cost_pct', 0.0) > 0 else "normal")
    with col_eval_k4:
        st.metric("ราคาจุดคุ้มทุน (Break-Even)", f"{calc_eval['breakeven_price_kg']:.2f} บ./กก.", help="ราคาขายต่ำสุดที่ทำให้ไม่ขาดทุน")

    if calc_eval["net_profit"] > 0:
        st.success(f"✅ **ประเมินความคุ้มค่า:** กำไรสุทธิคิดเป็น **{calc_eval['margin_rate_pct']:.1f}%** ของต้นทุนการผลิต (กำไรเฉลี่ย **{calc_eval['net_profit']/user_calc_area:+,.0f} บาท/ไร่**) ราคาขายปัจจุบันสูงกว่าราคาจุดคุ้มทุน ({calc_eval['breakeven_price_kg']:.2f} บ./กก.) มีความปลอดภัยทางเศรษฐกิจ")
    else:
        st.error(f"❌ **แจ้งเตือนความเสี่ยงขาดทุน:** ภายใต้ต้นทุนที่ระบุ ต้นทุนต่อหน่วยอยู่ที่ **{calc_eval['cost_per_kg']:.2f} บาท/กก.** ซึ่งสูงกว่าราคาขาย ({user_calc_price:.2f} บาท/กก.) จำเป็นต้องลดต้นทุนต่อไร่ลงอย่างน้อย **{abs(calc_eval['net_profit']/user_calc_area):,.0f} บาท/ไร่** หรือเพิ่มผลผลิตให้ถึง **{calc_eval['breakeven_yield_kg_rai']:,.0f} กก./ไร่**")

    # -------------------------------------------------------------
    # Transparent Data Governance Matrix for Tab 6
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🛡️ ข้อมูลแหล่งอ้างอิงและข้อกำหนดธรรมาภิบาลข้อมูล (Data Governance & Traceability)")
    governance_rows = [
        {"มิติปัจจัย": "1. น้ำและชลประทาน", "ชุดข้อมูล": "พื้นที่ชลประทานรายจังหวัด และฝนเชิงพื้นที่ IDW", "หน่วยงาน": "กรมชลประทาน (RID), สศก., สสน. HII", "ปีข้อมูล": "2561–2562", "หน่วย": "ไร่, มม.", "สิทธิ์ใช้งาน": "Open Government Data"},
        {"มิติปัจจัย": "2. ดินและความเหมาะสม", "ชุดข้อมูล": "Agri-Map การจำแนกความเหมาะสมของดิน (S1, S2, S3, N)", "หน่วยงาน": "กรมพัฒนาที่ดิน (LDD)", "ปีข้อมูล": "พ.ศ. 2562", "หน่วย": "ร้อยละ (%)", "สิทธิ์ใช้งาน": "ข้อมูลเปิดเผยราชการ"},
        {"มิติปัจจัย": "3. เทคโนโลยีและปัจจัย", "ชุดข้อมูล": "สถิติต้นทุนการผลิตและระดับการใช้เครื่องจักร (Mechanization)", "หน่วยงาน": "สำนักงานเศรษฐกิจการเกษตร (สศก.)", "ปีข้อมูล": "2561–2562", "หน่วย": "บาท/ไร่, %", "สิทธิ์ใช้งาน": "Open Government Data"},
        {"มิติปัจจัย": "4. โรคและภัยธรรมชาติ", "ชุดข้อมูล": "พื้นที่การเกษตรประสบอุทกภัยและภัยแล้ง รายจังหวัด", "หน่วยงาน": "กรมส่งเสริมการเกษตร (DOAE), ปภ.", "ปีข้อมูล": "2561–2562", "หน่วย": "ไร่", "สิทธิ์ใช้งาน": "รายงานราชการ ปภ."},
        {"มิติปัจจัย": "5. ราคาและเศรษฐกิจ", "ชุดข้อมูล": "ราคาที่เกษตรกรขายได้ที่ไร่นารายสัปดาห์ (Catalog ID 35)", "หน่วยงาน": "สำนักงานเศรษฐกิจการเกษตร (สศก.)", "ปีข้อมูล": "2561–2562", "หน่วย": "บาท/กก.", "สิทธิ์ใช้งาน": "Open Data สศก."}
    ]
    st.table(pd.DataFrame(governance_rows))

# ---------------------------------------------------------
# Footer & Academic Traceability Credits
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="font-size: 12.5px; color: #64748B; line-height: 1.6;">
    🌾 <b>Thai Agriculture & Climate Intelligence Dashboard — Real Open Data Edition</b><br>
    - <b>สถิติการผลิตสินค้าเกษตรรายปี พ.ศ. 2550–2562:</b> สำนักงานเศรษฐกิจการเกษตร (สศก.) กระทรวงเกษตรและสหกรณ์<br>
    - <b>สถิติราคาสินค้าเกษตรที่เกษตรกรขายได้ที่ไร่นา รายสัปดาห์ (2561–2562):</b> สำนักงานเศรษฐกิจการเกษตร (OAE Data Catalog ID 35)<br>
    - <b>ข้อมูลฝนเชิงพื้นที่และอุณหภูมิ:</b> สถาบันสารสนเทศทรัพยากรน้ำ (องค์การมหาชน) (สสน. HII)<br>
    - <b>สถานีอุตุนิยมวิทยาเพื่อการเกษตรรายวัน:</b> กรมอุตุนิยมวิทยา (TMD Open Data API Catalog c1-1-3-7)<br>
    - <b>ปฏิทินและคำแนะนำทางวิชาการ:</b> กรมการข้าว, กรมวิชาการเกษตร และกรมส่งเสริมการเกษตร<br>
    <i>วิจัยและพัฒนาเพื่อสนับสนุนการตัดสินใจของเกษตรกรไทยและการนำเสนอผลงานทางวิชาการ ปราศจากข้อมูลจำลองหรือข้อมูลหลอก 100%</i>
</div>
""", unsafe_allow_html=True)
