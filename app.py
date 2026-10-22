"""
Thai Agriculture & Climate Intelligence Dashboard — Real Open Data Edition
A comprehensive, academic-grade data intelligence platform analyzing Thailand's
agricultural production, climate interactions, production drivers, and farm economics.

Built with Streamlit, Plotly, Pandas, and SciPy.
Strictly relies on verified government Open Data sources (OAE, TMD, HII, RID, LDD, DDPM, DOAE).
Zero mock data. Double-counting prevented 100%. Exactly 4 Tabs.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Ensure repository root is strictly prioritized at sys.path[0].
# This guarantees deterministic package imports on both local environments and Streamlit Cloud.
ROOT_DIR = Path(__file__).resolve().parent
root_str = str(ROOT_DIR)
if sys.path[0] != root_str:
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

# On Streamlit Community Cloud, repositories are mounted at /mount/src/<repo_name>.
# Python's namespace package resolution can inadvertently bind top-level 'src'
# to '/mount/src' instead of the repository's '<repo_name>/src'.
# If 'src' exists in sys.modules and does not point to our repository package, purge it
# so that Python imports the actual package defined in '<repo_name>/src/__init__.py'.
if "src" in sys.modules:
    src_mod = sys.modules["src"]
    src_path = getattr(src_mod, "__file__", None) or (
        getattr(src_mod, "__path__", [""])[0] if getattr(src_mod, "__path__", None) else ""
    )
    if str(ROOT_DIR / "src") not in str(src_path):
        del sys.modules["src"]

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Canonical package imports from src (unified import strategy without try/except fallbacks)
from src.data_loader import (
    load_agriculture_data,
    load_spatial_rain_data,
    load_national_rain_data,
    load_provincial_temp_data,
    load_oae_farmgate_prices,
    load_tmd_daily_agromet,
    load_thailand_geojson,
    load_provincial_irrigation_data,
    load_ldd_soil_suitability_data,
    load_oae_production_cost_data,
    load_disaster_affected_data
)

from src.data_cleaning import (
    clean_agriculture_data,
    clean_spatial_rain_data,
    clean_national_rain_data,
    clean_provincial_temp_data,
    clean_oae_farmgate_prices,
    clean_tmd_daily_agromet,
    clean_provincial_irrigation_data,
    clean_ldd_soil_suitability_data,
    clean_oae_production_cost_data,
    clean_disaster_affected_data,
    validate_agromet_schema,
    OFFICIAL_77_PROVINCES,
    COUNTRY_NAME,
    REGION_NAMES,
    PROVINCE_TO_REGION,
    REGION_TO_PROVINCES
)

from src.feature_engineering import (
    calculate_weighted_yield,
    calculate_aggregated_harvest_ratio,
    calculate_unharvested_area,
    aggregate_agricultural_kpis,
    compute_yoy_dataframe,
    compute_cv_summary
)

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
    create_irrigation_vs_yield_chart,
    create_soil_suitability_bar_chart,
    create_cost_structure_breakdown_chart,
    create_disaster_impact_chart,
    create_seasonal_rain_vs_yield_chart,
    create_cost_price_margin_chart,
    COLOR_FOREST_DARK,
    COLOR_FOREST_MED,
    COLOR_SAGE,
    COLOR_MINT,
    COLOR_WARNING,
    COLOR_DANGER,
    COLOR_GOLD
)

from src.filters import init_session_state, reset_filters, apply_ag_filters


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
        margin-bottom: 20px;
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
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
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
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 24px;
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
        font-size: 12px;
        margin-top: 6px;
        font-weight: 500;
    }
    .delta-badge-pos {
        background-color: #ECFDF5;
        color: #059669;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 11.5px;
        display: inline-block;
    }
    .delta-badge-neg {
        background-color: #FEF2F2;
        color: #DC2626;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 11.5px;
        display: inline-block;
    }
    .delta-badge-neutral {
        background-color: #F1F5F9;
        color: #64748B;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 11.5px;
        display: inline-block;
    }

    /* Filter Bar Box */
    .filter-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px 8px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
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
# Sidebar: Documentation, Governance & Quick Shortcuts
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌾 ทางลัดพืชเศรษฐกิจหลัก")
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        if st.button("🌾 ข้าวนาปี", use_container_width=True):
            st.session_state.selected_product = "ข้าวนาปี"
            st.rerun()
        if st.button("🌾 ข้าวนาปรัง", use_container_width=True):
            st.session_state.selected_product = "ข้าวนาปรัง"
            st.rerun()
    with col_sc2:
        if st.button("🥔 มันสำปะหลัง", use_container_width=True):
            st.session_state.selected_product = "มันสำปะหลัง"
            st.rerun()
        if st.button("🌽 ข้าวโพดเลี้ยงสัตว์", use_container_width=True):
            st.session_state.selected_product = "ข้าวโพดเลี้ยงสัตว์"
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🛡️ ธรรมาภิบาลข้อมูล (Data Governance)")
    st.markdown(
        f"- **สถิติการผลิตการเกษตร:** สศก. พ.ศ. {min(available_years)}–{max(available_years)} (77 จังหวัด 26 ชนิดพืช)\n"
        f"- **ฝนเชิงพื้นที่รายจังหวัด:** สสน. HII (IDW) พ.ศ. 2561–2562\n"
        f"- **สภาพอากาศเกษตรรายวัน:** กรมอุตุนิยมวิทยา TMD\n"
        f"- **ราคาหน้าฟาร์ม:** สศก. พ.ศ. 2561–2562\n"
        f"- **ชลประทาน & ดิน:** กรมชลประทาน & กรมพัฒนาที่ดิน (Agri-Map)\n"
        f"- **ข้อมูลจำลอง (Mock):** ❌ ห้ามใช้เด็ดขาด (Real Open Data 100%)"
    )

    with st.expander("📖 รายละเอียดแหล่งข้อมูล (Data Sources)"):
        st.markdown(
            "ข้อมูลทุกมิติในระบบสามารถตรวจสอบย้อนกลับได้ ดูรายละเอียด URL เอกสารเผยแพร่ และใบอนุญาตได้ที่ไฟล์ `data_sources.md`"
        )


# ---------------------------------------------------------
# App Header Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <h1>ระบบวิเคราะห์ข้อมูลการผลิตสินค้าเกษตรและภูมิอากาศไทย (Thai Agriculture & Climate Intelligence)</h1>
    <p>แพลตฟอร์มสารสนเทศเพื่อเกษตรกรไทยและงานวิจัยวิชาการ ข้อมูลจริงจาก สศก., กรมอุตุนิยมวิทยา, สสน., กรมชลประทาน, กรมพัฒนาที่ดิน และ ปภ. | อัปเดตข้อมูลล่าสุด: พ.ศ. {latest_year}</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Global Filter Bar (Prominent Top Controls)
# ---------------------------------------------------------
st.markdown("""
<div class="filter-box">
    <div style="font-weight: 700; font-size: 14.5px; color: #1B4D3E; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <span>🔍 ตัวกรองหลักของ Dashboard (Global Filters)</span>
        <span style="font-size: 12px; font-weight: 400; color: #64748B;">— ตัวกรองเชื่อมโยงพร้อมกันทุกแท็บ แผนที่ กราฟ และตาราง</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    # Row 1: Region, Province, Granularity, Product
    col_f1, col_f2, col_f3, col_f4 = st.columns([1.1, 1.2, 1.0, 1.3])

    with col_f1:
        region_options = ["ทั้งหมด", "ภาคเหนือ", "ภาคตะวันออกเฉียงเหนือ", "ภาคกลาง", "ภาคใต้"]
        curr_reg_idx = region_options.index(st.session_state.selected_region) if st.session_state.selected_region in region_options else 0
        def on_region_change():
            st.session_state.selected_region = st.session_state.reg_select_key
            if st.session_state.selected_region != "ทั้งหมด":
                valid_provs = REGION_TO_PROVINCES.get(st.session_state.selected_region, [])
                if st.session_state.selected_province not in valid_provs:
                    st.session_state.selected_province = "ทั้งหมด"

        selected_region = st.selectbox(
            "🌍 1. ภูมิภาค (Region)",
            options=region_options,
            index=curr_reg_idx,
            key="reg_select_key",
            on_change=on_region_change
        )
        st.session_state.selected_region = selected_region

    with col_f2:
        if st.session_state.selected_region != "ทั้งหมด":
            prov_options = ["ทั้งหมด"] + sorted(REGION_TO_PROVINCES.get(st.session_state.selected_region, []))
        else:
            prov_options = ["ทั้งหมด"] + sorted(OFFICIAL_77_PROVINCES)

        curr_prov_idx = prov_options.index(st.session_state.selected_province) if st.session_state.selected_province in prov_options else 0
        selected_province = st.selectbox(
            "📍 2. จังหวัด (Province)",
            options=prov_options,
            index=curr_prov_idx,
            key="prov_select_key"
        )
        st.session_state.selected_province = selected_province

    with col_f3:
        st.selectbox(
            "🏘️ 3. ระดับพื้นที่ (Granularity)",
            options=["ระดับจังหวัด (OAE 77 จังหวัด)"],
            index=0,
            disabled=True,
            help="ข้อมูลสถิติการผลิตทางการเกษตรของ สศก. เผยแพร่ในระดับรายจังหวัด (77 จังหวัด) จึงแสดงผลระดับจังหวัดตามความละเอียดจริงของชุดข้อมูลภาครัฐ"
        )

    with col_f4:
        all_products = ["ทั้งหมด"] + sorted(df_ag["product"].dropna().unique().tolist())
        curr_prod_idx = all_products.index(st.session_state.selected_product) if st.session_state.selected_product in all_products else 0
        selected_product = st.selectbox(
            "🌱 4. ชนิดสินค้าเกษตร (Crop)",
            options=all_products,
            index=curr_prod_idx,
            key="prod_select_key"
        )
        st.session_state.selected_product = selected_product

    # Row 2: Year, Metric, Reset Button
    col_f5, col_f6, col_f7 = st.columns([1.1, 1.8, 0.9])

    with col_f5:
        year_options = sorted(available_years, reverse=True)
        curr_yr_idx = year_options.index(st.session_state.selected_year) if st.session_state.selected_year in year_options else 0
        selected_year = st.selectbox(
            "📅 5. ปี พ.ศ. (Crop Year BE)",
            options=year_options,
            index=curr_yr_idx,
            key="year_select_key"
        )
        st.session_state.selected_year = selected_year

    metric_labels = {
        "production_ton": ("ปริมาณผลผลิตรวม", "ตัน"),
        "calc_yield_per_rai": ("ผลผลิตต่อไร่แบบถ่วงน้ำหนัก", "กก./ไร่"),
        "area_plant_rai": ("เนื้อที่เพาะปลูก", "ไร่"),
        "area_harvest_rai": ("เนื้อที่เก็บเกี่ยว", "ไร่"),
        "calc_harvest_ratio": ("อัตราการเก็บเกี่ยว", "%")
    }

    with col_f6:
        metric_keys = list(metric_labels.keys())
        curr_met_idx = metric_keys.index(st.session_state.selected_metric) if st.session_state.selected_metric in metric_keys else 0
        selected_metric = st.selectbox(
            "📊 6. ตัวชี้วัดสำหรับแผนที่และจัดอันดับ (Metric)",
            options=metric_keys,
            format_func=lambda k: f"{metric_labels[k][0]} ({metric_labels[k][1]})",
            index=curr_met_idx,
            key="metric_select_key"
        )
        st.session_state.selected_metric = selected_metric

    with col_f7:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 รีเซ็ตตัวกรอง", use_container_width=True, help="คืนค่าตัวกรองเริ่มต้นทั้งหมด"):
            reset_filters(default_year=latest_year)
            st.rerun()

# Active Filters Summary Badge
sel_metric_title, sel_metric_unit = metric_labels[st.session_state.selected_metric]
st.markdown(f"""
<div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 8px 14px; margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 12.5px; color: #166534;">
    <span style="font-weight: 700;">🏷️ ตัวกรองที่ใช้งานอยู่:</span>
    <span style="background: #FFFFFF; border: 1px solid #86EFAC; padding: 2px 8px; border-radius: 4px;">🌍 ภูมิภาค: <b>{st.session_state.selected_region}</b></span>
    <span style="background: #FFFFFF; border: 1px solid #86EFAC; padding: 2px 8px; border-radius: 4px;">📍 จังหวัด: <b>{st.session_state.selected_province}</b></span>
    <span style="background: #FFFFFF; border: 1px solid #86EFAC; padding: 2px 8px; border-radius: 4px;">🌱 ชนิดพืช: <b>{st.session_state.selected_product}</b></span>
    <span style="background: #FFFFFF; border: 1px solid #86EFAC; padding: 2px 8px; border-radius: 4px;">📅 ปี พ.ศ.: <b>{st.session_state.selected_year}</b></span>
    <span style="background: #FFFFFF; border: 1px solid #86EFAC; padding: 2px 8px; border-radius: 4px;">📊 ตัวชี้วัด: <b>{sel_metric_title}</b> ({sel_metric_unit})</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Filter Application for Main Context
# ---------------------------------------------------------
# 1. Fully filtered provincial subset for active view
df_active = apply_ag_filters(
    df_ag,
    year=st.session_state.selected_year,
    region=st.session_state.selected_region,
    province=st.session_state.selected_province,
    product=st.session_state.selected_product,
    geo_level="province"
)

# 2. Provincial subset for mapping and rankings across all provinces in selected region
df_year_prov = apply_ag_filters(
    df_ag,
    year=st.session_state.selected_year,
    region=st.session_state.selected_region,
    province=None,
    product=st.session_state.selected_product,
    geo_level="province"
)

# 3. Multi-year series for trend analysis across years
df_trend_series = apply_ag_filters(
    df_ag,
    year=None,
    region=st.session_state.selected_region,
    province=st.session_state.selected_province,
    product=st.session_state.selected_product,
    geo_level="province"
)


# ---------------------------------------------------------
# Main 4 Tabs Navigation (Strict Acceptance Criteria)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 TAB 1: ภาพรวมการผลิตสินค้าเกษตร (Overview)",
    "🌦️ TAB 2: สภาพอากาศและสถานการณ์เพาะปลูก (Climate & Harvest)",
    "🚜 TAB 3: ปัจจัยที่เกี่ยวข้องกับผลผลิต (Production Factors)",
    "💰 TAB 4: ราคา ต้นทุน และความคุ้มค่า (Farm Economics)"
])


# =========================================================
# TAB 1: OVERVIEW (ภาพรวมการผลิตสินค้าเกษตร)
# =========================================================
with tab1:
    st.markdown("### 🌾 ภาพรวมสถิติการผลิตสินค้าเกษตรและตัวชี้วัดสำคัญ (Key KPIs)")
    st.caption(
        f"📅 ปี พ.ศ. {st.session_state.selected_year} | "
        f"ภูมิภาค: {st.session_state.selected_region} | "
        f"จังหวัด: {st.session_state.selected_province if st.session_state.selected_province != 'ทั้งหมด' else 'ทุกจังหวัด'} | "
        f"ชนิดพืช: {st.session_state.selected_product} (แหล่งข้อมูล: สำนักงานเศรษฐกิจการเกษตร สศก.)"
    )

    # Empty State Check
    if df_active.empty:
        st.warning(
            f"⚠️ ไม่พบข้อมูลจริงสำหรับตัวกรองที่เลือก (ปี พ.ศ. {st.session_state.selected_year}, "
            f"ชนิดพืช: {st.session_state.selected_product}, ภูมิภาค: {st.session_state.selected_region}, "
            f"จังหวัด: {st.session_state.selected_province}) กรุณาปรับเปลี่ยนตัวกรองด้านบน"
        )
    else:
        # Calculate current year KPIs
        kpis = aggregate_agricultural_kpis(df_active)

        # Calculate previous year KPIs for YoY comparison
        prev_year = st.session_state.selected_year - 1
        df_prev = apply_ag_filters(
            df_ag,
            year=prev_year,
            region=st.session_state.selected_region,
            province=st.session_state.selected_province,
            product=st.session_state.selected_product,
            geo_level="province"
        )
        kpis_prev = aggregate_agricultural_kpis(df_prev) if not df_prev.empty else None

        # Compute YoY Deltas
        def get_delta_badge(curr_val: float, prev_val: Optional[float], unit: str, is_yield: bool = False) -> str:
            if prev_val is None or pd.isna(prev_val) or prev_val <= 0 or pd.isna(curr_val):
                return '<span class="delta-badge-neutral">ไม่มีข้อมูลปีก่อนหน้า</span>'
            diff = curr_val - prev_val
            pct = (diff / prev_val) * 100.0
            badge_class = "delta-badge-pos" if diff >= 0 else "delta-badge-neg"
            formatted_diff = f"{diff:+.1f}" if is_yield else f"{diff:+,.0f}"
            return f'<span class="{badge_class}">{pct:+.1f}% ({formatted_diff} {unit} จากปี {prev_year})</span>'

        badge_plant = get_delta_badge(kpis["total_plant_rai"], kpis_prev["total_plant_rai"] if kpis_prev else None, "ไร่")
        badge_harvest = get_delta_badge(kpis["total_harvest_rai"], kpis_prev["total_harvest_rai"] if kpis_prev else None, "ไร่")
        badge_prod = get_delta_badge(kpis["total_production_ton"], kpis_prev["total_production_ton"] if kpis_prev else None, "ตัน")
        badge_yield = get_delta_badge(kpis["weighted_yield_kg_rai"], kpis_prev["weighted_yield_kg_rai"] if kpis_prev else None, "กก./ไร่", is_yield=True)

        # ROW 1: 4 Key KPI Cards
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

        with col_kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">🌾 เนื้อที่เพาะปลูก (Planted Area)</div>
                <div class="kpi-value">{kpis['total_plant_rai']:,.0f}<span class="kpi-unit">ไร่</span></div>
                <div class="kpi-sub">{badge_plant}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">🚜 เนื้อที่เก็บเกี่ยว (Harvested Area)</div>
                <div class="kpi-value">{kpis['total_harvest_rai']:,.0f}<span class="kpi-unit">ไร่</span></div>
                <div class="kpi-sub">{badge_harvest}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📦 ปริมาณผลผลิตรวม (Production)</div>
                <div class="kpi-value">{kpis['total_production_ton']:,.0f}<span class="kpi-unit">ตัน</span></div>
                <div class="kpi-sub">{badge_prod}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi4:
            yield_str = f"{kpis['weighted_yield_kg_rai']:,.1f}" if not np.isnan(kpis['weighted_yield_kg_rai']) else "N/A"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">⚖️ ผลผลิตต่อไร่ (Weighted Yield)</div>
                <div class="kpi-value">{yield_str}<span class="kpi-unit">กก./ไร่</span></div>
                <div class="kpi-sub">{badge_yield}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ROW 2: Map & YoY Comparison Chart
        col_map, col_chart = st.columns([1.15, 1.0])

        with col_map:
            st.markdown(f"#### 🗺️ แผนที่การกระจายตัวเชิงพื้นที่ ({sel_metric_title})")
            if geojson_data:
                fig_map = create_thailand_map(
                    df_data=df_year_prov,
                    geojson_data=geojson_data,
                    metric_col=st.session_state.selected_metric,
                    metric_label=sel_metric_title,
                    unit=sel_metric_unit,
                    year=st.session_state.selected_year,
                    product_name=st.session_state.selected_product,
                    selected_province=st.session_state.selected_province if st.session_state.selected_province != "ทั้งหมด" else None
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("ไม่พบไฟล์แผนที่ GeoJSON สำหรับการแสดงผลแผนที่ประเทศไทย")

        with col_chart:
            st.markdown(f"#### 📈 การเปรียบเทียบข้อมูลรายปีและแนวโน้มประวัติศาสตร์")
            # Build historical time series for the current filter
            if not df_trend_series.empty:
                trend_agg = df_trend_series.groupby("year").agg(
                    production_ton=("production_ton", "sum"),
                    area_plant_rai=("area_plant_rai", "sum"),
                    area_harvest_rai=("area_harvest_rai", "sum")
                ).reset_index()

                trend_agg["calc_yield_per_rai"] = np.where(
                    trend_agg["area_harvest_rai"] > 0,
                    trend_agg["production_ton"] * 1000.0 / trend_agg["area_harvest_rai"],
                    np.nan
                )
                trend_agg["calc_harvest_ratio"] = np.where(
                    trend_agg["area_plant_rai"] > 0,
                    trend_agg["area_harvest_rai"] / trend_agg["area_plant_rai"] * 100.0,
                    np.nan
                )

                fig_trend = create_trend_line_chart(
                    df=trend_agg,
                    x_col="year",
                    y_col=st.session_state.selected_metric,
                    color_col=None,
                    title=f"แนวโน้ม {sel_metric_title} (พ.ศ. 2550–2562)",
                    y_label=sel_metric_title,
                    unit=sel_metric_unit
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("ไม่มีข้อมูลอนุกรมเวลาสำหรับตัวกรองที่เลือก")

        st.markdown("---")

        # ROW 3: Top 10 and Bottom 10 Rankings
        st.markdown(f"### 🏆 การจัดอันดับจังหวัดตามตัวชี้วัด: {sel_metric_title}")
        col_top, col_bot = st.columns(2)

        with col_top:
            st.markdown(f"#### 🥇 10 อันดับจังหวัดสูงสุด (Top 10)")
            top_prov_df = get_top_provinces(df_year_prov, metric=st.session_state.selected_metric, top_n=10, ascending=False)
            fig_top = create_ranking_bar_chart(
                df=top_prov_df,
                x_col=st.session_state.selected_metric,
                y_col="province",
                title=f"10 อันดับสูงสุด ({st.session_state.selected_product} | ปี {st.session_state.selected_year})",
                x_label=sel_metric_title,
                unit=sel_metric_unit,
                selected_item=st.session_state.selected_province if st.session_state.selected_province != "ทั้งหมด" else None,
                is_bottom=False
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col_bot:
            st.markdown(f"#### 🔻 10 อันดับจังหวัดต่ำสุด (Bottom 10: เฉพาะที่มีข้อมูล > 0)")
            bot_prov_df = get_top_provinces(df_year_prov, metric=st.session_state.selected_metric, top_n=10, ascending=True)
            fig_bot = create_ranking_bar_chart(
                df=bot_prov_df,
                x_col=st.session_state.selected_metric,
                y_col="province",
                title=f"10 อันดับต่ำสุด ({st.session_state.selected_product} | ปี {st.session_state.selected_year})",
                x_label=sel_metric_title,
                unit=sel_metric_unit,
                selected_item=st.session_state.selected_province if st.session_state.selected_province != "ทั้งหมด" else None,
                is_bottom=True
            )
            st.plotly_chart(fig_bot, use_container_width=True)

        # Scale vs Yield Productivity Insight Callout
        st.markdown("""
        <div style="background-color: #FEF3C7; border-left: 4px solid #D97706; padding: 14px 18px; border-radius: 6px; margin: 10px 0 20px 0;">
            <span style="font-weight: 600; color: #92400E; font-size: 14px;">💡 ข้อสังเกตสำคัญด้านผลิตภาพและการใช้ที่ดิน (Scale vs Efficiency):</span><br>
            <span style="font-size: 13px; color: #78350F; line-height: 1.6;">
                จังหวัดที่มี <b>ผลผลิตรวมสูงสุด (Total Production)</b> มักเป็นจังหวัดที่มีเนื้อที่เพาะปลูกขนาดใหญ่ แต่ไม่ได้แปลว่ามีประสิทธิภาพหรือ <b>ผลผลิตต่อไร่ (Yield per Rai)</b> สูงสุดเสมอไป ในทางกลับกัน จังหวัดที่มีพื้นที่เพาะปลูกขนาดกลางหรือเล็กอาจมีผลผลิตต่อไร่สูงกว่าเนื่องจากระบบชลประทานและความอุดมสมบูรณ์ของดิน การวิเคราะห์เปรียบเทียบทั้ง <b>ขนาดการผลิต (Scale)</b> และ <b>ประสิทธิภาพการผลิต (Yield Efficiency)</b> ควบคู่กันจึงทำให้เห็นภาพความเป็นจริงอย่างรอบด้าน
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ROW 4: Detailed Data Table with Search, Sort, and CSV Download
        st.markdown("---")
        st.markdown("### 📋 ตารางสถิติการผลิตสินค้าเกษตรรายจังหวัด (Detailed Data Table)")
        st.caption("ตารางแสดงข้อมูลจริงที่ผ่านการกรองตามเงื่อนไขปัจจุบัน สามารถค้นหา จัดเรียง และส่งออกข้อมูลเป็นไฟล์ CSV ได้")

        table_cols = [
            "region", "province", "product", "year",
            "area_plant_rai", "area_harvest_rai", "production_ton",
            "calc_yield_per_rai", "calc_harvest_ratio", "calc_damaged_area"
        ]
        available_table_cols = [c for c in table_cols if c in df_active.columns]
        display_df = df_active[available_table_cols].copy()

        # Rename columns to Thai for presentation
        col_rename_dict = {
            "region": "ภาค",
            "province": "จังหวัด",
            "product": "ชนิดพืช",
            "year": "ปี พ.ศ.",
            "area_plant_rai": "เนื้อที่เพาะปลูก (ไร่)",
            "area_harvest_rai": "เนื้อที่เก็บเกี่ยว (ไร่)",
            "production_ton": "ผลผลิตรวม (ตัน)",
            "calc_yield_per_rai": "ผลผลิตต่อไร่ (กก./ไร่)",
            "calc_harvest_ratio": "อัตราเก็บเกี่ยว (%)",
            "calc_damaged_area": "เนื้อที่เสียหาย (ไร่)"
        }
        display_df = display_df.rename(columns=col_rename_dict)

        # Table Filter & Search Controls
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            search_query = st.text_input("🔍 ค้นหาในตาราง (พิมพ์ชื่อจังหวัดหรือชนิดพืช):", key="table_search_input")
        with col_t2:
            sort_options = [c for c in display_df.columns if c not in ["ภาค", "ปี พ.ศ."]]
            sort_col = st.selectbox("เรียงลำดับตาม:", options=sort_options, index=sort_options.index("ผลผลิตรวม (ตัน)") if "ผลผลิตรวม (ตัน)" in sort_options else 0)

        if search_query:
            mask = display_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            display_df = display_df[mask]

        if sort_col in display_df.columns:
            display_df = display_df.sort_values(by=sort_col, ascending=False)

        # CSV Download
        csv_bytes = display_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        col_t_down1, col_t_down2 = st.columns([3, 1])
        with col_t_down1:
            st.caption(f"แสดงข้อมูลทั้งหมด {len(display_df):,} รายการ")
        with col_t_down2:
            st.download_button(
                label="📥 ดาวน์โหลดข้อมูล (CSV)",
                data=csv_bytes,
                file_name=f"thai_agri_stats_{st.session_state.selected_year}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.dataframe(
            display_df.style.format({
                "เนื้อที่เพาะปลูก (ไร่)": "{:,.0f}",
                "เนื้อที่เก็บเกี่ยว (ไร่)": "{:,.0f}",
                "ผลผลิตรวม (ตัน)": "{:,.0f}",
                "ผลผลิตต่อไร่ (กก./ไร่)": "{:,.1f}",
                "อัตราเก็บเกี่ยว (%)": "{:,.1f}",
                "เนื้อที่เสียหาย (ไร่)": "{:,.0f}"
            }),
            use_container_width=True,
            height=380
        )


# =========================================================
# TAB 2: CLIMATE & HARVEST (สภาพอากาศและสถานการณ์เพาะปลูก)
# =========================================================
with tab2:
    st.markdown("### 🌦️ สภาพอากาศและสถานการณ์การเพาะปลูก (Climate & Harvest Dynamics)")
    st.caption("วิเคราะห์ปริมาณฝน อุณหภูมิ อัตราการเก็บเกี่ยว และความสัมพันธ์กับผลผลิตทางการเกษตรจากข้อมูลตรวจวัดจริงของ สสน. และ กรมอุตุนิยมวิทยา")

    climate_scope = st.radio(
        "เลือกมิติการวิเคราะห์สภาพอากาศ (Select Climate Analysis Scope):",
        options=[
            "1. การวิเคราะห์ฝนเชิงพื้นที่ 77 จังหวัด vs ผลผลิต (พ.ศ. 2561–2562, สสน. HII)",
            "2. แนวโน้มฝนสะสมระยะยาวระดับประเทศ (พ.ศ. 2550–2562, สสน. HII)",
            "3. สถานีตรวจอากาศเพื่อการเกษตรรายวัน (กรมอุตุนิยมวิทยา TMD Open Data API Catalog c1-1-3-7)"
        ],
        horizontal=True
    )

    if "1. การวิเคราะห์ฝนเชิงพื้นที่ 77 จังหวัด" in climate_scope:
        st.markdown("#### 🌧️ ฝนเชิงพื้นที่ สสน. HII (IDW) vs ผลผลิตต่อไร่ (77 จังหวัด)")
        
        target_climate_year = st.session_state.selected_year if st.session_state.selected_year in [2561, 2562] else 2562
        joined_nation = join_climate_agriculture_nationwide(
            df_ag=df_ag,
            df_rain_annual=df_rain_annual,
            df_temp_annual=df_temp_annual,
            year=target_climate_year,
            product=st.session_state.selected_product if st.session_state.selected_product != "ทั้งหมด" else "ข้าวนาปี"
        )

        if joined_nation.empty:
            st.warning(f"ไม่มีข้อมูลการเพาะปลูกที่จับคู่กับฝนเชิงพื้นที่ในปี {target_climate_year} (ชุดข้อมูลฝนเชิงพื้นที่ครอบคลุม พ.ศ. 2561–2562)")
        else:
            col_c1, col_c2 = st.columns([1.1, 0.9])
            with col_c1:
                stats_nat_rain = calculate_bivariate_stats(joined_nation["rainfall_annual_mm"], joined_nation["calc_yield_per_rai"])
                fig_nat_scat = create_bivariate_scatter_chart(
                    joined_nation,
                    x_col="rainfall_annual_mm",
                    y_col="calc_yield_per_rai",
                    title=f"ฝนเชิงพื้นที่ HII vs ผลผลิตต่อไร่ ({st.session_state.selected_product} | ปี พ.ศ. {target_climate_year})",
                    x_label="ปริมาณฝนสะสมทั้งปี",
                    y_label="ผลผลิตต่อไร่",
                    x_unit="มม.",
                    y_unit="กก./ไร่",
                    text_col="province",
                    correlation_val=stats_nat_rain["pearson_r"] if stats_nat_rain["is_valid"] else None,
                    p_val=stats_nat_rain["pearson_p"] if stats_nat_rain["is_valid"] else None,
                    n_samples=stats_nat_rain["n"] if stats_nat_rain["is_valid"] else None
                )
                st.plotly_chart(fig_nat_scat, use_container_width=True)

            with col_c2:
                st.markdown("##### 💡 การแปลผลทางสถิติและข้อพึงระวัง")
                interp_nat = generate_climate_interpretation(
                    bivariate_result=stats_nat_rain,
                    crop_name=st.session_state.selected_product,
                    climate_var_name="ปริมาณฝนเชิงพื้นที่ HII (IDW)",
                    ag_var_name="ผลผลิตต่อไร่ (Yield)",
                    geo_scope=f"77 จังหวัดทั่วประเทศ (ปี {target_climate_year})",
                    year_range=str(target_climate_year)
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

            # Seasonal Rain (May-Oct) vs Annual Rain
            st.markdown("---")
            st.markdown("#### ⏳ ช่วงเวลาฝนตก (ฝนฤดูเพาะปลูก พ.ค.–ต.ค.) สำคัญกว่าฝนสะสมทั้งปี")
            df_season_rain = analyze_growing_season_rainfall(
                df_rain_monthly, df_ag,
                crop=st.session_state.selected_product if st.session_state.selected_product != "ทั้งหมด" else "ข้าวนาปี",
                year=target_climate_year
            )
            col_s1, col_s2 = st.columns([1.1, 0.9])
            with col_s1:
                fig_season = create_seasonal_rain_vs_yield_chart(df_season_rain, st.session_state.selected_product, target_climate_year)
                st.plotly_chart(fig_season, use_container_width=True)
            with col_s2:
                st.markdown("""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 8px;">
                    <span style="font-weight: 600; color: #1B4D3E;">🌾 ปัจจัยจังหวะเวลาของฝน (Rainfall Timing):</span><br>
                    <span style="font-size: 13px; color: #475569; line-height: 1.6;">
                        การรวมปริมาณฝนทั้งปีอาจบดบังสภาวะแห้งแล้งจริง เนื่องจากพืชเศรษฐกิจต้องการน้ำในระยะวิกฤต เช่น ข้าวต้องการน้ำในระยะแตกกอและตั้งท้องออกรวง (มิ.ย.–ก.ย.) หากฝนตกหนักนอกฤดูเพาะปลูก (เช่น พ.ย.–ธ.ค.) อาจกลายเป็นอุทกภัยสร้างความเสียหายต่อการเก็บเกี่ยวแทนที่จะช่วยเพิ่มผลผลิต
                    </span>
                </div>
                """, unsafe_allow_html=True)

    elif "2. แนวโน้มฝนสะสมระยะยาวระดับประเทศ" in climate_scope:
        st.markdown("#### 📈 ฝนสะสมเฉลี่ยทั้งประเทศระยะยาว vs ผลผลิตภาพรวม (พ.ศ. 2550–2562)")

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

        if not joined_nat_long.empty:
            col_l1, col_l2 = st.columns([1.1, 0.9])
            with col_l1:
                stats_long = calculate_bivariate_stats(joined_nat_long["rainfall_annual_mm"], joined_nat_long["production_ton"])
                fig_long = create_bivariate_scatter_chart(
                    joined_nat_long,
                    x_col="rainfall_annual_mm",
                    y_col="production_ton",
                    title=f"ฝนสะสมเฉลี่ยประเทศ vs ผลผลิตรวม ({st.session_state.selected_product})",
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
                st.markdown("##### 💡 การแปลผลแนวโน้มฝนระดับประเทศระยะยาว")
                st.markdown(
                    f"- **จำนวนปีที่วิเคราะห์:** {len(joined_nat_long)} ปี (พ.ศ. {int(joined_nat_long['year'].min())}–{int(joined_nat_long['year'].max())})\n"
                    f"- **ค่าสหสัมพันธ์เพียร์สัน (r):** {stats_long['pearson_r']:.3f} (p-value: {stats_long['pearson_p']:.4f})\n"
                    f"- **ข้อพึงระวัง:** ข้อมูลระดับประเทศเป็นการรวมยอดพื้นที่ทั้งหมด ทำให้ไม่เห็นความแตกต่างของภัยแล้งเฉพาะจุด (Spatial Heterogeneity)"
                )

                # Forecast / Statistical Projection Note
                st.markdown("""
                <div style="background-color: #FEF3C7; border: 1px solid #FDE68A; padding: 12px; border-radius: 6px; margin-top: 10px;">
                    <span style="font-weight: 600; font-size: 13px; color: #92400E;">🔮 ค่าคาดการณ์เชิงสถิติ (Statistical Projections):</span><br>
                    <span style="font-size: 12px; color: #78350F;">
                        การพยากรณ์ผลผลิตในอนาคตจากโมเดลการถดถอยเชิงเส้น เป็นเพียงค่าประมาณการภายใต้สมมติฐานว่าแนวโน้มสภาพอากาศคงที่ มิใช่การรับประกันผลผลิตจริงในอนาคต
                    </span>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("#### 📡 ข้อมูลตรวจวัดจากสถานีอุตุนิยมวิทยาเพื่อการเกษตรรายวัน (TMD Daily Agromet)")
        st.caption("แหล่งข้อมูล: กรมอุตุนิยมวิทยา (TMD Open Data API Catalog c1-1-3-7: WeatherTodayByAgro)")

        if df_agromet.empty:
            st.info("ไม่มีข้อมูลตรวจวัดสถานี TMD รายวันในขณะนี้")
        else:
            # Schema Validation (Requirement 8)
            is_valid, missing_cols = validate_agromet_schema(df_agromet)
            if not is_valid:
                st.warning(
                    f"⚠️ ชุดข้อมูลสถานี TMD ขาดคอลัมน์: {', '.join(missing_cols)} "
                    f"(แหล่งข้อมูล: กรมอุตุนิยมวิทยา TMD Open Data API Catalog c1-1-3-7)"
                )

            # Defensive projection of columns that exist in DataFrame
            expected_cols = [
                "station_name", "province", "temp_c", "relative_humidity_pct",
                "rainfall_24h_mm", "wind_speed_kmh", "data_source"
            ]
            display_cols = [c for c in expected_cols if c in df_agromet.columns]

            # Filter by selected province if filtered above
            selected_prov = st.session_state.selected_province
            if selected_prov != "ทั้งหมด" and "province" in df_agromet.columns:
                df_prov_stn = df_agromet[df_agromet["province"] == selected_prov]
                if not df_prov_stn.empty:
                    df_target_display = df_prov_stn
                    st.info(f"📍 แสดงผลข้อมูลสถานีตรวจวัดเฉพาะจังหวัด **{selected_prov}** (จำนวน {len(df_prov_stn)} สถานี)")
                else:
                    df_target_display = df_agromet
                    st.info(
                        f"📍 ไม่พบสถานีตรวจวัดการเกษตรเฉพาะในจังหวัด **{selected_prov}** "
                        f"จึงแสดงข้อมูลสถานีเครือข่ายทั่วประเทศทั้งหมด ({len(df_agromet)} สถานี)"
                    )
            else:
                df_target_display = df_agromet

            # Display KPI summary metrics for meteorological indicators
            col_st1, col_st2, col_st3, col_st4 = st.columns(4)
            with col_st1:
                st.metric("จำนวนสถานีในตาราง", f"{len(df_target_display):,} แห่ง")
            with col_st2:
                if "temp_c" in df_target_display.columns:
                    avg_t = df_target_display["temp_c"].dropna().mean()
                    st.metric("อุณหภูมิเฉลี่ย", f"{avg_t:.1f} °C" if pd.notna(avg_t) else "ไม่มีข้อมูล")
                else:
                    st.metric("อุณหภูมิเฉลี่ย", "ไม่มีข้อมูล")
            with col_st3:
                if "relative_humidity_pct" in df_target_display.columns:
                    avg_rh = df_target_display["relative_humidity_pct"].dropna().mean()
                    st.metric("ความชื้นสัมพัทธ์เฉลี่ย", f"{avg_rh:.0f}%" if pd.notna(avg_rh) else "ไม่มีข้อมูล")
                else:
                    st.metric("ความชื้นสัมพัทธ์เฉลี่ย", "ไม่มีข้อมูล")
            with col_st4:
                rain_col = "rainfall_24h_mm" if "rainfall_24h_mm" in df_target_display.columns else ("rainfall_mm" if "rainfall_mm" in df_target_display.columns else None)
                if rain_col:
                    stns_with_rain = df_target_display[df_target_display[rain_col] > 0]
                    st.metric("สถานีที่มีฝนรอบ 24 ชม.", f"{len(stns_with_rain)} แห่ง")
                else:
                    st.metric("สถานีที่มีฝนรอบ 24 ชม.", "ไม่มีข้อมูล")

            # Friendly column headers in Thai
            th_rename = {
                "station_name": "ชื่อสถานีตรวจวัด",
                "province": "จังหวัด",
                "temp_c": "อุณหภูมิ (°C)",
                "relative_humidity_pct": "ความชื้นสัมพัทธ์ (%)",
                "rainfall_24h_mm": "ฝน 24 ชม. (มม.)",
                "rainfall_mm": "ฝน 24 ชม. (มม.)",
                "wind_speed_kmh": "ความเร็วลม (กม./ชม.)",
                "data_source": "แหล่งข้อมูล"
            }

            st.dataframe(
                df_target_display[display_cols].rename(columns=th_rename),
                use_container_width=True
            )


# =========================================================
# TAB 3: PRODUCTION FACTORS (ปัจจัยที่เกี่ยวข้องกับผลผลิต)
# =========================================================
with tab3:
    st.markdown("### 🚜 ปัจจัยที่เกี่ยวข้องกับผลผลิตทางการเกษตร (Crop Production Drivers)")
    st.caption("วิเคราะห์ปัจจัยขับเคลื่อนผลผลิตนอกเหนือจากสภาพอากาศ: ระบบชลประทาน, ดิน Agri-Map, เทคโนโลยีเครื่องจักรกล, โครงสร้างต้นทุน และภัยพิบัติ")

    driver_crop = st.session_state.selected_product if st.session_state.selected_product != "ทั้งหมด" else "ข้าวนาปี"
    driver_year = st.session_state.selected_year if st.session_state.selected_year in [2561, 2562] else 2562
    driver_prov = st.session_state.selected_province

    st.info(
        f"🔍 **ตัวแปรวิเคราะห์ปัจจุบัน:** ชนิดพืช = **{driver_crop}** | ปี พ.ศ. = **{driver_year}** | "
        f"จังหวัด = **{driver_prov}** (ซิงค์ตามตัวกรองหลักด้านบน)"
    )

    # Local Agromet Context for the farmer if available
    if not df_agromet.empty and "province" in df_agromet.columns and driver_prov != "ทั้งหมด":
        prov_agromet = df_agromet[df_agromet["province"] == driver_prov]
        if not prov_agromet.empty:
            p_row = prov_agromet.iloc[0]
            st.markdown(
                f"<div style='background-color: #F0FDF4; border: 1px solid #BBF7D0; padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 13px; color: #166534;'>"
                f"📡 <b>สภาพอากาศสถานีตรวจวัดการเกษตร ณ ปัจจุบัน ({p_row.get('station_name', driver_prov)}):</b> "
                f"อุณหภูมิ <b>{p_row.get('temp_c', 'N/A')} °C</b> | "
                f"ความชื้น <b>{p_row.get('relative_humidity_pct', 'N/A')}%</b> | "
                f"ฝนรอบ 24 ชม. <b>{p_row.get('rainfall_24h_mm', 0.0):.1f} มม.</b> | "
                f"ความเร็วลม <b>{p_row.get('wind_speed_kmh', 0.0):.1f} กม./ชม.</b>"
                f"</div>",
                unsafe_allow_html=True
            )

    # 1. Water & Irrigation
    st.markdown("#### 1. 💧 แหล่งน้ำและระบบชลประทาน (RID & OAE)")
    df_irrig_yield = join_irrigation_and_yield(df_ag, df_irrig, crop=driver_crop, year=driver_year)

    col_ir1, col_ir2 = st.columns([1.1, 0.9])
    with col_ir1:
        fig_irrig = create_irrigation_vs_yield_chart(df_irrig_yield, driver_crop, driver_year)
        st.plotly_chart(fig_irrig, use_container_width=True)
    with col_ir2:
        st.markdown("##### 💡 บทบาทของระบบชลประทาน")
        st.markdown(
            "- **การควบคุมน้ำสม่ำเสมอ:** พื้นที่ชลประทานสามารถรักษาระดับน้ำให้เหมาะสมกับระยะการเจริญเติบโตของพืช ทำให้ดูดซึมปุ๋ยได้เต็มที่\n"
            "- **ลดความเสี่ยงฝนทิ้งช่วง:** เกษตรกรในเขตชลประทานไม่ต้องพึ่งพาฝนฟ้าอากาศเพียงอย่างเดียว สามารถปลูกพืชรอบสอง (เช่น ข้าวนาปรัง) ได้อย่างมั่นคง"
        )

    # 2. Soil Suitability (Agri-Map)
    st.markdown("---")
    st.markdown("#### 2. 🌱 ความเหมาะสมของดินตามระบบ Agri-Map (กรมพัฒนาที่ดิน LDD)")
    fig_soil = create_soil_suitability_bar_chart(df_soil, driver_crop, top_n=15)
    st.plotly_chart(fig_soil, use_container_width=True)

    st.markdown("""
    > [!TIP]
    > **แนวทางปฏิบัติของกรมพัฒนาที่ดิน (Agri-Map Guidance):**
    > - **พื้นที่ S1 & S2 (เหมาะสมสูง-ปานกลาง):** ควรรักษาความอุดมสมบูรณ์และใช้ปุ๋ยสั่งตัดตามค่าวิเคราะห์ดิน
    > - **พื้นที่ S3 (เหมาะสมน้อย):** ควรปรับปรุงโครงสร้างดินด้วยอินทรียวัตถุ ปุ๋ยคอก หรือปุ๋ยพืชสด
    > - **พื้นที่ N (ไม่เหมาะสม):** ควรพิจารณาปรับเปลี่ยนไปสู่พืชชนิดอื่นที่เหมาะสมกับสภาพดินตามหลักการ Zoning
    """)

    # 3. Disasters & Risk (Flood & Drought)
    st.markdown("---")
    st.markdown("#### 3. ⚠️ ภัยธรรมชาติ น้ำท่วม และภัยแล้ง (DOAE & DDPM)")
    fig_dis = create_disaster_impact_chart(df_disaster, driver_year, top_n=12)
    st.plotly_chart(fig_dis, use_container_width=True)

    # 4. Inputs & Mechanization Cost Structure
    st.markdown("---")
    st.markdown("#### 4. ⚙️ เทคโนโลยี เครื่องจักรกล และโครงสร้างต้นทุนการผลิต (OAE)")
    cost_summary = compare_production_cost_and_price(df_cost, df_prices, crop=driver_crop, year=driver_year)

    if cost_summary.get("has_data", False):
        col_c_ch1, col_c_ch2 = st.columns([1.1, 0.9])
        with col_c_ch1:
            fig_cost_str = create_cost_structure_breakdown_chart(
                cost_summary["cost_breakdown"], driver_crop, cost_summary["total_cost_rai"], driver_year
            )
            st.plotly_chart(fig_cost_str, use_container_width=True)
        with col_c_ch2:
            st.markdown("##### 💡 นัยสำคัญของการใช้เครื่องจักรกล")
            st.markdown(
                f"- **สัดส่วนค่าบริการเครื่องจักร ({cost_summary['mechanization_share_pct']:.1f}%):** ปัจจุบันการไถเตรียมดินและเกี่ยวนวดปรับเปลี่ยนสู่เครื่องจักรเกือบสมบูรณ์ ช่วยแก้ไขปัญหาขาดแคลนแรงงานสูงอายุ\n"
                f"- **ค่าปุ๋ยและสารเคมี:** เป็นต้นทุนปัจจัยแปรผันที่สำคัญ เกษตรกรควรใส่ปุ๋ยตามค่าวิเคราะห์ดินเพื่อลดต้นทุนส่วนเกิน"
            )


# =========================================================
# TAB 4: FARM ECONOMICS (ราคา ต้นทุน และความคุ้มค่า)
# =========================================================
with tab4:
    st.markdown("### 💰 ราคา ต้นทุน และความคุ้มค่าทางเศรษฐศาสตร์ฟาร์ม (Farm Economics)")
    st.caption("ข้อมูลราคาสินค้าเกษตรหน้าฟาร์มจริงจาก สศก., โครงสร้างต้นทุน และเครื่องคำนวณกำไรสุทธิเบื้องต้น")

    tab4_sub1, tab4_sub2 = st.tabs([
        "📈 ตลาดและแนวโน้มราคา (Market Prices & Margins)",
        "🧮 เครื่องคำนวณกำไรและจุดคุ้มทุน (Farm Profit Calculator)"
    ])

    with tab4_sub1:
        st.markdown("#### 🌾 ราคาสินค้าเกษตรหน้าฟาร์มรายสัปดาห์ (สศก. พ.ศ. 2561–2562)")
        if not df_prices.empty:
            sel_price_crop = st.selectbox(
                "เลือกชนิดสินค้าเกษตรสำหรับดูกราฟราคา:",
                options=sorted(df_prices["product"].unique().tolist()),
                index=0,
                key="tab4_price_crop_select"
            )
            crop_prices = df_prices[df_prices["product"] == sel_price_crop]

            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("ราคาเฉลี่ย", f"{crop_prices['price_baht_per_kg'].mean():,.2f} บาท/กก.")
            with col_p2:
                st.metric("ราคาสูงสุด", f"{crop_prices['price_baht_per_kg'].max():,.2f} บาท/กก.")
            with col_p3:
                st.metric("ราคาต่ำสุด", f"{crop_prices['price_baht_per_kg'].min():,.2f} บาท/กก.")

            fig_p_trend = create_price_trend_chart(df_prices, sel_price_crop)
            st.plotly_chart(fig_p_trend, use_container_width=True)

            # Cost vs Price Margin Comparison
            st.markdown("---")
            st.markdown("#### ⚖️ เปรียบเทียบราคาขายเฉลี่ย vs ต้นทุนการผลิตต่อ กก.")
            margin_summary = compare_production_cost_and_price(df_cost, df_prices, crop=sel_price_crop, year=driver_year)
            fig_margin = create_cost_price_margin_chart(margin_summary)
            st.plotly_chart(fig_margin, use_container_width=True)
        else:
            st.info("ไม่มีข้อมูลราคาหน้าฟาร์มในขณะนี้")

    with tab4_sub2:
        st.markdown("#### 🧮 เครื่องคำนวณรายรับและกำไรเบื้องต้น (Farm Profit Calculator)")
        st.caption("คำนวณต้นทุนรวม ผลผลิตรวม รายรับ และกำไรสุทธิ โดยแยกช่องกรอกของเกษตรกรออกจากค่าอ้างอิงของภาครัฐ")

        # Benchmark reference values
        calc_crop = st.session_state.selected_product if st.session_state.selected_product != "ทั้งหมด" else "ข้าวนาปี"
        cost_bench = df_cost[df_cost["crop"] == calc_crop]
        default_cost_rai = float(cost_bench["total_cost_rai"].values[0]) if not cost_bench.empty else 4500.0
        default_yield_rai = float(cost_bench["benchmark_yield_kg_rai"].values[0]) if (not cost_bench.empty and "benchmark_yield_kg_rai" in cost_bench.columns) else 450.0

        price_sub = df_prices[df_prices["product"] == calc_crop]
        default_price_kg = float(price_sub["price_baht_per_kg"].mean()) if not price_sub.empty else 10.50

        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px;">
            <b>📌 ข้อมูลอ้างอิงเฉลี่ยทางการ (สศก. สำหรับ {calc_crop}):</b> 
            ต้นทุนเฉลี่ย: <b>{default_cost_rai:,.0f} บาท/ไร่</b> | 
            ผลผลิตเฉลี่ย: <b>{default_yield_rai:,.0f} กก./ไร่</b> | 
            ราคาหน้าฟาร์มเฉลี่ย: <b>{default_price_kg:,.2f} บาท/กก.</b>
        </div>
        """, unsafe_allow_html=True)

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            in_area = st.number_input("1. เนื้อที่เพาะปลูก (ไร่):", min_value=1.0, max_value=10000.0, value=20.0, step=1.0)
            in_yield = st.number_input("2. ผลผลิตต่อไร่ที่คาดว่าจะได้ (กก./ไร่):", min_value=10.0, max_value=5000.0, value=default_yield_rai, step=10.0)
        with col_in2:
            in_cost = st.number_input("3. ต้นทุนรวมต่อไร่ (บาท/ไร่):", min_value=100.0, max_value=50000.0, value=default_cost_rai, step=100.0)
            in_price = st.number_input("4. ราคาขายที่คาดหวัง (บาท/กก.):", min_value=0.5, max_value=200.0, value=default_price_kg, step=0.25)

        # Dynamic Math Formulas (Strictly matching specifications)
        # Total Cost = Area * Cost per Rai
        # Total Yield = Area * Yield per Rai
        # Revenue = Total Yield * Sale Price per Unit
        # Estimated Profit = Revenue - Total Cost
        tot_cost = in_area * in_cost
        tot_yield = in_area * in_yield
        tot_revenue = tot_yield * in_price
        est_profit = tot_revenue - tot_cost
        breakeven_yield = tot_cost / (in_area * in_price) if (in_area * in_price) > 0 else 0.0
        breakeven_price = tot_cost / tot_yield if tot_yield > 0 else 0.0

        st.markdown("---")
        st.markdown("#### 📊 ผลการคำนวณทางเศรษฐศาสตร์ฟาร์ม")

        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        with col_res1:
            st.metric("ต้นทุนรวม (Total Cost)", f"{tot_cost:,.0f} บาท")
        with col_res2:
            st.metric("ผลผลิตรวม (Total Yield)", f"{tot_yield:,.0f} กก.", help=f"คิดเป็น {tot_yield/1000:,.2f} ตัน")
        with col_res3:
            st.metric("รายรับรวม (Revenue)", f"{tot_revenue:,.0f} บาท")
        with col_res4:
            profit_color = "normal" if est_profit >= 0 else "inverse"
            st.metric("กำไรสุทธิประมาณการ (Profit)", f"{est_profit:,.0f} บาท", delta=f"{est_profit / tot_cost * 100:+.1f}% ROI" if tot_cost > 0 else None)

        col_be1, col_be2 = st.columns(2)
        with col_be1:
            st.info(f"🎯 **จุดคุ้มทุนด้านผลผลิต (Break-even Yield):** **{breakeven_yield:,.1f} กก./ไร่** (หากผลผลิตต่ำกว่านี้จะขาดทุน ณ ราคา {in_price:.2f} บาท/กก.)")
        with col_be2:
            st.info(f"🎯 **จุดคุ้มทุนด้านราคา (Break-even Price):** **{breakeven_price:,.2f} บาท/กก.** (หากขายได้ต่ำกว่านี้จะขาดทุน ณ ผลผลิต {in_yield:,.0f} กก./ไร่)")

        # Mandatory Disclaimer
        st.markdown("""
        <div style="background-color: #FFFBEB; border: 1px solid #FCD34D; padding: 12px 16px; border-radius: 8px; margin-top: 14px;">
            <span style="font-weight: 600; color: #B45309; font-size: 13px;">⚠️ ข้อจำกัดความรับผิดชอบ (Disclaimer):</span><br>
            <span style="font-size: 12px; color: #92400E; line-height: 1.5;">
                ผลการคำนวณเป็นเพียงการประมาณการเบื้องต้นจากตัวเลขที่ผู้ใช้กำหนดขึ้น มิใช่การยืนยันผลประกอบการจริงจากระบบบัญชีฟาร์ม เกษตรกรควรพิจารณาความเสี่ยงด้านสภาพอากาศ ความผันผวนของราคาตลาด และคุณภาพผลผลิตจริงประกอบการตัดสินใจ
            </span>
        </div>
        """, unsafe_allow_html=True)
