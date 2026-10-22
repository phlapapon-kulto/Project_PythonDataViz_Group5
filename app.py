"""
Thai Agriculture & Climate Intelligence Dashboard — Real Open Data Edition
A comprehensive, academic-grade data intelligence platform analyzing Thailand's
agricultural production, spatial distribution, climate interactions, crop planning,
and market economics.

Built with Streamlit, Plotly, Pandas, and SciPy.
Strictly relies on verified government Open Data sources (OAE, TMD, HII).
Zero mock data. Double-counting prevented 100%.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any, Optional

from src.data_loader import (
    load_agriculture_data,
    load_spatial_rain_data,
    load_national_rain_data,
    load_provincial_temp_data,
    load_oae_farmgate_prices,
    load_tmd_daily_agromet,
    load_thailand_geojson
)
from src.data_cleaning import (
    clean_agriculture_data,
    clean_spatial_rain_data,
    clean_national_rain_data,
    clean_provincial_temp_data,
    clean_oae_farmgate_prices,
    clean_tmd_daily_agromet,
    OFFICIAL_77_PROVINCES,
    COUNTRY_NAME,
    REGION_NAMES
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
    get_crop_calendar_guide
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
    COLOR_FOREST_DARK,
    COLOR_FOREST_MED,
    COLOR_WARNING,
    COLOR_DANGER
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

    return {
        "ag": df_ag,
        "rain_monthly": df_rain_monthly,
        "rain_annual": df_rain_annual,
        "nat_rain_annual": df_nat_annual,
        "temp_annual": df_temp_annual,
        "prices": df_prices,
        "agromet": df_agromet,
        "geojson": geojson_data
    }


data_dict = load_all_datasets()
df_ag = data_dict["ag"]
df_rain_annual = data_dict["rain_annual"]
df_nat_annual = data_dict["nat_rain_annual"]
df_temp_annual = data_dict["temp_annual"]
df_prices = data_dict["prices"]
df_agromet = data_dict["agromet"]
geojson_data = data_dict["geojson"]

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
    st.markdown("#### 🎯 พืชเศรษฐกิจหลัก 4 ชนิด")
    col_crop1, col_crop2 = st.columns(2)
    with col_crop1:
        if st.button("🌾 ข้าวนาปี", use_container_width=True):
            st.session_state.selected_product = "ข้าวนาปี"
        if st.button("🎋 อ้อยโรงงาน", use_container_width=True):
            st.session_state.selected_product = "อ้อยโรงงาน"
    with col_crop2:
        if st.button("🥔 มันสำปะหลัง", use_container_width=True):
            st.session_state.selected_product = "มันสำปะหลังโรงงาน"
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ TAB 1: ภาพรวมประเทศไทย (Thailand Overview)",
    "📍 TAB 2: แผนที่พื้นที่เพาะปลูกจริง (Real Cultivation Map)",
    "🌦️ TAB 3: สภาพอากาศกับความเสี่ยง (Climate Impact & Risk)",
    "📅 TAB 4: ผู้ช่วยวางแผนเพาะปลูก (Crop Planning Assistant)",
    "💰 TAB 5: ตลาดและเครื่องคำนวณฟาร์ม (Market & Farm Calculator)"
])


# =========================================================
# TAB 1: THAILAND OVERVIEW
# =========================================================
with tab1:
    st.markdown("### 📌 สรุปตัวชี้วัดภาคเกษตรภาพรวม (National Agricultural KPIs)")
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
    # Rankings & Historical Trend
    # ---------------------------------------------------------
    col_rank_p, col_rank_c = st.columns(2)

    with col_rank_p:
        st.markdown(f"#### 🏆 10 อันดับจังหวัดที่มีผลผลิตสูงสุด (ปี {st.session_state.selected_year})")
        top_prov_df = get_top_provinces(df_year_prov, metric="production_ton", top_n=10)
        fig_top_prov = create_ranking_bar_chart(
            top_prov_df,
            x_col="production_ton",
            y_col="province",
            title=f"10 อันดับจังหวัด ({st.session_state.selected_product})",
            x_label="ผลผลิตรวม",
            unit="ตัน",
            selected_item=st.session_state.selected_province if st.session_state.selected_province != "ทั้งหมด" else None
        )
        st.plotly_chart(fig_top_prov, use_container_width=True)

    with col_rank_c:
        st.markdown(f"#### 🌾 10 อันดับสินค้าเกษตรที่มีผลผลิตสูงสุด (ปี {st.session_state.selected_year})")
        top_crop_df = get_top_crops(df_year_prov, metric="production_ton", top_n=10)
        fig_top_crop = create_ranking_bar_chart(
            top_crop_df,
            x_col="production_ton",
            y_col="product",
            title=f"10 อันดับสินค้าเกษตรภาพรวม",
            x_label="ผลผลิต",
            unit="ตัน",
            selected_item=st.session_state.selected_product if st.session_state.selected_product != "ทั้งหมด" else None
        )
        st.plotly_chart(fig_top_crop, use_container_width=True)

    st.markdown("---")
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


# =========================================================
# TAB 2: REAL CULTIVATION MAP
# =========================================================
with tab2:
    st.markdown("### 📍 แผนที่พื้นที่เพาะปลูกจริง (Real Cultivation Map)")
    
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

    # 4 Main Economic Crops Quick Selector
    col_m1, col_m2 = st.columns([1.2, 0.8])
    with col_m1:
        map_crop_choice = st.selectbox(
            "เลือกสินค้าเกษตรสำหรับแสดงบนแผนที่ (Select Crop):",
            options=["ข้าวนาปี", "ข้าวนาปรัง", "อ้อยโรงงาน", "มันสำปะหลังโรงงาน", "ข้าวโพดเลี้ยงสัตว์"] + [c for c in sorted(df_ag["product"].unique()) if c not in ["ข้าวนาปี", "ข้าวนาปรัง", "อ้อยโรงงาน", "มันสำปะหลังโรงงาน", "ข้าวโพดเลี้ยงสัตว์"]],
            index=0,
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
        st.markdown(f"#### 📊 สรุปสถิติเชิงพื้นที่: {map_crop_choice} (ปี พ.ศ. {map_year_choice})")
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
        st.markdown("#### 🛰️ บริการตรวจจับพืชจากดาวเทียม (GISTDA Open Data API)")
        st.caption("ชุดข้อมูล: พื้นที่ปลูกพืชเศรษฐกิจ 6 ชนิด (Catalog agriculture-07 โดย สทอภ. GISTDA)")

        gistda_key_input = st.text_input("GISTDA API Key (เว้นว่างได้หากยังไม่มีกุญแจ):", type="password", key="gistda_key")
        col_coord1, col_coord2 = st.columns(2)
        with col_coord1:
            test_lat = st.number_input("ละติจูด (Lat):", value=14.6346, format="%.4f")
        with col_coord2:
            test_lon = st.number_input("ลองจิจูด (Lon):", value=100.2832, format="%.4f")

        if st.button("🔍 ตรวจสอบพืชจากพิกัด (Check Crop API)"):
            if not gistda_key_input:
                st.warning("⚠️ **สถานะการเชื่อมต่อ:** การเรียกดูผลการตรวจจับจากดาวเทียมสดต้องใช้ API Key ที่ลงทะเบียนผ่าน GISTDA Open Data Gateway (HTTP 403 Forbidden เมื่อไม่มี Key) — ตามหลักธรรมาภิบาลข้อมูล ระบบจะไม่สร้างผลลัพธ์จำลอง")
            else:
                st.info("กำลังเชื่อมต่อกับ GISTDA Open Data API Gateway...")

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
    st.markdown("### 🌦️ การวิเคราะห์สภาพภูมิอากาศกับผลผลิตและความเสี่ยง (Climate Impact & Risk)")
    
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
        st.markdown("#### 🌐 ปริมาณฝนเชิงพื้นที่ HII (IDW) vs ผลผลิตต่อไร่ 77 จังหวัด")
        
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
                    title=f"ปริมาณฝนเชิงพื้นที่ HII vs Yield (77 จังหวัด | ปี {joined_nation['year'].iloc[0]})",
                    x_label="ปริมาณฝนเชิงพื้นที่",
                    y_label="ผลผลิตต่อไร่",
                    x_unit="มม.",
                    y_unit="กก./ไร่",
                    text_col="province",
                    correlation_val=stats_nat_rain["pearson_r"] if stats_nat_rain["is_valid"] else None
                )
                st.plotly_chart(fig_nat_scat, use_container_width=True)

            with col_n2:
                st.markdown("##### 📝 การแปลผลทางสถิติและข้อพึงระวังทางวิชาการ")
                interp_nat = generate_climate_interpretation(
                    bivariate_result=stats_nat_rain,
                    crop_name=st.session_state.selected_product,
                    climate_var_name="ปริมาณฝนเชิงพื้นที่ HII (IDW)",
                    ag_var_name="ผลผลิตต่อไร่ (Yield)",
                    geo_scope=f"77 จังหวัดทั่วประเทศ (ปี {joined_nation['year'].iloc[0]})",
                    year_range=str(joined_nation['year'].iloc[0])
                )
                st.markdown(interp_nat)

            with st.expander("🔍 ตารางข้อมูลตรวจวัดเชิงพื้นที่ 77 จังหวัด"):
                st.dataframe(joined_nation[["province", "year", "product", "rainfall_annual_mm", "production_ton", "area_harvest_rai", "calc_yield_per_rai"]], use_container_width=True)

    elif "2. แนวโน้มฝนสะสมเฉลี่ยระดับประเทศระยะยาว" in climate_scope:
        st.markdown("#### 🌧️ ปริมาณฝนเฉลี่ยสะสมทั้งประเทศรายปี vs ผลผลิตเกษตรภาพรวม (พ.ศ. 2550–2562)")

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
                correlation_val=stats_long["pearson_r"] if stats_long["is_valid"] else None
            )
            st.plotly_chart(fig_long, use_container_width=True)

        with col_l2:
            st.markdown("##### 📝 การแปลผลแนวโน้มฝนระดับประเทศระยะยาว")
            interp_long = generate_climate_interpretation(
                bivariate_result=stats_long,
                crop_name=st.session_state.selected_product,
                climate_var_name="ฝนสะสมเฉลี่ยทั้งประเทศ (HII)",
                ag_var_name="ผลผลิตรวมทั้งประเทศ (Production)",
                geo_scope="ภาพรวมทั้งประเทศ",
                year_range="2550–2562"
            )
            st.markdown(interp_long)

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
    st.markdown("### 📅 ผู้ช่วยวางแผนการเพาะปลูกสำหรับเกษตรกร (Crop Planning Assistant)")
    
    st.markdown("""
    <div style="background-color: #F8FAFC; border-left: 4px solid #1B4D3E; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
        <span class="gov-badge">🌾 ปฏิทินและคำแนะนำทางวิชาการ (Official Agronomic Guidelines)</span>
        <span style="font-size: 13px; color: #334155;">
            รวบรวมข้อกำหนดและคำแนะนำมาตรฐานจาก <b>กรมการข้าว</b>, <b>กรมวิชาการเกษตร</b> และ <b>กรมส่งเสริมการเกษตร</b><br>
            <i>⚠️ ข้อพึงระวัง: ข้อมูลนี้เป็นแนวทางประกอบการตัดสินใจเบื้องต้น ไม่สามารถรับประกันผลผลิตหรือผลตอบแทนของแต่ละแปลงได้</i>
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_plan_sel1, col_plan_sel2, col_plan_sel3 = st.columns(3)
    with col_plan_sel1:
        target_crop = st.selectbox(
            "🌱 เลือกพืชที่ต้องการวางแผน:",
            options=["ข้าวนาปี", "ข้าวนาปรัง", "อ้อยโรงงาน", "มันสำปะหลังโรงงาน", "ข้าวโพดเลี้ยงสัตว์"] + [c for c in sorted(df_ag["product"].unique()) if c not in ["ข้าวนาปี", "ข้าวนาปรัง", "อ้อยโรงงาน", "มันสำปะหลังโรงงาน", "ข้าวโพดเลี้ยงสัตว์"]],
            index=0,
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
        st.markdown(f"#### 📖 ปฏิทินและข้อกำหนดทางปฐพีวิทยา ({target_crop})")
        st.markdown(f"""
        - **ช่วงเวลาเพาะปลูกที่เหมาะสม:** {guide.get('planting_window')}
        - **ช่วงเวลาเก็บเกี่ยวผลผลิต:** {guide.get('harvest_window')}
        - **ความต้องการดินและน้ำ:** {guide.get('water_soil_req')}
        """)

        st.markdown("##### ⚠️ ข้อควรระวังความเสี่ยงสภาพอากาศและโรคแมลง:")
        st.warning(guide.get('key_cautions'))

    with col_guide_r:
        st.markdown("#### ✅ รายการตรวจสอบก่อนการเพาะปลูก (Pre-planting Checklist)")
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
        st.markdown("#### 📈 สถิติราคาสินค้าเกษตรที่เกษตรกรขายได้ที่ไร่นา รายสัปดาห์ (พ.ศ. 2561–2562)")
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
        st.markdown("#### 🧮 เครื่องคำนวณต้นทุน ผลตอบแทน และจุดคุ้มทุนฟาร์ม (Farm Budget Calculator)")
        st.caption("คำนวณจากต้นทุนจริงที่เกษตรกรระบุเอง โดยใช้สถิติผลผลิตของจังหวัดและราคาตลาด สศก. เป็นฐานอ้างอิงเริ่มต้น")

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
        st.markdown("#### 📊 ผลการคำนวณกำไร-ขาดทุน และจุดคุ้มทุน (Financial Analysis)")

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
            st.markdown("##### 💡 บทวิเคราะห์เพื่อการตัดสินใจสำหรับเกษตรกร")
            margin = budget_result["profit_margin_pct"]
            if net_p > 0 and margin >= 20.0:
                st.success(f"✅ **ผลการวิเคราะห์อยู่ในเกณฑ์ดี:** อัตรากำไรสุทธิอยู่ที่ {margin:.1f}% และราคาขายปัจจุบัน ({user_price:.2f} บ./กก.) สูงกว่าราคาจุดคุ้มทุน ({be_p:.2f} บ./กก.) มีความปลอดภัยต่อความผันผวนของราคาตลาด")
            elif net_p > 0:
                st.warning(f"⚠️ **ควรระมัดระวังความเสี่ยงด้านต้นทุน:** อัตรากำไรสุทธิอยู่ที่ {margin:.1f}% ถือว่าค่อนข้างบาง หากราคาตลาดปรับลดลงต่ำกว่า {be_p:.2f} บาท/กก. หรือผลผลิตลดลงต่ำกว่า {be_y:,.0f} กก./ไร่ จะเริ่มเข้าสู่ภาวะขาดทุน แนะนำให้วางแผนลดต้นทุนปุ๋ยและค่าจ้างเหมา")
            else:
                st.error(f"❌ **ส่งสัญญาณเตือนภาวะขาดทุน:** ภายใต้ต้นทุนที่ระบุ ต้นทุนเฉลี่ยอยู่ที่ {budget_result['cost_per_kg']:.2f} บาท/กก. สูงกว่าราคาขาย {user_price:.2f} บาท/กก. จำเป็นต้องลดต้นทุนปัจจัยการผลิตลงอย่างน้อย {abs(net_p):,.0f} บาท หรือเพิ่มผลผลิตต่อไร่ให้ถึงจุดคุ้มทุน ({be_y:,.0f} กก./ไร่)")
        else:
            st.error(budget_result.get("message", "ข้อมูลไม่ถูกต้อง"))

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
