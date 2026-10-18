"""
Thai Agriculture & Climate Intelligence Dashboard
A comprehensive, academic-grade data intelligence platform analyzing Thailand's
agricultural production, spatial distribution, climate interactions, and production risk.

Built with Streamlit, Plotly, Pandas, and SciPy.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json

from src.data_loader import (
    load_agriculture_data,
    load_suphan_climate_data,
    load_spatial_rain_data,
    load_national_rain_data,
    load_provincial_temp_data,
    load_thailand_geojson
)
from src.data_cleaning import (
    clean_agriculture_data,
    clean_suphan_climate_data,
    clean_spatial_rain_data,
    clean_national_rain_data,
    clean_provincial_temp_data,
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
    join_climate_agriculture_suphan,
    join_climate_agriculture_nationwide,
    calculate_bivariate_stats,
    generate_climate_interpretation,
    classify_area_yield_quadrants,
    get_top_provinces,
    get_top_crops
)
from src.visualizations import (
    create_thailand_map,
    create_ranking_bar_chart,
    create_trend_line_chart,
    create_bivariate_scatter_chart,
    create_correlation_heatmap,
    create_quadrant_scatter_chart,
    create_yoy_bar_chart,
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
    /* Main container and typography */
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
        font-size: 12.5px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
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
        color: #52B788;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Section Cards */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #1B4D3E;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Badge & Alert styles */
    .risk-badge {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .safe-badge {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Data Loading & Processing
# ---------------------------------------------------------
@st.cache_data(show_spinner="กำลังโหลดและประมวลผลข้อมูล...")
def load_all_datasets():
    # 1. Agriculture Data
    raw_ag = load_agriculture_data()
    df_ag = clean_agriculture_data(raw_ag)

    # 2. Suphan Buri Climate Data
    try:
        raw_suphan = load_suphan_climate_data()
        df_suphan = clean_suphan_climate_data(raw_suphan)
    except Exception:
        df_suphan = pd.DataFrame()

    # 3. HII Spatial Rain Data
    try:
        raw_rain = load_spatial_rain_data()
        df_rain_monthly, df_rain_annual = clean_spatial_rain_data(raw_rain)
    except Exception:
        df_rain_monthly = pd.DataFrame()
        df_rain_annual = pd.DataFrame()

    # 4. HII National Rain Data
    try:
        raw_nat = load_national_rain_data()
        df_nat_monthly, df_nat_annual = clean_national_rain_data(raw_nat)
    except Exception:
        df_nat_monthly = pd.DataFrame()
        df_nat_annual = pd.DataFrame()

    # 5. HII Provincial Temp Data
    try:
        raw_temp = load_provincial_temp_data()
        df_temp_monthly, df_temp_annual = clean_provincial_temp_data(raw_temp)
    except Exception:
        df_temp_monthly = pd.DataFrame()
        df_temp_annual = pd.DataFrame()

    # 6. GeoJSON
    try:
        geojson_data = load_thailand_geojson()
    except Exception:
        geojson_data = None

    return {
        "ag": df_ag,
        "suphan_climate": df_suphan,
        "rain_monthly": df_rain_monthly,
        "rain_annual": df_rain_annual,
        "nat_rain_annual": df_nat_annual,
        "temp_annual": df_temp_annual,
        "geojson": geojson_data
    }


data_dict = load_all_datasets()
df_ag = data_dict["ag"]
df_suphan = data_dict["suphan_climate"]
df_rain_annual = data_dict["rain_annual"]
df_nat_annual = data_dict["nat_rain_annual"]
df_temp_annual = data_dict["temp_annual"]
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

    # Year Selector
    year_options = available_years
    curr_yr_idx = year_options.index(st.session_state.selected_year) if st.session_state.selected_year in year_options else len(year_options) - 1
    selected_year = st.selectbox(
        "📅 ปี พ.ศ. (Year BE)",
        options=year_options,
        index=curr_yr_idx,
        key="selected_year_box"
    )
    st.session_state.selected_year = selected_year

    # Product Selector
    all_products = ["ทั้งหมด"] + sorted(df_ag["product"].unique().tolist())
    curr_prod_idx = all_products.index(st.session_state.selected_product) if st.session_state.selected_product in all_products else 0
    selected_product = st.selectbox(
        "🌱 ชนิดสินค้าเกษตร (Crop)",
        options=all_products,
        index=curr_prod_idx,
        key="selected_product_box"
    )
    st.session_state.selected_product = selected_product

    # Province Selector
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
    st.markdown("#### ℹ️ ขอบเขตข้อมูล (Data Scope)")
    st.markdown(
        f"- **ปีข้อมูลเกษตร:** พ.ศ. {min(available_years)}–{max(available_years)} (13 ปี)\n"
        f"- **จำนวนสินค้าเกษตร:** 26 รายการ\n"
        f"- **ขอบเขตพื้นที่:** 77 จังหวัดทั่วประเทศไทย\n"
        f"- **สถานีสภาพอากาศสุพรรณบุรี:** พ.ศ. 2550–2568\n"
        f"- **ฝนเชิงพื้นที่ HII:** พ.ศ. 2561–2569\n"
        f"- **สถานะข้อมูล:** เชื่อมโยงข้อมูลจริง ป้องกันการนับซ้ำ 100%"
    )

    with st.expander("📖 เอกสารพจนานุกรมข้อมูล (Data Dictionary)"):
        st.markdown(
            "สามารถอ่านรายละเอียดระเบียบวิธีวิจัย แหล่งอ้างอิง URL และใบอนุญาตได้ที่ไฟล์ `data_sources.md` ในระบบ"
        )


# ---------------------------------------------------------
# App Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>ระบบวิเคราะห์ข้อมูลการผลิตสินค้าเกษตรและภูมิอากาศไทย (Thai Agriculture & Climate Intelligence)</h1>
    <p>แพลตฟอร์มวิเคราะห์เชิงลึกระดับวิชาการ สำรวจการกระจายตัวเชิงพื้นที่ ประสิทธิภาพการผลิต และความสัมพันธ์กับสภาพภูมิอากาศ</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Filter Application for Main Context
# ---------------------------------------------------------
# Provincial subset for Year and Product (for mapping and rankings)
df_year_prov = apply_ag_filters(
    df_ag,
    year=st.session_state.selected_year,
    province=None,  # keep all provinces for maps/rankings
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

# National equivalent multi-year series (for comparison if province selected)
df_national_trend = apply_ag_filters(
    df_ag,
    year=None,
    province="ทั้งประเทศ",
    product=st.session_state.selected_product,
    geo_level="country"
)


# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🗺️ TAB 1: ภาพรวมประเทศไทย (Thailand Overview)",
    "🌦️ TAB 2: สภาพอากาศกับผลผลิต (Climate Impact)",
    "⚠️ TAB 3: ประสิทธิภาพและความเสี่ยง (Production & Risk)"
])


# =========================================================
# TAB 1: THAILAND OVERVIEW
# =========================================================
with tab1:
    st.markdown("### 📌 สรุปตัวชี้วัดภาพรวม (Key Performance Indicators)")

    # Compute KPIs safely
    kpis = aggregate_agricultural_kpis(df_active)

    # YoY Production change calculation for current active filter
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
    # Map & Rankings Section
    # ---------------------------------------------------------
    col_map, col_rank = st.columns([1.1, 0.9])

    with col_map:
        st.markdown("#### 🗺️ แผนที่การกระจายตัวเชิงพื้นที่ (Choropleth Map)")

        metric_dict = {
            "production_ton": ("ปริมาณผลผลิตรวม", "ตัน"),
            "area_plant_rai": ("เนื้อที่เพาะปลูก", "ไร่"),
            "area_harvest_rai": ("เนื้อที่เก็บเกี่ยว", "ไร่"),
            "calc_yield_per_rai": ("ผลผลิตต่อไร่แบบถ่วงน้ำหนัก", "กก./ไร่"),
            "calc_harvest_ratio": ("อัตราการเก็บเกี่ยว", "%")
        }

        # Add Spatial Rainfall option if year is in HII coverage (2561 or 2562)
        if st.session_state.selected_year in [2561, 2562] and not df_rain_annual.empty:
            metric_dict["rainfall_annual_mm"] = ("ปริมาณน้ำฝนเชิงพื้นที่ HII (IDW)", "มม.")

        map_metric_key = st.selectbox(
            "เลือกตัวชี้วัดที่ต้องการแสดงบนแผนที่ (Select Map Metric):",
            options=list(metric_dict.keys()),
            format_func=lambda k: f"{metric_dict[k][0]} ({metric_dict[k][1]})"
        )
        metric_label, metric_unit = metric_dict[map_metric_key]

        # Prepare Map DataFrame
        if map_metric_key == "rainfall_annual_mm":
            map_data = df_rain_annual[df_rain_annual["year"] == st.session_state.selected_year].copy()
        else:
            # Aggregate by province for current year & product
            map_data = df_year_prov.groupby("province").agg(
                production_ton=("production_ton", "sum"),
                area_plant_rai=("area_plant_rai", "sum"),
                area_harvest_rai=("area_harvest_rai", "sum")
            ).reset_index()
            map_data["calc_yield_per_rai"] = np.where(
                map_data["area_harvest_rai"] > 0,
                map_data["production_ton"] * 1000.0 / map_data["area_harvest_rai"],
                np.nan
            )
            map_data["calc_harvest_ratio"] = np.where(
                map_data["area_plant_rai"] > 0,
                map_data["area_harvest_rai"] / map_data["area_plant_rai"] * 100.0,
                np.nan
            )

        if geojson_data:
            sel_prov_arg = st.session_state.selected_province if st.session_state.selected_province != "ทั้งหมด" else None
            map_fig = create_thailand_map(
                df_data=map_data,
                geojson_data=geojson_data,
                metric_col=map_metric_key,
                metric_label=metric_label,
                unit=metric_unit,
                year=st.session_state.selected_year,
                product_name=st.session_state.selected_product,
                selected_province=sel_prov_arg
            )
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.warning("ไม่สามารถโหลดไฟล์ GeoJSON ขอบเขตจังหวัดได้ แสดงเฉพาะข้อมูลในตาราง")

        st.caption("💡 **วิธีอ่านแผนที่:** นำเมาส์ไปชี้ (Hover) เพื่อดูชื่อจังหวัด ค่า และหน่วยวัด | พื้นที่สีเทาคือไม่มีการบันทึกข้อมูลการเพาะปลูกสำหรับพืชและปีนี้")

    with col_rank:
        st.markdown("#### 🏆 การจัดอันดับ (Top 10 Rankings)")
        rank_tab1, rank_tab2 = st.tabs(["Top 10 จังหวัด", "Top 10 สินค้าเกษตร"])

        with rank_tab1:
            top_prov_metric = map_metric_key if map_metric_key in ["production_ton", "area_plant_rai", "calc_yield_per_rai"] else "production_ton"
            top_p_label, top_p_unit = metric_dict.get(top_prov_metric, ("ผลผลิตรวม", "ตัน"))
            top_prov_df = get_top_provinces(df_year_prov, metric=top_prov_metric, top_n=10)
            
            fig_top_prov = create_ranking_bar_chart(
                top_prov_df,
                x_col=top_prov_metric,
                y_col="province",
                title=f"10 อันดับจังหวัดที่มี {top_p_label} สูงสุด (ปี {st.session_state.selected_year})",
                x_label=top_p_label,
                unit=top_p_unit,
                selected_item=st.session_state.selected_province if st.session_state.selected_province != "ทั้งหมด" else None
            )
            st.plotly_chart(fig_top_prov, use_container_width=True)

        with rank_tab2:
            # Top crops in current selected province / year
            top_crop_df = get_top_crops(df_year_prov, metric="production_ton", top_n=10)
            fig_top_crop = create_ranking_bar_chart(
                top_crop_df,
                x_col="production_ton",
                y_col="product",
                title=f"10 อันดับสินค้าเกษตรที่มีผลผลิตสูงสุด (ปี {st.session_state.selected_year})",
                x_label="ผลผลิต",
                unit="ตัน",
                selected_item=st.session_state.selected_product if st.session_state.selected_product != "ทั้งหมด" else None
            )
            st.plotly_chart(fig_top_crop, use_container_width=True)

    # ---------------------------------------------------------
    # Multi-Year Trend Section
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📈 แนวโน้มการผลิตรายปี พ.ศ. 2550–2562 (Multi-Year Historical Trend)")

    col_trend_ctrl1, col_trend_ctrl2 = st.columns([1, 1])
    with col_trend_ctrl1:
        trend_metric = st.selectbox(
            "เลือกตัวแปรแนวโน้ม (Trend Variable):",
            options=["production_ton", "area_plant_rai", "calc_yield_per_rai", "calc_harvest_ratio"],
            format_func=lambda k: metric_dict.get(k, (k, ""))[0]
        )
    t_label, t_unit = metric_dict.get(trend_metric, (trend_metric, ""))

    # Aggregate trend data across years
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

    scope_title = f"จังหวัด{st.session_state.selected_province}" if st.session_state.selected_province != "ทั้งหมด" else "ภาพรวมทั้งประเทศ (ผลรวม 77 จังหวัด)"
    crop_title = f"สินค้า: {st.session_state.selected_product}"
    
    fig_trend = create_trend_line_chart(
        trend_data,
        x_col="year",
        y_col=trend_metric,
        color_col=None,
        title=f"แนวโน้ม {t_label} ({scope_title} | {crop_title})",
        y_label=t_label,
        unit=t_unit
    )
    st.plotly_chart(fig_trend, use_container_width=True)


# =========================================================
# TAB 2: CLIMATE IMPACT
# =========================================================
with tab2:
    st.markdown("### 🌦️ การวิเคราะห์ความสัมพันธ์ระหว่างสภาพอากาศและผลผลิตเกษตร")
    
    st.info(
        "💡 **คำอธิบายขอบเขตข้อมูลสภาพอากาศ:**\n"
        "- **สถานีตรวจวัดสุพรรณบุรี (Station Data):** เป็นข้อมูลตรวจวัดจริงเฉพาะจุด ณ สถานีอุตุนิยมวิทยาจังหวัดสุพรรณบุรี ครอบคลุม พ.ศ. 2550–2562 (13 ปีต่อเนื่อง) ซึ่งจับคู่กับข้อมูลเกษตรเฉพาะจังหวัดสุพรรณบุรี\n"
        "- **ข้อมูลฝนเชิงพื้นที่ HII (IDW Interpolated):** เป็นค่าประมาณการเชิงพื้นที่รายจังหวัด ครอบคลุม พ.ศ. 2561–2562 (2 ปี ทุกจังหวัด)\n"
        "- **ข้อมูลฝนสะสมภาพรวมประเทศ (1970–2025):** ค่าเฉลี่ยปริมาณฝนสะสมทั้งประเทศรายเดือน"
    )

    climate_scope = st.radio(
        "เลือกมิติการวิเคราะห์สภาพอากาศ (Select Climate Scope):",
        options=[
            "1. เจาะลึกสถานีอุตุนิยมวิทยาสุพรรณบุรี (พ.ศ. 2550–2562, 13 ปีต่อเนื่อง)",
            "2. การวิเคราะห์ฝนเชิงพื้นที่ 77 จังหวัดทั่วประเทศ (พ.ศ. 2561–2562, สสน. HII)",
            "3. แนวโน้มฝนสะสมเฉลี่ยระดับประเทศระยะยาว (พ.ศ. 2550–2562, สสน. HII)"
        ],
        horizontal=True
    )

    if "1. เจาะลึกสถานีอุตุนิยมวิทยาสุพรรณบุรี" in climate_scope:
        # Suphan Buri Station Analysis
        joined_suphan = join_climate_agriculture_suphan(
            df_ag=df_ag,
            df_climate_suphan=df_suphan,
            product=st.session_state.selected_product
        )

        if joined_suphan.empty:
            st.warning(f"ไม่มีข้อมูลการผลิตของสินค้า '{st.session_state.selected_product}' ในจังหวัดสุพรรณบุรี ในช่วง พ.ศ. 2550–2562")
        else:
            # Suphan Climate KPIs
            mean_rain = joined_suphan["rainfall_mm"].mean()
            mean_rain_days = joined_suphan["rainy_days"].mean()
            mean_temp = joined_suphan["temp_avg_c"].mean()
            max_temp = joined_suphan["temp_max_c"].max()
            mean_yield = calculate_weighted_yield(joined_suphan)

            col_ckpi1, col_ckpi2, col_ckpi3, col_ckpi4, col_ckpi5 = st.columns(5)
            with col_ckpi1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">ฝนสะสมเฉลี่ย (Rainfall)</div>
                    <div class="kpi-value">{mean_rain:,.1f}<span class="kpi-unit">มม./ปี</span></div>
                    <div class="kpi-sub">สถานีสุพรรณบุรี (13 ปี)</div>
                </div>
                """, unsafe_allow_html=True)

            with col_ckpi2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">จำนวนวันฝนตก (Rainy Days)</div>
                    <div class="kpi-value">{mean_rain_days:,.0f}<span class="kpi-unit">วัน/ปี</span></div>
                    <div class="kpi-sub">เฉลี่ยต่อปี</div>
                </div>
                """, unsafe_allow_html=True)

            with col_ckpi3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">อุณหภูมิเฉลี่ย (Avg Temp)</div>
                    <div class="kpi-value">{mean_temp:.2f}<span class="kpi-unit">°C</span></div>
                    <div class="kpi-sub">เฉลี่ยทั้งปี</div>
                </div>
                """, unsafe_allow_html=True)

            with col_ckpi4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">อุณหภูมิสูงสุดที่พบ (Max Temp)</div>
                    <div class="kpi-value">{max_temp:.1f}<span class="kpi-unit">°C</span></div>
                    <div class="kpi-sub">จุดสูงสุดในรอบ 13 ปี</div>
                </div>
                """, unsafe_allow_html=True)

            with col_ckpi5:
                yield_txt = f"{mean_yield:,.1f}" if not np.isnan(mean_yield) else "N/A"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Yield ถ่วงน้ำหนัก ({st.session_state.selected_product})</div>
                    <div class="kpi-value">{yield_txt}<span class="kpi-unit">กก./ไร่</span></div>
                    <div class="kpi-sub">จ.สุพรรณบุรี</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Scatter Plots & Correlation
            col_scat1, col_scat2 = st.columns(2)

            with col_scat1:
                # Rainfall vs Yield
                stats_rain = calculate_bivariate_stats(joined_suphan["rainfall_mm"], joined_suphan["calc_yield_per_rai"])
                r_val_rain = stats_rain["pearson_r"] if stats_rain["is_valid"] else None
                fig_rain_scat = create_bivariate_scatter_chart(
                    joined_suphan,
                    x_col="rainfall_mm",
                    y_col="calc_yield_per_rai",
                    title="ปริมาณน้ำฝน vs ผลผลิตต่อไร่ (Rainfall vs Yield)",
                    x_label="ปริมาณน้ำฝน",
                    y_label="ผลผลิตต่อไร่",
                    x_unit="มม.",
                    y_unit="กก./ไร่",
                    correlation_val=r_val_rain
                )
                st.plotly_chart(fig_rain_scat, use_container_width=True)

            with col_scat2:
                # Temperature vs Yield
                stats_temp = calculate_bivariate_stats(joined_suphan["temp_avg_c"], joined_suphan["calc_yield_per_rai"])
                r_val_temp = stats_temp["pearson_r"] if stats_temp["is_valid"] else None
                fig_temp_scat = create_bivariate_scatter_chart(
                    joined_suphan,
                    x_col="temp_avg_c",
                    y_col="calc_yield_per_rai",
                    title="อุณหภูมิเฉลี่ย vs ผลผลิตต่อไร่ (Avg Temp vs Yield)",
                    x_label="อุณหภูมิเฉลี่ย",
                    y_label="ผลผลิตต่อไร่",
                    x_unit="°C",
                    y_unit="กก./ไร่",
                    correlation_val=r_val_temp
                )
                st.plotly_chart(fig_temp_scat, use_container_width=True)

            # Correlation Heatmap & Interpretation Panel
            col_heat, col_interp = st.columns([1, 1.2])

            with col_heat:
                climate_vars = ["rainfall_mm", "rainy_days", "temp_avg_c", "temp_max_c", "temp_min_c", "rh_avg", "calc_yield_per_rai", "production_ton"]
                friendly_labels = {
                    "rainfall_mm": "ปริมาณฝน",
                    "rainy_days": "จำนวนวันฝนตก",
                    "temp_avg_c": "อุณหภูมิเฉลี่ย",
                    "temp_max_c": "อุณหภูมิสูงสุด",
                    "temp_min_c": "อุณหภูมิต่ำสุด",
                    "rh_avg": "ความชื้นเฉลี่ย",
                    "calc_yield_per_rai": "ผลผลิตต่อไร่",
                    "production_ton": "ผลผลิตรวม"
                }
                sub_corr_df = joined_suphan[climate_vars].rename(columns=friendly_labels).corr()
                fig_heat = create_correlation_heatmap(sub_corr_df, title="เมทริกซ์สหสัมพันธ์ (Correlation Heatmap: สุพรรณบุรี)")
                st.plotly_chart(fig_heat, use_container_width=True)

            with col_interp:
                st.markdown("#### 📝 การแปลความหมายผลลัพธ์ (Scientific Interpretation)")
                interp_text = generate_climate_interpretation(
                    bivariate_result=stats_rain,
                    crop_name=st.session_state.selected_product,
                    climate_var_name="ปริมาณน้ำฝนรายปี (Rainfall)",
                    ag_var_name="ผลผลิตต่อไร่ (Yield)",
                    geo_scope="สถานีอุตุนิยมวิทยาจังหวัดสุพรรณบุรี",
                    year_range="2550–2562"
                )
                st.markdown(interp_text)

            # Traceability Data Table
            with st.expander("🔍 ตารางข้อมูลตรวจวัดที่จับคู่กัน (Matched Traceability Dataset - 13 ปี)"):
                trace_cols = ["year", "product", "province", "rainfall_mm", "rainy_days", "temp_avg_c", "temp_max_c", "production_ton", "area_harvest_rai", "calc_yield_per_rai"]
                st.dataframe(joined_suphan[trace_cols].style.format({
                    "rainfall_mm": "{:,.1f}",
                    "rainy_days": "{:.0f}",
                    "temp_avg_c": "{:.2f}",
                    "temp_max_c": "{:.1f}",
                    "production_ton": "{:,.0f}",
                    "area_harvest_rai": "{:,.0f}",
                    "calc_yield_per_rai": "{:,.1f}"
                }), use_container_width=True)

    elif "2. การวิเคราะห์ฝนเชิงพื้นที่ 77 จังหวัด" in climate_scope:
        # Nationwide 77 provinces with HII Spatial Rain
        st.markdown("#### 🌐 การวิเคราะห์ฝนเชิงพื้นที่ HII (IDW) เทียบผลผลิตเกษตร 77 จังหวัด (ปี พ.ศ. 2561–2562)")
        
        joined_nation = join_climate_agriculture_nationwide(
            df_ag=df_ag,
            df_rain_annual=df_rain_annual,
            df_temp_annual=df_temp_annual,
            year=st.session_state.selected_year if st.session_state.selected_year in [2561, 2562] else 2562,
            product=st.session_state.selected_product
        )

        if joined_nation.empty:
            st.warning("ไม่มีข้อมูลการเพาะปลูกที่จับคู่กับฝนเชิงพื้นที่ในปีที่เลือก")
        else:
            col_n1, col_n2 = st.columns(2)
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
                st.markdown("#### 📝 การแปลผลระดับจังหวัดทั่วประเทศ")
                interp_nat = generate_climate_interpretation(
                    bivariate_result=stats_nat_rain,
                    crop_name=st.session_state.selected_product,
                    climate_var_name="ปริมาณฝนเชิงพื้นที่ HII",
                    ag_var_name="ผลผลิตต่อไร่ (Yield)",
                    geo_scope=f"77 จังหวัดทั่วประเทศ (ปี {joined_nation['year'].iloc[0]})",
                    year_range=str(joined_nation['year'].iloc[0])
                )
                st.markdown(interp_nat)

            with st.expander("🔍 ตารางข้อมูลฝนเชิงพื้นที่และผลผลิต 77 จังหวัด"):
                st.dataframe(joined_nation[["province", "year", "product", "rainfall_annual_mm", "production_ton", "area_harvest_rai", "calc_yield_per_rai"]], use_container_width=True)

    else:
        # National Long-Term Rain Trend
        st.markdown("#### 🌧️ ปริมาณฝนเฉลี่ยสะสมทั้งประเทศรายปี vs ผลผลิตเกษตรภาพรวม (พ.ศ. 2550–2562)")
        
        # Merge national rain with national agricultural production
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

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            stats_long = calculate_bivariate_stats(joined_nat_long["rainfall_annual_mm"], joined_nat_long["production_ton"])
            fig_long = create_bivariate_scatter_chart(
                joined_nat_long,
                x_col="rainfall_annual_mm",
                y_col="production_ton",
                title=f"ฝนสะสมภาพรวมประเทศ vs ผลผลิตรวม ({st.session_state.selected_product})",
                x_label="ฝนสะสมเฉลี่ยทั้งประเทศ",
                y_label="ผลผลิตรวม",
                x_unit="มม.",
                y_unit="ตัน",
                correlation_val=stats_long["pearson_r"] if stats_long["is_valid"] else None
            )
            st.plotly_chart(fig_long, use_container_width=True)

        with col_l2:
            st.markdown("#### 📝 การแปลผลแนวโน้มฝนระดับประเทศระยะยาว")
            interp_long = generate_climate_interpretation(
                bivariate_result=stats_long,
                crop_name=st.session_state.selected_product,
                climate_var_name="ฝนสะสมเฉลี่ยทั้งประเทศ (HII)",
                ag_var_name="ผลผลิตรวมทั้งประเทศ (Production)",
                geo_scope="ภาพรวมทั้งประเทศ",
                year_range="2550–2562"
            )
            st.markdown(interp_long)


# =========================================================
# TAB 3: PRODUCTION & RISK
# =========================================================
with tab3:
    st.markdown("### ⚠️ การวิเคราะห์ประสิทธิภาพการผลิต ความผันผวน และสัญญาณความเสี่ยง")

    # Risk Controls
    col_rctrl1, col_rctrl2 = st.columns([1, 2])
    with col_rctrl1:
        threshold = st.slider(
            "🎚️ เกณฑ์อัตราการเก็บเกี่ยวสำหรับตรวจสอบ (Harvest Ratio Review Threshold %):",
            min_value=70.0,
            max_value=98.0,
            value=st.session_state.harvest_threshold,
            step=1.0
        )
        st.session_state.harvest_threshold = threshold

    with col_rctrl2:
        st.info(
            "📌 **หลักการระบุความเสี่ยง:**\n"
            f"- รายการที่อัตราการเก็บเกี่ยวต่ำกว่า **{threshold:.0f}%** จะถูกจัดเป็น **'สัญญาณที่ควรตรวจสอบ (Review Flag)'**\n"
            "- ข้อควรระวัง: สัญญาณนี้สะท้อนเนื้อที่เพาะปลูกที่ไม่ได้เก็บเกี่ยว ซึ่งอาจเกิดจากภัยธรรมชาติ โรคแมลง หรือปัจจัยทางเศรษฐกิจ ไม่สามารถสรุปได้ทันทีว่าเป็นภัยพิบัติทางสภาพอากาศ"
        )

    # Risk KPIs
    sub_risk = df_active.copy()
    flagged_records = sub_risk[sub_risk["calc_harvest_ratio"] < threshold]
    flagged_count = len(flagged_records)
    total_unharvested = kpis["unharvested_area_rai"]
    flagged_unharvested = flagged_records["calc_damaged_area"].sum() if not flagged_records.empty else 0.0

    col_rkpi1, col_rkpi2, col_rkpi3, col_rkpi4 = st.columns(4)
    with col_rkpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">รายการที่พบสัญญาณตรวจทาน (Flagged Items)</div>
            <div class="kpi-value">{flagged_count:,}<span class="kpi-unit">รายการ</span></div>
            <div class="kpi-sub">อัตราเก็บเกี่ยว < {threshold:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_rkpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">เนื้อที่ไม่ได้เก็บเกี่ยวรวม (Unharvested)</div>
            <div class="kpi-value">{total_unharvested:,.0f}<span class="kpi-unit">ไร่</span></div>
            <div class="kpi-sub">เนื้อที่เสียหายหรือละทิ้ง</div>
        </div>
        """, unsafe_allow_html=True)

    with col_rkpi3:
        # Calculate CV of Production across years
        cv_prod_val = "N/A"
        if not df_trend_series.empty:
            yearly_prod = df_trend_series.groupby("year")["production_ton"].sum()
            cv_num = (yearly_prod.std() / yearly_prod.mean() * 100.0) if yearly_prod.mean() > 0 else np.nan
            if not np.isnan(cv_num):
                cv_prod_val = f"{cv_num:.1f} %"

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ความผันผวนผลผลิต (Volatility CV)</div>
            <div class="kpi-value">{cv_prod_val}</div>
            <div class="kpi-sub">ความผันแปรตามปี (13 ปี)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_rkpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">สัดส่วนพื้นที่ตรวจทาน (Flagged Area)</div>
            <div class="kpi-value">{flagged_unharvested:,.0f}<span class="kpi-unit">ไร่</span></div>
            <div class="kpi-sub">ในกลุ่มที่หลุดเกณฑ์</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Quadrant Analysis: Planted Area vs Yield
    # ---------------------------------------------------------
    st.markdown("#### 🎯 การวิเคราะห์การจัดตำแหน่งผลิตภาพ (Area vs Yield Quadrants)")
    st.caption("การแบ่ง 4 ส่วน (Quadrants) โดยใช้ค่ามัธยฐาน (Median) ของพื้นที่เพาะปลูกและผลผลิตต่อไร่ เพื่อแยกกลุ่มพืช/จังหวัดที่มีศักยภาพแตกต่างกัน")

    # Use province-level records for current year
    if st.session_state.selected_product == "ทั้งหมด":
        # Group by product
        quad_base = df_year_prov.groupby("product").agg(
            area_plant_rai=("area_plant_rai", "sum"),
            area_harvest_rai=("area_harvest_rai", "sum"),
            production_ton=("production_ton", "sum")
        ).reset_index()
        quad_base["calc_yield_per_rai"] = quad_base["production_ton"] * 1000.0 / quad_base["area_harvest_rai"]
        quad_df, a_med, y_med = classify_area_yield_quadrants(quad_base, area_col="area_plant_rai", yield_col="calc_yield_per_rai", label_col="product")
        fig_quad = create_quadrant_scatter_chart(quad_df, x_col="area_plant_rai", y_col="calc_yield_per_rai", size_col="production_ton", label_col="product", area_median=a_med, yield_median=y_med, title=f"Quadrant พืชเกษตร: พื้นที่เพาะปลูก vs ผลผลิตต่อไร่ (ปี {st.session_state.selected_year})")
        st.plotly_chart(fig_quad, use_container_width=True)
    else:
        # Group by province for selected product
        quad_base = df_year_prov.groupby("province").agg(
            area_plant_rai=("area_plant_rai", "sum"),
            area_harvest_rai=("area_harvest_rai", "sum"),
            production_ton=("production_ton", "sum")
        ).reset_index()
        quad_base["calc_yield_per_rai"] = quad_base["production_ton"] * 1000.0 / quad_base["area_harvest_rai"]
        quad_df, a_med, y_med = classify_area_yield_quadrants(quad_base, area_col="area_plant_rai", yield_col="calc_yield_per_rai", label_col="province")
        fig_quad = create_quadrant_scatter_chart(quad_df, x_col="area_plant_rai", y_col="calc_yield_per_rai", size_col="production_ton", label_col="province", area_median=a_med, yield_median=y_med, title=f"Quadrant จังหวัดที่ปลูก {st.session_state.selected_product} (ปี {st.session_state.selected_year})")
        st.plotly_chart(fig_quad, use_container_width=True)

    # ---------------------------------------------------------
    # YoY Analysis & Volatility (CV) Ranking
    # ---------------------------------------------------------
    st.markdown("---")
    col_yoy, col_cv = st.columns([1, 1])

    with col_yoy:
        st.markdown("#### 📊 การเติบโตรายปี (Year-over-Year Growth)")
        if not df_trend_series.empty:
            trend_annual_agg = df_trend_series.groupby("year").agg(
                production_ton=("production_ton", "sum"),
                area_plant_rai=("area_plant_rai", "sum"),
                area_harvest_rai=("area_harvest_rai", "sum")
            ).reset_index()
            trend_annual_agg["calc_yield_per_rai"] = trend_annual_agg["production_ton"] * 1000.0 / trend_annual_agg["area_harvest_rai"]
            trend_annual_agg["calc_harvest_ratio"] = trend_annual_agg["area_harvest_rai"] / trend_annual_agg["area_plant_rai"] * 100.0
            
            yoy_metric_choice = st.selectbox(
                "เลือกตัวแปรสำหรับดูกราฟ YoY:",
                options=["production_ton", "area_plant_rai", "calc_yield_per_rai"],
                format_func=lambda k: {"production_ton": "ผลผลิตรวม (Production)", "area_plant_rai": "เนื้อที่เพาะปลูก (Planted)", "calc_yield_per_rai": "ผลผลิตต่อไร่ (Yield)"}[k]
            )

            trend_annual_agg["yoy_growth"] = (trend_annual_agg[yoy_metric_choice] - trend_annual_agg[yoy_metric_choice].shift(1)) / trend_annual_agg[yoy_metric_choice].shift(1) * 100.0
            fig_yoy = create_yoy_bar_chart(
                trend_annual_agg,
                year_col="year",
                yoy_col="yoy_growth",
                title=f"การเติบโต YoY (%) ของ {yoy_metric_choice}",
                metric_label=yoy_metric_choice
            )
            st.plotly_chart(fig_yoy, use_container_width=True)

    with col_cv:
        st.markdown("#### 🌪️ การจัดอันดับความผันผวน (Volatility CV %)")
        st.caption("พืชที่มีค่า CV สูงจะมีความเสี่ยงด้านความไม่แน่นอนของผลผลิตในรอบ 13 ปี")
        cv_summary_df = compute_cv_summary(df_ag[df_ag["geo_level"] == "province"], group_col="product", metric_col="production_ton", min_years=5)
        st.dataframe(
            cv_summary_df[["product", "cv_pct", "mean", "std", "n_years"]].rename(columns={
                "product": "สินค้าเกษตร",
                "cv_pct": "CV (%)",
                "mean": "ผลผลิตเฉลี่ย (ตัน)",
                "std": "ส่วนเบี่ยงเบนมาตรฐาน",
                "n_years": "จำนวนปี"
            }).style.format({
                "CV (%)": "{:.1f}%",
                "ผลผลิตเฉลี่ย (ตัน)": "{:,.0f}",
                "ส่วนเบี่ยงเบนมาตรฐาน": "{:,.0f}"
            }),
            height=340,
            use_container_width=True
        )

    # ---------------------------------------------------------
    # Comprehensive Risk Table & CSV Download
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📋 ตารางติดตามความเสี่ยงและตรวจสอบข้อมูล (Risk & Review Table)")

    table_data = df_active.copy()
    table_data["สถานะการตรวจสอบ"] = np.where(table_data["calc_harvest_ratio"] < threshold, "⚠️ สัญญาณตรวจทาน", "ปกติ")
    
    display_cols = [
        "year", "province", "product", "area_plant_rai", "area_harvest_rai",
        "production_ton", "calc_yield_per_rai", "calc_harvest_ratio",
        "calc_damaged_area", "สถานะการตรวจสอบ"
    ]
    rename_display = {
        "year": "ปี พ.ศ.",
        "province": "จังหวัด",
        "product": "สินค้าเกษตร",
        "area_plant_rai": "เนื้อที่เพาะปลูก (ไร่)",
        "area_harvest_rai": "เนื้อที่เก็บเกี่ยว (ไร่)",
        "production_ton": "ผลผลิตรวม (ตัน)",
        "calc_yield_per_rai": "ผลผลิตต่อไร่ (กก./ไร่)",
        "calc_harvest_ratio": "อัตราเก็บเกี่ยว (%)",
        "calc_damaged_area": "พื้นที่ไม่ได้เก็บเกี่ยว (ไร่)"
    }
    
    show_table = table_data[display_cols].rename(columns=rename_display)

    # CSV Download Button
    csv_bytes = show_table.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 ดาวน์โหลดข้อมูลตารางความเสี่ยง (Download CSV)",
        data=csv_bytes,
        file_name=f"risk_monitoring_{st.session_state.selected_year}_{st.session_state.selected_province}.csv",
        mime="text/csv"
    )

    st.dataframe(
        show_table.style.format({
            "เนื้อที่เพาะปลูก (ไร่)": "{:,.0f}",
            "เนื้อที่เก็บเกี่ยว (ไร่)": "{:,.0f}",
            "ผลผลิตรวม (ตัน)": "{:,.0f}",
            "ผลผลิตต่อไร่ (กก./ไร่)": "{:,.1f}",
            "อัตราเก็บเกี่ยว (%)": "{:,.1f}%",
            "พื้นที่ไม่ได้เก็บเกี่ยว (ไร่)": "{:,.0f}"
        }),
        height=380,
        use_container_width=True
    )

st.markdown("---")
st.caption("🌾 **Thai Agriculture & Climate Intelligence Dashboard** | ข้อมูลสถิติการผลิตสินค้าเกษตรรายปี พ.ศ. 2550–2562 (สศก.) | ข้อมูลอุตุนิยมวิทยาและฝนเชิงพื้นที่ (สสน. HII & กรมอุตุนิยมวิทยา) | วิจัยและพัฒนาสำหรับการนำเสนอผลงานทางวิชาการ")
