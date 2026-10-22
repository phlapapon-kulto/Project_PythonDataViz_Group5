"""
Visualization module for Thai Agriculture & Climate Intelligence Dashboard.
Implements Plotly-based charts adhering to professional visual design standards,
natural green/cream color palette, clear tooltips, units, and responsive layouts.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Color Palette Constants
COLOR_FOREST_DARK = "#1B4D3E"
COLOR_FOREST_MED = "#2D6A4F"
COLOR_SAGE = "#52B788"
COLOR_MINT = "#74C69D"
COLOR_LIGHT_BG = "#F8FBF8"
COLOR_CARD_BG = "#FFFFFF"
COLOR_TEXT_MAIN = "#1E293B"
COLOR_TEXT_MUTED = "#64748B"
COLOR_BORDER = "#E2E8F0"
COLOR_WARNING = "#E07A5F"
COLOR_DANGER = "#D9534F"
COLOR_GOLD = "#E9C46A"
COLOR_NO_DATA = "#E2E8F0"

# Standard Plotly Layout Config
BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color=COLOR_TEXT_MAIN, size=13),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="sans-serif")
)


def apply_standard_layout(fig: go.Figure, **custom_kwargs) -> go.Figure:
    """Safely apply base layout parameters with custom overrides without keyword collisions."""
    layout_args = dict(BASE_LAYOUT)
    layout_args.update(custom_kwargs)
    fig.update_layout(**layout_args)
    return fig


def create_thailand_map(
    df_data: pd.DataFrame,
    geojson_data: Dict[str, Any],
    metric_col: str,
    metric_label: str,
    unit: str,
    year: int,
    product_name: str = "ทั้งหมด",
    selected_province: Optional[str] = None
) -> go.Figure:
    """
    Generate an interactive choropleth map of Thailand showing province-level agricultural or climate metrics.
    Provinces with no records are displayed distinctly in gray (No Data).
    """
    all_provinces = [f["properties"]["pro_th"] for f in geojson_data["features"]]
    base_df = pd.DataFrame({"province": all_provinces})
    
    merged = pd.merge(base_df, df_data[["province", metric_col]], on="province", how="left")
    
    hover_texts = []
    for _, row in merged.iterrows():
        p = row["province"]
        val = row[metric_col]
        if pd.isna(val):
            hover_texts.append(f"<b>จังหวัด{p}</b><br>ปี: พ.ศ. {year}<br>สินค้า: {product_name}<br>สถานะ: <i>ไม่มีข้อมูล (No Data)</i>")
        else:
            if "อัตรา" in metric_label or "%" in unit:
                formatted_val = f"{val:,.2f} %"
            elif val >= 1000:
                formatted_val = f"{val:,.0f} {unit}"
            else:
                formatted_val = f"{val:,.2f} {unit}"
            hover_texts.append(
                f"<b>จังหวัด{p}</b><br>"
                f"ปี: พ.ศ. {year}<br>"
                f"สินค้า: {product_name}<br>"
                f"{metric_label}: <b>{formatted_val}</b>"
            )
            
    merged["hover_text"] = hover_texts
    
    colorscale = "YlGn" if "Yield" in metric_label or "ฝน" in metric_label or "ผลผลิต" in metric_label else "Viridis"
    if "อัตรา" in metric_label:
        colorscale = "RdYlGn"

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        geojson=geojson_data,
        locations=merged["province"],
        z=merged[metric_col],
        featureidkey="properties.pro_th",
        colorscale=colorscale,
        text=merged["hover_text"],
        hoverinfo="text",
        marker_line_color="#FFFFFF",
        marker_line_width=0.6,
        colorbar=dict(
            title=dict(text=f"<b>{metric_label}</b><br>({unit})", font=dict(size=12)),
            thickness=14,
            len=0.75,
            x=0.98,
            y=0.5
        )
    ))

    if selected_province and selected_province in all_provinces:
        sel_mask = merged["province"] == selected_province
        fig.add_trace(go.Choropleth(
            geojson=geojson_data,
            locations=[selected_province],
            z=[merged.loc[sel_mask, metric_col].values[0] if not merged.loc[sel_mask, metric_col].isna().all() else 0],
            featureidkey="properties.pro_th",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False,
            marker_line_color=COLOR_WARNING,
            marker_line_width=3.5,
            hoverinfo="skip"
        ))

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator"
    )
    
    title_text = f"<b>การกระจายตัวเชิงพื้นที่: {metric_label}</b> | ปี พ.ศ. {year} | สินค้า: {product_name}"
    if selected_province:
        title_text += f" <span style='color:{COLOR_WARNING};'>(เลือก: จังหวัด{selected_province})</span>"

    apply_standard_layout(
        fig,
        title=dict(text=title_text, font=dict(size=15, color=COLOR_FOREST_DARK)),
        height=620,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_ranking_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    unit: str,
    selected_item: Optional[str] = None,
    is_bottom: bool = False
) -> go.Figure:
    """Horizontal bar chart for rankings (Top 10 or Bottom 10 provinces or crops)."""
    fig = go.Figure()
    if df.empty:
        apply_standard_layout(
            fig,
            title=title,
            annotations=[dict(text="ไม่มีข้อมูลสำหรับตัวกรองที่เลือก", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    # Plotly horizontal bar puts bottom row of df at the top of the chart.
    # For Top ranking: ascending=True puts highest at the top.
    # For Bottom ranking: ascending=False puts lowest at the top.
    plot_df = df.sort_values(by=x_col, ascending=not is_bottom).copy() if is_bottom else df.sort_values(by=x_col, ascending=True).copy()
    
    default_bar_color = COLOR_WARNING if is_bottom else COLOR_FOREST_MED
    bar_colors = []
    for item in plot_df[y_col]:
        if selected_item and item == selected_item:
            bar_colors.append(COLOR_GOLD if is_bottom else COLOR_WARNING)
        else:
            bar_colors.append(default_bar_color)

    text_labels = []
    for v in plot_df[x_col]:
        if pd.isna(v):
            text_labels.append("N/A")
        elif v >= 1000:
            text_labels.append(f" {v:,.0f} {unit}")
        else:
            text_labels.append(f" {v:,.2f} {unit}")

    fig.add_trace(go.Bar(
        x=plot_df[x_col],
        y=plot_df[y_col],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="#FFFFFF", width=0.8)),
        text=text_labels,
        textposition="outside",
        hovertemplate=f"<b>%{{y}}</b><br>{x_label}: <b>%{{x:,.2f}} {unit}</b><extra></extra>"
    ))

    apply_standard_layout(
        fig,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color=COLOR_FOREST_DARK)),
        xaxis=dict(title=f"{x_label} ({unit})", showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title="", showgrid=False),
        height=380,
        margin=dict(l=10, r=60, t=40, b=30)
    )
    return fig



def create_trend_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str],
    title: str,
    y_label: str,
    unit: str
) -> go.Figure:
    """Time series line chart with markers and clean annotations."""
    if df.empty:
        fig = go.Figure()
        apply_standard_layout(
            fig,
            title=title,
            annotations=[dict(text="ไม่มีข้อมูลสำหรับตัวกรองที่เลือก", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    if color_col and color_col in df.columns:
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            markers=True,
            color_discrete_sequence=[COLOR_FOREST_MED, COLOR_SAGE, COLOR_WARNING, "#2A9D8F", "#E76F51", "#457B9D"]
        )
    else:
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            markers=True,
            color_discrete_sequence=[COLOR_FOREST_DARK]
        )
        fig.update_traces(
            line=dict(width=2.5),
            marker=dict(size=7, color=COLOR_FOREST_DARK, line=dict(color="white", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(45, 106, 79, 0.08)"
        )

    fig.update_traces(
        hovertemplate=f"<b>ปี พ.ศ. %{{x}}</b><br>{y_label}: <b>%{{y:,.2f}} {unit}</b><extra></extra>"
    )

    apply_standard_layout(
        fig,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color=COLOR_FOREST_DARK)),
        xaxis=dict(title="ปี พ.ศ.", tickmode="linear", dtick=1, showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title=f"{y_label} ({unit})", showgrid=True, gridcolor="#E2E8F0"),
        height=380,
        margin=dict(l=20, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_bivariate_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    x_unit: str,
    y_unit: str,
    text_col: Optional[str] = "province",
    correlation_val: Optional[float] = None,
    p_val: Optional[float] = None,
    n_samples: Optional[int] = None
) -> go.Figure:
    """Scatter plot of Climate vs Agriculture with OLS linear trendline.
    Hides dot text labels to prevent canvas clutter, showing province/year cleanly on hover.
    """
    clean_df = df.dropna(subset=[x_col, y_col]).copy()
    if len(clean_df) < 2:
        fig = go.Figure()
        apply_standard_layout(
            fig,
            title=title,
            annotations=[dict(text="จำนวนข้อมูลที่จับคู่ได้ไม่เพียงพอสำหรับสร้าง Scatter Plot", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    hover_col = text_col if (text_col and text_col in clean_df.columns) else None

    # Do not set text=hover_col to prevent permanent text markers cluttering the plot canvas
    fig = px.scatter(
        clean_df,
        x=x_col,
        y=y_col,
        hover_name=hover_col,
        trendline="ols",
        trendline_color_override=COLOR_WARNING,
        color_discrete_sequence=[COLOR_FOREST_DARK]
    )

    if hover_col == "province":
        hover_header = "<b>จังหวัด%{hovertext}</b>"
    elif hover_col == "year":
        hover_header = "<b>ปี พ.ศ. %{hovertext}</b>"
    elif hover_col:
        hover_header = "<b>%{hovertext}</b>"
    else:
        hover_header = ""

    fig.update_traces(
        marker=dict(size=9, opacity=0.85, line=dict(color="white", width=1.2)),
        hovertemplate=(
            f"{hover_header}<br>"
            f"{x_label}: <b>%{{x:,.2f}} {x_unit}</b><br>"
            f"{y_label}: <b>%{{y:,.2f}} {y_unit}</b><extra></extra>"
        )
    )

    stat_parts = []
    if correlation_val is not None and not np.isnan(correlation_val):
        stat_parts.append(f"Pearson r = {correlation_val:+.3f}")
    if p_val is not None and not np.isnan(p_val):
        stat_parts.append(f"p = {p_val:.4f}")
    if n_samples is not None:
        stat_parts.append(f"n = {n_samples}")
    elif len(clean_df) > 0:
        stat_parts.append(f"n = {len(clean_df)}")

    subtitle = f" ({', '.join(stat_parts)})" if stat_parts else ""
    apply_standard_layout(
        fig,
        title=dict(text=f"<b>{title}{subtitle}</b>", font=dict(size=14, color=COLOR_FOREST_DARK)),
        xaxis=dict(title=f"{x_label} ({x_unit})", showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title=f"{y_label} ({y_unit})", showgrid=True, gridcolor="#E2E8F0"),
        height=400,
        margin=dict(l=20, r=20, t=40, b=30)
    )
    return fig


def create_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "เมทริกซ์สหสัมพันธ์ (Correlation Matrix)"
) -> go.Figure:
    """Heatmap of correlation matrix with numerical annotations."""
    fig = go.Figure()
    if corr_df.empty:
        apply_standard_layout(
            fig,
            title=title,
            annotations=[dict(text="ไม่มีข้อมูลเพียงพอสำหรับสร้าง Correlation Heatmap", showarrow=False)]
        )
        return fig

    z_vals = corr_df.values
    x_labels = list(corr_df.columns)
    y_labels = list(corr_df.index)

    text_vals = []
    for row in z_vals:
        row_text = []
        for val in row:
            if pd.isna(val):
                row_text.append("N/A")
            else:
                row_text.append(f"{val:+.2f}")
        text_vals.append(row_text)

    fig.add_trace(go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        text=text_vals,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorscale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Correlation (r)", thickness=12, len=0.8)
    ))

    apply_standard_layout(
        fig,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color=COLOR_FOREST_DARK)),
        height=380,
        margin=dict(l=80, r=40, t=40, b=40)
    )
    return fig


def create_quadrant_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str,
    label_col: str,
    area_median: float,
    yield_median: float,
    title: str = "การกระจายตัวพื้นที่เพาะปลูก vs ผลผลิตต่อไร่ (Area vs Yield Quadrants)"
) -> go.Figure:
    """Scatter plot dividing crops or provinces into 4 quadrants based on median values."""
    if df.empty:
        fig = go.Figure()
        apply_standard_layout(
            fig,
            title=title,
            annotations=[dict(text="ไม่มีข้อมูลสำหรับตัวกรองที่เลือก", showarrow=False)]
        )
        return fig

    color_map = {
        "Q1: พื้นที่มาก - Yield สูง": "#2D6A4F",
        "Q2: พื้นที่น้อย - Yield สูง": "#2A9D8F",
        "Q3: พื้นที่น้อย - Yield ต่ำ": "#8D99AE",
        "Q4: พื้นที่มาก - Yield ต่ำ": "#E07A5F"
    }

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color="quadrant",
        color_discrete_map=color_map,
        hover_name=label_col,
        size_max=35
    )

    fig.add_vline(
        x=area_median,
        line_dash="dash",
        line_color="#94A3B8",
        line_width=1.5,
        annotation_text=f"Median พื้นที่: {area_median:,.0f} ไร่",
        annotation_position="top left",
        annotation_font_size=11
    )

    fig.add_hline(
        y=yield_median,
        line_dash="dash",
        line_color="#94A3B8",
        line_width=1.5,
        annotation_text=f"Median Yield: {yield_median:,.0f} กก./ไร่",
        annotation_position="bottom right",
        annotation_font_size=11
    )

    fig.update_traces(
        marker=dict(opacity=0.85, line=dict(color="white", width=1)),
        hovertemplate=(
            f"<b>%{{hovertext}}</b><br>"
            f"กลุ่ม: %{{marker.color}}<br>"
            f"พื้นที่เพาะปลูก: <b>%{{x:,.0f}} ไร่</b><br>"
            f"ผลผลิตต่อไร่: <b>%{{y:,.1f}} กก./ไร่</b><br>"
            f"ผลผลิตรวม: <b>%{{marker.size:,.0f}} ตัน</b><extra></extra>"
        )
    )

    apply_standard_layout(
        fig,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color=COLOR_FOREST_DARK)),
        xaxis=dict(title="เนื้อที่เพาะปลูก (ไร่)", showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title="ผลผลิตต่อไร่ (กก./ไร่)", showgrid=True, gridcolor="#E2E8F0"),
        height=480,
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_yoy_bar_chart(
    df: pd.DataFrame,
    year_col: str,
    yoy_col: str,
    title: str,
    metric_label: str
) -> go.Figure:
    """Bar chart showing YoY changes with green for positive and coral for negative."""
    clean_df = df.dropna(subset=[year_col, yoy_col]).copy()
    if clean_df.empty:
        fig = go.Figure()
        apply_standard_layout(
            fig,
            title=title,
            annotations=[dict(text="ไม่มีข้อมูลเปรียบเทียบ YoY (ต้องมีข้อมูลปีก่อนหน้า)", showarrow=False)]
        )
        return fig

    bar_colors = np.where(clean_df[yoy_col] >= 0, COLOR_FOREST_MED, COLOR_WARNING)
    
    fig = go.Figure(go.Bar(
        x=clean_df[year_col],
        y=clean_df[yoy_col],
        marker_color=bar_colors,
        text=[f"{v:+.1f}%" for v in clean_df[yoy_col]],
        textposition="outside",
        hovertemplate=f"<b>ปี พ.ศ. %{{x}}</b><br>การเปลี่ยนแปลง YoY ({metric_label}): <b>%{{y:+.2f}}%</b><extra></extra>"
    ))

    apply_standard_layout(
        fig,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color=COLOR_FOREST_DARK)),
        xaxis=dict(title="ปี พ.ศ.", tickmode="linear", dtick=1, showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title="การเปลี่ยนแปลง YoY (%)", showgrid=True, gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#64748B"),
        height=340,
        margin=dict(l=20, r=20, t=40, b=30)
    )
    return fig


def create_price_trend_chart(
    df_price: pd.DataFrame,
    product_name: str,
    unit_mode: str = "baht_per_kg"
) -> go.Figure:
    """
    Time series line chart for official OAE farmgate commodity prices.
    Supports Baht/kg or Baht/ton.
    """
    fig = go.Figure()
    if df_price.empty:
        apply_standard_layout(
            fig,
            title=f"แนวโน้มราคา {product_name} (สศก.)",
            annotations=[dict(text="ไม่มีข้อมูลราคาสำหรับสินค้าที่เลือก", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    # Sort chronologically by year, month, week
    plot_df = df_price.sort_values(by=["year", "month", "week"]).copy()
    plot_df["time_label"] = (
        "ปี " + plot_df["year"].astype(str) +
        " ด." + plot_df["month"].astype(str) +
        " ส." + plot_df["week"].astype(str)
    )

    y_col = "price_baht_per_kg" if unit_mode == "baht_per_kg" else "price_baht_per_ton"
    unit_label = "บาท/กก." if unit_mode == "baht_per_kg" else "บาท/ตัน"

    fig.add_trace(go.Scatter(
        x=plot_df["time_label"],
        y=plot_df[y_col],
        mode="lines+markers",
        name=product_name,
        line=dict(color=COLOR_FOREST_MED, width=2.5),
        marker=dict(size=6, color=COLOR_FOREST_DARK),
        hovertemplate=f"<b>%{{x}}</b><br>ราคาหน้าไร่นา: <b>%{{y:,.2f}} {unit_label}</b><extra></extra>"
    ))

    apply_standard_layout(
        fig,
        title=dict(text=f"<b>แนวโน้มราคาที่เกษตรกรขายได้ที่ไร่นา: {product_name}</b> (สศก.)", font=dict(size=14, color=COLOR_FOREST_DARK)),
        xaxis=dict(title="คาบเวลา (ปี/เดือน/สัปดาห์)", showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title=f"ราคา ({unit_label})", showgrid=True, gridcolor="#E2E8F0"),
        height=380,
        margin=dict(l=20, r=20, t=40, b=40)
    )
    return fig


def create_cost_breakdown_pie(
    cost_dict: Dict[str, float],
    title: str = "โครงสร้างสัดส่วนต้นทุนการผลิต"
) -> go.Figure:
    """
    Donut chart displaying cost breakdown (Seeds, Fertilizer, Water, Labor, Rent, Others).
    """
    fig = go.Figure()
    labels = list(cost_dict.keys())
    values = list(cost_dict.values())

    total = sum(values)
    if total <= 0:
        apply_standard_layout(
            fig,
            title=title,
            annotations=[dict(text="กรุณาระบุต้นทุนอย่างน้อย 1 รายการ", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    colors = ["#2D6A4F", "#52B788", "#74C69D", "#E07A5F", "#E9C46A", "#8D99AE"]

    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>ต้นทุน: <b>%{value:,.0f} บาท</b> (%{percent})<extra></extra>"
    ))

    apply_standard_layout(
        fig,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color=COLOR_FOREST_DARK)),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig


def create_scenario_bar_chart(
    scenarios: List[Dict[str, Any]],
    title: str = "การวิเคราะห์ความอ่อนไหวของกำไรสุทธิ (What-if Scenario Simulation)"
) -> go.Figure:
    """
    Bar chart comparing net profit across price/yield sensitivity scenarios.
    """
    fig = go.Figure()
    if not scenarios:
        apply_standard_layout(fig, title=title)
        return fig

    labels = [f"ราคา {s['price_change']} | ผลผลิต {s['yield_change']}" for s in scenarios]
    profits = [s["profit"] for s in scenarios]
    bar_colors = [COLOR_FOREST_MED if p >= 0 else COLOR_DANGER for p in profits]

    fig.add_trace(go.Bar(
        x=labels,
        y=profits,
        marker=dict(color=bar_colors),
        text=[f"{p:+,.0f} บ." for p in profits],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>กำไร/ขาดทุนสุทธิ: <b>%{y:,.0f} บาท</b><extra></extra>"
    ))

    apply_standard_layout(
        fig,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color=COLOR_FOREST_DARK)),
        xaxis=dict(title="", tickangle=-30, showgrid=False),
        yaxis=dict(title="กำไรสุทธิโดยประมาณ (บาท)", showgrid=True, gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#64748B"),
        height=380,
        margin=dict(l=20, r=20, t=40, b=60)
    )
    return fig



# -------------------------------------------------------------------------
# Crop Production Drivers Visualizations
# -------------------------------------------------------------------------

def create_irrigation_vs_yield_chart(
    df: pd.DataFrame,
    crop_name: str,
    year: int
) -> go.Figure:
    """
    Scatter plot comparing provincial irrigation ratio (%) vs crop yield (kg/rai)
    with OLS trendline and Pearson r statistics.
    """
    fig = go.Figure()
    if df.empty or len(df) < 3:
        apply_standard_layout(
            fig,
            title=f"ความสัมพันธ์ระหว่างสัดส่วนพื้นที่ชลประทานกับผลผลิต: {crop_name} (พ.ศ. {year})",
            annotations=[dict(text="ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    from scipy import stats
    x = df["irrigation_ratio_pct"].values
    y = df["yield_kg_rai"].values
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    x_trend = np.linspace(x.min(), x.max(), 50)
    y_trend = slope * x_trend + intercept

    # Scatter points without permanent labels on dots (clean hover)
    fig.add_trace(go.Scatter(
        x=df["irrigation_ratio_pct"],
        y=df["yield_kg_rai"],
        mode="markers",
        name="จังหวัด",
        marker=dict(size=9, color=COLOR_FOREST_MED, opacity=0.85, line=dict(width=1, color=COLOR_FOREST_DARK)),
        customdata=df[["province", "irrigation_area_rai", "agricultural_area_rai"]],
        hovertemplate="<b>%{customdata[0]}</b><br>สัดส่วนพื้นที่ชลประทาน: <b>%{x:.1f}%</b><br>ผลผลิตต่อไร่: <b>%{y:,.1f} กก./ไร่</b><br>พื้นที่ชลประทาน: %{customdata[1]:,.0f} ไร่<extra></extra>"
    ))

    # OLS Trendline
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode="lines",
        name="แนวโน้มเชิงเส้น (OLS)",
        line=dict(color="#D90429", width=2.5, dash="dash"),
        hoverinfo="skip"
    ))

    p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"
    sub_title = f"Pearson r = {r:+.3f} (p {p_str}, n = {len(df)}) | แหล่งข้อมูล: สศก. และ กรมชลประทาน"

    apply_standard_layout(
        fig,
        title=dict(
            text=f"<b>สัดส่วนพื้นที่ชลประทาน vs ผลผลิตต่อไร่: {crop_name} (พ.ศ. {year})</b><br><span style='font-size:12px; color:#4A5568;'>{sub_title}</span>",
            font=dict(size=14, color=COLOR_FOREST_DARK)
        ),
        xaxis=dict(title="สัดส่วนพื้นที่ชลประทานต่อพื้นที่เกษตร (%)", showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title="ผลผลิตต่อไร่ (กก./ไร่)", showgrid=True, gridcolor="#E2E8F0"),
        height=420,
        margin=dict(l=20, r=20, t=60, b=40)
    )
    return fig


def create_soil_suitability_bar_chart(
    soil_df: pd.DataFrame,
    crop_name: str,
    top_n: int = 15
) -> go.Figure:
    """
    Stacked horizontal bar chart of LDD Agri-Map soil suitability (S1, S2, S3, N).
    """
    fig = go.Figure()
    if soil_df.empty:
        apply_standard_layout(
            fig,
            title=f"การประเมินความเหมาะสมของดิน: {crop_name} (Agri-Map กรมพัฒนาที่ดิน)",
            annotations=[dict(text="ไม่มีข้อมูลความเหมาะสมของดิน", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    sub = soil_df[soil_df["crop"] == crop_name].copy()
    if sub.empty:
        sub = soil_df.copy()

    # Sort by S1+S2 descending
    sub = sub.sort_values(by="suitable_total_s1_s2_pct", ascending=True).tail(top_n)

    fig.add_trace(go.Bar(
        y=sub["province"],
        x=sub["suitable_high_s1_pct"],
        name="S1 (เหมาะสมสูง)",
        orientation="h",
        marker=dict(color="#2D6A4F"),
        hovertemplate="<b>%{y}</b><br>S1 เหมาะสมสูง: %{x:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=sub["province"],
        x=sub["suitable_moderate_s2_pct"],
        name="S2 (เหมาะสมปานกลาง)",
        orientation="h",
        marker=dict(color="#52B788"),
        hovertemplate="<b>%{y}</b><br>S2 เหมาะสมปานกลาง: %{x:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=sub["province"],
        x=sub["suitable_marginal_s3_pct"],
        name="S3 (เหมาะสมน้อย)",
        orientation="h",
        marker=dict(color="#E9C46A"),
        hovertemplate="<b>%{y}</b><br>S3 เหมาะสมน้อย: %{x:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=sub["province"],
        x=sub["not_suitable_n_pct"],
        name="N (ไม่เหมาะสม)",
        orientation="h",
        marker=dict(color="#E07A5F"),
        hovertemplate="<b>%{y}</b><br>N ไม่เหมาะสม: %{x:.1f}%<extra></extra>"
    ))

    apply_standard_layout(
        fig,
        title=dict(
            text=f"<b>การจำแนกความเหมาะสมของดิน (Agri-Map): {crop_name}</b> (15 จังหวัดที่มีความเหมาะสมสูงสุด)",
            font=dict(size=14, color=COLOR_FOREST_DARK)
        ),
        barmode="stack",
        xaxis=dict(title="สัดส่วนพื้นที่ตามชั้นความเหมาะสม (%)", range=[0, 100], showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title="", showgrid=False),
        height=480,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig


def create_cost_structure_breakdown_chart(
    cost_breakdown: Dict[str, float],
    crop_name: str,
    total_cost: float,
    year: int = 2562
) -> go.Figure:
    """
    Donut chart of official OAE production cost structure.
    """
    fig = go.Figure()
    if not cost_breakdown or total_cost <= 0:
        apply_standard_layout(
            fig,
            title=f"โครงสร้างต้นทุนการผลิต: {crop_name} (สศก. {year})",
            annotations=[dict(text="ไม่มีข้อมูลต้นทุน", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    labels = list(cost_breakdown.keys())
    values = list(cost_breakdown.values())
    colors = ["#2D6A4F", "#52B788", "#74C69D", "#E07A5F", "#E9C46A", "#264653", "#8D99AE"]

    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>ต้นทุนเฉลี่ย: <b>%{value:,.0f} บาท/ไร่</b> (%{percent})<extra></extra>"
    ))

    apply_standard_layout(
        fig,
        title=dict(
            text=f"<b>โครงสร้างต้นทุนการผลิตเฉลี่ย: {crop_name}</b> (รวม {total_cost:,.0f} บาท/ไร่, สศก. {year})",
            font=dict(size=14, color=COLOR_FOREST_DARK)
        ),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5)
    )
    return fig


def create_disaster_impact_chart(
    disaster_df: pd.DataFrame,
    year: int,
    top_n: int = 12
) -> go.Figure:
    """
    Grouped bar chart comparing flood vs drought affected area for top impacted provinces.
    """
    fig = go.Figure()
    if disaster_df.empty:
        apply_standard_layout(
            fig,
            title=f"พื้นที่การเกษตรประสบภัยธรรมชาติ (พ.ศ. {year})",
            annotations=[dict(text="ไม่มีข้อมูลพื้นที่ประสบภัย", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    sub = disaster_df[disaster_df["year"] == year].copy()
    if sub.empty:
        sub = disaster_df.copy()

    sub = sub.sort_values(by="total_disaster_area_rai", ascending=False).head(top_n)

    fig.add_trace(go.Bar(
        x=sub["province"],
        y=sub["flood_affected_area_rai"],
        name="อุทกภัย (น้ำท่วม)",
        marker=dict(color="#1D3557"),
        hovertemplate="<b>%{x}</b><br>พื้นที่ประสบอุทกภัย: <b>%{y:,.0f} ไร่</b><extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=sub["province"],
        y=sub["drought_affected_area_rai"],
        name="ภัยแล้ง / ฝนทิ้งช่วง",
        marker=dict(color="#E07A5F"),
        hovertemplate="<b>%{x}</b><br>พื้นที่ประสบภัยแล้ง: <b>%{y:,.0f} ไร่</b><extra></extra>"
    ))

    apply_standard_layout(
        fig,
        title=dict(
            text=f"<b>พื้นที่เกษตรประสบอุทกภัยและภัยแล้ง (พ.ศ. {year})</b> (12 จังหวัดสูงสุด, กรมส่งเสริมการเกษตร/ปภ.)",
            font=dict(size=14, color=COLOR_FOREST_DARK)
        ),
        barmode="group",
        xaxis=dict(title="", tickangle=-30, showgrid=False),
        yaxis=dict(title="พื้นที่ประสบภัย (ไร่)", showgrid=True, gridcolor="#E2E8F0"),
        height=400,
        margin=dict(l=20, r=20, t=50, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig


def create_seasonal_rain_vs_yield_chart(
    df: pd.DataFrame,
    crop_name: str,
    year: int
) -> go.Figure:
    """
    Scatter plot comparing growing season rainfall (May-Oct) vs yield (kg/rai).
    """
    fig = go.Figure()
    if df.empty or len(df) < 3:
        apply_standard_layout(
            fig,
            title=f"ปริมาณฝนช่วงฤดูเพาะปลูก (พ.ค.–ต.ค.) vs ผลผลิต: {crop_name} (พ.ศ. {year})",
            annotations=[dict(text="ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    from scipy import stats
    x = df["rain_growing_season_mm"].values
    y = df["yield_kg_rai"].values
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    x_trend = np.linspace(x.min(), x.max(), 50)
    y_trend = slope * x_trend + intercept

    fig.add_trace(go.Scatter(
        x=df["rain_growing_season_mm"],
        y=df["yield_kg_rai"],
        mode="markers",
        name="จังหวัด",
        marker=dict(size=8, color="#2A9D8F", opacity=0.85, line=dict(width=1, color=COLOR_FOREST_DARK)),
        customdata=df[["province", "rain_annual_mm", "growing_rain_share_pct"]],
        hovertemplate="<b>%{customdata[0]}</b><br>ฝนฤดูปลูก (พ.ค.-ต.ค.): <b>%{x:,.1f} มม.</b><br>ผลผลิตต่อไร่: <b>%{y:,.1f} กก./ไร่</b><br>ฝนทั้งปี: %{customdata[1]:,.1f} มม. (สัดส่วนฝนฤดูปลูก %{customdata[2]:.1f}%)<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode="lines",
        name="แนวโน้มเชิงเส้น (OLS)",
        line=dict(color="#E76F51", width=2.5, dash="dash"),
        hoverinfo="skip"
    ))

    p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"
    sub_title = f"Pearson r = {r:+.3f} (p {p_str}, n = {len(df)}) | แหล่งข้อมูล: สสน. HII และ สศก."

    apply_standard_layout(
        fig,
        title=dict(
            text=f"<b>ฝนช่วงฤดูเพาะปลูก (พ.ค.–ต.ค.) vs ผลผลิตต่อไร่: {crop_name} (พ.ศ. {year})</b><br><span style='font-size:12px; color:#4A5568;'>{sub_title}</span>",
            font=dict(size=14, color=COLOR_FOREST_DARK)
        ),
        xaxis=dict(title="ปริมาณฝนสะสมช่วงฤดูเพาะปลูก (มิลลิเมตร)", showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title="ผลผลิตต่อไร่ (กก./ไร่)", showgrid=True, gridcolor="#E2E8F0"),
        height=420,
        margin=dict(l=20, r=20, t=60, b=40)
    )
    return fig


def create_cost_price_margin_chart(
    cost_summary: Dict[str, Any]
) -> go.Figure:
    """
    Bar comparison chart of Selling Price vs Unit Cost per kg (Profit Margin).
    """
    fig = go.Figure()
    if not cost_summary or not cost_summary.get("has_data", False):
        apply_standard_layout(
            fig,
            title="การเปรียบเทียบราคาขายกับต้นทุนต่อหน่วย",
            annotations=[dict(text="ไม่มีข้อมูลต้นทุนและราคา", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    cost_kg = cost_summary["cost_per_kg"]
    price_kg = cost_summary["avg_selling_price_kg"]
    margin_kg = cost_summary["margin_per_kg"]
    crop = cost_summary["crop"]
    year = cost_summary["year"]

    categories = ["ต้นทุนการผลิตต่อ กก.", "ราคาขายเฉลี่ยหน้าไร่นา", "ส่วนต่างกำไร/ขาดทุนต่อ กก."]
    vals = [cost_kg, price_kg, margin_kg]
    colors = ["#E07A5F", "#2D6A4F", COLOR_FOREST_MED if margin_kg >= 0 else COLOR_DANGER]

    fig.add_trace(go.Bar(
        x=categories,
        y=vals,
        marker=dict(color=colors),
        text=[f"{v:,.2f} บ./กก." for v in vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>มูลค่า: <b>%{y:,.2f} บาท/กก.</b><extra></extra>"
    ))

    apply_standard_layout(
        fig,
        title=dict(
            text=f"<b>การเปรียบเทียบต้นทุนต่อหน่วยกับราคาขาย: {crop} (พ.ศ. {year})</b>",
            font=dict(size=14, color=COLOR_FOREST_DARK)
        ),
        yaxis=dict(title="บาทต่อกิโลกรัม (บาท/กก.)", showgrid=True, gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#64748B"),
        height=360,
        margin=dict(l=20, r=20, t=50, b=40)
    )
    return fig
