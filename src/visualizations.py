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
    selected_item: Optional[str] = None
) -> go.Figure:
    """Horizontal bar chart for rankings (Top 10 provinces or crops)."""
    fig = go.Figure()
    if df.empty:
        apply_standard_layout(
            fig,
            title=title,
            annotations=[dict(text="ไม่มีข้อมูลสำหรับตัวกรองที่เลือก", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    plot_df = df.sort_values(by=x_col, ascending=True).copy()
    
    bar_colors = []
    for item in plot_df[y_col]:
        if selected_item and item == selected_item:
            bar_colors.append(COLOR_WARNING)
        else:
            bar_colors.append(COLOR_FOREST_MED)

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
    text_col: Optional[str] = "year",
    correlation_val: Optional[float] = None
) -> go.Figure:
    """Scatter plot of Climate vs Agriculture with OLS linear trendline."""
    clean_df = df.dropna(subset=[x_col, y_col]).copy()
    if len(clean_df) < 2:
        fig = go.Figure()
        apply_standard_layout(
            fig,
            title=title,
            annotations=[dict(text="จำนวนข้อมูลที่จับคู่ได้ไม่เพียงพอสำหรับสร้าง Scatter Plot", showarrow=False, font=dict(size=14, color=COLOR_TEXT_MUTED))]
        )
        return fig

    fig = px.scatter(
        clean_df,
        x=x_col,
        y=y_col,
        text=text_col if text_col in clean_df.columns else None,
        trendline="ols",
        trendline_color_override=COLOR_WARNING,
        color_discrete_sequence=[COLOR_FOREST_DARK]
    )

    fig.update_traces(
        marker=dict(size=10, opacity=0.85, line=dict(color="white", width=1.5)),
        textposition="top center",
        hovertemplate=(
            f"<b>ปี พ.ศ. %{{text}}</b><br>"
            f"{x_label}: <b>%{{x:,.2f}} {x_unit}</b><br>"
            f"{y_label}: <b>%{{y:,.2f}} {y_unit}</b><extra></extra>"
        )
    )

    subtitle = f" (Pearson r = {correlation_val:+.3f})" if correlation_val is not None and not np.isnan(correlation_val) else ""
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
