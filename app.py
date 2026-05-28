import os
import streamlit as st
import pandas as pd
import numpy as np 
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime

# =================================================================
# --- 0. API Keys Setup ---
# =================================================================
# 设置 Tavily API Key (全局)
os.environ["TAVILY_API_KEY"] = "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"

# =================================================================
# --- 1. Executive Visual Configuration (McKinsey Strategic UI) ---
# =================================================================
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    :root { --cm-blue: #051C2C; --cm-terracotta: #A64B35; --cm-sage: #A4B6B0; --cm-beige: #F8F9FA; --cm-yellow: #FDB913; }
    .main { background-color: #FAFAFA; font-family: 'Inter', sans-serif; }
    
    .header-box {
        background-color: var(--cm-blue);
        color: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 10px 0 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        letter-spacing: 1px;
    }
    
    .filter-container {
        background-color: white;
        padding: 20px 25px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border-top: 4px solid var(--cm-blue);
        margin-bottom: 20px;
    }
    
    div[data-testid="stTabs"] button {
        flex: 1; font-family: 'Inter', sans-serif; font-weight: 600; color: #6C757D;
        background-color: #F8F9FA; border: 1px solid #EAECEF; border-bottom: 2px solid transparent;
        border-radius: 8px 8px 0 0; padding: 14px 10px; margin: 0 4px; font-size: 1.1rem;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: white; color: var(--cm-blue); border-bottom: 4px solid var(--cm-terracotta);
    }
    
    .mckinsey-table { width: 100%; border-collapse: collapse; margin: 5px 0 25px 0; background-color: #ffffff; }
    .th-main { color: white; font-family: 'Inter', sans-serif; font-weight: 600; text-align: center !important; padding: 10px 6px; font-size: 0.95rem; border: 1px solid #ffffff;}
    .th-sub { color: white; font-family: 'Inter', sans-serif; font-weight: 500; text-align: center !important; padding: 8px 4px; font-size: 0.85rem; border: 1px solid #ffffff;}
    .th-dark { background-color: #051C2C; }
    .th-cy { background-color: #112E43; }
    .th-py { background-color: #5C7080; }
    .th-var { background-color: #A64B35; }
    .th-yellow { background-color: #FDB913; color: #051C2C !important; }
    
    .mckinsey-table td { padding: 10px 14px; border: 1px solid #EAECEF; font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #333333; text-align: right; }
    .mckinsey-table td.cell-merged { text-align: center !important; vertical-align: middle !important; background-color: #FAFAFA; font-weight: 600; border-right: 1px solid #D1D5DB !important; border-bottom: 1px solid #EAECEF !important; }
    .mckinsey-table td.cell-detail-left { text-align: left !important; vertical-align: middle !important; }
    
    .td-divider { border-right: 2px solid #CBD5E1 !important; }
    .th-divider { border-right: 2px solid #ffffff !important; }
    
    .subtotal-row { background-color: #F4F7F9 !important; font-weight: 600; }
    .subtotal-row td { color: #051C2C !important; }
    .total-row { background-color: #E2ECF1 !important; font-weight: 700; border-top: 1px solid #051C2C !important; border-bottom: 1px solid #051C2C !important; }
    .total-row td { color: #051C2C !important; }
    .grand-total-row { background-color: #D0DFE7 !important; font-weight: 800; border-top: 2px solid #051C2C !important; border-bottom: 3px double #051C2C !important; }
    .grand-total-row td { color: #051C2C !important; }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# =================================================================
# --- 2. Core Formatting & Hoisted Utility Functions ---
# =================================================================
def format_volume(val):
    if pd.isna(val) or val == 0: return "0"
    if abs(val) >= 1_000_000: return f"{val/1_000_000:,.1f}M"
    elif abs(val) >= 1_000: return f"{val/1_000:,.1f}k"
    return f"{val:,.0f}"

def fmt_val(val, is_pct=False):
    if is_pct:
        if pd.isna(val) or np.isinf(val) or val == 0: return "0.0%"
        sign = "+" if val > 0 else ""
        color = "#28a745" if val >= 0 else "#dc3545"
        return f'<span style="color:{color}; font-weight:700;">{sign}{val*100:.1f}%</span>'
    else:
        if pd.isna(val): return "0"
        return f"{val:,.0f}"

def format_variance_cell(val, is_pct=False):
    if is_pct:
        if pd.isna(val) or np.isinf(val) or val == 0: return "0.0%"
        sign = "+" if val > 0 else ""
        color = "#28a745" if val > 0 else "#dc3545"
        return f'<span style="color:{color}; font-weight:600;">{sign}{val*100:.1f}%</span>'
    else:
        sign = "+" if val > 0 else ""
        color = "#28a745" if val > 0 else "#dc3545"
        return f'<span style="color:{color}; font-weight:600;">{sign}{val:,.2f}</span>'

def custom_metric_card(title, cy_val, py_val, delta_pct, cy_format, py_format):
    delta_color = "#28a745" if delta_pct >= 0 else "#dc3545"
    delta_sign = "+" if delta_pct > 0 else ""
    return f"""
    <div style="background-color: white; border-radius: 8px; padding: 15px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border-top: 3px solid #A64B35; border-left: 4px solid #051C2C; margin-bottom: 15px; height: 100%;">
        <div style="color: #6C757D; font-size: 0.95rem; font-weight: 600; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #051C2C; margin-bottom: 5px;">{cy_format}</div>
        <div style="font-size: 0.9rem; color: #888; font-weight: 500;">
            PY: {py_format} <span style="color: {delta_color}; font-weight: 700; margin-left: 8px;">({delta_sign}{delta_pct:.1f}%)</span>
        </div>
    </div>
    """

def mini_metric_card(title, value, color="#051C2C"):
    return f"""
    <div style="background-color: #F8F9FA; border-radius: 6px; padding: 10px 15px; border-left: 3px solid {color}; margin-bottom: 10px; height: 100%;">
        <div style="color: #6C757D; font-size: 0.8rem; font-weight: 600; margin-bottom: 2px;">{title}</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: {color};">{value}</div>
    </div>
    """

def complex_mini_card(title, tot_val, fit_val, mice_val, curr_sym="", color="#051C2C"):
    return f"""
    <div style="background-color: #F8F9FA; border-radius: 6px; padding: 10px 15px; border-left: 3px solid {color}; margin-bottom: 10px; height: 100%;">
        <div style="color: #6C757D; font-size: 0.8rem; font-weight: 600; margin-bottom: 2px;">{title}</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: {color};">{curr_sym}{tot_val/1e6:.2f}M</div>
        <div style="font-size: 0.75rem; color: #888; margin-top: 3px; font-weight: 500;">
            FIT: {curr_sym}{fit_val/1e6:.2f}M &nbsp;|&nbsp; MICE: {curr_sym}{mice_val/1e6:.2f}M
        </div>
    </div>
    """

def safe_offset(d, years):
    if not isinstance(d, datetime.date): return None
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d + datetime.timedelta(days=365 * years)

# =================================================================
# --- 3. Core Data Cleaning & Strategic Mapping Logic ---
# =================================================================
@st.cache_data
def load_and_clean(file):
    data = pd.read_csv(file, low_memory=False)
    data.columns = [col.strip() for col in data.columns]
    
    mapping = {
        'CONSUMPTION_CALENDAR[Month Name]': 'Month',
        'CONSUMPTION_CALENDAR[Consumption_month_num]': 'Month_Num',
        'CONSUMPTION_CALENDAR[Consumption_year]': 'Year',
        'CONSUMPTION_CALENDAR[Consumption_date]': 'Cons_Date',
        'SALES_CALENDAR[Sales_date]': 'Sales_Date',
        'REF_SALES_MARKET[Market]': 'Market',
        'REF_DESTINATION[Resort]': 'Resort',
        'REF_CML_AGENCY[Group_TA_cml]': 'TA_Group',
        'REF_DESTINATION[Destination type Asia]': 'Dest_Type',
        '[BVSTS___final]': 'BV_Euro',        
        '[BVSTS_loc_final]': 'BV_Locale',  
        '[HN_final]': 'HN',
        'REF_SALES_SEGMENT[Segment_type]': 'Segment',
        'Channel Group': 'Channel_Group',
        'Team Group': 'Team_Group',
        'reChannel': 'reChannel'
    }
    data.rename(columns=mapping, inplace=True, errors='ignore')
    
    if 'BV_Euro' not in data.columns: data['BV_Euro'] = 0.0
    if 'BV_Locale' not in data.columns: data['BV_Locale'] = 0.0
    if 'HN' not in data.columns: data['HN'] = 0.0
    
    str_cols = ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month', 'Segment', 'Channel_Group', 'Team_Group', 'reChannel']
    for col in str_cols:
        if col in data.columns: 
            data[col] = data[col].astype(str).str.strip()

    if 'Dest_Type' in data.columns:
        data['Dest_Type'] = data['Dest_Type'].str.replace('Interzone mountain', 'IZ Mountain', case=False)
        data['Dest_Type'] = data['Dest_Type'].str.replace('Interzone sun', 'IZ Sun', case=False)
        data = data[~data['Dest_Type'].str.lower().str.contains('other', na=False)]

    for col in ['BV_Euro', 'BV_Locale', 'HN']:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '').replace(' ', ''), errors='coerce').fillna(0)
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    
    if 'Sales_Date' in data.columns: data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    if 'Cons_Date' in data.columns: data['Cons_Date'] = pd.to_datetime(data['Cons_Date'], errors='coerce')
    return data

def sanitize_channels(df_mat):
    if 'Channel_Group' not in df_mat.columns: return df_mat
    df_mat['Channel_Group'] = df_mat['Channel_Group'].astype(str).str.strip()
    
    is_dir = df_mat['Channel_Group'].str.lower().isin(['direct', 'semi-direct', 'mice-corp', 'mice corp', 'mice'])
    is_indir = df_mat['Channel_Group'].str.lower().isin(['indirect', 'indirect-ctrip', 'indirect - ctrip', 'indirect-meituan', 'indirect - meituan'])
    
    if 'Segment' in df_mat.columns:
        me_mask = df_mat['Segment'].str.lower().str.contains('m&e|mice', na=False)
        is_dir = is_dir | me_mask
        df_mat.loc[me_mask, 'Team_Group'] = 'MICE'
        df_mat.loc[me_mask, 'reChannel'] = 'MICE'
    
    df_mat.loc[is_dir, 'Channel_Group'] = 'Direct'
    df_mat.loc[is_indir, 'Channel_Group'] = 'Indirect'
    df_mat.loc[(~is_dir) & (~is_indir), 'Channel_Group'] = 'Direct'
    
    if 'Team_Group' in df_mat.columns: df_mat['Team_Group'] = df_mat['Team_Group'].replace(['nan', 'None', '', '(blank)'], '-')
    if 'reChannel' in df_mat.columns: df_mat['reChannel'] = df_mat['reChannel'].replace(['nan', 'None', '', '(blank)'], '-')
    if 'Segment' in df_mat.columns:
        df_mat['Segment'] = df_mat['Segment'].replace(['individual', 'Individual', 'fit', 'FIT'], 'FIT')
        df_mat['Segment'] = df_mat['Segment'].replace(['m&e', 'M&E', 'mice', 'MICE'], 'MICE')
        
    return df_mat

def assign_strategic_tags(idf):
    d = idf.copy()
    d['Strat_Port'] = 'EC w/o Ctrip'
    d.loc[d['reChannel'].str.lower() == 'ctrip', 'Strat_Port'] = 'Ctrip'
    d.loc[d['Segment'].str.upper() == 'MICE', 'Strat_Port'] = 'MICE'
    d.loc[(d['Team_Group'].str.upper() == 'TA') & (d['Strat_Port'] != 'Ctrip') & (d['Strat_Port'] != 'MICE'), 'Strat_Port'] = 'TA'
    
    d['Strat_Zone'] = 'IZ'
    is_china = d['Dest_Type'].str.contains('China|GC', case=False) | d['Resort'].str.contains('Anji|Changbaishan|Guilin|Lijiang|Beidahu|Yabuli|Xianlin|Taicang', case=False)
    is_asia = d['Dest_Type'].str.contains('Asia|ESAP', case=False) & (~is_china)
    is_sun = d['Dest_Type'].str.contains('Sun', case=False)
    is_mountain = d['Dest_Type'].str.contains('Mountain|Snow|Ski', case=False)
    
    d.loc[is_china & is_sun, 'Strat_Zone'] = 'GC SUN'
    d.loc[is_china & is_mountain, 'Strat_Zone'] = 'GC mountain'
    d.loc[is_asia & is_sun, 'Strat_Zone'] = 'ESAP SUN'
    d.loc[is_asia & is_mountain, 'Strat_Zone'] = 'ESAP mountain'
    d.loc[d['Dest_Type'].str.contains('IZ|Interzone', case=False), 'Strat_Zone'] = 'IZ'
    return d

def build_strategic_summary_matrix(cy_df, py_df, bv_col):
    cy_g = cy_df.groupby(['Strat_Port', 'Strat_Zone'])[bv_col].sum().reset_index()
    py_g = py_df.groupby(['Strat_Port', 'Strat_Zone'])[bv_col].sum().reset_index()
    merged = pd.merge(cy_g, py_g, on=['Strat_Port', 'Strat_Zone'], how='outer', suffixes=('_CY', '_PY')).fillna(0)
    merged['Variance'] = merged[f'{bv_col}_CY'] - merged[f'{bv_col}_PY']
    merged['Var_Pct'] = np.where(merged[f'{bv_col}_PY'] > 0, merged['Variance'] / merged[f'{bv_col}_PY'] * 100, 0)
    return merged

# =================================================================
# --- 4. Plotting & AI Engines ---
# =================================================================
def draw_pacing_curve_m(df_curve, cy_label, py_label, curr_symbol, info_text):
    if df_curve is None or df_curve.empty: return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)
    
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['CY_M'], name=cy_label, mode='lines', line=dict(color='#051C2C', width=3), customdata=df_curve['CY_abs'], hovertemplate='<b>CY OTB:</b> %{customdata:,.0f} €<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['PY_M'], name=py_label, mode='lines', line=dict(color='#A4B6B0', width=2, dash='dash'), customdata=df_curve['PY_abs'], hovertemplate='<b>PY OTB:</b> %{customdata:,.0f} €<extra></extra>'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['Gap_M'].clip(lower=0), fill='tozeroy', line=dict(color='rgba(0,128,0,0)'), fillcolor='rgba(40,167,69,0.25)', name='Ahead (+)', customdata=df_curve['Gap_abs'], hovertemplate='<b>Ahead (+):</b> %{customdata:,.0f} €<extra></extra>'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['Gap_M'].clip(upper=0), fill='tozeroy', line=dict(color='rgba(255,0,0,0)'), fillcolor='rgba(220,53,69,0.25)', name='Behind (-)', customdata=df_curve['Gap_abs'], hovertemplate='<b>Behind (-):</b> %{customdata:,.0f} €<extra></extra>'), row=2, col=1)
    
    df_curve['Abs_Gap'] = df_curve['Gap_M'].abs()
    if not df_curve['Abs_Gap'].isna().all():
        max_idx = df_curve['Abs_Gap'].idxmax()
        min_idx = df_curve['Abs_Gap'].idxmin()
        
        max_gap = df_curve.loc[max_idx, 'Gap_M']
        min_gap = df_curve.loc[min_idx, 'Gap_M']
        
        max_color = "#28a745" if max_gap >= 0 else "#dc3545"
        min_color = "#28a745" if min_gap >= 0 else "#dc3545"
        
        fig.add_annotation(
            x=df_curve.loc[max_idx, 'Sales_Date'], y=df_curve.loc[max_idx, 'CY_M'],
            text=f"🔥 Max Absolute Gap<br>Gap: <b><span style='color:{max_color}'>{max_gap:+.2f} M€</span></b><br>{cy_label}: {df_curve.loc[max_idx, 'CY_M']:.2f}M€<br>{py_label}: {df_curve.loc[max_idx, 'PY_M']:.2f}M€",
            showarrow=True, arrowhead=2, row=1, col=1, ax=0, ay=-60 if df_curve.loc[max_idx, 'CY_M'] > df_curve.loc[max_idx, 'PY_M'] else 60,
            bgcolor="rgba(255,255,255,0.95)", bordercolor=max_color, borderwidth=2, font=dict(color="#051C2C", size=11)
        )
        
        fig.add_annotation(
            x=df_curve.loc[min_idx, 'Sales_Date'], y=df_curve.loc[min_idx, 'CY_M'],
            text=f"❄️ Min Absolute Gap<br>Gap: <b><span style='color:{min_color}'>{min_gap:+.2f} M€</span></b><br>{cy_label}: {df_curve.loc[min_idx, 'CY_M']:.2f}M€<br>{py_label}: {df_curve.loc[min_idx, 'PY_M']:.2f}M€",
            showarrow=True, arrowhead=2, row=1, col=1, ax=0, ay=-60 if df_curve.loc[min_idx, 'CY_M'] > df_curve.loc[min_idx, 'PY_M'] else 60,
            bgcolor="rgba(255,255,255,0.95)", bordercolor=min_color, borderwidth=2, font=dict(color="#051C2C", size=11)
        )
    
    fig.update_yaxes(ticksuffix="M", tickformat=".1f", row=1, col=1)
    fig.update_yaxes(ticksuffix="M", tickformat=".1f", row=2, col=1)
    fig.update_layout(title=dict(text=f"<b>Cumulative Pacing Trajectory (M€)</b><br><sup style='color:gray;'>{info_text}</sup>", font=dict(family="Playfair Display", size=18)), hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center'))
    return fig

def draw_weekly_pace_chart_m(df_curve, cy_label, py_label, curr_symbol, info_text):
    if df_curve is None or df_curve.empty: return go.Figure()
    df_weekly = df_curve.resample('W-MON', on='Sales_Date').sum().reset_index()
    df_weekly['Weekly_Gap_M'] = df_weekly['CY_inc_M'] - df_weekly['PY_inc_M']
    df_weekly['Weekly_Gap_abs'] = df_weekly['CY_inc_abs'] - df_weekly['PY_inc_abs'] 
    
    fig = go.Figure()
    colors = ['rgba(40,167,69,0.85)' if val >= 0 else 'rgba(220,53,69,0.85)' for val in df_weekly['Weekly_Gap_M']]
    
    fig.add_trace(go.Bar(x=df_weekly['Sales_Date'], y=df_weekly['Weekly_Gap_M'], name='Weekly Net Flow Variance', marker_color=colors, customdata=df_weekly['Weekly_Gap_abs'], hovertemplate='<b>Net Flow Var:</b> %{customdata:,.0f} €<extra></extra>'))
    fig.add_trace(go.Scatter(x=df_weekly['Sales_Date'], y=df_weekly['CY_inc_M'], name=f"{cy_label} Flow Velocity", mode='lines+markers', line=dict(color='#051C2C', width=2), customdata=df_weekly['CY_inc_abs'], hovertemplate='<b>CY Flow:</b> %{customdata:,.0f} €<extra></extra>'))
    fig.add_trace(go.Scatter(x=df_weekly['Sales_Date'], y=df_weekly['PY_inc_M'], name=f"{py_label} Flow Velocity", mode='lines+markers', line=dict(color='#A4B6B0', width=2, dash='dash'), customdata=df_weekly['PY_inc_abs'], hovertemplate='<b>PY Flow:</b> %{customdata:,.0f} €<extra></extra>'))
    
    fig.update_yaxes(ticksuffix="M", tickformat=".1f")
    fig.update_layout(title=dict(text=f"<b>Weekly Incremental Booking Velocity (M€)</b><br><sup style='color:gray;'>{info_text}</sup>", font=dict(family="Playfair Display", size=18)), hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center'))
    return fig

try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxx" 

llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.1)

def generate_weekly_diagnostics(context_info, matrix_summary_str, search_intel_str, chat_history, current_prompt):
    sys_prompt = f"""You are the Elite Executive Intelligence Brain of ClubMed, serving simultaneously as our Senior Strategic Analyst, Chief Financial Officer (CFO), and Global Sales Director. 
    Data Scope Environment: {context_info}

    ================================================================================
    🚨 【第一重盾牌：大盘数字绝对锚定铁律（ANTI-HALLUCINATION ANCHOR）】
    ================================================================================
    1. 你必须、且只能从输入数据最顶部的【=== GLOBAL OMNI GRAND TOTAL ===】明文块中读取大盘宏观总量（包含当前总预订量CY、去年总预订量PY、绝对差额Variance、同比增速%）。
    2. 报告开篇涉及大盘总盘子的数字，必须与该明文块【一字不差地精准引用】！严禁跳过明文块去肉眼心算下方细分矩阵的表格，严禁自行发挥编造任何大盘总额。
    3. 报告的战略总调性必须严格受到大盘总差额（Variance）正负号的绝对强控：
       - 若 Variance 为正，开篇定调必须是：“大盘总量报喜，但需穿透增长质量”；
       - 若 Variance 为负，开篇定调必须是：“系统性失血，全面防守止血”。

    ================================================================================
    🚨 【第二重盾牌：Tavily 实时动态情报舱（EXTERNAL INTELLIGENCE SYNTHESIS）】
    ================================================================================
    下方【Tavily 实时联网检索情报】白纸黑字记录了当前全网最新的航班运力、地缘政治、签证政策及消费心态动态。你必须将其作为归因的核心依据，完成与内部财务底盘数据的“化学反应”：
    
    【Tavily 实时联网检索情报】：
    {search_intel_str}
    
    1. 运力与成本对齐：如果情报提到某航线直航加密或票价大幅回落，且报表中该战区数据泛绿，你必须敏锐指出这是【大交通运力释放带来的直接通道红利】；反之，若燃油税高企，必须对齐分析其对散客出行的压制。
    2. 消费心态两极分化：结合最新的宏观经济信心指数，分析中产阶层是否在滑向“极致性价比”（对应国内村平替高增长），高净值人群的“老钱风”是否依旧稳固（对应长线IZ或海岛直销长锁定）。
    
    ================================================================================
    🚨 【新增核心指令：外部情报前置概述】
    ================================================================================
    阅读【Tavily 实时联网检索情报】的原始内容，并在报告的开头撰写一段【近期宏观经济总结】。该总结需包含：
    1. 近期关键航线运力变化（增班、停航、票价波动）。
    2. 签证政策与地缘政治事件（免签、落地签、限制）。
    3. 消费者信心与宏观经济情绪（消费降级、奢侈游韧性）。
    4. 其他任何与当前消费季节及目的地战区直接相关的突发信号。
    要求：概述精炼，约 200-300 字，能够提供给决策者做参考。

    ================================================================================
    🚨 【第三重盾牌：微观浮点数数量级审计（MAGNITUDE INTUITION）】
    ================================================================================
    下方的渠道与战区交叉矩阵中，所有数字均为以【百万欧元（M€）】为基础单位的浮点数！你必须具备清晰的商业量级体感，严禁混淆数量级：
    - 浮点数 `+0.01` = 0.01M€ = 1万欧元（约8万人民币）：在财务上属于【无战略意义的统计噪音】。
    - 浮点数 `+0.03` = 0.03M€ = 3万欧元（约24万人民币）：顶多是2-3个家庭的零散订单，或一个小团队单子。
    - 浮点数 `+0.20` = 0.20M€ = 20万欧元：属于常规的业务波动。
    - 浮点数 `>+0.50` = 0.50M€以上：这才属于能惊动高管层的战略性大异动或大单。
    - 【死命令】：严禁被变动百分比唬住！如果某个区域的绝对差额（Variance）绝对值小于 0.1M€，请在分析时直接忽略或定性为常规摆动，【绝对禁止】使用“雪崩式下滑、暴涨几千万、系统性溃败、跨国年会巨额大单”等夸张戏剧化词汇去描述日常噪音！

    ================================================================================
    🚨 【第四重盾牌：度假村冬夏双重形态与 P&L 资产雷达】
    ================================================================================
    1. 季节常识熔断：
       - S1 (1-4月滑雪季) vs S2 (7-8月暑期盛夏旺季)：山岳度假村（Mountain）在夏天自动切换为【夏季山地避暑、花海观赏与亲子夏令营产品】！夏天日本雪场本来就没雪，日本雪村出现微弱下滑属于正常停摆，【绝对禁止】在7-8月盛夏报告中脑补任何“滑雪需求下降/没雪”的低智商归因！国内雪村暑期泛绿则需精准归因为【避暑营大获成功】或【冬季极早鸟前置锁单】。
    2. P&L 与资产合同雷达：
    - Leased Contract (重资产直营L): 包含 Bali, Bintan, Phuket, Cherating, Kani, Finolhu, Kabira, Sahoro 以及所有 IZ 远途村。
      * 财务逻辑：承担全部运营固定成本（Rent/FTE/Energy）。盈亏平衡点高，但越过后新增客人的边际利润率极高（约80%）。
      * 洞察要求：如果在旺季这些 L 村出现散客（FIT）暴跌，必须拉响“底层利润(ROCV)崩塌”的最高红色警报；若大涨，定性为“绝对利润收割机”。
    - Managed Contract (轻资产管理M): 包含绝大部分国内村 (Guilin, Lijiang, Yabuli, Beidahu, Changbaishan, Anji, Taicang 等) 及日本部分雪村 (Tomamu, Kiroro)。
      * 财务逻辑：无度假村直接固定成本压力，靠收取按约定的利润抽成（约20% Contracted Margin）。
      * 洞察要求：属于抗压安全垫，其增长是纯顶线贡献，但对公司绝对利润绝对值拉动作用弱于 L 村。

    🚨 【地理常识与货币常识硬防线】
    - ESAP Mountain: 专指日本滑雪度假村（Sahoro, Tomamu, Kiroro Peak, Kiroro Grand）。
    - GC mountain: 专指中国国内滑雪度假村（北大壶，长白山，亚布力）。
    - GC Sun: 专指中国国内阳光度假村（丽江，桂林）。
    - ESAP SUN: 严格包含马尔代夫（Kani、Finolhu）及东南亚阳光度假村（Bali, Phuket, Cherating, Bintan, Kabira, Kota Kinabalu, 。
    - IZ (Interzone): 专指真正的高客单、跨洲远途长线市场（如欧洲阿尔卑斯、北美等）。
    - 货币单位常识：宏观大盘营业额统一使用百万欧元（M€）。当分析到中国本土细分产品（如家庭早鸟、闺蜜游套餐）的实际销售单价时，几百或几千的数字在常识上明显属于人民币，请使用（¥）或人民币进行表述，绝对禁止写成几千欧元。

    ================================================================================
    🚨 【战术指令与渠道高情商博弈红线】
    ================================================================================
    给出 2-3 条击中要害的实战方案。
    - 🚨 【商务绝对红线】：【绝对禁止】在报告中提出任何“直接降低给携程等核心渠道佣金率”的业余财务方案！
    - 💡 【高情商反击】：采取“价值/库存交换”。向携程提供独家、非标的专属体验包，用固定库存去强行置换其核心搜索页面的【免费置顶曝光位】；或通过直销私域盲盒预售、机酒隐形打包等手段绕过比价。

    ================================================================================
    🚨 【格式与币种强制规范】
    ================================================================================
    1. 请直接以高级总裁商业备忘录风格输出，多用精炼的段落与 Bullet Points，展现高级管理层的果断与冷静。
    2. 严禁任何公文、邮件元数据（如“致：...”、“日期：...”）。
    3. 宏观大盘营业额严格使用百万欧元（M€）；具体提及中国本土促销产品、细分套餐定价（如家庭早鸟、非标盲盒）时，必须使用（¥）或人民币表述，绝对禁止将几千人民币写成几千欧元。
    """
    
    messages = [SystemMessage(content=sys_prompt)]
    for msg in chat_history:
        if msg["role"] == "user": 
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant": 
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=current_prompt + f"\nData context:\n{matrix_summary_str}"))
    
    try:
        return llm.invoke(messages).content
    except Exception as e:
        return f"Diagnostics engine timeout. Error: {e}"

    
# =================================================================
# --- 6. Main Operational UI Flow ---
# =================================================================
if uploaded_file := st.sidebar.file_uploader("Upload SalesData.csv", type=['csv']):
    df = load_and_clean(uploaded_file)
    
    st.markdown("<div class='filter-container'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; color: #A64B35; font-size: 1.1rem; font-weight: 600;'>🌍 Global Parameter Controls</h4>", unsafe_allow_html=True)
    
    tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns(5)
    with tcol1:
        bv_sel = st.selectbox("Currency", ["Euro (€)", "Locale (Original)"])
        bv_col = "BV_Euro" if "Euro" in bv_sel else "BV_Locale"
        curr_sym = "€" if "Euro" in bv_sel else ""
    with tcol2: sel_mkt = st.multiselect("Source Market", sorted(df['Market'].unique()))
    with tcol3: sel_dest = st.multiselect("Dest. Type", sorted(df['Dest_Type'].unique()))
    with tcol4: sel_resort = st.multiselect("Resort", sorted(df['Resort'].unique()))
    with tcol5: sel_ta = st.multiselect("Travel Agency", sorted(df['TA_Group'].unique()))
    st.markdown("</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📅 Consumption Window")
        cons_mode = st.radio("Filter By:", ["Quick Select (Year/Season)", "Custom Date Range"])
        if cons_mode == "Quick Select (Year/Season)":
            sel_y = st.selectbox("Consumption Year", sorted(df['Year'].unique(), reverse=True), index=0)
            season = st.radio("Season Focus", ["All Year", "S1 (Jan-Jun)", "S2 (Jul-Dec)"])
            cons_desc = f"{season} {sel_y}"; cy_label, py_label = f"CY {sel_y}", f"PY {sel_y-1}"
            df_cons_filtered = df[df['Year'] == sel_y]
            if season != "All Year":
                m_range = [1,6] if "S1" in season else [7,12]
                df_cons_filtered = df_cons_filtered[df_cons_filtered['Month_Num'].between(*m_range)]
            c_start, c_end = None, None
        else:
            sel_y, season = None, None
            max_c = df['Cons_Date'].max().date() if not df['Cons_Date'].dropna().empty else datetime.date.today()
            c_start = st.date_input("Cons. Start", max_c - datetime.timedelta(days=180))
            c_end = st.date_input("Cons. End", max_c); cons_desc = f"{c_start} to {c_end}"
            cy_label, py_label = f"CY ({c_start.year})", f"PY ({c_start.year-1})"
            df_cons_filtered = df[(df['Cons_Date'].dt.date >= c_start) & (df['Cons_Date'].dt.date <= c_end)]

        st.divider()
        st.markdown("### ⏱️ Booking Window (Sales)")
        preset = st.selectbox("Quick Range Select", ["From Sales Opening", "Last 3 Months", "Last 1 Month", "Custom Range"], index=0)
        max_s = df['Sales_Date'].max().date() if not df['Sales_Date'].dropna().empty else datetime.date.today()
        
        if preset == "From Sales Opening":
            start_date = df_cons_filtered['Sales_Date'].min().date() if not df_cons_filtered.empty and not pd.isna(df_cons_filtered['Sales_Date'].min()) else max_s - datetime.timedelta(days=365)
            end_date = max_s
        elif preset == "Custom Range":
            start_date = st.date_input("Sales Start", max_s - datetime.timedelta(days=90)); end_date = st.date_input("Sales End", max_s)
        else:
            days = 90 if "3 Month" in preset else 30
            start_date = max_s - datetime.timedelta(days=days); end_date = max_s
        
        try: py_start, py_end = start_date.replace(year=start_date.year-1), end_date.replace(year=end_date.year-1)
        except: py_start, py_end = start_date - datetime.timedelta(days=365), end_date - datetime.timedelta(days=365)

    def apply_filters(idf, mode, y, seas, cs, ce, ss, se):
        d = idf.copy()
        d = d[(d['Sales_Date'].dt.date >= ss) & (d['Sales_Date'].dt.date <= se)]
        if mode == "Quick Select (Year/Season)":
            if y is not None: d = d[d['Year'] == y]
            if seas and seas != "All Year":
                m_range = [1,6] if "S1" in seas else [7,12]
                d = d[d['Month_Num'].between(*m_range)]
        else: 
            if cs and ce: d = d[(d['Cons_Date'].dt.date >= cs) & (d['Cons_Date'].dt.date <= ce)]
            
        if sel_mkt: d = d[d['Market'].isin(sel_mkt)]
        if sel_ta: d = d[d['TA_Group'].isin(sel_ta)]
        if sel_dest: d = d[d['Dest_Type'].isin(sel_dest)]
        if sel_resort: d = d[d['Resort'].isin(sel_resort)]
        return d
        
    def apply_filters_no_mkt(idf, mode, y, seas, cs, ce, ss, se):
        d = idf.copy()
        d = d[(d['Sales_Date'].dt.date >= ss) & (d['Sales_Date'].dt.date <= se)]
        if mode == "Quick Select (Year/Season)":
            if y is not None: d = d[d['Year'] == y]
            if seas and seas != "All Year":
                m_range = [1,6] if "S1" in seas else [7,12]
                d = d[d['Month_Num'].between(*m_range)]
        else: 
            if cs and ce: d = d[(d['Cons_Date'].dt.date >= cs) & (d['Cons_Date'].dt.date <= ce)]
            
        if sel_ta: d = d[d['TA_Group'].isin(sel_ta)]
        if sel_dest: d = d[d['Dest_Type'].isin(sel_dest)]
        if sel_resort: d = d[d['Resort'].isin(sel_resort)]
        return d

    actual_y = sel_y if cons_mode.startswith("Quick") and sel_y is not None else (c_start.year if c_start else datetime.date.today().year)
    
    df_cy_raw = apply_filters(df, cons_mode, sel_y if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, c_start, c_end, start_date, end_date)
    df_py_raw = apply_filters(df, cons_mode, sel_y-1 if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, safe_offset(c_start, -1), safe_offset(c_end, -1), py_start, py_end)
    df_ppy_raw = apply_filters(df, cons_mode, actual_y-2 if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, safe_offset(c_start, -2), safe_offset(c_end, -2), safe_offset(start_date, -2), safe_offset(end_date, -2))

    df_cy_tags = assign_strategic_tags(sanitize_channels(df_cy_raw[~df_cy_raw['Segment'].str.lower().str.contains('mission', na=False)]))
    df_py_tags = assign_strategic_tags(sanitize_channels(df_py_raw[~df_py_raw['Segment'].str.lower().str.contains('mission', na=False)]))
    df_ppy_tags = assign_strategic_tags(sanitize_channels(df_ppy_raw[~df_ppy_raw['Segment'].str.lower().str.contains('mission', na=False)]))

    st.markdown(f"<div class='header-box'>ClubMed Executive Intelligence Hub</div>", unsafe_allow_html=True)
    mkt_txt = ", ".join(sel_mkt) if sel_mkt else "All Markets"
    dest_txt = ", ".join(sel_dest) if sel_dest else "All Destinations"
    chart_info = f"Market: {mkt_txt} | Destination: {dest_txt} | Currency: {bv_sel.split(' ')[0]} | Cons: {cons_desc}"

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Executive Dashboard", "🎢 Trajectory & Velocity", "🎯 Strategic Decision Canvas", "📋 Automated Weekly Diagnostics", "🤖 Strategic AI Advisor"])

    # =================================================================
    # 📊 TAB 1: EXECUTIVE DASHBOARD
    # =================================================================
    with tab1:
        st.markdown(f"<h3 style='margin-bottom:20px; font-weight: 700; color: #051C2C;'>Pacing Summary: {cy_label} vs {py_label}</h3>", unsafe_allow_html=True)
        cy_v, py_v = df_cy_tags[bv_col].sum(), df_py_tags[bv_col].sum()
        cy_h, py_h = df_cy_tags['HN'].sum(), df_py_tags['HN'].sum()
        cy_adr = cy_v/cy_h if cy_h > 0 else 0
        py_adr = py_v/py_h if py_h > 0 else 0
        
        pct_v = (cy_v-py_v)/py_v*100 if py_v else 0
        pct_h = (cy_h-py_h)/py_h*100 if py_h else 0
        pct_adr = (cy_adr-py_adr)/py_adr*100 if py_adr else 0

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(custom_metric_card(f"Paced BV ({bv_sel.split(' ')[0]})", cy_v, py_v, pct_v, f"{curr_sym}{format_volume(cy_v)}", f"{curr_sym}{format_volume(py_v)}"), unsafe_allow_html=True)
        with c2: st.markdown(custom_metric_card("Paced HN", cy_h, py_h, pct_h, format_volume(cy_h), format_volume(py_h)), unsafe_allow_html=True)
        with c3: st.markdown(custom_metric_card("Current ADR", cy_adr, py_adr, pct_adr, f"{curr_sym}{cy_adr:,.0f}", f"{curr_sym}{py_adr:,.0f}"), unsafe_allow_html=True)

        col_l, col_r = st.columns([2, 1])
        with col_l:
            cy_g = df_cy_tags.groupby('Dest_Type')[[bv_col]].sum().reset_index()
            py_g = df_py_tags.groupby('Dest_Type')[[bv_col]].sum().reset_index()
            combined = pd.merge(cy_g, py_g, on='Dest_Type', how='outer', suffixes=('_CY', '_PY')).fillna(0)
            combined['YoY_Pct'] = np.where(combined[f'{bv_col}_PY'] > 0, (combined[f'{bv_col}_CY'] - combined[f'{bv_col}_PY']) / combined[f'{bv_col}_PY'] * 100, 0)
            text_cy = [f"<b>{format_volume(cy)}<br>({pct:+.1f}%)</b>" if py > 0 else f"<b>{format_volume(cy)}</b>" for cy, py, pct in zip(combined[f'{bv_col}_CY'], combined[f'{bv_col}_PY'], combined['YoY_Pct'])]
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=combined['Dest_Type'], y=combined[f'{bv_col}_CY']/1000, name=cy_label, marker_color='#051C2C', text=text_cy, textposition='outside'))
            fig_bar.add_trace(go.Bar(x=combined['Dest_Type'], y=combined[f'{bv_col}_PY']/1000, name=py_label, marker_color='#A4B6B0', text=[f"<b>{format_volume(v)}</b>" for v in combined[f'{bv_col}_PY']], textposition='outside'))
            fig_bar.update_yaxes(ticksuffix="k", tickformat=",")
            fig_bar.update_layout(title=f"Booking Pace by Dest Type<br><sup style='color:gray;'>{chart_info}</sup>", barmode='group', margin=dict(t=100))
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_r:
            fig_pie = px.pie(cy_g, values=bv_col, names='Dest_Type', title=f"{cy_label} Share", color_discrete_sequence=['#051C2C', '#A64B35', '#A4B6B0', '#EAECEF'])
            fig_pie.update_traces(textinfo='percent+label', hole=.3); st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<hr style='margin: 30px 0; border-top: 2px solid #EAECEF;'/>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='font-weight: 700; color: #051C2C; margin-bottom: 5px;'>🏢 Channel Structure Deep-dive Matrix</h3>", unsafe_allow_html=True)
        
        grp_cols = ['Segment', 'Channel_Group', 'Team_Group', 'reChannel']
        cy_matrix_grp = df_cy_tags.groupby(grp_cols, dropna=False).agg({bv_col: 'sum', 'HN': 'sum'}).reset_index()
        py_matrix_grp = df_py_tags.groupby(grp_cols, dropna=False).agg({bv_col: 'sum', 'HN': 'sum'}).reset_index()
        m_df = pd.merge(cy_matrix_grp, py_matrix_grp, on=grp_cols, how='outer', suffixes=('_CY', '_PY')).fillna(0)
        for c in grp_cols: m_df[c] = m_df[c].astype(str).str.strip().replace(['nan', 'None', ''], '-')
        m_df = m_df.groupby(grp_cols).sum().reset_index().sort_values(['Segment', 'Channel_Group', f'{bv_col}_CY'], ascending=[True, True, False])
        
        html_out = [CSS_STYLE, '<table class="mckinsey-table"><thead><tr>']
        html_out.append('<th rowspan="2" class="th-main th-dark" style="width:8%;">Segment</th><th rowspan="2" class="th-main th-dark" style="width:10%;">Channel Group</th><th rowspan="2" class="th-main th-dark" style="width:10%;">Team Group</th><th rowspan="2" class="th-main th-dark" style="width:12%;">reChannel</th><th colspan="3" class="th-main th-cy">Current Period</th><th colspan="3" class="th-main th-py" style="border-left: 2px solid #ffffff;">Previous Period</th><th colspan="3" class="th-main th-var" style="border-left: 2px solid #ffffff;">Variance</th></tr><tr>')
        html_out.append('<th class="th-sub th-cy">BV</th><th class="th-sub th-cy">HN</th><th class="th-sub th-cy th-divider">ADR</th><th class="th-sub th-py">BV</th><th class="th-sub th-py">HN</th><th class="th-sub th-py th-divider">ADR</th><th class="th-sub th-var">BV %</th><th class="th-sub th-var">HN %</th><th class="th-sub th-var">ADR %</th></tr></thead><tbody>')
        
        seg_rowspan_dict, ch_rowspan_dict, tm_rowspan_dict = {}, {}, {}
        show_ch_subtotal, show_tm_subtotal = {}, {}
        
        for s in m_df['Segment'].unique():
            df_s = m_df[m_df['Segment'] == s]
            s_rows = 0
            for ch in df_s['Channel_Group'].unique():
                df_ch = df_s[df_s['Channel_Group'] == ch]
                tms = df_ch['Team_Group'].unique()
                n_ch = len(tms) > 1
                ch_rows = 0
                for tm in tms:
                    df_tm = df_ch[df_ch['Team_Group'] == tm]
                    n_tm = len(df_tm) > 1
                    t_rows = len(df_tm) + (1 if n_tm else 0)
                    tm_rowspan_dict[(s, ch, tm)] = t_rows
                    show_tm_subtotal[(s, ch, tm)] = n_tm
                    ch_rows += t_rows
                if n_ch: ch_rows += 1
                ch_rowspan_dict[(s, ch)] = ch_rows
                show_ch_subtotal[(s, ch)] = n_ch
                s_rows += ch_rows
            s_rows += 1
            seg_rowspan_dict[s] = s_rows

        gt_cy_b, gt_cy_h, gt_py_b, gt_py_h = 0, 0, 0, 0
        for s in m_df['Segment'].unique():
            df_s = m_df[m_df['Segment'] == s]
            seg_cy_bv, seg_cy_hn = df_s[f'{bv_col}_CY'].sum(), df_s['HN_CY'].sum()
            seg_py_bv, seg_py_hn = df_s[f'{bv_col}_PY'].sum(), df_s['HN_PY'].sum()
            gt_cy_b += seg_cy_bv; gt_cy_h += seg_cy_hn
            gt_py_b += seg_py_bv; gt_py_h += gt_py_h
            
            first_s = True
            for ch in df_s['Channel_Group'].unique():
                df_ch = df_s[df_s['Channel_Group'] == ch]
                ch_cy_bv, ch_cy_hn = df_ch[f'{bv_col}_CY'].sum(), df_ch['HN_CY'].sum()
                ch_py_bv, ch_py_hn = df_ch[f'{bv_col}_PY'].sum(), df_ch['HN_PY'].sum()
                first_ch = True
                
                for tm in df_ch['Team_Group'].unique():
                    df_tm = df_ch[df_ch['Team_Group'] == tm]
                    tm_cy_bv, tm_cy_hn = df_tm[f'{bv_col}_CY'].sum(), df_tm['HN_CY'].sum()
                    tm_py_bv, tm_py_hn = df_tm[f'{bv_col}_PY'].sum(), df_tm['HN_PY'].sum()
                    first_tm = True
                    
                    for idx, row in df_tm.iterrows():
                        html_out.append('<tr>')
                        if first_s: html_out.append(f'<td rowspan="{seg_rowspan_dict[s]}" class="cell-merged" style="border-right: 2px solid #051C2C !important;">{s}</td>'); first_s = False
                        if first_ch: html_out.append(f'<td rowspan="{ch_rowspan_dict[(s, ch)]}" class="cell-merged">{ch}</td>'); first_ch = False
                        if first_tm: html_out.append(f'<td rowspan="{tm_rowspan_dict[(s, ch, tm)]}" class="cell-merged">{tm}</td>'); first_tm = False
                        cb, chn = row[f'{bv_col}_CY'], row['HN_CY']
                        pb, phn = row[f'{bv_col}_PY'], row['HN_PY']
                        ca = cb/chn if chn>0 else 0
                        pa = pb/phn if phn>0 else 0
                        html_out.append(f'<td class="cell-detail-left">{row["reChannel"]}</td><td>{fmt_val(cb)}</td><td>{fmt_val(chn)}</td><td class="td-divider">{fmt_val(ca)}</td><td>{fmt_val(pb)}</td><td>{fmt_val(phn)}</td><td class="td-divider">{fmt_val(pa)}</td><td>{fmt_val((cb-pb)/pb if pb>0 else 0, True)}</td><td>{fmt_val((chn-phn)/phn if phn>0 else 0, True)}</td><td>{fmt_val((ca-pa)/pa if pa>0 else 0, True)}</td></tr>')
                    
                    if show_tm_subtotal[(s, ch, tm)]:
                        tm_ca = tm_cy_bv / tm_cy_hn if tm_cy_hn > 0 else 0
                        tm_pa = tm_py_bv / tm_py_hn if tm_py_hn > 0 else 0
                        html_out.append(f'<tr class="subtotal-row" style="background-color:#FAFDFC !important;"><td class="cell-detail-left" style="font-style:italic; padding-left:15px; font-weight:500; color:#5C7080 !important;">{tm} Subtotal</td><td>{fmt_val(tm_cy_bv)}</td><td>{fmt_val(tm_cy_hn)}</td><td class="td-divider">{fmt_val(tm_ca)}</td><td>{fmt_val(tm_py_bv)}</td><td>{fmt_val(tm_py_hn)}</td><td class="td-divider">{fmt_val(tm_pa)}</td><td>{fmt_val((tm_cy_bv-tm_py_bv)/tm_py_bv if tm_py_bv>0 else 0, True)}</td><td>{fmt_val((tm_cy_hn-tm_py_hn)/tm_py_hn if tm_py_hn>0 else 0, True)}</td><td>{fmt_val((tm_ca-tm_pa)/tm_pa if tm_pa>0 else 0, True)}</td></tr>')
                
                if show_ch_subtotal[(s, ch)]:
                    ch_ca = ch_cy_bv / ch_cy_hn if ch_cy_hn > 0 else 0
                    ch_pa = ch_py_bv / ch_py_hn if ch_py_hn > 0 else 0
                    html_out.append(f'<tr class="subtotal-row"><td colspan="2" class="cell-detail-left" style="padding-left:15px; font-weight:600;">{ch} Total</td><td>{fmt_val(ch_cy_bv)}</td><td>{fmt_val(ch_cy_hn)}</td><td class="td-divider">{fmt_val(ch_ca)}</td><td>{fmt_val(ch_py_bv)}</td><td>{fmt_val(ch_py_hn)}</td><td class="td-divider">{fmt_val(ch_pa)}</td><td>{fmt_val((ch_cy_bv-ch_py_bv)/ch_py_bv if ch_py_bv>0 else 0, True)}</td><td>{fmt_val((ch_cy_hn-ch_py_hn)/ch_py_hn if ch_py_hn>0 else 0, True)}</td><td>{fmt_val((ch_ca-ch_pa)/ch_pa if ch_pa>0 else 0, True)}</td></tr>')

            seg_ca = seg_cy_bv / seg_cy_hn if seg_cy_hn > 0 else 0
            seg_pa = seg_py_bv / seg_py_hn if seg_py_hn > 0 else 0
            html_out.append(f'<tr class="total-row"><td colspan="3" class="cell-detail-left" style="font-weight:700;">{s} OMNI TOTAL</td><td>{fmt_val(seg_cy_bv)}</td><td>{fmt_val(seg_cy_hn)}</td><td class="td-divider">{fmt_val(seg_ca)}</td><td>{fmt_val(seg_py_bv)}</td><td>{fmt_val(seg_py_hn)}</td><td class="td-divider">{fmt_val(seg_pa)}</td><td>{fmt_val((seg_cy_bv-seg_py_bv)/seg_py_bv if seg_py_bv>0 else 0, True)}</td><td>{fmt_val((seg_cy_hn-seg_py_hn)/seg_py_hn if seg_py_hn>0 else 0, True)}</td><td>{fmt_val((seg_ca-seg_pa)/seg_pa if seg_pa>0 else 0, True)}</td></tr>')
        
        gt_adr_cy = gt_cy_b / gt_cy_h if gt_cy_h > 0 else 0
        gt_adr_py = gt_py_b / gt_py_h if gt_py_h > 0 else 0
        html_out.append(f'<tr class="grand-total-row" style="background-color:#E2ECF1 !important;"><td colspan="4" class="cell-detail-left" style="font-weight:800; color:#A64B35 !important;">SUM OF CORE DEMAND (FIT + MICE COMBINED)</td><td>{fmt_val(gt_cy_b)}</td><td>{fmt_val(gt_cy_h)}</td><td class="td-divider">{fmt_val(gt_adr_cy)}</td><td>{fmt_val(gt_py_b)}</td><td>{fmt_val(gt_py_h)}</td><td class="td-divider">{fmt_val(gt_adr_py)}</td><td>{fmt_val((gt_cy_b-gt_py_b)/gt_py_b if gt_py_b>0 else 0, True)}</td><td>{fmt_val((gt_cy_h-gt_py_h)/gt_py_h if gt_py_h>0 else 0, True)}</td><td>{fmt_val((gt_adr_cy-gt_adr_py)/gt_adr_py if gt_adr_py>0 else 0, True)}</td></tr>')
        
        html_out.append('</tbody></table>')
        st.markdown("".join(html_out), unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 30px 0; border-top: 2px solid #EAECEF;'/>", unsafe_allow_html=True)
        
        c_title, c_filt = st.columns([3, 1])
        with c_title:
            st.markdown(f"<h3 style='font-weight: 700; color: #051C2C; margin-bottom: 5px;'>🏔️ Resort-Level Performance</h3>", unsafe_allow_html=True)
        with c_filt:
            all_ports = ['EC w/o Ctrip', 'Ctrip', 'MICE', 'TA']
            is_china = any('CHINA' in m.upper() for m in sel_mkt) if sel_mkt else True
            if is_china:
                sel_port = st.multiselect("Strat Port (Resort Tbl)", all_ports, default=[], placeholder="Choose options...")
            else:
                sel_port = st.multiselect("Strat Port (Resort Tbl)", all_ports, default=[], disabled=True)

        mkt_text = ", ".join(sel_mkt) if sel_mkt else "All Markets"
        port_text = ", ".join(sel_port) if sel_port else "All Portfolios"
        
        st.markdown(f"""
        <div style="background-color:#F8F9FA; padding:10px 15px; border-radius:6px; border-left:4px solid #A4B6B0; margin-bottom:15px;">
            <p style="color:#051C2C; font-size:0.95rem; font-weight:600; margin-bottom:0;">
                <span style="color:#6C757D;">Market:</span> {mkt_text} &nbsp;|&nbsp; 
                <span style="color:#6C757D;">Consumption:</span> {cons_desc} &nbsp;|&nbsp; 
                <span style="color:#6C757D;">Strategic Portfolio:</span> {port_text} &nbsp;|&nbsp; 
                <span style="color:#6C757D;">Currency:</span> {bv_sel}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<p style='color:#6C757D; font-size: 0.85rem; margin-bottom:15px; font-style:italic;'>* Note: IZ (Interzone) resorts are consolidated into a single macro line item for high-level visibility.</p>", unsafe_allow_html=True)

        t1_cy = df_cy_tags.copy()
        t1_py = df_py_tags.copy()

        if sel_port:
            t1_cy = t1_cy[t1_cy['Strat_Port'].isin(sel_port)]
            t1_py = t1_py[t1_py['Strat_Port'].isin(sel_port)]

        t1_cy.loc[t1_cy['Strat_Zone'] == 'IZ', 'Resort'] = 'INTERZONE CONSOLIDATED'
        t1_py.loc[t1_py['Strat_Zone'] == 'IZ', 'Resort'] = 'INTERZONE CONSOLIDATED'

        grp_cy = t1_cy.groupby(['Strat_Zone', 'Resort']).agg({bv_col:'sum', 'HN':'sum'}).reset_index()
        grp_py = t1_py.groupby(['Strat_Zone', 'Resort']).agg({bv_col:'sum', 'HN':'sum'}).reset_index()

        t1_m = pd.merge(grp_cy, grp_py, on=['Strat_Zone', 'Resort'], how='outer', suffixes=('_CY', '_PY')).fillna(0)

        html_t1 = ['<table class="mckinsey-table"><thead><tr>']
        html_t1.append('<th rowspan="2" class="th-main th-dark" style="width:14%;">Strategic Zone</th><th rowspan="2" class="th-main th-dark" style="width:16%;">Resort / Pool</th><th colspan="3" class="th-main th-cy">Current Period (CY)</th><th colspan="3" class="th-main th-py" style="border-left: 2px solid #ffffff;">Previous Period (PY)</th><th colspan="3" class="th-main th-var" style="border-left: 2px solid #ffffff;">YoY Variance %</th></tr><tr>')
        html_t1.append('<th class="th-sub th-cy">BV</th><th class="th-sub th-cy">HN</th><th class="th-sub th-cy th-divider">ADR</th><th class="th-sub th-py">BV</th><th class="th-sub th-py">HN</th><th class="th-sub th-py th-divider">ADR</th><th class="th-sub th-var">BV %</th><th class="th-sub th-var">HN %</th><th class="th-sub th-var">ADR %</th></tr></thead><tbody>')

        grand_cy_b, grand_cy_h, grand_py_b, grand_py_h = 0, 0, 0, 0
        sorted_zones = sorted(t1_m['Strat_Zone'].unique())
        
        for zone in sorted_zones:
            z_df = t1_m[t1_m['Strat_Zone'] == zone].sort_values(f'{bv_col}_CY', ascending=False)
            
            sub_cy_b, sub_cy_h = z_df[f'{bv_col}_CY'].sum(), z_df['HN_CY'].sum()
            sub_py_b, sub_py_h = z_df[f'{bv_col}_PY'].sum(), z_df['HN_PY'].sum()
            grand_cy_b += sub_cy_b; grand_cy_h += sub_cy_h
            grand_py_b += sub_py_b; grand_py_h += sub_py_h
            
            is_iz = (zone == 'IZ')
            rowspan = 1 if is_iz else len(z_df) + 1 
            first = True
            
            for _, row in z_df.iterrows():
                html_t1.append('<tr>')
                if first:
                    html_t1.append(f'<td rowspan="{rowspan}" class="cell-merged" style="border-right: 2px solid #051C2C !important;">{zone}</td>')
                    first = False
                
                cb, chn = row[f'{bv_col}_CY'], row['HN_CY']
                pb, phn = row[f'{bv_col}_PY'], row['HN_PY']
                ca = cb/chn if chn>0 else 0
                pa = pb/phn if phn>0 else 0
                
                html_t1.append(f'<td class="cell-detail-left">{row["Resort"]}</td><td>{fmt_val(cb)}</td><td>{fmt_val(chn)}</td><td class="td-divider">{fmt_val(ca)}</td><td>{fmt_val(pb)}</td><td>{fmt_val(phn)}</td><td class="td-divider">{fmt_val(pa)}</td><td>{fmt_val((cb-pb)/pb if pb>0 else 0, True)}</td><td>{fmt_val((chn-phn)/phn if phn>0 else 0, True)}</td><td>{fmt_val((ca-pa)/pa if pa>0 else 0, True)}</td></tr>')
            
            if not is_iz:
                sub_ca = sub_cy_b / sub_cy_h if sub_cy_h > 0 else 0
                sub_pa = sub_py_b / sub_py_h if sub_py_h > 0 else 0
                html_t1.append(f'<tr class="subtotal-row"><td class="cell-detail-left" style="font-weight:600;">{zone} Subtotal</td><td>{fmt_val(sub_cy_b)}</td><td>{fmt_val(sub_cy_h)}</td><td class="td-divider">{fmt_val(sub_ca)}</td><td>{fmt_val(sub_py_b)}</td><td>{fmt_val(sub_py_h)}</td><td class="td-divider">{fmt_val(sub_pa)}</td><td>{fmt_val((sub_cy_b-sub_py_b)/sub_py_b if sub_py_b>0 else 0, True)}</td><td>{fmt_val((sub_cy_h-sub_py_h)/sub_py_h if sub_py_h>0 else 0, True)}</td><td>{fmt_val((sub_ca-sub_pa)/sub_pa if sub_pa>0 else 0, True)}</td></tr>')

        gt_ca = grand_cy_b / grand_cy_h if grand_cy_h > 0 else 0
        gt_pa = grand_py_b / grand_py_h if grand_py_h > 0 else 0
        html_t1.append(f'<tr class="grand-total-row" style="background-color:#E2ECF1 !important;"><td colspan="2" class="cell-detail-left" style="font-weight:800; color:#A64B35 !important;">GLOBAL RESORT TOTAL</td><td>{fmt_val(grand_cy_b)}</td><td>{fmt_val(grand_cy_h)}</td><td class="td-divider">{fmt_val(gt_ca)}</td><td>{fmt_val(grand_py_b)}</td><td>{fmt_val(grand_py_h)}</td><td class="td-divider">{fmt_val(gt_pa)}</td><td>{fmt_val((grand_cy_b-grand_py_b)/grand_py_b if grand_py_b>0 else 0, True)}</td><td>{fmt_val((grand_cy_h-grand_py_h)/grand_py_h if grand_py_h>0 else 0, True)}</td><td>{fmt_val((gt_ca-gt_pa)/gt_pa if gt_pa>0 else 0, True)}</td></tr>')
        
        html_t1.append('</tbody></table>')
        st.markdown("".join(html_t1), unsafe_allow_html=True)

    # =================================================================
    # 🎢 TAB 2: TRAJECTORY & VELOCITY
    # =================================================================
    with tab2:
        def get_curve_m(idf, cy_y, mode, seas, cs, ce, se):
            d_cy = apply_filters(idf, mode, cy_y, seas, cs, ce, datetime.date(2000,1,1), se)
            d_py = apply_filters(idf, mode, cy_y-1, seas, safe_offset(cs, -1), safe_offset(ce, -1), datetime.date(2000,1,1), se-datetime.timedelta(days=365))
            
            c_d = d_cy.groupby('Sales_Date')[bv_col].sum().reset_index()
            p_d = d_py.groupby('Sales_Date')[bv_col].sum().reset_index()
            p_d['Sales_Date'] = p_d['Sales_Date'] + pd.DateOffset(years=1)
            if c_d.empty and p_d.empty: return None
            tline = pd.date_range(start=min(c_d['Sales_Date'].min() if not c_d.empty else pd.to_datetime(se), p_d['Sales_Date'].min() if not p_d.empty else pd.to_datetime(se)), end=pd.to_datetime(se))
            df_t = pd.DataFrame({'Sales_Date': tline})
            c_d = pd.merge(df_t, c_d, on='Sales_Date', how='left').fillna(0)
            p_d = pd.merge(df_t, p_d, on='Sales_Date', how='left').fillna(0)
            res = df_t.copy()
            
            res['CY_inc_abs'] = c_d[bv_col]
            res['PY_inc_abs'] = p_d[bv_col]
            res['CY_abs'] = res['CY_inc_abs'].cumsum()
            res['PY_abs'] = res['PY_inc_abs'].cumsum()
            res['Gap_abs'] = res['CY_abs'] - res['PY_abs']
            
            res['CY_inc_M'] = res['CY_inc_abs'] / 1_000_000
            res['PY_inc_M'] = res['PY_inc_abs'] / 1_000_000
            res['CY_M'] = res['CY_abs'] / 1_000_000
            res['PY_M'] = res['PY_abs'] / 1_000_000
            res['Gap_M'] = res['Gap_abs'] / 1_000_000
            return res
            
        curve_data = get_curve_m(df, sel_y if cons_mode.startswith("Quick") else c_start.year, cons_mode, season, c_start, c_end, end_date)
        st.plotly_chart(draw_pacing_curve_m(curve_data, cy_label, py_label, curr_sym, chart_info), use_container_width=True)
        st.plotly_chart(draw_weekly_pace_chart_m(curve_data, cy_label, py_label, curr_sym, chart_info), use_container_width=True)

        st.markdown("---")
        st.markdown("<h3 style='color:#051C2C; font-weight:700;'>Rolling 15-Days Sales Momentum (CY vs PY vs PPY)</h3>", unsafe_allow_html=True)
        cy_15 = df_cy_tags.groupby(df_cy_tags['Sales_Date'].dt.date)[bv_col].sum().reset_index().tail(15)
        py_15 = df_py_tags.groupby(df_py_tags['Sales_Date'].dt.date)[bv_col].sum().reset_index().tail(15)
        ppy_15 = df_ppy_tags.groupby(df_ppy_tags['Sales_Date'].dt.date)[bv_col].sum().reset_index().tail(15)
        
        tot_cy_15 = cy_15[bv_col].sum()
        tot_py_15 = py_15[bv_col].sum()
        tot_ppy_15 = ppy_15[bv_col].sum()
        
        yoy_growth_15 = (tot_cy_15 - tot_py_15) / tot_py_15 * 100 if tot_py_15 > 0 else 0
        
        fig_trend_15 = go.Figure()
        fig_trend_15.add_trace(go.Scatter(x=cy_15['Sales_Date'], y=cy_15[bv_col]/1_000_000, name='CY Rolling 15D Daily Flow', mode='lines+markers', line=dict(color='#051C2C', width=3), customdata=cy_15[bv_col], hovertemplate='CY Absolute: %{customdata:,.0f} €<extra></extra>'))
        fig_trend_15.add_trace(go.Scatter(x=cy_15['Sales_Date'], y=py_15[bv_col]/1_000_000, name='PY Corresponding Daily Flow', mode='lines', line=dict(color='#A4B6B0', width=2, dash='dash'), customdata=py_15[bv_col], hovertemplate='PY Absolute: %{customdata:,.0f} €<extra></extra>'))
        fig_trend_15.add_trace(go.Scatter(x=cy_15['Sales_Date'], y=ppy_15[bv_col]/1_000_000, name='PPY Corresponding Daily Flow', mode='lines', line=dict(color='#A64B35', width=1.5, dash='dot'), customdata=ppy_15[bv_col], hovertemplate='PPY Absolute: %{customdata:,.0f} €<extra></extra>'))
        
        fig_trend_15.add_annotation(
            xref="paper", yref="paper", x=0.98, xanchor="right", y=0.95, yanchor="top",
            text=f"<b>Rolling 15-Days Strategic Accumulation:</b><br>• CY Aggregate: {tot_cy_15/1_000_000:.2f} M€<br>• PY Aggregate: {tot_py_15/1_000_000:.2f} M€ (YoY Var: <span style='color:{'#28a745' if yoy_growth_15>=0 else '#dc3545'}; font-weight:700;'>{yoy_growth_15:+.1f}%</span>)<br>• PPY Aggregate: {tot_ppy_15/1_000_000:.2f} M€",
            showarrow=False, bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#051C2C", borderwidth=1.5, borderpad=12
        )
        fig_trend_15.update_yaxes(ticksuffix="M", tickformat=".1f", title_text="Daily Velocity Profile (M€)")
        fig_trend_15.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.15, x=0.5, xanchor='center'))
        st.plotly_chart(fig_trend_15, use_container_width=True)

    # =================================================================
    # 🎯 TAB 3: STRATEGIC DECISION CANVAS
    # =================================================================
    with tab3:
        st.markdown("<h2 style='color:#051C2C; font-weight:700;'>🎯 Advanced Decision Support Canvas</h2>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<h3 style='color:#051C2C; font-weight:600;'>Booking Lead-Time & Decision Window Monitor</h3>", unsafe_allow_html=True)
        df_cy_tags['Lead_Time'] = (df_cy_tags['Cons_Date'] - df_cy_tags['Sales_Date']).dt.days
        df_py_tags['Lead_Time'] = (df_py_tags['Cons_Date'] - df_py_tags['Sales_Date']).dt.days
        bins = [-999, 7, 30, 60, 90, 99999]
        labels = ['0-7 Days (Last-Minute)', '8-30 Days (Short-Lead)', '31-60 Days (Regular)', '61-90 Days (Early-Bird)', '90+ Days (Long-Lock)']
        df_cy_tags['Lead_Bucket'] = pd.cut(df_cy_tags['Lead_Time'], bins=bins, labels=labels)
        df_py_tags['Lead_Bucket'] = pd.cut(df_py_tags['Lead_Time'], bins=bins, labels=labels)
        lt_cy = df_cy_tags.groupby('Lead_Bucket', observed=False)[bv_col].sum().reset_index()
        lt_py = df_py_tags.groupby('Lead_Bucket', observed=False)[bv_col].sum().reset_index()
        lt_m = pd.merge(lt_cy, lt_py, on='Lead_Bucket', suffixes=('_CY', '_PY'))
        
        lt_cy_total = lt_m[f'{bv_col}_CY'].sum()
        lt_py_total = lt_m[f'{bv_col}_PY'].sum()
        
        lt_m['Label_CY'] = [f"<b>{v/1_000_000:.2f}</b><br>({(v/lt_cy_total*100):.1f}%)" if lt_cy_total > 0 else "0" for v in lt_m[f'{bv_col}_CY']]
        lt_m['Label_PY'] = [f"<b>{v/1_000_000:.2f}</b><br>({(v/lt_py_total*100):.1f}%)" if lt_py_total > 0 else "0" for v in lt_m[f'{bv_col}_PY']]
        
        fig_lt = go.Figure()
        fig_lt.add_trace(go.Bar(x=lt_m['Lead_Bucket'], y=lt_m[f'{bv_col}_CY']/1_000_000, name=cy_label, marker_color='#051C2C', text=lt_m['Label_CY'], textposition='inside'))
        fig_lt.add_trace(go.Bar(x=lt_m['Lead_Bucket'], y=lt_m[f'{bv_col}_PY']/1_000_000, name=py_label, marker_color='#A4B6B0', text=lt_m['Label_PY'], textposition='outside'))
        fig_lt.update_yaxes(ticksuffix="M", tickformat=".1f", title_text="Volume (M€)")
        fig_lt.update_layout(barmode='group', margin=dict(t=50))
        st.plotly_chart(fig_lt, use_container_width=True)
        
        st.markdown("<h4 style='color:#051C2C; margin-top:20px;'>🔍 Product Mix Breakdown by Decision Window</h4>", unsafe_allow_html=True)
        sel_bucket = st.selectbox("Select specific Lead-Time Window to inspect product composition:", labels)
        
        mix_cy = df_cy_tags[df_cy_tags['Lead_Bucket'] == sel_bucket].groupby('Strat_Zone')[bv_col].sum().reset_index()
        mix_py = df_py_tags[df_py_tags['Lead_Bucket'] == sel_bucket].groupby('Strat_Zone')[bv_col].sum().reset_index()
        mix_m = pd.merge(mix_cy, mix_py, on='Strat_Zone', how='outer', suffixes=('_CY', '_PY')).fillna(0)
        
        mix_m['Var_Abs'] = mix_m[f'{bv_col}_CY'] - mix_m[f'{bv_col}_PY']
        mix_m['Var_Pct'] = np.where(mix_m[f'{bv_col}_PY'] > 0, mix_m['Var_Abs'] / mix_m[f'{bv_col}_PY'] * 100, 0)
        
        mix_m['Label_Text'] = [f"<b>{c/1e6:.1f}</b><br><span style='color:{'#28a745' if p>=0 else '#dc3545'};'>{p:+.1f}%</span>" for c, p in zip(mix_m[f'{bv_col}_CY'], mix_m['Var_Pct'])]
        
        fig_mix = go.Figure()
        fig_mix.add_trace(go.Bar(x=mix_m['Strat_Zone'], y=mix_m[f'{bv_col}_CY']/1_000_000, name=cy_label, marker_color='#051C2C', text=mix_m['Label_Text'], textposition='outside'))
        fig_mix.add_trace(go.Bar(x=mix_m['Strat_Zone'], y=mix_m[f'{bv_col}_PY']/1_000_000, name=py_label, marker_color='#A4B6B0', text=[f"<b>{v/1e6:.1f}</b>" for v in mix_m[f'{bv_col}_PY']], textposition='outside'))
        fig_mix.update_yaxes(ticksuffix="M", tickformat=".1f")
        fig_mix.update_layout(barmode='group', margin=dict(t=50), title=f"Product Mix for [{sel_bucket}]")
        st.plotly_chart(fig_mix, use_container_width=True)
            
        st.markdown("---")
        st.markdown("<h3 style='color:#051C2C; font-weight:600;'>Greater China & HK Source Market Strategic Corridor (Sankey Matrix)</h3>", unsafe_allow_html=True)
        
        df_cy_sankey_base = apply_filters_no_mkt(df, cons_mode, sel_y if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, c_start, c_end, start_date, end_date)
        df_cy_sankey_tags = assign_strategic_tags(sanitize_channels(df_cy_sankey_base[~df_cy_sankey_base['Segment'].str.lower().str.contains('mission', na=False)]))
        
        sk_filtered = df_cy_sankey_tags[df_cy_sankey_tags['Market'].str.lower().str.contains('china|hong|香港|中国', na=False)].copy()
        
        required_cols = ['Src_Node', 'Strat_Zone', bv_col]
        if not sk_filtered.empty:
            sk_filtered['Src_Node'] = np.where(sk_filtered['Market'].str.lower().str.contains('hong|香港', na=False), 'CM Hong Kong', 'CM China')
            
            if all(col in sk_filtered.columns for col in required_cols):
                sk_grp = sk_filtered.groupby(['Src_Node', 'Strat_Zone'])[bv_col].sum().reset_index()
                
                total_vol = sk_grp[bv_col].sum()
                src_tot = sk_grp.groupby('Src_Node')[bv_col].sum()
                dest_tot = sk_grp.groupby('Strat_Zone')[bv_col].sum()
                
                src_nodes = sk_grp['Src_Node'].unique().tolist()
                dest_nodes = sk_grp['Strat_Zone'].unique().tolist()
                nodes = src_nodes + dest_nodes
                node_map = {name: i for i, name in enumerate(nodes)}
                
                mckinsey_colors = {
                    'CM China': '#051C2C',      # Navy
                    'CM Hong Kong': '#FDB913',  # Yellow
                    'GC SUN': '#A4B6B0',        
                    'GC mountain': '#5C7080',   
                    'ESAP SUN': '#D0DFE7',      
                    'ESAP mountain': '#112E43', 
                    'IZ': '#EAECEF'             
                }
                node_colors = [mckinsey_colors.get(n, '#A4B6B0') for n in nodes]
                
                def get_rgba(hex_color, alpha=0.15):
                    hex_color = hex_color.lstrip('#')
                    if len(hex_color) == 6:
                        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        return f'rgba({r}, {g}, {b}, {alpha})'
                    return f'rgba(200, 200, 200, {alpha})'

                link_colors = [get_rgba(mckinsey_colors.get(src, '#A4B6B0')) for src in sk_grp['Src_Node']]

                node_labels = []
                for n in nodes:
                    if n in src_tot:
                        v = src_tot[n]
                        node_labels.append(f"{n}<br>{v/total_vol*100:.1f}%<br>{v/1e6:.1f}M€")
                    elif n in dest_tot:
                        v = dest_tot[n]
                        node_labels.append(f"{n}<br>{v/total_vol*100:.1f}%<br>{v/1e6:.1f}M€")
                    else:
                        node_labels.append(n)

                sources = [node_map[s] for s in sk_grp['Src_Node']]
                targets = [node_map[t] for t in sk_grp['Strat_Zone']]
                values = sk_grp[bv_col].round(0).tolist()
                
                fig_sankey = go.Figure(data=[go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=25, 
                        thickness=20,
                        line=dict(color="white", width=0.5),
                        label=node_labels, 
                        color=node_colors,
                        hoverlabel=dict(bgcolor="white", font=dict(color="black"))
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                        hovertemplate='%{source.label} → %{target.label}<br><b>Flow Volume: %{value}€</b><extra></extra>'
                    )
                )])
                
                st.markdown("""
                <style>
                .sankey-text, .sankey text {
                    fill: #000000 !important;
                    text-shadow: none !important;
                    font-weight: 600 !important;
                }
                </style>
                """, unsafe_allow_html=True)

                fig_sankey.update_layout(
                    height=600,
                    margin=dict(l=160, r=160, t=50, b=50),
                    title_text="Source Market → Strategic Zone Flow (M€)",
                    title_font=dict(size=16, color="#051C2C", family="Playfair Display"),
                    font=dict(size=13, color="black", family="Inter, sans-serif"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_sankey, use_container_width=True, theme=None)
                
            else:
                st.info("No corridor records matched for China/HK markers.")
        else:
            st.info("No corridor records matched for China/HK markers.")

        st.markdown("---")
        st.markdown("<h3 style='color:#051C2C; font-weight:600;'>Strategic Channel Cannibalization & Margin Quality Radar</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #6C757D; font-size: 0.95rem; font-style: italic; margin-top:-10px;'>* Note: Variance percentage (%) displayed on top of bars represents the YoY ADR (Average Daily Rate) change for the respective channel and zone.</p>", unsafe_allow_html=True)
        
        adr_cy = df_cy_tags.groupby(['Strat_Zone', 'Channel_Group']).agg({bv_col:'sum', 'HN':'sum'}).reset_index()
        adr_py = df_py_tags.groupby(['Strat_Zone', 'Channel_Group']).agg({bv_col:'sum', 'HN':'sum'}).reset_index()
        adr_cy['ADR_CY'] = adr_cy[bv_col] / adr_cy['HN']
        adr_py['ADR_PY'] = adr_py[bv_col] / adr_py['HN']
        
        adr_m = pd.merge(adr_cy, adr_py, on=['Strat_Zone', 'Channel_Group'], how='left', suffixes=('', '_OLD'))
        adr_m['YoY_Growth'] = (adr_m['ADR_CY'] - adr_m['ADR_PY']) / adr_m['ADR_PY'] * 100
        
        adr_m['Label_Text'] = [f"<b>{v:,.0f}</b><br><span style='color:{'#28a745' if g>=0 else '#dc3545'}; font-weight:700;'>{g:+.1f}%</span>" if pd.notna(g) else f"<b>{v:,.0f}</b>" for v, g in zip(adr_m['ADR_CY'], adr_m['YoY_Growth'])]
        
        fig_adr_comp = px.bar(adr_m, x='Strat_Zone', y='ADR_CY', color='Channel_Group', barmode='group', text='Label_Text', color_discrete_sequence=['#051C2C', '#A64B35'])
        fig_adr_comp.update_traces(textposition='outside', textfont=dict(size=14))
        fig_adr_comp.update_xaxes(title_text=None)
        fig_adr_comp.update_yaxes(title_text="ADR")
        st.plotly_chart(fig_adr_comp, use_container_width=True)

        # ---------------------------------------------------------------------------------
        # 🌟 CORE ENGINE REPLACEMENT: Dynamic Pickup Forecast Matrix (FIT/MICE Decoupled)
        # ---------------------------------------------------------------------------------
        st.markdown("---")
        st.markdown("<h3 style='color:#051C2C; font-weight:700;'>Dynamic Baseline Forecast Matrix</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color:#F8F9FA; padding:20px; border-radius:8px; border-left:4px solid #A64B35; margin-bottom:20px;">
            <h4 style="margin-top:0; color:#051C2C; font-family:'Playfair Display', serif;">🧠 Next-Gen Forecasting: Decoupled Demand & Formulaic Transparency</h4>
            <p style="font-size:0.95rem; margin-bottom:10px; font-weight: 500;">Methodology Breakdown:</p>
            <ul style="font-size:0.9rem; color:#444; line-height: 1.6;">
                <li><b>1. FIT Pace Ratio</b>: <code>FIT OTB (Ref) / FIT Final (Ref)</code>. Measures historical booking completion rate at this specific window.</li>
                <li><b>2. FIT Unbooked Gap</b>: <code>(FIT OTB (CY) / FIT Pace Ratio) - FIT OTB (CY)</code>. The raw remaining demand yet to be picked up.</li>
                <li><b>3. L15 Velocity Factor</b>: <code>CY Last 15 Days FIT Pickup / Ref Last 15 Days FIT Pickup</code>. Adjusts the gap based on current market momentum.</li>
                <li><b>4. Lead-Time D-Factor</b>: <code>Early Bird Share (Ref) / Early Bird Share (CY)</code>. Dampens future demand if current bookings are heavily front-loaded (>90 days). Maximum 1.0.</li>
                <li><b>5. FIT Dynamic Forecast</b>: <code>FIT OTB (CY) + (FIT Unbooked Gap * Velocity Factor * D-Factor)</code>.</li>
                <li><b>6. Total Omni Forecast</b>: <code>FIT Dynamic Forecast + MICE OTB (CY) + MICE Pipeline Override (Manual)</code>. MICE block demand is decoupled from historical pacing curves.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        latest_sales_date = df['Sales_Date'].dropna().max().date() if not df['Sales_Date'].dropna().empty else datetime.date.today()
        st.info(f"**📅 System Data Cutoff Date:** {latest_sales_date}")

        ref_choice = st.radio("⚙️ EXTRACTED PACE RATIO REFERENCE (Select Benchmark):", ["PY (Previous Year)", "PPY (Pre-Previous Year)"], horizontal=True)
        
        is_py = "PY" in ref_choice and "PPY" not in ref_choice
        df_ref_tags = df_py_tags if is_py else df_ppy_tags
        ref_y_val = actual_y - 1 if is_py else actual_y - 2
        ref_label = "PY" if is_py else "PPY"

        df_ref_full = apply_filters(df, cons_mode, ref_y_val if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, safe_offset(c_start, -(actual_y-ref_y_val)), safe_offset(c_end, -(actual_y-ref_y_val)), datetime.date(2000, 1, 1), datetime.date(2099, 12, 31))
        df_ref_full_tags = assign_strategic_tags(sanitize_channels(df_ref_full[~df_ref_full['Segment'].str.lower().str.contains('mission', na=False)]))

        def compress_iz(idf):
            d = idf.copy()
            d.loc[d['Strat_Zone'] == 'IZ', 'Resort'] = 'INTERZONE CONSOLIDATED'
            if 'Cons_Date' in d.columns and 'Sales_Date' in d.columns:
                d['Is_Early'] = (d['Cons_Date'] - d['Sales_Date']).dt.days >= 90
            return d

        fc_cy = compress_iz(df_cy_tags)
        fc_ref = compress_iz(df_ref_tags)
        fc_full = compress_iz(df_ref_full_tags)

        # Global Velocity (L15)
        fit_l15_cy = fc_cy[(fc_cy['Segment'] != 'MICE') & (fc_cy['Sales_Date'].dt.date >= latest_sales_date - datetime.timedelta(days=15))][bv_col].sum()
        max_ref_s = fc_ref['Sales_Date'].max().date() if not fc_ref['Sales_Date'].dropna().empty else safe_offset(latest_sales_date, -1)
        fit_l15_ref = fc_ref[(fc_ref['Segment'] != 'MICE') & (fc_ref['Sales_Date'].dt.date >= max_ref_s - datetime.timedelta(days=15))][bv_col].sum()
        global_velocity = fit_l15_cy / fit_l15_ref if fit_l15_ref > 0 else 1.0
        global_velocity = max(0.5, min(1.5, global_velocity))

        # FIT vs MICE Aggregation
        def agg_metrics(d, is_full=False):
            fit_mask = d['Segment'] != 'MICE'
            mice_mask = d['Segment'] == 'MICE'
            fit_df = d[fit_mask].groupby(['Strat_Zone', 'Resort']).agg(
                FIT_OTB=(bv_col, 'sum'),
                FIT_Early=(bv_col, lambda x: x[d.loc[x.index, 'Is_Early']].sum()) if not is_full else (bv_col, 'sum')
            ).reset_index()
            mice_df = d[mice_mask].groupby(['Strat_Zone', 'Resort']).agg(MICE_OTB=(bv_col, 'sum')).reset_index()
            return pd.merge(fit_df, mice_df, on=['Strat_Zone', 'Resort'], how='outer').fillna(0)

        resort_cy = agg_metrics(fc_cy)
        resort_ref = agg_metrics(fc_ref)
        resort_full = agg_metrics(fc_full, is_full=True)
        
        resort_ref = resort_ref.rename(columns={'FIT_OTB': 'FIT_OTB_REF', 'FIT_Early': 'FIT_Early_REF'})
        resort_full = resort_full.rename(columns={'FIT_OTB': 'FIT_FULL_REF', 'MICE_OTB': 'MICE_FULL_REF'})
        
        fc_master = resort_cy.merge(resort_ref[['Strat_Zone', 'Resort', 'FIT_OTB_REF', 'FIT_Early_REF']], on=['Strat_Zone', 'Resort'], how='outer').fillna(0)
        fc_master = fc_master.merge(resort_full[['Strat_Zone', 'Resort', 'FIT_FULL_REF', 'MICE_FULL_REF']], on=['Strat_Zone', 'Resort'], how='outer').fillna(0)

        st.markdown("#### ✍️ Bottom-Up Resort Assumption Editor")
        st.markdown("<p style='font-size:0.85rem; color:#6C757D;'>Directly input the expected MICE pipeline in the table below. The model will automatically inject it into the final forecast.</p>", unsafe_allow_html=True)
        
        edit_df = fc_master[['Strat_Zone', 'Resort', 'FIT_OTB', 'MICE_OTB']].copy()
        edit_df['FIT_OTB_M'] = edit_df['FIT_OTB'] / 1_000_000
        edit_df['MICE_OTB_M'] = edit_df['MICE_OTB'] / 1_000_000
        edit_df['MICE_Pipeline_Override (M€)'] = 0.0

        edited_df = st.data_editor(
            edit_df[['Strat_Zone', 'Resort', 'FIT_OTB_M', 'MICE_OTB_M', 'MICE_Pipeline_Override (M€)']],
            column_config={
                "MICE_Pipeline_Override (M€)": st.column_config.NumberColumn(
                    "➕ MICE Pipeline Override (M€)", help="Manual input for expected MICE pipeline (in M€)", min_value=0.0, step=0.1, format="%.2f"
                ),
                "Strat_Zone": "Strategic Zone", "Resort": "Resort / Pool",
                "FIT_OTB_M": st.column_config.NumberColumn("FIT OTB (M€)", disabled=True, format="%.2f"),
                "MICE_OTB_M": st.column_config.NumberColumn("MICE OTB (M€)", disabled=True, format="%.2f"),
            }, hide_index=True, use_container_width=True
        )

        fc_master = fc_master.merge(edited_df[['Strat_Zone', 'Resort', 'MICE_Pipeline_Override (M€)']], on=['Strat_Zone', 'Resort'])
        
        # Calculations Logic
        fc_master['Early_CY_Pct'] = np.where(fc_master['FIT_OTB'] > 0, fc_master['FIT_Early'] / fc_master['FIT_OTB'], 0)
        fc_master['Early_REF_Pct'] = np.where(fc_master['FIT_OTB_REF'] > 0, fc_master['FIT_Early_REF'] / fc_master['FIT_OTB_REF'], 0)
        fc_master['D_Factor'] = np.where(fc_master['Early_CY_Pct'] > 0, fc_master['Early_REF_Pct'] / fc_master['Early_CY_Pct'], 1.0)
        fc_master['D_Factor'] = fc_master['D_Factor'].clip(upper=1.0) 

        fc_master['FIT_Pace'] = np.where(fc_master['FIT_FULL_REF'] > 0, fc_master['FIT_OTB_REF'] / fc_master['FIT_FULL_REF'], 0.75)
        fc_master['FIT_Pace'] = np.where(fc_master['FIT_Pace'] <= 0, 1.0, fc_master['FIT_Pace'])
        
        fc_master['FIT_Unbooked_Gap'] = np.maximum(0, (fc_master['FIT_OTB'] / fc_master['FIT_Pace']) - fc_master['FIT_OTB'])
        
        fc_master['FIT_Predicted'] = fc_master['FIT_OTB'] + (fc_master['FIT_Unbooked_Gap'] * global_velocity * fc_master['D_Factor'])
        fc_master['MICE_Predicted'] = fc_master['MICE_OTB'] + (fc_master['MICE_Pipeline_Override (M€)'] * 1_000_000)
        fc_master['Total_Predicted'] = fc_master['FIT_Predicted'] + fc_master['MICE_Predicted']
        fc_master['Total_Ref_Full'] = fc_master['FIT_FULL_REF'] + fc_master['MICE_FULL_REF']

        fc_master['Var_Ref_Abs'] = fc_master['Total_Predicted'] - fc_master['Total_Ref_Full']
        fc_master['Var_Ref_Pct'] = np.where(fc_master['Total_Ref_Full'] > 0, fc_master['Var_Ref_Abs'] / fc_master['Total_Ref_Full'], 0)
        fc_master = fc_master.sort_values(['Strat_Zone', 'Total_Predicted'], ascending=[True, False])

        # 📌 Top Reference Parameters Calculation
        gt_cy_fit = fc_cy[fc_cy['Segment'] != 'MICE'][bv_col].sum()
        gt_cy_mice = fc_cy[fc_cy['Segment'] == 'MICE'][bv_col].sum()
        gt_cy_tot = gt_cy_fit + gt_cy_mice
        
        gt_ref_fit = fc_ref[fc_ref['Segment'] != 'MICE'][bv_col].sum()
        gt_ref_mice = fc_ref[fc_ref['Segment'] == 'MICE'][bv_col].sum()
        gt_ref_tot = gt_ref_fit + gt_ref_mice

        gt_full_fit = fc_full[fc_full['Segment'] != 'MICE'][bv_col].sum()
        gt_full_mice = fc_full[fc_full['Segment'] == 'MICE'][bv_col].sum()
        gt_full_tot = gt_full_fit + gt_full_mice
        
        gt_fit_ref_otb = resort_ref['FIT_OTB_REF'].sum()
        gt_fit_ref_full = resort_full['FIT_FULL_REF'].sum()
        gt_fit_pace = gt_fit_ref_otb / gt_fit_ref_full if gt_fit_ref_full > 0 else 0
        
        gt_early_cy = resort_cy['FIT_Early'].sum()
        gt_fit_cy = resort_cy['FIT_OTB'].sum()
        gt_early_ref = resort_ref['FIT_Early_REF'].sum()
        
        gt_early_cy_pct = gt_early_cy / gt_fit_cy if gt_fit_cy > 0 else 0
        gt_early_ref_pct = gt_early_ref / gt_fit_ref_otb if gt_fit_ref_otb > 0 else 0
        gt_d_factor = gt_early_ref_pct / gt_early_cy_pct if gt_early_cy_pct > 0 else 1.0
        gt_d_factor = min(1.0, gt_d_factor)

        st.markdown("<h4 style='color:#051C2C; margin-top:30px;'>📊 Global Reference Parameters (For Pace & Gap Context)</h4>", unsafe_allow_html=True)
        rcol1, rcol2, rcol3, rcol4, rcol5, rcol6 = st.columns(6)
        with rcol1: st.markdown(complex_mini_card("Current CY OTB", gt_cy_tot, gt_cy_fit, gt_cy_mice, curr_sym), unsafe_allow_html=True)
        with rcol2: st.markdown(complex_mini_card(f"{ref_label} Actual OTB", gt_ref_tot, gt_ref_fit, gt_ref_mice, curr_sym, "#5C7080"), unsafe_allow_html=True)
        with rcol3: st.markdown(complex_mini_card(f"{ref_label} Final Actuals", gt_full_tot, gt_full_fit, gt_full_mice, curr_sym, "#112E43"), unsafe_allow_html=True)
        with rcol4: st.markdown(mini_metric_card("Global FIT Pace", f"{gt_fit_pace*100:.1f}%"), unsafe_allow_html=True)
        with rcol5: st.markdown(mini_metric_card("L15 Velocity Ratio", f"{global_velocity:.2f}x"), unsafe_allow_html=True)
        with rcol6: st.markdown(mini_metric_card("Lead-Time D-Factor", f"{gt_d_factor:.2f}", "#A64B35"), unsafe_allow_html=True)

        dashboard_title_info = f"Market: {mkt_txt} | Cons: {cons_desc}"
        st.markdown(f"<h4 style='color:#051C2C; margin-top:30px; font-weight:700;'>🏢 Resort-Level Decoupled Forecast Dashboard ({dashboard_title_info}) - Unit: M€</h4>", unsafe_allow_html=True)
        
        # 📌 Simplified Dashboard Table
        html_pred = [CSS_STYLE, '<table class="mckinsey-table"><thead><tr>']
        html_pred.append(f'<th rowspan="2" class="th-main th-dark" style="width:14%;">Strategic Zone</th><th rowspan="2" class="th-main th-dark" style="width:16%;">Resort / Pool</th><th colspan="3" class="th-main th-cy" style="border-right: 2px solid #ffffff;">Tuned Forecast (CY)</th><th rowspan="2" class="th-main th-py" style="width:12%; border-right: 2px solid #ffffff;">[H] {ref_label} Full Final</th><th colspan="2" class="th-main th-var" style="width:16%;">Variance (G vs H)</th></tr><tr>')
        html_pred.append('<th class="th-sub th-cy">FIT Calc.</th><th class="th-sub th-cy">MICE Calc.</th><th class="th-sub th-yellow" style="font-weight:700;">[G] TOTAL FINISH</th><th class="th-sub th-var">Abs</th><th class="th-sub th-var">Var %</th></tr></thead><tbody>')
        
        grand_tot_pred, grand_tot_ref, grand_fit_pred, grand_mice_pred = 0, 0, 0, 0
        for zone in fc_master['Strat_Zone'].unique():
            z_df = fc_master[fc_master['Strat_Zone'] == zone]
            sub_tot_pred = z_df['Total_Predicted'].sum()
            sub_tot_ref = z_df['Total_Ref_Full'].sum()
            sub_fit_pred = z_df['FIT_Predicted'].sum()
            sub_mice_pred = z_df['MICE_Predicted'].sum()
            
            grand_tot_pred += sub_tot_pred
            grand_tot_ref += sub_tot_ref
            grand_fit_pred += sub_fit_pred
            grand_mice_pred += sub_mice_pred
            
            zone_rowspan = len(z_df) + 1 
            first = True
            
            for _, row in z_df.iterrows():
                html_pred.append('<tr>')
                if first:
                    html_pred.append(f'<td rowspan="{zone_rowspan}" class="cell-merged" style="border-right: 2px solid #051C2C !important;">{zone}</td>')
                    first = False
                
                f_pred = row['FIT_Predicted'] / 1_000_000
                m_pred = row['MICE_Predicted'] / 1_000_000
                tot_m = row['Total_Predicted'] / 1_000_000
                ref_m = row['Total_Ref_Full'] / 1_000_000
                v_ref_a = row['Var_Ref_Abs'] / 1_000_000
                v_ref_p = row['Var_Ref_Pct']
                
                html_pred.append(f'<td class="cell-detail-left">{row["Resort"]}</td><td>{f_pred:.2f}</td><td>{m_pred:.2f}</td><td style="border-right: 2px solid #CBD5E1 !important; background-color:#FEF9E7;"><b style="color:#051C2C;">{tot_m:.2f}</b></td><td style="border-right: 2px solid #CBD5E1 !important;">{ref_m:.2f}</td>')
                html_pred.append(f'<td>{format_variance_cell(v_ref_a)}</td><td>{format_variance_cell(v_ref_p, is_pct=True)}</td></tr>')
            
            sub_var_abs = (sub_tot_pred - sub_tot_ref) / 1_000_000
            sub_var_pct = (sub_tot_pred - sub_tot_ref) / sub_tot_ref if sub_tot_ref > 0 else 0
            html_pred.append(f'<tr class="subtotal-row"><td class="cell-detail-left"><b>{zone} Subtotal</b></td><td>{sub_fit_pred/1e6:.2f}</td><td>{sub_mice_pred/1e6:.2f}</td><td style="border-right: 2px solid #CBD5E1 !important; background-color:#FEF9E7;"><b style="color:#051C2C;">{sub_tot_pred/1e6:.2f}</b></td><td style="border-right: 2px solid #CBD5E1 !important;"><b>{sub_tot_ref/1e6:.2f}</b></td><td>{format_variance_cell(sub_var_abs)}</td><td>{format_variance_cell(sub_var_pct, True)}</td></tr>')

        gt_var_abs = (grand_tot_pred - grand_tot_ref) / 1_000_000
        gt_var_pct = (grand_tot_pred - grand_tot_ref) / grand_tot_ref if grand_tot_ref > 0 else 0
        html_pred.append(f'<tr class="grand-total-row"><td colspan="2" class="cell-detail-left"><b>GLOBAL OMNI OUTLOOK FORECAST</b></td><td><b>{grand_fit_pred/1e6:.2f}</b></td><td><b>{grand_mice_pred/1e6:.2f}</b></td><td style="border-right: 2px solid #CBD5E1 !important; background-color:#FDB913;"><b style="color:#051C2C; font-size:1.05rem;">{grand_tot_pred/1e6:.2f}</b></td><td style="border-right: 2px solid #CBD5E1 !important;"><b>{grand_tot_ref/1e6:.2f}</b></td><td>{format_variance_cell(gt_var_abs)}</td><td>{format_variance_cell(gt_var_pct, True)}</td></tr>')
        
        html_pred.append('</tbody></table>')
        st.markdown("".join(html_pred), unsafe_allow_html=True)


    # =================================================================
    # 📋 TAB 4: AUTOMATED WEEKLY DIAGNOSTICS 
    # =================================================================
    with tab4:
        st.markdown("<h2 style='color:#051C2C; font-weight:700;'>📋 Automated Weekly Executive Diagnostics</h2>", unsafe_allow_html=True)
        strat_matrix = build_strategic_summary_matrix(df_cy_tags, df_py_tags, bv_col)
        
        st.markdown("""
        <div style="background-color:#F4F7F9; padding:18px 22px; border-radius:6px; border-left:5px solid #051C2C; margin-bottom:20px;">
            <h4 style="margin-top:0; color:#051C2C; font-weight:600;">🧠 McKinsey Framework Multi-Dimensional Attribution Logic:</h4>
            <p style="font-size:0.92rem; color:#333; line-height:1.6; margin-bottom:0;">
                本交叉矩阵的核心逻辑在于<b>“多维立体盈亏归因”</b>。它打破了传统单一渠道或单一目的地的平面化分析孤岛，通过将<b>4大核心分销组合入口（Strategic Portfolio）</b>与<b>5大战略目的地战区（Strategic Zone）</b>在底盘进行全面交叉。每一条记录能直接向管理层进行高穿透力的本质指引。
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Tabular Visual Overlay: Strategic Port vs Zone Variance")
        
        tab4_cy_tot = df_cy_tags[bv_col].sum()
        tab4_py_tot = df_py_tags[bv_col].sum()
        tab4_diff = tab4_cy_tot - tab4_py_tot
        tab4_pct = (tab4_diff / tab4_py_tot * 100) if tab4_py_tot > 0 else 0.0

        t4c1, t4c2, t4c3, t4c4 = st.columns(4)
        with t4c1:
            st.markdown(f'''
            <div style="text-align:center; background-color:white; padding:12px; border-radius:6px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border-top:4px solid #051C2C;">
                <div style="color:#6C757D; font-size:0.85rem; font-weight:600; margin-bottom:4px;">PORTFOLIO CY TOTAL</div>
                <div style="font-size:1.5rem; font-weight:700; color:#051C2C;">{curr_sym}{format_volume(tab4_cy_tot)}</div>
            </div>
            ''', unsafe_allow_html=True)
        with t4c2:
            st.markdown(f'''
            <div style="text-align:center; background-color:white; padding:12px; border-radius:6px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border-top:4px solid #5C7080;">
                <div style="color:#6C757D; font-size:0.85rem; font-weight:600; margin-bottom:4px;">PORTFOLIO PY TOTAL</div>
                <div style="font-size:1.5rem; font-weight:700; color:#5C7080;">{curr_sym}{format_volume(tab4_py_tot)}</div>
            </div>
            ''', unsafe_allow_html=True)
        with t4c3:
            diff_c = "#28a745" if tab4_diff >= 0 else "#dc3545"
            diff_s = "+" if tab4_diff > 0 else ""
            st.markdown(f'''
            <div style="text-align:center; background-color:white; padding:12px; border-radius:6px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border-top:4px solid {diff_c};">
                <div style="color:#6C757D; font-size:0.85rem; font-weight:600; margin-bottom:4px;">ABSOLUTE VARIANCE</div>
                <div style="font-size:1.5rem; font-weight:700; color:{diff_c};">{curr_sym}{diff_s}{format_volume(tab4_diff)}</div>
            </div>
            ''', unsafe_allow_html=True)
        with t4c4:
            pct_c = "#28a745" if tab4_pct >= 0 else "#dc3545"
            pct_s = "+" if tab4_pct > 0 else ""
            st.markdown(f'''
            <div style="text-align:center; background-color:white; padding:12px; border-radius:6px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border-top:4px solid {pct_c};">
                <div style="color:#6C757D; font-size:0.85rem; font-weight:600; margin-bottom:4px;">YOY GROWTH %</div>
                <div style="font-size:1.5rem; font-weight:700; color:{pct_c};">{pct_s}{tab4_pct:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        strat_matrix['CY_M'] = strat_matrix[f'{bv_col}_CY'] / 1_000_000
        strat_matrix['PY_M'] = strat_matrix[f'{bv_col}_PY'] / 1_000_000
        strat_matrix['Variance_M'] = strat_matrix['Variance'] / 1_000_000
        
        fig_diag_bar = px.bar(
            strat_matrix, x='Strat_Zone', y='Variance_M', color='Strat_Port', barmode='group', 
            text=strat_matrix['Variance_M'].apply(lambda x: f"{x:+.2f}"), 
            color_discrete_sequence=['#051C2C', '#1D263B', '#A64B35', '#A4B6B0'], 
            title="Strategic Portfolio YoY Net Variance Matrix"
        )
        fig_diag_bar.update_traces(textposition='outside')
        fig_diag_bar.update_layout(
            xaxis_title="Strategic Zone",
            yaxis_title="Variance (M€)",
            legend_title="Strategic Portfolio",
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_diag_bar, use_container_width=True)
        
        if st.button("🚀 Trigger McKinsey Lean Executive Weekly Diagnostic Report with Live Tavily Radar"):
            
            try:
                tavily_client = TavilySearchAPIWrapper()
            except Exception as e:
                st.error("🔑 无法初始化 Tavily 引擎，请检查 TAVILY_API_KEY 环境变量配置。")
                st.stop()

            # 📌 1. 后台 Python 自动执行数据硬审计
            cy_total = strat_matrix[f'{bv_col}_CY'].sum()
            py_total = strat_matrix[f'{bv_col}_PY'].sum()
            var_total = cy_total - py_total
            pct_total = (var_total / py_total * 100) if py_total > 0 else 0.0
            
            # 锁定最大出血点和最大增长点 (剔除总数行)
            valid_rows = strat_matrix[strat_matrix['Strat_Zone'] != 'GLOBAL OMNI TOTAL']
            
            if not valid_rows.empty:
                max_drag_row = valid_rows.loc[valid_rows['Variance'].idxmin()]
                max_grow_row = valid_rows.loc[valid_rows['Variance'].idxmax()]
                drag_zone = max_drag_row['Strat_Zone']
                grow_zone = max_grow_row['Strat_Zone']
            else:
                drag_zone = "出境度假村"
                grow_zone = "国内度假村"
            
            # 📌 2. 仪式感流式滚动提示窗
            status_placeholder = st.empty()
            
            with status_placeholder.container():
                st.info(f"📊 内部财务审计完毕：大盘差异 {var_total/1e6:+.2f}M{curr_sym}。已锁定核心病灶区 【{drag_zone}】 与增长引擎区 【{grow_zone}】...")
            
            # 📌 3. 智能化组装“宏观+微观双轨搜索词”
            season_context = cons_desc 
            
            query_macro_drag = f"{season_context} {drag_zone} 中国游客 旅游签证 消费信心 趋势"
            query_micro_drag = f"{season_context} {drag_zone} 航班运力 直航增班 机票价格 燃油税"
            query_macro_grow = f"{season_context} {grow_zone} 旅游 亲子夏令营 避暑 市场调研报告"
            
            search_queries = [query_macro_drag, query_micro_drag, query_macro_grow]
            
            # 📌 4. 触发 Tavily 并发集群检索
            search_results_raw = []
            for idx, q in enumerate(search_queries):
                with status_placeholder.container():
                    st.spinner(f"✈️ 实时雷达已联网：正在精准检索网络情报 [{idx+1}/{len(search_queries)}]: '{q}' ...")
                try:
                    res = tavily_client.results(query=q, max_results=2)
                    for r in res:
                        search_results_raw.append(f"🔍 信号源: {r['title']}\n📝 摘要: {r['content']}\n")
                except Exception as e:
                    search_results_raw.append(f"⚠️ 检索失败 '{q}': {e}")
            
            search_intel_str = "\n".join(search_results_raw)
            
            with status_placeholder.container():
                st.success("👔 外部情报网络脱水完毕！正在交由 CFO 与销售总监联合撰写最终总裁战略备忘录...")
            
            # 📌 5. 拼接最权威置顶的 Grand Total 护栏
            total_header = (
                f"=== GLOBAL OMNI GRAND TOTAL ===\n"
                f"CY Total: {cy_total/1e6:.2f}M{curr_sym}\n"
                f"PY Total: {py_total/1e6:.2f}M{curr_sym}\n"
                f"Variance: {var_total/1e6:+.2f}M{curr_sym} ({pct_total:+.1f}%)\n"
                f"================================\n\n"
            )
            matrix_str = total_header + strat_matrix.to_string(index=False)
            
            # 📌 6. 唤醒模型，吐出字字珠玑的决策报告
            chat_history = []
            task_prompt = "请将内部财务表格中的每一个异动点，无缝缝合进 Tavily 检索回来的实时运力和宏观经济情报中，完成多维深度归因，并输出具体的战术实战行动方案。"
            
            report_out = generate_weekly_diagnostics(
                chart_info, 
                matrix_str, 
                search_intel_str, 
                chat_history, 
                task_prompt
            )
            
            status_placeholder.empty()
            st.markdown("---")
            st.markdown("### 🏢 Executive Weekly Advisory Insight (Tavily Live Net)")
            st.write(report_out)

    # =================================================================
    # 🤖 TAB 5: STRATEGIC AI ADVISOR
    # =================================================================
    with tab5:
        if "messages" not in st.session_state: st.session_state.messages = []
        chat_container = st.container()
        with chat_container:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Ask for strategic gap analysis..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"): st.write(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing context, geopolitics & browsing trends..."):
                        insights = generate_weekly_diagnostics(chart_info, df_cy_tags.head(10).to_string(), "No real-time search for quick chat.", st.session_state.messages[:-1], prompt)
                        st.info(insights)
                        st.session_state.messages.append({"role": "assistant", "content": insights})

# =================================================================
# 🌟 WELCOME SCREEN (Complete Landing Block)
# =================================================================
else:
    welcome_html = """
    <div style="padding: 5rem 2rem; text-align: center; background: linear-gradient(135deg, #051C2C 0%, #1D263B 100%); border-radius: 16px; margin-top: 1rem; box-shadow: 0 20px 40px rgba(0,0,0,0.15);">
        <div style="font-size: 4.5rem; margin-bottom: 0.5rem; color: #A64B35; font-family: serif;">Ψ</div>
        <h1 style="font-family: 'Playfair Display', serif; font-size: 3.5rem; color: #FFFFFF; margin-bottom: 1rem; letter-spacing: 1px;">Executive Intelligence Hub</h1>
        <p style="font-family: 'Inter', sans-serif; font-size: 1.15rem; color: #A4B6B0; max-width: 650px; margin: 0 auto; line-height: 1.6; font-weight: 300;">
            Elevate your sales strategy. Please upload your Sales Data via the sidebar to unlock multi-currency pacing analytics, consumption date precision, full channel-matrix matrix and AI-driven insights.
        </p>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    card_style = "padding: 2rem 1.5rem; background-color: #FFFFFF; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); border-top: 4px solid #A64B35; height: 100%; text-align: center;"
    with c1:
        st.markdown(f'''<div style="{card_style}"><div style="font-size: 2.5rem; margin-bottom: 1rem;">📅</div><h3 style="font-family: 'Playfair Display', serif; color: #051C2C; font-size: 1.4rem; margin-bottom: 0.5rem;">Dual-Date Precision</h3><p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Cross-filter by exact Booking Window and Consumption Dates to pinpoint holiday and campaign performance.</p></div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''<div style="{card_style}"><div style="font-size: 2.5rem; margin-bottom: 1rem;">🌍</div><h3 style="font-family: 'Playfair Display', serif; color: #051C2C; font-size: 1.4rem; margin-bottom: 0.5rem;">Omni-Channel Matrix</h3><p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Deep dive into Segment, Channel groups, and Team structures with McKinsey-grade cross-tabulation and subtotaling.</p></div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''<div style="{card_style}"><div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div><h3 style="font-family: 'Playfair Display', serif; color: #051C2C; font-size: 1.4rem; margin-bottom: 0.5rem;">Conversational AI</h3><p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">A strategic partner with full contextual memory, powered by real-time macro-intelligence and geopolitical context.</p></div>''', unsafe_allow_html=True)
