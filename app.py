import streamlit as st
import pandas as pd
import numpy as np 
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime

# =================================================================
# --- 1. Executive Visual Configuration (McKinsey Strategic UI) ---
# =================================================================
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    :root { --cm-blue: #051C2C; --cm-terracotta: #A64B35; --cm-sage: #A4B6B0; --cm-beige: #F8F9FA; }
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

def generate_weekly_diagnostics(context_info, matrix_summary_str, chat_history, current_prompt):
    sys_prompt = f"""You are the Elite Strategic Revenue Director for ClubMed. 
    Data Scope Environment: {context_info}
    
    🚨 RIGOROUS REGIONAL DEFINITIONS & GEOGRAPHIC ACCURACY (CRITICAL MAPPING CHECK):
    - GC mountain: Refers EXCLUSIVELY to China domestic ski resorts (e.g., Changbaishan, Beidahu, Yabuli). NEVER the European Alps.
    - IZ (Interzone): Long-haul destinations. This INCLUDES European Alps ski (e.g., Grand Massif, Val Thorens, etc.), Canada ski, and Long-haul sun destinations (e.g., Mauritius, Seychelles). NEVER Southeast Asia or Maldives.
    - ESAP mountain: Refers EXCLUSIVELY to Japan Ski products (Sahoro, Tomamu, Kiroro, etc.).
    - ESAP SUN: Covers Southeast Asia beach & tropical sun destinations INCLUDING the Maldives (Kani, Finolhu), Phuket, Bali, Cherating, etc.
    - ABSOLUTE FORBIDDEN MISTAKES: Do NOT confuse GC mountain with the Alps. Do NOT confuse IZ with SE Asia. You will be penalized if you make these fundamental errors.
    
    🚨 MACRO RADAR & GEOPOLITICAL INTELLIGENCE:
    1. Sino-Japanese Relations: Cooling trends restrict proactive official promotion of Japan travel products, adding direct headwinds to ESAP Mountain performance.
    2. Long-Haul Logistics: Geopolitical spikes in the Middle East elevate fuel surcharges, filtering demand out of traditional IZ corridors.
    
    💰 DISTRIBUTION PORTFOLIO MATRIX & REALISTIC STRATEGIC PRAGMATISM:
    - Channel Portfolios: EC w/o Ctrip, Ctrip, MICE, TA.
    - CRITICAL BUSINESS REALISM CONSTRAINT: Do NOT recommend unfeasible measures like "forcing Ctrip to drop commission under 8% or cutting off inventory distribution". In a revenue contraction or volatile recovery window, higher-margin direct channels are preferred, but volume placement via aggregate OTA platforms remains absolutely vital. High-commission realized volume is exponentially better than zero-occupancy empty rooms. Focus on smart margin balancing, product bundling, and tactical inventory gating instead.
    
    🚨 FORMAT CONSTRAINT:
    Start directly with the core analysis paragraphs and actionable bullet points. Do NOT output metadata like "致：...", "发件人：...", "主题：...", or "日期：...".
    """
    messages = [SystemMessage(content=sys_prompt)]
    for msg in chat_history:
        if msg["role"] == "user": messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant": messages.append(AIMessage(content=msg["content"]))
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

    df_cy_raw = apply_filters(df, cons_mode, sel_y if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, c_start, c_end, start_date, end_date)
    df_py_raw = apply_filters(df, cons_mode, sel_y-1 if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, c_start.replace(year=c_start.year-1) if c_start else None, c_end.replace(year=c_end.year-1) if c_end else None, py_start, py_end)
    
    ref_y = sel_y if cons_mode.startswith("Quick") else c_start.year
    df_ppy_raw = apply_filters(df, cons_mode, ref_y-2 if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, c_start.replace(year=c_start.year-2) if c_start else None, c_end.replace(year=c_end.replace(year=c_end.year-2)) if c_end else None, start_date.replace(year=start_date.year-2), end_date.replace(year=end_date.year-2))

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
            gt_py_b += seg_py_bv; gt_py_h += seg_py_hn
            
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
            d_py = apply_filters(idf, mode, cy_y-1, seas, cs.replace(year=cs.year-1) if cs else None, ce.replace(year=ce.replace(year=ce.year-1)) if ce else None, datetime.date(2000,1,1), se-datetime.timedelta(days=365))
            
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
        
        if not sk_filtered.empty:
            sk_filtered['Src_Node'] = np.where(sk_filtered['Market'].str.lower().str.contains('hong|香港', na=False), 'CM Hong Kong', 'CM China')
            sk_grp = sk_filtered.groupby(['Src_Node', 'Strat_Zone'])[bv_col].sum().reset_index()
        
            total_vol = sk_grp[bv_col].sum()
        
            src_tot = sk_grp.groupby('Src_Node')[bv_col].sum()
            src_labels = {s: f"{s}<br>{v/total_vol*100:.1f}%<br>{v/1e6:.1f}M€" for s, v in src_tot.items()}
        
            dest_tot = sk_grp.groupby('Strat_Zone')[bv_col].sum()
            dest_labels = {d: f"{d}<br>{v/total_vol*100:.1f}%<br>{v/1e6:.1f}M€" for d, v in dest_tot.items()}
        
            nodes = list(src_tot.index) + list(dest_tot.index)
            node_map = {name: i for i, name in enumerate(nodes)}
            node_display_labels = [src_labels.get(n, dest_labels.get(n, n)) for n in nodes]
        
            # ==== 📌 桑基图回退到上一彩色版版本 ====
            color_palette = px.colors.qualitative.Set3  
            node_colors = [color_palette[i % len(color_palette)] for i in range(len(nodes))]
            
            sources = [node_map[s] for s in sk_grp['Src_Node']]
            targets = [node_map[t] for t in sk_grp['Strat_Zone']]
            values = sk_grp[bv_col].round(0).tolist()
            link_texts = [f"Flow Volume: {v/1e6:.2f}M€" for v in values]
            
            fig_sankey = go.Figure(data=[go.Sankey(
                arrangement="snap",  # 自动调整节点布局
                node=dict(
                    pad=30,          
                    thickness=25,    
                    line=dict(color="white", width=0.5),
                    label=node_display_labels,
                    color=node_colors,
                    # 强制设定文本颜色为黑色，移除任何阴影或背景效果
                    hoverlabel=dict(bgcolor="white", font=dict(color="black")),
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(100, 100, 150, 0.4)",  # 统一浅色连线
                    customdata=link_texts,
                    hovertemplate='%{source.label} → %{target.label}<br><b>%{customdata}</b><extra></extra>'
                )
            )])
            
            fig_sankey.update_layout(
                height=650,
                # 📌 强制锁定纯黑色字体
                font=dict(size=13, family="Inter", color="black"),
                margin=dict(l=20, r=20, t=40, b=20),
                title_text="Source Market → Strategic Zone Flow (M€)",
                title_font=dict(size=16, color="#051C2C"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # 📌 加入 theme=None: 彻底阻断 Streamlit 对文字强加的灰色半透明底板/阴影特性，还原本色的极简黑字！
            st.plotly_chart(fig_sankey, use_container_width=True, theme=None)
            
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
        
        ta_only = df_cy_tags[(~df_cy_tags['TA_Group'].str.lower().isin(['direct', 'semi-direct', 'nan', '-', 'none', 'web', 'individual', 'fit'])) & (df_cy_tags['Strat_Port'] == 'TA')]
        ta_rank = ta_only.groupby('TA_Group')[bv_col].sum().reset_index().sort_values(bv_col, ascending=False).head(5)
        grand_total_pacing_bv = df_cy_tags[bv_col].sum()
        if not ta_rank.empty and grand_total_pacing_bv > 0:
            ta_rank['Market_Contribution_Rate'] = ta_rank[bv_col] / grand_total_pacing_bv
            fig_ta_rank = px.bar(ta_rank, x='TA_Group', y='Market_Contribution_Rate', text=ta_rank['Market_Contribution_Rate'].apply(lambda x: f"<b>{x*100:.1f}%</b>"), title="Top 5 Wholesaler/TA Groups Market Contribution Rate", color_discrete_sequence=['#1D263B'])
            fig_ta_rank.update_traces(textposition='outside')
            st.plotly_chart(fig_ta_rank, use_container_width=True)

        # ---------------------------------------------------------------------------------
        # 🌟 CORE ENGINE REPLACEMENT: Dynamic Pickup Forecast Matrix (Velocity Tuned)
        # ---------------------------------------------------------------------------------
        st.markdown("---")
        st.markdown("<h3 style='color:#051C2C; font-weight:700;'>Dynamic Baseline Forecast Matrix</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        ### 🧠 McKinsey Methodology: Historical Curve + Velocity Tuning Projection

        **1. Historical Pace Ratio (HPR)** Measures historical booking velocity to define our baseline denominator.  
        > `Pace Ratio = Historical OTB up to Exact Cutoff Date / Historical Season Final Realized (100%)`

        **2. Velocity Tuning Factor (L15D)** Tracks recent 15-day momentum relative to historical speed. (Multiplier > 1 implies acceleration).  
        > `Velocity Factor = CY Last 15 Days Intake Flow / PY Last 15 Days Intake Flow`

        **3. Tuned Predicted Final** Applies the momentum adjustment only to the remaining unbooked gap.  
        > `Tuned Forecast = Current OTB + [(Current OTB / Pace Ratio) - Current OTB] × Velocity Factor`
        """)
        
        latest_sales_date = df['Sales_Date'].dropna().max().date() if not df['Sales_Date'].dropna().empty else datetime.date.today()
        st.info(f"**📅 System Data Cutoff Date:** {latest_sales_date}")

        ref_choice = st.radio("⚙️ EXTRACTED PACE RATIO REFERENCE (Select Benchmark):", ["PY (Previous Year)", "PPY (Pre-Previous Year)"], horizontal=True)
        
        is_py = "PY" in ref_choice and "PPY" not in ref_choice
        df_ref_tags = df_py_tags if is_py else df_ppy_tags
        ref_y_val = sel_y - 1 if is_py else sel_y - 2
        ref_label = "PY" if is_py else "PPY"

        df_ref_full = apply_filters(df, cons_mode, ref_y_val if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, c_start.replace(year=c_start.year-(sel_y-ref_y_val)) if c_start else None, c_end.replace(year=c_end.year-(sel_y-ref_y_val)) if c_end else None, datetime.date(2000, 1, 1), datetime.date(2099, 12, 31))
        df_ref_full_tags = assign_strategic_tags(sanitize_channels(df_ref_full[~df_ref_full['Segment'].str.lower().str.contains('mission', na=False)]))

        full_ref_total_bv = df_ref_full_tags[bv_col].sum()
        current_ref_otb_bv = df_ref_tags[bv_col].sum()
        
        pace_ratio = current_ref_otb_bv / full_ref_total_bv if full_ref_total_bv > 0 else 0.78
        
        cy_15d_tot = df_cy_tags.groupby(df_cy_tags['Sales_Date'].dt.date)[bv_col].sum().tail(15).sum()
        ref_15d_tot = df_ref_tags.groupby(df_ref_tags['Sales_Date'].dt.date)[bv_col].sum().tail(15).sum()
        velocity_factor = cy_15d_tot / ref_15d_tot if ref_15d_tot > 0 else 1.0

        cy_tot_bv = df_cy_tags[bv_col].sum()
        ref_tot_bv = df_ref_tags[bv_col].sum()
        var_abs = cy_tot_bv - ref_tot_bv
        var_pct = var_abs / ref_tot_bv if ref_tot_bv > 0 else 0
        
        st.markdown(f"<h4 style='margin-bottom: 10px; color:#051C2C;'>Selected Period Cumulative Overview ({cy_label} vs {ref_label})</h4>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div style='padding:15px; border-radius:8px; background:white; border-top: 4px solid #051C2C; box-shadow:0 2px 8px rgba(0,0,0,0.05);'><div style='font-size:0.85rem; color:#6C757D; font-weight:600;'>{cy_label} Volume (OTB)</div><div style='font-size:1.8rem; font-weight:700; color:#051C2C;'>{cy_tot_bv/1e6:.2f}M€</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='padding:15px; border-radius:8px; background:white; border-top: 4px solid #5C7080; box-shadow:0 2px 8px rgba(0,0,0,0.05);'><div style='font-size:0.85rem; color:#6C757D; font-weight:600;'>{ref_label} Volume (OTB)</div><div style='font-size:1.8rem; font-weight:700; color:#5C7080;'>{ref_tot_bv/1e6:.2f}M€</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div style='padding:15px; border-radius:8px; background:white; border-top: 4px solid {'#28a745' if var_abs>=0 else '#dc3545'}; box-shadow:0 2px 8px rgba(0,0,0,0.05);'><div style='font-size:0.85rem; color:#6C757D; font-weight:600;'>Variance (Abs)</div><div style='font-size:1.8rem; font-weight:700; color:{'#28a745' if var_abs>=0 else '#dc3545'};'>{var_abs/1e6:+.2f}M€</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div style='padding:15px; border-radius:8px; background:white; border-top: 4px solid {'#28a745' if var_pct>=0 else '#dc3545'}; box-shadow:0 2px 8px rgba(0,0,0,0.05);'><div style='font-size:0.85rem; color:#6C757D; font-weight:600;'>Variance (%)</div><div style='font-size:1.8rem; font-weight:700; color:{'#28a745' if var_pct>=0 else '#dc3545'};'>{var_pct*100:+.1f}%</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; margin-bottom:20px;">
            <div style="background-color:#FAFAFA; padding:15px; border-radius:8px; border:1px solid #EAECEF; width:48%;">
                <div style="color:#6C757D; font-size:0.9rem; font-weight:600;">ACTIVE PACE RATIO ({ref_label})</div>
                <div style="font-size:1.5rem; font-weight:700; color:#051C2C;">{pace_ratio*100:.2f}%</div>
            </div>
            <div style="background-color:#FAFAFA; padding:15px; border-radius:8px; border:1px solid #EAECEF; width:48%;">
                <div style="color:#6C757D; font-size:0.9rem; font-weight:600;">L15D VELOCITY TUNING FACTOR</div>
                <div style="font-size:1.5rem; font-weight:700; color:#A64B35;">{velocity_factor:.2f}x</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        df_cy_tags['Baseline_Final'] = df_cy_tags[bv_col] / pace_ratio
        df_cy_tags['Predicted_Final'] = df_cy_tags[bv_col] + (df_cy_tags['Baseline_Final'] - df_cy_tags[bv_col]) * velocity_factor
        
        df_cy_tags.loc[df_cy_tags['Strat_Zone'] == 'IZ', 'Resort'] = 'INTERZONE CONSOLIDATED'
        cy_pred_g = df_cy_tags.groupby(['Strat_Zone', 'Resort'])['Predicted_Final'].sum().reset_index()
        
        df_ref_full_tags.loc[df_ref_full_tags['Strat_Zone'] == 'IZ', 'Resort'] = 'INTERZONE CONSOLIDATED'
        ref_full_g = df_ref_full_tags.groupby(['Strat_Zone', 'Resort'])[bv_col].sum().reset_index().rename(columns={bv_col: 'Ref_Full_Final'})
        
        f_matrix = pd.merge(cy_pred_g, ref_full_g, on=['Strat_Zone', 'Resort'], how='outer').fillna(0)
        
        f_matrix['Var_Ref_Abs'] = f_matrix['Predicted_Final'] - f_matrix['Ref_Full_Final']
        f_matrix['Var_Ref_Pct'] = np.where(f_matrix['Ref_Full_Final'] > 0, f_matrix['Var_Ref_Abs'] / f_matrix['Ref_Full_Final'], 0)
        
        f_matrix = f_matrix.sort_values(['Strat_Zone', 'Predicted_Final'], ascending=[True, False])
        
        dashboard_title_info = f"Market: {mkt_txt} | Cons: {cons_desc}"
        st.markdown(f"<h4 style='color:#051C2C; margin-top:30px; font-weight:700;'>🏢 Resort-Level Tuned Forecast Dashboard ({dashboard_title_info}) - Unit: M€</h4>", unsafe_allow_html=True)
        
        html_pred = [CSS_STYLE, '<table class="mckinsey-table"><thead><tr>']
        html_pred.append(f'<th rowspan="2" class="th-main th-dark" style="width:20%;">Strategic Zone</th><th rowspan="2" class="th-main th-dark" style="width:20%;">Resort / Pool</th><th rowspan="2" class="th-main th-cy" style="width:15%;">Tuned Forecast Final (CY)</th><th rowspan="2" class="th-main th-py" style="width:15%; border-right: 2px solid #ffffff;">{ref_label} Full Final Realized</th><th colspan="2" class="th-main th-var" style="width:30%;">Vs {ref_label} Full Variance</th></tr><tr>')
        html_pred.append('<th class="th-sub th-var">Abs</th><th class="th-sub th-var">Var %</th></tr></thead><tbody>')
        
        grand_tot_pred, grand_tot_ref = 0, 0
        for zone in f_matrix['Strat_Zone'].unique():
            z_df = f_matrix[f_matrix['Strat_Zone'] == zone]
            sub_tot_pred = z_df['Predicted_Final'].sum()
            sub_tot_ref = z_df['Ref_Full_Final'].sum()
            grand_tot_pred += sub_tot_pred
            grand_tot_ref += sub_tot_ref
            
            zone_rowspan = len(z_df) + 1 
            
            first = True
            for _, row in z_df.iterrows():
                html_pred.append('<tr>')
                if first:
                    html_pred.append(f'<td rowspan="{zone_rowspan}" class="cell-merged" style="border-right: 2px solid #051C2C !important;">{zone}</td>')
                    first = False
                
                p_m = row['Predicted_Final'] / 1_000_000
                ref_m = row['Ref_Full_Final'] / 1_000_000
                v_ref_a = row['Var_Ref_Abs'] / 1_000_000
                v_ref_p = row['Var_Ref_Pct']
                
                html_pred.append(f'<td class="cell-detail-left">{row["Resort"]}</td><td><b style="color:#051C2C;">{p_m:.2f}</b></td><td style="border-right: 2px solid #CBD5E1 !important;">{ref_m:.2f}</td>')
                html_pred.append(f'<td>{format_variance_cell(v_ref_a)}</td><td>{format_variance_cell(v_ref_p, is_pct=True)}</td></tr>')
            
            sub_var_abs = (sub_tot_pred - sub_tot_ref) / 1_000_000
            sub_var_pct = (sub_tot_pred - sub_tot_ref) / sub_tot_ref if sub_tot_ref > 0 else 0
            html_pred.append(f'<tr class="subtotal-row"><td class="cell-detail-left"><b>{zone} Subtotal</b></td><td><b>{sub_tot_pred/1e6:.2f}</b></td><td style="border-right: 2px solid #CBD5E1 !important;"><b>{sub_tot_ref/1e6:.2f}</b></td><td>{format_variance_cell(sub_var_abs)}</td><td>{format_variance_cell(sub_var_pct, True)}</td></tr>')

        gt_var_abs = (grand_tot_pred - grand_tot_ref) / 1_000_000
        gt_var_pct = (grand_tot_pred - grand_tot_ref) / grand_tot_ref if grand_tot_ref > 0 else 0
        html_pred.append(f'<tr class="grand-total-row"><td colspan="2" class="cell-detail-left"><b>GLOBAL OMNI OUTLOOK FORECAST</b></td><td><b>{grand_tot_pred/1e6:.2f}</b></td><td style="border-right: 2px solid #CBD5E1 !important;"><b>{grand_tot_ref/1e6:.2f}</b></td><td>{format_variance_cell(gt_var_abs)}</td><td>{format_variance_cell(gt_var_pct, True)}</td></tr>')
        
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
        
        if st.button("🚀 Trigger McKinsey Lean Executive Weekly Diagnostic Report"):
            with st.spinner("AI 正在结合硬性渠道成本结构与宏观政经地缘变局破局中..."):
                matrix_str = strat_matrix.to_string(index=False)
                chat_history = []
                report_out = generate_weekly_diagnostics(chart_info, matrix_str, chat_history, "基于以上战区及端口数据，融合中日政策环境限制、燃油成本等地缘要素，生成一份深度管理层复盘建议报告。")
                st.markdown("---")
                st.markdown("### 🏢 Executive Weekly Advisory Insight")
                st.success(report_out)

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
                        insights = generate_weekly_diagnostics(chart_info, df_cy_tags.head(10).to_string(), st.session_state.messages[:-1], prompt)
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
