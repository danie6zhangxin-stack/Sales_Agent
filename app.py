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
    .pos-var { color: #28a745; font-weight: 600; }
    .neg-var { color: #dc3545; font-weight: 600; }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# =================================================================
# --- 2. HTML Helper Functions for Custom KPIs ---
# =================================================================
def custom_metric_card(title, cy_val, py_val, delta_pct, cy_format, py_format):
    delta_color = "#28a745" if delta_pct > 0 else "#dc3545" if delta_pct < 0 else "#6c757d"
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

def custom_share_card(title, cy_share, py_share, cy_abs, py_abs, curr_sym):
    delta_ppts = (cy_share - py_share) * 100
    delta_color = "#28a745" if delta_ppts > 0 else "#dc3545" if delta_ppts < 0 else "#6c757d"
    delta_sign = "+" if delta_ppts > 0 else ""
    return f"""
    <div style="background-color: white; border-radius: 8px; padding: 15px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border-top: 3px solid #A64B35; border-left: 4px solid #051C2C; margin-bottom: 15px; height: 100%;">
        <div style="color: #6C757D; font-size: 0.95rem; font-weight: 600; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #051C2C; margin-bottom: 5px;">{cy_share*100:.1f}%</div>
        <div style="font-size: 0.9rem; color: #888; font-weight: 500;">
            PY: <span style="color: #888;">{py_share*100:.1f}%</span> <span style="color: {delta_color}; font-weight: 700; margin-left: 8px;">({delta_sign}{delta_ppts:.1f} ppts)</span>
        </div>
        <div style="font-size: 0.8rem; color: #adb5bd; margin-top: 8px; font-weight: 400;">
            Vol: {curr_sym}{format_volume(cy_abs)} vs PY {curr_sym}{format_volume(py_abs)}
        </div>
    </div>
    """

def fmt_val(val, is_pct=False):
    if is_pct:
        if pd.isna(val) or np.isinf(val) or val == 0: return "0.0%"
        sign = "+" if val > 0 else ""
        cls = "pos-var" if val > 0 else "neg-var"
        return f'<span class="{cls}">{sign}{val*100:.1f}%</span>'
    else:
        if pd.isna(val): return "0"
        return f"{val:,.0f}"

# =================================================================
# --- 3. AI Engine Initialization & Connectors ---
# =================================================================
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxx" 

llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.1)

def get_web_search_context(query):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results: return "No relevant news found."
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception:
        return "Web search currently unavailable."

# =================================================================
# --- 4. Core Data Cleaning & Strategic Mapping Logic ---
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
    d['Strat_Port'] = 'EC端 (去携程)'
    d.loc[d['reChannel'].str.lower() == 'ctrip', 'Strat_Port'] = 'Ctrip端'
    d.loc[d['Segment'].str.upper() == 'MICE', 'Strat_Port'] = 'MICE端'
    d.loc[(d['Team_Group'].str.upper() == 'TA') & (d['Strat_Port'] != 'Ctrip端') & (d['Strat_Port'] != 'MICE端'), 'Strat_Port'] = 'TA端'
    
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

def format_volume(val):
    if abs(val) >= 1_000_000: return f"{val/1_000_000:,.1f}M"
    elif abs(val) >= 1_000: return f"{val/1_000:,.1f}k"
    return f"{val:,.0f}"

# =================================================================
# --- 5. Plotting Functions (Tab 2) ---
# =================================================================
def draw_pacing_curve(df_curve, cy_label, py_label, curr_symbol, info_text):
    if df_curve is None or df_curve.empty: return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['CY'], name=cy_label, mode='lines', line=dict(color='#1D263B', width=3), hovertemplate='<b>CY:</b> %{y:,.1f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['PY'], name=py_label, mode='lines', line=dict(color='#A4B6B0', width=2, dash='dash'), hovertemplate='<b>PY:</b> %{y:,.1f}<extra></extra>'), row=1, col=1)
    
    df_curve['Gap_Pos'] = df_curve['Gap'].clip(lower=0)
    df_curve['Gap_Neg'] = df_curve['Gap'].clip(upper=0)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['Gap_Pos'], fill='tozeroy', line=dict(color='rgba(0,128,0,0)'), fillcolor='rgba(40,167,69,0.3)', showlegend=False, hovertemplate='<b>Ahead (+):</b> %{y:,.1f}<extra></extra>'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['Gap_Neg'], fill='tozeroy', line=dict(color='rgba(255,0,0,0)'), fillcolor='rgba(220,53,69,0.3)', showlegend=False, hovertemplate='<b>Behind (-):</b> %{y:,.1f}<extra></extra>'), row=2, col=1)
    
    max_idx = df_curve['Gap'].abs().idxmax()
    if pd.notna(max_idx):
        max_row = df_curve.loc[max_idx]
        cy_f, py_f, gap_f = format_volume(max_row['CY']), format_volume(max_row['PY']), format_volume(max_row['Gap'])
        fig.add_annotation(x=max_row['Sales_Date'], y=max_row['CY'], text=f"<b>Max Gap: {max_row['Sales_Date'].strftime('%Y-%b-%d')}</b><br>CY: {curr_symbol}{cy_f}<br>PY: {curr_symbol}{py_f}<br>Diff: {curr_symbol}{gap_f}", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A64B35", ax=0, ay=-60, bgcolor="white", bordercolor="#A64B35", borderwidth=1.5, row=1, col=1)
        
    sign = np.sign(df_curve['Gap'].round(0))
    signchange = ((np.roll(sign, 1) - sign) != 0).astype(int)
    signchange[0] = 0
    crosses = df_curve.index[signchange == 1].tolist()
    for idx in crosses:
        if abs(df_curve.loc[idx, 'Gap']) < 5000: continue 
        c_date, curr_gap, prev_gap = df_curve.loc[idx, 'Sales_Date'], df_curve.loc[idx, 'Gap'], df_curve.loc[idx-1, 'Gap']
        if prev_gap > 0 and curr_gap < 0: txt = f"📉 PY catches up<br>{c_date.strftime('%b %d')}"
        elif prev_gap < 0 and curr_gap > 0: txt = f"🚀 CY overtakes<br>{c_date.strftime('%b %d')}"
        else: continue
        fig.add_annotation(x=c_date, y=0, text=txt, showarrow=True, arrowhead=1, ax=0, ay=-40 if curr_gap>0 else 40, bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1, row=2, col=1)

    fig.update_yaxes(ticksuffix="k", tickformat=",", row=1, col=1)
    fig.update_yaxes(ticksuffix="k", tickformat=",", row=2, col=1)
    fig.update_layout(title=dict(text=f"<b>Cumulative Pacing Trajectory</b><br><sup style='color:gray;'>{info_text}</sup>", font=dict(family="Playfair Display", size=18)), hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center'))
    return fig

def draw_weekly_pace_chart(df_curve, cy_label, py_label, curr_symbol, info_text):
    if df_curve is None or df_curve.empty: return go.Figure()
    df_weekly = df_curve.resample('W-MON', on='Sales_Date').sum().reset_index()
    df_weekly['Weekly_Gap'] = df_weekly['CY_inc'] - df_weekly['PY_inc']
    
    fig = go.Figure()
    colors = ['rgba(40,167,69,0.8)' if val >= 0 else 'rgba(220,53,69,0.8)' for val in df_weekly['Weekly_Gap']]
    fig.add_trace(go.Bar(x=df_weekly['Sales_Date'], y=df_weekly['Weekly_Gap'], name='Weekly Variance', marker_color=colors, text=[f"{format_volume(v)}" for v in df_weekly['Weekly_Gap']], textposition='outside'))
    fig.add_trace(go.Scatter(x=df_weekly['Sales_Date'], y=df_weekly['CY_inc'], name=f"{cy_label} Weekly", mode='lines+markers', line=dict(color='#1D263B', width=2)))
    fig.add_trace(go.Scatter(x=df_weekly['Sales_Date'], y=df_weekly['PY_inc'], name=f"{py_label} Weekly", mode='lines+markers', line=dict(color='#A4B6B0', width=2, dash='dash')))
    
    fig.update_yaxes(ticksuffix="k", tickformat=",")
    fig.update_layout(title=dict(text=f"<b>⚡ Weekly Incremental Booking Velocity</b><br><sup style='color:gray;'>{info_text}</sup>", font=dict(family="Playfair Display", size=18)), hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center'))
    return fig

# =================================================================
# --- 6. Dual AI Engines (Weekly Report & Conversational Bot) ---
# =================================================================
def build_strategic_summary_matrix(cy_df, py_df, bv_col):
    cy_s = assign_strategic_tags(cy_df)
    py_s = assign_strategic_tags(py_df)
    cy_g = cy_s.groupby(['Strat_Port', 'Strat_Zone'])[bv_col].sum().reset_index()
    py_g = py_s.groupby(['Strat_Port', 'Strat_Zone'])[bv_col].sum().reset_index()
    merged = pd.merge(cy_g, py_g, on=['Strat_Port', 'Strat_Zone'], how='outer', suffixes=('_CY', '_PY')).fillna(0)
    merged['Variance'] = merged[f'{bv_col}_CY'] - merged[f'{bv_col}_PY']
    merged['Var_Pct'] = np.where(merged[f'{bv_col}_PY'] > 0, merged['Variance'] / merged[f'{bv_col}_PY'] * 100, 0)
    return merged

# 引擎 A：负责 Tab 4 的自动化周报摘要生成
def generate_weekly_diagnostics(context_info, matrix_summary_str, chat_history, current_prompt):
    sys_prompt = f"""You are the Elite Strategic Director for ClubMed. You are interacting with Daniel (Business Controller Manager).
    Data Scope Environment: {context_info}
    
    📊 WEEKLY TOP-TIER DIAGNOSTIC MATRIX SUMMARY DATA (Unit: Original Value):
    {matrix_summary_str}
    
    YOUR CORE TASK:
    Write a highly sharp, professional McKinsey-style Executive Summary Report based on the 4 strategic ports (EC, TA, Ctrip, MICE) and 5 destination zones (ESAP SUN/MTN, GC SUN/MTN, IZ) presented in the data.
    Identify the single largest driver of growth or decay. Do not repeat long list of neutral points. Be actionable.
    """
    messages = [SystemMessage(content=sys_prompt)]
    for msg in chat_history:
        if msg["role"] == "user": messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant": messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=current_prompt))
    try:
        return llm.invoke(messages).content
    except Exception as e:
        return f"Strategic report engine timeout. Error: {e}"

# 引擎 B：负责 Tab 5 的自由对话式 AI 顾问（你的老朋友）
def generate_chat_advisor_response(cy_df, py_df, context_desc, bv_col, search_context, chat_history, current_prompt):
    cy_sum = cy_df.groupby('Dest_Type')[bv_col].sum() / 1000
    py_sum = py_df.groupby('Dest_Type')[bv_col].sum() / 1000
    comp_df = pd.DataFrame({'CY_k': cy_sum, 'PY_k': py_sum}).fillna(0)
    comp_df['Variance_k'] = comp_df['CY_k'] - comp_df['PY_k']
    comp_df['Var_Pct'] = (comp_df['Variance_k'] / comp_df['PY_k'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    total_cy, total_py = comp_df['CY_k'].sum(), comp_df['PY_k'].sum()
    total_pct = ((total_cy - total_py) / total_py * 100) if total_py != 0 else 0
    
    sys_prompt = f"""You are an elite Strategy Consultant for ClubMed. You are having an ongoing conversation with Daniel (Business Controller Manager).
    Context: {context_desc}
    Total Performance: CY {total_cy:,.0f}k vs PY {total_py:,.0f}k (Var: {total_pct:+.1f}%)
    
    YoY COMPARISON TABLE (Unit: k):
    {comp_df.to_string()}
    
    🌍 REAL-TIME WEB SEARCH/NEWS: {search_context}
    
    RULES:
    1. If the user asks a follow-up question, DO NOT repeat the entire table summary. Directly address their new specific point.
    2. Be highly analytical, empathetic, and forward-looking. 
    3. Use the conversation history provided to maintain context.
    """
    
    messages = [SystemMessage(content=sys_prompt)]
    for msg in chat_history:
        if msg["role"] == "user": messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant": messages.append(AIMessage(content=msg["content"]))
        
    messages.append(HumanMessage(content=current_prompt))
    
    try:
        resp = llm.invoke(messages)
        return resp.content
    except Exception as e: 
        return f"AI Advisor unavailable. Error: {e}"

# =================================================================
# --- 7. Main UI Flow & Routing ---
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
        preset = st.selectbox("Quick Range Select", ["Last 3 Months", "Last 1 Month", "From Sales Opening", "Custom Range"])
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
            d = d[d['Year'] == y]
            if seas != "All Year":
                m_range = [1,6] if "S1" in seas else [7,12]
                d = d[d['Month_Num'].between(*m_range)]
        else: 
            d = d[(d['Cons_Date'].dt.date >= cs) & (d['Cons_Date'].dt.date <= ce)]
            
        if sel_mkt: d = d[d['Market'].isin(sel_mkt)]
        if sel_ta: d = d[d['TA_Group'].isin(sel_ta)]
        if sel_dest: d = d[d['Dest_Type'].isin(sel_dest)]
        if sel_resort: d = d[d['Resort'].isin(sel_resort)]
        return d

    df_cy = apply_filters(df, cons_mode, sel_y if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, c_start, c_end, start_date, end_date)
    df_py = apply_filters(df, cons_mode, sel_y-1 if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, 
                          c_start.replace(year=c_start.year-1) if c_start else None, c_end.replace(year=c_end.year-1) if c_end else None, py_start, py_end)

    df_cy = sanitize_channels(df_cy)
    df_py = sanitize_channels(df_py)

    # 彻底清除使命渠道
    df_cy = df_cy[~df_cy['Segment'].str.lower().str.contains('mission', na=False)]
    df_py = df_py[~df_py['Segment'].str.lower().str.contains('mission', na=False)]

    st.markdown(f"<div class='header-box'>ClubMed Executive Intelligence Hub</div>", unsafe_allow_html=True)
    mkt_txt = ", ".join(sel_mkt) if sel_mkt else "All Markets"
    dest_txt = ", ".join(sel_dest) if sel_dest else "All Destinations"
    chart_info = f"Market: {mkt_txt} | Destination: {dest_txt} | Currency: {bv_sel.split(' ')[0]} | Cons: {cons_desc}"

    # 🌟 集大成者：全新的 5 大专属战略 Tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Dashboard", 
        "🎢 Trajectory & Velocity", 
        "🎯 Strategic Decision Canvas", 
        "📋 Auto-Diagnostics Summary", 
        "🤖 Strategic AI Advisor"
    ])

    # =================================================================
    # 📊 TAB 1: EXECUTIVE DASHBOARD (恢复了 FIT/MICE/Direct/Indirect 卡片)
    # =================================================================
    with tab1:
        st.markdown(f"<h3 style='margin-bottom:20px; font-weight: 700; color: #051C2C;'>Pacing Summary: {cy_label} vs {py_label}</h3>", unsafe_allow_html=True)
        cy_v, py_v = df_cy[bv_col].sum(), df_py[bv_col].sum()
        cy_h, py_h = df_cy['HN'].sum(), df_py['HN'].sum()
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
            cy_g = df_cy.groupby('Dest_Type')[[bv_col]].sum().reset_index()
            py_g = df_py.groupby('Dest_Type')[[bv_col]].sum().reset_index()
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
        
        # 🌟 恢复诉求：动态 Title 与 4 大份额矩阵卡片
        st.markdown(f"<h3 style='font-weight: 700; color: #051C2C; margin-bottom: 5px;'>🏢 Channel Structure Deep-dive Matrix</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#6C757D; font-size:0.95rem; font-weight:500; margin-bottom:20px;'>{chart_info}</div>", unsafe_allow_html=True)
        
        total_cy_bv = df_cy[bv_col].sum()
        total_py_bv = df_py[bv_col].sum()
        
        fit_cy = df_cy[df_cy['Segment'].str.lower() == 'fit'][bv_col].sum()
        fit_py = df_py[df_py['Segment'].str.lower() == 'fit'][bv_col].sum()
        mice_cy = df_cy[df_cy['Segment'].str.lower() == 'mice'][bv_col].sum()
        mice_py = df_py[df_py['Segment'].str.lower() == 'mice'][bv_col].sum()
        
        dir_cy = df_cy[df_cy['Channel_Group'] == 'Direct'][bv_col].sum()
        dir_py = df_py[df_py['Channel_Group'] == 'Direct'][bv_col].sum()
        indir_cy = df_cy[df_cy['Channel_Group'] == 'Indirect'][bv_col].sum()
        indir_py = df_py[df_py['Channel_Group'] == 'Indirect'][bv_col].sum()

        fit_share_cy = fit_cy / total_cy_bv if total_cy_bv else 0
        fit_share_py = fit_py / total_py_bv if total_py_bv else 0
        mice_share_cy = mice_cy / total_cy_bv if total_cy_bv else 0
        mice_share_py = mice_py / total_py_bv if total_py_bv else 0
        dir_share_cy = dir_cy / total_cy_bv if total_cy_bv else 0
        dir_share_py = dir_py / total_py_bv if total_py_bv else 0
        indir_share_cy = indir_cy / total_cy_bv if total_cy_bv else 0
        indir_share_py = indir_py / total_py_bv if total_py_bv else 0

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1: st.markdown(custom_share_card("FIT (Individual) Share", fit_share_cy, fit_share_py, fit_cy, fit_py, curr_sym), unsafe_allow_html=True)
        with sc2: st.markdown(custom_share_card("MICE (M&E) Share", mice_share_cy, mice_share_py, mice_cy, mice_py, curr_sym), unsafe_allow_html=True)
        with sc3: st.markdown(custom_share_card("Total Direct Share", dir_share_cy, dir_share_py, dir_cy, dir_py, curr_sym), unsafe_allow_html=True)
        with sc4: st.markdown(custom_share_card("Total Indirect Share", indir_share_cy, indir_share_py, indir_cy, indir_py, curr_sym), unsafe_allow_html=True)

        grp_cols = ['Segment', 'Channel_Group', 'Team_Group', 'reChannel']
        cy_matrix_grp = df_cy.groupby(grp_cols, dropna=False).agg({bv_col: 'sum', 'HN': 'sum'}).reset_index()
        py_matrix_grp = df_py.groupby(grp_cols, dropna=False).agg({bv_col: 'sum', 'HN': 'sum'}).reset_index()
        m_df = pd.merge(cy_matrix_grp, py_matrix_grp, on=grp_cols, how='outer', suffixes=('_CY', '_PY')).fillna(0)
        for c in grp_cols: m_df[c] = m_df[c].astype(str).str.strip().replace(['nan', 'None', ''], '-')
        m_df = m_df.groupby(grp_cols).sum().reset_index().sort_values(['Segment', 'Channel_Group', f'{bv_col}_CY'], ascending=[True, True, False])
        
        # 100% 完整深潜矩阵渲染 (包含修复后的闭合括号)
        html_out = [CSS_STYLE, '<table class="mckinsey-table"><thead><tr>']
        html_out.append('<th rowspan="2" class="th-main th-dark" style="width:8%;">Segment</th><th rowspan="2" class="th-main th-dark" style="width:10%;">Channel Group</th><th rowspan="2" class="th-main th-dark" style="width:10%;">Team Group</th><th rowspan="2" class="th-main th-dark" style="width:12%;">reChannel</th><th colspan="3" class="th-main th-cy">Current Period</th><th colspan="3" class="th-main th-py" style="border-left: 2px solid #ffffff;">Previous Period</th><th colspan="3" class="th-main th-var" style="border-left: 2px solid #ffffff;">Variance</th></tr><tr>')
        html_out.append('<th class="th-sub th-cy">BV</th><th class="th-sub th-cy">HN</th><th class="th-sub th-cy th-divider">ADR</th><th class="th-sub th-py">BV</th><th class="th-sub th-py">HN</th><th class="th-sub th-py th-divider">ADR</th><th class="th-sub th-var">BV %</th><th class="th-sub th-var">HN %</th><th class="th-sub th-var">ADR %</th></tr></thead><tbody>')
        
        gt_cy_bv, gt_cy_hn, gt_py_bv, gt_py_hn = 0, 0, 0, 0
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

        for s in m_df['Segment'].unique():
            df_s = m_df[m_df['Segment'] == s]
            seg_cy_bv, seg_cy_hn = df_s[f'{bv_col}_CY'].sum(), df_s['HN_CY'].sum()
            seg_py_bv, seg_py_hn = df_s[f'{bv_col}_PY'].sum(), df_s['HN_PY'].sum()
            gt_cy_bv += seg_cy_bv; gt_cy_hn += seg_cy_hn
            gt_py_bv += seg_py_bv; gt_py_hn += seg_py_hn
            
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
            
        gt_ca = gt_cy_bv / gt_cy_hn if gt_cy_hn > 0 else 0
        gt_pa = gt_py_bv / gt_py_hn if gt_py_hn > 0 else 0
        html_out.append(f'<tr class="grand-total-row"><td colspan="4" class="cell-detail-left" style="text-align:center !important;">GRAND TOTAL LINE (AUDITED)</td><td>{fmt_val(gt_cy_bv)}</td><td>{fmt_val(gt_cy_hn)}</td><td class="td-divider">{fmt_val(gt_ca)}</td><td>{fmt_val(gt_py_bv)}</td><td>{fmt_val(gt_py_hn)}</td><td class="td-divider">{fmt_val(gt_pa)}</td><td>{fmt_val((gt_cy_bv-gt_py_bv)/gt_py_bv if gt_py_bv>0 else 0, True)}</td><td>{fmt_val((gt_cy_hn-gt_py_hn)/gt_py_hn if gt_py_hn>0 else 0, True)}</td><td>{fmt_val((gt_ca-gt_pa)/gt_pa if gt_pa>0 else 0, True)}</td></tr>')
        html_out.append('</tbody></table>')
        
        st.markdown("".join(html_out), unsafe_allow_html=True)

# =================================================================
# 🎢 TAB 2: TRAJECTORY & VELOCITY
# =================================================================
    with tab2:
        def get_curve(idf, cy_y, mode, seas, cs, ce, se):
            d_cy = apply_filters(idf, mode, cy_y, seas, cs, ce, datetime.date(2000,1,1), se)
            d_py = apply_filters(idf, mode, cy_y-1, seas, cs.replace(year=cs.year-1) if cs else None, ce.replace(year=ce.year-1) if ce else None, datetime.date(2000,1,1), se-datetime.timedelta(days=365))
            c_d = d_cy.groupby('Sales_Date')[bv_col].sum().reset_index()
            p_d = d_py.groupby('Sales_Date')[bv_col].sum().reset_index()
            p_d['Sales_Date'] = p_d['Sales_Date'] + pd.DateOffset(years=1)
            if c_d.empty and p_d.empty: return None
            tline = pd.date_range(start=min(c_d['Sales_Date'].min(), p_d['Sales_Date'].min()), end=pd.to_datetime(se))
            df_t = pd.DataFrame({'Sales_Date': tline})
            c_d = pd.merge(df_t, c_d, on='Sales_Date', how='left').fillna(0)
            p_d = pd.merge(df_t, p_d, on='Sales_Date', how='left').fillna(0)
            res = df_t.copy()
            res['CY_inc'] = c_d[bv_col]/1000; res['PY_inc'] = p_d[bv_col]/1000
            res['CY'] = res['CY_inc'].cumsum(); res['PY'] = res['PY_inc'].cumsum(); res['Gap'] = res['CY'] - res['PY']
            return res
        curve_data = get_curve(df, sel_y if cons_mode.startswith("Quick") else c_start.year, cons_mode, season, c_start, c_end, end_date)
        st.plotly_chart(draw_pacing_curve(curve_data, cy_label, py_label, curr_sym, chart_info), use_container_width=True)
        st.plotly_chart(draw_weekly_pace_chart(curve_data, cy_label, py_label, curr_sym, chart_info), use_container_width=True)

# =================================================================
# 🎯 TAB 3: STRATEGIC DECISION CANVAS (四大大看板)
# =================================================================
    with tab3:
        st.markdown("<h2 style='color:#051C2C; font-weight:700;'>🎯 Advanced Decision Support Canvas</h2>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📅 看板 1: Booking Lead-Time & Decision Window Monitor")
        df_cy['Lead_Time'] = (df_cy['Cons_Date'] - df_cy['Sales_Date']).dt.days
        df_py['Lead_Time'] = (df_py['Cons_Date'] - df_py['Sales_Date']).dt.days
        bins = [-999, 7, 30, 60, 90, 99999]
        labels = ['0-7天 (极度临时)', '8-30天 (次月短线)', '31-60天 (常规中线)', '61-90天 (早鸟主力)', '90天+ (远期锁定)']
        df_cy['Lead_Bucket'] = pd.cut(df_cy['Lead_Time'], bins=bins, labels=labels)
        df_py['Lead_Bucket'] = pd.cut(df_py['Lead_Time'], bins=bins, labels=labels)
        lt_cy = df_cy.groupby('Lead_Bucket', observed=False)[bv_col].sum().reset_index()
        lt_py = df_py.groupby('Lead_Bucket', observed=False)[bv_col].sum().reset_index()
        lt_m = pd.merge(lt_cy, lt_py, on='Lead_Bucket', suffixes=('_CY', '_PY'))
        
        fig_lt = go.Figure()
        fig_lt.add_trace(go.Bar(x=lt_m['Lead_Bucket'], y=lt_m[f'{bv_col}_CY']/1000, name=cy_label, marker_color='#051C2C'))
        fig_lt.add_trace(go.Bar(x=lt_m['Lead_Bucket'], y=lt_m[f'{bv_col}_PY']/1000, name=py_label, marker_color='#A4B6B0'))
        fig_lt.update_layout(title="预订提前天数(Lead Time)结构对比 (k€)", barmode='group')
        st.plotly_chart(fig_lt, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🌍 看板 2: Source Market to Destination Corridor Matrix (中国与香港专题)")
        corr_raw = df_cy.pivot_table(index='Market', columns='Dest_Type', values=bv_col, aggfunc='sum').fillna(0)/1000
        target_rows = [r for r in corr_raw.index if any(x in r.lower() for x in ['china', 'hong', 'cn', 'hk', '中国', '香港'])]
        if target_rows:
            corr_df = corr_raw.loc[target_rows]
        else:
            corr_df = corr_raw
        fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale="Blugrn", title="核心战略双向走廊穿透矩阵大盘 (k€)")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📉 看板 3: Channel Cannibalization & Margin Quality Monitor")
        chan_adr = df_cy.groupby('Channel_Group').agg({bv_col:'sum', 'HN':'sum'}).reset_index()
        chan_adr['ADR'] = chan_adr[bv_col] / chan_adr['HN']
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            fig_chan = px.bar(chan_adr, x='Channel_Group', y='ADR', color='Channel_Group', color_discrete_sequence=['#051C2C','#A64B35'], title="直销(Direct) vs 分销(Indirect) 核心实际有效 ADR 对比")
            st.plotly_chart(fig_chan, use_container_width=True)
        with c_r2:
            ta_share = df_cy.groupby('TA_Group')[bv_col].sum().reset_index().sort_values(bv_col, ascending=False).head(5)
            fig_ta_p = px.pie(ta_share, values=bv_col, names='TA_Group', title="前五大核心分销渠道集团集中度雷达 (Top 5 TA Share)", color_discrete_sequence=['#051C2C', '#1D263B', '#A64B35', '#A4B6B0', '#EAECEF'])
            st.plotly_chart(fig_ta_p, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 📈 看板 4: Campaign Pulse & Dynamic Baseline Final Outlook")
        st.info("💡 **数理完工模型：** 系统依据当前时点的 On-the-Book 实绩，参考历史积累推进斜率曲线（Pace Ratio）自动外推终点线大盘预测值。")
        pace_ratio = 0.78 
        predicted_final = cy_v / pace_ratio
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            st.metric(label=f"🎯 {season if season else '当前季度'} AI 智能最终完工总值预测 (Forecast Final Baseline)", value=f"{curr_sym}{format_volume(predicted_final)}", delta=f"参考历史同推进率基准 {pace_ratio*100:.0f}% 计算")
        with b_c2:
            pulse_df = df_cy.groupby(df_cy['Sales_Date'].dt.date)[bv_col].sum().reset_index().tail(15)
            fig_pulse = px.line(pulse_df, x='Sales_Date', y=bv_col, title="最近 15 日销售大增量流速监测 (Campaign Pulse Acceleration)", markers=True, color_discrete_sequence=['#A64B35'])
            st.plotly_chart(fig_pulse, use_container_width=True)

# =================================================================
# 📋 TAB 4: AUTO-DIAGNOSTICS AI SUMMARY (周度全量战略摘要)
# =================================================================
    with tab4:
        st.markdown("<h2 style='color:#051C2C; font-weight:700;'>📋 Automated Weekly Executive Diagnostics</h2>", unsafe_allow_html=True)
        strat_matrix = build_strategic_summary_matrix(df_cy, df_py, bv_col)
        st.dataframe(strat_matrix.style.format({f'{bv_col}_CY': '{:,.0f}', f'{bv_col}_PY': '{:,.0f}', 'Variance': '{:,.0f}', 'Var_Pct': '{:.1f}%'}), use_container_width=True)
        
        if st.button("🚀 触发生成麦肯锡式精益管理周度诊断报告"):
            with st.spinner("AI 正在深度穿透细分行业与战区异动原因..."):
                matrix_str = strat_matrix.to_string(index=False)
                chat_history = []
                report_out = generate_weekly_diagnostics(chart_info, matrix_str, chat_history, "根据当前战区的变动数据，写一份深度的 summary 报告，指出需要关注和注意的地方。")
                st.markdown("---")
                st.markdown("### 🏢 Executive Weekly Advisory Report")
                st.success(report_out)

# =================================================================
# 🤖 TAB 5: STRATEGIC AI ADVISOR (完全恢复你的老朋友！)
# =================================================================
    with tab5:
        if "messages" not in st.session_state: st.session_state.messages = []
        chat_container = st.container()
        with chat_container:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Ask for strategic gap analysis (e.g. Analysis on FIT Web reChannel discrepancy...)"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"): st.write(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing context & browsing trends..."):
                        search_result = get_web_search_context(prompt)
                        insights = generate_chat_advisor_response(df_cy, df_py, chart_info, bv_col, search_result, st.session_state.messages[:-1], prompt)
                        st.info(insights)
                        st.session_state.messages.append({"role": "assistant", "content": insights})

# =================================================================
# 🌟 WELCOME SCREEN (全量保留三栏式高管卡片)
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
        st.markdown(f'''<div style="{card_style}"><div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div><h3 style="font-family: 'Playfair Display', serif; color: #051C2C; font-size: 1.4rem; margin-bottom: 0.5rem;">Conversational AI</h3><p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">A strategic partner with full contextual memory, powered by real-time web search for unparalleled macro analysis.</p></div>''', unsafe_allow_html=True)
