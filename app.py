import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime

# --- 1. Executive Visual Configuration ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { --cm-blue: #1D263B; --cm-terracotta: #A64B35; --cm-sage: #A4B6B0; --cm-beige: #F8F9FA; }
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    div[data-testid="stMetric"] { background-color: white; border-radius: 6px; padding: 15px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); border-top: 3px solid var(--cm-terracotta); border-left: 4px solid var(--cm-blue); }
    .stDataFrame { border: 1px solid #EAECEF; border-radius: 6px; overflow: hidden; background-color: white; }
    div[data-testid="stChatMessage"] { background-color: transparent !important; border: none !important; border-bottom: 1px solid #EAECEF !important; padding: 1.5rem 0.5rem !important; }
    div[data-testid="stChatMessageAvatarUser"], div[data-testid="stChatMessageAvatarAssistant"] { background-color: var(--cm-blue) !important; color: white !important; }
    div[data-testid="stChatMessageAvatarUser"] svg, div[data-testid="stChatMessageAvatarAssistant"] svg { fill: white !important; color: white !important; }
    .stSidebar { background-color: white !important; border-right: 1px solid #EAECEF; }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 2. AI Engine Initialization ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxx" 

llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.1)

# --- 3. Core Data Cleaning ---
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
        '[HN_final]': 'HN'
    }
    data.rename(columns=mapping, inplace=True, errors='ignore')
    
    for col in data.columns:
        if col in ['BV_Euro', 'BV_Locale', 'HN', 'Sales_Date']: continue
        c_up = col.upper()
        if 'BV' in c_up and 'EURO' in c_up and 'BV_Euro' not in data.columns: 
            data.rename(columns={col: 'BV_Euro'}, inplace=True)
        elif 'BV' in c_up and ('LOCAL' in c_up or 'LOCALE' in c_up) and 'BV_Locale' not in data.columns: 
            data.rename(columns={col: 'BV_Locale'}, inplace=True)
        elif 'HN' in c_up and 'FINAL' in c_up and 'HN' not in data.columns: 
            data.rename(columns={col: 'HN'}, inplace=True)
        elif 'SALES_DATE' in c_up and 'Sales_Date' not in data.columns: 
            data.rename(columns={col: 'Sales_Date'}, inplace=True)
            
    if 'BV_Euro' not in data.columns: data['BV_Euro'] = 0.0
    if 'BV_Locale' not in data.columns: data['BV_Locale'] = 0.0
    if 'HN' not in data.columns: data['HN'] = 0.0
    
    for col in ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month']:
        if col in data.columns: data[col] = data[col].astype(str).str.strip()
    
    for col in ['BV_Euro', 'BV_Locale', 'HN']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '').replace(' ', ''), errors='coerce').fillna(0)
    
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    data['Month_Num'] = pd.to_numeric(data['Month_Num'], errors='coerce').fillna(0).astype(int)
    
    if 'Sales_Date' in data.columns: data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    if 'Cons_Date' in data.columns: data['Cons_Date'] = pd.to_datetime(data['Cons_Date'], errors='coerce')
        
    return data

# --- 🌟 Pacing Curve & Variance Generator (NEW) ---
def get_pacing_curve_data(df, cy_year, cons_mode, season, c_start, c_end, sel_markets, sel_ta, bv_col, s_end):
    # Filter CY (Ignore Sales Date initially to get full history up to s_end)
    d_cy = df[df['Year'] == cy_year].copy()
    if cons_mode == "Quick Select (Year/Season)":
        if season == "S1 (Jan-Jun)": d_cy = d_cy[d_cy['Month_Num'].between(1, 6)]
        elif season == "S2 (Jul-Dec)": d_cy = d_cy[d_cy['Month_Num'].between(7, 12)]
    else:
        d_cy = d_cy[(d_cy['Cons_Date'].dt.date >= c_start) & (d_cy['Cons_Date'].dt.date <= c_end)]
        
    if sel_markets: d_cy = d_cy[d_cy['Market'].isin(sel_markets)]
    if sel_ta: d_cy = d_cy[d_cy['TA_Group'].isin(sel_ta)]
    d_cy = d_cy[d_cy['Sales_Date'].dt.date <= s_end]

    # Filter PY (Shifted by 1 year)
    d_py = df[df['Year'] == cy_year - 1].copy()
    if cons_mode == "Quick Select (Year/Season)":
        if season == "S1 (Jan-Jun)": d_py = d_py[d_py['Month_Num'].between(1, 6)]
        elif season == "S2 (Jul-Dec)": d_py = d_py[d_py['Month_Num'].between(7, 12)]
    else:
        try: py_c_s, py_c_e = c_start.replace(year=c_start.year-1), c_end.replace(year=c_end.year-1)
        except ValueError: py_c_s, py_c_e = c_start - datetime.timedelta(days=365), c_end - datetime.timedelta(days=365)
        d_py = d_py[(d_py['Cons_Date'].dt.date >= py_c_s) & (d_py['Cons_Date'].dt.date <= py_c_e)]
        
    if sel_markets: d_py = d_py[d_py['Market'].isin(sel_markets)]
    if sel_ta: d_py = d_py[d_py['TA_Group'].isin(sel_ta)]
    
    try: py_s_end = s_end.replace(year=s_end.year-1)
    except ValueError: py_s_end = s_end - datetime.timedelta(days=365)
    d_py = d_py[d_py['Sales_Date'].dt.date <= py_s_end]

    # Aggregate & Cumulative sum
    cy_daily = d_cy.groupby('Sales_Date')[bv_col].sum().reset_index()
    py_daily = d_py.groupby('Sales_Date')[bv_col].sum().reset_index()

    # Shift PY Sales Dates forward 1 year to align axes!
    py_daily['Sales_Date'] = py_daily['Sales_Date'] + pd.DateOffset(years=1)

    if cy_daily.empty and py_daily.empty: return None

    # Create master timeline
    min_date = min(cy_daily['Sales_Date'].min(), py_daily['Sales_Date'].min())
    if pd.isna(min_date): return None
    
    timeline = pd.date_range(start=min_date, end=pd.to_datetime(s_end), freq='D')
    df_time = pd.DataFrame({'Sales_Date': timeline})

    cy_daily = pd.merge(df_time, cy_daily, on='Sales_Date', how='left').fillna(0)
    py_daily = pd.merge(df_time, py_daily, on='Sales_Date', how='left').fillna(0)

    cy_daily['CY'] = cy_daily[bv_col].cumsum() / 1000
    py_daily['PY'] = py_daily[bv_col].cumsum() / 1000
    
    df_curve = pd.DataFrame({'Sales_Date': timeline, 'CY': cy_daily['CY'], 'PY': py_daily['PY']})
    df_curve['Gap'] = df_curve['CY'] - df_curve['PY']
    return df_curve

def draw_pacing_curve(df_curve, cy_label, py_label, curr_symbol):
    if df_curve is None or df_curve.empty: return go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)
    
    # 1. Cumulative Lines
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['CY'], name=cy_label, mode='lines', line=dict(color='#1D263B', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['PY'], name=py_label, mode='lines', line=dict(color='#A4B6B0', width=2, dash='dash')), row=1, col=1)
    
    # 2. Variance Area
    df_curve['Gap_Pos'] = df_curve['Gap'].clip(lower=0)
    df_curve['Gap_Neg'] = df_curve['Gap'].clip(upper=0)
    
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['Gap_Pos'], name='Ahead (+)', fill='tozeroy', line=dict(color='rgba(0,128,0,0)'), fillcolor='rgba(40,167,69,0.3)', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['Gap_Neg'], name='Behind (-)', fill='tozeroy', line=dict(color='rgba(255,0,0,0)'), fillcolor='rgba(220,53,69,0.3)', showlegend=False), row=2, col=1)

    # 3. Max Gap Annotation
    max_idx = df_curve['Gap'].abs().idxmax()
    if pd.notna(max_idx):
        max_row = df_curve.loc[max_idx]
        fig.add_annotation(
            x=max_row['Sales_Date'], y=max_row['CY'],
            text=f"<b>Max Gap: {max_row['Sales_Date'].strftime('%Y-%b-%d')}</b><br>CY: {curr_symbol}{max_row['CY']:,.0f}k<br>PY: {curr_symbol}{max_row['PY']:,.0f}k<br>Diff: {curr_symbol}{max_row['Gap']:+,.0f}k",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A64B35",
            ax=0, ay=-60, bgcolor="white", bordercolor="#A64B35", borderwidth=1.5, row=1, col=1
        )
        
    # 4. Zero-Crossing Points (Catch up points)
    sign = np.sign(df_curve['Gap'].round(1))
    signchange = ((np.roll(sign, 1) - sign) != 0).astype(int)
    signchange[0] = 0
    crosses = df_curve.index[signchange == 1].tolist()
    
    for idx in crosses:
        if abs(df_curve.loc[idx, 'Gap']) < 5: continue # Ignore micro fluctuations
        c_date, curr_gap = df_curve.loc[idx, 'Sales_Date'], df_curve.loc[idx, 'Gap']
        prev_gap = df_curve.loc[idx-1, 'Gap']
        
        if prev_gap > 0 and curr_gap < 0: txt = f"📉 PY catches up<br>{c_date.strftime('%b %d')}"
        elif prev_gap < 0 and curr_gap > 0: txt = f"🚀 CY overtakes<br>{c_date.strftime('%b %d')}"
        else: continue
            
        fig.add_annotation(x=c_date, y=0, text=txt, showarrow=True, arrowhead=1, ax=0, ay=-40 if curr_gap>0 else 40, bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1, row=2, col=1)

    fig.update_layout(
        title=dict(text="<b>Cumulative Pacing Trajectory & Variance Tracking</b>", font=dict(family="Playfair Display", size=18)),
        hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center'), margin=dict(t=80, b=10)
    )
    
    # Format X axes as Year/Month
    fig.update_xaxes(dtick="M1", tickformat="%Y-%b", showgrid=True, gridcolor='rgba(0,0,0,0.05)', row=1, col=1)
    fig.update_xaxes(dtick="M1", tickformat="%Y-%b", showgrid=True, gridcolor='rgba(0,0,0,0.05)', row=2, col=1)
    
    fig.update_yaxes(title_text=f"Cumulative Vol. ({curr_symbol}k)", showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Gap vs PY", showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinecolor='black', row=2, col=1)
    
    return fig

# --- Plotting Bar & Pie ---
def draw_charts(cy_df, py_df, cy_label, py_label, bv_col, dynamic_title):
    cy_g = cy_df.groupby('Dest_Type')[[bv_col]].sum().reset_index()
    py_g = py_df.groupby('Dest_Type')[[bv_col]].sum().reset_index()
    
    cy_g[bv_col] /= 1000
    py_g[bv_col] /= 1000
    
    combined = pd.merge(cy_g, py_g, on='Dest_Type', how='outer', suffixes=('_CY', '_PY')).fillna(0)
    combined['YoY_Pct'] = np.where(combined[f'{bv_col}_PY'] > 0, (combined[f'{bv_col}_CY'] - combined[f'{bv_col}_PY']) / combined[f'{bv_col}_PY'] * 100, 0)
    
    fig_bar = go.Figure()
    text_cy = [f"<b>{cy:,.0f}k<br>({pct:+.1f}%)</b>" if py > 0 else f"<b>{cy:,.0f}k</b>" for cy, py, pct in zip(combined[f'{bv_col}_CY'], combined[f'{bv_col}_PY'], combined['YoY_Pct'])]
    fig_bar.add_trace(go.Bar(x=combined['Dest_Type'], y=combined[f'{bv_col}_CY'], name=cy_label, marker_color='#1D263B', text=text_cy, textposition='auto', textfont=dict(size=14)))
    fig_bar.add_trace(go.Bar(x=combined['Dest_Type'], y=combined[f'{bv_col}_PY'], name=py_label, marker_color='#A4B6B0', text=[f"<b>{v:,.0f}k</b>" for v in combined[f'{bv_col}_PY']], textposition='auto', textfont=dict(size=14)))
    fig_bar.update_layout(title=dict(text=dynamic_title, font=dict(family="Playfair Display", size=18)), barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=70, b=0), legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'), yaxis=dict(visible=False))

    fig_pie = px.pie(cy_g, values=bv_col, names='Dest_Type', title=f"<b>{cy_label} Share</b>", color_discrete_sequence=['#1D263B', '#A64B35', '#A4B6B0', '#EAECEF'])
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', hole=.3)
    fig_pie.update_layout(showlegend=False, margin=dict(t=50, b=0, l=0, r=0))

    return fig_bar, fig_pie

# --- AI Insights Generator ---
def generate_macro_insights(cy_data, py_data, context_desc, bv_col):
    cy_total = cy_data[bv_col].sum() / 1000
    py_total = py_data[bv_col].sum() / 1000
    pct = ((cy_total - py_total) / py_total * 100) if py_total > 0 else 0
    currency = "k€" if "Euro" in bv_col else "k (Locale)"
    
    sys_prompt = "You are a Strategy Consultant for ClubMed. Analyze the variance between CY and PY. Focus on macro-environmental shifts and strategic behavior. Use 4-5 sentences."
    user_prompt = f"Context: {context_desc}\nCurrency: {currency}\nCY: {cy_total:,.0f} | PY: {py_total:,.0f}\nVariance: {pct:+.1f}%\nBreakdown:\n{cy_data.groupby('Dest_Type')[bv_col].sum().to_string()}"
    
    try:
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        return resp.content
    except:
        return "Insight generation currently unavailable."

def extract_dataframe(resp):
    if isinstance(resp, pd.DataFrame): return resp
    if isinstance(resp, dict) and 'value' in resp and isinstance(resp['value'], pd.DataFrame): return resp['value']
    try: return pd.DataFrame(resp)
    except: return None

# ==========================================
# 🌟 4. Main Business UI
# ==========================================
if uploaded_file := st.sidebar.file_uploader("Upload SalesData.csv", type=['csv']):
    df = load_and_clean(uploaded_file)
    
    with st.sidebar:
        st.markdown("### 🛠️ Global Filters")
        
        bv_selection = st.radio("Display Currency:", ["Euro (€)", "Locale (Original)"])
        bv_col = "BV_Euro" if "Euro" in bv_selection else "BV_Locale"
        currency_symbol = "€" if "Euro" in bv_selection else ""
        
        sel_markets = st.multiselect("Market Select", sorted(df['Market'].unique()))
        sel_ta = st.multiselect("Travel Agency Select", sorted(df['TA_Group'].unique()))

        st.divider()
        
        st.markdown("### 📅 Consumption Window")
        cons_mode = st.radio("Filter By:", ["Quick Select (Year/Season)", "Custom Date Range"])
        
        if cons_mode == "Quick Select (Year/Season)":
            sel_year = st.selectbox("Consumption Year", sorted(df['Year'].unique(), reverse=True), index=0)
            season = st.radio("Season Focus", ["All Year", "S1 (Jan-Jun)", "S2 (Jul-Dec)"])
            cons_start, cons_end = None, None
            cy_label, py_label = f"CY {sel_year}", f"PY {sel_year-1}"
        else:
            sel_year, season = None, None
            max_cons_date = df['Cons_Date'].max().date() if not df['Cons_Date'].dropna().empty else datetime.date.today()
            col_cs, col_ce = st.columns(2)
            cons_start = col_cs.date_input("Cons. Start", value=max_cons_date - datetime.timedelta(days=180))
            cons_end = col_ce.date_input("Cons. End", value=max_cons_date)
            cy_label, py_label = f"CY ({cons_start.year})", f"PY ({cons_start.year - 1})"

        st.divider()
        
        st.markdown("### 📅 Booking Window (Sales)")
        preset = st.selectbox("Quick Range Select", ["Last 3 Months", "Last Week", "Last 1 Month", "Custom Range"])
        max_sales_date = df['Sales_Date'].max().date() if not df['Sales_Date'].dropna().empty else datetime.date.today()
        
        if preset == "Custom Range":
            col_s, col_e = st.columns(2)
            start_date = col_s.date_input("Sales Start", value=max_sales_date - datetime.timedelta(days=90))
            end_date = col_e.date_input("Sales End", value=max_sales_date)
        else:
            days_map = {"Last Week": 7, "Last 1 Month": 30, "Last 3 Months": 90}
            start_date = max_sales_date - datetime.timedelta(days=days_map[preset])
            end_date = max_sales_date

        if start_date <= end_date:
            py_start, py_end = start_date.replace(year=start_date.year-1), end_date.replace(year=end_date.year-1)
        else: st.error("Date Error"); st.stop()

    def apply_ui_filters(input_df, c_mode, y_val, seas_val, c_start, c_end, s_start, s_end):
        d = input_df.copy()
        d = d[(d['Sales_Date'].dt.date >= s_start) & (d['Sales_Date'].dt.date <= s_end)]
        if c_mode == "Quick Select (Year/Season)":
            d = d[d['Year'] == y_val]
            if seas_val == "S1 (Jan-Jun)": d = d[d['Month_Num'].between(1, 6)]
            elif seas_val == "S2 (Jul-Dec)": d = d[d['Month_Num'].between(7, 12)]
        else:
            d = d[(d['Cons_Date'].dt.date >= c_start) & (d['Cons_Date'].dt.date <= c_end)]
            
        if sel_markets: d = d[d['Market'].isin(sel_markets)]
        if sel_ta: d = d[d['TA_Group'].isin(sel_ta)]
        return d

    df_cy_base = apply_ui_filters(df, cons_mode, sel_year, season, cons_start, cons_end, start_date, end_date)
    
    if cons_mode == "Quick Select (Year/Season)":
        py_y_val = sel_year - 1
        py_c_start, py_c_end = None, None
        base_cy_year = sel_year
    else:
        py_y_val = None
        base_cy_year = cons_start.year
        try:
            py_c_start, py_c_end = cons_start.replace(year=cons_start.year-1), cons_end.replace(year=cons_end.year-1)
        except ValueError:
            py_c_start, py_c_end = cons_start - datetime.timedelta(days=365), cons_end - datetime.timedelta(days=365)
            
    df_py_base = apply_ui_filters(df, cons_mode, py_y_val, season, py_c_start, py_c_end, py_start, py_end)

    # --- Dashboard Header & KPIs ---
    st.markdown(f"### 📈 Executive Booking Pacing: {cy_label} vs {py_label}")
    
    cy_bv, py_bv = df_cy_base[bv_col].sum() / 1000, df_py_base[bv_col].sum() / 1000
    cy_hn, py_hn = df_cy_base['HN'].sum(), df_py_base['HN'].sum()
    cy_adr, py_adr = (cy_bv * 1000) / cy_hn if cy_hn > 0 else 0, (py_bv * 1000) / py_hn if py_hn > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Paced BV ({bv_selection.split(' ')[0]})", f"{currency_symbol}{cy_bv:,.0f}k", f"{(cy_bv-py_bv)/py_bv*100:.1f}%" if py_bv>0 else None)
    c2.metric("Paced HN", f"{cy_hn:,.0f}", f"{(cy_hn-py_hn)/py_hn*100:.1f}%" if py_hn>0 else None)
    c3.metric("Current ADR", f"{currency_symbol}{cy_adr:,.0f}", f"{(cy_adr-py_adr)/py_adr*100:.1f}%" if py_adr>0 else None)

    # --- Static Charts (Bar & Pie) ---
    mkt_t = ", ".join(sel_markets) if sel_markets else "All Markets"
    if cons_mode == "Quick Select (Year/Season)": cons_desc = f"{season} {sel_year}"
    else: cons_desc = f"{cons_start} to {cons_end}"
    
    chart_title = f"<b>Booking Pace by Destination Type Asia</b><br><sup style='color: gray;'>{mkt_t} | Consumption: {cons_desc}</sup>"
    
    fig_bar, fig_pie = draw_charts(df_cy_base, df_py_base, cy_label, py_label, bv_col, chart_title)
    
    col_left, col_right = st.columns([2, 1])
    with col_left: st.plotly_chart(fig_bar, use_container_width=True)
    with col_right: st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================
    # 🌟 5. Dynamic Cumulative Pacing Trajectory
    # ==========================================
    st.divider()
    st.markdown("### 🎢 Cumulative Booking Pacing & Variance Trajectory")
    
    df_curve = get_pacing_curve_data(df, base_cy_year, cons_mode, season, cons_start, cons_end, sel_markets, sel_ta, bv_col, end_date)
    fig_curve = draw_pacing_curve(df_curve, cy_label, py_label, currency_symbol)
    st.plotly_chart(fig_curve, use_container_width=True)


    # ==========================================
    # 🌟 6. AI Macro & Strategy Advisor
    # ==========================================
    st.divider()
    st.markdown("### 🤖 Strategy & Macro Advisor")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("E.g., What are the macro factors driving this trend?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Compiling global data..."):
                strict_instr = "YOU MUST OUTPUT EXACTLY ONE CODE BLOCK ENCLOSED IN ```python AND ```. NO TEXT."
                agent = Agent([df_cy_base, df_py_base], config={"llm": llm, "save_charts": False, "custom_instructions": strict_instr})
                
                hacked_prompt = f"""User: "{prompt}". Output only code to filter dfs[0] and dfs[1] by entity name in prompt, add 'Period' column, and result = pd.concat([cy, py])."""
                
                try:
                    response_raw = agent.chat(hacked_prompt)
                    combined_df = extract_dataframe(response_raw)
                    if not isinstance(combined_df, pd.DataFrame) or 'Period' not in combined_df.columns:
                        ai_cy_df, ai_py_df = df_cy_base.copy(), df_py_base.copy()
                    else:
                        ai_cy_df = combined_df[combined_df['Period'] == 'CY']
                        ai_py_df = combined_df[combined_df['Period'] == 'PY']
                    
                    full_context = f"Question: {prompt} | Currency: {bv_selection} | Cons_Window: {cons_desc} | Sales_Window: {start_date} to {end_date}"
                    insights = generate_macro_insights(ai_cy_df, ai_py_df, full_context, bv_col)
                    st.info(insights)
                    st.session_state.messages.append({"role": "assistant", "content": insights})
                except:
                    st.error("Analysis failed. Try rephrasing.")
else:
    welcome_html = """
    <div style="padding: 5rem 2rem; text-align: center; background: linear-gradient(135deg, #1D263B 0%, #2A3650 100%); border-radius: 16px; margin-top: 1rem; box-shadow: 0 20px 40px rgba(0,0,0,0.15);">
        <div style="font-size: 4.5rem; margin-bottom: 0.5rem; color: #A64B35; font-family: serif;">Ψ</div>
        <h1 style="font-family: 'Playfair Display', serif; font-size: 3.5rem; color: #FFFFFF; margin-bottom: 1rem; letter-spacing: 1px;">Executive Intelligence Hub</h1>
        <p style="font-family: 'Inter', sans-serif; font-size: 1.15rem; color: #A4B6B0; max-width: 650px; margin: 0 auto; line-height: 1.6; font-weight: 300;">
            Elevate your sales strategy. Please upload your Sales Data via the sidebar to unlock multi-currency pacing analytics, consumption date precision, and AI-driven macroeconomic insights.
        </p>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    
    card_style = "padding: 2rem 1.5rem; background-color: #FFFFFF; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); border-top: 4px solid #A64B35; height: 100%; text-align: center;"
    
    with c1:
        st.markdown(f'''
        <div style="{card_style}">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📅</div>
            <h3 style="font-family: 'Playfair Display', serif; color: #1D263B; font-size: 1.4rem; margin-bottom: 0.5rem;">Dual-Date Precision</h3>
            <p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Cross-filter by exact Booking Window and Consumption Dates to pinpoint holiday and campaign performance.</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with c2:
        st.markdown(f'''
        <div style="{card_style}">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🌍</div>
            <h3 style="font-family: 'Playfair Display', serif; color: #1D263B; font-size: 1.4rem; margin-bottom: 0.5rem;">Global Perspective</h3>
            <p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Instantly toggle between Euro (€) and Locale currencies, with automated visualizations for Market and category shares.</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with c3:
        st.markdown(f'''
        <div style="{card_style}">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div>
            <h3 style="font-family: 'Playfair Display', serif; color: #1D263B; font-size: 1.4rem; margin-bottom: 0.5rem;">Macro AI Advisor</h3>
            <p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Transform raw sales variances into boardroom-ready narratives, connecting data shifts with global macroeconomic trends.</p>
        </div>
        ''', unsafe_allow_html=True)
