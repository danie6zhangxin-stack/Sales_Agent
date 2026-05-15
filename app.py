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

# --- 1. Executive Visual Configuration (McKinsey Strategic UI) ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    :root { --cm-blue: #051C2C; --cm-terracotta: #A64B35; --cm-sage: #A4B6B0; --cm-beige: #F8F9FA; }
    .main { background-color: #FAFAFA; font-family: 'Inter', sans-serif; }
    
    /* 🌟 Centralized Boxed Header Styling */
    .header-box {
        background-color: var(--cm-blue);
        color: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 30px 0 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .premium-sub-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--cm-blue);
        margin: 40px 0 20px 0;
        text-align: center;
        width: 100%;
        border-bottom: 2px solid var(--cm-terracotta);
        padding-bottom: 10px;
        display: inline-block;
    }
    
    /* 🌟 Top Control Panel */
    .filter-container {
        background-color: white;
        padding: 20px 25px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border-top: 4px solid var(--cm-blue);
        margin-bottom: 25px;
    }
    
    div[data-testid="stMetric"] { background-color: white; border-radius: 6px; padding: 15px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border-top: 3px solid var(--cm-terracotta); border-left: 4px solid var(--cm-blue); }
    
    /* 🌟 Professional Chat Icons */
    div[data-testid="stChatMessageAvatarUser"] { background-color: #34495E !important; }
    div[data-testid="stChatMessageAvatarAssistant"] { background-color: #051C2C !important; }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 2. AI Engine & Web Search Initialization ---
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
    except Exception as e:
        return "Web search currently unavailable. Focus strictly on internal data."

# --- 3. Core Data Cleaning (VIP Mapping) ---
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
    if 'BV_Euro' not in data.columns: data['BV_Euro'] = 0.0
    if 'BV_Locale' not in data.columns: data['BV_Locale'] = 0.0
    if 'HN' not in data.columns: data['HN'] = 0.0
    for col in ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month']:
        if col in data.columns: data[col] = data[col].astype(str).str.strip()
    for col in ['BV_Euro', 'BV_Locale', 'HN']:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '').replace(' ', ''), errors='coerce').fillna(0)
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    if 'Sales_Date' in data.columns: data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    if 'Cons_Date' in data.columns: data['Cons_Date'] = pd.to_datetime(data['Cons_Date'], errors='coerce')
    return data

# --- 4. Plotting & Chart Generation ---
def draw_pacing_curve(df_curve, cy_label, py_label, curr_symbol, info_text):
    if df_curve is None or df_curve.empty: return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['CY'], name=cy_label, mode='lines', line=dict(color='#1D263B', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['PY'], name=py_label, mode='lines', line=dict(color='#A4B6B0', width=2, dash='dash')), row=1, col=1)
    
    df_curve['Gap_Pos'] = df_curve['Gap'].clip(lower=0)
    df_curve['Gap_Neg'] = df_curve['Gap'].clip(upper=0)
    
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['Gap_Pos'], fill='tozeroy', line=dict(color='rgba(0,128,0,0)'), fillcolor='rgba(40,167,69,0.3)', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_curve['Sales_Date'], y=df_curve['Gap_Neg'], fill='tozeroy', line=dict(color='rgba(255,0,0,0)'), fillcolor='rgba(220,53,69,0.3)', showlegend=False), row=2, col=1)
    
    max_idx = df_curve['Gap'].abs().idxmax()
    if pd.notna(max_idx):
        max_row = df_curve.loc[max_idx]
        fig.add_annotation(
            x=max_row['Sales_Date'], y=max_row['CY'],
            text=f"<b>Max Gap: {max_row['Sales_Date'].strftime('%Y-%b-%d')}</b><br>CY: {curr_symbol}{max_row['CY']:,.0f}k<br>PY: {curr_symbol}{max_row['PY']:,.0f}k<br>Diff: {curr_symbol}{max_row['Gap']:+,.0f}k",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A64B35",
            ax=0, ay=-60, bgcolor="white", bordercolor="#A64B35", borderwidth=1.5, row=1, col=1
        )
        
    sign = np.sign(df_curve['Gap'].round(1))
    signchange = ((np.roll(sign, 1) - sign) != 0).astype(int)
    signchange[0] = 0
    crosses = df_curve.index[signchange == 1].tolist()
    for idx in crosses:
        if abs(df_curve.loc[idx, 'Gap']) < 5: continue 
        c_date, curr_gap, prev_gap = df_curve.loc[idx, 'Sales_Date'], df_curve.loc[idx, 'Gap'], df_curve.loc[idx-1, 'Gap']
        if prev_gap > 0 and curr_gap < 0: txt = f"📉 PY catches up<br>{c_date.strftime('%b %d')}"
        elif prev_gap < 0 and curr_gap > 0: txt = f"🚀 CY overtakes<br>{c_date.strftime('%b %d')}"
        else: continue
        fig.add_annotation(x=c_date, y=0, text=txt, showarrow=True, arrowhead=1, ax=0, ay=-40 if curr_gap>0 else 40, bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1, row=2, col=1)

    # 🌟 Format Y-axis with 'k'
    fig.update_yaxes(ticksuffix="k", tickformat=",", row=1, col=1)
    fig.update_yaxes(ticksuffix="k", tickformat=",", row=2, col=1)
    
    fig.update_layout(title=dict(text=f"<b>Cumulative Pacing Trajectory</b><br><sup style='color:gray;'>{info_text}</sup>", font=dict(family="Playfair Display", size=18)),
                      hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center'))
    return fig

def draw_weekly_pace_chart(df_curve, cy_label, py_label, curr_symbol, info_text):
    if df_curve is None or df_curve.empty: return go.Figure()
    df_weekly = df_curve.resample('W-MON', on='Sales_Date').sum().reset_index()
    df_weekly['Weekly_Gap'] = df_weekly['CY_inc'] - df_weekly['PY_inc']
    
    fig = go.Figure()
    colors = ['rgba(40,167,69,0.8)' if val >= 0 else 'rgba(220,53,69,0.8)' for val in df_weekly['Weekly_Gap']]
    fig.add_trace(go.Bar(x=df_weekly['Sales_Date'], y=df_weekly['Weekly_Gap'], marker_color=colors, text=[f"{v:+,.0f}k" for v in df_weekly['Weekly_Gap']], textposition='outside'))
    fig.add_trace(go.Scatter(x=df_weekly['Sales_Date'], y=df_weekly['CY_inc'], name=f"{cy_label} Weekly", mode='lines+markers', line=dict(color='#1D263B', width=2)))
    fig.add_trace(go.Scatter(x=df_weekly['Sales_Date'], y=df_weekly['PY_inc'], name=f"{py_label} Weekly", mode='lines+markers', line=dict(color='#A4B6B0', width=2, dash='dash')))
    
    # 🌟 Format Y-axis with 'k'
    fig.update_yaxes(ticksuffix="k", tickformat=",")
    
    fig.update_layout(title=dict(text=f"<b>⚡ Weekly Incremental Booking Velocity</b><br><sup style='color:gray;'>{info_text}</sup>", font=dict(family="Playfair Display", size=18)),
                      hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center'))
    return fig

# --- 5. AI Advisor with Pivot Table (Anti-Hallucination) ---
def generate_macro_insights(cy_df, py_df, context_desc, bv_col, search_context):
    cy_sum = cy_df.groupby('Dest_Type')[bv_col].sum() / 1000
    py_sum = py_df.groupby('Dest_Type')[bv_col].sum() / 1000
    comp_df = pd.DataFrame({'CY_k': cy_sum, 'PY_k': py_sum}).fillna(0)
    comp_df['Variance_k'] = comp_df['CY_k'] - comp_df['PY_k']
    comp_df['Var_Pct'] = (comp_df['Variance_k'] / comp_df['PY_k'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    total_cy, total_py = comp_df['CY_k'].sum(), comp_df['PY_k'].sum()
    total_pct = ((total_cy - total_py) / total_py * 100) if total_py != 0 else 0
    
    sys_prompt = "You are a Strategy Consultant for ClubMed. You MUST base your analysis on the YoY COMPARISON TABLE below. Do NOT assume data is missing; if CY is 0, it means zero sales. Integrate Web Search Context if relevant."
    user_prompt = f"""Context: {context_desc}
    Total Performance: CY {total_cy:,.0f}k vs PY {total_py:,.0f}k (Var: {total_pct:+.1f}%)
    
    YoY COMPARISON TABLE (Unit: k):
    {comp_df.to_string()}
    
    🌍 WEB SEARCH CONTEXT:
    {search_context}
    
    Explain the performance trend and answer the user's question clearly in 4-5 sentences."""
    
    try:
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        return resp.content
    except: return "AI Advisor unavailable."

# ==========================================
# 🌟 6. Main UI Flow & Routing
# ==========================================
if uploaded_file := st.sidebar.file_uploader("Upload SalesData.csv", type=['csv']):
    df = load_and_clean(uploaded_file)
    
    # 🌟 Global Filters (Top)
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

    # 🌟 Sidebar Time Filters
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
            max_c = df['Cons_Date'].max().date(); c_start = st.date_input("Cons. Start", max_c - datetime.timedelta(days=180))
            c_end = st.date_input("Cons. End", max_c); cons_desc = f"{c_start} to {c_end}"
            cy_label, py_label = f"CY ({c_start.year})", f"PY ({c_start.year-1})"
            df_cons_filtered = df[(df['Cons_Date'].dt.date >= c_start) & (df['Cons_Date'].dt.date <= c_end)]

        st.divider()
        st.markdown("### ⏱️ Booking Window (Sales)")
        preset = st.selectbox("Quick Range Select", ["Last 3 Months", "Last 1 Month", "From Sales Opening", "Custom Range"])
        max_s = df['Sales_Date'].max().date() if not df['Sales_Date'].dropna().empty else datetime.date.today()
        
        # 🌟 From Sales Opening Logic
        if preset == "From Sales Opening":
            start_date = df_cons_filtered['Sales_Date'].min().date() if not df_cons_filtered.empty and not pd.isna(df_cons_filtered['Sales_Date'].min()) else max_s - datetime.timedelta(days=365)
            end_date = max_s
            st.info(f"Opening Date Detetced: {start_date}")
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
        else: d = d[(d['Cons_Date'].dt.date >= cs) & (d['Cons_Date'].dt.date <= ce)]
        if sel_mkt: d = d[d['Market'].isin(sel_mkt)]
        if sel_ta: d = d[d['TA_Group'].isin(sel_ta)]
        if sel_dest: d = d[d['Dest_Type'].isin(sel_dest)]
        if sel_resort: d = d[d['Resort'].isin(sel_resort)]
        return d

    df_cy = apply_filters(df, cons_mode, sel_y if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, c_start, c_end, start_date, end_date)
    df_py = apply_filters(df, cons_mode, sel_y-1 if cons_mode.startswith("Quick") else None, season if cons_mode.startswith("Quick") else None, 
                          c_start.replace(year=c_start.year-1) if c_start else None, c_end.replace(year=c_end.year-1) if c_end else None, py_start, py_end)

    # 🌟 Centralized Box Header
    st.markdown(f"<div class='header-box'>Executive Booking Pacing: {cy_label} vs {py_label}</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    cy_v, py_v = df_cy[bv_col].sum()/1000, df_py[bv_col].sum()/1000
    cy_h, py_h = df_cy['HN'].sum(), df_py['HN'].sum()
    c1.metric(f"Paced BV ({bv_sel.split(' ')[0]})", f"{curr_sym}{cy_v:,.0f}k", f"{(cy_v-py_v)/py_v*100:.1f}%" if py_v!=0 else None)
    c2.metric("Paced HN", f"{cy_h:,.0f}", f"{(cy_h-py_h)/py_h*100:.1f}%" if py_h!=0 else None)
    c3.metric("Current ADR", f"{curr_sym}{(cy_v*1000)/cy_h if cy_h>0 else 0:,.0f}", None)

    mkt_txt = ", ".join(sel_mkt) if sel_mkt else "All Markets"
    chart_info = f"Currency: {bv_sel} | Market: {mkt_txt} | Cons: {cons_desc}"
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        cy_g = df_cy.groupby('Dest_Type')[[bv_col]].sum().reset_index()
        py_g = df_py.groupby('Dest_Type')[[bv_col]].sum().reset_index()
        combined = pd.merge(cy_g, py_g, on='Dest_Type', how='outer', suffixes=('_CY', '_PY')).fillna(0)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=combined['Dest_Type'], y=combined[f'{bv_col}_CY']/1000, name=cy_label, marker_color='#051C2C', text=[f"<b>{v:,.0f}k</b>" for v in combined[f'{bv_col}_CY']/1000], textposition='outside', textangle=0))
        fig_bar.add_trace(go.Bar(x=combined['Dest_Type'], y=combined[f'{bv_col}_PY']/1000, name=py_label, marker_color='#A4B6B0', text=[f"<b>{v:,.0f}k</b>" for v in combined[f'{bv_col}_PY']/1000], textposition='outside', textangle=0))
        fig_bar.update_layout(title=f"Booking Pace by Dest Type<br><sup style='color:gray;'>{chart_info}</sup>", barmode='group', margin=dict(t=100))
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_r:
        fig_pie = px.pie(cy_g, values=bv_col, names='Dest_Type', title=f"{cy_label} Share", color_discrete_sequence=['#051C2C', '#A64B35', '#A4B6B0', '#EAECEF'])
        fig_pie.update_traces(textinfo='percent+label', hole=.3); st.plotly_chart(fig_pie, use_container_width=True)

    # --- Pacing Trajectory ---
    st.markdown("<div align='center' class='premium-sub-header'>🎢 Booking Trajectory & Velocity Analysis</div>", unsafe_allow_html=True)
    
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
        res = df_t.copy(); res['CY_inc'] = c_d[bv_col]/1000; res['PY_inc'] = p_d[bv_col]/1000
        res['CY'] = res['CY_inc'].cumsum(); res['PY'] = res['PY_inc'].cumsum(); res['Gap'] = res['CY'] - res['PY']
        return res

    curve_data = get_curve(df, sel_y if cons_mode.startswith("Quick") else c_start.year, cons_mode, season, c_start, c_end, end_date)
    st.plotly_chart(draw_pacing_curve(curve_data, cy_label, py_label, curr_sym, chart_info), use_container_width=True)
    st.plotly_chart(draw_weekly_pace_chart(curve_data, cy_label, py_label, curr_sym, chart_info), use_container_width=True)

    # --- AI Chat ---
    st.markdown("<div align='center' class='premium-sub-header'>🤖 Strategic AI Advisor</div>", unsafe_allow_html=True)
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Ask for strategic gap analysis (e.g. Why are we behind PY despite the early bird promo?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing YoY Comparisons & Scanning Web..."):
                search_result = get_web_search_context(prompt)
                # Ensure we pass the precise table to avoid hallucination
                insights = generate_macro_insights(df_cy, df_py, chart_info, bv_col, search_result)
                st.info(insights)
                st.session_state.messages.append({"role": "assistant", "content": insights})

else:
    # ==========================================
    # 🌟 7. Full Premium Welcome UI (Preserved!)
    # ==========================================
    welcome_html = """
    <div style="padding: 5rem 2rem; text-align: center; background: linear-gradient(135deg, #051C2C 0%, #1D263B 100%); border-radius: 16px; margin-top: 1rem; box-shadow: 0 20px 40px rgba(0,0,0,0.15);">
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
            <h3 style="font-family: 'Playfair Display', serif; color: #051C2C; font-size: 1.4rem; margin-bottom: 0.5rem;">Dual-Date Precision</h3>
            <p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Cross-filter by exact Booking Window and Consumption Dates to pinpoint holiday and campaign performance.</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with c2:
        st.markdown(f'''
        <div style="{card_style}">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🌍</div>
            <h3 style="font-family: 'Playfair Display', serif; color: #051C2C; font-size: 1.4rem; margin-bottom: 0.5rem;">Global Perspective</h3>
            <p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Instantly toggle between Euro (€) and Locale currencies, with automated visualizations for Market and category shares.</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with c3:
        st.markdown(f'''
        <div style="{card_style}">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div>
            <h3 style="font-family: 'Playfair Display', serif; color: #051C2C; font-size: 1.4rem; margin-bottom: 0.5rem;">Macro AI Advisor</h3>
            <p style="color: #6c757d; font-size: 0.95rem; line-height: 1.5;">Powered by real-time web search. Transform raw sales variances into boardroom-ready narratives using global trends.</p>
        </div>
        ''', unsafe_allow_html=True)
