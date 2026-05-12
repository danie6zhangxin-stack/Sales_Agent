import streamlit as st
import pandas as pd
import numpy as np 
from pandasai import Agent
from langchain_openai import ChatOpenAI
import matplotlib
matplotlib.use('Agg') # 强制离线渲染
import os

# --- 1. 高端商业视觉配置 (McKinsey x ClubMed Theme) ---
st.set_page_config(page_title="ClubMed Executive Intelligence", layout="wide", page_icon="Ψ")

CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    :root { 
        --cm-blue: #1D263B;      
        --cm-terracotta: #A64B35; 
        --cm-beige: #F8F9FA;     
    }
    
    .main { background-color: var(--cm-beige); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--cm-blue); }
    
    div[data-testid="stMetric"] { 
        background-color: white; border-radius: 6px; padding: 15px 20px; 
        box-shadow: 0 2px 10px rgba(0,0,0,0.02); border-top: 3px solid var(--cm-terracotta); 
    }
    div[data-testid="stMetricValue"] { color: var(--cm-blue); font-weight: 600; font-size: 28px; }
    
    .stDataFrame { border: 1px solid #EAECEF; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.01); background-color: white; }
    
    div[data-testid="stChatMessage"] { 
        background-color: transparent !important; 
        border: none !important; 
        border-bottom: 1px solid #EAECEF !important; 
        padding: 1.5rem 0.5rem !important; 
        margin-bottom: 0 !important; 
    }
    div[data-testid="stChatMessage"]:last-child { border-bottom: none !important; }
    
    div[data-testid="stChatMessageAvatarUser"], div[data-testid="stChatMessageAvatarAssistant"] { 
        background-color: var(--cm-blue) !important; color: white !important;
    }
    
    .stSidebar { background-color: white !important; border-right: 1px solid #EAECEF; }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- 2. AI 引擎初始化 ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except:
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # ⚠️ 填入你的真实Key

llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.1)

# --- 3. 核心数据清洗引擎 ---
@st.cache_data
def load_and_clean(file):
    data = pd.read_csv(file, low_memory=False)
    data.columns = [col.strip() for col in data.columns]
    mapping = {
        'CONSUMPTION_CALENDAR[Month Name]': 'Month',
        'CONSUMPTION_CALENDAR[Consumption_month_num]': 'Month_Num',
        'CONSUMPTION_CALENDAR[Consumption_year]': 'Year',
        'REF_SALES_MARKET[Market]': 'Market',
        'REF_DESTINATION[Resort]': 'Resort',
        'REF_CML_AGENCY[Group_TA_cml]': 'TA_Group',
        'REF_DESTINATION[Destination type Asia]': 'Dest_Type',
        '[BVSTS___final]': 'BV',
        '[HN_final]': 'HN'
    }
    data.rename(columns=mapping, inplace=True, errors='ignore')
    
    for col in ['Market', 'Resort', 'TA_Group', 'Dest_Type', 'Month']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()
    
    for col in ['BV', 'HN']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    data['ADR'] = (data['BV'] / data['HN']).replace([float('inf'), -float('inf')], 0).fillna(0)
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    
    if 'Month_Num' in data.columns:
        data['Month_Num'] = pd.to_numeric(data['Month_Num'], errors='coerce').fillna(0).astype(int)
        
    return data

# ==========================================
# 🌟 核心提取器 (防崩溃必备)
# ==========================================
def extract_dataframe(resp):
    if isinstance(resp, pd.DataFrame): return resp
    if hasattr(resp, 'to_pandas'): return resp.to_pandas()
    if hasattr(resp, 'dataframe'): return resp.dataframe
    if hasattr(resp, '_df'): return resp._df
    try: return pd.DataFrame(resp)
    except: return None

# --- 4. 业务逻辑与界面展示 ---
with st.sidebar:
    st.markdown("<h2 style='color:#A64B35;'>ClubMed Ψ Hub</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload SalesData.csv", type=['csv'])
    st.divider()

if uploaded_file:
    df = load_and_clean(uploaded_file)
    
    with st.sidebar:
        st.markdown("### ⚙️ Global Controls")
        years = sorted([y for y in df['Year'].unique() if y > 2000], reverse=True)
        sel_year = st.selectbox("Current Year Filter", years) if years else 2026
        
        markets = sorted([str(m) for m in df['Market'].unique() if str(m).strip() != '' and str(m).lower() != 'nan'])
        def_markets = [m for m in markets if any(k in m.lower() for k in ['china', 'hong kong', 'hk', 'cn'])]
        sel_markets = st.multiselect("Active Markets (Dashboard)", markets, default=def_markets)

    st.markdown(f"### 📈 Executive Performance ({sel_year} vs {sel_year-1})")
    df_cy = df[df['Year'] == sel_year]
    df_py = df[df['Year'] == sel_year - 1]
    
    c1, c2, c3 = st.columns(3)
    cy_bv = df_cy['BV'].sum()
    py_bv = df_py['BV'].sum()
    c1.metric("Total BV", f"€ {cy_bv:,.0f}", f"{(cy_bv-py_bv)/py_bv*100:.1f}%" if py_bv>0 else None)
    c2.metric("Total HN", f"{df_cy['HN'].sum():,.0f}")
    c3.metric("Avg ADR", f"€ {df_cy['BV'].sum()/df_cy['HN'].sum() if df_cy['HN'].sum()>0 else 0:,.2f}")

    if sel_markets:
        st.markdown(f"#### 📊 Regional Drill-down by Destination Type")
        df_cy_filtered = df_cy[df_cy['Market'].isin(sel_markets)]
        df_py_filtered = df_py[df_py['Market'].isin(sel_markets)]
        
        cy_target = df_cy_filtered.groupby(['Market', 'Dest_Type'])[['BV', 'HN']].sum().reset_index()
        py_target = df_py_filtered.groupby(['Market', 'Dest_Type'])[['BV', 'HN']].sum().reset_index()
        
        dash_df = pd.merge(cy_target, py_target, on=['Market', 'Dest_Type'], how='outer', suffixes=(f'_{sel_year}', f'_{sel_year-1}')).fillna(0)
        if not dash_df.empty:
            dash_df['YoY(%)'] = np.where(dash_df[f'BV_{sel_year-1}'] > 0, 
                                        (dash_df[f'BV_{sel_year}'] - dash_df[f'BV_{sel_year-1}']) / dash_df[f'BV_{sel_year-1}'] * 100, 0)
            
            display_cols = ['Market', 'Dest_Type', f'BV_{sel_year}', f'BV_{sel_year-1}', 'YoY(%)']
            st.dataframe(dash_df[display_cols].style.format({
                f'BV_{sel_year}': '€ {:,.0f}', f'BV_{sel_year-1}': '€ {:,.0f}', 'YoY(%)': '{:+.1f}%'
            }).background_gradient(subset=['YoY(%)'], cmap='RdYlGn', vmin=-15, vmax=15), use_container_width=True, hide_index=True)

    # ==========================================
    # 🌟 模块 C：智能 AI 决策顾问
    # ==========================================
    st.divider()
    st.markdown("### 🤖 Strategy Advisor (Deep Dive Table)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    history_context = ""
    if len(st.session_state.messages) > 0:
        history_context = "\n\n=== RECENT CONVERSATION ===\n"
        for msg in st.session_state.messages[-4:]:
            if isinstance(msg["content"], str) and not msg["content"].endswith(".png"):
                history_context += f"{msg['role'].upper()}: {msg['content'][:200]}...\n"

    # 🌟 终极强压指令：双向大写转换，彻底消灭 0 行数据 Bug
    custom_instr = f"""
    You are a Data Analyst for ClubMed writing Python code for PandasAI.

    === CRITICAL EXECUTION RULES ===
    1. YOU MUST OUTPUT VALID PYTHON CODE ENCLOSED IN ```python AND ```.
    2. The dataframe is provided to you as `dfs[0]`. Assign the final result to `result`.
    3. NEVER output conversational text. ONLY output the code block.

    === BULLETPROOF FILTERING RULES ===
    1. YEAR: Create a new filtered df first: `df_filtered = dfs[0][dfs[0]['Year'] == 2026]`.
    2. TARGET MATCH (CRITICAL): You MUST convert BOTH the column AND your search string to uppercase to avoid empty dataframes! 
       Correct Example: `df_filtered = df_filtered[df_filtered['TA_Group'].str.upper() == 'NJ XXY'.upper()]`
    3. MONTHS: Use `Month_Num` (1-12). For "Jan to May", use `df_filtered = df_filtered[df_filtered['Month_Num'].between(1, 5)]`. NEVER use string matching for months.

    === OUTPUT FORMAT ===
    1. Group `df_filtered` by `Dest_Type` (and `Month` if requested). Calculate sum of `BV` and `HN`. Then calculate `ADR` = `BV` / `HN`.
    2. MUST call `.reset_index()` at the end so `result` is a flat dataframe. 
    3. Rename columns cleanly (e.g., 'Month', 'Destination Type', 'Business Volume').
    4. Round to 2 decimals. NO PLOTTING allowed.
    
    === MEMORY ===
    {history_context}
    """

    agent = Agent(df, config={"llm": llm, "custom_instructions": custom_instr, "save_charts": False, "enable_cache": False})

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if isinstance(m["content"], pd.DataFrame): 
                st.dataframe(m["content"], use_container_width=True, hide_index=True)
            elif isinstance(m["content"], str) and m["content"].endswith(".png"):
                st.image(m["content"])
            else: 
                st.markdown(m["content"])

    if prompt := st.chat_input("Analyze NJ XXY's performance from Jan to May 2026."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Generating code and executing analysis..."):
                try:
                    response = agent.chat(prompt)
                    safe_df = extract_dataframe(response)
                    
                    if safe_df is not None:
                        # 🌟 智能防御机制：如果是空表，直接标红提醒！
                        if safe_df.empty:
                            st.warning("⚠️ 查无数据 (Empty Table): The analysis executed successfully, but 0 records matched your filters. (Check if the Year, Month, or TA Group exists).")
                        else:
                            st.markdown("**📊 Analysis Results:**")
                            st.dataframe(safe_df, use_container_width=True, hide_index=True)
                        st.session_state.messages.append({"role": "assistant", "content": safe_df})
                    else:
                        res_str = str(response)
                        st.markdown(res_str)
                        st.session_state.messages.append({"role": "assistant", "content": res_str})
                except Exception as e:
                    st.error(f"Analysis Error: {e}")
else:
    st.markdown("""
        <div style="height: 60vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; color: #1D263B;">Ψ Executive Strategy Hub</h1>
            <p style="color: #6c757d; max-width: 550px; font-size: 1.1rem; line-height: 1.6;">
                Upload your dataset to begin. Accurate YoY Dashboards and an Intelligent Data Analyst are ready for your inquiries.
            </p>
        </div>
    """, unsafe_allow_html=True)
