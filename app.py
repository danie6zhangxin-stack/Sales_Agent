import streamlit as st
import pandas as pd
from pandasai import Agent
from pandasai.llm import GoogleGemini

# --- 1. 页面设置 ---
st.set_page_config(page_title="Club Med Sales AI Agent", layout="wide", page_icon="🤖")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTextInput { border-radius: 20px; }
    </style>
    """, unsafe_allow_stdio=True)

# 优先读取 Streamlit Cloud 的 Secrets，如果本地测试没配则用默认值
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = "你的API_KEY" 

llm = GoogleGemini(api_key=api_key)

# --- 2. 数据侧边栏 ---
with st.sidebar:
    st.header("📂 数据中心")
    uploaded_file = st.file_uploader("上传 SalesData.csv", type=['csv'])
    st.info("🔒 内存即时处理，关闭后数据销毁，绝对安全。")

# --- 3. 核心主程序 ---
st.title("🤖 Club Med 销售洞察智能体")

if uploaded_file:
    with st.spinner("正在加载并进行财务级数据清洗..."):
        # 1. 读取原始数据
        df = pd.read_csv(uploaded_file)
        
        # 清除表头前后可能隐藏的空格
        df.columns = [col.strip() for col in df.columns]

        # 2. 字段翻译映射 (根据你的截图业务逻辑)
        col_mapping = {
            'CONSUMPTION_CALENDAR[Month Name]': 'Consumption Month',
            'CONSUMPTION_CALENDAR[Consumption_month_num]': 'Consumption Month Num',
            'REF_SALES_MARKET[Business_unit]': 'Business Unit',
            'REF_SALES_MARKET[Market]': 'Market',
            'VENTES[Resort]': 'Resort',
            'SALES_CALENDAR[Sales_date]': 'Sales Date',
            'CONSUMPTION_CALENDAR[Consumption_year]': 'Consumption Year',
            'REF_CML_AGENCY[Group_TA_cml]': 'TA Group',
            'REF_CML_AGENCY[Region_cml]': 'TA Region',
            'REF_CML_AGENCY[Channel_cml]': 'TA Channel',
            'REF_CML_AGENCY[Channel_ type_cml]': 'TA Channel Type',
            'SALES_CALENDAR[Sales_Month_name]': 'Sales Month',
            'REF_DESTINATION[Destination type Asia]': 'Destination Type',
            'REF_DESTINATION[Resort]': 'Destination Resort',
            '[BVSTS___final]': 'BV (EUR)',
            '[HN_final]': 'HN',
            '[BVSTS_loc_final]': 'BV (Local Currency)'
        }
        df.rename(columns=col_mapping, inplace=True)

        # 3. 数据类型清理 (去除千分位逗号并转为数字)
        for num_col in ['BV (EUR)', 'HN', 'BV (Local Currency)']:
            if num_col in df.columns:
                df[num_col] = df[num_col].astype(str).str.replace(',', '').str.replace(' ', '').astype(float)
        
        # 4. 空值逻辑处理 (无 TA 名字的标记为 Direct/Non-TA)
        if 'TA Group' in df.columns:
            df['TA Group'] = df['TA Group'].fillna('Direct/Non-TA')

        # 5. Club Med 专属 AI 业务指令集 (Prompt)
        custom_instructions = """
        你是 Club Med 的首席数据分析师。在回答问题时，请严格遵守以下业务逻辑：
        
        1. 【口径法则】：当用户问“X月的销售额”时，默认使用 'Consumption Month' 和 'Consumption Year'（入住时间）来计算。只有当用户明确说“下单时间”或“预订月份”时，才使用 'Sales Month'，并在回答前提醒用户：“当前查询基于下单时间 (Sales Calendar)”。
        2. 【币种法则】：'BV (EUR)' 是欧元，'BV (Local Currency)' 是原币种。默认计算销售额时，请使用 'BV (EUR)' 进行统一汇总。除非用户特别指明要看人民币(RMB/CNY)或原币，才使用 'BV (Local Currency)'。
        3. 【TA 过滤法则】：'TA Group' 列中值为 'Direct/Non-TA' 的数据代表无明确旅行社信息的直销订单。当用户查询“某个具体 TA 的业绩”或“TA 排名”时，请过滤掉 'Direct/Non-TA'。但在计算某个度假村 (Destination Resort) 或市场 (Market) 的总业绩时，必须包含所有数据（不能过滤掉它）。
        4. 【展示规则】：数值请保留两位小数并加上千分位逗号。如果发现销售额有显著差异，请主动按度假村拆解并分析原因。主动生成美观的柱状图或折线图。
        """

        # 【最新升级】：使用新版的 Agent
        agent = Agent(df, config={
            "llm": llm,
            "custom_instructions
