import streamlit as st
from zhipuai import ZhipuAI
import base64
from PIL import Image
from io import BytesIO
import json
import pandas as pd
import plotly.express as px

# --- 页面配置 ---
st.set_page_config(page_title="生态AI评估系统", layout="wide", initial_sidebar_state="expanded")

# --- CSS美化 ---
st.markdown("""
<style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745; }
    .calc-box { background-color: #f0f7ff; padding: 20px; border-radius: 8px; border: 1px solid #cce5ff; }
    .param-explain { background-color: #ffffff; padding: 10px; margin-top:5px; border-left: 3px solid #007bff; font-size: 0.9em; color: #555; }
    .visual-box { background-color: #fff8dc; padding: 15px; border-radius: 8px; border: 1px solid #faebcc; }
    .warning-card { background-color: #fff3cd; padding: 20px; border-radius: 8px; border-left: 5px solid #ffc107; }
    .chat-box { background-color: #e9ecef; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 生态系统生产总值 (GEP) 智慧决策平台")
st.caption("Powered by 智谱 GLM-4V | 核心算法：谢高地(2015)生态服务价值当量因子法")

# --- 初始化 Session State ---
if 'report_context' not in st.session_state:
    st.session_state.report_context = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- 核心：谢高地（2015）生态服务价值当量因子法---
def get_xie_gaodi_table():
    # 根据谢高地（2015）生态服务价值当量因子法录入数据
    data = {
        "二级分类": [
            "旱地 (农田)", "水田 (农田)", 
            "针叶林 (森林)", "针阔混交 (森林)", "阔叶林 (森林)", "灌木林 (森林)",
            "草原 (草地)", "灌草丛 (草地)", "草甸 (草地)",
            "湿地 (湿地)", 
            "荒漠 (荒漠)", "裸地 (荒漠)",
            "水系 (水域)", "冰川积雪 (水域)"
        ],
        "食物生产": [0.85, 1.36, 0.22, 0.31, 0.29, 0.19, 0.10, 0.38, 0.22, 0.51, 0.01, 0.00, 0.80, 0.00],
        "原料生产": [0.40, 0.09, 0.52, 0.71, 0.66, 0.43, 0.14, 0.56, 0.33, 0.50, 0.03, 0.00, 0.23, 0.00],
        "水资源供给": [0.02, -2.63, 0.27, 0.37, 0.34, 0.22, 0.08, 0.31, 0.18, 2.59, 0.02, 0.00, 8.29, 2.16],
        "气体调节": [0.67, 1.11, 1.70, 2.35, 2.17, 1.41, 0.51, 1.97, 1.14, 1.90, 0.11, 0.02, 0.77, 0.18],
        "气候调节": [0.36, 0.57, 5.07, 7.03, 6.50, 4.23, 1.34, 5.21, 3.02, 3.60, 0.10, 0.00, 2.29, 0.54],
        "净化环境": [0.10, 0.17, 1.49, 1.99, 1.93, 1.28, 0.44, 1.72, 1.00, 3.60, 0.31, 0.10, 5.55, 0.16],
        "水文调节": [0.27, 2.72, 3.34, 3.51, 4.74, 3.35, 0.98, 3.82, 2.21, 24.23, 0.21, 0.03, 102.24, 7.13],
        "土壤保持": [1.03, 0.01, 2.06, 2.86, 2.65, 1.72, 0.62, 2.40, 1.39, 2.31, 0.13, 0.02, 0.93, 0.00],
        "维持养分循环": [0.12, 0.19, 0.16, 0.22, 0.20, 0.13, 0.05, 0.18, 0.11, 0.18, 0.01, 0.00, 0.07, 0.00],
        "生物多样性": [0.13, 0.21, 1.88, 2.60, 2.41, 1.57, 0.56, 2.18, 1.27, 7.87, 0.12, 0.02, 2.55, 0.01],
        "美学景观": [0.06, 0.09, 0.82, 1.14, 1.06, 0.69, 0.25, 0.96, 0.56, 4.73, 0.05, 0.01, 1.89, 0.09]
    }
    return pd.DataFrame(data).set_index("二级分类")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 专家控制台")
    api_key = st.text_input("请输入智谱 API Key", type="password")
    st.markdown("---")
    st.info("🔎 **功能说明：**\n\n1. **数据透明化**：AI 必须解释每个系数(E, K)的选取理由。\n2. **精准量化**：健康度和固碳量将给出具体数字。\n3. **生态顾问**：与AI顾问进行对话，例如我想种果树是否合适？")
    if st.button("🗑️ 清空对话历史"):
        st.session_state.chat_history = []
        st.rerun()

# --- 辅助函数 ---
def safe_format_money(value):
    try:
        clean_val = str(value).replace(',', '').replace('¥', '').replace('元', '')
        num_val = float(clean_val)
        return f"¥{num_val:,.0f}"
    except:
        return str(value)

# --- 主功能区 ---
uploaded_file = st.file_uploader("📸 上传环境样本...", type=["jpg", "png", "jpeg"])

if uploaded_file and api_key:
    image = Image.open(uploaded_file)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(image, caption='待评估地块', use_container_width=True)
    with c2:
        st.write("#### 🕵️‍♂️ AI 正在进行深度溯源分析...")
        
        if st.button("🚀 生成深度溯源评估报告", type="primary"):
            st.session_state.chat_history = []
            
            with st.spinner("正在逐项核对参数来源... 计算固碳量... 推演生态后果..."):
                try:
                    # 转码
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    client = ZhipuAI(api_key=api_key)
                    
                    # --- 核心提示词升级：要求解释 S, E, K 来源，强制数字，详细推导 ---
                    prompt = """
                    你是一位严谨的生态经济学教授。请对这张照片进行 GEP 核算。
                    
                    【任务一：透明化计算过程 (核心要求)】
                    1. 识别生态类型。
                    2. **必须引用谢高地(2015)标准**。
                    3. 展示公式：V = S × E × K。
                       - **S (面积)**：假设图示区域为1公顷。
                       - **E (当量因子)**：必须明确指出参考了谢高地表中的哪一列（如：因为是森林，选取了“森林”列的气体调节系数3.50）。
                       - **K (修正系数)**：详细解释修正理由。例如：“K定为1.2，因为图中植被郁闭度极高，且位于亚热带，生物量高于平均水平”。
                    4. 列出具体的乘法算式。
                    
                    【任务二：精准量化 (拒绝模糊)】
                    1. 健康度：根据植被覆盖和人为干扰，给出一个 0-100 的整数。
                    2. 固碳量：给出一个具体的估算数字（单位：吨/年），例如：15.5。
                    
                    【任务三：深度后果推导 (利用机会成本法)】
                    如果开发成商业用地，请推导连锁反应。
                    格式要求：详细描述“起因 -> 过程 -> 结果”。例如：“地表硬化 -> 雨水下渗减少 -> 地表径流激增 -> 城市内涝风险增加”。

                    【任务四：详细专家建议】
                    给出详细的专家建议，每个建议包含是什么，为什么，怎么样。每个建议不少于500字。
                    
                    请严格输出 JSON：
                    {
                        "meta": {
                            "ecosystem_name": "名称",
                            "health_score": 85, 
                            "carbon": 120.5
                        },
                        "visual_evidence": ["线索1", "线索2"],
                        "calculation_process": {
                            "formula": "V = S × E × K",
                            "param_explanation": {
                                "S_explain": "面积取值理由...",
                                "E_explain": "E值选取理由（对应谢高地表哪一项）...",
                                "K_explain": "K值修正理由（基于图中什么特征）..."
                            },
                            "math_detail": "具体算式"
                        },
                        "total_value": 50000,
                        "breakdown": {
                            "供给服务": 5000,
                            "调节服务": 25000,
                            "文化服务": 10000,
                            "支持服务": 10000
                        },
                        "simulation": {
                            "scenario": "商业开发后果",
                            "loss_value": 50000,
                            "chain_of_consequence": "详细的推导链条（起因->过程->结果）"
                        },
                        "suggestions": [{"title": "建议1", "content": "内容"}, {"title": "建议2", "content": "内容"}]
                    }
                    """

                    response = client.chat.completions.create(
                        model="glm-4v-plus",
                        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_str}}, {"type": "text", "text": prompt}]}]
                    )
                    
                    result_text = response.choices[0].message.content
                    clean_json = result_text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean_json)
                    
                    st.session_state.report_data = data
                    st.session_state.report_context = json.dumps(data, ensure_ascii=False)

                except Exception as e:
                    st.error("AI 思考超时，请重试！")
                    st.write(e)

    # --- 展示报告 ---
    if 'report_data' in st.session_state:
        data = st.session_state.report_data
        
        st.divider()
        # 1. 精准仪表盘 (显示具体数值)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌿 类型", data['meta'].get('ecosystem_name', '未知'))
        m2.metric("❤️ 健康度", f"{data['meta'].get('health_score', 0)} 分") # 显示具体分
        m3.metric("🌳 年固碳量", f"{data['meta'].get('carbon', 0)} 吨/年") # 显示具体吨数
        m4.metric("💰 年度总值", safe_format_money(data.get('total_value', 0)), delta="GEP")

        st.markdown("---")

        # 2. 分栏详情
        logic_col, chart_col = st.columns([1.4, 1])
        with logic_col:
            st.subheader("🧮 价值量化与参数溯源")
            tab1, tab2 = st.tabs(["📝 参数选取理由", "📖 谢高地标准表"])
            
            with tab1:
                # 视觉证据
                st.markdown('<div class="visual-box"><b>👁️ 视觉依据：</b></div>', unsafe_allow_html=True)
                for ev in data.get('visual_evidence', []):
                    st.write(f"- {ev}")
                
                st.write("")
                
                # 计算逻辑 (深度揭秘 S, E, K)
                st.markdown('<div class="calc-box"><b>📐 算法参数详解：</b></div>', unsafe_allow_html=True)
                calc = data.get('calculation_process', {})
                params = calc.get('param_explanation', {})
                
                st.latex(r"V = S \times E \times K")
                
                # 专门的参数解释框
                st.markdown(f"""
                <div class='param-explain'>
                <b>1. 面积 (S)：</b> {params.get('S_explain', '默认标准单位1公顷')}
                </div>
                <div class='param-explain'>
                <b>2. 当量因子 (E) 来源：</b> {params.get('E_explain', '参考谢高地标准')}
                </div>
                <div class='param-explain'>
                <b>3. 修正系数 (K) 推导：</b> {params.get('K_explain', '根据长势修正')}
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.markdown("**最终运算明细：**")
                st.code(calc.get('math_detail', ''), language='python')
                
            with tab2:
                st.markdown("**📚 生态服务价值当量因子表 (谢高地 2015)**")
                st.dataframe(get_xie_gaodi_table(), use_container_width=True, height=400)

        with chart_col:
            st.subheader("📊 结构分析与后果推演")
            breakdown = data.get('breakdown', {})
            df = pd.DataFrame(dict(r=list(breakdown.values()), theta=list(breakdown.keys())))
            fig = px.line_polar(df, r='r', theta='theta', line_close=True, title="生态服务功能雷达图")
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)
            
            # 破坏模拟 (详细推导版)
            sim = data.get('simulation', {})
            st.markdown(f"""
            <div class="warning-card">
                <h4>🚧 机会成本分析：{sim.get('scenario', '开发模拟')}</h4>
                <p>若破坏此区域，预计年生态损失：<br>
                <span style="font-size:22px;color:#d9534f"><b>-{safe_format_money(sim.get('loss_value', 0))}</b></span></p>
                <hr style="border-top: 1px dashed #bbb;">
                <p><b>🔗 后果推演链条：</b></p>
                <p style="font-style: italic;">{sim.get('chain_of_consequence', '推导中...')}</p>
            </div>
            """, unsafe_allow_html=True)

        # 3. 专家建议
        st.divider()
        st.subheader("🛡️ 专家实施方案建议")
        for i, s in enumerate(data.get('suggestions', []), 1):
            with st.expander(f"📝 方案 {i}: {s.get('title', '保护建议')}", expanded=True):
                st.markdown(s.get('content', ''))

        # 4. AI 问答区
        st.divider()
        st.subheader("💬 AI 生态顾问在线")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("针对刚才的 S/E/K 参数或推导过程提问..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                msg_ph = st.empty()
                full_res = ""
                chat_client = ZhipuAI(api_key=api_key)
                # 上下文包含解释理由
                ctx = f"用户提问：{prompt}\n基于此报告：{st.session_state.report_context}"
                try:
                    res = chat_client.chat.completions.create(
                        model="glm-4", messages=[{"role": "user", "content": ctx}], stream=True
                    )
                    for chunk in res:
                        if chunk.choices[0].delta.content:
                            full_res += chunk.choices[0].delta.content
                            msg_ph.markdown(full_res + "▌")
                    msg_ph.markdown(full_res)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_res})
                except:
                    st.error("顾问掉线了")

elif not api_key:
    st.warning("👈 请先在左侧填入智谱 API Key")