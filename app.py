import io
import json
import re
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# =========================================================
# 1) Page Configuration (Premium UI: Linear/Notion + Lux, LIGHT text)
# =========================================================
st.set_page_config(page_title="Report Mate", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg: #F7F8FA;
  --bg2:#F3F4F7;
  --panel: rgba(255,255,255,0.86);
  --panel2: rgba(255,255,255,0.96);
  --border: rgba(15,23,42,0.10);
  --border2: rgba(15,23,42,0.14);
  --text: rgba(17,24,39,0.92);   /* 거의 검정 */
  --muted: rgba(17,24,39,0.62);
  --muted2: rgba(17,24,39,0.52);
  --accent: #6D5EF7;
  --accent2: #00B7FF;
  --shadow: 0 18px 60px rgba(2,6,23,0.10);
  --shadow2: 0 12px 40px rgba(2,6,23,0.08);
}

html, body, [class*="css"]{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background:
    radial-gradient(1200px 800px at 18% 10%, rgba(109,94,247,0.14), transparent 55%),
    radial-gradient(900px 600px at 85% 15%, rgba(0,183,255,0.10), transparent 50%),
    radial-gradient(700px 700px at 55% 88%, rgba(0,0,0,0.04), transparent 45%),
    linear-gradient(180deg, var(--bg), var(--bg2)) !important;
  color: var(--text) !important;
}

/* Make container feel premium + centered */
.block-container{
  padding-top: 2.2rem;
  padding-bottom: 3rem;
  max-width: 980px;
}

/* Sidebar styling */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.72);
  border-right: 1px solid rgba(15,23,42,0.10);
}
section[data-testid="stSidebar"] *{
  color: var(--text) !important;
}

/* Hero */
.hero{
  padding: 28px 26px;
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(109,94,247,0.16), rgba(0,183,255,0.08));
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  margin: 8px 0 18px 0;
  position: relative;
  overflow: hidden;
}
.hero:before{
  content:"";
  position:absolute;
  inset:-2px;
  background: radial-gradient(900px 320px at 15% 18%, rgba(255,255,255,0.65), transparent 60%);
  pointer-events:none;
}
.badge{
  display:inline-flex;
  gap:8px;
  align-items:center;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.65);
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
}
.hero-title{
  margin-top: 10px;
  font-size: 34px;
  font-weight: 900;
  letter-spacing: -0.02em;
  line-height: 1.12;
}
.hero-sub{
  margin-top: 8px;
  font-size: 15px;
  color: var(--muted);
  line-height: 1.65;
  max-width: 78ch;
}
.kpi{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top: 12px;
}
.pill{
  background: rgba(255,255,255,0.70);
  border: 1px solid var(--border);
  padding: 7px 10px;
  border-radius: 999px;
  font-size: 12px;
  color: var(--muted);
  font-weight: 700;
}

/* Cards (Glass) */
.glass{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 22px;
  box-shadow: var(--shadow2);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  margin: 14px 0 18px 0;
}
.card-title{
  font-size: 13px;
  color: var(--muted);
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.h3{
  font-size: 18px;
  font-weight: 900;
  letter-spacing: -0.01em;
  margin: 0 0 8px 0;
}
.help{
  font-size: 13px;
  color: var(--muted);
  line-height: 1.6;
  margin-top: 6px;
}

/* Inputs */
.stTextInput>div>div>input,
.stTextArea textarea,
.stFileUploader section,
.stSelectbox>div>div{
  border-radius: 14px !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  background: rgba(255,255,255,0.80) !important;
  color: var(--text) !important;
}
.stTextArea textarea::placeholder,
.stTextInput input::placeholder{
  color: rgba(17,24,39,0.40) !important;
}

/* Buttons */
.stButton>button{
  width:100%;
  border-radius: 14px;
  border: 1px solid rgba(15,23,42,0.10);
  background: linear-gradient(135deg, rgba(109,94,247,1), rgba(0,183,255,0.90));
  color: white !important;
  font-weight: 900;
  padding: 12px 14px;
  box-shadow: 0 12px 30px rgba(109,94,247,0.22);
  transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
}
.stButton>button:hover{
  transform: translateY(-1px);
  box-shadow: 0 16px 44px rgba(109,94,247,0.28);
  filter: brightness(1.02);
}

/* Secondary button */
.secondary-btn .stButton>button{
  background: rgba(255,255,255,0.78) !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  box-shadow: none !important;
  color: rgba(17,24,39,0.86) !important;
}
.secondary-btn .stButton>button:hover{
  transform: translateY(-1px);
  box-shadow: 0 14px 36px rgba(2,6,23,0.10) !important;
  filter: none !important;
}

/* Tabs */
[data-baseweb="tab-list"]{
  background: rgba(255,255,255,0.70);
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 14px;
  padding: 6px;
}
[data-baseweb="tab"]{
  border-radius: 12px;
  color: var(--muted) !important;
  font-weight: 900;
}
[aria-selected="true"]{
  background: rgba(255,255,255,0.95) !important;
  color: var(--text) !important;
}

/* Popover button */
div[data-testid="stPopover"] > button{
  background: rgba(255,255,255,0.78) !important;
  color: rgba(17,24,39,0.88) !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  border-radius: 10px !important;
  padding: 2px 8px !important;
  font-size: 12px !important;
  min-height: 26px !important;
}

hr{
  border-color: rgba(15,23,42,0.10) !important;
}
.small{
  font-size: 12px;
  color: var(--muted2);
}
.footer{
  text-align:center;
  color: rgba(17,24,39,0.45);
  font-size: 12px;
  margin-top: 34px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 2) Session State
# =========================================================
def init_state():
    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "context_text" not in st.session_state:
        st.session_state["context_text"] = ""
    if "last_inputs" not in st.session_state:
        st.session_state["last_inputs"] = {}
    if "expansion_level" not in st.session_state:
        st.session_state["expansion_level"] = 0

init_state()

# =========================================================
# 3) Sidebar
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model_name = st.text_input("Model", value="gpt-4o-mini")

    st.divider()
    st.markdown("### 📝 Draft Options")
    base_paras = st.select_slider("소절당 문단 수(기본)", options=[2, 3], value=2)
    min_chars_per_para = st.select_slider("문단 최소 글자 수", options=[200, 250, 300, 400], value=200)
    tone_setting = st.selectbox("어조", ["Academic", "Formal", "Analytical"], index=0)
    expand_additional = st.select_slider("확장 시 소절당 추가 문단", options=[1, 2], value=1)

    st.divider()
    if st.button("새 프로젝트 시작", use_container_width=True):
        st.session_state.clear()
        init_state()
        st.rerun()

# =========================================================
# 4) Hero Header (Title changed to "Report Mate")
# =========================================================
st.markdown(
    """
<div class="hero">
  <div class="badge">✨ Academic Drafting Assistant</div>
  <div class="hero-title">Report Mate</div>
  <div class="hero-sub">
    선행연구 PDF를 기반으로 <b>개요(간결)</b>와 <b>초안(소절·문단 단위)</b>을 생성합니다.
    본문에는 출처를 연결하는 <b>[REF:파일명,p숫자]</b> 태그가 포함되며, 필요 시 <b>확장 버튼</b>으로 분량을 늘릴 수 있습니다.
  </div>
  <div class="kpi">
    <div class="pill">🧩 소절(1.1…)</div>
    <div class="pill">📍 REF 팝오버</div>
    <div class="pill">➕ 초안 확장</div>
    <div class="pill">📋 간결 개요</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 5) Main Inputs
# =========================================================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Research Context</div>', unsafe_allow_html=True)

topic = st.text_input("연구 주제", placeholder="예: 생성형 AI가 대학생의 학술적 글쓰기에 미치는 영향")
col1, col2 = st.columns(2)
with col1:
    purpose = st.text_input("연구 목적", placeholder="연구를 통해 무엇을 밝히고 싶나요?")
with col2:
    hypothesis = st.text_input("연구 가설", placeholder="예상되는 결론은 무엇인가요?")

uploaded_files = st.file_uploader(
    "선행연구 PDF 업로드 (다중 선택 가능)",
    type=["pdf"],
    accept_multiple_files=True
)

st.markdown(
    '<div class="help">Tip: 텍스트 추출이 안 되는 스캔 PDF는 내용이 비어 보일 수 있어요. 가능한 텍스트 기반 PDF를 업로드해 주세요.</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 6) Core Logic (No RAG / No export)
# =========================================================
def get_combined_text_with_meta(files, max_pages_each=10, max_chars=35000):
    text_data = ""
    for f in files:
        reader = PdfReader(io.BytesIO(f.getvalue()))
        file_name = f.name
        for i, page in enumerate(reader.pages[:max_pages_each]):
            content = page.extract_text()
            if content:
                text_data += f"\n[SOURCE: {file_name}, PAGE: {i+1}]\n{content}\n"
    return text_data[:max_chars]

def tone_instructions(tone: str) -> str:
    if tone == "Academic":
        return "학술적·객관적 문체로, 정의-근거-논증 연결을 분명히 하되 과도한 수사는 피할 것."
    if tone == "Formal":
        return "격식을 갖춘 문체로, 문장 구조를 정돈하고 단정적 표현은 근거와 함께 제시할 것."
    if tone == "Analytical":
        return "분석적 문체로, 비교·대조·비판적 논의(한계/공백)를 더 적극적으로 포함할 것."
    return "학술적 문체를 유지할 것."

def build_initial_prompt(topic, purpose, hypothesis, context, base_paras, min_chars_per_para, tone):
    system_msg = f"""
당신은 석사학위 논문을 다수 지도한 전문 학술 에디터입니다.
제공된 자료에 근거해 엄밀한 학술 문체(석사 논문 수준)로 서술하며, 주장-근거-비판적 논의-연구 공백/기여를 명료하게 연결합니다.
{tone_instructions(tone)}
반드시 지정한 JSON 스키마로만 출력하세요.
""".strip()

    user_msg = f"""
주제: {topic}
목적: {purpose}
가설: {hypothesis}

[자료 원문]
{context}

[요구사항]
1) detailed_outline (간결):
- 각 섹션(서론/이론적 배경/연구방법/결론)당 6~10문장 이내로 전개 전략만 요약.

2) interactive_draft (석사 수준, 기본 분량 강화):
- 각 섹션을 소절로 나누어 작성 (예시 구조를 반드시 반영):
  • 서론: 1.1 연구 배경, 1.2 문제 제기, 1.3 연구 목적/질문, 1.4 연구 기여/구성
  • 이론적 배경: 2.1 핵심 개념 정의, 2.2 선행연구 흐름, 2.3 한계/논쟁점, 2.4 연구 공백 및 연구모형 시사점
  • 연구방법: 3.1 연구설계, 3.2 표본/자료, 3.3 측정(변수/도구), 3.4 분석전략, 3.5 타당도·윤리
  • 결론: 4.1 결과 요약(예상 포함), 4.2 이론적 함의, 4.3 실천적 함의, 4.4 한계 및 후속연구
- 각 소절은 최소 {base_paras}개 문단으로 작성.
- 각 문단은 최소 {min_chars_per_para}자 이상(한국어 기준).
- 각 문단에 최소 1개의 인용 태그 [REF:파일명,p숫자]를 반드시 포함(가능하면 2개).
- 논리 전개: (주장/요지 → 근거와 선행연구 연결 → 비판적 논의/한계 → 연구 공백 및 본 연구 위치화)를 균형 있게 포함.

3) source_map:
- 각 [REF:...] 태그에 대응하는 근거(해당 페이지의 핵심 요약)를 구체적으로 작성.

4) REF 규칙:
- 태그 포맷은 반드시 정확히 [REF:파일명,p숫자]
- 파일명은 [SOURCE: ...]에 나온 파일명을 그대로 사용
- 페이지 숫자는 [PAGE: ...]를 근거로 사용

[반드시 아래 JSON으로만 출력]
{{
  "detailed_outline": {{
    "서론": "...",
    "이론적 배경": "...",
    "연구방법": "...",
    "결론": "..."
  }},
  "interactive_draft": {{
    "서론": "...",
    "이론적 배경": "...",
    "연구방법": "...",
    "결론": "..."
  }},
  "source_map": {{
    "[REF:파일명,p숫자]": "이 REF가 지지하는 핵심 근거(해당 페이지 내용) 요약"
  }}
}}
""".strip()
    return system_msg, user_msg

def build_expand_prompt(topic, purpose, hypothesis, context, current_result, add_paras, min_chars_per_para, tone):
    system_msg = f"""
당신은 석사학위 논문을 다수 지도한 전문 학술 에디터입니다.
기존 초안을 더 전문적이고 더 길게 확장합니다. 근거(REF) 밀도와 논리 연결을 강화하세요.
{tone_instructions(tone)}
반드시 지정한 JSON 스키마로만 출력하세요.
""".strip()

    user_msg = f"""
주제: {topic}
목적: {purpose}
가설: {hypothesis}

[자료 원문]
{context}

[기존 결과(JSON)]
{json.dumps(current_result, ensure_ascii=False)}

[확장 요구사항]
- interactive_draft만 확장 (detailed_outline는 유지 또는 약간 정리).
- 각 소절(예: 1.1, 1.2...)마다 문단을 추가로 {add_paras}개씩 더 작성.
- 새로 추가되는 각 문단은 최소 {min_chars_per_para}자 이상.
- 새 문단마다 최소 1개의 [REF:파일명,p숫자] 포함(가능하면 2개).
- source_map은 기존 매핑을 유지하고, 새로 추가된 REF가 있으면 반드시 추가/보강.

[반드시 아래 JSON으로만 출력]
{{
  "detailed_outline": {{
    "서론": "...",
    "이론적 배경": "...",
    "연구방법": "...",
    "결론": "..."
  }},
  "interactive_draft": {{
    "서론": "...",
    "이론적 배경": "...",
    "연구방법": "...",
    "결론": "..."
  }},
  "source_map": {{
    "[REF:파일명,p숫자]": "이 REF가 지지하는 핵심 근거(해당 페이지 내용) 요약"
  }}
}}
""".strip()
    return system_msg, user_msg

def call_openai_json(api_key, model, system_msg, user_msg, temperature=0.45):
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return json.loads(resp.choices[0].message.content)

def render_text_with_ref_popovers(text, source_map):
    parts = re.split(r"(\[REF:[^\]]+\])", text)
    buffer = ""
    for part in parts:
        if part.startswith("[REF:"):
            if buffer.strip():
                st.markdown(buffer)
            buffer = ""
            ref_info = source_map.get(part, "상세 출처 정보를 불러올 수 없습니다.")
            with st.popover(f"📍 {part}"):
                st.markdown(f"**상세 근거:**\n\n{ref_info}")
        else:
            buffer += part
    if buffer.strip():
        st.markdown(buffer)

# =========================================================
# 7) Actions
# =========================================================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Actions</div>', unsafe_allow_html=True)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    generate_clicked = st.button("🚀 분석 및 상세 초안 생성", type="primary")
with btn_col2:
    st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
    expand_clicked = st.button("➕ 초안 확장(추가 작성)", disabled=(st.session_state["result"] is None))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="small">• 확장은 기존 초안을 바탕으로 소절마다 문단을 추가합니다.</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if generate_clicked:
    if not user_api_key:
        st.error("API 키를 입력해주세요.")
    elif not uploaded_files:
        st.warning("분석할 논문 파일을 업로드해주세요.")
    elif not topic:
        st.warning("연구 주제를 입력해주세요.")
    else:
        with st.spinner("선행연구들을 교차 분석하며 석사 수준의 초안을 작성 중입니다..."):
            try:
                context = get_combined_text_with_meta(uploaded_files, max_pages_each=10)
                st.session_state["context_text"] = context
                st.session_state["expansion_level"] = 0
                st.session_state["last_inputs"] = {
                    "topic": topic,
                    "purpose": purpose,
                    "hypothesis": hypothesis,
                    "base_paras": base_paras,
                    "min_chars_per_para": min_chars_per_para,
                    "tone_setting": tone_setting,
                    "model_name": model_name,
                }

                system_msg, user_msg = build_initial_prompt(
                    topic, purpose, hypothesis, context, base_paras, min_chars_per_para, tone_setting
                )
                st.session_state["result"] = call_openai_json(
                    api_key=user_api_key,
                    model=model_name,
                    system_msg=system_msg,
                    user_msg=user_msg,
                    temperature=0.45,
                )
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")

if expand_clicked:
    if not user_api_key:
        st.warning("먼저 OpenAI API Key를 입력해주세요.")
    elif st.session_state["result"] is None:
        st.warning("먼저 초안을 생성해주세요.")
    else:
        with st.spinner("초안을 더 전문적으로 확장 작성 중입니다..."):
            try:
                last = st.session_state.get("last_inputs", {})
                topic0 = last.get("topic", topic)
                purpose0 = last.get("purpose", purpose)
                hypothesis0 = last.get("hypothesis", hypothesis)
                model0 = last.get("model_name", model_name)
                tone0 = last.get("tone_setting", tone_setting)

                context = st.session_state.get("context_text", "")
                system_msg, user_msg = build_expand_prompt(
                    topic0,
                    purpose0,
                    hypothesis0,
                    context,
                    st.session_state["result"],
                    expand_additional,
                    min_chars_per_para,
                    tone0,
                )
                st.session_state["result"] = call_openai_json(
                    api_key=user_api_key,
                    model=model0,
                    system_msg=system_msg,
                    user_msg=user_msg,
                    temperature=0.50,
                )
                st.session_state["expansion_level"] += 1
            except Exception as e:
                st.error(f"확장 중 오류가 발생했습니다: {e}")

# =========================================================
# 8) Results
# =========================================================
if st.session_state["result"]:
    res = st.session_state["result"]

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Results</div>', unsafe_allow_html=True)

    if st.session_state.get("expansion_level", 0) > 0:
        st.success(f"초안이 {st.session_state['expansion_level']}회 확장되었습니다.")

    tab1, tab2 = st.tabs(["📋 상세 설계 개요(간결)", "✍️ 각주 포함 초안(전문적)"])

    with tab1:
        for section, detail in res.get("detailed_outline", {}).items():
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown(f"<div class='h3'>{section}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='help'>{detail}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        source_map = res.get("source_map", {})
        for section, text in res.get("interactive_draft", {}).items():
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown(f"<div class='h3'>{section}</div>", unsafe_allow_html=True)
            render_text_with_ref_popovers(text, source_map)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <div class="glass">
          <div class="card-title">Status</div>
          <div class="help">선행연구 PDF를 업로드하고 “분석 및 상세 초안 생성”을 실행하면 결과가 여기에 표시됩니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="footer">© 2026 Report Mate</div>', unsafe_allow_html=True)
