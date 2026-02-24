import io
import json
import re
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# -----------------------------
# 1. Page Configuration & Apple UX Style
# -----------------------------
st.set_page_config(page_title="Report Mate", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #FBFBFD;
        color: #1D1D1F;
    }

    .report-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.04);
        margin-bottom: 25px;
        border: 1px solid #F2F2F7;
    }

    .main-header {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-align: center;
        padding-top: 40px;
        margin-bottom: 5px;
    }

    .sub-header {
        font-size: 17px;
        color: #86868B;
        text-align: center;
        margin-bottom: 40px;
    }

    .stTextInput>div>div>input, .stFileUploader section, .stTextArea textarea, .stSelectbox>div>div {
        border-radius: 12px !important;
    }

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background-color: #0071E3;
        color: white;
        font-weight: 600;
        padding: 12px;
        transition: all 0.2s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #0077ED;
        box-shadow: 0 4px 15px rgba(0,113,227,0.3);
    }

    div[data-testid="stPopover"] > button {
        background-color: #F5F5F7 !important;
        color: #0071E3 !important;
        border: 1px solid #D2D2D7 !important;
        border-radius: 8px !important;
        padding: 2px 8px !important;
        font-size: 12px !important;
        min-height: 24px !important;
        margin: 0 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# 2. Session State Initialization
# -----------------------------
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

# -----------------------------
# 3. Sidebar (Gemini-style options, but keep your core UX)
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model_name = st.text_input("Model", value="gpt-4o-mini")

    st.divider()
    st.subheader("📝 Draft Options (기본 길이 강화)")
    # Gemini style: length & tone
    base_paras = st.select_slider("소절당 문단 수(기본)", options=[2, 3], value=2)
    min_chars_per_para = st.select_slider("문단 최소 글자 수", options=[200, 250, 300, 400], value=200)

    tone_setting = st.selectbox("어조", ["Academic", "Formal", "Analytical"], index=0)
    # 확장 버튼 눌렀을 때 추가되는 문단 수
    expand_additional = st.select_slider("확장 시 소절당 추가 문단", options=[1, 2], value=1)

    st.divider()
    if st.button("새 프로젝트 시작", use_container_width=True):
        st.session_state.clear()
        init_state()
        st.rerun()

# -----------------------------
# 4. Main UI
# -----------------------------
st.markdown('<div class="main-header">Report Mate</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">선행연구를 분석하여 논문의 논리 구조를 설계합니다.</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown("#### 🖋️ 연구 맥락 설정")

    topic = st.text_input("연구 주제", placeholder="예: 생성형 AI가 대학생의 학술적 글쓰기에 미치는 영향")

    col1, col2 = st.columns(2)
    with col1:
        purpose = st.text_input("연구 목적", placeholder="연구를 통해 무엇을 밝히고 싶나요?")
    with col2:
        hypothesis = st.text_input("연구 가설", placeholder="예상되는 결론은 무엇인가요?")

    uploaded_files = st.file_uploader(
        "선행연구 PDF 업로드 (다중 선택 가능)", type=["pdf"], accept_multiple_files=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 5. Business Logic
# -----------------------------
def get_combined_text_with_meta(files, max_pages_each=10, max_chars=35000):
    """
    Extract first N pages from each PDF with [SOURCE:..., PAGE:...] tags.
    (Keep your original lightweight approach; not RAG)
    """
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
"""

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
"""
    return system_msg, user_msg

def build_expand_prompt(topic, purpose, hypothesis, context, current_result, add_paras, min_chars_per_para, tone):
    system_msg = f"""
당신은 석사학위 논문을 다수 지도한 전문 학술 에디터입니다.
기존 초안을 더 전문적이고 더 길게 확장합니다. 근거(REF) 밀도와 논리 연결을 강화하세요.
{tone_instructions(tone)}
반드시 지정한 JSON 스키마로만 출력하세요.
"""

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
"""
    return system_msg, user_msg

def call_openai_json(api_key, model, system_msg, user_msg, temperature=0.45):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return json.loads(response.choices[0].message.content)

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

# -----------------------------
# 6. Actions: Generate + Expand (Buttons)
# -----------------------------
colA, colB = st.columns(2)
with colA:
    generate_clicked = st.button("🚀 분석 및 상세 초안 생성", type="primary")
with colB:
    expand_clicked = st.button("➕ 초안 확장(추가 작성)", disabled=(st.session_state["result"] is None))

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
                    temperature=0.5,
                )
                st.session_state["expansion_level"] += 1
            except Exception as e:
                st.error(f"확장 중 오류가 발생했습니다: {e}")

# -----------------------------
# 7. Result Display
# -----------------------------
if st.session_state["result"]:
    res = st.session_state["result"]
    st.markdown("---")

    if st.session_state.get("expansion_level", 0) > 0:
        st.info(f"초안이 {st.session_state['expansion_level']}회 확장되었습니다.")

    tab1, tab2 = st.tabs(["📋 상세 설계 개요(간결)", "✍️ 각주 포함 초안(전문적)"])

    with tab1:
        for section, detail in res.get("detailed_outline", {}).items():
            st.markdown(
                f"""
                <div class="report-card">
                    <div style="color: #0071E3; font-weight: 700; font-size: 18px; margin-bottom: 12px;">{section}</div>
                    <div style="color: #424245; line-height: 1.7; font-size: 15px;">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab2:
        source_map = res.get("source_map", {})
        for section, text in res.get("interactive_draft", {}).items():
            st.markdown(f"### {section}")
            render_text_with_ref_popovers(text, source_map)
            st.divider()
else:
    st.markdown(
        "<br><br><p style='text-align: center; color: #BFBFC3;'>선행연구를 업로드하고 분석을 시작하여 논문 초안을 확인하세요.</p>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<p style="text-align: center; color: #D2D2D7; font-size: 12px; margin-top: 50px;">© 2026 Report Mate. Designed for Academic Excellence.</p>',
    unsafe_allow_html=True,
)
