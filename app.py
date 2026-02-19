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

    .stTextInput>div>div>input, .stFileUploader section {
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
if "result" not in st.session_state:
    st.session_state["result"] = None
if "context_text" not in st.session_state:
    st.session_state["context_text"] = ""
if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = {}
if "expansion_level" not in st.session_state:
    st.session_state["expansion_level"] = 0  # counts how many times user expanded

# -----------------------------
# 3. Sidebar (Settings)
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model_name = st.text_input("Model", value="gpt-4o-mini")

    st.divider()
    st.markdown("### 🧩 Draft Length")
    # 기본 길이: 소절당 2~3문단
    base_paras = st.slider("기본 문단 수(소절당)", min_value=2, max_value=3, value=2, step=1)
    # 확장 시 추가되는 문단 수
    expand_additional = st.slider("확장 시 추가 문단(소절당)", min_value=1, max_value=2, value=1, step=1)
    min_chars_per_para = st.number_input("문단 최소 글자수", min_value=120, max_value=600, value=200, step=10)

    st.divider()
    if st.button("새 프로젝트 시작", use_container_width=True):
        st.session_state.clear()
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
    """Extract first N pages (default 10) with SOURCE/PAGE tags."""
    text_data = ""
    for f in files:
        reader = PdfReader(io.BytesIO(f.getvalue()))
        file_name = f.name
        for i, page in enumerate(reader.pages[:max_pages_each]):
            content = page.extract_text()
            if content:
                text_data += f"\n[SOURCE: {file_name}, PAGE: {i+1}]\n{content}\n"
    return text_data[:max_chars]


def build_initial_prompt(topic, purpose, hypothesis, context, base_paras, min_chars_per_para):
    system_msg = """
당신은 석사학위 논문을 다수 지도한 전문 학술 에디터입니다.
제공된 자료에 근거해 엄밀한 학술 문체(석사 논문 수준)로 서술하고, 주장-근거-비판적 논의-연구 공백/기여의 연결을 분명히 합니다.
반드시 지정한 JSON 스키마로만 출력하세요.
"""

    user_msg = f"""
주제: {topic}
목적: {purpose}
가설: {hypothesis}

[자료 원문]
{context}

[출력 언어]
- 한국어

[핵심 요구사항]
1) detailed_outline (간결): 각 섹션(서론/이론적 배경/연구방법/결론)당 6~10문장 이내로 '전개 전략'을 요약(어떤 논문을 어떤 논거로 어떻게 엮을지).
2) interactive_draft (전문적/충분한 분량):
   - 각 섹션을 소절로 나누어 작성: 예) 서론은 1.1~1.4, 이론적 배경은 2.1~2.4, 연구방법은 3.1~3.5, 결론은 4.1~4.4(필요 시 조정 가능).
   - 각 소절은 최소 {base_paras}개 문단으로 작성.
   - 각 문단은 최소 {min_chars_per_para}자 이상(한국어 기준)으로 작성.
   - 각 문단에는 최소 1개의 인용 태그 [REF:파일명,p숫자]를 반드시 포함(가능하면 2개).
   - 문단 전개는 (주장/요지 → 근거와 선행연구 연결 → 비판적 논의/한계 → 연구 공백 및 본 연구의 위치화) 요소를 균형 있게 포함.
   - 막연한 추정 표현만 반복하지 말고, 자료에 근거한 연결 논리를 명시.
3) source_map:
   - 각 [REF:...] 태그에 대응하는 '해당 페이지의 핵심 근거 요약'을 구체적으로 작성(왜 그 문장을 지지하는지).
4) REF 규칙:
   - REF 태그 포맷은 반드시 정확히 [REF:파일명,p숫자]
   - 파일명은 SOURCE에 나온 파일명을 그대로 사용할 것.
   - 페이지 숫자는 SOURCE의 PAGE 값을 근거로 할 것.

[JSON 스키마 - 반드시 이 구조로만 출력]
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


def build_expand_prompt(topic, purpose, hypothesis, context, current_result, add_paras, min_chars_per_para):
    system_msg = """
당신은 석사학위 논문을 다수 지도한 전문 학술 에디터입니다.
아래의 기존 초안을 더 전문적이고 더 길게 '확장'합니다. 근거(REF) 밀도를 유지하면서 논리적 연결과 비판적 논의를 강화하세요.
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
- interactive_draft만 '더 길게' 확장하세요. (detailed_outline은 기존 수준 유지 또는 약간만 다듬기)
- 각 소절(예: 1.1, 1.2...)마다 문단을 추가로 {add_paras}개씩 더 작성하세요.
- 새로 추가되는 각 문단은 최소 {min_chars_per_para}자 이상.
- 새 문단마다 최소 1개의 [REF:파일명,p숫자]를 반드시 포함(가능하면 2개).
- source_map에는 새로 등장한 REF가 있다면 반드시 추가하고, 기존 REF 매핑도 유지/보강하세요.
- 기존 서술과 모순되지 않게 하되, 학술적 연결어/개념 정교화/한계 및 연구공백을 더 명확히 하세요.

[JSON 스키마 - 반드시 이 구조로만 출력]
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


def call_openai_json(api_key, model, system_msg, user_msg, temperature=0.4):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return json.loads(response.choices[0].message.content)


def render_text_with_ref_popovers(text, source_map):
    """
    Render markdown-ish text, splitting out [REF:...] as popover buttons.
    Uses st.markdown for buffered text to preserve headings like '### 1.1 ...'
    """
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
# 6. Generate / Expand Buttons
# -----------------------------
colA, colB = st.columns(2)

with colA:
    generate_clicked = st.button("🚀 분석 및 상세 초안 생성", type="primary")

with colB:
    expand_clicked = st.button("➕ 초안 확장(추가 작성)", type="secondary", disabled=(st.session_state["result"] is None))

if generate_clicked:
    if not user_api_key or not uploaded_files or not topic:
        st.warning("API 키, 주제, 그리고 파일을 모두 입력했는지 확인해주세요.")
    else:
        with st.spinner("선행연구들을 교차 분석하며 상세 설계안/초안을 작성 중입니다..."):
            try:
                context = get_combined_text_with_meta(uploaded_files)
                st.session_state["context_text"] = context
                st.session_state["last_inputs"] = {
                    "topic": topic,
                    "purpose": purpose,
                    "hypothesis": hypothesis,
                    "base_paras": base_paras,
                    "min_chars_per_para": min_chars_per_para,
                    "model_name": model_name,
                }
                st.session_state["expansion_level"] = 0

                system_msg, user_msg = build_initial_prompt(
                    topic=topic,
                    purpose=purpose,
                    hypothesis=hypothesis,
                    context=context,
                    base_paras=base_paras,
                    min_chars_per_para=min_chars_per_para,
                )
                st.session_state["result"] = call_openai_json(
                    api_key=user_api_key,
                    model=model_name,
                    system_msg=system_msg,
                    user_msg=user_msg,
                    temperature=0.4,
                )
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")

if expand_clicked:
    # 확장은 "현재 결과 + 같은 컨텍스트"로 추가 생성
    if not user_api_key:
        st.warning("먼저 OpenAI API Key를 입력해주세요.")
    elif st.session_state["result"] is None:
        st.warning("먼저 초안을 생성해주세요.")
    else:
        with st.spinner("초안을 더 전문적으로 확장 작성 중입니다..."):
            try:
                context = st.session_state.get("context_text", "")
                last = st.session_state.get("last_inputs", {})

                # 입력값이 바뀌었더라도, 확장은 '마지막 생성 기준'으로 일관되게 진행
                topic0 = last.get("topic", topic)
                purpose0 = last.get("purpose", purpose)
                hypothesis0 = last.get("hypothesis", hypothesis)
                model0 = last.get("model_name", model_name)

                system_msg, user_msg = build_expand_prompt(
                    topic=topic0,
                    purpose=purpose0,
                    hypothesis=hypothesis0,
                    context=context,
                    current_result=st.session_state["result"],
                    add_paras=expand_additional,
                    min_chars_per_para=min_chars_per_para,
                )
                st.session_state["result"] = call_openai_json(
                    api_key=user_api_key,
                    model=model0,
                    system_msg=system_msg,
                    user_msg=user_msg,
                    temperature=0.45,
                )
                st.session_state["expansion_level"] += 1
            except Exception as e:
                st.error(f"확장 중 오류가 발생했습니다: {e}")

# -----------------------------
# 7. Result Display (Apple UX)
# -----------------------------
if st.session_state["result"]:
    res = st.session_state["result"]
    st.markdown("---")

    # 확장 상태 표시
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
        draft = res.get("interactive_draft", {})

        for section, text in draft.items():
            st.markdown(f"### {section}")
            # 소절(### 1.1 ...)이 있으면 그대로 마크다운으로 표시되도록 처리
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
