import io
import json
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# -----------------------------
# 1. Page Configuration & Apple UX Style
# -----------------------------
st.set_page_config(page_title="Report Mate", layout="centered")

# iOS/Apple 느낌의 커스텀 CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #FBFBFD;
        color: #1D1D1F;
    }
    
    .stButton>button {
        border-radius: 20px;
        border: none;
        background-color: #0071E3;
        color: white;
        padding: 8px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0077ED;
        transform: scale(1.02);
    }
    
    .report-card {
        background: white;
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #E5E5E7;
    }
    
    .cite-badge {
        display: inline-block;
        background: #F5F5F7;
        color: #0071E3;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        cursor: pointer;
        margin-left: 4px;
        border: 1px solid #D2D2D7;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 2. Session State Initialization
# -----------------------------
if "result" not in st.session_state:
    st.session_state["result"] = None

# -----------------------------
# 3. Sidebar (Simplified)
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model_name = st.text_input("Model", value="gpt-4o-mini")
    
    st.divider()
    if st.button("Clear All Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# -----------------------------
# 4. Main UI
# -----------------------------
st.markdown("<h1 style='text-align: center; font-weight: 800;'>Report Mate</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #86868B;'>선행연구 기반 맞춤형 논문 설계 도구</p>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown("### 🖋️ Research Context")
    c1, c2 = st.columns(2)
    with c1:
        topic = st.text_input("주제", placeholder="무엇을 연구하시나요?")
        purpose = st.text_input("목적", placeholder="연구의 의도는 무엇인가요?")
    with c2:
        hypothesis = st.text_input("가설", placeholder="예상되는 결과는?")
        uploaded_files = st.file_uploader("선행연구 PDF 업로드", type=["pdf"], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 5. Logic: Extraction & Generation
# -----------------------------
def get_combined_text(files):
    text_data = ""
    for f in files:
        reader = PdfReader(io.BytesIO(f.getvalue()))
        file_name = f.name
        # 각 논문의 핵심인 앞부분 12페이지 추출
        for i, page in enumerate(reader.pages[:12]):
            content = page.extract_text()
            if content:
                text_data += f"\n[DOC: {file_name}, PAGE: {i+1}]\n{content}\n"
    return text_data[:35000]

if st.button("🚀 분석 및 초안 생성", type="primary", use_container_width=True):
    if not user_api_key or not uploaded_files or not topic:
        st.error("API 키, 주제, 그리고 파일을 모두 확인해주세요.")
    else:
        with st.spinner("논문들을 교차 분석하여 상세 설계를 진행 중입니다..."):
            try:
                client = OpenAI(api_key=user_api_key)
                raw_context = get_combined_text(uploaded_files)
                
                system_msg = "당신은 전문 학술 에디터입니다. 제공된 자료를 분석하여 '논문별 특징이 반영된 아주 상세한 개요'와 '클릭 가능한 출처가 포함된 초안'을 JSON으로 작성하세요."
                
                user_msg = f"""
                주제: {topic} / 목적: {purpose} / 가설: {hypothesis}
                
                [자료 원문]
                {raw_context}
                
                [요구사항]
                1. outline: 각 섹션별로 어떤 논문의 어떤 이론을 인용할지 구체적인 전략을 포함하여 상세히 작성.
                2. draft: 본문 중간에 출처가 필요한 시점에 반드시 [REF:파일명,페이지] 태그를 삽입할 것. 
                3. 반드시 JSON 형식을 지킬 것.
                
                JSON 구조 예시:
                {{
                    "detailed_outline": {{ "서론": "전략적 내용...", "이론적 배경": "구체적 분석..." }},
                    "interactive_draft": {{ "서론": "내용 [REF:A논문,p1] 내용...", "본문": "..." }},
                    "source_map": {{ "[REF:A논문,p1]": "A논문 1페이지의 ~이론을 인용함" }}
                }}
                """
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                st.session_state["result"] = json.loads(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")

# -----------------------------
# 6. Result Display (iOS Style)
# -----------------------------
if st.session_state["result"]:
    res = st.session_state["result"]
    
    st.markdown("---")
    tab1, tab2 = st.tabs(["📋 상세 설계 개요", "✍️ 인터랙티브 초안"])
    
    with tab1:
        for section, detail in res.get("detailed_outline", {}).items():
            st.markdown(f'<div class="report-card"><b>{section}</b><br><p style="color: #424245; font-size: 0.95rem;">{detail}</p></div>', unsafe_allow_html=True)
            
    with tab2:
        source_map = res.get("source_map", {})
        for section, text in res.get("interactive_draft", {}).items():
            st.markdown(f"#### {section}")
            
            # 텍스트 내의 [REF:...]를 찾아 Streamlit UI 요소로 변환
            parts = text.split("[")
            display_text = ""
            
            container = st.container()
            with container:
                # 간단한 구현을 위해 텍스트와 도움말(각주)을 조합
                for part in parts:
                    if "]" in part:
                        ref_key_inner, rest = part.split("]", 1)
                        ref_key = "[" + ref_key_inner + "]"
                        st.write(display_text, Maryland="inline") # 이전 텍스트 출력
                        display_text = rest # 나머지 텍스트 저장
                        
                        # 클릭(Hover) 시 정보를 보여주는 각주 버튼
                        with st.expander(f"📍 출처: {ref_key_inner}"):
                            st.caption(source_map.get(ref_key, "상세 출처 정보를 불러올 수 없습니다."))
                    else:
                        display_text += part
                st.write(display_text)
            st.divider()

else:
    st.markdown("<br><br><p style='text-align: center; color: #BFBFC3;'>상단 버튼을 눌러 분석을 시작하세요.</p>", unsafe_allow_html=True)
