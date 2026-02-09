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

# iOS/Apple 느낌의 커스텀 CSS (애니메이션 및 미니멀리즘 강조)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #FBFBFD;
        color: #1D1D1F;
    }
    
    /* 카드 디자인 */
    .report-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.04);
        margin-bottom: 25px;
        border: 1px solid #F2F2F7;
    }
    
    /* 제목 스타일 */
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
    
    /* 입력창 및 버튼 모서리 둥글게 */
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
    
    /* 각주 팝오버 버튼 스타일 */
    div[data-testid="stPopover"] > button {
        background-color: #F5F5F7 !important;
        color: #0071E3 !important;
        border: 1px solid #D2D2D7 !important;
        border-radius: 8px !important;
        padding: 2px 8px !important;
        font-size: 12px !important;
        min-height: 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 2. Session State Initialization
# -----------------------------
if "result" not in st.session_state:
    st.session_state["result"] = None

# -----------------------------
# 3. Sidebar (설정 간소화)
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model_name = st.text_input("Model", value="gpt-4o-mini")
    
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
        
    uploaded_files = st.file_uploader("선행연구 PDF 업로드 (다중 선택 가능)", type=["pdf"], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 5. Business Logic: PDF Analysis & AI Generation
# -----------------------------
def get_combined_text_with_meta(files):
    text_data = ""
    for f in files:
        reader = PdfReader(io.BytesIO(f.getvalue()))
        file_name = f.name
        # 논문의 핵심 정보가 포함된 앞부분 10페이지 추출
        for i, page in enumerate(reader.pages[:10]):
            content = page.extract_text()
            if content:
                text_data += f"\n[SOURCE: {file_name}, PAGE: {i+1}]\n{content}\n"
    return text_data[:35000]

if st.button("🚀 분석 및 상세 초안 생성", type="primary"):
    if not user_api_key or not uploaded_files or not topic:
        st.warning("API 키, 주제, 그리고 파일을 모두 입력했는지 확인해주세요.")
    else:
        with st.spinner("선행연구들을 교차 분석하며 상세 설계안을 작성 중입니다..."):
            try:
                client = OpenAI(api_key=user_api_key)
                context = get_combined_text_with_meta(uploaded_files)
                
                system_msg = "당신은 전문 학술 에디터입니다. 제공된 자료를 분석하여 '논문별 특징이 반영된 아주 상세한 개요'와 '클릭 가능한 출처가 포함된 초안'을 JSON으로 작성하세요."
                
                user_msg = f"""
                주제: {topic} / 목적: {purpose} / 가설: {hypothesis}
                
                [자료 원문]
                {context}
                
                [요구사항]
                1. detailed_outline: 각 섹션별로 어떤 논문의 어떤 논거를 인용하여 전개할지 전략을 상세히 서술.
                2. interactive_draft: 본문 중간중간 출처가 필요한 시점에 [REF:파일명,p숫자] 태그를 반드시 삽입할 것.
                3. source_map: 각 [REF:...] 태그에 대응하는 상세 근거(해당 페이지의 핵심 내용)를 설명.
                4. 언어는 한국어로 작성.
                
                반드시 아래 구조의 JSON으로만 출력:
                {{
                    "detailed_outline": {{ "서론": "...", "이론적 배경": "...", "연구방법": "...", "결론": "..." }},
                    "interactive_draft": {{ "서론": "...", "이론적 배경": "...", "연구방법": "...", "결론": "..." }},
                    "source_map": {{ "[REF:파일명,p숫자]": "이 논문에서 강조한 ~내용을 인용함" }}
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
                st.error(f"분석 중 오류가 발생했습니다: {e}")

# -----------------------------
# 6. Result Display (Apple UX)
# -----------------------------
if st.session_state["result"]:
    res = st.session_state["result"]
    
    st.markdown("---")
    tab1, tab2 = st.tabs(["📋 상세 설계 개요", "✍️ 각주 포함 초안"])
    
    with tab1:
        for section, detail in res.get("detailed_outline", {}).items():
            st.markdown(f'''
                <div class="report-card">
                    <div style="color: #0071E3; font-weight: 600; font-size: 18px; margin-bottom: 12px;">{section}</div>
                    <div style="color: #424245; line-height: 1.7; font-size: 15px;">{detail}</div>
                </div>
            ''', unsafe_allow_html=True)
            
    with tab2:
        source_map = res.get("source_map", {})
        for section, text in res.get("interactive_draft", {}).items():
            st.markdown(f"#### {section}")
            
            # [REF:...] 패턴을 찾아 텍스트와 팝오버로 분리 렌더링
            parts = re.split(r'(\[REF:[^\]]+\])', text)
            
            # 텍스트 단락 구성을 위한 컨테이너
            para_container = st.container()
            with para_container:
                # 스트림릿에서 텍스트와 위젯을 인라인처럼 보이게 배치
                cols = st.columns([100]) # 넓은 단일 컬럼
                with cols[0]:
                    buffer = ""
                    for part in parts:
                        if part.startswith("[REF:"):
                            st.write(buffer) # 지금까지 쌓인 텍스트 출력
                            buffer = ""
                            ref_info = source_map.get(part, "상세 출처 정보를 불러올 수 없습니다.")
                            with st.popover(f"📍 {part}"):
                                st.markdown(f"**상세 근거:**\n{ref_info}")
                        else:
                            buffer += part
                    if buffer:
                        st.write(buffer)
            st.divider()
else:
    st.markdown("<br><br><p style='text-align: center; color: #BFBFC3;'>선행연구를 업로드하고 분석을 시작하여 논문 초안을 확인하세요.</p>", unsafe_allow_html=True)

st.markdown('<p style="text-align: center; color: #D2D2D7; font-size: 12px; margin-top: 50px;">© 2026 Report Mate. Designed for Academic Excellence.</p>', unsafe_allow_html=True)
