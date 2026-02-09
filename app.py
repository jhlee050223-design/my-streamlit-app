import os
import io
import json
import base64
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

# -----------------------------
# 1. 환경 설정 및 초기화
# -----------------------------
load_dotenv()
DEFAULT_MODEL = "gpt-4o-mini" 

st.set_page_config(
    page_title="Report mate - 다중 논문 분석",
    layout="wide",
)

if "result" not in st.session_state:
    st.session_state["result"] = None
if "pdf_files_dict" not in st.session_state:
    st.session_state["pdf_files_dict"] = {}

# -----------------------------
# 2. 유틸리티 함수
# -----------------------------
def center_title(text: str):
    st.markdown(
        f"""
        <style>
          .rm-title {{ text-align: center; font-size: 32px; font-weight: 800; color: #1E3A8A; padding: 10px 0; }}
          .rm-sub {{ text-align: center; opacity: 0.8; margin-top: -10px; margin-bottom: 20px; font-size: 16px; }}
        </style>
        <div class="rm-title">{text}</div>
        <div class="rm-sub">여러 권의 논문 자료를 분석하여 학술적 개요와 초안 작성을 돕습니다.</div>
        """,
        unsafe_allow_html=True,
    )

def read_pdf_text(uploaded_files: List) -> str:
    """여러 개의 PDF에서 텍스트를 추출하고 구조화합니다."""
    all_text = []
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = f"\n[출처 파일: {uploaded_file.name}]\n"
            # 각 논문당 핵심 내용이 몰려있는 앞부분 10페이지 위주로 추출
            for page in reader.pages[:10]:
                content = page.extract_text()
                if content:
                    text += content
            all_text.append(text)
        except Exception as e:
            st.error(f"{uploaded_file.name} 읽기 실패: {e}")
    
    combined = "\n".join(all_text)
    # LLM 컨텍스트 한계를 고려하여 최대 약 30,000자 제한
    return combined[:30000] + ("..." if len(combined) > 30000 else "")

def pdf_viewer_iframe(pdf_bytes: bytes, height: int = 800):
    """Base64 인코딩을 통한 PDF 뷰어"""
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}px" 
        style="border: 1px solid #E2E8F0; border-radius: 12px;" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

@dataclass
class GenerateParams:
    topic: str
    purpose: str
    hypothesis: str
    citation_style: str
    writing_style: str
    language: str
    model: str

def call_openai_api(prompt: str, model: str, api_key: str) -> Dict[str, Any]:
    """OpenAI API 호출 및 JSON 파싱"""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional academic research assistant. You must respond in valid JSON format only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# -----------------------------
# 3. 사이드바 구성 (설정 및 API 키)
# -----------------------------
with st.sidebar:
    st.header("🔐 API 설정")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if not user_api_key:
        st.warning("분석을 시작하려면 API 키를 입력하세요.")
    
    st.divider()
    st.header("📝 작성 옵션")
    citation_style = st.selectbox("인용 스타일", ["APA", "MLA", "Chicago", "IEEE"], index=0)
    writing_style = st.selectbox("문체 스타일", ["학술적(Professional)", "간결(Concise)", "설명적(Descriptive)"], index=0)
    language = st.selectbox("출력 언어", ["한국어", "English"], index=0)
    model_name = st.text_input("사용 모델", value=DEFAULT_MODEL)

    st.divider()
    if st.button("🔄 모든 데이터 초기화", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# -----------------------------
# 4. 메인 화면 구성
# -----------------------------
center_title("리포트 메이트 (Report mate)")

# 상단 입력부
st.markdown("### 1. 연구 기본 정보")
row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    topic = st.text_input("연구 주제", placeholder="예: 생성형 AI의 교육적 활용")
with row1_col2:
    purpose = st.text_input("연구 목적", placeholder="예: 학습 효율성 증진 효과 분석")
with row1_col3:
    hypothesis = st.text_input("연구 가설", placeholder="예: AI 튜터 사용군이 일반 학습군보다 성취도가 높을 것이다")

st.markdown("### 2. 논문 자료 업로드 (다중 파일 가능)")
uploaded_files = st.file_uploader(
    "참고할 PDF 논문들을 모두 업로드하세요.", 
    type=["pdf"], 
    accept_multiple_files=True
)

# 세션에 파일 데이터 캐싱
if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state["pdf_files_dict"]:
            st.session_state["pdf_files_dict"][f.name] = f.getvalue()

st.divider()

# -----------------------------
# 5. 분석 및 결과 뷰어 (2컬럼 레이아웃)
# -----------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📁 업로드된 자료 확인")
    if st.session_state["pdf_files_dict"]:
        file_names = list(st.session_state["pdf_files_dict"].keys())
        selected_file = st.selectbox("내용을 확인할 파일을 선택하세요", file_names)
        pdf_viewer_iframe(st.session_state["pdf_files_dict"][selected_file])
    else:
        st.info("업로드된 논문이 없습니다. 위에서 PDF 파일을 추가해 주세요.")

with right_col:
    st.subheader("💡 AI 분석 및 초안 생성")
    
    # 생성 버튼
    if st.button("🚀 분석 및 초안 작성 시작", type="primary", use_container_width=True):
        if not user_api_key:
            st.error("사이드바에 OpenAI API Key를 입력해야 합니다.")
        elif not uploaded_files:
            st.error("분석할 PDF 파일을 최소 하나 이상 업로드하세요.")
        elif not topic:
            st.error("연구 주제를 입력하세요.")
        else:
            with st.spinner("여러 논문 데이터를 통합 분석 중입니다. 잠시만 기다려 주세요..."):
                try:
                    # 텍스트 추출 및 프롬프트 빌드
                    context_text = read_pdf_text(uploaded_files)
                    
                    prompt = f"""
                    당신은 전문적인 연구 보조원입니다. 다음 제공된 여러 편의 논문 내용을 바탕으로 연구 리포트의 개요와 초안을 작성하세요.
                    
                    [연구 정보]
                    - 주제: {topic}
                    - 목적: {purpose}
                    - 가설: {hypothesis}
                    
                    [제공된 논문 텍스트 발췌]
                    {context_text}
                    
                    [지시 사항]
                    1. 제공된 텍스트의 내용을 바탕으로 인용을 포함하여 작성할 것.
                    2. 인용 스타일은 {citation_style}를 따를 것.
                    3. 문체는 {writing_style}로, 언어는 {language}로 작성할 것.
                    4. 결과는 반드시 다음 JSON 구조를 유지할 것:
                    {{
                        "outline": {{
                            "서론": ["소제목1", "소제목2"],
                            "이론적 배경": ["소제목1", "소제목2"],
                            "연구방법": ["소제목1"],
                            "결론": ["소제목1"]
                        }},
                        "draft": {{
                            "서론": "초안 내용...",
                            "이론적 배경": "초안 내용...",
                            "연구방법": "초안 내용...",
                            "결론": "초안 내용..."
                        }},
                        "references": ["참고문헌1", "참고문헌2"]
                    }}
                    """
                    
                    # API 호출
                    result = call_openai_api(prompt, model_name, user_api_key)
                    st.session_state["result"] = result
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {e}")

    # 결과 출력 탭
    if st.session_state["result"]:
        res = st.session_state["result"]
        tab1, tab2, tab3 = st.tabs(["📊 상세 개요", "📝 섹션별 초안", "📚 참고문헌"])
        
        with tab1:
            for section, subs in res.get("outline", {}).items():
                with st.expander(f"**{section}**", expanded=True):
                    for sub in subs:
                        st.markdown(f"- {sub}")
        
        with tab2:
            for section, content in res.get("draft", {}).items():
                st.markdown(f"#### {section}")
                st.info(content)
        
        with tab3:
            for ref in res.get("references", []):
                st.markdown(f"- {ref}")
    else:
        st.caption("분석 시작 버튼을 누르면 AI가 생성한 개요와 초안이 여기에 표시됩니다.")

st.markdown("---")
st.caption("© 2024 Report Mate - Academic Writing Assistant")
