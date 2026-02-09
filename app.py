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
# 1. 환경 설정 및 세션 초기화
# -----------------------------
load_dotenv()
DEFAULT_MODEL = "gpt-4o-mini" 

st.set_page_config(
    page_title="Report Mate - 학술 연구 보조",
    layout="wide",
)

if "result" not in st.session_state:
    st.session_state["result"] = None
if "pdf_files_dict" not in st.session_state:
    st.session_state["pdf_files_dict"] = {}

# -----------------------------
# 2. 유틸리티 함수 (텍스트 추출 및 뷰어)
# -----------------------------
def center_title(text: str):
    st.markdown(
        f"""
        <style>
          .rm-title {{ text-align: center; font-size: 32px; font-weight: 800; color: #1E3A8A; padding: 10px 0; }}
          .rm-sub {{ text-align: center; opacity: 0.8; margin-top: -10px; margin-bottom: 20px; font-size: 16px; }}
        </style>
        <div class="rm-title">{text}</div>
        <div class="rm-sub">선행연구 종합 분석 및 각주가 포함된 초안 작성을 지원합니다.</div>
        """,
        unsafe_allow_html=True,
    )

def read_pdf_text_with_metadata(uploaded_files: List) -> str:
    """파일명과 페이지 번호를 포함하여 텍스트를 추출합니다 (각주 생성용)"""
    structured_context = []
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            file_name = uploaded_file.name
            # 주요 내용이 있는 앞부분 위주로 추출
            for page_num, page in enumerate(reader.pages[:15]):
                content = page.extract_text()
                if content:
                    # AI가 출처를 명확히 알 수 있도록 텍스트 덩어리마다 메타데이터 태깅
                    structured_context.append(f"--- SOURCE: {file_name}, PAGE: {page_num+1} ---\n{content}\n")
        except Exception as e:
            st.error(f"{file_name} 읽기 중 오류: {e}")
    
    combined = "\n".join(structured_context)
    return combined[:35000] # 토큰 제한 고려

def pdf_viewer_iframe(pdf_bytes: bytes, height: int = 800):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}px" 
        style="border: 1px solid #E2E8F0; border-radius: 12px;" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# -----------------------------
# 3. 사이드바 (API 키 및 옵션)
# -----------------------------
with st.sidebar:
    st.header("🔐 API & 설정")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    
    st.divider()
    citation_style = st.selectbox("인용 스타일", ["APA", "MLA", "Chicago", "IEEE"], index=0)
    language = st.selectbox("출력 언어", ["한국어", "English"], index=0)
    model_name = st.text_input("사용 모델", value=DEFAULT_MODEL)

    if st.button("🔄 초기화", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# -----------------------------
# 4. 메인 화면 구성
# -----------------------------
center_title("리포트 메이트 (Report mate)")

st.markdown("### 📝 연구 주제 및 선행연구 업로드")
c1, c2, c3 = st.columns(3)
with c1: topic = st.text_input("연구 주제", placeholder="예: 생성형 AI의 교육적 효과")
with c2: purpose = st.text_input("연구 목적", placeholder="예: 학습 성취도 변화 분석")
with c3: hypothesis = st.text_input("연구 가설", placeholder="예: 맞춤형 피드백이 성적을 높일 것이다")

uploaded_files = st.file_uploader("선행연구 PDF 파일들을 업로드하세요 (다중 선택 가능)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state["pdf_files_dict"]:
            st.session_state["pdf_files_dict"][f.name] = f.getvalue()

st.divider()

# -----------------------------
# 5. 분석 실행 및 결과 레이아웃
# -----------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📁 자료 확인 (Viewer)")
    if st.session_state["pdf_files_dict"]:
        selected_file = st.selectbox("파일 선택", list(st.session_state["pdf_files_dict"].keys()))
        pdf_viewer_iframe(st.session_state["pdf_files_dict"][selected_file])
    else:
        st.info("업로드된 자료가 없습니다.")

with right_col:
    st.subheader("🤖 AI 선행연구 종합 분석")
    
    if st.button("🚀 종합 개요 및 초안 생성", type="primary", use_container_width=True):
        if not user_api_key:
            st.error("사이드바에 API 키를 입력해주세요.")
        elif not uploaded_files:
            st.error("분석할 선행연구 파일을 업로드해주세요.")
        else:
            with st.spinner("NotebookLM 방식으로 자료를 교차 분석 중입니다..."):
                try:
                    client = OpenAI(api_key=user_api_key)
                    context_data = read_pdf_text_with_metadata(uploaded_files)
                    
                    system_prompt = f"""
                    당신은 학술 논문 작성 조교입니다. NotebookLM과 같이 제공된 소스(선행연구)만을 바탕으로 답변해야 합니다.
                    사용자가 입력한 주제에 맞춰 '세부 개요'와 '초안'을 작성하세요.
                    
                    [핵심 요구사항]
                    1. 모든 초안의 문장 또는 단락 끝에는 반드시 출처를 각주 형태로 표기하세요. (예: [파일명, p.숫자])
                    2. 여러 파일의 내용을 종합하여 '이론적 배경'과 '선행연구 검토' 섹션을 풍부하게 작성하세요.
                    3. 학술적인 톤을 유지하고, 한국어로 답변하세요.
                    """
                    
                    user_prompt = f"""
                    연구 주제: {topic}
                    연구 목적: {purpose}
                    연구 가설: {hypothesis}
                    인용 스타일: {citation_style}

                    [제공된 선행연구 자료]
                    {context_data}

                    결과는 반드시 아래 JSON 구조로만 출력하세요:
                    {{
                        "outline": {{
                            "서론": ["소제목1", "소제목2"],
                            "이론적 배경": ["소제목1", "소제목2"],
                            "연구방법": ["소제목1"],
                            "결론": ["소제목1"]
                        }},
                        "draft": {{
                            "서론": "각주가 포함된 초안 내용...",
                            "이론적 배경": "각주가 포함된 초안 내용...",
                            "연구방법": "각주가 포함된 초안 내용...",
                            "결론": "각주가 포함된 초안 내용..."
                        }},
                        "references": ["사용한 참고문헌 리스트"]
                    }}
                    """
                    
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3,
                        response_format={"type": "json_object"}
                    )
                    
                    st.session_state["result"] = json.loads(response.choices[0].message.content)
                    st.success("분석 완료!")
                except Exception as e:
                    st.error(f"오류 발생: {e}")

    # 결과 디스플레이
    if st.session_state["result"]:
        res = st.session_state["result"]
        t1, t2, t3 = st.tabs(["📊 논리 개요", "📝 각주 포함 초안", "📚 참고문헌"])
        
        with t1:
            for sec, subs in res.get("outline", {}).items():
                with st.expander(sec, expanded=True):
                    for s in subs: st.write(f"• {s}")
        
        with t2:
            for sec, content in res.get("draft", {}).items():
                st.markdown(f"**{sec}**")
                st.write(content)
                st.divider()
        
        with t3:
            for r in res.get("references", []):
                st.markdown(f"- {r}")
    else:
        st.caption("자료를 업로드하고 생성 버튼을 누르면 NotebookLM급 분석 결과가 표시됩니다.")

st.markdown("---")
st.caption("본 도구는 선행연구 데이터를 바탕으로 논문 작성을 돕는 연구 보조 도구입니다.")
