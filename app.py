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
# Config
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("REPORT_MATE_MODEL", "gpt-4.1-mini")

st.set_page_config(
    page_title="Report mate",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
def center_title(text: str):
    # Streamlit 상단을 "중앙 정렬"처럼 보이게 하는 간단한 CSS 트릭
    st.markdown(
        f"""
        <style>
          .rm-title {{
            text-align: center;
            font-size: 28px;
            font-weight: 700;
            padding: 6px 0 2px 0;
          }}
          .rm-sub {{
            text-align: center;
            opacity: 0.75;
            margin-top: -6px;
            margin-bottom: 10px;
          }}
        </style>
        <div class="rm-title">{text}</div>
        <div class="rm-sub">논문 자료 분석 · 학술 개요/초안 작성 보조</div>
        """,
        unsafe_allow_html=True,
    )

def read_pdf_text(pdf_bytes: bytes, max_chars: int = 20000) -> str:
    """PDF 텍스트 추출(간단 RAG 전처리). 긴 경우 앞부분만."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    for i, page in enumerate(reader.pages[:30]):  # 과도한 페이지는 제한
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            chunks.append("")
    text = "\n".join(chunks).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...텍스트가 길어 일부만 사용됨...]"
    return text

def pdf_viewer_iframe(pdf_bytes: bytes, height: int = 780):
    """업로드한 PDF를 좌측 패널에 표시(브라우저 내장 PDF 뷰어)."""
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{b64}"
            width="100%"
            height="{height}"
            style="border: 1px solid rgba(0,0,0,0.12); border-radius: 10px;"
            type="application/pdf"
        ></iframe>
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

def build_prompt(params: GenerateParams, pdf_text: str, bibliography: List[str]) -> str:
    bib_block = "\n".join([f"- {b}" for b in bibliography]) if bibliography else "- (없음)"
    return f"""
당신은 연구 보조 AI입니다. 사용자가 입력한 주제/목적/가설과 제공된 원문(PDF 텍스트 발췌)을 바탕으로,
학술적 개요와 초안을 생성하세요.

요구사항:
- 반드시 아래 섹션 구조로 "세부 개요"를 작성:
  1) 서론
  2) 이론적 배경
  3) 연구방법
  4) 결과(예상/가정 가능. 단, 실제 데이터가 없음을 명시)
  5) 결론
- 각 섹션에는 소제목(2~5개) + 각 소제목별 핵심 bullet(2~5개)을 포함
- 이어서 "초안 텍스트"를 섹션 구조 그대로 작성 (과장 금지, 학술 문체)
- 인용은 사용자가 고른 스타일({params.citation_style})을 따르되,
  원문 출처가 불명확하면 (출처불명)으로 표시하고 과도한 단정은 피함
- 언어: {params.language}
- 문체/스타일: {params.writing_style}
- 출력은 반드시 JSON 하나만 반환

JSON 스키마(반드시 준수):
{{
  "outline": {{
    "서론": [{{"title": "...", "bullets": ["...", "..."]}}],
    "이론적 배경": [{{"title": "...", "bullets": ["...", "..."]}}],
    "연구방법": [{{"title": "...", "bullets": ["...", "..."]}}],
    "결과": [{{"title": "...", "bullets": ["...", "..."]}}],
    "결론": [{{"title": "...", "bullets": ["...", "..."]}}]
  }},
  "draft": {{
    "서론": "...",
    "이론적 배경": "...",
    "연구방법": "...",
    "결과": "...",
    "결론": "..."
  }},
  "bibliography_suggestions": ["...", "..."]
}}

사용자 입력:
- 주제: {params.topic}
- 연구 목적: {params.purpose}
- 가설: {params.hypothesis}

사용자 참고문헌(있다면 우선 활용):
{bib_block}

PDF 텍스트(발췌):
\"\"\"
{pdf_text}
\"\"\"
""".strip()

def call_openai_json(prompt: str, model: str) -> Dict[str, Any]:
    """
    Responses API/Chat 계열은 계속 진화하므로,
    여기서는 가장 호환성 높은 'JSON만 출력하라' 방식 + 파싱으로 안전하게 처리합니다.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env를 확인하세요.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # (권장) Responses API 사용: platform 문서에서 추천 :contentReference[oaicite:1]{index=1}
    # SDK 버전에 따라 필드명이 조금씩 다를 수 있어, 실패 시 Chat Completions로 폴백합니다.
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
        )
        text = resp.output_text
    except Exception:
        # 폴백: Chat Completions :contentReference[oaicite:2]{index=2}
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        text = resp.choices[0].message.content or ""

    # JSON 파싱(모델이 코드펜스를 섞는 경우 제거)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text else text
        text = text.replace("json", "", 1).strip()

    return json.loads(text)

def init_session():
    st.session_state.setdefault("topic", "")
    st.session_state.setdefault("purpose", "")
    st.session_state.setdefault("hypothesis", "")
    st.session_state.setdefault("pdf_bytes", None)
    st.session_state.setdefault("pdf_text", "")
    st.session_state.setdefault("outline_json", None)
    st.session_state.setdefault("draft_json", None)
    st.session_state.setdefault("bib_items", [])
    st.session_state.setdefault("progress", 0)

init_session()

# -----------------------------
# UI: Header (Top center title)
# -----------------------------
center_title("리포트 메이트 (Report mate)")

# -----------------------------
# Sidebar: Bibliography + Styles
# -----------------------------
with st.sidebar:
    st.header("설정")
    st.caption("참고문헌 리스트 / 양식 / 스타일 변경")

    citation_style = st.selectbox("인용 스타일", ["APA", "MLA", "Chicago", "IEEE"], index=0)
    writing_style = st.selectbox("문체", ["학술적(기본)", "간결", "서술적", "비평적"], index=0)
    language = st.selectbox("언어", ["한국어", "English"], index=0)
    model = st.text_input("모델", value=DEFAULT_MODEL)

    st.divider()

    st.subheader("참고문헌 리스트")
    new_bib = st.text_input("항목 추가", placeholder="예: Author, A. (2023). Title. Journal...")
    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button("추가", use_container_width=True) and new_bib.strip():
            st.session_state["bib_items"].append(new_bib.strip())
            st.rerun()
    with col_clear:
        if st.button("비우기", use_container_width=True):
            st.session_state["bib_items"] = []
            st.rerun()

    if st.session_state["bib_items"]:
        for i, b in enumerate(st.session_state["bib_items"], start=1):
            st.write(f"{i}. {b}")

# -----------------------------
# Top: Topic input row
# -----------------------------
st.markdown("### 주제 입력")
top_c1, top_c2, top_c3 = st.columns([2, 2, 2])
with top_c1:
    st.session_state["topic"] = st.text_input("주제", value=st.session_state["topic"], placeholder="연구 주제를 입력하세요")
with top_c2:
    st.session_state["purpose"] = st.text_input("연구 목적", value=st.session_state["purpose"], placeholder="연구 목적을 입력하세요")
with top_c3:
    st.session_state["hypothesis"] = st.text_input("가설", value=st.session_state["hypothesis"], placeholder="가설을 입력하세요")

st.markdown("---")

# -----------------------------
# Center: Upload area
# -----------------------------
st.markdown("### 자료 업로드")
uploaded = st.file_uploader("인용 원문 소스(PDF)를 업로드하세요", type=["pdf"], accept_multiple_files=False)
if uploaded is not None:
    st.session_state["pdf_bytes"] = uploaded.read()
    st.session_state["pdf_text"] = ""  # 새 업로드면 텍스트 재추출
    st.success("PDF 업로드 완료")

# -----------------------------
# Action buttons + Progress (Bottom)
# -----------------------------
action_c1, action_c2, action_c3 = st.columns([1.2, 1.2, 3.6])

with action_c1:
    gen = st.button("개요/초안 생성", type="primary", use_container_width=True)
with action_c2:
    reset = st.button("초기화", use_container_width=True)
with action_c3:
    st.session_state["progress"] = st.slider("진행률", 0, 100, int(st.session_state["progress"]), 1)

if reset:
    for k in ["topic", "purpose", "hypothesis", "pdf_bytes", "pdf_text", "outline_json", "draft_json", "progress"]:
        st.session_state[k] = "" if k in ["topic", "purpose", "hypothesis", "pdf_text"] else None
    st.session_state["bib_items"] = []
    st.session_state["progress"] = 0
    st.rerun()

# -----------------------------
# Main 2-split: Left PDF, Right AI output
# -----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("인용된 원문 소스 (PDF)")
    if st.session_state["pdf_bytes"]:
        pdf_viewer_iframe(st.session_state["pdf_bytes"], height=820)
    else:
        st.info("좌측에 표시할 PDF가 아직 없습니다. 위에서 PDF를 업로드하세요.")

with right:
    st.subheader("AI 제안: 논리 구조 및 초안")
    tabs = st.tabs(["개요", "초안", "참고문헌 제안", "JSON 원본"])

    if gen:
        # 진행률을 "아래"에서 보여주고 싶으면 slider 대신 progress bar를 써도 됩니다.
        st.session_state["progress"] = 5

        if not st.session_state["topic"].strip():
            st.error("주제를 입력하세요.")
        elif not st.session_state["purpose"].strip():
            st.error("연구 목적을 입력하세요.")
        elif not st.session_state["hypothesis"].strip():
            st.error("가설을 입력하세요.")
        elif not st.session_state["pdf_bytes"]:
            st.error("PDF를 업로드하세요.")
        else:
            try:
                st.session_state["progress"] = 20
                if not st.session_state["pdf_text"]:
                    st.session_state["pdf_text"] = read_pdf_text(st.session_state["pdf_bytes"])
                st.session_state["progress"] = 45

                params = GenerateParams(
                    topic=st.session_state["topic"].strip(),
                    purpose=st.session_state["purpose"].strip(),
                    hypothesis=st.session_state["hypothesis"].strip(),
                    citation_style=citation_style,
                    writing_style=writing_style,
                    language=language,
                    model=model.strip() or DEFAULT_MODEL,
                )
                prompt = build_prompt(params, st.session_state["pdf_text"], st.session_state["bib_items"])
                st.session_state["progress"] = 65

                result = call_openai_json(prompt=prompt, model=params.model)
                st.session_state["outline_json"] = result.get("outline")
                st.session_state["draft_json"] = result.get("draft")
                st.session_state["bib_suggestions"] = result.get("bibliography_suggestions", [])
                st.session_state["raw_json"] = result
                st.session_state["progress"] = 100
                st.success("생성 완료!")
            except json.JSONDecodeError:
                st.session_state["progress"] = 0
                st.error("모델 출력이 JSON 형식이 아니어서 파싱에 실패했습니다. (모델/프롬프트를 조정해 보세요)")
            except Exception as e:
                st.session_state["progress"] = 0
                st.error(f"오류: {e}")

    outline = st.session_state.get("outline_json")
    draft = st.session_state.get("draft_json")
    bib_suggestions = st.session_state.get("bib_suggestions", [])

    with tabs[0]:
        if outline:
            for section, items in outline.items():
                st.markdown(f"#### {section}")
                for it in items:
                    st.markdown(f"- **{it.get('title','')}**")
                    bullets = it.get("bullets", [])
                    for b in bullets:
                        st.markdown(f"  - {b}")
        else:
            st.caption("아직 생성된 개요가 없습니다.")

    with tabs[1]:
        if draft:
            for section, text in draft.items():
                st.markdown(f"#### {section}")
                st.text_area(label=f"{section} 초안", value=text, height=180)
        else:
            st.caption("아직 생성된 초안이 없습니다.")

    with tabs[2]:
        if bib_suggestions:
            st.write("AI가 제안한 참고문헌(초안):")
            for b in bib_suggestions:
                st.markdown(f"- {b}")
        else:
            st.caption("아직 제안된 참고문헌이 없습니다.")

    with tabs[3]:
        raw = st.session_state.get("raw_json")
        if raw:
            st.json(raw)
        else:
            st.caption("아직 JSON 원본이 없습니다.")

# -----------------------------
# Bottom: Progress bar (진짜 하단 진행률)
# -----------------------------
st.markdown("---")
st.progress(int(st.session_state["progress"]))