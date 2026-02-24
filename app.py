import io
import json
import re
import math
from typing import List, Dict, Tuple

import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# Optional (but recommended) deps:
# pip install numpy pandas
import numpy as np
import pandas as pd


# =========================================================
# 1) Page Configuration & Apple UX Style
# =========================================================
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


# =========================================================
# 2) Session State Initialization
# =========================================================
def _init_state():
    defaults = {
        "result": None,
        "context_text": "",
        "last_inputs": {},
        "expansion_level": 0,
        # RAG store
        "rag_chunks": [],         # list[dict]: {text, file, page, chunk_id}
        "rag_embs": None,         # np.ndarray [N, D]
        "rag_ready": False,
        "rag_fingerprint": "",    # to know if we should rebuild
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# =========================================================
# 3) Sidebar (Settings + RAG + Length)
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model_name = st.text_input("Model", value="gpt-4o-mini")

    st.divider()
    st.markdown("### 🧩 Draft Length")
    base_paras = st.slider("기본 문단 수(소절당)", min_value=2, max_value=3, value=2, step=1)
    expand_additional = st.slider("확장 시 추가 문단(소절당)", min_value=1, max_value=2, value=1, step=1)
    min_chars_per_para = st.number_input("문단 최소 글자수", min_value=120, max_value=600, value=200, step=10)

    st.divider()
    st.markdown("### 🔎 RAG (검색 기반 인용 강화)")
    use_rag = st.checkbox("RAG 사용 (PDF에서 관련 문단을 검색해 컨텍스트 구성)", value=True)
    rag_top_k = st.slider("RAG Top-K (섹션별 가져올 조각 수)", 3, 12, 6, 1)
    rag_max_pages_each = st.slider("PDF당 최대 읽을 페이지 수", 5, 30, 12, 1)
    rag_chunk_chars = st.slider("Chunk 크기(문자 수)", 400, 1400, 900, 50)
    rag_overlap_chars = st.slider("Chunk overlap(문자 수)", 0, 400, 150, 10)

    # Embedding model (stable default; change if you use a different one)
    embedding_model = st.text_input("Embedding model", value="text-embedding-3-small")

    st.divider()
    if st.button("새 프로젝트 시작", use_container_width=True):
        st.session_state.clear()
        _init_state()
        st.rerun()


# =========================================================
# 4) Main UI
# =========================================================
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


# =========================================================
# 5) Helpers: PDF extraction, chunking, embeddings, retrieval
# =========================================================
def get_pdf_pages_text(files, max_pages_each=10) -> List[Dict]:
    """
    Return list of dicts: {file, page, text}
    """
    pages = []
    for f in files:
        reader = PdfReader(io.BytesIO(f.getvalue()))
        file_name = f.name
        for i, page in enumerate(reader.pages[:max_pages_each]):
            content = page.extract_text() or ""
            content = content.strip()
            if content:
                pages.append({"file": file_name, "page": i + 1, "text": content})
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple char-based chunker with overlap.
    """
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == n:
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def build_rag_store(files, api_key: str, embed_model: str, max_pages_each: int, chunk_size: int, overlap: int):
    """
    Build chunk list + embeddings and store in session_state.
    """
    client = OpenAI(api_key=api_key)

    pages = get_pdf_pages_text(files, max_pages_each=max_pages_each)
    chunks = []
    for p in pages:
        for j, ch in enumerate(chunk_text(p["text"], chunk_size, overlap)):
            chunks.append(
                {
                    "chunk_id": f'{p["file"]}::p{p["page"]}::c{j+1}',
                    "file": p["file"],
                    "page": p["page"],
                    "text": ch,
                }
            )

    if not chunks:
        raise ValueError("PDF에서 텍스트를 추출할 수 없습니다. (스캔본/이미지 PDF일 수 있어요)")

    # Embed in batches
    texts = [c["text"] for c in chunks]
    embs = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=embed_model, input=batch)
        embs.extend([d.embedding for d in resp.data])

    embs_np = np.array(embs, dtype=np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-12
    embs_np = embs_np / norms

    st.session_state["rag_chunks"] = chunks
    st.session_state["rag_embs"] = embs_np
    st.session_state["rag_ready"] = True


def embed_query(text: str, api_key: str, embed_model: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=embed_model, input=text)
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


def rag_retrieve(query: str, top_k: int, api_key: str, embed_model: str) -> List[Dict]:
    """
    Return top_k chunks: [{file,page,text,score,chunk_id}, ...]
    """
    if not st.session_state.get("rag_ready", False):
        return []

    q = embed_query(query, api_key, embed_model)
    embs = st.session_state["rag_embs"]  # [N, D]
    sims = embs @ q  # cosine because normalized

    k = min(top_k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    out = []
    chunks = st.session_state["rag_chunks"]
    for i in idx:
        c = chunks[int(i)]
        out.append(
            {
                "file": c["file"],
                "page": c["page"],
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "score": float(sims[int(i)]),
            }
        )
    return out


def build_section_context_rag(topic_: str, purpose_: str, hypothesis_: str, api_key: str, embed_model: str, top_k: int):
    """
    Build a compact, section-targeted context using RAG retrieval.
    """
    sections = {
        "서론": ["연구 배경", "문제 제기", "연구 필요성", "연구 질문"],
        "이론적 배경": ["핵심 개념 정의", "선행연구", "이론", "연구 공백"],
        "연구방법": ["연구 설계", "표본", "측정", "변수", "분석 방법", "타당도"],
        "결론": ["요약", "함의", "한계", "후속 연구"],
    }

    context_parts = []
    for sec, hints in sections.items():
        q = f"{topic_} / {purpose_} / {hypothesis_} / {sec} / {' '.join(hints)}"
        hits = rag_retrieve(q, top_k=top_k, api_key=api_key, embed_model=embed_model)
        if hits:
            context_parts.append(f"\n=== SECTION CONTEXT: {sec} ===\n")
            for h in hits:
                # Include SOURCE/PAGE tag for the model to create [REF:파일명,p숫자]
                context_parts.append(
                    f"[SOURCE: {h['file']}, PAGE: {h['page']}]\n{h['text']}\n"
                )

    return ("\n".join(context_parts)).strip()


def get_combined_text_with_meta(files, max_pages_each=10, max_chars=35000) -> str:
    """
    Fallback non-RAG context: first N pages combined.
    """
    text_data = ""
    for f in files:
        reader = PdfReader(io.BytesIO(f.getvalue()))
        file_name = f.name
        for i, page in enumerate(reader.pages[:max_pages_each]):
            content = page.extract_text() or ""
            content = content.strip()
            if content:
                text_data += f"\n[SOURCE: {file_name}, PAGE: {i+1}]\n{content}\n"
    return text_data[:max_chars]


def call_openai_json(api_key, model, system_msg, user_msg, temperature=0.45) -> Dict:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return json.loads(resp.choices[0].message.content)


# =========================================================
# 6) Prompt builders (RAG-ready)
# =========================================================
def build_initial_prompt(topic_, purpose_, hypothesis_, context, base_paras_, min_chars_):
    system_msg = """
당신은 석사학위 논문을 다수 지도한 전문 학술 에디터입니다.
제공된 자료에 근거해 엄밀한 학술 문체(석사 논문 수준)로 서술하고, 주장-근거-비판적 논의-연구 공백/기여의 연결을 분명히 합니다.
반드시 지정한 JSON 스키마로만 출력하세요.
"""

    user_msg = f"""
주제: {topic_}
목적: {purpose_}
가설: {hypothesis_}

[자료 원문]
{context}

[출력 언어]
- 한국어

[핵심 요구사항]
1) detailed_outline (간결): 각 섹션(서론/이론적 배경/연구방법/결론)당 6~10문장 이내로 '전개 전략' 요약.
2) interactive_draft (전문적/충분한 분량):
   - 각 섹션을 소절로 나누어 작성: 예) 서론은 1.1~1.4, 이론적 배경은 2.1~2.4, 연구방법은 3.1~3.5, 결론은 4.1~4.4 (필요 시 조정 가능).
   - 각 소절은 최소 {base_paras_}개 문단.
   - 각 문단은 최소 {min_chars_}자 이상.
   - 각 문단에는 최소 1개의 인용 태그 [REF:파일명,p숫자] 포함(가능하면 2개).
   - 문단 전개에 (주장/요지 → 근거 연결 → 비판적 논의/한계 → 연구 공백 및 본 연구의 위치화) 요소를 균형 있게 포함.
3) source_map:
   - 각 [REF:...] 태그에 대응하는 '해당 페이지의 핵심 근거 요약'을 구체적으로 작성.
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


def build_expand_prompt(topic_, purpose_, hypothesis_, context, current_result, add_paras_, min_chars_):
    system_msg = """
당신은 석사학위 논문을 다수 지도한 전문 학술 에디터입니다.
아래의 기존 초안을 더 전문적이고 더 길게 '확장'합니다. 근거(REF) 밀도를 유지하면서 논리적 연결과 비판적 논의를 강화하세요.
반드시 지정한 JSON 스키마로만 출력하세요.
"""
    user_msg = f"""
주제: {topic_}
목적: {purpose_}
가설: {hypothesis_}

[자료 원문]
{context}

[기존 결과(JSON)]
{json.dumps(current_result, ensure_ascii=False)}

[확장 요구사항]
- interactive_draft만 '더 길게' 확장(소절별 문단 추가).
- 각 소절(예: 1.1, 1.2...)마다 문단을 추가로 {add_paras_}개씩 더 작성.
- 새로 추가되는 각 문단은 최소 {min_chars_}자 이상.
- 새 문단마다 최소 1개의 [REF:파일명,p숫자] 포함(가능하면 2개).
- source_map에는 새로 등장한 REF가 있다면 반드시 추가하고, 기존 REF 매핑도 유지/보강.
- 기존 서술과 모순되지 않게 하되, 학술적 연결어/개념 정교화/한계 및 연구공백을 더 명확히.

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


# =========================================================
# 7) Rendering + Reference export helpers
# =========================================================
def render_text_with_ref_popovers(text: str, source_map: Dict[str, str]):
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


def extract_refs_from_draft(draft: Dict[str, str]) -> List[str]:
    all_text = "\n".join(draft.values())
    refs = re.findall(r"\[REF:[^\]]+\]", all_text)
    # preserve order, unique
    seen = set()
    out = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def parse_ref_tag(ref_tag: str) -> Tuple[str, int]:
    # [REF:파일명,p숫자]
    m = re.match(r"\[REF:(.+?),p(\d+)\]", ref_tag)
    if not m:
        return ref_tag, -1
    return m.group(1), int(m.group(2))


def build_reference_table(source_map: Dict[str, str], used_refs: List[str]) -> pd.DataFrame:
    rows = []
    for ref in used_refs:
        file_name, page = parse_ref_tag(ref)
        rows.append(
            {
                "ref_tag": ref,
                "file": file_name,
                "page": page,
                "evidence_summary": source_map.get(ref, ""),
            }
        )
    return pd.DataFrame(rows)


def export_markdown_with_footnotes(draft: Dict[str, str], source_map: Dict[str, str]) -> str:
    """
    Convert [REF:...] into footnote markers [^n] and append footnotes list.
    """
    used_refs = extract_refs_from_draft(draft)
    ref_to_idx = {r: i + 1 for i, r in enumerate(used_refs)}

    md_parts = ["# Report Mate Draft\n"]
    for sec, txt in draft.items():
        md_parts.append(f"\n## {sec}\n")
        # Replace refs with footnotes
        def _repl(m):
            r = m.group(0)
            idx = ref_to_idx.get(r, None)
            return f"[^{idx}]" if idx is not None else r

        converted = re.sub(r"\[REF:[^\]]+\]", _repl, txt)
        md_parts.append(converted.strip() + "\n")

    md_parts.append("\n---\n\n## References (Footnotes)\n")
    for r in used_refs:
        idx = ref_to_idx[r]
        # Keep the original tag in the footnote for traceability
        md_parts.append(f"[^{idx}]: {r} — {source_map.get(r, '상세 근거 없음')}\n")
    return "\n".join(md_parts).strip()


# =========================================================
# 8) Generate / Expand Actions (with RAG)
# =========================================================
colA, colB = st.columns(2)
with colA:
    generate_clicked = st.button("🚀 분석 및 상세 초안 생성", type="primary")
with colB:
    expand_clicked = st.button("➕ 초안 확장(추가 작성)", type="secondary", disabled=(st.session_state["result"] is None))

def compute_files_fingerprint(files) -> str:
    # cheap fingerprint: file names + sizes
    if not files:
        return ""
    parts = []
    for f in files:
        try:
            parts.append(f"{f.name}:{len(f.getvalue())}")
        except Exception:
            parts.append(f"{f.name}:?")
    return "|".join(parts)

if generate_clicked:
    if not user_api_key or not uploaded_files or not topic:
        st.warning("API 키, 주제, 그리고 파일을 모두 입력했는지 확인해주세요.")
    else:
        with st.spinner("선행연구를 분석하며 상세 설계안/초안을 작성 중입니다..."):
            try:
                st.session_state["expansion_level"] = 0

                # ---- Build context (RAG or fallback) ----
                context = ""
                files_fp = compute_files_fingerprint(uploaded_files)
                rag_fp = f"{files_fp}::{rag_max_pages_each}::{rag_chunk_chars}::{rag_overlap_chars}::{embedding_model}"

                if use_rag:
                    # Build RAG store if not ready or fingerprint changed
                    if (not st.session_state.get("rag_ready", False)) or (st.session_state.get("rag_fingerprint", "") != rag_fp):
                        st.session_state["rag_ready"] = False
                        st.session_state["rag_fingerprint"] = rag_fp
                        build_rag_store(
                            files=uploaded_files,
                            api_key=user_api_key,
                            embed_model=embedding_model,
                            max_pages_each=rag_max_pages_each,
                            chunk_size=rag_chunk_chars,
                            overlap=rag_overlap_chars,
                        )
                    context = build_section_context_rag(
                        topic_=topic,
                        purpose_=purpose,
                        hypothesis_=hypothesis,
                        api_key=user_api_key,
                        embed_model=embedding_model,
                        top_k=rag_top_k,
                    )

                    if not context.strip():
                        # fallback if retrieval context is empty for some reason
                        context = get_combined_text_with_meta(uploaded_files, max_pages_each=rag_max_pages_each)
                else:
                    context = get_combined_text_with_meta(uploaded_files, max_pages_each=rag_max_pages_each)

                st.session_state["context_text"] = context
                st.session_state["last_inputs"] = {
                    "topic": topic,
                    "purpose": purpose,
                    "hypothesis": hypothesis,
                    "base_paras": base_paras,
                    "min_chars_per_para": min_chars_per_para,
                    "model_name": model_name,
                    "use_rag": use_rag,
                    "rag_top_k": rag_top_k,
                    "rag_max_pages_each": rag_max_pages_each,
                    "rag_chunk_chars": rag_chunk_chars,
                    "rag_overlap_chars": rag_overlap_chars,
                    "embedding_model": embedding_model,
                }

                system_msg, user_msg = build_initial_prompt(
                    topic_=topic,
                    purpose_=purpose,
                    hypothesis_=hypothesis,
                    context=context,
                    base_paras_=base_paras,
                    min_chars_=min_chars_per_para,
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

                # Rebuild context (keeps RAG settings consistent with last run)
                context = st.session_state.get("context_text", "")
                # If user changed PDFs after initial generation, context might be stale;
                # but we keep last run consistency intentionally.
                # If you'd rather rebuild context on expand, set context again here.

                system_msg, user_msg = build_expand_prompt(
                    topic_=topic0,
                    purpose_=purpose0,
                    hypothesis_=hypothesis0,
                    context=context,
                    current_result=st.session_state["result"],
                    add_paras_=expand_additional,
                    min_chars_=min_chars_per_para,
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


# =========================================================
# 9) Result Display + Reference Export
# =========================================================
if st.session_state["result"]:
    res = st.session_state["result"]
    st.markdown("---")

    if st.session_state.get("expansion_level", 0) > 0:
        st.info(f"초안이 {st.session_state['expansion_level']}회 확장되었습니다.")

    tab1, tab2, tab3 = st.tabs(["📋 상세 설계 개요(간결)", "✍️ 각주 포함 초안(전문적)", "📤 Reference Export"])

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
            render_text_with_ref_popovers(text, source_map)
            st.divider()

    with tab3:
        source_map = res.get("source_map", {})
        draft = res.get("interactive_draft", {})

        used_refs = extract_refs_from_draft(draft)
        ref_df = build_reference_table(source_map, used_refs)

        st.markdown("#### 🔗 사용된 REF 목록")
        st.caption("초안에 실제로 등장한 [REF:...]만 추출해 내보냅니다.")
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

        # --- Export: CSV / JSON / Markdown with footnotes ---
        csv_bytes = ref_df.to_csv(index=False).encode("utf-8-sig")
        json_bytes = json.dumps(
            {"used_refs": used_refs, "source_map_used": {r: source_map.get(r, "") for r in used_refs}},
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        md_text = export_markdown_with_footnotes(draft, source_map).encode("utf-8")

        colx, coly, colz = st.columns(3)
        with colx:
            st.download_button(
                "⬇️ REF Table (CSV)",
                data=csv_bytes,
                file_name="references_table.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with coly:
            st.download_button(
                "⬇️ REF Map (JSON)",
                data=json_bytes,
                file_name="references_map.json",
                mime="application/json",
                use_container_width=True,
            )
        with colz:
            st.download_button(
                "⬇️ Draft + Footnotes (MD)",
                data=md_text,
                file_name="draft_with_footnotes.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("#### ✅ Export 팁")
        st.markdown(
            """
- **CSV**: 논문/페이지별 근거를 정리해서 검토·보완할 때 유용
- **JSON**: 다른 시스템(Word/Notion 변환, DB 저장, 후처리 파이프라인)에 연결하기 좋음
- **MD**: [REF]를 각주로 바꿔서 문서 형태로 바로 활용 가능
"""
        )

else:
    st.markdown(
        "<br><br><p style='text-align: center; color: #BFBFC3;'>선행연구를 업로드하고 분석을 시작하여 논문 초안을 확인하세요.</p>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<p style="text-align: center; color: #D2D2D7; font-size: 12px; margin-top: 50px;">© 2026 Report Mate. Designed for Academic Excellence.</p>',
    unsafe_allow_html=True,
)
