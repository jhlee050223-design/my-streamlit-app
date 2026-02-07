import os
import re
import json
import uuid
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import faiss
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
load_dotenv()

APP_TITLE = "리포트 메이트 (Report mate)"
DATA_DIR = Path("data")
PROJECTS_JSON = DATA_DIR / "projects.json"
PROJECTS_DIR = DATA_DIR / "projects"

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"  # 비용/속도 밸런스(필요시 변경)

# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PROJECTS_JSON.exists():
        PROJECTS_JSON.write_text(json.dumps({"projects": []}, ensure_ascii=False, indent=2), encoding="utf-8")

def load_projects() -> Dict[str, Any]:
    ensure_dirs()
    return json.loads(PROJECTS_JSON.read_text(encoding="utf-8"))

def save_projects(obj: Dict[str, Any]):
    PROJECTS_JSON.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def slugify(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9a-zA-Z가-힣_()-]", "", name)
    return name[:60] if len(name) > 60 else name

def token_len(text: str, model: str = CHAT_MODEL) -> int:
    # tiktoken은 모델별 encoding이 다를 수 있어 fallback 처리
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def chunk_text(text: str, max_tokens: int = 900, overlap: int = 120) -> List[str]:
    # 간단한 토큰 기준 chunker
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if token_len(candidate) <= max_tokens:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # 너무 긴 단락은 문장 단위로 쪼갬
            if token_len(p) > max_tokens:
                sentences = re.split(r"(?<=[.!?。？！])\s+", p)
                tmp = ""
                for s in sentences:
                    cand = (tmp + " " + s).strip() if tmp else s
                    if token_len(cand) <= max_tokens:
                        tmp = cand
                    else:
                        if tmp:
                            chunks.append(tmp)
                        tmp = s
                if tmp:
                    chunks.append(tmp)
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)

    # overlap(문자 기준 간단) — MVP용
    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = out[-1]
            tail = prev[-overlap*4:]  # 대략치
            out.append((tail + "\n\n" + chunks[i]).strip())
        return out
    return chunks

def read_pdf_text(pdf_path: Path) -> Tuple[str, int]:
    doc = fitz.open(pdf_path)
    pages = doc.page_count
    texts = []
    for i in range(pages):
        page = doc.load_page(i)
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts).strip(), pages

def pdf_to_base64(pdf_path: Path) -> str:
    b = pdf_path.read_bytes()
    return base64.b64encode(b).decode("utf-8")

# ----------------------------
# OpenAI wrapper
# ----------------------------
@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_hint: str
    text: str

class OA:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 없습니다. .env에 설정하세요.")
        self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # OpenAI embeddings: list[str] -> (n, d)
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return vecs

    def chat_json(self, system: str, user: str, schema_hint: str = "") -> Dict[str, Any]:
        # JSON만 반환하도록 강제(프롬프트 방식)
        prompt = f"""{user}

반드시 JSON만 출력해. 다른 텍스트는 출력하지 마.
{schema_hint}
"""
        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        # 안전한 JSON 파싱(앞뒤 잡텍스트 제거 시도)
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            raise ValueError("모델 출력에서 JSON을 찾지 못했습니다:\n" + content)
        return json.loads(m.group(0))

    def chat_text(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()

# ----------------------------
# Vector store (FAISS)
# ----------------------------
class VectorStore:
    def __init__(self, dim: int, index_path: Path, meta_path: Path):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatIP(dim)  # cosine 유사도는 정규화로 처리
        self.meta: List[ChunkRecord] = []

        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            meta_raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.meta = [ChunkRecord(**r) for r in meta_raw]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    def add(self, vecs: np.ndarray, records: List[ChunkRecord]):
        vecs = self._normalize(vecs.astype(np.float32))
        self.index.add(vecs)
        self.meta.extend(records)

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        raw = [r.__dict__ for r in self.meta]
        self.meta_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, qvec: np.ndarray, k: int = 8) -> List[ChunkRecord]:
        qvec = qvec.astype(np.float32)
        qvec = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
        D, I = self.index.search(qvec, k)
        hits = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.meta):
                continue
            hits.append(self.meta[idx])
        return hits

# ----------------------------
# App state helpers
# ----------------------------
def get_project_dir(project_id: str) -> Path:
    d = PROJECTS_DIR / project_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "docs").mkdir(exist_ok=True)
    return d

def project_files(project_id: str) -> List[Path]:
    d = get_project_dir(project_id) / "docs"
    return sorted(d.glob("*.pdf"))

def get_store_paths(project_id: str) -> Tuple[Path, Path]:
    d = get_project_dir(project_id)
    return d / "faiss.index", d / "chunks.json"

# ----------------------------
# Core features
# ----------------------------
SYS_RESEARCH_ASSISTANT = (
    "당신은 학술 논문 교정 전문가이자 연구 방법론 컨설턴트다. "
    "사용자가 제시한 연구 주제에 맞춰 근거 중심으로 요약하고, 논리 구조를 엄밀하게 제안한다. "
    "확실한 근거가 부족하면 '추가 자료 필요'를 명시한다. "
    "인용은 (DocID/DocName, ChunkID)로 출처를 남긴다."
)

def build_kp_extraction_prompt(topic: str, retrieved: List[ChunkRecord]) -> str:
    ctx = "\n\n".join(
        [f"[{r.doc_name} | chunk={r.chunk_id}] {r.text[:1800]}" for r in retrieved]
    )
    return f"""
연구 주제: {topic}

아래는 참고문헌에서 검색된 근거 발췌다.
각 발췌를 읽고, 연구 주제와 관련된 핵심 주장/결과/데이터를 추출해 카테고리별로 분류해줘.

출력 JSON 스키마:
{{
  "categories": [
    {{
      "name": "카테고리명(예: 이론/방법/결과/한계/변수/데이터셋 등)",
      "items": [
        {{
          "claim": "핵심 주장/결과 요약(1~2문장)",
          "evidence": "근거가 되는 문장/수치 요약",
          "source": {{
            "doc_name": "...",
            "chunk_id": "..."
          }},
          "notes": "추가 해석/주의점(없으면 빈 문자열)"
        }}
      ]
    }}
  ]
}}

근거 발췌:
{ctx}
"""

def build_outline_prompt(topic: str, purpose: str, hypothesis: str, extracted_json: Dict[str, Any]) -> str:
    return f"""
연구 주제: {topic}
연구 목적: {purpose}
가설: {hypothesis}

아래는 참고자료에서 뽑은 핵심 근거/분류 결과야(JSON).
이 근거들을 바탕으로 '서론-이론적 배경-연구 방법-결과-결론' 구조의 세부 개요를 작성해줘.
각 섹션마다:
- 핵심 논지 bullet
- 섹션에 들어갈 대표 문장(학술적 문체 1~2개)
- 해당 문장에 연결되는 출처(doc_name, chunk_id)
- 근거 부족이면 '추가 자료 필요' 표시

분량은 너무 길지 않게(섹션당 6~10 bullet).

근거 JSON:
{json.dumps(extracted_json, ensure_ascii=False, indent=2)}
"""

def build_rewrite_prompt(memo: str, citation_style: str, retrieved: List[ChunkRecord]) -> str:
    ctx = "\n\n".join([f"[{r.doc_name} | chunk={r.chunk_id}] {r.text[:1600]}" for r in retrieved])
    return f"""
사용자 메모(평어/구어체 가능):
{memo}

요구사항:
1) 메모를 학술 논문 문체로 교정(전문 용어/논리 연결 강화)
2) 메모 속 주장 중, 아래 근거 발췌로 뒷받침 가능한 부분은 인용을 붙여줘
3) 각주는 {citation_style} 스타일로, 문장 끝에 [^n] 형태로 달고, 하단에 각주 목록을 만들어줘
4) 근거가 없으면 '추가 자료 필요'라고 표시

근거 발췌:
{ctx}

출력 형식:
- 교정된 본문(마크다운)
- 각주 목록(마크다운, [^1]: ...)

주의: 실제 서지정보(저자/연도/제목)를 자동으로 확정할 수 없으면 doc_name 기반으로 임시 표기하고,
사용자가 DOI/서지정보를 채우도록 안내해.
"""

# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    ensure_dirs()
    projects_obj = load_projects()

    # Sidebar: global settings
    st.sidebar.header("설정")
    citation_style = st.sidebar.selectbox(
        "각주/인용 스타일",
        ["APA (임시)", "MLA (임시)", "Chicago (임시)", "IEEE (임시)"],
        index=0
    )
    top_k = st.sidebar.slider("검색 근거 개수(K)", 3, 15, 8)

    st.sidebar.divider()
    st.sidebar.caption("참고: MVP에서는 PDF의 서지정보 자동 추출이 제한적이라 doc_name 기반 임시 인용을 사용합니다.")

    # Main: project controls (상단: 새프로젝트 만들기, 주제 입력창) :contentReference[oaicite:6]{index=6}
    colA, colB, colC = st.columns([1.4, 1.2, 1.4], gap="large")

    with colA:
        st.subheader("프로젝트")
        existing = projects_obj["projects"]
        proj_names = ["(새 프로젝트)"] + [p["name"] for p in existing]
        choice = st.selectbox("선택", proj_names, index=0)

        new_name = ""
        if choice == "(새 프로젝트)":
            new_name = st.text_input("새 프로젝트 이름", value="")
            if st.button("새프로젝트 만들기"):
                if not new_name.strip():
                    st.warning("프로젝트 이름을 입력해줘.")
                else:
                    project_id = str(uuid.uuid4())[:8]
                    p = {"id": project_id, "name": new_name.strip(), "topic": "", "created_at": pd.Timestamp.now().isoformat()}
                    projects_obj["projects"].insert(0, p)
                    save_projects(projects_obj)
                    st.success("프로젝트 생성 완료! 왼쪽에서 선택해줘.")
                    st.rerun()

    # Determine active project
    active_project = None
    if choice != "(새 프로젝트)":
        active_project = next((p for p in existing if p["name"] == choice), None)

    with colB:
        st.subheader("연구 주제")
        if active_project:
            topic = st.text_input("주제 입력", value=active_project.get("topic", ""))
            if st.button("주제 저장"):
                active_project["topic"] = topic
                save_projects(projects_obj)
                st.success("저장 완료")
        else:
            st.info("프로젝트를 먼저 선택/생성해줘.")

    with colC:
        st.subheader("자료 업로드 (PDF)")  # 중앙 : 자료 업로드 :contentReference[oaicite:7]{index=7}
        if active_project:
            uploads = st.file_uploader("여러 PDF 업로드", type=["pdf"], accept_multiple_files=True)
            if uploads:
                docs_dir = get_project_dir(active_project["id"]) / "docs"
                saved = 0
                for uf in uploads:
                    out = docs_dir / f"{slugify(Path(uf.name).stem)}.pdf"
                    out.write_bytes(uf.read())
                    saved += 1
                st.success(f"{saved}개 저장 완료")
        else:
            st.info("프로젝트를 먼저 선택/생성해줘.")

    st.divider()

    if not active_project:
        st.stop()

    # Main area: bottom list + progress (하단 : 논문 리스트/진행률) :contentReference[oaicite:8]{index=8}
    docs = project_files(active_project["id"])
    left, right = st.columns([1.05, 1.25], gap="large")

    with left:
        st.subheader("논문 리스트 / 원문 보기")
        if not docs:
            st.warning("PDF를 업로드해줘.")
        else:
            doc_labels = [d.name for d in docs]
            selected = st.selectbox("원문 선택", doc_labels, index=0)
            sel_path = next(d for d in docs if d.name == selected)

            # PDF preview (간단 iframe)
            b64 = pdf_to_base64(sel_path)
            pdf_display = f"""
            <iframe src="data:application/pdf;base64,{b64}" width="100%" height="720" type="application/pdf"></iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)

            st.download_button("선택 PDF 다운로드", data=sel_path.read_bytes(), file_name=sel_path.name)

    with right:
        st.subheader("AI 작업 공간")
        topic = active_project.get("topic", "").strip()
        if not topic:
            st.warning("연구 주제를 먼저 저장해줘.")
            st.stop()

        # Load/Open store
        oa = OA()

        # Initialize store (need dim)
        # dim은 첫 임베딩 결과로 확정(없으면 더미로 시작했다가 생성)
        index_path, meta_path = get_store_paths(active_project["id"])
        dim = None
        if index_path.exists() and meta_path.exists():
            # 임시로 meta의 임베딩 dim을 알 수 없어서, 첫 쿼리 임베딩으로 dim 맞춰 재로딩 방식 대신
            # Index를 그대로 읽고 dim은 index.d로 가져옴
            idx = faiss.read_index(str(index_path))
            dim = idx.d
        else:
            # 기본 dim: text-embedding-3-large는 보통 3072지만, 확실히 하려면 첫 embed로 확인
            dim = 3072

        store = VectorStore(dim=dim, index_path=index_path, meta_path=meta_path)

        # Progress indicators
        total_docs = len(docs)
        indexed_chunks = len(store.meta)
        st.caption(f"업로드 논문: {total_docs}개 | 인덱싱된 청크: {indexed_chunks}개")

        # Controls
        tab1, tab2, tab3 = st.tabs(["기능1: 핵심내용/데이터 추출", "기능2: 세부 개요 작성", "기능3: 메모 교정 + 각주"])

        # ---- Feature 1
        with tab1:
            st.write("업로드한 논문에서 주제 관련 핵심 내용/데이터를 추출하고 카테고리별로 분류합니다. :contentReference[oaicite:9]{index=9}")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("1) PDF 인덱싱(처음 1회 권장)"):
                    if not docs:
                        st.warning("PDF가 없습니다.")
                    else:
                        prog = st.progress(0, text="PDF 텍스트 추출/청크/임베딩 중...")
                        all_records: List[ChunkRecord] = []
                        all_texts: List[str] = []

                        for i, pdf in enumerate(docs):
                            text, pages = read_pdf_text(pdf)
                            chunks = chunk_text(text)
                            doc_id = pdf.stem
                            for ci, ch in enumerate(chunks):
                                chunk_id = f"{doc_id}_c{ci:04d}"
                                rec = ChunkRecord(
                                    chunk_id=chunk_id,
                                    doc_id=doc_id,
                                    doc_name=pdf.name,
                                    page_hint=f"{pages} pages",
                                    text=ch
                                )
                                all_records.append(rec)
                                all_texts.append(ch)

                            prog.progress((i + 1) / max(1, len(docs)), text=f"처리중: {pdf.name}")

                        # batch embed
                        if all_texts:
                            # 너무 길면 배치로 나눔
                            B = 64
                            vec_list = []
                            for j in range(0, len(all_texts), B):
                                vec_list.append(oa.embed_texts(all_texts[j:j+B]))
                            vecs = np.vstack(vec_list)
                            store.add(vecs, all_records)
                            store.save()
                            st.success(f"인덱싱 완료: {len(all_records)} chunks 추가")
                        else:
                            st.warning("추출된 텍스트가 없습니다.")
                        prog.empty()

            with col2:
                query = st.text_input("빠른 검색(주제/키워드)", value=topic)
                if st.button("2) 주제 기반 핵심내용 추출/분류"):
                    if len(store.meta) == 0:
                        st.warning("먼저 'PDF 인덱싱'을 수행해줘.")
                    else:
                        qvec = oa.embed_texts([query])
                        hits = store.search(qvec, k=top_k)

                        schema = "JSON only. categories: list of {name, items:[{claim,evidence,source:{doc_name,chunk_id},notes}]}"
                        payload = oa.chat_json(
                            system=SYS_RESEARCH_ASSISTANT,
                            user=build_kp_extraction_prompt(topic=topic, retrieved=hits),
                            schema_hint=schema
                        )
                        st.session_state["extracted"] = payload
                        st.success("추출/분류 완료")
                        st.json(payload)

        # ---- Feature 2
        with tab2:
            st.write("연구 목적/가설을 입력하면 '서론-이론적 배경-연구 방법-결과-결론' 세부 개요를 생성합니다. :contentReference[oaicite:10]{index=10}")
            purpose = st.text_area("연구 목적", height=80, placeholder="예) OO 이론을 바탕으로 XX가 YY에 미치는 영향을 검증")
            hypothesis = st.text_area("가설", height=80, placeholder="예) H1: X가 증가할수록 Y는 증가한다.")
            if st.button("세부 개요 생성"):
                extracted = st.session_state.get("extracted")
                if not extracted:
                    st.warning("먼저 기능1에서 핵심내용 추출/분류를 수행해줘.")
                elif not purpose.strip() or not hypothesis.strip():
                    st.warning("연구 목적과 가설을 입력해줘.")
                else:
                    outline_md = oa.chat_text(
                        system=SYS_RESEARCH_ASSISTANT,
                        user=build_outline_prompt(topic, purpose, hypothesis, extracted)
                    )
                    st.session_state["outline"] = outline_md
                    st.success("개요 생성 완료")
                    st.markdown(outline_md)

        # ---- Feature 3
        with tab3:
            st.write("평어 메모를 학술 문체로 교정하고, 근거가 있는 주장에는 각주를 자동 생성합니다. :contentReference[oaicite:11]{index=11}")
            memo = st.text_area("내 메모 붙여넣기", height=180, placeholder="예) 이 논문은 대체로 A가 중요하다고 말한다...")
            if st.button("메모 교정 + 각주 생성"):
                if len(store.meta) == 0:
                    st.warning("먼저 'PDF 인덱싱'을 수행해줘.")
                elif not memo.strip():
                    st.warning("메모를 입력해줘.")
                else:
                    qvec = oa.embed_texts([memo + "\n\n" + topic])
                    hits = store.search(qvec, k=top_k)

                    rewritten = oa.chat_text(
                        system=SYS_RESEARCH_ASSISTANT,
                        user=build_rewrite_prompt(memo, citation_style, hits)
                    )
                    st.session_state["rewrite"] = rewritten
                    st.success("교정/각주 생성 완료")
                    st.markdown(rewritten)

    st.caption("MVP: PDF 텍스트 기반(표/그림 OCR, 정확한 서지정보 자동완성은 다음 단계에서 확장 권장).")

if __name__ == "__main__":
    main()