# 📌 Report mate (리포트 메이트)

논문 자료 분석 및 학술적 개요/초안 작성을 돕는 연구용 창작 AI 프로토타입입니다.  
Streamlit 기반 UI에서 PDF 업로드, 연구 주제 입력, 개요 및 초안 생성을 지원합니다.

---

## ✨ Features

- 📄 PDF 업로드 및 화면 좌측 뷰어 표시
- 🧾 연구 주제 / 목적 / 가설 입력 UI
- 🧠 AI 개요(논리 구조) 및 초안 출력 영역 (우측)
- 📚 사이드바 참고문헌 및 스타일 설정 UI
- 📊 진행률 표시(progress)

---

## 🖥️ UI Layout

- **상단 중앙:** 앱 타이틀 (Report mate)
- **사이드바:** 참고문헌 리스트 / 인용 스타일 / 문체 설정
- **상단 입력:** 주제, 연구 목적, 가설 입력
- **중앙 업로드:** PDF 파일 업로드
- **하단:** 진행률 표시
- **2분할 화면**
  - 왼쪽: 업로드된 PDF 원문
  - 오른쪽: AI가 생성한 개요 및 초안

---

## 📁 Project Structure

```txt
report-mate/
  app.py
  requirements.txt
  README.md
  📌 Requirements
```

Python 3.9 이상 권장

Streamlit

## 🛠️ Tech Stack

Frontend/UI: Streamlit

PDF Viewer: iframe 기반 PDF 렌더링

Future Plan: OpenAI API 기반 논문 개요/초안 자동 생성

## 🚀 Future Improvements (Planned)

OpenAI API 연동을 통한 실제 개요/초안 생성

PDF 텍스트 추출 및 근거 기반 인용 자동화(RAG)

참고문헌 스타일 자동 변환(APA/MLA/IEEE 등)

결과 export 기능 (.docx / .pdf / .md)

GitHub 기반 버전 관리 및 협업 기능 확장
