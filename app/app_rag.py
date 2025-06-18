# app/app_rag.py

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# 기본 설정
disease_groups = [
    "aki",
    "ckd",
    "ns",
    "gn",
    "electrolyte"
]

VECTOR_STORE_PATHS = {
    "aki": "vector_store_aki_md_ko",
    "ckd": "vector_store_ckd_md_ko",
    "ns": "vector_store_ns_md_ko",
    "gn": "vector_store_gn_md_ko",
    "electrolyte": "vector_store_ed_md_ko"
}

# 질병군 추론 함수
def detect_disease_group_from_question(question: str) -> str:
    q = question.lower()
    if "aki" in q or "acute kidney injury" in q or "급성 신손상" in q:
        return "aki"
    elif "ckd" in q or "chronic kidney disease" in q or "만성 콩팥병" in q:
        return "ckd"
    elif "nephrotic" in q or "신증후군" in q:
        return "ns"
    elif "glomerulonephritis" in q or "사구체신염" in q:
        return "gn"
    elif "electrolyte" in q or "전해질" in q:
        return "electrolyte"
    else:
        return "aki"  # 기본값

# UI 구성
st.title("🔎 신장내과 문서 기반 질의응답 시스템")
st.markdown("""
질병 관련 혈액검사 수치를 입력하고, 궁금한 내용을 자유롭게 물어보세요.

""")

# 수치 입력창
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    bun = st.text_input("BUN")
with col2:
    creatinine = st.text_input("Creatinine")
with col3:
    egfr = st.text_input("eGFR")
with col4:
    sodium = st.text_input("Na")
with col5:
    potassium = st.text_input("K")

# 자연어 질문
question = st.text_input("🗨️ 질의:", placeholder="예: 급성 신손상의 정의는?")

if question:
    disease_group = detect_disease_group_from_question(question)
    vector_path = VECTOR_STORE_PATHS[disease_group]

    # 모델 로드 및 검색
    with st.spinner(f"📂 {disease_group.upper()} 문서 검색 중..."):
        embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
        db = FAISS.load_local(vector_path, embedding_model)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # QA 체인 구성
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0.3),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa.invoke(question)
        st.markdown("#### 💬 답변:")
        st.write(result['result'])

        st.markdown("#### 📄 참조 문서:")
        for i, doc in enumerate(result['source_documents']):
            st.write(f"{i+1}. {doc.metadata['source']}")
