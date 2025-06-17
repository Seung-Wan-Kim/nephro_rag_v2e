# app/app_rag.py

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os

# 🔸 벡터 저장소 경로들
VECTOR_PATHS = {
    "AKI": "vector_store_aki_ko",
    "CKD": "vector_store_ckd_ko",
    "Nephrotic Syndrome": "vector_store_ns_ko",
    "Glomerulonephritis": "vector_store_gn_ko",
    "Electrolyte Disorders": "vector_store_electrolyte_ko",
}

# 🔸 중요 검사 항목 리스트
TEST_ITEMS = [
    "Creatinine", "BUN", "eGFR", "Albumin", "Na", "K", "Cl", "HCO3", "Ca", "Phosphorus"
]

st.set_page_config(page_title="Nephrology RAG", layout="wide")
st.title("🧪 신장내과 질병 예측 RAG 시스템")

st.markdown("#### 1. 혈액검사 수치 입력")
col1, col2 = st.columns(2)
user_inputs = {}

with col1:
    for item in TEST_ITEMS[:5]:
        user_inputs[item] = st.text_input(f"{item} 수치", placeholder="예: 1.2")

with col2:
    for item in TEST_ITEMS[5:]:
        user_inputs[item] = st.text_input(f"{item} 수치", placeholder="예: 3.8")

st.markdown("---")
question = st.text_area("#### 2. 자연어 질문 입력", placeholder="예: 이 수치로 볼 때 어떤 질병 가능성이 있나요?")

if st.button("🔍 질의 응답 시작"):
    # ✅ 문서 임베딩 모델 로드
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

    # ✅ 모든 질병군에서 문서 검색
    all_docs = []
    for disease, path in VECTOR_PATHS.items():
        if os.path.exists(path):
            db = FAISS.load_local(path, embedding_model)
            docs = db.similarity_search(question, k=2)
            all_docs.extend(docs)

    if not all_docs:
        st.warning("❌ 관련된 문서를 찾을 수 없습니다.")
    else:
        # ✅ 응답 생성
        llm = OpenAI(temperature=0.3)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=all_docs, question=question)

        st.markdown("### 🧾 응답 결과")
        st.success(response)

        # ✅ 문서 출처 표시
        st.markdown("### 📄 참조한 문서 (요약)")
        for i, doc in enumerate(all_docs):
            st.markdown(f"**{i+1}.** {doc.page_content[:300]}...")

