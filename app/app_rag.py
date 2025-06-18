import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os

# 벡터 저장소 경로 매핑
vector_paths = {
    "AKI": "vector_store_aki_ko",
    "CKD": "vector_store_ckd_ko",
    "NS": "vector_store_ns_ko",
    "GN": "vector_store_gn_ko",
    "Electrolyte": "vector_store_electrolyte_ko",
}

# 임베딩 모델 정의
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

# 질병군 자동 추론 (간단 키워드 기반)
def detect_disease_group_from_query(query):
    if "급성" in query or "AKI" in query:
        return "AKI"
    elif "만성" in query or "CKD" in query:
        return "CKD"
    elif "신증후군" in query or "NS" in query:
        return "NS"
    elif "사구체신염" in query or "GN" in query:
        return "GN"
    elif "전해질" in query or "electrolyte" in query:
        return "Electrolyte"
    else:
        return "CKD"

# Streamlit 앱 시작
st.title("💉 신장내과 RAG 진단 지원 시스템")

# 20개 혈액검사 수치 입력 받기
st.subheader("🧪 혈액검사 수치 입력")
input_values = {}
cols = st.columns(5)
test_items = [
    "BUN", "Creatinine", "BUN/Cr ratio", "eGFR", "Na",
    "K", "Cl", "CO2", "Ca", "IP",
    "Hb", "PTH", "Vitamin D", "ALP", "LDH",
    "Lactate", "Albumin", "Total Protein", "Uric Acid", "Glucose"
]

for i, item in enumerate(test_items):
    with cols[i % 5]:
        input_values[item] = st.text_input(f"{item}", key=item)

# 자연어 질의 입력
st.subheader("💬 질의 입력")
query = st.text_area("질문을 입력하세요 (예: 급성 신손상의 정의는?)")

# 버튼 클릭 시 검색 수행
if st.button("🔍 진단 정보 확인"):
    if not query:
        st.warning("질문을 입력해주세요.")
    else:
        # 질병군 자동 추정
        detected_disease = detect_disease_group_from_query(query)
        vector_path = vector_paths.get(detected_disease)

        if not vector_path or not os.path.exists(f"{vector_path}/index.faiss"):
            st.error("선택된 질병군의 벡터스토어를 찾을 수 없습니다.")
        else:
            # 벡터 DB 로드
            db = FAISS.load_local(vector_path, embedding_model)

            # 유사 문서 검색
            docs = db.similarity_search(query)

            # LLM 응답 (OpenAI API 사용)
            llm = OpenAI(temperature=0.3)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=query)

            st.markdown("### 📘 답변")
            st.write(answer)

            st.markdown("📎 관련 문서")
            for i, doc in enumerate(docs):
                st.markdown(f"**[{i+1}]** {doc.page_content[:300]}...")
