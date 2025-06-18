""import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# -------------------- 설정 --------------------
# 벡터 DB 경로 자동 선택 함수
def get_vector_path_from_question(question):
    keywords = {
        "aki": ["aki", "급성신손상"],
        "ckd": ["ckd", "만성신질환", "만성콩팥병"],
        "ns": ["nephrotic", "신증후군"],
        "gn": ["glomerulonephritis", "사구체신염"],
        "electrolyte": ["electrolyte", "전해질"]
    }
    for folder, keys in keywords.items():
        for key in keys:
            if key.lower() in question.lower():
                return f"vector_store_{folder}/"
    return "vector_store_aki/"  # 기본값

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Nephrology RAG System", layout="wide")
st.title("🧠 신장내과 진단 지원 시스템")

# 수치 입력 칼럼 구성
st.subheader("1. 혈액 검사 수치 입력")
cols = st.columns(4)

input_labels = [
    "BUN", "Creatinine", "B/C ratio", "eGFR", "Na", "K", "Cl", "CO2", "Ca", "IP",
    "Hb", "PTH", "Vitamin D", "ALP", "LDH", "Lactate", "Albumin", "Proteinuria", "CRP", "Glucose"
]

user_inputs = {}
for i, label in enumerate(input_labels):
    with cols[i % 4]:
        user_inputs[label] = st.text_input(f"{label}")

# 결과 확인 버튼 1 (수치 기반 진단용)
if st.button("수치 기반 결과 확인"):
    st.markdown("👉 이 기능은 향후 구현될 예정입니다. 현재는 자연어 질문만 지원됩니다.")

# 자연어 질문
st.subheader("2. 자연어 질문")
query = st.text_area("질문을 입력하세요", placeholder="예: 급성신손상의 정의는?")

# 결과 확인 버튼 2 (RAG)
if st.button("자연어 기반 질의 결과 확인") and query:
    # 벡터 경로 추출
    vector_path = get_vector_path_from_question(query)

    # 벡터 파일 존재 확인
    if not (os.path.exists(os.path.join(vector_path, "index.faiss")) and os.path.exists(os.path.join(vector_path, "index.pkl"))):
        st.error(f"해당 질병군에 대한 벡터 데이터가 존재하지 않습니다: {vector_path}")
    else:
        # 임베딩 모델 불러오기
        embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

        # 벡터 DB 로드
        try:
            db = FAISS.load_local(vector_path, embedding_model, allow_dangerous_deserialization=True)
        except ValueError as e:
            st.error(f"FAISS 로딩 오류: {str(e)}")
            st.stop()

        # QA 체인 생성
        qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.3), chain_type="stuff", retriever=db.as_retriever())

        # 답변 생성
        with st.spinner("답변 생성 중..."):
            result = qa.run(query)
        st.markdown("#### 📘 답변")
        st.write(result)

# 참고
st.markdown("---")
st.markdown("📁 *본 시스템은 5개 주요 신장내과 질환군(AKI, CKD, NS, GN, Electrolyte)의 문서 임베딩 기반 RAG 시스템입니다.*")
