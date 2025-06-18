
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# -------------------- 벡터 DB 경로 자동 선택 함수 --------------------
def get_vector_path_from_question(question):
    keywords = {
        "aki": ["aki", "급성신손상"],
        "ckd": ["ckd", "만성신질환", "만성콩팥병"],
        "ns": ["nephrotic", "신증후군"],
        "gn": ["glomerulonephritis", "사구체신염"],
        "electrolyte": ["electrolyte", "전해질"]
    }
    folder_map = {
        "aki": "vector_store_aki_ko/",
        "ckd": "vector_store_ckd_ko/",
        "ns": "vector_store_ns_ko/",
        "gn": "vector_store_gn_ko/",
        "electrolyte": "vector_store_electrolyte_ko/"
    }
    for folder, keys in keywords.items():
        for key in keys:
            if key.lower() in question.lower():
                return folder_map[folder]
    return folder_map["aki"]

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Nephrology RAG System", layout="wide")
st.title("🧠 신장내과 진단 지원 시스템")

# 수치 입력 칼럼 구성
st.subheader("1. 혈액 검사 수치 입력")
input_labels = [
    "BUN", "Creatinine", "eGFR", "Albumin", "Proteinuria",
    "Hb", "CRP", "Na", "K"
]

cols = st.columns(3)
user_inputs = {}
for i, label in enumerate(input_labels):
    with cols[i % 3]:
        val = st.text_input(f"{label}", key=label)
        user_inputs[label] = float(val) if val.strip() else None

# -------------------- 수치 기반 결과 확인 --------------------
def analyze_inputs(inputs):
    results = []
    suggestions = []

    # AKI
    if inputs["Creatinine"] and inputs["Creatinine"] >= 1.5:
        results.append(("AKI", 80))
        if not inputs["eGFR"]:
            suggestions.append("eGFR")
    # CKD
    if inputs["eGFR"] and inputs["eGFR"] < 60:
        results.append(("CKD", 85))
        if not inputs["Proteinuria"]:
            suggestions.append("Proteinuria")
    # GN
    if inputs["Proteinuria"] and inputs["Proteinuria"] >= 1.0:
        results.append(("Glomerulonephritis", 75))
    # NS
    if inputs["Proteinuria"] and inputs["Proteinuria"] > 3.5:
        if inputs["Albumin"] and inputs["Albumin"] < 3.0:
            results.append(("Nephrotic Syndrome", 90))
        else:
            results.append(("Nephrotic Syndrome", 70))
        if not inputs["Albumin"]:
            suggestions.append("Albumin")

    return results, suggestions

if st.button("수치 기반 결과 확인"):
    results, suggestions = analyze_inputs(user_inputs)
    if results:
        st.markdown("### 🔬 진단 결과")
        for disease, score in results:
            st.write(f"✅ {disease} 가능성: {score}%")
    else:
        st.info("❗ 명확한 진단 기준을 충족하지 않습니다.")

    if suggestions:
        st.warning("📌 더 정확한 분석을 위해 다음 항목 입력을 권장합니다: " + ", ".join(set(suggestions)))

# -------------------- 자연어 기반 질의응답 --------------------
st.subheader("2. 자연어 질문")
query = st.text_area("질문을 입력하세요", placeholder="예: 급성신손상의 정의는?")

if st.button("자연어 기반 질의 결과 확인") and query:
    vector_path = get_vector_path_from_question(query)

    if not (os.path.exists(os.path.join(vector_path, "index.faiss")) and os.path.exists(os.path.join(vector_path, "index.pkl"))):
        st.error(f"해당 질병군에 대한 벡터 데이터가 존재하지 않습니다: {vector_path}")
    else:
        embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
        try:
            db = FAISS.load_local(vector_path, embedding_model, allow_dangerous_deserialization=True)
        except ValueError as e:
            st.error(f"FAISS 로딩 오류: {str(e)}")
            st.stop()

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo"),
            chain_type="stuff",
            retriever=db.as_retriever()
        )
        with st.spinner("답변 생성 중..."):
            result = qa.run(query)
        st.markdown("#### 📘 답변")
        st.write(result)

st.markdown("---")
st.markdown("📁 *본 시스템은 AKI, CKD, NS, GN 질환군을 기반으로 수치 + 자연어 질의를 지원합니다.*")
