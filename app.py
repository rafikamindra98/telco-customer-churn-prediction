from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")
ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "churn_artifacts" / "churn_model.joblib")


bundle = load_bundle()
st.title("Telco Customer Churn Predictor")
st.write("Prototipe portofolio untuk menandai pelanggan yang perlu ditinjau dalam program retensi.")
st.caption("Skor prediksi belum diuji kalibrasinya dan tidak menyatakan kepastian pelanggan akan berhenti.")

with st.expander("Model dan hasil evaluasi"):
    st.write("XGBoost · ROC-AUC 0,8478 · Average precision 0,6693")
    st.write("Pada data uji: recall 79,14%, precision 51,21%, 296 churn terdeteksi dan 282 alarm keliru.")
    st.write("Threshold dipilih dari prediksi out-of-fold data latih dengan target recall minimal 80%. Target ini merupakan asumsi demonstrasi.")
    st.caption("Dataset contoh: Kaggle Telco Customer Churn. Tidak ada horizon prediksi masa depan yang telah divalidasi.")

labels = {
    "tenure": "Masa berlangganan (bulan)",
    "MonthlyCharges": "Tagihan bulanan (satuan dataset)",
    "TotalCharges": "Total tagihan (satuan dataset)",
    "SeniorCitizen": "Pelanggan senior (0 = tidak, 1 = ya)",
    "gender": "Gender", "Partner": "Memiliki pasangan", "Dependents": "Memiliki tanggungan",
    "PhoneService": "Layanan telepon", "MultipleLines": "Beberapa saluran telepon",
    "InternetService": "Layanan internet", "OnlineSecurity": "Keamanan online",
    "OnlineBackup": "Backup online", "DeviceProtection": "Perlindungan perangkat",
    "TechSupport": "Dukungan teknis", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming film", "Contract": "Jenis kontrak",
    "PaperlessBilling": "Tagihan tanpa kertas", "PaymentMethod": "Metode pembayaran"
}

with st.form("customer_form"):
    columns = st.columns(3)
    inputs = {}
    for index, feature in enumerate(bundle["feature_names"]):
        with columns[index % 3]:
            label = labels.get(feature, feature)
            if feature in bundle["categorical_features"]:
                inputs[feature] = st.selectbox(label, bundle["category_options"][feature], key=feature)
            elif feature == "tenure":
                inputs[feature] = st.number_input(label, min_value=0, max_value=72, value=12, step=1)
            else:
                inputs[feature] = st.number_input(label, min_value=0.0, value=65.0 if feature == "MonthlyCharges" else 780.0, step=0.5)
    total_missing = st.checkbox("Total tagihan tidak tersedia (gunakan imputasi pipeline)")
    submitted = st.form_submit_button("Hitung skor churn", type="primary")

if submitted:
    problems = []
    if inputs["PhoneService"] == "No" and inputs["MultipleLines"] != "No phone service":
        problems.append("Tanpa layanan telepon, MultipleLines harus 'No phone service'.")
    if inputs["PhoneService"] == "Yes" and inputs["MultipleLines"] == "No phone service":
        problems.append("Dengan layanan telepon, MultipleLines harus 'Yes' atau 'No'.")
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for feature in addons:
        if inputs["InternetService"] == "No" and inputs[feature] != "No internet service":
            problems.append(f"Tanpa internet, {feature} harus 'No internet service'.")
        elif inputs["InternetService"] != "No" and inputs[feature] == "No internet service":
            problems.append(f"Dengan internet aktif, {feature} harus 'Yes' atau 'No'.")
    if problems:
        for problem in problems:
            st.error(problem)
        st.stop()
    if total_missing:
        inputs["TotalCharges"] = np.nan
    customer = pd.DataFrame([inputs], columns=bundle["feature_names"])
    pipeline = bundle["pipeline"]
    positive_index = list(pipeline.classes_).index(1)
    score = float(pipeline.predict_proba(customer)[0, positive_index])
    threshold = float(bundle["threshold"])
    st.metric("Skor prediksi churn (0–1)", f"{score:.4f}")
    st.caption(f"Threshold tetap: {threshold:.6f}")
    if score >= threshold:
        st.warning("Ditandai untuk tinjauan retensi: skor berada pada atau di atas threshold.")
    else:
        st.info("Tidak ditandai oleh model: skor berada di bawah threshold. Churn tetap mungkin terjadi.")
    st.dataframe(customer, hide_index=True, use_container_width=True)
