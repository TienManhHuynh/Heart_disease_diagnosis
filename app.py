"""Demo local: python -m streamlit run app.py (sau khi cài dependencies)."""

import streamlit as st

from heart_disease.config import PROJECT_ROOT
from heart_disease.inference import load_selected_pipeline, predict_patient


@st.cache_resource
def cached_pipeline(signature):
    return load_selected_pipeline()


def main():
    st.set_page_config(page_title="Heart Disease | Project 3.1", layout="wide")
    st.title("Heart Disease Diagnosis - Project 3.1")
    st.caption("Demo học tập trên Cleveland; kết quả không phải chẩn đoán y khoa.")
    artifact = PROJECT_ROOT / "artifacts" / "models" / "selected_pipeline.joblib"
    if not artifact.exists():
        st.info("Chạy .venv/Scripts/python.exe -m heart_disease benchmark để tạo mô hình.")
        st.stop()
    metadata = PROJECT_ROOT / "artifacts" / "metadata" / "selected_model.json"
    try:
        pipeline, information = cached_pipeline(
            (artifact.stat().st_mtime_ns, metadata.stat().st_mtime_ns)
        )
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    st.caption(f"Mô hình: {information['selected_key']} | Chọn bằng CV/validation")
    with st.form("patient"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("age (tuổi)", min_value=1, max_value=120, value=50)
            sex = st.selectbox("sex (0=nữ, 1=nam)", [0, 1])
            cp = st.selectbox("cp (loại đau ngực)", [1, 2, 3, 4])
            trestbps = st.number_input("trestbps (mmHg)", min_value=1, value=120)
            chol = st.number_input("chol (mg/dL)", min_value=1, value=200)
            fbs = st.selectbox("fbs (>120 mg/dL: 1=có)", [0, 1])
            restecg = st.selectbox("restecg", [0, 1, 2])
        with right:
            thalach = st.number_input("thalach (lần/phút)", min_value=1, value=150)
            exang = st.selectbox("exang (đau khi gắng sức: 1=có)", [0, 1])
            oldpeak = st.number_input("oldpeak", min_value=0.0, value=1.0)
            slope = st.selectbox("slope", [1, 2, 3])
            ca = st.selectbox("ca", [0, 1, 2, 3])
            thal = st.selectbox("thal", [3, 6, 7])
        submitted = st.form_submit_button("Dự đoán")
    patient = dict(
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal,
    )
    if submitted:
        try:
            prediction = predict_patient(pipeline, patient)
        except (ValueError, TypeError) as exc:
            st.error(f"Đầu vào không hợp lệ: {exc}")
        else:
            st.subheader("Kết quả của mô hình")
            st.write("Lớp 1 — có bệnh" if prediction["label"] == 1 else "Lớp 0 — không bệnh")
            st.metric("Xác suất ước lượng cho lớp 1", f"{prediction['disease_probability']:.1%}")
            st.caption(
                "Xác suất chưa được hiệu chỉnh/kiểm định lâm sàng; chỉ dùng để minh họa bài học."
            )
            with st.expander("Xem dữ liệu đã nhập"):
                st.json(patient)


if __name__ == "__main__":
    main()
