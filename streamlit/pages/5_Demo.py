import streamlit as st
import pandas as pd
import joblib

st.title("🔍 Live Demo")

@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")

model = load_model()
#todo: eventuell mehr eingabe möglichkeiten hinzufügen, falls die Struckturellen Variablen grossen einfluss haben.
review = st.text_area("Review eingeben")
verified = st.checkbox("Verifiziert")

if st.button("Analysieren"):
    df = pd.DataFrame([{
        "review_text": review,
        "supplier_response": None,
        "verified": int(verified),
        "review_text_clean": review
    }])

    pred = model.predict(df)[0]

    if pred == 1:
        st.success("😊 Positiv")
    else:
        st.error("😡 Negativ")