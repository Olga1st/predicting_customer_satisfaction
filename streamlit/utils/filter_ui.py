import streamlit as st
from src.analysis.reporting import get_available_filters


def render_filters():
    filters = get_available_filters()

    model = st.selectbox(
        "Model",
        ["All"] + filters.get("model_name", [])
    )

    feature = st.selectbox(
        "Feature Type",
        ["All"] + filters.get("feature_type", [])
    )

    sampling = st.selectbox(
        "Sampling",
        ["All"] + filters.get("sampling", [])
    )

    return {
        "model_name": None if model == "All" else model,
        "feature_type": None if feature == "All" else feature,
        "sampling": None if sampling == "All" else sampling,
    }