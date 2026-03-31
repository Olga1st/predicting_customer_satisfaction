import streamlit as st
import pandas as pd

st.title("🤖 Modeling, Evaluation & Interpretability")

# --- Models ---
#st.subheader("Models Compared")

#st.markdown("""
#- **XGBoost (Gradient Boosting)** → High performance
#- **Random Forest** → Baseline model
#""")

df = pd.read_csv("models/model_comparison.csv")

st.subheader("Model Comparison")
st.dataframe(df)
if st.button("Re-run Model Comparison"):
    from src.models.compare_models import compare_models
    df = compare_models()
    st.dataframe(df)
    
st.subheader("Performance Comparison")

st.bar_chart(df.set_index("model")[["f1_score", "cv_mean_f1"]])

best_model = df.iloc[0]["model"]

st.success(f"🏆 Best Model: {best_model}")

# --- Interpretation ---
st.subheader("Model Interpretation")

st.markdown("""
### Key Insights

- Text features dominate prediction
- Negative keywords strongly influence predictions
- Structured features add stability
""")

# --- Validation ---
st.subheader("Validation Strategy")

st.markdown("""
- Train/Test Split (80/20)
- Cross Validation (5-fold)
- No Data Leakage (Pipeline-based TF-IDF)
- Class balance checked
""")

# --- Why XGBoost Wins ---
st.subheader("Why XGBoost Performs Better")

st.markdown("""
- Captures non-linear relationships
- Handles sparse TF-IDF vectors efficiently
- Better generalization
""")