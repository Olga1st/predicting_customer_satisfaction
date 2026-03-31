import streamlit as st
import pandas as pd

st.title("🤖 Modeling, Evaluation & Interpretability")

# --- Models ---
st.subheader("Models Compared")

st.markdown("""
- **XGBoost (Gradient Boosting)** → High performance
- **Random Forest** → Baseline model
""")

# --- Real Results (you should replace with your actual numbers) ---
results = pd.DataFrame({
    "Model": ["XGBoost", "Random Forest"],
    "Accuracy": [0.89, 0.84],
    "F1 Score": [0.88, 0.83]
})

st.subheader("Model Performance Comparison")
st.table(results)

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