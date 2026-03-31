import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("🔍 Model Interpretability (SHAP)")

@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")

model_pipeline = load_model()

# Extract parts of pipeline
model = model_pipeline.named_steps["model"]
preprocessor = model_pipeline.named_steps["preprocessing"]

st.subheader("How the Model Makes Decisions")

st.markdown("""
SHAP (SHapley Additive exPlanations) shows how each feature contributes to a prediction.
- Positive values → increase prediction
- Negative values → decrease prediction
""")

# Load sample data
df = pd.read_csv("data/processed/reviews_clean.csv").sample(100)

# Transform data
X_transformed = preprocessor.transform(df)

# Create SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_transformed)

# --- Global Feature Importance ---
st.subheader("Global Feature Importance")

fig1, ax1 = plt.subplots()
shap.plots.bar(shap_values, show=False)
st.pyplot(fig1)

# --- Beeswarm Plot ---
st.subheader("Feature Impact Distribution")

fig2, ax2 = plt.subplots()
shap.plots.beeswarm(shap_values, show=False)
st.pyplot(fig2)


st.subheader("Explain a Single Prediction")

user_input = st.text_area("Enter a review to explain")

if st.button("Explain Prediction"):
    input_df = pd.DataFrame([{
        "review_text": user_input,
        "supplier_response": None,
        "verified": 0,
        "review_text_clean": user_input
    }])

    X_input = preprocessor.transform(input_df)
    shap_values_input = explainer(X_input)

    st.write("### SHAP Explanation")

    fig3, ax3 = plt.subplots()
    shap.plots.waterfall(shap_values_input[0], show=False)
    st.pyplot(fig3)
    
    st.markdown("""
### How to read this:
- Red → pushes toward negative
- Blue → pushes toward positive
- Bigger bars → stronger influence
""")
    
    feature_names = preprocessor.get_feature_names_out()
    shap_values.feature_names = feature_names
    