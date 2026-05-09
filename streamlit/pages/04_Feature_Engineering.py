import streamlit as st
from src.analysis.reporting import analyze_feature_impact
from src.analysis.visualization import plot_feature_impact

st.title("🧠 Feature Engineering")

st.header("🔧 Feature Types")

st.markdown("""
We compare two approaches:

### 1. TF-IDF
- Fast
- Interpretable
- Strong baseline

### 2. Embeddings (SentenceTransformers)
- Capture semantics
- Multilingual understanding
- More powerful but heavier
""")

st.header("📊 Feature Impact")

df = analyze_feature_impact()

st.dataframe(df)

if not df.empty:
    fig = plot_feature_impact(df)
    st.pyplot(fig)

st.header("🔍 Structured Features")

st.markdown("""
Additional features:

- review_length
- verified
- has_response

👉 These improve model performance
""")

st.header("💡 Insights")

st.markdown("""
- TF-IDF often performs well
- Embeddings help with multilingual data
- Best approach depends on business trade-off:
  - speed vs accuracy
""")

st.header("💡 Alternative 'Text Representation' Approaches")

st.markdown("""
## Features Used:

### 1. TF-IDF
- captures word importance

### 2. Embeddings
- multilingual semantic understanding

### 3. Structured Features
- review length
- verified
- has_response

## Insight:
👉 Combining text + metadata improves performance
""")