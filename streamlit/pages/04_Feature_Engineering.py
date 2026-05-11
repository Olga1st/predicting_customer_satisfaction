import streamlit as st
#from src.analysis.reporting import analyze_feature_impact
#from src.analysis.visualization import plot_feature_impact

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

st.header("🔍 Structured Features")

st.markdown("""
Additional features:

- review_length
- verified
- has_negation
- sentiment

👉 These improve model performance
""")

st.header("💡 Insights")

st.markdown("""
- TF-IDF lexical signal is strong but limited by semantic ambiguity
- Embeddings help with multilingual data and semantic nuances
- Structured features provide stability signals
- Combining text + metadata improves performance
- Best approach depends on business trade-off:
  - speed vs accuracy
""")

