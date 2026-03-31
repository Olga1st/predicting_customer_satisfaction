import streamlit as st

st.title("⚙️ Data Processing & Feature Engineering Pipeline")

st.subheader("Data Pipeline")

st.markdown("""
### End-to-End Pipeline

1. Data Collection (Scraper / Raw Data)
2. Data Cleaning & Normalization
3. Feature Engineering
4. Vectorization (TF-IDF)
5. Model Input Preparation
""")

st.subheader("Technologies Used")

st.markdown("""
- **pandas** → Data manipulation  
- **NLTK** → Text preprocessing  
- **scikit-learn** → TF-IDF, Pipelines  
- **XGBoost** → Gradient boosting model  
- **joblib** → Model persistence  
""")

st.subheader("Feature Engineering")

st.markdown("""
### Text Features
- TF-IDF (n-grams)
- Stopword removal
- Lemmatization

### Structured Features
- Review length
- Verified purchase (binary)
- Supplier response presence
""")

st.subheader("Pipeline Design")

st.markdown("""
- ColumnTransformer combines:
  - Text features (TF-IDF)
  - Numerical features (scaled)
- Ensures:
  - No data leakage
  - Reproducibility
""")