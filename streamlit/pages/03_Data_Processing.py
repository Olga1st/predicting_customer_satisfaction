import streamlit as st

st.title("🧹 Data Processing")

st.header("⚠️ Data Problems")

st.markdown("""

- Missing texts
- Duplicate reviews
- Invalid ratings
- Very short / meaningless text
- System-generated responses
- mixed languages

👉 These problems degrade model performance.
""")

st.header("🧽 Cleaning Steps")

st.markdown("""
            
- Remove duplicates
- Remove empty reviews
- Remove very short texts
- Remove reply texts
- Validate ratings (1–5)
- Handle missing values if needed
- Translate non-English reviews
- Normalize structured features
""")

st.success("✅ Result: Clean, reliable dataset for modeling")

st.header("🔁 Why This Matters")

st.markdown("""
Without proper cleaning:
- Model learns noise
- Bias increases
- Predictions become unreliable

👉 Cleaning is critical for real-world ML systems
""")