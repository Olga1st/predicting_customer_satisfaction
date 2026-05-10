# =========================================================
# FILE:
# streamlit/pages/07_Representation_vs_Sampling.py
# =========================================================

import streamlit as st
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Representation vs Sampling",
    layout="wide"
)

# =========================================================
# LOAD DATA
# =========================================================

error_df = pd.read_csv(
    "reports/page3_error_summary.csv"
)

# =========================================================
# TITLE
# =========================================================

st.title("🧠 Feature Representation vs Sampling")

st.markdown(
    """
    Balancing alone did not fully resolve classification errors.

    We therefore investigated whether performance is primarily driven by:
    
    - sampling strategy  
    - or feature representation
    """
)

st.markdown("---")

# =========================================================
# SECTION 1 — GLOBAL COMPARISON
# =========================================================

st.header("1️⃣ Global Performance Comparison")

col1, col2 = st.columns(2)

with col1:

    st.image(
        "reports/figures/page3_macro_f1.png",
        use_container_width=True
    )

with col2:

    st.image(
        "reports/figures/page3_rmse.png",
        use_container_width=True
    )

# =========================================================
# GLOBAL LEGEND
# =========================================================

st.markdown(
    """
<span style="color:#1f77b4;">●</span> Embeddings + Basic Features &nbsp;&nbsp;
<span style="color:#2ca02c;">●</span> Embeddings + Structural Features &nbsp;&nbsp;
<span style="color:#ff7f0e;">●</span> TF-IDF + Structural Features &nbsp;&nbsp;
<span style="color:#444444;">— —</span> Best
""",
    unsafe_allow_html=True
)

# =========================================================
# INTERPRETATION
# =========================================================

st.info(
    """
    ### Initial Observation
    
    - Sampling strategies produced only limited improvements  
    - Undersampling consistently reduced performance  
    - Embedding-based representations consistently outperformed TF-IDF  
    
    ---
    
    Representation quality appears to have a larger impact on model behavior than sampling strategy alone.
    """
)

# =========================================================
# SECTION 2 — ERROR STRUCTURE
# =========================================================

st.markdown("---")

st.header("2️⃣ Error Structure Analysis")

col1, col2 = st.columns(2)

with col1:

    st.image(
        "reports/figures/page3_cm_embeddings.png",
        use_container_width=True
    )

with col2:

    st.image(
        "reports/figures/page3_cm_tfidf.png",
        use_container_width=True
    )

# =========================================================
# ERROR SUMMARY
# =========================================================

st.subheader("📋 Error Summary")

st.table(error_df)

# =========================================================
# INTERPRETATION
# =========================================================

st.info(
    """
    ### Key Insight
    
    TF-IDF produced:
    
    - larger prediction distances  
    - more severe misclassifications  
    - weaker separation of neighboring classes  
    
    Embeddings instead tended to produce nearby prediction errors,
    suggesting a better preservation of semantic relationships.
    """
)

# =========================================================
# SECTION 3 — SCIENTIFIC CONCLUSION
# =========================================================

st.markdown("---")

st.header("3️⃣ Scientific Conclusion")

st.success(
    """
    ### Core Finding
    
    Sampling strategies produced only limited improvements,
    while feature representation consistently changed both
    overall performance and error behavior.
    
    ---
    
    Embeddings achieved:
    
    - higher Macro-F1  
    - lower RMSE  
    - fewer critical errors  
    - smaller prediction distances  
    
    This suggests that representation quality appears to be the dominant factor in handling semantic ambiguity.
    
    ---
    
    ### Transition to Final Phase
    
    This raised an important question:
    
    Could TF-IDF and embeddings capture complementary signals rather than competing representations?
    
    → This motivated the development of hybrid models in the final phase.
    """
)