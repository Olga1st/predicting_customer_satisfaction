# =========================================================
# FILE:
# streamlit/pages/05_Baseline_Behavior.py
# =========================================================

import streamlit as st
import pandas as pd
from PIL import Image

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Baseline Behavior",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📊 Baseline Behavior")

st.markdown(
    """
    We first compare a simplified classification setup with the real-world multi-class problem.
    """
)

st.markdown("---")

# =========================================================
# SECTION 1 — BINARY BASELINE
# =========================================================

st.header("1️⃣ Simplified Binary Classification")

st.caption(
    "Controlled setup before introducing ordinal ambiguity."
)

col1, col2 = st.columns([1.2, 1])

with col1:

    st.image(
        "reports/figures/page1_binary_performance.png",
        use_container_width=True
    )

with col2:

    st.info(
        """
        ### Observation
        
        Both feature representations perform strongly in the simplified setting.
        
        Embeddings consistently outperform TF-IDF, but the task itself remains relatively easy due to the absence of fine-grained semantic distinctions.
        """
    )

# =========================================================
# SECTION 2 — MULTI-CLASS BREAKDOWN
# =========================================================

st.markdown("---")

st.header("2️⃣ Real-World Multi-Class Classification")

st.caption(
    "Introducing semantic overlap and ordinal ambiguity."
)

# =========================================================
# METRICS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Embeddings")

    emb_df = pd.DataFrame({
        "Metric": ["Macro-F1", "RMSE"],
        "Value": [0.684, 0.480]
    })

    st.table(emb_df)

with col2:

    st.subheader("TF-IDF")

    tfidf_df = pd.DataFrame({
        "Metric": ["Macro-F1", "RMSE"],
        "Value": [0.469, 0.709]
    })

    st.table(tfidf_df)

# =========================================================
# CONFUSION MATRICES
# =========================================================

st.subheader("🔍 Normalized Confusion Matrices")

col1, col2 = st.columns(2)

with col1:

    st.image(
        "reports/figures/page1_cm_embeddings.png",
        use_container_width=True
    )

with col2:

    st.image(
        "reports/figures/page1_cm_tfidf.png",
        use_container_width=True
    )

# =========================================================
# FINAL INSIGHT
# =========================================================

st.success(
    """
    ### Scientific Insight
    
    Performance drops substantially in the realistic multi-class setting.
    
    The confusion matrices reveal that:
    
    - extreme ratings remain relatively separable  
    - neighboring classes exhibit strong semantic overlap  
    - TF-IDF shows substantially larger confusion between adjacent ordinal classes  
    
    Embeddings preserve semantic relationships more effectively and therefore produce more stable predictions.
    
    ---
    
    ### Why Macro-F1?
    
    The dataset is strongly imbalanced. Macro-F1 evaluates all classes equally and therefore provides a more reliable estimate of robustness across minority classes.
    
    ---
    
    ### Why RMSE?
    
    Ratings are ordinal by nature. RMSE captures the distance between prediction errors and therefore reflects ordinal severity more effectively than standard classification metrics.
    """
)