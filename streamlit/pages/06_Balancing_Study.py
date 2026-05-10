# =========================================================
# FILE:
# streamlit/pages/06_Balancing_Study.py
# =========================================================

import streamlit as st
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Balancing Study",
    layout="wide"
)

# =========================================================
# LOAD DATA
# =========================================================

performance_df = pd.read_csv(
    "reports/page2_performance_summary.csv"
)

# =========================================================
# TITLE
# =========================================================

st.title("⚖️ Balancing Study")

st.markdown(
    """
    Testing whether class imbalance is the dominant source of classification failure.
    """
)

st.markdown("---")

# =========================================================
# SECTION 1 — OBSERVED IMBALANCE
# =========================================================

st.header("1️⃣ Observed Imbalance")

col1, col2 = st.columns([1.3, 1])

with col1:

    st.image(
        "reports/figures/page2_original_distribution.png",
        use_container_width=True
    )

with col2:

    st.info(
        """
        ### Initial Observation
        
        - Strong dominance of high-rating reviews  
        - Minority classes heavily underrepresented  
        - Weak separability of neighboring classes  
        
        ---
        
        This raised the hypothesis that class imbalance may contribute significantly to classification failure.
        """
    )

# =========================================================
# SECTION 2 — BEFORE VS AFTER
# =========================================================

st.markdown("---")

st.header("2️⃣ Before vs After Balancing")

col1, col2 = st.columns(2)

with col1:

    st.image(
        "reports/figures/page2_before_balancing.png",
        use_container_width=True
    )

with col2:

    st.image(
        "reports/figures/page2_after_balancing.png",
        use_container_width=True
    )

# =========================================================
# SECTION 3 — PERFORMANCE IMPACT
# =========================================================

st.markdown("---")

st.header("3️⃣ Performance Impact")

st.table(performance_df)

st.caption(
    """
    Grouped balancing setup:
    oversampling + augmentation for minority classes,
    undersampling for majority class.
    """
)

# =========================================================
# SECTION 4 — SCIENTIFIC INTERPRETATION
# =========================================================

st.success(
    """
    ### Scientific Interpretation
    
    Balancing reduced distributional bias and improved robustness under the grouped-class setup.
    
    However, the approach did not resolve semantic ambiguity between neighboring rating levels.
    
    ---
    
    The grouped setup simplifies the decision boundary and therefore cannot be interpreted as a direct solution to the original five-class problem.
    
    ---
    
    ### Key Conclusion
    
    Class imbalance amplifies the problem,
    but semantic overlap remains the dominant challenge.
    
    This motivated a deeper investigation into:
    
    - feature representations  
    - semantic separability  
    - alternative modeling strategies
    """
)