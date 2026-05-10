import streamlit as st
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Final Model Evolution",
    layout="wide"
)

# =========================================================
# LOAD DATA
# =========================================================

comparison_df = pd.read_csv("reports/final_model_comparison.csv")
per_class_df = pd.read_csv("reports/final_per_class_f1.csv")

# =========================================================
# TITLE
# =========================================================

st.title("🚀 Final Model Evolution")

st.markdown(
    """
    After identifying feature representation as the key driver of performance,
    we combine all insights into a final optimized hybrid model.
    """
)

st.markdown("---")

# =========================================================
# SECTION 1 — STRATEGY
# =========================================================

st.header("1️⃣ Final Modeling Strategy")

col1, col2 = st.columns([1.3, 1])

with col1:

    st.info(
        """
        ### Motivation
        
        Previous findings showed:
        
        - sampling strategies had limited impact  
        - feature representation was dominant  
        - semantic ambiguity remained the core challenge  
        """
    )

with col2:

    st.markdown("### Final Model Pipeline")

    pipeline_html = """
    <div style="
        display:flex;
        flex-direction:column;
        align-items:center;
        gap:12px;
        margin-top:20px;
    ">

        <div style="
            background-color:#EAF2FF;
            padding:12px 24px;
            border-radius:10px;
            width:260px;
            text-align:center;
            font-weight:600;
            border:1px solid #4A6FA5;
        ">
            XGBoost
        </div>

        <div style="font-size:24px;">⬇️</div>

        <div style="
            background-color:#EAF2FF;
            padding:12px 24px;
            border-radius:10px;
            width:260px;
            text-align:center;
            font-weight:600;
            border:1px solid #4A6FA5;
        ">
            Structural Features
        </div>

        <div style="font-size:24px;">⬇️</div>

        <div style="
            background-color:#EAF2FF;
            padding:12px 24px;
            border-radius:10px;
            width:260px;
            text-align:center;
            font-weight:600;
            border:1px solid #4A6FA5;
        ">
            Embeddings
        </div>

        <div style="font-size:24px;">⬇️</div>

        <div style="
            background-color:#EAF2FF;
            padding:12px 24px;
            border-radius:10px;
            width:260px;
            text-align:center;
            font-weight:600;
            border:1px solid #4A6FA5;
        ">
            TF-IDF
        </div>

        <div style="font-size:24px;">⬇️</div>

        <div style="
            background-color:#EAF2FF;
            padding:12px 24px;
            border-radius:10px;
            width:260px;
            text-align:center;
            font-weight:600;
            border:1px solid #4A6FA5;
        ">
            Hyperparameter Optimization
        </div>

        <div style="font-size:24px;">⬇️</div>

        <div style="
            background-color:#DFF5E1;
            padding:14px 24px;
            border-radius:12px;
            width:280px;
            text-align:center;
            font-weight:700;
            border:2px solid #2E8B57;
            font-size:18px;
        ">
            Final Hybrid Model
        </div>

    </div>
    """

    st.markdown(
        pipeline_html,
        unsafe_allow_html=True
    )
# =========================================================
# SECTION 2 — EVOLUTION
# =========================================================

st.header("2️⃣ Model Evolution")

st.image(
    "reports/figures/model_evolution.png",
    use_container_width=True
)

st.info(
    """
    ### Interpretation
    
    Each step reflects a targeted decision derived from prior experiments.
    
    The strongest gains come from:
    
    - improved feature representation  
    - hybrid semantic signals  
    - hyperparameter optimization  
    
    The model does not improve randomly — it improves structurally.
    """
)

# =========================================================
# SECTION 3 — PER CLASS
# =========================================================

st.header("3️⃣ Baseline vs Hybrid Model")

st.image(
    "reports/figures/final_per_class_comparison.png",
    use_container_width=True
)

st.markdown(
    """
    ### Insight
    
    - consistent improvement across all classes  
    - strongest gains in mid-range classes  
    - reduced class imbalance sensitivity  
    
    → confirms improved semantic stability of hybrid approach
    """
)

# =========================================================
# SECTION 4 — SUMMARY
# =========================================================

st.header("4️⃣ Final Metrics")

st.dataframe(
    comparison_df[["stage", "macro_f1_mean", "rmse"]],
    use_container_width=True
)

# =========================================================
# SECTION 5 — CONCLUSION
# =========================================================

st.header("5️⃣ Scientific Conclusion")

st.success(
    """
    ### Key Finding
    
    Performance improvements are not driven by sampling,
    but by representation quality and hybrid feature design.
    
    ---
    
    The final model integrates:
    
    - semantic embeddings  
    - lexical TF-IDF signals  
    - structural metadata  
    
    → resulting in the most robust and stable performance across all classes.
    """
)