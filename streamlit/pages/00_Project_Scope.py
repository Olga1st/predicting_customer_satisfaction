import streamlit as st

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Project Scope",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📦 Project Scope & Business Problem")

st.markdown(
    """
    This project investigates whether customer satisfaction can be predicted automatically
    from textual customer reviews within the automotive supply chain domain.
    """
)

st.markdown("---")

# =========================================================
# SECTION 1
# =========================================================

st.header("1️⃣ Supply Chain Context")

st.markdown(
    """
    <div style='text-align:center;font-size:28px;padding:20px;'>
    Supplier → Warehouse → Store → Customer → Review
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    """
    Customer reviews represent valuable operational feedback signals.

    Delayed deliveries, damaged products, poor communication,
    or return handling issues are often reflected directly
    in customer satisfaction ratings.
    """
)

# =========================================================
# SECTION 2
# =========================================================

st.header("2️⃣ Business Problem")

col1, col2 = st.columns(2)

with col1:

    st.success(
        """
        ### Objective

        Predict customer satisfaction automatically
        from textual reviews.

        Target:

        - 5-class rating prediction
        - ordinal classification problem
        """
    )

with col2:

    st.warning(
        """
        ### Main Challenges

        - semantic ambiguity
        - neighboring rating overlap
        - class imbalance
        - noisy real-world reviews
        - multilingual customer feedback
        """
    )

# =========================================================
# SECTION 3
# =========================================================

st.header("3️⃣ Business Value")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Customer Retention",
        "↑"
    )

with col2:
    st.metric(
        "Operational Insight",
        "↑"
    )

with col3:
    st.metric(
        "Complaint Reduction",
        "↑"
    )

st.markdown(
    """
    ### Why this matters

    Automatic satisfaction prediction enables companies to:

    - identify dissatisfied customers earlier
    - detect operational weaknesses
    - prioritize customer support interventions
    - improve long-term brand reputation
    """
)