import streamlit as st
import pandas as pd

# =========================================================
# LOAD DATA
# =========================================================

DATA_PATH = "data/processed/reviews_processed.csv"

@st.cache_data

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


df = load_data()

# =========================================================
# KPIS
# =========================================================

num_reviews = len(df)
avg_rating = df["rating"].mean()
response_rate = (
    df["supplier_response"]
    .notna()
    .mean() * 100
)

num_companies = df["company"].nunique()

# =========================================================
# TITLE
# =========================================================

st.title("📊 Exploratory Data Analysis")

st.markdown(
    """
    The objective of this stage is to understand the structure,
    quality and statistical characteristics of the collected dataset.
    """
)

st.markdown("---")

# =========================================================
# SECTION 1
# =========================================================

st.header("1️⃣ Dataset Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Total Reviews",
        f"{num_reviews:,}"
    )

with kpi2:
    st.metric(
        "Average Rating",
        f"{avg_rating:.2f}"
    )

with kpi3:
    st.metric(
        "Supplier Response Rate",
        f"{response_rate:.1f}%"
    )

with kpi4:
    st.metric(
        "Companies",
        num_companies
    )


# =========================================================
# SECTION 2
# =========================================================

st.header("2️⃣ Customer Satisfaction Distribution")

st.image(
    str("reports/figures/rating_distribution.png"),
    use_container_width=True
)

st.info(
    """
    The dataset is strongly imbalanced.

    High ratings dominate the distribution,
    while low and mid-range ratings occur significantly less frequently.

    This imbalance later becomes a central challenge for model evaluation.
    """
)

# =========================================================
# SECTION 3
# =========================================================

st.header("3️⃣ Temporal Analysis")

st.image(
    str("reports/figures/temporal_analysis.png"),
    use_container_width=True
)

st.info(
    """
    The dataset spans more than a decade of customer feedback,
    reflecting evolving customer behavior and language usage patterns.
    """
)



# =========================================================
# SECTION 5
# =========================================================

st.header("5️⃣ Frequent Terms")

col1, col2 = st.columns(2)

with col1:

    st.image(
        str("reports/figures/positive_terms.png"),
        use_container_width=True
    )

with col2:

    st.image(
        str("reports/figures/negative_terms.png"),
        use_container_width=True
    )

st.info(
    """
    Positive reviews are dominated by delivery speed,
    quality and service-related expressions.

    Negative reviews contain significantly more operational problem indicators,
    including complaints about delays, support or product quality.

    The overlap between many expressions already hints at the semantic ambiguity
    later observed during classification.
    """
)


# =========================================================
# SECTION 6
# =========================================================

st.header("6️⃣ Initial Insights")

st.success(
    """
    ### Key Observations

    - strong class imbalance
    - noisy real-world text
    - semantic overlap between ratings
    - multilingual and heterogeneous review structures

    ---

    These observations motivated the extensive preprocessing,
    feature engineering and representation studies
    performed in the next stages.
    """
)