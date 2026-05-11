from pathlib import Path
import streamlit as st
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Exploratory Data Analysis",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "reviews_processed.csv"

FIG_DIR = BASE_DIR / "reports" / "figures"

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# =========================================================
# KPIs
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
    statistical properties and real-world challenges of the collected dataset.
    """
)

st.markdown("---")

# =========================================================
# SECTION 1 — DATASET OVERVIEW
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

st.info(
    """
    The dataset contains real-world customer reviews collected from multiple companies
    in the automotive retail sector.

    The combination of textual, temporal and metadata features
    enables both sentiment and behavioral analysis.
    """
)

# =========================================================
# SECTION 2 — RATING DISTRIBUTION
# =========================================================

st.header("2️⃣ Customer Satisfaction Distribution")
col1, col2 = st.columns([1.3, 1])

with col1:

    st.image(
        FIG_DIR / "rating_distribution.png",
        use_container_width=True
    )
with col2:
    st.info(
    """
    The dataset is strongly imbalanced.

    High ratings dominate the distribution,
    while minority ratings occur substantially less frequently.

    This imbalance later becomes one of the major modeling challenges.
    """
    )



# =========================================================
# SECTION 3 — TEMPORAL ANALYSIS
# =========================================================

st.header("3️⃣ Temporal Analysis")

col1, col2 = st.columns([1.3, 1])

with col1:
    st.image(
        FIG_DIR / "temporal_analysis.png",
        use_container_width=True
    )

with col2:
    st.info(
    """
    The dataset spans more than a decade of customer feedback.

    This introduces evolving language usage,
    changing customer behavior and temporal variation in review patterns.
    """
    )

# =========================================================
# SECTION 4 — GEOGRAPHIC & LANGUAGE DIVERSITY
# =========================================================

st.header("4️⃣ Geographic and Language Diversity")

col1, col2 = st.columns(2)

with col1:

    st.image(
        FIG_DIR / "geographic_distribution.png",
        use_container_width=True
    )

with col2:

    st.image(
        FIG_DIR / "language_distribution.png",
        use_container_width=True
    )

st.info(
    """
    Reviews originate from multiple geographic regions
    and contain multilingual language structures.

    This increases linguistic variability
    and motivates robust preprocessing and semantic feature extraction techniques.
    """
)

# =========================================================
# SECTION 5 — TEXT CHARACTERISTICS
# =========================================================

st.header("5️⃣ Text Characteristics")

col1, col2 = st.columns(2)

with col1:

    st.image(
        FIG_DIR / "positive_terms.png",
        use_container_width=True
    )

with col2:

    st.image(
        FIG_DIR / "negative_terms.png",
        use_container_width=True
    )

    st.info(
        """
        Positive reviews are dominated by delivery speed,
        service quality and product satisfaction.

        Negative reviews contain stronger operational problem indicators,
        including complaints about delays, support and product quality.

        At the same time, substantial vocabulary overlap exists between rating classes,
        already hinting at the semantic ambiguity later observed during classification.
        """
    )

# =========================================================
# SECTION 6 — INITIAL INSIGHTS
# =========================================================

st.header("6️⃣ Initial Insights")

st.success(
    """
    ### Key Observations

    - strong class imbalance
    - semantic overlap between ratings
    - multilingual and heterogeneous review structures
    - noisy real-world customer language

    ---

    These observations motivated the preprocessing,
    feature engineering and representation studies
    performed in the next stages of the project.
    """
)