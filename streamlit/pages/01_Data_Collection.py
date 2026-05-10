import streamlit as st

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Data Collection",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🛠️ Data Collection & Scraping Pipeline")

st.markdown(
    """
    Customer reviews were collected from Trustpilot
    for German companies in the automotive parts retail sector.
    """
)

st.markdown("---")

# =========================================================
# SECTION 1
# =========================================================

st.header("1️⃣ Scraping Pipeline")

st.markdown(
    """
    <div style='text-align:center;font-size:26px;padding:20px;'>
    Trustpilot → Selenium Scraper → Cleaning → Standardization → ML Dataset
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    """
    Selenium was used because Trustpilot relies heavily on dynamically loaded content.

    The scraping pipeline automatically iterated through:

    - multiple companies
    - multiple pages
    - asynchronously loaded review elements
    """
)

# =========================================================
# SECTION 2
# =========================================================

st.header("2️⃣ Collected Attributes")

col1, col2, col3 = st.columns(3)

with col1:

    st.success(
        """
        ### Review Information

        - review_text
        - rating
        - date
        - language
        """
    )

with col2:

    st.success(
        """
        ### Customer Metadata

        - location
        - verified
        - company
        - domain
        """
    )

with col3:

    st.success(
        """
        ### Engineered Signals

        - sentiment
        - review_length
        - has_negation
        - supplier_response
        """
    )


# =========================================================
# SECTION 3
# =========================================================

st.header("3️⃣ Scraping Challenges")

challenge_df = {
    "Challenge": [
        "Dynamic content",
        "Bot detection",
        "Duplicates",
        "Different formats",
        "Incomplete reviews"
    ],
    "Mitigation": [
        "Selenium rendering",
        "Human-like delays",
        "Cleaning pipeline",
        "Standardization",
        "Partial filtering"
    ]
}

st.table(challenge_df)

# =========================================================
# SECTION 4
# =========================================================

st.header("4️⃣ Ethics & Data Usage")

st.warning(
    """
    - Only publicly accessible reviews were collected
    - No personal or sensitive information was used
    - Data was processed exclusively for educational purposes
    - Reviews remain subject to Trustpilot platform ownership
    """
)

