import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.load_data import load_raw_data
from src.data.load_data import load_and_clean_data

st.title("📊 Data Analysis")

df = load_raw_data()
df = load_and_clean_data()

# =====================
# DISTRIBUTION
# =====================
st.header("📈 Rating Distribution")

fig, ax = plt.subplots()
sns.countplot(x="rating", data=df, ax=ax)
st.pyplot(fig)

st.markdown("""
🔎 Insight:
- Strong imbalance towards **5-star reviews**
- Typical for online platforms
""")

# =====================
# TEXT LENGTH
# =====================
st.header("📝 Review Length Distribution")

df["length"] = df["review_text"].apply(lambda x: len(str(x).split()))

fig, ax = plt.subplots()
sns.histplot(df["length"], bins=50, ax=ax)
st.pyplot(fig)

st.markdown("""
🔎 Insight:
- Most reviews are short
- Long reviews often contain detailed complaints
""")

# =====================
# VERIFIED
# =====================
st.header("✅ Verified Reviews")

fig, ax = plt.subplots()
sns.countplot(x="verified", data=df, ax=ax)
st.pyplot(fig)

st.markdown("""
🔎 Insight:
- Verified users may be more trustworthy
""")

st.dataframe(df.head())

