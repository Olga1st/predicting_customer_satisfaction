import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Project Overview & Exploratory Data Analysis")

df = pd.read_csv("data/processed/reviews_clean.csv")

# --- KPIs ---
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(df))
col2.metric("Average Rating", round(df["rating"].mean(), 2))
col3.metric("Verified Reviews (%)", round(df["verified"].mean()*100, 2))

# --- Dataset Preview ---
st.subheader("Dataset Overview")
st.dataframe(df.head())

# --- Ratings Distribution ---
st.subheader("Ratings Distribution")
fig, ax = plt.subplots()
df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
st.pyplot(fig)

# --- Verified vs Non Verified ---
st.subheader("Verified vs Non-Verified (Average Rating)")
verified_avg = df.groupby("verified")["rating"].mean()
st.bar_chart(verified_avg)

# --- Supplier Response Impact ---
st.subheader("Impact of Supplier Response on Rating")
df["has_response"] = df["supplier_response"].notna().astype(int)
response_avg = df.groupby("has_response")["rating"].mean()
st.bar_chart(response_avg)

# --- Geographic Analysis ---
st.subheader("Geographical Analysis (Rating by CountryWebSite)")
geo1 = df.groupby("company_site")["rating"].mean().sort_values()
st.bar_chart(geo1)

st.subheader("Geographical Analysis (Rating by CostomerCountry)")
geo2 = df.groupby("location")["rating"].mean().sort_values()
st.bar_chart(geo2)

# --- Company Analysis ---
st.subheader("Top Companies by Rating")
top_companies = df.groupby("company")["rating"].mean().sort_values(ascending=False).head(10)
st.bar_chart(top_companies)

# --- Additional Insight ---
st.subheader("Review Length vs Rating")
df["review_length"] = df["review_text"].astype(str).apply(lambda x: len(x.split()))
st.scatter_chart(df[["review_length", "rating"]])

# --- Issue Analysis ---
st.subheader("Wordcloud")
text = " ".join(df["review_text_clean"].dropna())
wordcloud = WordCloud(width=800, height=400).generate(text)

fig, ax = plt.subplots()
ax.imshow(wordcloud)
ax.axis("off")
st.pyplot(fig)

#ToDo: um die bereinigten häufigsten worte erweitern?!