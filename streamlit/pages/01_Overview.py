import streamlit as st

st.title("📌 Business Overview")

st.header("🎯 Business Case")

st.markdown("""
Companies receive thousands of customer reviews across platforms like Trustpilot.

👉 The challenge:
- Reviews are **unstructured text**
- Manual analysis does **not scale**
- Negative feedback is often missed

🎯 Goal of this system:
- Automatically predict **customer satisfaction (1–5 stars)**
- Detect **unhappy customers early**
- Enable **data-driven decisions**
""")

st.header("🌐 Data Source: Trustpilot")

st.markdown("""
We use real-world customer reviews collected via web scraping.

### 🔍 Web Scraping Overview
- Source: Trustpilot
- Data:
  - Review text
  - Rating (1–5)
  - Verified flag
  - Company response

### ⚙️ Pipeline
1. Crawl Trustpilot pages
2. Extract structured data
3. Store as JSON
4. Feed into ML pipeline

⚠️ Ethical Consideration:
- Respect robots.txt
- Rate-limited scraping

## 🚨 Key KPI
👉 Recall for low ratings (1–2 stars)
""")

st.info("👉 This system transforms raw reviews into actionable insights.")

