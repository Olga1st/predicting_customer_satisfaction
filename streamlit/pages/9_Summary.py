import streamlit as st
from src.analysis.reporting import get_best_model
from src.analysis.business_insights import generate_business_summary

#Achtung: Hier fehlen die Funktionen 
# get_best_model und generate_business_summary,
# die in src.analysis.reporting bzw. src.analysis.business_insights
# definiert sein sollten. Hier muss nachgearbeitet werden!
#Ergebnis sollte "a more detailed business report" liefern.
st.title("📌 Summary & Business Insights")

best = get_best_model()

summary = generate_business_summary(best)

st.text(summary)

# Alternative: 
st.title("📌 Summary & Conclusion")

st.markdown("""
## 🏆 Best Model
- Undersampling + TF-IDF

## 📊 Key Findings
- Imbalance heavily impacts performance
- Business KPI ≠ accuracy
- Feature engineering is critical

## 🚀 Next Steps
- Deploy API
- Add real-time monitoring
- Improve multilingual handling
""")