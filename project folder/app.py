import streamlit as st
import pickle

# Load saved model and tfidf
model = pickle.load(open('model (1).pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Title
st.title("📰 Fake News Detector")
st.subheader("Paste any news article to check if it's Real or Fake")
st.divider()

# Input
news = st.text_area("Enter News Article Here", height=200)

# Button
if st.button("🔍 Check News"):
    if news.strip() == "":
        st.warning("Please enter some news text!")
    else:
        input_data = tfidf.transform([news])
        prediction = model.predict(input_data)
        confidence = model.predict_proba(input_data).max() * 100

        st.divider()

        if prediction[0] == 0:
            st.error(f"❌ FAKE NEWS")
            st.metric("Confidence", f"{confidence:.2f}%")
        else:
            st.success(f"✅ REAL NEWS")
            st.metric("Confidence", f"{confidence:.2f}%")

st.divider()
st.caption("Built with Python & Machine Learning | Accuracy: 98.64%")