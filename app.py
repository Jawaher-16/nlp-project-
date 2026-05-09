import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("🤖 AI vs Human Text Detector")

st.write("Enter text and the model will predict whether it was written by AI or a human.")

# User input
text = st.text_area("✍ Enter text here")

# Prediction button
if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Clean text
        cleaned_text = text.lower().strip()

        # Transform input
        text_vec = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_vec)[0]

        # Prediction probabilities
        probability = model.predict_proba(text_vec)[0]

        # Result
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success("🤖 AI Generated Text label 1")
        else:
            st.success("🧑 Human Written Text label 0")

        # Confidence
        st.subheader("📈 Prediction Confidence")

        st.write({
            "Human": f"{probability[0] * 100:.2f}%",
            "AI": f"{probability[1] * 100:.2f}%"
        })
