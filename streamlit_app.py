import streamlit as st
import os
from news_analysis import (
    is_model_trained,
    get_trained_model,
    get_analyzation_result,
    save_result_in_file,
)

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    h1 {
        color: #333;
        text-align: center;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.title("📰 Fake News Detector")
    st.markdown("### Analyse Sentiment and Detect Fake News!")

    # Check model status
    if is_model_trained():
        st.success("Model trained and ready. Select an article to analyze.", icon="✅")
    else:
        st.info(
            "Model training will begin on first analysis. This may take a moment.",
            icon="ℹ️",
        )

    # File uploader
    uploaded_file = st.file_uploader("Select Article Text File", type=["txt"])

    if uploaded_file is not None:
        # Read file content
        try:
            raw_text = uploaded_file.read().decode("utf-8")
            st.text_area("Article Content", raw_text, height=200)

            if st.button("Analyse Article"):
                with st.spinner("Analyzing..."):
                    # Get model
                    vectorizer, clf = get_trained_model()
                    
                    # Analyze
                    result = get_analyzation_result(clf, vectorizer, raw_text)
                    sentiment = result["sentiment"]
                    label = result["label"]
                    accuracy = result["confidence"]

                    # Save result (optional, matching original logic)
                    save_result_in_file(raw_text, sentiment, label, accuracy)

                    # Display results
                    st.divider()
                    st.subheader("Analysis Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Sentiment")
                        if sentiment == "positive":
                            st.success(f"😊 {sentiment.capitalize()}")
                        elif sentiment == "negative":
                            st.error(f"😞 {sentiment.capitalize()}")
                        else:
                            st.warning(f"😐 {sentiment.capitalize()}")

                    with col2:
                        st.markdown("#### Real or Fake")
                        if label == "REAL":
                            st.success(f"✅ {accuracy}% {label}")
                        else:
                            st.error(f"🚫 {accuracy}% {label}")
                            
        except Exception as e:
            st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
