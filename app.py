"""Streamlit front-end for the SMS Spam Classifier MLOps pipeline."""
import streamlit as st

from src.predict import load_metrics, load_model, predict_message

st.set_page_config(page_title="SMS Spam Classifier MLOps", page_icon="🛡️", layout="centered")

@st.cache_resource
def get_model():
    return load_model()

def main():
    st.title("🛡️ SMS Spam Classifier MLOps Dashboard")
    st.markdown("Classify SMS messages using a production-ready ML pipeline tracked with MLflow.")

    model = get_model()
    metrics_data = load_metrics()

    # Sidebar for metrics & model info
    st.sidebar.header("📊 Model Performance")
    if metrics_data:
        st.sidebar.text(f"Best Model: {metrics_data.get('best_model', 'N/A')}")
        st.sidebar.text(f"Trained At: {metrics_data.get('trained_at', 'N/A')}")
        metrics = metrics_data.get("metrics", {})
        st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        st.sidebar.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
        st.sidebar.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        st.sidebar.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    else:
        st.sidebar.warning("Metrics file not found. Run training first.")

    # Main prediction area
    st.subheader("💬 Test a Message")
    default_text = "WINNER!! Claim your free prize now, call 09061701461"
    user_input = st.text_area("Enter SMS text to classify:", value=default_text, height=100)

    if st.button("Classify Message", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text to classify.")
            return

        result = predict_message(model, user_input)
        label = result["label"]
        prob = result["spam_probability"]

        if label == "Spam":
            st.error(f"🚨 Prediction: **{label}** (Spam Probability: `{prob:.2%}`)")
        else:
            st.success(f"✅ Prediction: **{label}** (Spam Probability: `{prob:.2%}`)")

        with st.expander("🔍 View Preprocessed Text"):
            st.code(result["cleaned"])

if __name__ == "__main__":
    main()
