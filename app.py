import joblib
import pandas as pd
import streamlit as st

from features import extract_features, validate_url, get_indicators

st.set_page_config(page_title="URL Threat Classifier", page_icon="🔗")


@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib"), joblib.load("feature_cols.joblib")


def trust_band(score):
    if score >= 70:
        return "High trust", "success"
    if score >= 40:
        return "Use caution", "warning"
    return "Low trust", "error"


model, feature_cols = load_model()

st.title("URL threat classifier")
st.write("Paste a URL to get a trust score, a predicted class and the signals behind it.")

raw = st.text_input("URL", placeholder="example.com/login")

if st.button("Classify URL"):
    ok, url, message = validate_url(raw)
    if not ok:
        st.error(message)
    else:
        row, values = extract_features(url, feature_cols)
        probs = pd.Series(model.predict_proba(pd.DataFrame([row], columns=feature_cols))[0],
                          index=model.classes_)

        trust = float(probs.get("benign", 0.0)) * 100
        label, level = trust_band(trust)

        left, right = st.columns(2)
        left.metric("Trust score", f"{trust:.0f} / 100")
        right.metric("Predicted class", probs.idxmax())
        getattr(st, level)(f"{label}. Risk score: {100 - trust:.0f} / 100.")
        st.progress(int(round(trust)))

        st.subheader("What looks suspicious")
        indicators = get_indicators(url, values)

        for status, text in indicators:
            if status == "warn":
                st.warning(text)
            elif status == "info":
                st.info(text)
            else:
                st.success(text)

        if probs.idxmax() != "benign" and not any(s == "warn" for s, _ in indicators):
            st.info(
                "No rule-based warnings, but the model's mix of character counts "
                f"(digits, dots, hyphens, special characters) resembles {probs.idxmax()} URLs."
            )

        st.subheader("Probability by class")

        probability_df = pd.DataFrame({
            "Class": probs.index,
            "Probability": probs.values
        })

        st.bar_chart(
            probability_df,
            x="Class",
            y="Probability"
        )