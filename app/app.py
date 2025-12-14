import streamlit as st
import joblib, json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# To access src files (like preprocess)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import wordopt  

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.write("Compare different ML pipelines and explore what influenced the predictions.")

# -----------------------------
# Load available configurations
# -----------------------------
with open("app/model_configs.json") as f:
    model_configs = json.load(f)

feature_type = st.selectbox("Choose Feature Extractor", list(model_configs.keys()))
model_name = st.selectbox("Choose Model", model_configs[feature_type])

text_input = st.text_area("Enter a news headline or short text:")

# -----------------------------
# Prediction Section
# -----------------------------
if st.button("Predict"):
    model_path = f"outputs/models/{feature_type}_{model_name}.pkl"
    vectorizer_path = f"outputs/vectorizers/{feature_type}_vectorizer.pkl"
    metric_path = f"outputs/metrics/{feature_type}_{model_name}_metrics.json"
   
    if not os.path.exists(model_path):
        st.error("Model not found. Please train it first.")
    else:
        # Load model, vectorizer, and metrics
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        with open(metric_path, "r") as f:
            metrics = json.load(f)

        # Preprocess and vectorize
        clean = wordopt(text_input)
        if feature_type == "word2vec":
            from src.features import get_word2vec_features
            X, _ = get_word2vec_features([clean])
        else:
            X = vectorizer.transform([clean])

        pred_proba = model.predict_proba(X)[0]
        pred_label = model.predict(X)[0]
        label_name = "🟥 Fake News" if pred_label == 1 else "🟩 Real News"

        st.markdown(f"## Prediction: **{label_name}**")
        st.write(f"**Probability:** {pred_proba[pred_label]*100:.2f}%")
        st.progress(float(pred_proba[pred_label]))

        # -----------------------------
        # Metrics summary
        # -----------------------------
        st.subheader("📊 Model Performance Metrics")
        st.json(metrics)

        # -----------------------------
        # Explainability (word importance)
        # -----------------------------
        if feature_type == "tfidf":
            st.subheader("🧩 Top Influential Words")

            feature_names = np.array(vectorizer.get_feature_names_out())
            coefs = model.coef_[0] if hasattr(model, "coef_") else None

            if coefs is not None:
                input_features = X.toarray().flatten()
                word_contrib = [
                    (feature_names[i], coefs[i] * input_features[i])
                    for i in np.where(input_features > 0)[0]
                ]
                word_contrib = sorted(word_contrib, key=lambda x: abs(x[1]), reverse=True)[:15]

                if word_contrib:
                    df = pd.DataFrame(word_contrib, columns=["Word", "Contribution"])
                    df["Impact"] = df["Contribution"].apply(lambda x: "Fake ↑" if x > 0 else "Real ↓")

                    st.dataframe(df.style.background_gradient(
                        subset=["Contribution"], cmap="RdYlGn_r", vmin=-0.2, vmax=0.2))

                    # Bar chart
                    st.subheader("📈 Word Impact Visualization")
                    plt.figure(figsize=(8, 4))
                    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in df["Contribution"]]
                    plt.barh(df["Word"], df["Contribution"], color=colors)
                    plt.xlabel("Contribution to Prediction")
                    plt.gca().invert_yaxis()
                    st.pyplot(plt)

                    # Word Cloud
                    st.subheader("☁️ Word Cloud of Influential Words")
                    word_weights = {w: abs(c) for w, c in word_contrib}
                    wc = WordCloud(width=800, height=300, background_color="white",
                                   colormap="RdYlGn_r").generate_from_frequencies(word_weights)
                    st.image(wc.to_array(), use_container_width=True)
                else:
                    st.info("No significant words found in this text.")
            else:
                st.info("This model type doesn’t provide coefficients (e.g., XGBoost).")

        # -----------------------------
        # Hidden Text Insights
        # -----------------------------
        st.subheader("🔎 Hidden Text Insights")
        words = clean.split()
        st.write(f"**Word Count:** {len(words)}")
        st.write(f"**Unique Words:** {len(set(words))}")
        avg_len = np.mean([len(w) for w in words]) if words else 0
        st.write(f"**Average Word Length:** {avg_len:.2f}")

        # Clickbait detector
        clickbait_words = ["breaking", "shocking", "exclusive", "urgent", "revealed", "secret", "alert"]
        found_clickbait = [w for w in words if w.lower() in clickbait_words]
        if found_clickbait:
            st.warning(f"⚠️ Clickbait-like words detected: {', '.join(found_clickbait)}")

        # Probability table
        st.subheader("📉 Class Probabilities")
        st.write(pd.DataFrame({
            "Class": ["Real News", "Fake News"],
            "Probability": [pred_proba[0], pred_proba[1]]
        }).set_index("Class"))

        st.markdown("---")
        st.caption("Built with ❤️ — Compare models, visualize influence, and explore text insights.")
