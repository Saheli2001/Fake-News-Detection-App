import shap
import lime
from lime.lime_text import LimeTextExplainer
import numpy as np

def explain_with_shap(model, vectorizer, text):
    """Return SHAP explanation for a single text input."""
    try:
        # Create a callable prediction function for SHAP as sklearn models are not callable directly
        predict_fn = lambda x: model.predict_proba(vectorizer.transform(x))
        explainer = shap.Explainer(predict_fn, shap.maskers.Text(tokenizer=str.split))
        shap_values = explainer([text])

        return shap_values
    except Exception as e:
        print(f"[SHAP Error] {e}")
        return None

def explain_with_lime(model, vectorizer, text):
    """Return LIME explanation for a single text input."""
    try:
        class_names = ['Fake', 'Real']
        explainer = LimeTextExplainer(class_names=class_names)

        def predict_fn(texts):
            X = vectorizer.transform(texts)
            return model.predict_proba(X)
        
        exp = explainer.explain_instance(text, predict_fn, num_features=10)
        return exp
    except Exception as e:
        print(f"[LIME Error] {e}")
        return None
    



