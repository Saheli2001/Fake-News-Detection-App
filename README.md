# Fake News Detector

My first NLP app — built to try out all the models and techniques I’ve learned so far. The motive here was not to give you 90% accuracy on blind test but to code everything that i have learned to make myself understand everything clearly. But the goal is to improve this project to make it more interactive , more fun and more accurate. :)

Paste some news, pick a feature extractor and model, and see if it’s Real or Fake. Compare different pipelines and explore what words influenced the prediction.

---

## Features
- Feature extractors: TF-IDF, n-grams, Word2Vec
- Models: Logistic Regression, Random Forest, XGBoost  
- Confidence scores and word-level insights  
- Simple clickbait detection  

---

## How to Run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app/app.py
