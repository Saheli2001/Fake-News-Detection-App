import joblib, json, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from .preprocess import wordopt
from .features import get_tfidf_features, get_ngram_features, get_word2vec_features
from .models import get_model
import os


def train(feature_type, model_name, train_path='data/train.csv', val_path='data/test.csv', lemmatize=False):

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    X_train = train['text'].apply(lambda x: wordopt(x, lemmatize=lemmatize)).tolist()
    X_val = val['text'].apply(lambda x: wordopt(x, lemmatize=lemmatize)).tolist()
    y_train = train['class']
    y_val = val['class']

    if feature_type == 'tfidf':
        X_train_feat, vectorizer = get_tfidf_features(X_train)
        X_val_feat = vectorizer.transform(X_val)

    elif feature_type == 'ngram':
        X_train_feat, vectorizer = get_ngram_features(X_train)
        X_val_feat = vectorizer.transform(X_val)
    
    elif feature_type == 'word2vec':
        X_train_feat, w2v_model = get_word2vec_features(X_train)
        X_val_feat, vectorizer = get_word2vec_features(X_val)

    else:
        raise ValueError("Wrong feature type.. ")

    model = get_model(model_name)
    model.fit(X_train_feat, y_train)

    preds = model.predict(X_val_feat)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/vectorizers", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    joblib.dump(model, f'outputs/models/{feature_type}_{model_name}.pkl')
    joblib.dump(vectorizer, f'outputs/vectorizers/{feature_type}_vectorizer.pkl')

    metrics = {
        'accuracy': acc,
        'f1_score': f1,
    }

    with open(f'outputs/metrics/{feature_type}_{model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Model: {feature_type}+{model_name} | Acc={acc:.3f} | F1={f1:.3f}")
    return metrics
