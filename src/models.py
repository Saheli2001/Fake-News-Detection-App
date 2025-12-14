from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_model(name):
    if name == "logistic":
        return LogisticRegression(max_iter=500)
    elif name == "random_forest":
        return RandomForestClassifier(n_estimators=200)
    elif name == "xgboost":
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("Unknown model name")