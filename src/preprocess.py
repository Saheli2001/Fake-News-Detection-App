import re
import string
import spacy

def wordopt(text, lemmatize=True):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) 
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    if lemmatize:
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        doc = nlp(text)
        text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

    return text