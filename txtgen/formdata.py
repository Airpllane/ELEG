
import pandas as pd
import xgboost as xgb
import nltk
import pickle

#%%

clf = xgb.XGBClassifier()
clf.load_model('XGB.json')

with open('vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)

#%%

data = pd.read_csv('../data/txtgen/eng_sentences.tsv', sep = '\t', names = ['id', 'lang', 'text']).sample(frac = 1)['text']
data = pd.DataFrame(data, columns = ['text'])

#%%

data['tok'] = data['text'].map(nltk.word_tokenize)
data['tok'] = data['tok'].map(nltk.pos_tag)
data['tok'] = [' '.join([tup[1] for tup in sentence]) for sentence in data['tok']]
data['tok'] = list(vectorizer.transform(data['tok']).toarray())
data['pred'] = clf.predict(data['tok'].tolist())

#%%

'''
del data['tok']
'''
data.to_csv('../data/txtgen/preds.csv', columns = ['text', 'pred'], index = False)