
import pandas as pd
import xgboost as xgb
import nltk
import pickle

#%%

clf = xgb.XGBClassifier()
clf.load_model('XGB.json')

with open('vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)

#%% Image captions 1

data = pd.read_csv('../data/txtgen/add_captions8k.csv', usecols = ['caption']).sample(frac = 1)
data = data.rename(columns = {'caption': 'text'})
data = data.dropna()

#%%

data['tok'] = data['text'].map(nltk.word_tokenize)
data['tok'] = data['tok'].map(nltk.pos_tag)
data['tok'] = [' '.join([tup[1] for tup in sentence]) for sentence in data['tok']]
data['tok'] = list(vectorizer.transform(data['tok']).toarray())
data['pred'] = clf.predict(data['tok'].tolist())

#%%

data.to_csv('../data/txtgen/preds.csv', mode = 'a', columns = ['text', 'pred'], index = False, header = False)

#%% Image captions 2

data = pd.read_csv('../data/txtgen/add_captions30k.csv', usecols = ['caption']).sample(frac = 1)
data = data.rename(columns = {'caption': 'text'})
data = data.dropna()

#%%

data['tok'] = data['text'].map(nltk.word_tokenize)
data['tok'] = data['tok'].map(nltk.pos_tag)
data['tok'] = [' '.join([tup[1] for tup in sentence]) for sentence in data['tok']]
data['tok'] = list(vectorizer.transform(data['tok']).toarray())
data['pred'] = clf.predict(data['tok'].tolist())

#%%

data.to_csv('../data/txtgen/preds.csv', mode = 'a', columns = ['text', 'pred'], index = False, header = False)

#%% Newspapers

data = pd.DataFrame()

with pd.read_csv('../data/txtgen/add_newspapers.csv', usecols = ['text'], 
                 chunksize = 10 ** 4) as reader:
    for chunk in reader:
        chunk['tok'] = chunk['text'].map(nltk.word_tokenize)
        chunk['tok'] = chunk['tok'].map(nltk.pos_tag)
        chunk['tok'] = [' '.join([tup[1] for tup in sentence]) for sentence in chunk['tok']]
        chunk['tok'] = list(vectorizer.transform(chunk['tok']).toarray())
        chunk['pred'] = clf.predict(chunk['tok'].tolist())
        chunk = chunk.drop('tok', axis = 1)
        chunk = chunk.dropna()
        data = pd.concat([data, chunk])

#%%

data.to_csv('../data/txtgen/preds.csv', mode = 'a', columns = ['text', 'pred'], index = False, header = False)

#%% SNLIC

data = pd.DataFrame()

with pd.read_csv('../data/txtgen/add_snlic.csv', usecols = ['caption'], 
                 chunksize = 10 ** 4) as reader:
    for chunk in reader:
        chunk = chunk.rename(columns = {'caption': 'text'})
        chunk['tok'] = chunk['text'].map(nltk.word_tokenize)
        chunk['tok'] = chunk['tok'].map(nltk.pos_tag)
        chunk['tok'] = [' '.join([tup[1] for tup in sentence]) for sentence in chunk['tok']]
        chunk['tok'] = list(vectorizer.transform(chunk['tok']).toarray())
        chunk['pred'] = clf.predict(chunk['tok'].tolist())
        chunk = chunk.drop('tok', axis = 1)
        chunk = chunk.dropna()
        data = pd.concat([data, chunk])
        
#%%

data.to_csv('../data/txtgen/preds.csv', mode = 'a', columns = ['text', 'pred'], index = False, header = False)

#%% Wikipedia

data = pd.DataFrame()

with pd.read_csv('../data/txtgen/add_wikipedia.csv', usecols = ['caption'], 
                 chunksize = 10 ** 4) as reader:
    for chunk in reader:
        chunk = chunk.rename(columns = {'caption': 'text'})
        chunk['tok'] = chunk['text'].map(nltk.word_tokenize)
        chunk['tok'] = chunk['tok'].map(nltk.pos_tag)
        chunk['tok'] = [' '.join([tup[1] for tup in sentence]) for sentence in chunk['tok']]
        chunk['tok'] = list(vectorizer.transform(chunk['tok']).toarray())
        chunk['pred'] = clf.predict(chunk['tok'].tolist())
        chunk = chunk.drop('tok', axis = 1)
        chunk = chunk.dropna()
        data = pd.concat([data, chunk])

#%%

data.to_csv('../data/txtgen/preds.csv', mode = 'a', columns = ['text', 'pred'], index = False, header = False)