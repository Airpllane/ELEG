import pandas as pd
import nltk

data = pd.DataFrame()

#%% Newspapers

with pd.read_csv('../data/txtgen/old-newspaper.tsv', 
                 sep = '\t', names = ['lang', 'src', 'date', 'text'], 
                 chunksize = 10 ** 4) as reader:
    for chunk in reader:
        chunk = chunk[chunk['lang'] == 'English']
        chunk['text'] = chunk['text'].map(nltk.tokenize.sent_tokenize)
        chunk = chunk.explode('text')
        data = pd.concat([data, chunk])

#%%
data = data.dropna()
data.to_csv('../data/txtgen/add_newspapers.csv', index = False)

#%% Image captions 1

data = pd.read_csv('../data/txtgen/captions.txt', usecols = ['caption'])
data = data.dropna()

#%%

data['caption'] = data['caption'].str.strip()
data['caption'] = data['caption'].str.replace(' .', '.', regex = False)
data['caption'] = data['caption'].str.replace(' ,', ',', regex = False)
data['caption'] = data['caption'].str.replace(' \' ', '\' ', regex = False)
data['caption'] = data['caption'].str.replace(' " ', '"', regex = False)
data['caption'] = data['caption'].str.replace(' )', ')', regex = False)
data['caption'] = data['caption'].str.replace('( ', '(', regex = False)
data['caption'] = data['caption'].str.rstrip('.') + '.'
data['caption'] = data['caption'].apply(lambda x: x[0].upper() + x[1:])

#%%

data.to_csv('../data/txtgen/add_captions8k.csv', index = False)

#%% Image captions 2

data = pd.read_csv('../data/txtgen/captions30k.csv', sep = '|', usecols = [2]).rename(columns = {' comment': 'caption'})
data = data.dropna()
#%%

data['caption'] = data['caption'].str.strip()
data['caption'] = data['caption'].str.replace(' .', '.', regex = False)
data['caption'] = data['caption'].str.replace(' ,', ',', regex = False)
data['caption'] = data['caption'].str.replace(' \'', '\'', regex = False)
data['caption'] = data['caption'].str.replace(' " ', '"', regex = False)
data['caption'] = data['caption'].str.replace(' )', ')', regex = False)
data['caption'] = data['caption'].str.replace('( ', '(', regex = False)
data['caption'] = data['caption'].str.rstrip('.') + '.'
data['caption'] = data['caption'].apply(lambda x: x[0].upper() + x[1:])

#%%

data.to_csv('../data/txtgen/add_captions30k.csv', index = False)

#%% SNLIC

data1 = pd.read_csv('../data/txtgen/snlic/snli_1.0_train.csv', usecols = ['sentence1', 'sentence2'])
data2 = pd.read_csv('../data/txtgen/snlic/snli_1.0_dev.csv', usecols = ['sentence1', 'sentence2'])
data3 = pd.read_csv('../data/txtgen/snlic/snli_1.0_test.csv', usecols = ['sentence1', 'sentence2'])

data = pd.concat([data1, data2, data3])

#%%

data = data.melt().drop('variable', 1)
data = data.rename(columns = {'value': 'caption'})
data = data.drop_duplicates()
data = data.dropna()

#%%

data['caption'] = data['caption'].str.strip()
data['caption'] = data['caption'].str.replace(' .', '.', regex = False)
data['caption'] = data['caption'].str.replace(' ,', ',', regex = False)
data['caption'] = data['caption'].str.replace(' \'', '\'', regex = False)
data['caption'] = data['caption'].str.replace(' " ', '"', regex = False)
data['caption'] = data['caption'].str.replace(' )', ')', regex = False)
data['caption'] = data['caption'].str.replace('( ', '(', regex = False)
data['caption'] = data['caption'].str.rstrip('.') + '.'
data['caption'] = data['caption'].apply(lambda x: x[0].upper() + x[1:])

#%%

data.to_csv('../data/txtgen/add_snlic.csv', index = False)

#%% Wikipedia

with open('../data/txtgen/wikisent2.txt', 'r') as handle:
    data = pd.DataFrame(handle.readlines())
data = data.rename(columns = {0: 'caption'})
data = data.dropna()

#%%

data['caption'] = data['caption'].str.strip()

#%%

data.to_csv('../data/txtgen/add_wikipedia.csv', index = False)