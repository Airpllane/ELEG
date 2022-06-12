
import pandas as pd
import nltk
import pickle

from sklearn.feature_extraction.text import CountVectorizer

#%%

def sentences_to_tags(sentences, stype):
    """
    Преобразование предложений в список, состоящий из частей речи.
    
    Parameters
    ----------
    sentences : pd.DataFrame
        Набор данных, содержащий прдложения.
    stype : str
        Время предложений.
    Returns
    -------
    stc : pd.DataFrame
        Набор данных, содержащий список частей речи и время.
    """
    stc_tokd = sentences['text'].map(nltk.word_tokenize)
    stc_tagd = stc_tokd.map(nltk.pos_tag)
    stc_as_tags = [[tup[1] for tup in sentence] for sentence in stc_tagd]
    stc = pd.DataFrame()
    stc['as_tags'] = stc_as_tags
    stc['type'] = stype
    return stc

def get_convert_and_tag(data, tags, stype):
    '''
    Отбор и обработка предложений по тегу.
    Parameters
    ----------
    data : pd.DataFrame
        Набор исходных данных.
    tags : pd.DataFrame
        Набор исходных тегов.
    stype : str
        Время предложений.
    Returns
    -------
    stc : pd.DataFrame
        Набор данных, содержащий список частей речи и время.
    '''
    stc_tags = tags[tags['tag'] == stype]
    stc = data[data['id'].isin(stc_tags['id'])]
    stc = sentences_to_tags(stc, stype)
    return stc

#%%
#загрузка данных из файлов
data = pd.read_csv('../data/txtgen/eng_sentences.tsv', sep = '\t', names = ['id', 'lang', 'text'])
tags = pd.read_csv('../data/txtgen/tags.csv', sep = '\t', names = ['id', 'tag'])

#%%
#определение списка используемых времен
stypes = ['present simple', 'present continuous', 'past simple', 'present perfect', 'present perfect continuous', 'future simple']
#отбор предложений на основе списка времен
tags_and_type = pd.concat([get_convert_and_tag(data, tags, i) for i in stypes])

#%%
#замена тегов редких времен на 'other'
mask = tags_and_type.type == 'present perfect'
column_name = 'type'
tags_and_type.loc[mask, column_name] = 'other'
mask = tags_and_type.type == 'present perfect continuous'
tags_and_type.loc[mask, column_name] = 'other'

#%%
#перевод списка тегов в строку
tags_and_type['as_tags'] = tags_and_type['as_tags'].map(' '.join)

#%%
#сохранение в файл
tags_and_type.to_csv('../data/txtgen/tnt.csv', index = False)
#загрузка из файла
tags_and_type = pd.read_csv('../data/txtgen/tnt.csv')

#%%
#создание и обучение токенайзера
vectorizer = CountVectorizer()
vectorizer.fit(tags_and_type['as_tags'])

#%%
#сохранение токенайзера в файл
with open('vectorizer.pickle', 'wb') as handle:
    pickle.dump(vectorizer, handle)
