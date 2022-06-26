#отключение предупреждений
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd #используется для работы с данными
import numpy as np #используется для работы с данными
import seaborn as sns #визуализация результатов
import matplotlib.pyplot as plt #визуализация результатов
import pickle #для загрузки объектов

from sklearn import metrics #для рассчета метрик
from sklearn.model_selection import train_test_split, GridSearchCV #разделение данных на train и test, поиск оптимальных параметров для классификатора
from sklearn.pipeline import Pipeline #используется для поиска оптимальных параметров для классификатора

import xgboost as xgb #классификация с помощью алгоритма XGBoost

import nltk #содержит функции для обработки английского языка
#загрузка файлов, необходимых для работы nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

#%%
#загрузка данных, полученных в процессе предобработки
tags_and_type = pd.read_csv('../data/txtgen/tnt.csv')

with open('vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)
    
#%%
#преобразование списка частей речи в матрицу чисел
Xfs = vectorizer.transform(tags_and_type['as_tags']).toarray()
yfs = tags_and_type['type']
yfs_with_numbers = np.copy(yfs)

#преобразование названий классов в целые числа
for i in range(len(yfs_with_numbers)):
  if (yfs_with_numbers[i] == 'present simple'):
    yfs_with_numbers[i] = 0
  if (yfs_with_numbers[i] == 'present continuous'):
    yfs_with_numbers[i] = 1
  if (yfs_with_numbers[i] == 'past simple'):
    yfs_with_numbers[i] = 2
  if (yfs_with_numbers[i] == 'future simple'):
    yfs_with_numbers[i] = 3
  if (yfs_with_numbers[i] == 'other'):
    yfs_with_numbers[i] = 4
 
#разделение данных на train и test    
train_X, test_X, train_y, test_y = train_test_split(Xfs, yfs_with_numbers, test_size=0.3, stratify=yfs_with_numbers, random_state=0)

#создание классификатора
clf = xgb.XGBClassifier(random_state = 42, seed = 2, colsample_bytree = 0.6, subsample = 0.7, verbosity = 0)

#%%
#поиск лучших параметров с помощью GridSearch
pipe = Pipeline([('clf', clf)])

#определение предполагаемых параметров
param_grid = {
     'clf__n_estimators': [300],
}

grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid, 
                          cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)

grid_search.fit(train_X, train_y.astype(int))

#сохранение лучшего классификатора
clf = grid_search.best_estimator_

#сохранение модели в файл
#clf._final_estimator.save_model('XGB.json')

#%%

#получение результатов модели на тестовых данных
preds = clf.predict(test_X)

def print_stats(preds, target, labels):
    '''
    Вывод метрик.
    Parameters
    ----------
    preds : ndarray
        Правильные ответы.
    target : ndarray
        Результаты модели на тестовых данных.
    labels : list
        Названия классов.
    
    Returns
    -------
    None.
    '''
    print('Accuracy = %.3f' % metrics.accuracy_score(target, preds))
    print('-'*40)
    print('Classification report:')
    print(metrics.classification_report(target, preds))
    print('-'*40)
    print('Confusion matrix')
    cm = metrics.confusion_matrix(target, preds, labels = labels)
    cm = cm / np.sum(cm, axis=1)[:,None]
    sns.set(rc={'figure.figsize':(5,4)})
    sns.heatmap(cm, 
        xticklabels=['present simple', 'present continuous', 'past simple', 'future simple', 'other'],
        yticklabels=['present simple', 'present continuous', 'past simple', 'future simple', 'other'],
           annot=True, cmap = 'YlGnBu')
    plt.pause(0.05)
    
#вывод метрик
print_stats(test_y.astype(int), preds.astype(int), clf.classes_)

#%%
#тестирование модели на произвольных предложениях
input_variable = pd.Series(["I walk my dog", "I am going there", "Tom was perplexed", "I will go"])
input_variable = input_variable.map(nltk.word_tokenize)
input_variable = input_variable.map(nltk.pos_tag)
input_variable = [' '.join([tup[1] for tup in sentence]) for sentence in input_variable]
input_variable = vectorizer.transform(input_variable)
print(clf.predict(input_variable))