
import pandas as pd

#%%

data = pd.read_csv('../data/txtgen/preds.csv').sample(frac = 1)

print('Total: ' + str(len(data)))

mask_nonAscii = data['text'].map(lambda x: len(x.encode('ascii', errors='ignore')) != len(x))
nasc = data[mask_nonAscii]

print('Non-ASCII: ' + str(len(nasc)))

data = data.drop(data[mask_nonAscii].index)

print('Remaining: ' + str(len(data)))

present = data.loc[data['pred'] == 0]
past = data.loc[data['pred'] == 2]

print('Present: ' + str(len(present)))
print('Past: ' + str(len(past)))

present_sentences = ' '.join(present['text'].astype(str))
past_sentences = ' '.join(past['text'].astype(str))
print('Total chars (present): ' + str(len(present_sentences)))
print('Total chars (past): ' + str(len(past_sentences)))

#%%

with open('../data/txtgen/present_sentences.txt', 'w') as handle:
    handle.write(present_sentences)

with open('../data/txtgen/past_sentences.txt', 'w') as handle:
    handle.write(past_sentences)