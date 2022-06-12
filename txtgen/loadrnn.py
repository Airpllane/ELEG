import torch
import nltk

import random

from utils import all_characters, char_tensor
#from model import CharRNN

cuda = False

#%%

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

import pattern.en as pt

#%%

def generate(model, prime_str = 'A', predict_len = 200, temperature = 0.4, cuda = False):
    hidden = model.init_hidden(1)
    prime_input = char_tensor(prime_str).unsqueeze(0)

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = model(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char).unsqueeze(0)
        if cuda:
            inp = inp.cuda()

    return predicted

#%%

model_present = torch.load('CharRNN-1024x3-presentw_1_398.pt', map_location='cpu')
model_past = torch.load('CharRNN-1024x3-pastw_1_312.pt', map_location='cpu')
#model = torch.load('CharRNN-512x3-presentwl_1_343.pt', map_location='cpu')
#model = torch.load('CharRNN-512x3-pastwl_1_322.pt').cpu()

model = model_present
if cuda:
    model = model.cuda()

#%%

def generate_sentence(temperature, cuda = False):
    prime_str = '. '
    prime_input = char_tensor('. ').unsqueeze(0)
    hidden = model.init_hidden(1)
    
    gen_text = []
    
    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[:,p], hidden)
    inp = prime_input[:,-1]
    predicted = prime_str
        
    for i in range(200):
        output, hidden = model(inp, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        
        '''
        output_dist[75] *= i * 5 #.
        output_dist[82] *= i * 5 #?
        output_dist[62] *= i * 5 #!
        '''
        
        top_i = torch.multinomial(output_dist, 1)[0]
        
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char).unsqueeze(0)
        if cuda:
            inp = inp.cuda()
        if(predicted_char in ['.', '?', '!']):
            break
    
    return predicted[2:]

def generate_exercise(temperature):
    tgpos = ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    
    stc = generate_sentence(temperature)
    tgs = nltk.pos_tag(nltk.word_tokenize(stc))
    tgs = [[i[0], i[1]] for i in tgs]
    result = ''
    opt = []
    try:
        tgw = random.choice([i for i, pos in enumerate(tgs) if (pos[1] in tgpos)])
    except IndexError:
        print('Target POS not found in: ' + str(stc))
        return False, False
    for i, tg in enumerate(tgs):
        if i == tgw:
            result += ' '
            result += '__?__'
            try:
                opts = pt.lexeme(tg[0])
            except StopIteration:
                print(tg)
                return False, False
            if tg[0] in opts:
                opts.remove(tg[0])
            opt = [tg[0], random.sample(opts, min(2, len(opts)))]
        elif (tg[1] in ['.', '?', '!', ',', ':'] or "'" in tg[0]):
            result += tg[0]    
        else:
            result += ' '
            result += tg[0]
    result = result[1:]
    return result, opt