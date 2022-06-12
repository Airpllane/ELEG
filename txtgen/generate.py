import torch
import os

from utils import all_characters, n_characters, read_file, char_tensor, time_since
from model import CharRNN

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

if __name__ == '__main__':
    model = torch.load('CharRNN-test.pt').cuda()
    print(generate(model, cuda = True))