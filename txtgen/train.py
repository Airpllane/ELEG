import torch
import torch.nn as nn
import random
import time

from tqdm import tqdm

from utils import all_characters, n_characters, read_file, char_tensor, time_since
from model import CharRNN
from generate import generate

#%%

file, file_len = read_file('../data/txtgen/present_sentences.txt')

#%%

chunk_len = 200
n_epochs = 200
batch_size = 128
print_every = 20
cuda = True

#%%

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    #inp = Variable(inp)
    #target = Variable(target)
    if cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = model.init_hidden(batch_size)
    if cuda:
        hidden = hidden.cuda()
    model.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = model(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    optimizer.step()

    return loss.item() / chunk_len

def save(label):
    save_filename = f'CharRNN-{label}.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)
    return

#%%

model = CharRNN(n_characters, 512, n_characters, model_type='gru', n_layers = 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

model = torch.load('CharRNN-512-present(1.217).pt')

if cuda:
    model.cuda()

#%%

start = time.time()
all_losses = []
loss_avg = 0

#%%

try:
    print("Training for %d epochs..." % n_epochs)
    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train(*random_training_set(chunk_len, batch_size))
        loss_avg += loss

        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
            print(generate(model, '. ', 200, cuda = cuda), '\n')

    print("Saving...")
    save('512-present(' + str(loss)[:5] + ')')

except KeyboardInterrupt:
    print("Saving before quit...")
    save('512-present(' + str(loss)[:5] + ')')