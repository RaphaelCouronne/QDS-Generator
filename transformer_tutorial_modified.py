#%% Models

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.language_modeling import TransformerModel


#%% Load and Batch Data

import torchtext
from torchtext.data.utils import get_tokenizer



#%%

import os
import pandas as pd
from torchtext import data
from torchtext.data import Field
#tokenize = lambda x: x.split()
tokenize = get_tokenizer("basic_english")

# Parameters
BATCH_SIZE = 50
max_size_vocab = 10000
use_cuda = False
train_ratio = 0.9
n_songs_max = 5100 # max 5100

# Load Data
df = pd.read_csv(os.path.join("data","pop",'billboard_lyrics_1964-2015.csv'), encoding='latin-1')
corpus = df.Lyrics.dropna()
df = df.rename({"Lyrics" : "text"}, axis=1)

# Tiny df
df = df.iloc[:n_songs_max,:]

df.to_csv(os.path.join("data","pop",'billboard.csv'))


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=1/10, random_state=42)
train, val = train_test_split(train, test_size=1/9, random_state=42)

# Corpus dataset
corpus_train = " ".join(train.dropna()['text'].values)
corpus_val = " ".join(val.dropna()['text'].values)
corpus_test = " ".join(test.dropna()['text'].values)
df_corpus_train = pd.DataFrame({"text":corpus_train}, index=[0])
df_corpus_val = pd.DataFrame({"text":corpus_val}, index=[0])
df_corpus_test = pd.DataFrame({"text":corpus_test}, index=[0])
df_corpus_train.to_csv("data/pop/billboard_corpus_train.csv")
df_corpus_val.to_csv("data/pop/billboard_corpus_val.csv")
df_corpus_test.to_csv("data/pop/billboard_corpus_test.csv")

LABEL = Field(sequential=False, use_vocab=False) # TODO understand use_vocab
TEXT = Field(sequential=True, tokenize=tokenize, lower=True, use_vocab=True)

datafields = [("id", LABEL),("text", TEXT)]

train_txt, val_txt, test_txt = data.TabularDataset.splits(
    path='./data/pop/', format='csv',
    train='billboard_corpus_train.csv', validation='billboard_corpus_val.csv',
    test='billboard_corpus_test.csv',
    fields=datafields, skip_header=True)

TEXT.build_vocab(train_txt)





#%%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)


#%%

bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


#%% Initiate an instance


from model.language_modeling import TransformerModel

ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)


#%% Run the model

criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

#%% Loop

best_val_loss = float("inf")
epochs = 10 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

    # Print results given
    for batch_num in range(10):
        data, targets = get_batch(train_data, batch_num)
        optimizer.zero_grad()
        output = model(data)

        val, idx_max = output.max(dim=2)

        print(" ")
        text_in = " ".join([TEXT.vocab.itos[tok] for tok in data[:, batch_num]])
        text_out = " ".join([TEXT.vocab.itos[tok] for tok in idx_max[:, batch_num]])
        print(" ")

        print(text_in)
        print(text_out)

#%% Save the best model

torch.save(best_model.state_dict(), "data/pop/models/transformers.p")


#%% Evalutate model

test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


#%% Print what is shows

# Print results given
for batch_num in range(10):
    data, targets = get_batch(train_data, batch_num)
    optimizer.zero_grad()
    output = model(data)

    val, idx_max = output.max(dim=2)

    text_in = " ".join([TEXT.vocab.itos[tok] for tok in data[:, batch_num]])
    text_out = " ".join([TEXT.vocab.itos[tok] for tok in idx_max[:, batch_num]])

    print(" ")
    print(text_in)
    print(text_out)
    print(" ")

#%%

import numpy as np

num_batch = 10
words_sequence_ogigin = get_batch(train_data, 7)[0][:, num_batch].reshape(-1,1).cpu().reshape(-1).tolist()
words_sequence_aigenerated = []

temp = get_batch(train_data, 7)[0][:, num_batch].reshape(-1,1)

for i in range(100):

    output = model(temp)
    val, idx_max = output.max(dim=2)

    last_word = idx_max[-1,0]
    words_sequence_aigenerated.append(output[-1,0].cpu().detach().numpy().argsort()[-1])

    temp = idx_max

text_in = " ".join([TEXT.vocab.itos[tok] for tok in words_sequence_ogigin])
text_out = " ".join([TEXT.vocab.itos[tok] for tok in words_sequence_aigenerated])

print(text_in,"||| after that, generated via AI |||",text_out)

