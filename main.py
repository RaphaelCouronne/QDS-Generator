import pandas as pd
import os
from torchtext.data import TabularDataset
from torchtext import data


# Parameters
BATCH_SIZE = 10

# Load Data
df = pd.read_csv(os.path.join("data","pop",'billboard_lyrics_1964-2015.csv'), encoding='latin-1')
corpus = df.Lyrics.dropna()
df=df.rename({"Lyrics":"text"}, axis=1)

# Tiny df
df = df.iloc[:1000,:]

df.to_csv(os.path.join("data","pop",'billboard.csv'))



# Convert Data Using Bag of Word
from torchtext.data import Field
tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True, use_vocab=True)
LABEL = Field(sequential=False, use_vocab=False) # TODO understand use_vocab


datafields = [("id", LABEL),
                  ("Rank", None),
                  ("Song", None),
                  ("Artist", None),
                  ("Year", None),
                  ("text", TEXT),
                  ("Source", None)]

#%% Examples

# For multiple examles
music_train = TabularDataset(
           path="data/pop/billboard.csv", # the file path
           format='csv',
           skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as torchtext-tuto!
           fields=datafields)
train_dataset, test_dataset = music_train.split(split_ratio=0.8)


#%% Corpus
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.8, random_state=42)

# Corpus dataset
corpus_train = " ".join(train.dropna()['text'].values)
corpus_test = " ".join(test.dropna()['text'].values)
df_corpus_train = pd.DataFrame({"corpus":corpus_train}, index=[0])
df_corpus_test = pd.DataFrame({"corpus":corpus_test}, index=[0])
df_corpus_train.to_csv("data/pop/billboard_corpus_train.csv")
df_corpus_test.to_csv("data/pop/billboard_corpus_test.csv")

train_dataset, test_dataset = data.TabularDataset.splits(
    path='./data/pop/', format='csv',
    train='billboard_corpus_train.csv', validation='billboard_corpus_test.csv',
    fields=datafields, skip_header=True)


#%%

#Build Vocab
TEXT.build_vocab(music_train, vectors="glove.6B.200d")
#TEXT.build_vocab(music_train)

# Iterator
from torchtext.data import Iterator, BPTTIterator, BucketIterator

#train_iter = BPTTIterator(music_train, batch_size=10, bptt_len=20)
#train_iter = Iterator(music_train, batch_size=10)

train_iterator, test_iterator = data.BPTTIterator.splits(datasets=(train_dataset, test_dataset), # specify train and validation Tabulardataset
                                            batch_sizes=(BATCH_SIZE,BATCH_SIZE),  # batch size of train and validation
                                            sort_key=lambda x: len(x.text), # on what attribute the text should be sorted
                                            device=None, # -1 mean cpu and 0 or None mean gpu
                                            sort_within_batch=True,
                                            repeat=False,
                                            bptt_len=20)

"""
train_iterator, test_iterator = BucketIterator.splits(datasets=(train_dataset, test_dataset), # specify train and validation Tabulardataset
                                            batch_sizes=(BATCH_SIZE,BATCH_SIZE),  # batch size of train and validation
                                            sort_key=lambda x: len(x.text), # on what attribute the text should be sorted
                                            device=None, # -1 mean cpu and 0 or None mean gpu
                                            sort_within_batch=True,
                                            repeat=False)"""


# Show examples
print(TEXT.vocab.freqs.most_common(10))

#for i in range(10):
#    print(train_iterator.dataset.examples[i].Artist,
#          train_iterator.dataset.examples[i].Song,
#          train_iterator.dataset.examples[i].text,
#          "\n")




#%% Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp,
                 nhid, nlayers, bsz,
                 dropout=0.5, tie_weights=True):
        super(RNNModel, self).__init__()
        self.nhid, self.nlayers, self.bsz = nhid, nlayers, bsz
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.hidden = self.init_hidden(bsz)  # the input is a batched consecutive corpus
        # therefore, we retain the hidden state across batches

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        emb = self.drop(self.encoder(input))
        output, self.hidden = self.rnn(emb, self.hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        #return (V(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda()),
        #        V(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda())

        return (V(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                V(weight.new(self.nlayers, bsz, self.nhid).zero_()))

    def reset_history(self):
        self.hidden = tuple(V(v.data) for v in self.hidden)

#%% Instanciate model



weight_matrix = TEXT.vocab.vectors
model = RNNModel(weight_matrix.size(0),
                 weight_matrix.size(1), 200, 1, BATCH_SIZE)

model.encoder.weight.data.copy_(weight_matrix)



#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.7, 0.99))
n_tokens = weight_matrix.size(0)

n_epochs = 10

from tqdm import tqdm

for epoch in range(n_epochs):
    print("EPOCH ", epoch)
    epoch_loss = 0
    for n_iter, batch in enumerate(train_iterator):
        print("coucou train", n_iter)
        # reset the hidden state or else the model will try to backpropagate to the
        # beginning of the dataset, requiring lots of time and a lot of memory
        model.reset_history()

        optimizer.zero_grad()

        text, targets = batch.text, batch.target
        #text, targets = batch.text[:-1], batch.text[1:]
        try:
            prediction = model(text)

            # pytorch currently only supports cross entropy loss for inputs of 2 or 4 dimensions.
            # we therefore flatten the predictions out across the batch axis so that it becomes
            # shape (batch_size * sequence_length, n_tokens)
            # in accordance to this, we reshape the targets to be
            # shape (batch_size * sequence_length)
            loss = criterion(prediction.view(-1, n_tokens), targets.view(-1))
            loss.backward()

            optimizer.step()
        except:
            print("Problem with ID", n_iter)

    epoch_loss += loss.data * prediction.size(0) * prediction.size(1)

    epoch_loss /= len(music_train.examples[0].text)

    # monitor the loss
    test_loss = 0
    model.eval()
    for n_iter, batch in enumerate(test_iterator):
        print("coucou test", n_iter)
        model.reset_history()
        text, targets = batch.text, batch.target
        #text, targets = batch.text[:-1], batch.text[1:]
        try:
            prediction = model(text)
            loss = criterion(prediction.view(-1, n_tokens), targets.view(-1))
            test_loss += loss.data * text.size(0)
        except:
            print("Problem with ID", n_iter)
    test_loss /= len(test_dataset.examples[0].text)

    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, test_loss))



#%% What does the model in the end ????

batch_chosen = 2

x_in = " ".join([TEXT.vocab.itos[i] for i in text[:,batch_chosen]])
prediction = model(text)
_, indices_max = prediction.max(dim=2)
x_out = " ".join([TEXT.vocab.itos[i] for i in indices_max[:, batch_chosen]])


print(x_in)
print(x_out)


#%%

for n_iter, batch in enumerate(train_iterator):
    print(n_iter)