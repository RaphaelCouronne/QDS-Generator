import pandas as pd
import os
from torchtext.data import TabularDataset
from torchtext import data


print("Let's go !")

# Parameters
BATCH_SIZE = 4000
max_size_vocab = 10000
use_cuda = True
train_ratio = 0.9
n_songs_max = 5100 # max 5100

# Load Data
df = pd.read_csv(os.path.join("data","pop",'billboard_lyrics_1964-2015.csv'), encoding='latin-1')
corpus = df.Lyrics.dropna()
df = df.rename({"Lyrics" : "text"}, axis=1)

# Tiny df
df = df.iloc[:n_songs_max,:]

df.to_csv(os.path.join("data","pop",'billboard.csv'))



#%% With multiple examples


# Convert Data Using Bag of Word
from torchtext.data import Field
import torch
tokenize = lambda x: x.split()
if use_cuda:
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, use_vocab=True)
else:
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, use_vocab=True)


LABEL = Field(sequential=False, use_vocab=False) # TODO understand use_vocab



# Fields
datafields = [("id", LABEL),
                  ("Rank", None),
                  ("Song", None),
                  ("Artist", None),
                  ("Year", None),
                  ("text", TEXT),
                  ("Source", None)]

# For multiple examles
music_train = TabularDataset(
           path="data/pop/billboard.csv", # the file path
           format='csv',
           skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as torchtext-tuto!
           fields=datafields)

train_dataset, test_dataset = music_train.split(split_ratio=train_ratio)


#%% With a Corpus

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=1-train_ratio, random_state=42)

# Corpus dataset
corpus_train = " ".join(train.dropna()['text'].values)
corpus_test = " ".join(test.dropna()['text'].values)
df_corpus_train = pd.DataFrame({"text":corpus_train}, index=[0])
df_corpus_test = pd.DataFrame({"text":corpus_test}, index=[0])
df_corpus_train.to_csv("data/pop/billboard_corpus_train.csv")
df_corpus_test.to_csv("data/pop/billboard_corpus_test.csv")

datafields = [("id", LABEL),
                  ("text", TEXT)]

train_dataset, test_dataset = data.TabularDataset.splits(
    path='./data/pop/', format='csv',
    train='billboard_corpus_train.csv', validation='billboard_corpus_test.csv',
    fields=datafields, skip_header=True)


#%%

# Build Vocab
TEXT.build_vocab(train_dataset, vectors="glove.6B.200d", max_size=max_size_vocab)

# Create Iteratir
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


#%% Model
from model.language_modeling import RNNModel

# Data Embedding
weight_matrix = TEXT.vocab.vectors
n_words_vocab = weight_matrix.size(0) # Number of tokens
embedding_dimension = weight_matrix.size(1)

# Model hyperparameters
n_hidden = 200
n_layers = 1

# Instanciate Model
model = RNNModel(n_words_vocab,
                 embedding_dimension, n_hidden, n_layers, BATCH_SIZE, use_cuda=use_cuda)

# Initialize model
# TODO : I think encoding is initialized at W2V, and possibly modified during training ?
model.encoder.weight.data.copy_(weight_matrix)
if use_cuda:
    model.cuda()



#%% Run training loop

import torch.nn as nn
import torch.optim as optim

def print_batches(TEXT, BATCH_SIZE, text, model):
    for i, batch_chosen in enumerate(range(BATCH_SIZE)):
        x_in = " ".join([TEXT.vocab.itos[i] for i in text[:, batch_chosen]])
        prediction = model(text)
        _, indices_max = prediction.max(dim=2)
        x_out = " ".join([TEXT.vocab.itos[i] for i in indices_max[:, batch_chosen]])
        print("Example : Batch ",batch_chosen)
        print(x_in)
        print(x_out)
        print(" ")

        if i>10:
            break


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.7, 0.99))
n_tokens = weight_matrix.size(0)

train_losses = []
test_losses = []



n_epochs = 10

for epoch in range(n_epochs):
    print("EPOCH ", epoch)
    epoch_loss = 0
    loss = 0
    for n_iter, batch in enumerate(train_iterator):
        print("coucou train", n_iter)
        # reset the hidden state or else the model will try to backpropagate to the
        # beginning of the dataset, requiring lots of time and a lot of memory
        model.reset_history()

        optimizer.zero_grad()

        text, targets = batch.text, batch.target
        print(text.shape)


        if use_cuda:
            text = text.cuda()


        #text, targets = batch.text[:-1], batch.text[1:]
        try:
            prediction = model(text)

            # pytorch currently only supports cross entropy loss for inputs of 2 or 4 dimensions.
            # we therefore flatten the predictions out across the batch axis so that it becomes
            # shape (batch_size * sequence_length, n_tokens)
            # in accordance to this, we reshape the targets to be
            # shape (batch_size * sequence_length)
            loss = criterion(prediction.cpu().view(-1, n_tokens), targets.view(-1))
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


        if use_cuda:
            text = text.cuda()

        try:
            prediction = model(text)
            loss = criterion(prediction.cpu().view(-1, n_tokens), targets.view(-1))
            test_loss += loss.data * text.size(0)
        except:
            print("Problem with ID", n_iter)
    test_loss /= len(test_dataset.examples[0].text)

    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, test_loss))

    print_batches(TEXT, BATCH_SIZE, text, model)

    train_losses.append(float(test_loss.cpu().detach().numpy()))
    test_losses.append(float(epoch_loss.cpu().detach().numpy()))




