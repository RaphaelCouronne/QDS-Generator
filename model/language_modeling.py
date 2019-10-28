
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V


class RNNModel(nn.Module):
    print("Instanciate RNN")
    def __init__(self, ntoken, ninp,
                 nhid, nlayers, bsz,
                 dropout=0.5, tie_weights=True, use_cuda=False):
        super(RNNModel, self).__init__()
        self.nhid, self.nlayers, self.bsz = nhid, nlayers, bsz
        self.use_cuda = use_cuda
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
        if self.use_cuda:
            return (V(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda()),
                    V(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda())
        else:
            return (V(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    V(weight.new(self.nlayers, bsz, self.nhid).zero_()))

    def reset_history(self):
        self.hidden = tuple(V(v.data) for v in self.hidden)