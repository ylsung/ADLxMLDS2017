import torch
import torch.nn as nn
from torch.autograd import Variable

class S2VT(nn.Module):
    def __init__(self, embed_size, feature_size, hidden_size, num_layers, dropout, vocab_size):
        super(S2VT, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # embed_size = vocab_size
        direction = 2
        self.direction = direction
        if direction == 1:
            bidirectional = False
        elif direction == 2:
            bidirectional = True

        input_size = hidden_size*self.direction + embed_size
        # self.dropout = nn.Dropout(dropout)
        self.preprocess = nn.Sequential(
            nn.Linear(feature_size, input_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            )
        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            nn.Dropout(dropout),
            )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
            )
        self.output = nn.Sequential(
            nn.Linear(hidden_size*self.direction, vocab_size),
            nn.LogSoftmax(),
            )
    def forward(self, inputs, h, c):
        output, (h, c) = self.lstm(inputs, (h, c))
        return output, h, c
    def encode(self, inputs, h, c):
        output = self.preprocess(inputs)
        output, (h, c) = self.lstm(output, (h, c))
        return output, h, c
    def decode(self, pre_word, encode_output, h, c, padding):
        embed = self.embed(pre_word).unsqueeze(1)

        if padding:
            embed.data.fill_(0.0)
        # embed = pre_word
        # embed = embed.unsqueeze(1)
        encode_output = encode_output.unsqueeze(1)
  
        real_input = torch.cat((embed, encode_output), dim=2)
        output, (h, c) = self.lstm(real_input, (h, c))
        output = self.output(output.view(output.size(0), -1))
        return output, h, c 

    def init_hidden(self, batch):
        h = torch.zeros(self.num_layers*self.direction, batch, self.hidden_size)
        c = torch.zeros(self.num_layers*self.direction, batch, self.hidden_size)
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        h_v, c_v = Variable(h), Variable(c)
        return h_v, c_v