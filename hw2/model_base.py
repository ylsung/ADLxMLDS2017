import torch
import torch.nn as nn
from torch.autograd import Variable

def one_hot(label, vocab_size):
    label = label.cpu().data
    label = label.view(-1, 1)
    one_hot_tensor = torch.FloatTensor(label.size(0), vocab_size).zero_()

    one_hot_tensor.scatter_(dim=1, index=label.cpu(), value=1.0)
    if torch.cuda.is_available():
        one_hot_tensor = one_hot_tensor.cuda()


    return Variable(one_hot_tensor)


class RNNEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, dropout, direction):
        super(RNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.direction = direction
        if direction == 1:
            bidirectional = False
        elif direction == 2:
            bidirectional = True

        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
            )
        self.relu = nn.ReLU()
    def forward(self, inputs, h, c):
        output, (h, c) = self.lstm(inputs, (h, c))
        return output, h, c

    def init_hidden(self, batch):
        h = torch.zeros(self.num_layers * self.direction, batch, self.hidden_size)
        c = torch.zeros(self.num_layers * self.direction, batch, self.hidden_size)
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        h_v, c_v = Variable(h), Variable(c)
        return h_v, c_v

class attnRNNDecoder(nn.Module):
    def __init__(self, embed_size, frame_size, hidden_size, num_layers, dropout, vocab_size, direction):
        super(attnRNNDecoder, self).__init__()

        self.direction = direction
        if direction == 1:
            bidirectional = False
        elif direction == 2:
            bidirectional = True

        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            # nn.Tanh(),
            nn.Dropout(dropout),
            )

        self.e_dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        # self.input_preprocessing = nn.Linear(feature_size + embed_size, input_size)
        # self.down_sample = nn.Linear(feature_size, embed_size)
        # embed_size = vocab_size

        input_size = hidden_size * num_layers * self.direction + embed_size
        self.attn_prob = nn.Sequential(
            nn.Linear(input_size, frame_size),
            # nn.Dropout(dropout),
            nn.Softmax(),
            )
        lstm_input_size = hidden_size * self.direction + embed_size

        self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(lstm_input_size)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
            )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * self.direction, vocab_size),
            nn.LogSoftmax(),
            )
    def forward(self, pre_word, encoder_output, h, c):

        embed = self.embed(pre_word)
        # embed = one_hot(pre_word, self.vocab_size)

        attn_input = torch.cat((embed, h.permute(1, 0, 2).contiguous().view(embed.size(0), -1)), dim=1)
        # attn_input = self.bn1(attn_input)
        attn_weight = self.attn_prob(attn_input).unsqueeze(1)

        encoder_attn = torch.bmm(attn_weight, encoder_output)
        # print(embed)
        # print(encoder_attn)

        real_input = torch.cat((embed, encoder_attn.view(embed.size(0), -1)), dim=1)
        # real_input = self.bn2(real_input)

        real_input = real_input.unsqueeze(1)
        # real_input = self.e_dropout(real_input)
        # output = output.unsqueeze(1)
        output, (h, c) = self.lstm(real_input, (h, c)) 
        output = output.view(output.size(0), -1)
        output = self.output_layer(output)
        return output, h, c, attn_weight
    def init_hidden(self, batch):
        h = torch.zeros(self.num_layers * self.direction, batch, self.hidden_size)
        c = torch.zeros(self.num_layers * self.direction, batch, self.hidden_size)
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        h_v, c_v = Variable(h), Variable(c)
        return h_v, c_v

class ReLURNNEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, dropout, direction):
        super(ReLURNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.direction = direction
        if direction == 1:
            bidirectional = False
        elif direction == 2:
            bidirectional = True

        self.projection = nn.Sequential(
            nn.Linear(feature_size, hidden_size*self.direction),
            # nn.ReLU(),
            )
        self.lstm = nn.GRU(
            input_size=hidden_size*self.direction,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
            )
        self.relu = nn.ReLU()
    def forward(self, inputs, h, c):
        output = self.projection(inputs)
        for i in range(self.num_layers):
            output = self.relu(output)
            output, h = self.lstm(output, h)
        return output, h, c

    def init_hidden(self, batch):
        h = torch.zeros(self.direction, batch, self.hidden_size)
        c = torch.zeros(self.direction, batch, self.hidden_size)
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        h_v, c_v = Variable(h), Variable(c)
        return h_v, c_v

class ReLUattnRNNDecoder(nn.Module):
    def __init__(self, embed_size, frame_size, hidden_size, num_layers, dropout, vocab_size, direction):
        super(ReLUattnRNNDecoder, self).__init__()

        self.direction = direction
        if direction == 1:
            bidirectional = False
        elif direction == 2:
            bidirectional = True

        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            nn.Dropout(dropout),
            )
        # self.e_dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.input_preprocessing = nn.Linear(feature_size + embed_size, input_size)
        # self.down_sample = nn.Linear(feature_size, embed_size)
        input_size = hidden_size * 1 * self.direction + embed_size
        self.attn_prob = nn.Sequential(
            nn.Linear(input_size, frame_size),
            # nn.Dropout(dropout),
            nn.Softmax(),
            )
        lstm_input_size = hidden_size * self.direction + embed_size

        self.projection = nn.Sequential(
            nn.Linear(lstm_input_size, hidden_size*self.direction),
            # nn.ReLU(),
            )
        self.lstm = nn.GRU(
            input_size=hidden_size*self.direction,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
            )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * self.direction, vocab_size),
            nn.LogSoftmax(),
            )
        self.relu = nn.ReLU()
    def forward(self, pre_word, encoder_output, h, c):

        embed = self.embed(pre_word)

        attn_input = torch.cat((embed, h.permute(1, 0, 2).contiguous().view(embed.size(0), -1)), dim=1)
        attn_weight = self.attn_prob(attn_input).unsqueeze(1)

        encoder_attn = torch.bmm(attn_weight, encoder_output)

        real_input = torch.cat((embed.unsqueeze(1), encoder_attn), dim=2)
        # real_input = self.e_dropout(real_input)
        # output = output.unsqueeze(1)
        output = self.projection(real_input)
        for i in range(self.num_layers):
            # output = self.relu(output)
            output, h = self.lstm(output, h) 

        output = output.view(output.size(0), -1)
        output = self.output_layer(output)
        return output, h, c, attn_weight
    def init_hidden(self, batch):
        h = torch.zeros(self.direction, batch, self.hidden_size)
        c = torch.zeros(self.direction, batch, self.hidden_size)
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        h_v, c_v = Variable(h), Variable(c)
        return h_v, c_v


