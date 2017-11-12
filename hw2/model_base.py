import torch
import torch.nn as nn
from torch.autograd import Variable

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
            nn.Dropout(dropout),
            )
        # self.e_dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.input_preprocessing = nn.Linear(feature_size + embed_size, input_size)
        # self.down_sample = nn.Linear(feature_size, embed_size)
        input_size = hidden_size * num_layers * self.direction + embed_size
        self.attn_prob = nn.Sequential(
            nn.Linear(input_size, frame_size),
            # nn.Dropout(dropout),
            nn.Softmax(),
            )
        lstm_input_size = hidden_size * self.direction + embed_size
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

        attn_input = torch.cat((embed, h.permute(1, 0, 2).contiguous().view(embed.size(0), -1)), dim=1)
        attn_weight = self.attn_prob(attn_input).unsqueeze(1)

        encoder_attn = torch.bmm(attn_weight, encoder_output)

        real_input = torch.cat((embed.unsqueeze(1), encoder_attn), dim=2)
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

class Decoder(nn.Module):
    def __init__(self, embed_size, feature_size, hidden_size, num_layers, dropout, vocab_size):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.e_dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        input_size = feature_size + embed_size

        # self.input_preprocessing = nn.Linear(feature_size + embed_size, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
            )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size*2, vocab_size),
            nn.LogSoftmax(),
            )
    def forward(self, pre_word, image_feature, h, c):

        output = self.embed(pre_word)

        # output = self.e_dropout(output)
        image_feature = image_feature.view(image_feature.size(0), -1)

        output = torch.cat((output, image_feature), dim=1)

        output = output.unsqueeze(1)
        output, (h, c) = self.lstm(output, (h, c)) 
        output = output.view(output.size(0), -1)
        output = self.output_layer(output)
        return output, h, c
    def init_hidden(self, batch):
        h = torch.zeros(self.num_layers * 2, batch, self.hidden_size)
        c = torch.zeros(self.num_layers * 2, batch, self.hidden_size)
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        h_v, c_v = Variable(h), Variable(c)
        return h_v, c_v

class attnDecoder(nn.Module):
    def __init__(self, embed_size, frame_size, feature_size, hidden_size, num_layers, dropout, vocab_size):
        super(attnDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.e_dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.input_preprocessing = nn.Linear(feature_size + embed_size, input_size)
        # self.down_sample = nn.Linear(feature_size, embed_size)
        input_size = feature_size + embed_size
        self.attn_prob = nn.Sequential(
            nn.Linear(input_size, frame_size),
            nn.Dropout(dropout),
            nn.Softmax(),
            )
        lstm_input_size = feature_size // frame_size + embed_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
            )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size*2, vocab_size),
            nn.LogSoftmax(),
            )
    def forward(self, pre_word, image_feature, h, c):

        embed = self.embed(pre_word)

        embed = self.e_dropout(embed)
        # image_feature = nn.down_sample(image_feature)

        # image_feature = image_feature.view(image_feature.size(0), -1)

        attn_input = torch.cat((embed, image_feature.view(image_feature.size(0), -1)), dim=1)
        attn_weight = self.attn_prob(attn_input).unsqueeze(1)

        image_feature_attn = torch.bmm(attn_weight, image_feature)

        real_input = torch.cat((embed.unsqueeze(1), image_feature_attn), dim=2)
        real_input = self.e_dropout(real_input)
        # output = output.unsqueeze(1)
        output, (h, c) = self.lstm(real_input, (h, c)) 
        output = output.view(output.size(0), -1)
        output = self.output_layer(output)
        return output, h, c
    def init_hidden(self, batch):
        h = torch.zeros(self.num_layers * 2, batch, self.hidden_size)
        c = torch.zeros(self.num_layers * 2, batch, self.hidden_size)
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        h_v, c_v = Variable(h), Variable(c)
        return h_v, c_v

