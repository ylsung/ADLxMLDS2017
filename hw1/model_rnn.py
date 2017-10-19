import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable

class rnnModel(nn.Module):
    def __init__(self, params):
        super(rnnModel, self).__init__()

        # declare parameters
        self.save = params['save']
        feature_size = params['feature_size']
        num_layers = params['num_layers']
        hidden_size = params['hidden_size']
        num_output = params['num_output']
        self.num_output = num_output
        self.CUDA = params['CUDA']
        self.batch_size = params['batch_size']
        lr = params['lr']
        self.valid_tuple = params.get('valid')
        self.epoch = params['epoch']
        self.early_stop = params['early_stop']

        input_size = feature_size
        direction = 1
        bidirectional = False
        if direction == 1:
            bidirectional = False
        elif direction == 2:
            bidirectional = True

        # declare model structure    
        # one direction lstm
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional = bidirectional
            )
        self.h = torch.FloatTensor(num_layers*direction, self.batch_size, hidden_size).fill_(0.0)
        self.c = torch.FloatTensor(num_layers*direction, self.batch_size, hidden_size).fill_(0.0)
        self.criterion = nn.NLLLoss()
        self.criterion_all = nn.NLLLoss(size_average=False)
        if self.CUDA:
            self.h = self.h.cuda()
            self.c = self.c.cuda()
            self.criterion = self.criterion.cuda()

        # convert lstm output to label
        self.feature2label = nn.Linear(hidden_size*direction, num_output)
        self.logsoftmax = nn.LogSoftmax()

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.best_state = deepcopy(self.state_dict())

    def forward(self, inputs):
        self.h.fill_(0.0)
        self.c.fill_(0.0)
        h0 = Variable(self.h)
        c0 = Variable(self.c)
        output, hn = self.lstm(inputs, (h0, c0))
        # self.h = hn.data
        # self.c = cn.data
        output = self.feature2label(output)
        output = self.logsoftmax(output)
        return output

    def batch_data_gen(self, _id, _data, _label, _framelength):
        batch_index = 0
        end_epoch = 0
        while True:
            if batch_index + self.batch_size < _id.shape[0]:
                end_epoch = 0
                yield _id[batch_index: batch_index + self.batch_size], _data[batch_index: batch_index + self.batch_size],\
                    _label[batch_index: batch_index + self.batch_size], _framelength[batch_index: batch_index + self.batch_size],\
                    end_epoch
                batch_index += self.batch_size
            else:
                end_epoch = 1
                yield _id[-self.batch_size:], _data[-self.batch_size:], _label[-self.batch_size:], _framelength[-self.batch_size:], end_epoch
                batch_index = 0
                # shuffle_index = np.arange(_id.shape[0])
                # random.shuffle(shuffle_index)
                # _id = _id[shuffle_index]
                # _data = _data[shuffle_index]
                # _label = _label[shuffle_index]
                # _framelength = _framelength[shuffle_index]

    def train_iter(self, output_prob, label, mask):
        self.zero_grad()

        output_prob = output_prob * mask

        output_prob = output_prob.view(-1, self.num_output)

        label = label.view(-1)

        loss = self.criterion(output_prob, label)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().data.numpy()[0]

    def framelength2mask(self, data, framelength):
        mask = torch.FloatTensor(data.size(0), data.size(1), self.num_output).fill_(0.0)
        for i in range(len(framelength)):
            mask[i, :framelength[i], :] = 1.0
        return mask
    def valid_error(self, valid_data, valid_label, valid_mask):
        # assert valid_data.size(0) % self.batch_size == 0
        split_num = valid_data.size(0) // self.batch_size + 1
        total_valid_loss = 0.0
        for i in range(split_num):
            if i == split_num - 1:
                batch_valid_data = valid_data[-self.batch_size:]

                overlap = self.batch_size - (valid_data.size(0) - i * self.batch_size)
                valid_prob = self(batch_valid_data)[overlap:]

            else:
                batch_valid_data = valid_data[i * self.batch_size: (i + 1) * self.batch_size]
                valid_prob = self(batch_valid_data)
            batch_valid_label = valid_label[i * self.batch_size: (i + 1) * self.batch_size]
            batch_valid_mask = valid_mask[i * self.batch_size: (i + 1) * self.batch_size]
            # print(valid_prob.size())
            # print(batch_valid_mask.size())
            valid_prob = valid_prob * batch_valid_mask
            valid_prob = valid_prob.view(-1, self.num_output)
            batch_valid_label = batch_valid_label.view(-1)
            valid_loss = self.criterion_all(valid_prob, batch_valid_label)

            total_valid_loss += valid_loss.cpu().data.numpy()[0]
        return total_valid_loss / valid_data.size(0)
    def predict(self, test_data, test_framelength):
        if isinstance(test_data, np.ndarray):
            test_data_t = torch.from_numpy(test_data)
            if self.CUDA:
                test_data_t = test_data_t.cuda()
            test_data_v = Variable(test_data_t, volatile=True)
        elif isinstance(test, Variable):
            test_data_v = test_data
        self.eval()
        predict_list = []
        split_num = test_data_v.size(0) // self.batch_size + 1
        for i in range(split_num):
            if i == split_num - 1:
                batch_test_data = test_data_v[-self.batch_size:]
                overlap = self.batch_size - (test_data_v.size(0) - i * self.batch_size)
                test_prob = self(batch_test_data)[overlap:]
            else:
                batch_test_data = test_data_v[i * self.batch_size: (i + 1) * self.batch_size]
                test_prob = self(batch_test_data)
            _, test_pred = torch.max(test_prob, dim=2)
            test_pred = test_pred.cpu().data.numpy()
            predict_list.append(test_pred)
        self.train()
        return np.vstack(predict_list)
    def run_epoch(self, data_gen, valid_data_v, valid_label_v, valid_mask_v):
        end_epoch = 0
        while end_epoch == 0:
            batch_id, batch_data, batch_label, batch_framelength, end_epoch = next(data_gen)
            batch_data_t, batch_label_t = torch.from_numpy(batch_data), torch.from_numpy(batch_label)

            if self.CUDA:
                batch_data_t, batch_label_t = batch_data_t.cuda(), batch_label_t.cuda()
            batch_data_v, batch_label_v = Variable(batch_data_t), Variable(batch_label_t)

            batch_mask = self.framelength2mask(batch_data_v, batch_framelength)

            if self.CUDA:
                batch_mask_t = batch_mask.cuda()
            batch_mask_v = Variable(batch_mask_t)
            # output format [batch_size, sequence length, num_output]
            output_prob = self(batch_data_v)


            train_loss = self.train_iter(output_prob, batch_label_v, batch_mask_v)

        # validate
        valid_loss = self.valid_error(valid_data_v, valid_label_v, valid_mask_v)
        return train_loss, valid_loss
    def fit(self, train_tuple):
        # train_tuple's content 
        # [0] : id
        # [1] : feature
        # [2] : label
        # [3] : framelength
        train_id = train_tuple[0]
        train_data = train_tuple[1]
        train_label = train_tuple[2]
        train_framelength = train_tuple[3]
        if self.valid_tuple != None:
            valid_id = self.valid_tuple[0]
            valid_data = self.valid_tuple[1]
            valid_label = self.valid_tuple[2]
            valid_framelength = self.valid_tuple[3]
        else:
            valid_id = train_tuple[0]
            valid_data = train_tuple[1]
            valid_label = train_tuple[2]
            valid_framelength = train_tuple[3]

        valid_data_t, valid_label_t = torch.from_numpy(valid_data), torch.from_numpy(valid_label)
        if self.CUDA:
            valid_data_t, valid_label_t = valid_data_t.cuda(), valid_label_t.cuda()
        valid_data_v, valid_label_v = Variable(valid_data_t, volatile=True), Variable(valid_label_t, volatile=True)
        valid_mask = self.framelength2mask(valid_data_v, valid_framelength)
        if self.CUDA:
            valid_mask_t = valid_mask.cuda()
        valid_mask_v = Variable(valid_mask_t)

        data_gen = self.batch_data_gen(train_id, train_data, train_label, train_framelength)

        best_valid_loss = 1e7
        early_stop = 0
        for i in range(self.epoch):

            train_loss, valid_loss = self.run_epoch(data_gen, valid_data_v, valid_label_v, valid_mask_v)
            print('epoch %d, train loss: %.6f, valid loss: %.6f' % (i, train_loss, valid_loss))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.best_state = deepcopy(self.state_dict())
                early_stop = 0
            else:
                early_stop += 1
        if self.save:
            torch.save(self.best_state, 'model_rnn.th')
        self.load_state_dict(self.best_state)

def print_result(predict, framelength):
    phone_list = []
    for i in range(predict.shape[0]):
        phone = ''
        for j in range(framelength[i]):
            phone += str(predict[i][j])
        phone_list.append(phone)

        # print(i, phone)
    return phone_list

if __name__ == '__main__':

    feature_size = 4
    hidden_size = 10
    num_layers = 2
    num_output = 7
    total_size = 6

    _framelength = np.asarray(random.sample(range(1, 10), total_size))
    _id = np.arange(total_size)
    _feature = np.random.randn(total_size, 9, 4).astype(np.float32)
    for i in range(len(_framelength)):
        _feature[i, _framelength[i]:, :] = 0.0
    _label = np.zeros(shape=(total_size, 9), dtype=int)
    for i in range(len(_framelength)):
        _label[i, :_framelength[i]] = i + 1
    print(_framelength)
    print_result(_label, _framelength)
    train_tuple = (_id, _feature, _label, _framelength)
    params={
        'feature_size': feature_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_output': num_output,
        'CUDA': 1,
        'lr': 0.01,
        'save': 1,
        'batch_size': 4,
        'epoch': 50,
        'early_stop': 3
    }
    print(torch.cuda.is_available())
    model = rnnModel(params)
    model.cuda()
    print(model)
    model.fit(train_tuple)
    pred = model.predict(_feature, _framelength)
    # print(pred.shape)
    print(print_result(pred, _framelength))
    