import numpy as np
import random
from model_base import RNNEncoder, attnRNNDecoder, attnDecoder
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import sys

def data_gen(data, label_array, str_length_array, caption_length_array, name_array, batch_size, shuffle):
    start = 0
    end_epoch = 0
    while True:
        end = start + batch_size
        if end < data.shape[0]:
            end_epoch = 0
            # caption_sample = (caption_length_array[start: end] * np.random.rand(batch_size)).astype(int)
            caption_sample = 0
            start2end = range(start, end)
            yield data[start: end], label_array[start2end, caption_sample], str_length_array[start2end,\
              caption_sample], name_array[start: end], end_epoch
            start += batch_size
        else:
            end_epoch = 1
            end = data.shape[0]
            start = end - batch_size

            # caption_sample = (caption_length_array[start: end] * np.random.rand(batch_size)).astype(int)
            caption_sample = 0
            start2end = range(start, end)
            yield data[start: end], label_array[start2end, caption_sample], str_length_array[start2end,\
              caption_sample], name_array[start: end], end_epoch
            start = 0
            if shuffle:
                shuffle_index = np.arange(data.shape[0])
                random.shuffle(shuffle_index)
                data = data[shuffle_index]
                label_array = label_array[shuffle_index]
                str_length_array = str_length_array[shuffle_index]
                caption_length_array = caption_length_array[shuffle_index]
                name_array = name_array[shuffle_index]

def NumArray2Str(pred, num2word):
    out_list = []
    for i in range(pred.shape[0]):
        string = ''
        for j in range(1, pred.shape[1]):
            word = num2word[pred[i, j]]
            if word == 'EOS':
                string += '.'
                break
            if j != 1:
                string += ' '
            string += num2word[pred[i, j]]
        out_list.append(string)
    return np.stack(out_list)


def numpy2Variable(_input, _eval):
    _input_t = torch.from_numpy(_input)
    if torch.cuda.is_available():
        _input_t = _input_t.cuda()
    return Variable(_input_t, requires_grad=False, volatile=_eval)

def schedule_sampling(pred_pre, label_pre, prob_use_real):
    if random.random() < prob_use_real:
        return label_pre
    else:
        return pred_pre

def train_iteration(data_generator, decoder, optimizer, criterion, word2num, num2word):
    decoder.zero_grad()
    batch_data, batch_label, batch_str_length, batch_name, end_epoch = next(data_generator)

    # start_word = batch_label[:, 0] # start from BOS
    pred_word = np.zeros(shape=(batch_data.shape[0], batch_label.shape[1]), dtype=int)
    pred_word[:, 0] = batch_label[:, 0]
    mask = np.ones(shape=(batch_data.shape[0], 1), dtype=np.float32)
    batch_loss = 0.0
    total_string_length = 0.0
    batch_data_v, batch_label_v = numpy2Variable(batch_data, _eval=False), numpy2Variable(batch_label, _eval=False)
    h, c = decoder.init_hidden(batch_data.shape[0])
    weight_penalty = 0.0
    for i in range(1, batch_label.shape[1]):
        # print(i)
        # use real data as pre word
        pre_word = schedule_sampling(pred_word[:, i - 1], batch_label[:, i - 1], 0.5)
        pre_word = numpy2Variable(pre_word, _eval=False)
        # h, c = Variable(h.data), Variable(c.data)
        output, h, c, = decoder(pre_word, batch_data_v, h.detach(), c.detach())
        # weight_penalty += attn_weight.mean(dim=0)
        _, pred = torch.topk(output, 1)
        mask = mask * (batch_label[:, i - 1] != word2num['EOS']).reshape(-1, 1)
        mask_v = numpy2Variable(mask, _eval=False)

        loss = criterion(output * mask_v, batch_label_v[:, i])
        loss.backward()
        total_string_length += float(mask.sum())
        batch_loss += loss.cpu().data.numpy()[0]

        pred_word[:, i] = pred.cpu().data.numpy().reshape(-1)
    # weight_penalty = ((1.0 - weight_penalty) ** 2).mean()
    # weight_penalty.backward()
    batch_loss = batch_loss / total_string_length
    # batch_loss.backward()
    optimizer.step()

    return batch_loss, end_epoch

def eval(data_generator, decoder, word2num, criterion, max_length):
    end_epoch = 0
    iteration = 0.0
    total_loss = 0.0
    decoder.eval()
    pred_list = []
    while not end_epoch:
        batch_data, batch_label, batch_str_length, batch_name, end_epoch = next(data_generator)

        pred_word = np.zeros(shape=(batch_data.shape[0], max_length), dtype=int)
        pred_word[:, 0].fill(word2num['BOS'])
        mask = np.ones(shape=(batch_data.shape[0], 1), dtype=np.float32)
        batch_loss = 0.0
        total_string_length = 0.0
        batch_data_v, batch_label_v = numpy2Variable(batch_data, _eval=True), numpy2Variable(batch_label, _eval=True)
        h, c = decoder.init_hidden(batch_data.shape[0])
        for i in range(1, pred_word.shape[1]):
            # print(i)
            # use real data as pre word
            pre_word = numpy2Variable(pred_word[:, i - 1], _eval=True)
            # h, c = Variable(h.data), Variable(c.data)
            output, h, c, = decoder(pre_word, batch_data_v, h.detach(), c.detach())

            _, pred = torch.topk(output, 1)
            mask = mask * (pred_word[:, i - 1] != word2num['EOS']).reshape(-1, 1)
            mask_v = numpy2Variable(mask, _eval=True)

            loss = criterion(output * mask_v, batch_label_v[:, i])

            total_string_length += float(mask.sum())
            batch_loss += loss.cpu().data.numpy()[0]

            pred_word[:, i] = pred.cpu().data.numpy().reshape(-1)

        pred_list.append(pred_word)
        batch_loss = batch_loss / total_string_length
        total_loss += batch_loss
        iteration += 1.0
    decoder.train()
    return total_loss, np.vstack(pred_list)

def predict(data, decoder, batch_size, word2num, max_length):
    decoder.eval()
    start = 0
    predict_list = []
    end_epoch = 0
    while not end_epoch:
        end = start + batch_size
        pred_word = np.zeros(shape=(batch_size, max_length), dtype=int)
        pred_word[:, 0].fill(word2num['BOS'])

        h, c = decoder.init_hidden(batch_size)
        if end < data.shape[0]:
            end_epoch = 0
            overlap = batch_size
            batch_data = data[start: end]

            batch_data_v = numpy2Variable(batch_data, _eval=True)
            for i in range(1, max_length):
                # use real data as pre word
                pre_word = numpy2Variable(pred_word[:, i - 1], _eval=True)
                # h, c = Variable(h.data), Variable(c.data)
                output, h, c, = decoder(pre_word, batch_data_v, h.detach(), c.detach())

                # greedy search
                _, pred = torch.topk(output, 1)

                pred_word[:, i] = pred.cpu().data.numpy().reshape(-1)

        else:
            end_epoch = 1
            overlap = data.shape[0] - start
            batch_data = data[-batch_size:]

            batch_data_v = numpy2Variable(batch_data, _eval=True)
            for i in range(1, max_length):
                # use real data as pre word
                pre_word = numpy2Variable(pred_word[:, i - 1], _eval=True)
                # h, c = Variable(h.data), Variable(c.data)
                output, h, c, = decoder(pre_word, batch_data_v, h.detach(), c.detach())

                # greedy search
                _, pred = torch.topk(output, 1)

                pred_word[:, i] = pred.cpu().data.numpy().reshape(-1)
            pred_word = pred_word[-overlap:]
        predict_list.append(pred_word)

        start += batch_size
    decoder.train()
    return np.vstack(predict_list)



def train_epoch(data_generator, valid_data_generator, decoder, optimizer, criterion, word2num, num2word, max_length):
    end_epoch = 0
    iteration = 0.0
    total_loss = 0.0
    while not end_epoch:
        
        iteration += 1.0
        loss, end_epoch = train_iteration(data_generator, decoder, optimizer, criterion, word2num, num2word)
        total_loss += loss

    valid_loss, valid_pred = eval(valid_data_generator, decoder, word2num, criterion, max_length)

    total_loss /= iteration
    return total_loss, valid_loss, valid_pred

def train(args, train_data, label_array, str_length_array, caption_length_array, name_array, word2num, num2word, valid_dict, max_length):
    MaxEpoch = 600
    learning_rate = 1e-3
    batch_size = 100
    en_feature_size = train_data.shape[2]
    de_feature_size = train_data.shape[1] * train_data.shape[2]
    embed_size = 600
    hidden_size = 64
    num_layers = 2
    frame_size = train_data.shape[1]
    dropout = 0.5
    vocab_size = len(word2num)

    valid_data = valid_dict['valid_data']
    valid_label_array = valid_dict['valid_label_array']
    valid_str_length_array = valid_dict['valid_str_length_array']
    valid_caption_length_array = valid_dict['valid_caption_length_array']
    valid_name_array = valid_dict['valid_name_array']

    data_generator = data_gen(train_data, label_array, str_length_array, caption_length_array, name_array, batch_size, True)

    valid_data_generator = data_gen(valid_data, valid_label_array, valid_str_length_array, valid_caption_length_array,
        valid_name_array, batch_size, False)

    # encoder = RNNEncoder(feature_size, hidden_size, num_layers, dropout)
    decoder = attnDecoder(embed_size, frame_size, de_feature_size, hidden_size, num_layers, dropout, vocab_size)
    print(decoder)
    criterion = nn.NLLLoss(size_average=False)
    if torch.cuda.is_available():
        decoder.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    for epoch in range(MaxEpoch):
        loss, valid_loss, valid_pred = \
          train_epoch(data_generator, valid_data_generator, decoder, optimizer, criterion, word2num, num2word, max_length)

        train_pred = predict(train_data, decoder, batch_size, word2num, max_length)
        print('-------------------train------------------------')
        print('real')
        print(NumArray2Str(label_array[-2:, 0, :], num2word))
        print('predict')
        print(NumArray2Str(train_pred[-2:], num2word))

        print('epoch %d, train loss: %.4f, valid_loss: %.4f' % (epoch, loss, valid_loss))
        print('-------------------eval------------------------')
        print('real')
        print(NumArray2Str(valid_label_array[-2:, 0, :], num2word))
        print('predict')
        print(NumArray2Str(valid_pred[-2:], num2word))
        if epoch % 100 == 0:
            model_path = os.path.join('model', args.save, 'model_%d.pth' % epoch)
            torch.save(decoder.state_dict(), model_path)
        sys.stdout.flush()
    