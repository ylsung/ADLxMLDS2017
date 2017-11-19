import numpy as np
import random
from model_base import RNNEncoder, attnRNNDecoder, ReLURNNEncoder, ReLUattnRNNDecoder
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import sys
import os
import pickle
import heapq
import math
from copy import deepcopy

def data_gen(data, label_array, str_length_array, caption_length_array, name_array, batch_size, shuffle, choise):
    start = 0
    end_epoch = 0
    # _max = data.max()
    while True:
        end = start + batch_size
        if end < data.shape[0]:
            end_epoch = 0
            if choise:
                caption_sample = (caption_length_array[start: end] * np.random.rand(batch_size)).astype(int)
            else:
                caption_sample = 0
            start2end = range(start, end)
            if shuffle:
                # noise = np.random.normal(scale=_max/101.0, 
                #     size=(len(start2end), data.shape[1], data.shape[2])).astype(np.float32)
                # noise = 0.0
                pass
            else:
                noise = 0.0
            yield data[start: end], label_array[start2end, caption_sample], str_length_array[start2end,\
              caption_sample], name_array[start: end], end_epoch
            start += batch_size
        else:
            end_epoch = 1
            end = data.shape[0]
            # start = end - batch_size
            batch_size_temp = end - start
            if choise:
                caption_sample = (caption_length_array[start: end] * np.random.rand(batch_size_temp)).astype(int)
            else:
                caption_sample = 0
            start2end = range(start, end)
            if shuffle:
                # noise = np.random.normal(scale=_max/101.0, 
                #     size=(len(start2end), data.shape[1], data.shape[2])).astype(np.float32)
                # noise = 0.0
                pass
            else:
                noise = 0.0
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
def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x)))
def train_iteration(data_generator, decoder, encoder, optimizer_en, optimizer_de, criterion, word2num, num2word, ratio):
    decoder.zero_grad()
    encoder.zero_grad()
    batch_data, batch_label, batch_str_length, batch_name, end_epoch = next(data_generator)

    batch_data_v, batch_label_v = numpy2Variable(batch_data, _eval=False), numpy2Variable(batch_label, _eval=False)

    # batch_data_v[:, :, -3:].data.normal_(0, 1)
    ################
    # Encoder part
    ################
    h, c = encoder.init_hidden(batch_data.shape[0])
    encoder_output, h, c = encoder(batch_data_v, h, c)
    ##################################
    # declare the array to save pred
    ##################################

    # start_word = batch_label[:, 0] # start from BOS
    pred_word = np.zeros(shape=(batch_data.shape[0], batch_label.shape[1]), dtype=int)
    pred_word[:, 0].fill(word2num['BOS'])
    mask = np.ones(shape=(batch_data.shape[0], 1), dtype=np.float32)
    batch_loss = 0.0
    total_string_length = 0.0

    weight_penalty = 0.0
    ####################
    # Decoder part
    ####################
    # schedule_sampling_ratio = sigmoid(ratio * 3.0)
    for i in range(1, batch_label.shape[1]):
        # print(i)
        # use real data as pre word
        pre_word = schedule_sampling(pred_word[:, i - 1], batch_label[:, i - 1], ratio)
        pre_word = numpy2Variable(pre_word, _eval=False)
  
        output, h, c, attn_weight = decoder(pre_word, encoder_output, h, c)

        _, pred = torch.topk(output, 1)
        mask = mask * (batch_label[:, i - 1] != word2num['EOS']).reshape(-1, 1)
        mask_v = numpy2Variable(mask, _eval=False)

        weight_penalty += attn_weight * mask_v
        loss = criterion(output * mask_v, batch_label_v[:, i])

        total_string_length += float(mask.sum())
        batch_loss += loss

        pred_word[:, i] = pred.cpu().data.numpy().reshape(-1)
    weight_penalty = 10.0 * ((1.0 - weight_penalty) ** 2).mean()

    # weight_penalty.backward()
    batch_loss = batch_loss / total_string_length
    total_loss = batch_loss + weight_penalty
    total_loss.backward()

    # if ratio % 30 == 0:
    #     print(weight_penalty)
    #     print('grad of encoder')
    #     grad = 0.0
    #     for p in encoder.parameters():
    #         if isinstance(p.grad, torch.autograd.Variable):
    #             grad += torch.sum(torch.abs(p.grad)).cpu().data.numpy()[0]
    #     print(grad)
    #     grad = 0.0
    #     for p in decoder.parameters():

    #         if isinstance(p.grad, torch.autograd.Variable):
    #             grad += torch.sum(torch.abs(p.grad)).cpu().data.numpy()[0]
    #     print('grad of decoder')
    #     print(grad)

    # batch_loss.backward()
    optimizer_en.step()
    optimizer_de.step()

    return batch_loss.cpu().data.numpy()[0], end_epoch

def eval(data_generator, decoder, encoder, word2num, criterion, max_length):
    end_epoch = 0
    iteration = 0.0
    total_loss = 0.0
    decoder.eval()
    encoder.eval()
    pred_list = []
    while not end_epoch:
        batch_data, batch_label, batch_str_length, batch_name, end_epoch = next(data_generator)

        pred_word = np.zeros(shape=(batch_data.shape[0], max_length), dtype=int)
        pred_word[:, 0].fill(word2num['BOS'])
        mask = np.ones(shape=(batch_data.shape[0], 1), dtype=np.float32)
        batch_loss = 0.0
        total_string_length = 0.0
        batch_data_v, batch_label_v = numpy2Variable(batch_data, _eval=True), numpy2Variable(batch_label, _eval=True)
        h, c = encoder.init_hidden(batch_data.shape[0])
        ###############
        # Encode
        ###############
        encoder_output, h, c = encoder(batch_data_v, h, c)

        for i in range(1, pred_word.shape[1]):
            # print(i)
            # use real data as pre word
            pre_word = numpy2Variable(pred_word[:, i - 1], _eval=True)
            # h, c = Variable(h.data), Variable(c.data)
            output, h, c, _ = decoder(pre_word, encoder_output, h, c)

            _, pred = torch.topk(output, 1)
            mask = mask * (batch_label[:, i - 1] != word2num['EOS']).reshape(-1, 1)
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
    encoder.train()
    return total_loss, np.vstack(pred_list)

def predict(data, decoder, encoder, batch_size, word2num, max_length):
    decoder.eval()
    encoder.eval()
    start = 0
    predict_list = []
    end_epoch = 0
    while not end_epoch:
        end = start + batch_size
    
        # if end < data.shape[0]:
        if end < data.shape[0]:
            end_epoch = 0
        else:
            end_epoch = 1
        batch_data = data[start: end]
        pred_word = np.zeros(shape=(batch_data.shape[0], max_length), dtype=int)
        pred_word[:, 0].fill(word2num['BOS'])

        h, c = encoder.init_hidden(batch_data.shape[0])

        batch_data_v = numpy2Variable(batch_data, _eval=True)

        encoder_output, h, c = encoder(batch_data_v, h, c)
        for i in range(1, max_length):
            # use real data as pre word
            pre_word = numpy2Variable(pred_word[:, i - 1], _eval=True)
            # h, c = Variable(h.data), Variable(c.data)
            output, h, c, _ = decoder(pre_word, encoder_output, h, c)

            # greedy search
            _, pred = torch.topk(output, 1)

            pred_word[:, i] = pred.cpu().data.numpy().reshape(-1)

        predict_list.append(pred_word)

        start += batch_size
    decoder.train()
    encoder.train()
    return np.vstack(predict_list)
def beam_search(prob_array, index_array, candidates, output, h, c, batch_size, beam_size, beam_id, frame_id):
    # prob_array, index_array: the word probability and their index
    # candidates: a list, contains prob heap of each batch
    # prob heap: heap, size equal to beam size, format: (prob, pred_array, last word index)
    for i in range(len(candidates)):
        for j in range(beam_size):
            prob = candidates[i][beam_id][0] + prob_array[i][j] + np.random.rand() / 1e8
            frame_array = deepcopy(candidates[i][beam_id][1])
            frame_array[frame_id] = index_array[i][j]
            output_piece = (prob, frame_array, h[:, i, :].clone().unsqueeze(1), c[:, i, :].clone().unsqueeze(1))
            if len(output[i]) < beam_size:
                heapq.heappush(output[i], output_piece)

            else:
                heapq.heappushpop(output[i], output_piece)


    return output
def predict_BeamSearch(data, decoder, encoder, batch_size, word2num, max_length, beam_size):
    decoder.eval()
    encoder.eval()
    start = 0
    predict_list = []
    end_epoch = 0
    while not end_epoch:
        end = start + batch_size
        
        candidates = []
        beam_output = []

        if end < data.shape[0]:
            end_epoch = 0
        else:
            end_epoch = 1

        batch_data = data[start: end]

        pred_word = np.zeros(shape=(batch_data.shape[0], max_length), dtype=int)
        pred_word[:, 0].fill(word2num['BOS'])

        h, c = encoder.init_hidden(batch_data.shape[0])


        batch_data_v = numpy2Variable(batch_data, _eval=True)

        encoder_output, h, c = encoder(batch_data_v, h, c)

        for i in range(batch_data.shape[0]):
            candidates.append([(0.0, deepcopy(pred_word[i]), h[:, i, :].clone().unsqueeze(1), 
                c[:, i, :].clone().unsqueeze(1))])
            beam_output.append([])

        for i in range(1, max_length):
            beam_output = []
            for l in range(batch_data.shape[0]):
                beam_output.append([])
            for j in range(len(candidates[0])):

                pred_word_list = []
                for k in range(len(candidates)):
                    pred_word_list.append(candidates[k][j][1])
                    if k == 0:
                        h = candidates[k][j][2]
                        c = candidates[k][j][3]
                    else:
                        h = torch.cat((h, candidates[k][j][2]), dim=1)
                        c = torch.cat((c, candidates[k][j][3]), dim=1)
                pred_word = np.stack(pred_word_list)
                pre_word = numpy2Variable(pred_word[:, i - 1], _eval=True)
                output, h, c, _ = decoder(pre_word, encoder_output, h, c)
                
                prob, pred = torch.topk(output, beam_size)
                prob = prob.cpu().data.numpy()
                pred = pred.cpu().data.numpy()
                beam_search(prob, pred, candidates, beam_output, h, c, batch_data.shape[0], beam_size, j, i)
            candidates = beam_output
        output_list = []

        for i in range(batch_data.shape[0]):
            largest_output = heapq.nlargest(1, beam_output[i])[0][1]

            output_list.append(largest_output)
        final_pred = np.stack(output_list)

        predict_list.append(final_pred)

        start += batch_size
    decoder.train()
    encoder.train()
    return np.vstack(predict_list)


def train_epoch(data_generator, valid_data_generator, decoder, encoder, optimizer_en, optimizer_de, criterion, word2num,
    num2word, max_length, ratio):
    end_epoch = 0
    iteration = 0.0
    total_loss = 0.0
    print(ratio)
    while not end_epoch:
        
        iteration += 1.0
        loss, end_epoch = train_iteration(data_generator, decoder, encoder, optimizer_en, optimizer_de, criterion,
            word2num, num2word, ratio)
        total_loss += loss

    valid_loss, valid_pred = eval(valid_data_generator, decoder, encoder, word2num, criterion, max_length)

    total_loss /= iteration
    return total_loss, valid_loss, valid_pred

def train(args, train_data, label_array, str_length_array, caption_length_array, name_array, word2num, num2word, valid_dict, max_length):
    MaxEpoch = 151
    learning_rate_en = 1e-3
    learning_rate_de = 1e-3
    batch_size = 128
    en_feature_size = train_data.shape[2]
    de_feature_size = train_data.shape[1] * train_data.shape[2]
    embed_size = 200
    hidden_size = 200
    num_layers = 1
    frame_size = train_data.shape[1]
    dropout = 0.5
    vocab_size = len(word2num)
    direction = 2

    valid_data = valid_dict['valid_data']
    valid_label_array = valid_dict['valid_label_array']
    valid_str_length_array = valid_dict['valid_str_length_array']
    valid_caption_length_array = valid_dict['valid_caption_length_array']
    valid_name_array = valid_dict['valid_name_array']

    data_generator = data_gen(train_data, label_array, str_length_array, caption_length_array, name_array, batch_size, True, 1)

    valid_data_generator = data_gen(valid_data, valid_label_array, valid_str_length_array, valid_caption_length_array,
        valid_name_array, batch_size, False, 1)
    model_info_path = os.path.join('ADLxMLDS_hw2_model', args.save, 'model_info.pickle')
    with open(model_info_path, 'wb') as f:
        model_info = {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'embed_size': embed_size,
        'direction': direction,
        'batch_size': batch_size,
        }
        pickle.dump(model_info, f)

    # encoder = RNNEncoder(feature_size, hidden_size, num_layers, dropout)
    encoder = RNNEncoder(en_feature_size, hidden_size, num_layers, dropout, direction)
    decoder = attnRNNDecoder(embed_size, frame_size, hidden_size, num_layers, dropout, vocab_size, direction)
    print(encoder)
    print(decoder)
    criterion = nn.NLLLoss(size_average=False)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        criterion = criterion.cuda()
    optimizer_en = torch.optim.Adam(encoder.parameters(), lr=learning_rate_en)
    optimizer_de = torch.optim.Adam(decoder.parameters(), lr=learning_rate_de)
    switch = 1
    threshold = int(0.5 * MaxEpoch)
    for epoch in range(MaxEpoch):
        ratio = 1.0
        # if epoch < threshold:
        loss, valid_loss, valid_pred = train_epoch(data_generator, valid_data_generator, decoder, encoder, optimizer_en,
            optimizer_de, criterion, word2num, num2word, max_length, ratio)
        # else:
            
        #     if switch:
        #         switch = 0
        #         data_generator = data_gen(train_data, label_array, str_length_array, caption_length_array, 
        #             name_array, batch_size, True, 0)
        #     loss, valid_loss, valid_pred = train_epoch(data_generator, valid_data_generator, decoder, encoder, optimizer_en,
        #         optimizer_de, criterion, word2num, num2word, max_length, ratio)

        train_pred = predict(train_data, decoder, encoder, batch_size, word2num, max_length)
        # valid_pred = predict(valid_data, decoder, encoder, batch_size, word2num, max_length)

        print('epoch %d, train loss: %.4f, valid loss: %.4f' % (epoch, loss, valid_loss))


        start_train = np.random.randint(label_array.shape[0])
        print('-------------------train------------------------')
        print('real')
        print(NumArray2Str(label_array[start_train: start_train+2, 0, :], num2word))
        print('name')
        print(name_array[start_train: start_train+2])
        print('predict')
        print(NumArray2Str(train_pred[start_train: start_train+2], num2word))

        start_valid = np.random.randint(valid_label_array.shape[0])
        print('-------------------valid------------------------')
        print('real')
        print(NumArray2Str(valid_label_array[start_valid: start_valid+2, 0, :], num2word))
        print('name')
        print(valid_name_array[start_valid: start_valid+2])
        print('predict')
        print(NumArray2Str(valid_pred[start_valid: start_valid+2], num2word))

        if epoch % 30 == 0:
            decoder_path = os.path.join('ADLxMLDS_hw2_model', args.save, 'decoder_%d.pth' % epoch)
            encoder_path = os.path.join('ADLxMLDS_hw2_model', args.save, 'encoder_%d.pth' % epoch)
            torch.save(decoder.state_dict(), decoder_path)
            torch.save(encoder.state_dict(), encoder_path)

        sys.stdout.flush()

def test(args, test_data, test_name, word2num, num2word, max_length, file_type):
    model_id = 90
    decoder_path = os.path.join('ADLxMLDS_hw2_model', args.load, 'decoder_%d.pth' % model_id)
    encoder_path = os.path.join('ADLxMLDS_hw2_model', args.load, 'encoder_%d.pth' % model_id)
    # model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))

    en_feature_size = test_data.shape[2]
    frame_size = test_data.shape[1]

    vocab_size = len(word2num)

    model_info_path = os.path.join('ADLxMLDS_hw2_model', args.load, 'model_info.pickle')
    with open(model_info_path, 'rb') as f:
        model_info = pickle.load(f)
        embed_size = model_info['embed_size']
        hidden_size = model_info['hidden_size']
        num_layers = model_info['num_layers']
        dropout = model_info['dropout']
        direction = model_info['direction']
        batch_size = model_info['batch_size']

    encoder = RNNEncoder(en_feature_size, hidden_size, num_layers, dropout, direction)
    decoder = attnRNNDecoder(embed_size, frame_size, hidden_size, num_layers, dropout, vocab_size, direction)


    encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
    decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    # test_pred = predict(test_data, decoder, encoder, batch_size, word2num, max_length)
    test_pred = predict_BeamSearch(test_data, decoder, encoder, batch_size, word2num, max_length, 3)
    test_string = NumArray2Str(test_pred, num2word)

    test_string = test_string.reshape(-1, 1)
    test_name = test_name.reshape(-1, 1)
    out = np.concatenate((test_name, test_string), axis=1)

    if file_type == 'test':
        np.savetxt(args.test_out, out, fmt='%s', delimiter=',', comments='')
    elif file_type == 'peer':
        np.savetxt(args.peer_out, out, fmt='%s', delimiter=',', comments='')
