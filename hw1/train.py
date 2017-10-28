from model_rnn import rnnModel
from model_dnn import dnnModel
from model_best import lstmModel
from model_rnn_cnn import rnn_cnnModel

import torch
from time import time
import numpy as np
import os
from collections import Counter

def remove_middle(string):
    string = list(string)
    pre_flag = 0
    now_flag = 0
    for i in range(1, len(string)):
        if string[i] != string[i - 1]:
            now_flag = 1
        else:
            now_flag = 0

        if pre_flag and now_flag:
            string[i - 1] = string[i - 2]
        pre_flag = now_flag
    if string[0] != string[1]:
        string[0] = string[1]
    if string[-1] != string[-2]:
        string[-1] = string[-2]
    return ''.join(string)
def smooth(string):
    string = list(string)
    new_string = ''
    window = 7
    half_window = window // 2
    for i in range(len(string)):
        left = i - half_window if i >= half_window else 0
        right = i + half_window
        window_list = string[left:right]
        window_counter = Counter(window_list)
        # most_common = window_counter.most_common(2)
        # if len(most_common) == 2:
        #     if most_common[0][1] == most_common[1][1]:
        #         if most_common[0][0] == string[i]:
        #             new_string += most_common[0][0]
        #         elif most_common[1][0] == string[i]:
        #             new_string += most_common[1][0]
        #         else:
        #             new_string += most_common[0][0]
        #     else:
        #         new_string += most_common[0][0]
        # else:
        new_string += window_counter.most_common(1)[0][0]
    return new_string



# num = 0
# for phone in phone_list:
#     for i in range(1, len(phone) - 2):
#         if phone[i] == phone[i + 1]:
#             if phone[i - 1] != phone[i] and phone[i + 1] != phone[i + 2]:
#                 num += 1


def FeatureMat2PhoneStr(predict, framelength, transformer):
    phone_list = []
    for i in range(predict.shape[0]):
        phone = ''
        pre_flag = 0
        now_flag = 0
        for j in range(framelength[i]):
            phone += transformer.transform2char(str(predict[i][j]))
        # phone = remove_middle(phone)
        phone = smooth(phone)
        phone_list.append(phone)
        if i < 2:
            print(phone)
        # print(i, phone)
    return phone_list
def TrimSeq(seq_str):
    pre_char = ''
    out_str = ''
    for char in seq_str:
        if char != pre_char:
            out_str += char
        pre_char = char

    if out_str[0] == 'L':
        out_str = out_str[1:]
    if out_str[-1] == 'L':
        out_str = out_str[:-1]
    return out_str

def minEditDist(sm,sn):
    m, n = len(sm),len(sn)
    D = np.zeros(shape=(m + 1, n + 1), dtype=np.float32)
    for i in range(D.shape[0]):
        D[i][0] = i
    for i in range(D.shape[1]):
        D[0][i] = i
    # print(D)
    for i in range(1, m + 1):
      for j in range(1, n + 1):
        D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (0 if sm[i - 1] == sn[j - 1] else 1))
    # for i in range(0,m+1):
    #   print(D[i]) 
    return D[m][n]

def bestPath(pred_prob, trans_prob):
    # pred_prob shape = (batch, seq, num_output)
    # trans_prob shape = (num_output, num_output)

    predict_array = np.zeros(shape=(pred_prob.shape[0], pred_prob.shape[1]), dtype=np.int)
    for i in range(pred_prob.shape[0]):
        prob_map = np.zeros(shape=(pred_prob.shape[1], pred_prob.shape[2]), dtype=np.float32)
        prob_map.fill(-np.inf)
        parent_map = np.zeros(shape=(pred_prob.shape[1], pred_prob.shape[2]), dtype=np.int)
        for j in range(prob_map.shape[0]):
            # seq
            for k in range(prob_map.shape[1]):
                if j == 0:
                    prob_map[j][k] = np.log(pred_prob[i][j][k])
                else:
                    for l in range(prob_map.shape[1]):
                        # now state k, pre state l
                        prob_total = prob_map[j - 1][l] + np.log(trans_prob[l][k]) + np.log(pred_prob[i][j][k])

                        if prob_total > prob_map[j][k]:
                            prob_map[j][k] = prob_total
                            parent_map[j][k] = l

        fut_index = -1
        for j in range(predict_array.shape[1] - 1, -1, -1):
            if fut_index == -1:
                index = np.argmax(prob_map[j])
                # print(index)
                fut_index = parent_map[j][index]
                # print(fut_index)
                predict_array[i][j] = index
            else:
                predict_array[i][j] = fut_index
                # print(predict_array[i][j])
                fut_index = parent_map[j][fut_index]

    return predict_array


def avgEditDist(model, data_list, transformer):
    target_str_list = FeatureMat2PhoneStr(data_list[2], data_list[3], transformer)
    pred, _ = model.predict(data_list[1], data_list[3])

    # pred = bestPath(_, transition_matrix)
    pred_str_list = FeatureMat2PhoneStr(pred, data_list[3], transformer)

    # trim list
    Edit_dist_list = []
    for i in range(len(target_str_list)):
        target_str_list[i] = TrimSeq(target_str_list[i])
        pred_str_list[i] = TrimSeq(pred_str_list[i])
        Edit_dist_list.append(minEditDist(target_str_list[i], pred_str_list[i]))
    return np.mean(Edit_dist_list)


def train(args, transformer, train_tuple_list, valid_tuple_list, transition_matrix_list):
    # train_tuple's content 
    # [0] : id
    # [1] : feature
    # [2] : label
    # [3] : framelength
    feature_size = train_tuple_list[0][1].shape[2]
    hidden_size = 64
    num_layers = 3
    num_output = 48
    CUDA = args.cuda and torch.cuda.is_available()
    torch.manual_seed(int(time()))

    # train the model
    edit_train = []
    edit_valid = []
    for i in range(len(train_tuple_list)):
        params={
            'feature_size': feature_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_output': num_output,
            'CUDA': CUDA,
            'lr': args.lr,
            'save': args.save,
            'batch_size': args.batch_size,
            'epoch': args.epoch,
            'early_stop': args.early_stop,
            'gpu': args.gpu
        }
        if len(valid_tuple_list) != 0:
            params['valid'] =  valid_tuple_list[i]

        if args.model == 'rnn':
            model = rnnModel(params)
        elif args.model == 'dnn':
            model = dnnModel(params)
        elif args.model == 'rnn_cnn':
            model = rnn_cnnModel(params)
        elif args.model == 'lstm':
            model = lstmModel(params)

        if CUDA:
            model.cuda()
        print(model)
        if args.load_folder == '':
            model.fit(train_tuple_list[i])
        else:
            model_name = os.path.join('model', args.load_folder, 'model_%d.th' % i)
            model.load_state_dict(torch.load(model_name))

        print('fold: %d' % i)
        edit_train.append(avgEditDist(model, train_tuple_list[i], transformer))
        edit_valid.append(avgEditDist(model, valid_tuple_list[i], transformer))
        print('train_edit_distance: ', edit_train[i])

        print('valid_edit_distance: ', edit_valid[i])

        if args.save:
            model_name = os.path.join('model', args.save_folder, 'model_%d.th' % i)
            torch.save(model.state_dict(), model_name)
    print('avg train_edit_distance: ', np.mean(edit_train))
    print('avg valid_edit_distance: ', np.mean(edit_valid))

def test(args, transformer, test_tuple_list):
    feature_size = test_tuple_list[0][1].shape[2]
    hidden_size = 64
    num_layers = 3
    num_output = 48
    CUDA = args.cuda and torch.cuda.is_available()
    torch.manual_seed(int(time()))

    # train the model
    final_prob = np.zeros(shape=(
        test_tuple_list[0][1].shape[0], test_tuple_list[0][1].shape[1], num_output), dtype=np.float32)
    for i in range(len(test_tuple_list)):
        params={
            'feature_size': feature_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_output': num_output,
            'CUDA': CUDA,
            'lr': args.lr,
            'save': 0,
            'batch_size': args.batch_size,
            'epoch': args.epoch,
            'early_stop': args.early_stop,
            'gpu': args.gpu
        }

        if args.model == 'rnn':
            model = rnnModel(params)
        elif args.model == 'dnn':
            model = dnnModel(params)
        elif args.model == 'rnn_cnn':
            model = rnn_cnnModel(params)
        elif args.model == 'lstm':
            model = lstmModel(params)

        if CUDA:
            model.cuda()
        print(model)
        if args.load_folder != '':
            model_name = os.path.join('model', args.load_folder, 'model_%d.th' % i)
            model.load_state_dict(torch.load(model_name))

        _, test_prob = model.predict(test_tuple_list[i][1], test_tuple_list[i][2])
        final_prob = final_prob + test_prob

    final_prob = final_prob / float(len(test_tuple_list))

    final_pred = np.argmax(final_prob, axis=2)
    pred_str_list = FeatureMat2PhoneStr(final_pred, test_tuple_list[0][2], transformer)
    for i in range(len(pred_str_list)):
        pred_str_list[i] = TrimSeq(pred_str_list[i])

    pred_str_array = np.vstack(pred_str_list)
    pred_name = test_tuple_list[0][0].reshape(-1, 1)

    out = np.concatenate((pred_name, pred_str_array), axis=1)

    np.savetxt(args.write_file, out, fmt='%s', delimiter=',', header='id,phone_sequence', comments='')





