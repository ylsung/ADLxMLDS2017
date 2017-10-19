from model_rnn import rnnModel
import torch


def FeatureMat2PhoneStr(predict, framelength):
    phone_list = []
    for i in range(predict.shape[0]):
        phone = ''
        for j in range(framelength[i]):
            phone += str(predict[i][j])
        phone_list.append(phone)

        # print(i, phone)
    return phone_list

def train(args, transformer, train_tuple_list, valid_tuple_list):
    # train_tuple's content 
    # [0] : id
    # [1] : feature
    # [2] : label
    # [3] : framelength
    feature_size = train_tuple_list[0][1].shape[2]
    hidden_size = 40
    num_layers = 3
    num_output = 48
    CUDA = args.cuda and torch.cuda.is_available()

    # train the model
    for i in range(len(train_tuple_list)):
        params={
            'feature_size': feature_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_output': num_output,
            'CUDA': CUDA,
            'lr': args.lr,
            'save': args.save,
            'valid': valid_tuple_list[i],
            'batch_size': args.batch_size,
            'epoch': args.epoch,
            'early_stop': args.early_stop
        }

        if args.model == 'rnn':
            model = rnnModel(params)

        if CUDA:
            model.cuda()

        model.fit(train_tuple_list[i])

