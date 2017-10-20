from model_rnn import rnnModel
import torch


def FeatureMat2PhoneStr(predict, framelength, transformer):
    phone_list = []
    for i in range(predict.shape[0]):
        phone = ''
        for j in range(framelength[i]):
            phone += transformer.transform2char(str(predict[i][j]))
        phone_list.append(phone)

        print(i, phone)
    # return phone_list

def train(args, transformer, train_tuple_list, valid_tuple_list):
    # train_tuple's content 
    # [0] : id
    # [1] : feature
    # [2] : label
    # [3] : framelength
    feature_size = train_tuple_list[0][1].shape[2]
    hidden_size = 20
    num_layers = 2
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


        test_size = 64
        # FeatureMat2PhoneStr(train_tuple_list[i][2][:test_size], train_tuple_list[i][3][:test_size], transformer)
        train_pred = model.predict(train_tuple_list[i][1][:test_size], train_tuple_list[i][3][:test_size])
        print('train predict')
        FeatureMat2PhoneStr(train_pred, train_tuple_list[i][3][:test_size], transformer)
        # valid_pred = model.predict(valid_tuple_list[i][1][:test_size], valid_tuple_list[i][3][:test_size])
        print('valid predict')
        # FeatureMat2PhoneStr(valid_pred, valid_tuple_list[i][3][:test_size], transformer)

