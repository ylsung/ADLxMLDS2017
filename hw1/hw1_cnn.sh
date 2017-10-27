#!/bin/bash
python main.py --todo test --data_directory $1 --write_file $2 --cuda --batch_size 256 --epoch 100 --model 'rnn_cnn' --load_folder 'rnn_cnn' 
