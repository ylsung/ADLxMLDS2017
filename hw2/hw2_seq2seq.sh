#!/bin/bash
python main.py --todo 'train' --model 'seq2seq' --data 'MLDS_hw2_data' --test_out 'output/speical.txt' --peer_out 'output/peer_out.txt' --save 'seq2seq_embed600_hidden200*2_dp0.5_attpen_'
