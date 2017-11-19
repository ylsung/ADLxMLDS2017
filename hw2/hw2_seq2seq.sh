#!/bin/bash
git clone https://gitlab.com/louis2889184/ADLxMLDS_hw2_model.git
python main.py --todo 'test' --model 'seq2seq' --data $1 --test_out $2 --peer_out $3 --save 'embed_all_0_50_dp0.5' --load 'embed_all_0_50_dp0.5'
