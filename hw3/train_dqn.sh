python3 main.py --train_dqn \
--save_path 'ADLxMLDS2017_hw3_model/dqn_henorm_replay300000_target50_exp2000_lstart1000_freq4' \
--start_game 0 --model_id '' --games 1000001 --memory_size 300000 \
--load_path 'ADLxMLDS2017_hw3_model/dqn_henorm_replay300000_target50_exp2000_lstart1000_freq4' \
--batch_size 64 --decay_exploration 2000 \
--update_target 50 --learning_start 1000 --online_update_freq 4