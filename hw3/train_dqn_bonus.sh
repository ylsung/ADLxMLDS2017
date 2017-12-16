# python3 main_dqn_bonus.py --train_dqn_pm \
# --save_path 'ADLxMLDS2017_hw3_model/dqn_pm_henorm_replay10000_target50_exp2000_lstart1000_freq4' \
# --start_game 0 --model_id '' --games 1000001 --memory_size 10000 \
# --load_path 'ADLxMLDS2017_hw3_model/dqn_pm_henorm_replay10000_target50_exp2000_lstart1000_freq4' \
# --batch_size 64 --decay_exploration 2000 \
# --update_target 50 --learning_start 1000 --online_update_freq 4

# python3 main_dqn_bonus.py --train_dqn_duel \
# --save_path 'ADLxMLDS2017_hw3_model/dqn_duel_henorm_replay10000_target50_exp2000_lstart1000_freq4' \
# --start_game 0 --model_id '' --games 1000001 --memory_size 10000 \
# --load_path 'ADLxMLDS2017_hw3_model/dqn_duel_henorm_replay10000_target50_exp2000_lstart1000_freq4' \
# --batch_size 64 --decay_exploration 2000 \
# --update_target 50 --learning_start 1000 --online_update_freq 4

python3 main_dqn_bonus.py --train_dqn_db \
--save_path 'ADLxMLDS2017_hw3_model/dqn_db_henorm_replay10000_target50_exp2000_lstart1000_freq4' \
--start_game 0 --model_id '' --games 1000001 --memory_size 10000 \
--load_path 'ADLxMLDS2017_hw3_model/dqn_db_henorm_replay10000_target50_exp2000_lstart1000_freq4' \
--batch_size 64 --decay_exploration 2000 \
--update_target 50 --learning_start 1000 --online_update_freq 4