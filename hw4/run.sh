# python3.6 main.py --todo 'test' --load 'binary_style_mask_aug' --model_id 500 \
# --img_save 'sample' --te_data 'data/sample_testing_text.txt' \

python3.6 main.py --todo 'test' --load 'binary_style_mask_aug_stage2' --model_id 500 \
--img_save 'samples' --te_data $1 --nz 100 \
--image 96 --stage 2