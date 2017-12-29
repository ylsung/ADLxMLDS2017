from utils import load_tags, list2dict, load_style, create_logger, load_te_tags

import os
import argparse

def parser():
    parser = argparse.ArgumentParser(description='Conditional GAN')
    parser.add_argument('--todo', default='train', choices=['train', 'test'])
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--image', default=64, type=int)
    parser.add_argument('--degree', default=30.0, type=float)
    parser.add_argument('--max_epoch', default=500, type=int)
    parser.add_argument('--LAMBDA', default=10.0, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--nz', default=100, type=int)
    parser.add_argument('--nc', default=3, type=int)
    parser.add_argument('--tags', default='data/tags_clean.csv')
    parser.add_argument('--img_dir', default='data/faces')
    parser.add_argument('--root', default='ADLxMLDS_hw4_model')
    parser.add_argument('--save', default='baseline')
    parser.add_argument('--load', default='')
    parser.add_argument('--te_data', default='')
    parser.add_argument('--model_id', default=0, type=int)
    parser.add_argument('--sample_num', default=5, type=int)
    parser.add_argument('--img_save', default='samples')
    parser.add_argument('--noise_max', default=1.0, type=float)
    parser.add_argument('--stage', default=1, type=int)

    return parser.parse_args()



def main(args):

    # create folder to save model, logging, etc.
    if not os.path.exists(args.root):
        os.mkdir(args.root)

    save_path = os.path.join(args.root, args.save)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # initial logger 
    logger = create_logger(save_path, args.todo)

    logger.info(args)

    # load data
    style_list = load_style()
    style2id, id2style = list2dict(style_list)
    assert len(id2style) == 23

    if args.stage == 1:
        from train import train, test
    elif args.stage == 2:
        from train_stage_2 import train, test

    if args.todo == 'train':
        tags_dict, mask_dict = load_tags(args.tags, style2id)
        logger.info('tags shape: %s' % (len(tags_dict)))

        train(args, logger, tags_dict, mask_dict, id2style)
    elif args.todo == 'test':
        _id, _tags = load_te_tags(args.te_data, style2id)

        test(args, logger, _id, _tags, id2style)



if __name__ == '__main__':
    args = parser()
    main(args)