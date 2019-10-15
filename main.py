"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import KvqaFeatureDataset, Dictionary
import base_model
from train import train
import utils
from registry import dictionary_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--q_emb', type=str, default='fasttext-pkb', choices=dictionary_dict.keys())
    parser.add_argument('--op', type=str, default='')
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--use_both', action='store_true', help='use both train/val datasets to train?')
    parser.add_argument('--finetune_q', action='store_true', help='finetune question embedding?')
    parser.add_argument('--on_do_q', action='store_true', help='turn on dropout of question embedding?')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/ban-kvqa')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    parser.add_argument('--num_cv', type=int, default=5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if 'bert' in args.q_emb:
        dictionary = None
    else:
        dictionary_path = os.path.join(args.dataroot, dictionary_dict[args.q_emb]['dict'])
        dictionary = Dictionary.load_from_file(dictionary_path)
    dset = KvqaFeatureDataset('train', dictionary, tokenizer=dictionary_dict[args.q_emb]['tokenizer'])

    batch_size = args.batch_size

    num_val = int(len(dset) // args.num_cv) # Number of validation dataset
    scores = []
    bounds = []
    train_n_types = []
    val_n_types = []
    train_type_scores = []
    val_type_scores = []

    gen_cv = utils.gen_cv_dataset(dset, args.num_cv, num_val) # Cross-validation dataset generator
    for i, (train_dset, val_dset) in enumerate(gen_cv):
        logger.write('=' * 50)
        logger.write('Cross Validation {}'.format(i))
        constructor = 'build_%s' % args.model
        model = getattr(base_model, constructor)(dset, args.num_hid, args.op, args.gamma,
                                                 args.q_emb, args.on_do_q, args.finetune_q).cuda()
        if 'bert' not in args.q_emb:
            model.q_emb.w_emb.init_embedding(os.path.join(args.dataroot, dictionary_dict[args.q_emb]['embedding']))

        model = nn.DataParallel(model).cuda()

        optim = None
        epoch = 0

        # load snapshot
        if args.input is not None:
            print('loading %s' % args.input)
            model_data = torch.load(args.input)
            model.load_state_dict(model_data.get('model_state', model_data))
            optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
            optim.load_state_dict(model_data.get('optimizer_state', model_data))
            epoch = model_data['epoch'] + 1

        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)
        val_score, bound, train_n_type, val_n_type, train_type_score, val_type_score = \
            train(model, train_loader, eval_loader, args.epochs, os.path.join(args.output, 'cv{}'.format(i)), optim, epoch, logger)

        scores.append(val_score)
        bounds.append(bound)
        train_n_types.append(train_n_type)
        val_n_types.append(val_n_type)
        train_type_scores.append(train_type_score)
        val_type_scores.append(val_type_score)

    logger.write('\nMean val upper bound: {}'.format(sum(bounds) / args.num_cv))
    logger.write('\nMean val score: {}'.format(sum(scores) / args.num_cv))
    logger.write('\nAnswer type: '+', '.join(dset.idx2type))
    logger.write('\n'+'Number of examples for each type on train: {}'.format(torch.stack(train_n_types).mean(0)))
    logger.write('\n'+'Number of examples for each type on val: {}'.format(torch.stack(val_n_types).mean(0)))
    logger.write('\n'+'Mean score for each type on val: {}'.format(torch.stack(val_type_scores).mean(0)))
