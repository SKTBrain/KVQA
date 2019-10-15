"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import _pickle as cPickle
import os
import json
import itertools
import re
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import numpy as np
import torch
from torch.utils.data import Dataset
from konlpy.tag import Mecab, Kkma
from pytorch_pretrained_bert import BertTokenizer

import utils


COUNTING_ONLY = False

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word, sp=None):
        if sp is None:
            sentence = sentence.lower()
            sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
            words = sentence.split()
        else:
            words = sp(sentence)
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_kvqa(dataroot, name, img_id2val, label2ans, drop_img_inds=[]):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    question_path = os.path.join(
        os.path.join(dataroot, 'KVQA_annotations.json'))
    questions = sorted(json.load(open(question_path, encoding='utf-8')), key=lambda x: x['image'])
    idx2type = None
    type2idx = None
    if 'test'!=name[:4]: # train, val
        answer_path = os.path.join(dataroot, 'cache', '%s_target.kvqa.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        type2idx = {}
        idx2type = []
        entries = []
        for question, answer in zip(questions, answers):
            q_id, _ = os.path.splitext(question['image'])
            question['question_id'] = q_id
            question['image_id'] = q_id
            utils.assert_eq(q_id, answer['question_id'])
            img_id = q_id
            if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
                image_index = img_id2val[img_id]
                if image_index in drop_img_inds:
                    continue
                entry = _create_entry(image_index, question, answer)
                entry['answerable'] = int(question['answerable'])
                if question['answer_type'] == '네/아니요':
                    question['answer_type'] = 'yes/no'
                if question['answer_type'] not in type2idx:
                    type2idx[question['answer_type']] = len(idx2type)
                    idx2type.append(question['answer_type'])
                entry['answer_type'] = type2idx[question['answer_type']]
                entries.append(entry)
    else: # test
        entries = []
        for question in questions:
            img_id = int(question['image'].split('.')[0].split('_')[-1])
            q_id = img_id
            if not COUNTING_ONLY or is_howmany(question['question'], None, None):
                question['question_id'] = q_id
                question['image_id'] = q_id
                entry = _create_entry(img_id2val[img_id], question, None)
                entries.append(entry)
    return entries, type2idx, idx2type


class KvqaFeatureDataset(Dataset):
    def __init__(self, split, dictionary, dataroot='data', tokenizer='sp', drop_zero_detection=True):
        super(KvqaFeatureDataset, self).__init__()
        assert split in ['train']
        self.dataroot = dataroot

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.kvqa.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.kvqa.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s_imgid2idx.kvqa.pkl' % split),
                 'rb'))

        h5_path = os.path.join(dataroot, '%s_kvqa.hdf5' % split)

        print('loading features from h5 file')
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
            self.pos_boxes = np.array(hf.get('pos_boxes'))

        if drop_zero_detection:
            num_boxes = self.pos_boxes[:, 1] - self.pos_boxes[:, 0]
            drop_img_inds = np.where(num_boxes == 0)[0]

        self.entries, self.type2idx, self.idx2type = _load_kvqa(dataroot, split, self.img_id2idx, self.label2ans, drop_img_inds)

        if tokenizer == 'sp':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
            self.dictionary = self.tokenizer.vocab
        elif tokenizer == 'mecab':
            self.tokenizer = Mecab()
        elif tokenizer == 'kkma':
            self.tokenizer = Kkma()

        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(1)
        self.s_dim = self.spatials.size(1)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            if hasattr(self.tokenizer, 'morphs'):
                tokens = self.tokenizer.morphs(entry['question'].replace('.', ''))
                tokens = [self.dictionary.word2idx[token] for token in tokens[:max_length]]
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                    tokens = tokens + padding
            elif hasattr(self.tokenizer, 'tokenize'):
                tokens = self.tokenizer.tokenize(entry['question'])
                tokens = [self.dictionary[token] for token in tokens[:max_length]]
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary['[PAD]']] * (max_length - len(tokens))
                    tokens = tokens + padding
            else:
                tokens = self.tokenizer(entry['question'])
                tokens = [self.dictionary(token) for token in tokens[:max_length]]
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary('[PAD]')] * (max_length - len(tokens))
                    tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']

        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, spatials, question, target, entry['answerable'], entry['answer_type']
        else:
            return features, spatials, question, question_id, 0., -1

    def __len__(self):
        return len(self.entries)
