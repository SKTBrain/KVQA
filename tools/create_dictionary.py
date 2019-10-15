from __future__ import print_function
import os
import sys
import json
import argparse
import re
import numpy as np
from konlpy.tag import Mecab, Kkma
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary
from registry import dictionary_dict

def tokenize_kvqa(s):
    return s.replace('.', ' ').replace('  ', ' ')

def create_dictionary(dataroot, tk='mecab'):
    dictionary = Dictionary()
    if tk == 'mecab':
        tokenizer = Mecab()
    elif tk == 'kkma':
        tokenizer = Kkma()
    files = [
        'KVQA_annotations.json',
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path, encoding='utf-8'))
        for q in qs:
            dictionary.tokenize(tokenize_kvqa(q['question']), True, tokenizer.morphs)
    return dictionary


def create_embedding_init(idx2word, glove_file, format='stanford'):
    word2emb = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    if format == 'stanford':
        emb_dim = len(entries[0].split(' ')) - 1
    elif format == 'fasttext':
        entries = entries[1:]
        emb_dim = len(entries[0].strip().split(' ')) - 1
    elif format == 'word2vec':
        entries = ''.join(entries).replace('\n', '').split(']')
        _, word, vec = entries[0].split('\t')
        vals = list(map(float, re.sub('\s+', ' ', vec.replace('[', '')).strip().split(' ')))
        emb_dim = len(vals)
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        if format in ('stanford', 'fasttext'):
            vals = entry.strip().split(' ')
            word = vals[0]
            vals = list(map(float, vals[1:]))
        else:
            if entry == '':
                continue
            _, word, vec = entry.split('\t')
            vals = list(map(float, re.sub('\s+', ' ', vec.replace('[', '')).strip().split(' ')))
        word2emb[word] = np.array(vals)
    notFound = 0
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            notFound += 1
            print(word)
            continue
        weights[idx] = word2emb[word]
    print('not found %d/%d words' % (notFound, len(idx2word)))
    return weights, word2emb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', default='glove-rg', type=str)
    args = parser.parse_args()

    dataroot = 'data'
    emb = dictionary_dict[args.embedding]
    d = create_dictionary(dataroot, emb['tokenizer'])
    dict_path = os.path.join(dataroot, emb['dict'])
    d.dump_to_file(dict_path)

    d = Dictionary.load_from_file(dict_path)
    embedding_path = os.path.join(dataroot, emb['path'])
    weights, word2emb = create_embedding_init(d.idx2word, embedding_path, emb['format'])
    np.save(os.path.join(dataroot, emb['embedding']), weights)



