from __future__ import print_function
import _pickle as cPickle
import os
import sys
import json
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']


def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 1./3
    elif occurences == 2:
        return 2./3
    elif occurences == 3:
        return 1.
    else:
        return 1.


def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def preprocess_answer(answer):
    answer = process_punctuation(answer)
    answer = answer.replace(',', '')
    return answer


def get_answer_mode(answers):
    counts = {}
    for ans in answers:
        a = ans['answer']
        counts[a] = counts.get(a, 0) + 1
    max_count = 0
    mode = None
    for c in counts:
        if max_count < counts[c]:
            max_count = counts[c]
            mode = c
    return mode


def filter_answers(answers_dset, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence = {}

    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        gtruth = get_answer_mode(answers)
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        img_id, _ = os.path.splitext(ans_entry['image'])
        occurence[gtruth].add(img_id)
    for answer in list(occurence):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence


def create_ans2label(occurence, name, cache_root='data/cache'):
    """Note that this will also create label2ans.pkl at the same time

    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    utils.create_dir(cache_root)

    cache_file = os.path.join(cache_root, name+'_ans2label.kvqa.pkl')
    cPickle.dump(ans2label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_root, name+'_label2ans.kvqa.pkl')
    cPickle.dump(label2ans, open(cache_file, 'wb'))
    return ans2label


def compute_target(answers_dset, ans2label, split, cache_root='data/cache'):
    """Augment answers_dset with soft score as label

    ***answers_dset should be preprocessed***

    Write result into a cache file
    """
    target = []
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer['answer']
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = []
        scores = []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            score = get_score(answer_count[answer])
            scores.append(score)

        img_id, _ = os.path.splitext(ans_entry['image'])
        target.append({
            'question_id': img_id,
            'image_id': img_id,
            'labels': labels,
            'scores': scores
        })

    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, split + '_target.kvqa.pkl')
    cPickle.dump(target, open(cache_file, 'wb'))
    return target


def get_answer(qid, answers):
    for ans in answers:
        if ans['question_id'] == qid:
            return ans


def get_question(qid, questions):
    for question in questions:
        if question['question_id'] == qid:
            return question


if __name__ == '__main__':
    dataroot = 'data'
    answer_file = os.path.join(dataroot, 'KVQA_annotations.json')
    answers = json.load(open(answer_file, encoding='utf-8'))

    occurence = filter_answers(answers, 1)
    ans2label = create_ans2label(occurence, 'trainval')
    compute_target(answers, ans2label, 'train')
