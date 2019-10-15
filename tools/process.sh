## This code is modified from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa

## Process data
## Notice that 10-100 adaptive bottom-up attention features are used.
python tools/compute_softscore.py
python tools/create_dictionary.py --embedding glove-rg
python tools/create_dictionary.py --embedding word2vec-pkb
python tools/create_dictionary.py --embedding fasttext-pkb
python tools/adaptive_detection_features_converter.py