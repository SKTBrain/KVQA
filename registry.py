dictionary_dict = {
    'bert': {'dict': '', # BERT by Google
             'tokenizer': 'sp'},
    'bertrnn': {'dict': '', # BERT by Google
                'tokenizer': 'sp'},
    'glove-rg': {'path': 'word-embeddings/glove/glove.txt', # GloVe by ratsgo
                 'dict': 'dictionary_mecab.kvqa.pkl',
                 'tokenizer': 'mecab',
                 'embedding': 'glove_init.kvqa.npy',
                 'format': 'stanford'},
    'word2vec-pkb': {'path': 'word2vec/ko.tsv', # Word2vec by Kyubyong Park
                     'dict': 'dictionary_kkma.kvqa.pkl',
                     'tokenizer': 'kkma',
                     'embedding': 'word2vec_init.kvqa.npy',
                     'format': 'word2vec'},
    'fasttext-pkb': {'path': 'fasttext/ko.vec', # Word2vec by Kyubyong Park
                     'dict': 'dictionary_kkma.kvqa.pkl',
                     'tokenizer': 'kkma',
                     'embedding': 'ft_init.kvqa.npy',
                     'format': 'fasttext'}
}
