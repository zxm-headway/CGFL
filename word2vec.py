import pickle
import nltk
import re
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from ordered_set import OrderedSet
from gensim.models import Word2Vec,KeyedVectors
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import random


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN





def gen_corpus():
    input1 = './data/post'
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []
    doc_val_list = []

    f = open(input1 + '.txt', 'r', encoding='latin1')

    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
        elif temp[1].find('val') != -1:
            doc_val_list.append(line.strip())    
    f.close()

    doc_content_list = []
    f = open(input1 + '.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    random.shuffle(train_ids)

    train_ids_str = '\n'.join(str(index) for index in train_ids)

    val_ids = []
    for val_name in doc_val_list:
        val_id = doc_name_list.index(val_name)
        val_ids.append(val_id)
    # random.shuffle(val_ids)


    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    # print(test_ids)
    # random.shuffle(test_ids)

    test_ids_str = '\n'.join(str(index) for index in test_ids)

    ids = train_ids + val_ids + test_ids
    # print(ids)
    # print(len(ids))

    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])

    label_set = OrderedSet()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])


    label_list = list(label_set)
    labels = []
    for one in shuffle_doc_name_list:
        entry = one.split('\t')
        labels.append(label_list.index(entry[-1]))


    # shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    # shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
    word_freq = {}
    word_set = OrderedSet()
    lemmatizer = WordNetLemmatizer()
    for doc_words in shuffle_doc_words_list:

        # todo 开始用ntlk进行分词
        # words = doc_words.split()
        # words = nltk.word_tokenize(doc_words)

        pos_tagged = pos_tag(nltk.word_tokenize(doc_words))
        words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged]

        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    print("Vocab size: {}".format(vocab_size))

    word_doc_list = {}
    # lemmatize = WordNetLemmatizer()
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        # words = doc_words.split()
        # words = nltk.word_tokenize(doc_words)
        pos_tagged = pos_tag(nltk.word_tokenize(doc_words))
        words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged]
        appeared = set()

        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    id_word_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
        id_word_map[i] = vocab[i]

    return vocab,shuffle_doc_words_list



def get_sentences():
    _,data =gen_corpus()
    sentences = []

    
    # 做词性还原
    lemmatizer = WordNetLemmatizer()

    for i in range(len(data)):
        doc_words = data[i]
        pos_tagged = pos_tag(nltk.word_tokenize(doc_words))
        words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged]
        sentences.append(words)
    return sentences


# 得到文本的词向量
def get_word_embedding(path):
    vocab, _ = gen_corpus()
    # 词向量的维度
    embedding_size = 300
    # 词向量矩阵
    embedding_matrix = np.zeros((len(vocab), embedding_size))
    # 读取word2vec模型
    word2vec = KeyedVectors.load_word2vec_format(path, binary=False)
    for i in range(len(vocab)):
        if vocab[i] in word2vec:
            # 按照索引将词向量存储到embedding_matrix中
            embedding_matrix[i] = word2vec[vocab[i]]
    return embedding_matrix


# 暂时不用
class Word2Vec_train(nn.Module):
    def __init__(self, sentences, embedding_size,window_size):
        super(Word2Vec_train, self).__init__()
        self.model = Word2Vec(sentences=sentences, vector_size=embedding_size, window=window_size, min_count=1, workers=4)
        self.word_path = f'./word_embeddings/word2vec_dim768_{window_size}.kv'
        self.train_words()
        self.word2vec = KeyedVectors.load_word2vec_format(self.word_path, binary=False)

    def train_words(self):
      if not os.path.exists(self.word_path):
        self.model.wv.save_word2vec_format(self.word_path, binary=False)
      else:
        print('word2vec model already exists')

    def word2vec_path(self):
        return self.word_path

    def forward(self, x):
        return self.woed2vec[x]


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    # 窗口大小
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument("--embed_size", type=int, default=768)
    # parser.add_argument("--max_len", default=4096, type=int)
    # parser.add_argument("--hidden_size", type=int, default=512)
    return parser.parse_args(args)


def main(args):
    # 词向量路径

    sentences = get_sentences()
    Word2Vec_embedings = Word2Vec_train(sentences, embedding_size=args.embed_size, window_size=args.window_size)
    path = Word2Vec_embedings.word2vec_path()


if __name__ == '__main__':
    args = parse_args()
    main(args)
