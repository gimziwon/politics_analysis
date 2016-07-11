#coding: utf-8
import numpy as np
import re
import gensim
import jieba
import json
import ipdb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from gensim.matutils import corpus2csc
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel

import training

def load_json_file(input_path):
    json_file = open(input_path)
    json_array = json.load(json_file)

    return json_array

def extract_from_json(json_array, field_array):
    str_list = list()
    for json_obj in json_array:
        for field_name in field_array:
            str_list.append(json_obj[field_name])

    return str_list

def extract_from_json_with_answer(json_array, field_array):
    str_list = list()
    answer = list()

    for json_obj in json_array:
        for field_name in field_array:
            str_list.append(json_obj[field_name])
        answer.append(json_obj['answer'])

    return str_list, answer

def tokenize(str_list):
    jieba.set_dictionary('dict.txt.big.txt')

    texts = list()
    for comment in str_list:
        comment_tokens = jieba.cut(comment, cut_all = False)
        texts.append(" ".join(comment_tokens).split(" "))

    return texts

def remove_stop_words(texts):
    with open('stop_words_chinese.txt') as reader:
        stop_word_list = reader.read().splitlines()

    for comment in texts:
        for word in comment[:]:
            if word in stop_word_list:
                comment.remove(word)

    return texts

def convert_texts_to_corpus(texts, dictionary):
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus

def main():

    input_path = 'dataset/taipei_city.json'
    data, label = load_json(input_path)
    data = remove_stop_words(data)

    X, y = Tfidf(data, label, data)
    model.cross_validation(X, y, 5, DummyClassifier)
    model.cross_validation(X, y, 5, LogisticRegression, {'C':1e+5})

if __name__ == '__main__':
    main()