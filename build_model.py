#coding: utf-8
import numpy as np
import pandas as pd
import gensim
import sys
import json
import re
import jieba
import ipdb
import pickle

import preprocess
import training

from gensim.matutils import corpus2csc
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def convert_to_X_y(model_class, params, data, label):
	
	model = model_class(**params)
	mat = model[data]

	X = corpus2csc(mat)
	y = np.array(label)

	return X.T, y

def train_with_dummy(result_table, X, y, preprocess_model_name):
	params = {'strategy': 'most_frequent', 'random_state': 1234}
	experiement_result = training.cross_validation(X, y, n_fold=5, 
		model_class=DummyClassifier, params = params)
	experiement_result.insert(0, 'preprocess_model', preprocess_model_name)
	experiement_result.insert(0, 'params', str(('strategy', params['strategy'])))

	result_table = result_table.append(experiement_result)

	return result_table

def train_with_logistic_regression(result_table, X, y, preprocess_model_name):
	for C in range(-5, 6):
		params = {'C': 10**C, 'random_state': 1234}
		experiement_result = training.cross_validation(X, y, n_fold=5, 
			model_class=LogisticRegression, params = params)
		experiement_result.insert(0, 'preprocess_model', preprocess_model_name)
		experiement_result.insert(0, 'params', str(('C', params['C'])))

		result_table = result_table.append(experiement_result)

	return result_table

def train_with_random_forest(result_table, X, y, preprocess_model_name):
	for max_depth in np.arange(2,30.5,0.5):
		params = {'max_depth': max_depth, 'n_estimators': 1000, 'random_state': 1234}
		experiement_result = training.cross_validation(X, y, n_fold=5, 
			model_class=RandomForestClassifier, params=params)
		experiement_result.insert(0, 'preprocess_model', preprocess_model_name)
		experiement_result.insert(0, 'params', str(('max_depth', params['max_depth'])))

		result_table = result_table.append(experiement_result)

	return result_table
    
def main():

	input_path = 'dataset/taipei_city.json'
	json_array = preprocess.load_json_file(input_path)

	field_array = ['content']
	str_list, answer = preprocess.extract_from_json_with_answer(json_array['data'], field_array)

	texts = preprocess.tokenize(str_list)
	removed_texts = preprocess.remove_stop_words(texts)
    
	#dictionary = pickle.load(open('dictionary.obj', 'rb'))
	dictionary = corpora.Dictionary(removed_texts)
	data_corpus = preprocess.convert_texts_to_corpus(removed_texts, dictionary)

	#corpus = pickle.load(open('corpus.obj', 'rb'))
	
	result_table = pd.DataFrame()

	# preprocess with Tfidf model
	params = {"corpus": data_corpus}
	X, y = convert_to_X_y(TfidfModel, params, data_corpus, answer)
	result_table = train_with_dummy(result_table, X, y, 'tfidf')
	result_table = train_with_random_forest(result_table, X, y, 'tfidf')
	result_table = train_with_logistic_regression(result_table, X, y, 'tfidf')

	'''
	# preprocess with lda model
	for num_topics in [10, 50, 100, 150, 200]:
		params = {"corpus": data_corpus, "num_topics": num_topics}
		X, y = convert_to_X_y(LdaModel, params, data_corpus, answer)
		result_table = train_with_dummy(result_table, X, y, 'lda_'+str(params['num_topics']))
		result_table = train_with_random_forest(result_table, X, y, 'lda_'+str(params['num_topics']))
		result_table = train_with_logistic_regression(result_table, X, y, 'lda_'+str(params['num_topics']))
	
	# preprocess with lsi model
	for num_topics in [10, 50, 100, 150, 200]:
		params = {"corpus": data_corpus, "num_topics": num_topics}
		X, y = convert_to_X_y(LsiModel, params, data_corpus, answer)
		result_table = train_with_dummy(result_table, X, y, 'lsi_'+str(params['num_topics']))
		result_table = train_with_random_forest(result_table, X, y, 'lsi_'+str(params['num_topics']))
		result_table = train_with_logistic_regression(result_table, X, y, 'lsi_'+str(params['num_topics']))

	'''
	output_file = sys.argv[1]
	result_table.to_csv(output_file, sep='\t')
	
def build_corpus_dictionary():

    input_path = 'dataset/comments_array.json'
    json_array = preprocess.load_json_file(input_path)
    
    field_array = ['content']
    str_list = preprocess.extract_from_json(json_array, field_array)

    texts = preprocess.tokenize(str_list)
    removed_texts = preprocess.remove_stop_words(texts)
    
    dictionary = corpora.Dictionary(texts)
    corpus = preprocess.convert_texts_to_corpus(removed_texts, dictionary)

    import ipdb; ipdb.set_trace()
    

if __name__ == '__main__':
    main()