#coding: utf-8
import numpy as np
import pandas as pd
import evaluation

from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

def cross_validation(X, y, n_fold, model_class, params = {}):
    skf = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=1234)
    class_num = len(np.unique(y))

    train_table = pd.DataFrame()
    test_table = pd.DataFrame()
    for train_index, test_index in skf:
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        model = model_class()
        model.set_params(**params)
        model.fit(X_train, y_train)

        train_pred_prob = model.predict_proba(X_train)
        train_pred_label = model.predict(X_train)
        train_stats = evaluation.get_statistics(train_pred_prob, 
            train_pred_label, y_train)

        test_pred_prob = model.predict_proba(X_test)
        test_pred_label = model.predict(X_test)
        test_stats = evaluation.get_statistics(test_pred_prob, 
            test_pred_label, y_test)

        train_frame = pd.DataFrame(train_stats, index=[0])
        test_frame = pd.DataFrame(test_stats, index=[0])

        train_table = train_table.append(train_frame)
        test_table = test_table.append(test_frame)
    
    train_table_mean = train_table.mean()
    test_table_mean = test_table.mean()

    result_table = pd.DataFrame(index=[short_class_name(model_class)])
    for col in train_table:
        result_table['train_'+col] = train_table_mean[col]
        result_table['test_'+col] = test_table_mean[col]

    return result_table

def short_class_name(model_class):
    if model_class == RandomForestClassifier:
        return 'RF'
    elif model_class == LogisticRegression:
        return 'LG'
    elif model_class == DummyClassifier:
        return 'DM'
