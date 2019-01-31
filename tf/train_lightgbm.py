#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/21
from __future__ import unicode_literals
from __future__ import division
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import codecs
from sklearn import metrics

def len_diff(s1, s2):
    return abs(len(s1) - len(s2))


def len_diff_ratio(s1, s2):
    return 2 * abs(len(s1) - len(s2)) / (len(s1) + len(s2))


def shingle_similarity(s1, s2, size=1):
    """Shingle similarity of two sentences."""
    def get_shingles(text, size):
        shingles = set()
        for i in range(0, len(text) - size + 1):
            shingles.add(text[i:i + size])
        return shingles

    def jaccard(set1, set2):
        x = len(set1.intersection(set2))
        y = len(set1.union(set2))
        return x, y

    x, y = jaccard(get_shingles(s1, size), get_shingles(s2, size))
    return x / float(y) if (y > 0 and x > 2) else 0.0


def common_words(s1, s2):
    s1_common_cnt = len([w for w in s1 if w in s2])
    s2_common_cnt = len([w for w in s2 if w in s1])
    return (s1_common_cnt + s2_common_cnt) / (len(s1) + len(s2))


def tf_idf():
    pass


def wmd():
    pass
def gen_features(infile,outfile):
    file = codecs.open(infile,"r","utf-8")
    fo = codecs.open(outfile,"w","utf-8")
    i=0       
    for line in file:
        s1=line.strip().split('\t')[1]
        s2=line.strip().split('\t')[2]
        f1=common_words(s1,s2)
        f2=shingle_similarity(s1, s2)
        f3=shingle_similarity(s1, s2,2)
        f4=shingle_similarity(s1, s2,3)
        f5=len_diff(s1, s2)
        #f6=jaccard(s1, s2)
        f7=len_diff_ratio(s1,s2)
        if line.strip().split('\t')[3]!="":
            y=line.strip().split('\t')[3]
        fo.write(y+"\t"+str(f1)+"\t"+str(f2)+"\t"+str(f3)+"\t"+str(f4)+"\t"+str(f5)+"\t"+str(7)+"\r\n")


if __name__ == '__main__':
    gen_features("../data/atec_nlp_sim_train.csv","../data/train_features.csv")
    df_train = pd.read_csv('../data/train_features.csv', header=None, sep='\t')
    y_train = df_train[0].values
    X_train = df_train.drop(0, axis=1).values
    print X_train
    exit(1)
    #print y_train
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_train, y_train, reference=lgb_train)
    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)

    print('Save model...')
# save model to file
    gbm.save_model('model.txt')
    #bst.predict(X_train, num_iteration=gbm.best_iteration)
    print('Start predicting...')
# predict
    y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    for i in range(len(y_pred)):
        if y_pred[i]>0.31:
            y_pred[i]=1
        else:
            y_pred[i]=0
    score = metrics.f1_score(y_train, y_pred,average='binary')

    print('score: {0:f}'.format(score))
# eval
    #print('The rmse of prediction is:', mean_squared_error(y_train, y_pred) ** 0.5)

    #s1 = '怎么更改花呗手机号码'
    #s2 = '我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号'
    #print(len_diff(s1, s2))
    #print(shingle_similarity(s1, s2))
    #print(shingle_similarity(s1, s2, 2))
    #print(shingle_similarity(s1, s2, 3))

