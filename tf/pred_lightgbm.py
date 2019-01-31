# !/usr/bin/env python
import sys
import os

#import tensorflow as tf
import lightgbm as lgb
from dataset import Dataset
from train import FLAGS
from train_lightgbm import *
from sklearn import metrics

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    #tf.logging.set_verbosity(tf.logging.WARN)
    #
    
    gen_features(sys.argv[1],"train_features.csv")

    df_train = pd.read_csv('train_features.csv', header=None, sep='\t')
    y_true = df_train[0].values
    X_train = df_train.drop(0, axis=1).values
    bst = lgb.Booster(model_file='model.txt')
    y_pred=bst.predict(X_train)
    print('Start predicting...')
    with open("light_gbm.txt", 'w') as fo:
        lineno = 1
        for i in range(len(y_pred)):

            fo.write('{}\t{}\n'.format(lineno, y_pred[i]))
            lineno += 1
    
        #if y_pred[i]>0.31:
         #   y_pred[i]=1
       # else:
         #   y_pred[i]=0
    print y_pred
    '''
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    score = metrics.f1_score(y_true, y_pred,average='binary')
    print(precision)
    print(recall)
    print('score: {0:f}'.format(score))
    #main(sys.argv[1], sys.argv[2])
    '''
    '''
    if Upload==False:
        file = open(sys.argv[1]) 
        y_true=[]
        for line in file:
            y_true.append(int(line.strip().split('\t')[3]))
        file.close()
        file = open(sys.argv[2]) 
        y_pred=[]

        for line in file:
            y_pred.append(int(line.strip().split('\t')[1]))
        file.close()
        #print(y_true)
        from sklearn import metrics
        import numpy as np
        #####
        # Do classification task, 
        # then get the ground truth and the predict label named y_true and y_pred
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        score = metrics.f1_score(y_true, y_pred,average='binary')
        print(precision)
        print(recall)
        print('score: {0:f}'.format(score))
        '''