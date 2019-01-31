# !/usr/bin/env python
import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import numpy as np
from dataset import Dataset
from train import FLAGS

FLAGS.model_dir = '../model/rcnn_50_cv/'
FLAGS.max_document_length = 34
Upload=False
Train=False

def main(input_file, output_file,model):
  
    graph = tf.Graph()
    with graph.as_default():  # with tf.Graph().as_default() as g:
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            # saver = tf.train.Saver(tf.global_variables())
            meta_file = os.path.abspath(os.path.join(FLAGS.model_dir, model))
            #print meta_file 
            new_saver = tf.train.import_meta_graph(meta_file)
            #new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'checkpoints')))
            new_saver.restore(sess, meta_file.split(".")[0])
            # graph = tf.get_default_graph()

            # Get the placeholders from the graph by name
            # input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x1 = graph.get_tensor_by_name("input_x1:0")  # Tensor("input_x1:0", shape=(?, 15), dtype=int32)
            input_x2 = graph.get_tensor_by_name("input_x2:0")
            dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
            # Tensors we want to evaluate
            y_pred = graph.get_tensor_by_name("metrics/y_pred:0")
            # vars = tf.get_collection('vars')
            # for var in vars:
            #     print(var)

            e = graph.get_tensor_by_name("cosine:0")

            batches = dataset.batch_iter(input_file, FLAGS.batch_size, 1, shuffle=False)

            #with open(output_file, 'w') as fo:
            print("\nPredicting...\n")
            lineno = 1
            out=[]
            for batch in batches:
                    #print batch
                    #exit(1)
                x1_batch, x2_batch, _, _ = zip(*batch)
                y_pred_ = sess.run([e], {input_x1: x1_batch, input_x2: x2_batch, dropout_keep_prob: 1.0})
                for pred in y_pred_[0]:
                    out.append(pred)
            return out
                        #fo.write('{}    {}\n'.format(lineno, pred))
                        #lineno += 1

if __name__ == '__main__':

    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.WARN)
    # Generate batches for one epoch
    dataset = Dataset(data_file=sys.argv[1], is_training=False)
    data = dataset.process_data(data_file=sys.argv[1], sequence_length=FLAGS.max_document_length)
    
    #x1,x2,x3,x4,x5,x6,x7,x8,x9,x10=[],[],[],[],[],[],[],[],[],[]
    
    out1 = main(data, '1.txt','checkpoints/model1-8501.meta')
    x1 = np.asarray(out1)
    out2 = main(data, '2.txt','checkpoints/model2-19301.meta')
    x2 = np.asarray(out2)   
    out3 = main(data, '3.txt','checkpoints/model3-10601.meta')
    x3 = np.asarray(out3)
    out4 = main(data, '4.txt','checkpoints/model4-11901.meta')
    x4 = np.asarray(out4)
    out5 = main(data, '5.txt','checkpoints/model5-12201.meta')
    x5 = np.asarray(out5)
    out6 = main(data, '6.txt','checkpoints/model6-12901.meta')
    x6 = np.asarray(out6)
    out7 = main(data, '7.txt','checkpoints/model7-10401.meta')
    x7 = np.asarray(out7)
    out8 = main(data, '8.txt','checkpoints/model8-8101.meta')
    x8 = np.asarray(out8)
    out9 = main(data, '9.txt','checkpoints/model9-10601.meta')
    x9 = np.asarray(out9) 
    out10 = main(data, '10.txt','checkpoints/model10-11701.meta')
    x10 = np.asarray(out10)

    X_train=np.vstack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)).T
    #X_train=np.vstack((x1,x2,x3,x4,x5,x6,x7,x8,x9)).T
    #X_train=np.random.rand(102477,5)
    y_train=[]
    if Train==True:
        for line in open(sys.argv[1]):
            y_train.append(line.strip().decode('utf-8').split('\t')[3])
        y_train=np.asarray(y_train)
        print(y_train)
        lgb_train = lgb.Dataset(X_train, y_train)
        params = {
        'max_depth':6,

        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'l2', 'auc'},
        'num_leaves': 64,
        'learning_rate': 0.05,
        'scale_pos_weight':2,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
        }
        #params['is_unbalance']='true'
        #params['metric'] = 'auc'

        gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=50,
                    #valid_sets=lgb_eval,
                    #early_stopping_rounds=10
                    )
        gbm.save_model('lightgbm.txt')
    #bst.predict(X_train, num_iteration=gbm.best_iteration)
    print('Start predicting...')
# predict
    bst = lgb.Booster(model_file='lightgbm.txt')
    y_pred=bst.predict(X_train)
#y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    for i in range(len(y_pred)):
        if y_pred[i]>0.5:
            y_pred[i]=1
        else:
            y_pred[i]=0
    y=y_pred
    print y
    #score = metrics.f1_score(y_train, y_pred,average='binary')

    #print('score: {0:f}'.format(score))
    #y=(x1+x2+x3+x4+x5+x6+x7+x8+x9+x10)/10
    #print x1
    #print x2   
    #print x3
    #print x4
    #print x8   
    #y=(x1+x2+x3+x4+x8)/5
    with open(sys.argv[2],'w') as out:
        lineno = 1
        for pred in y:
            out.write('{}\t{}\n'.format(lineno, int(pred)))
            lineno += 1
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

        #####
        # Do classification task, 
        # then get the ground truth and the predict label named y_true and y_pred
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        score = metrics.f1_score(y_true, y_pred,average='binary')
        print(precision)
        print(recall)
        print('score: {0:f}'.format(score))
