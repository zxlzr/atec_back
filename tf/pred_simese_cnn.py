# !/usr/bin/env python
import sys
import os

import tensorflow as tf

from dataset import Dataset
from train import FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
FLAGS.model_dir = '../model/test1/'
FLAGS.max_document_length = 34
Upload=False

def main(input_file, output_file):
  
    graph = tf.Graph()
    with graph.as_default():  # with tf.Graph().as_default() as g:
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            # saver = tf.train.Saver(tf.global_variables())
            meta_file = os.path.abspath(os.path.join(FLAGS.model_dir, 'checkpoints/model-8700.meta'))
            new_saver = tf.train.import_meta_graph(meta_file)
            #new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'checkpoints')))
            new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'checkpoints')))
            # graph = tf.get_default_graph()

            # Get the placeholders from the graph by name
            # input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x1 = graph.get_tensor_by_name("input_x1:0")  # Tensor("input_x1:0", shape=(?, 15), dtype=int32)
            input_x2 = graph.get_tensor_by_name("input_x2:0")
            dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
            #dropout_emb = graph.get_tensor_by_name("dropout_emb:0")
            # Tensors we want to evaluate
            y_pred = graph.get_tensor_by_name("metrics/y_pred:0")
            # vars = tf.get_collection('vars')
            # for var in vars:
            #     print(var)

            e = graph.get_tensor_by_name("cosine:0")

            # Generate batches for one epoch
            dataset = Dataset(data_file=input_file, is_training=False)
            data = dataset.process_data(data_file=input_file, sequence_length=FLAGS.max_document_length)
            batches = dataset.batch_iter(data, FLAGS.batch_size, 1, shuffle=False)
            with open(output_file, 'w') as fo:
                print("\nPredicting...\n")
                lineno = 1
                for batch in batches:
                    #print batch
                    #exit(1)
                    x1_batch, x2_batch, _, _ = zip(*batch)
                    y_pred_ = sess.run([y_pred], {input_x1: x1_batch, input_x2: x2_batch, dropout_keep_prob: 1.0})
                    for pred in y_pred_[0]:
                        fo.write('{}\t{}\n'.format(lineno, int(pred)))
                        lineno += 1

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.WARN)
    main(sys.argv[1], sys.argv[2])
    
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
