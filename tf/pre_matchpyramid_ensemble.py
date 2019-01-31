# !/usr/bin/env python
import sys
import os

import tensorflow as tf

from dataset import Dataset
from train import FLAGS
import numpy as np 

FLAGS.model_dir = '../model/cnn2/'
FLAGS.max_document_length = 50
Upload=True
def dynamic_pooling_index(len1, len2, max_len1, max_len2):


    def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):

        stride1 = 1.0 * max_len1 / (len1_one+0.001)
        stride2 = 1.0 * max_len2 / (len2_one+0.001)
        idx1_one = [int(i / stride1) for i in range(max_len1)]
        idx2_one = [int(i / stride2) for i in range(max_len2)]
        mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
        index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2, 1, 0))
        return index_one

    index = []
    for i in range(len(len1)):
        index.append(dpool_index_(i, len1[i], len2[i], max_len1, max_len2))
    return np.array(index)

def main(input_file, output_file):
  
    graph = tf.Graph()
    with graph.as_default():  # with tf.Graph().as_default() as g:
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            # saver = tf.train.Saver(tf.global_variables())
            meta_file = os.path.abspath(os.path.join(FLAGS.model_dir, 'checkpoints/model-16020.meta'))
            new_saver = tf.train.import_meta_graph(meta_file)
            #new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'checkpoints')))
            new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'checkpoints')))
            # graph = tf.get_default_graph()

            # Get the placeholders from the graph by name
            # input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x1 = graph.get_tensor_by_name("input_x1:0")  # Tensor("input_x1:0", shape=(?, 15), dtype=int32)
            input_x2 = graph.get_tensor_by_name("input_x2:0")
            x_scene_len = graph.get_tensor_by_name("x_scene_len:0")
            x_title_len = graph.get_tensor_by_name("x_title_len:0")
            dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
            # Tensors we want to evaluate
            y_pred = graph.get_tensor_by_name("metrics/y_pred:0")
            dpool_index = graph.get_tensor_by_name("dpool_index:0")
            probs = graph.get_tensor_by_name("probs:0")
            # vars = tf.get_collection('vars')
            # for var in vars:
            #     print(var)

            #e = graph.get_tensor_by_name("cosine:0")

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
                    x1_batch, x2_batch, seqlen1, seqlen2 = zip(*batch)
                    dpool = dynamic_pooling_index(seqlen1, seqlen2, 50, 50)
                    y_pred_ = sess.run([probs], {dpool_index: dpool,input_x1: x1_batch, input_x2: x2_batch, dropout_keep_prob: 1.0,x_scene_len: seqlen1,x_title_len: seqlen2,})

                    for pred in y_pred_[0]:
                        fo.write('{}\t{}\n'.format(lineno, pred[1]))
                        lineno += 1

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.WARN)
    main(sys.argv[1], "matchpyramid.txt")
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
