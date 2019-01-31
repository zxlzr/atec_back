# !/usr/bin/env python
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf

from dataset import Dataset
from train import FLAGS
import numpy as np

FLAGS.max_document_length = 34
Upload=False

def main(input_file, output_file,model,flag):
    if flag==1:
        FLAGS.model_dir = '../model/rcnn_50epoch/'
    else:
        FLAGS.model_dir = '../model/cnn/'
    
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
                        #fo.write('{}\t{}\n'.format(lineno, pred))
                        #lineno += 1

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.WARN)
    # Generate batches for one epoch
    dataset = Dataset(data_file=sys.argv[1], is_training=False)
    data = dataset.process_data(data_file=sys.argv[1], sequence_length=FLAGS.max_document_length)
	
    #x1,x2,x3,x4,x5,x6,x7,x8,x9,x10=[],[],[],[],[],[],[],[],[],[]
	
    out1 = main(data, '1.txt','checkpoints/model1-13801.meta',1)
    x1 = np.asarray(out1)
    out2 = main(data, '2.txt','checkpoints/model2-10501.meta',1)
    x2 = np.asarray(out2)   
    out3 = main(data, '3.txt','checkpoints/model3-12901.meta',1)
    x3 = np.asarray(out3)
    out4 = main(data, '4.txt','checkpoints/model4-16301.meta',1)
    x4 = np.asarray(out4)
    out5 = main(data, '5.txt','checkpoints/model5-13601.meta',1)
    x5 = np.asarray(out5)
    out6 = main(data, '6.txt','checkpoints/model6-15101.meta',1)
    x6 = np.asarray(out6)
    out7 = main(data, '7.txt','checkpoints/model7-15801.meta',1)
    x7 = np.asarray(out7)
    out8 = main(data, '8.txt','checkpoints/model8-22701.meta',1)
    x8 = np.asarray(out8)
    out9 = main(data, '9.txt','checkpoints/model9-12501.meta',1)
    x9 = np.asarray(out9) 
    out10 = main(data, '10.txt','checkpoints/model10-13901.meta',1)
    x10 = np.asarray(out10)
    y1=(x1+x2+x3+x4+x5+x6+x7+x8+x9+x10)/10
    out1 = main(data, '1.txt','checkpoints/model1-4935.meta',2)
    x1 = np.asarray(out1)
    out2 = main(data, '2.txt','checkpoints/model2-4935.meta',2)
    x2 = np.asarray(out2)   
    out3 = main(data, '3.txt','checkpoints/model3-4935.meta',2)
    x3 = np.asarray(out3)
    out4 = main(data, '4.txt','checkpoints/model4-4935.meta',2)
    x4 = np.asarray(out4)
    out5 = main(data, '5.txt','checkpoints/model5-4935.meta',2)
    x5 = np.asarray(out5)
    out6 = main(data, '6.txt','checkpoints/model6-4935.meta',2)
    x6 = np.asarray(out6)
    out7 = main(data, '7.txt','checkpoints/model7-4935.meta',2)
    x7 = np.asarray(out7)
    out8 = main(data, '8.txt','checkpoints/model8-4935.meta',2)
    x8 = np.asarray(out8)
    out9 = main(data, '9.txt','checkpoints/model9-4935.meta',2)
    x9 = np.asarray(out9) 
    out10 = main(data, '10.txt','checkpoints/model10-4935.meta',2)
    x10 = np.asarray(out10)
    y2=(x1+x2+x3+x4+x5+x6+x7+x8+x9+x10)/10
    y=(y1+y2)/2
    #print x1
    #print x2	
    #print x3
    #print x4
    #print x8	
    #y=(x1+x2+x3+x4+x8)/5
    for i in range(len(y)):
        if y[i]>0.4:
            y[i]=1
        else:
            y[i]=0
    lineno = 1
    with open(sys.argv[2],'w') as out:
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
