# coding:utf-8
import tensorflow as tf
import os
import sys
import random
import codecs
from dataset import Dataset
from metrics import *
import datetime

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# hyper parameters
#tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability(Default: 0.5)")
#tf.flags.DEFINE_integer("train_epoch", 1, "number of epochs to run when training(Default: 5")
#tf.flags.DEFINE_integer("embedding_dim", 100, "word embedding dimension(Default: 100)")
#tf.flags.DEFINE_integer("batch_size", 128, "batch_size(Defualt: 128)")
tf.flags.DEFINE_integer("psize1", 3, "psize1(Defualt: 10)")
tf.flags.DEFINE_integer("psize2", 3, "psize2(Defualt: 40)")

#tf.flags.DEFINE_string("task", "train", "task(Default:train)")
tf.flags.DEFINE_boolean("Submit", True, "Submit or test")
# Data loading params
tf.flags.DEFINE_string("data_file", "../data/atec_nlp_sim_train_filtered.csv", "Training data file path.")
tf.flags.DEFINE_float("val_percentage", .2, "Percentage of the training data to use for validation. (default: 0.2)")
tf.flags.DEFINE_integer("random_seed", 123, "Random seed to split train and test. (default: None)")
tf.flags.DEFINE_integer("max_document_length", 50, "Max document length of each train pair. (default: 15)")

# Model Hyperparameters
tf.flags.DEFINE_string("model_class", "siamese", "Model class, one of {`siamese`, `textrcnn`}")
tf.flags.DEFINE_string("model_type", "cnn", "Model type, one of {`cnn`, `rnn`, `rcnn`} (default: rnn)")
tf.flags.DEFINE_boolean("char_model", True, "Character based syntactic model. if false, word based semantic model. (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character/word embedding (default: 300)")
tf.flags.DEFINE_string("word_embedding_type", "non-static", "One of `rand`, `static`, `non-static`, random init(rand) vs pretrained word2vec(static) vs pretrained word2vec + training(non-static)")
# If include CNN
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5,6", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
# If include RNN
tf.flags.DEFINE_string("rnn_cell", "lstm", "Rnn cell type, lstm or gru (default: lstm)")
tf.flags.DEFINE_integer("hidden_units", 100, "Number of hidden units (default: 50)")
tf.flags.DEFINE_integer("num_layers", 2, "Number of rnn layers (default: 3)")
tf.flags.DEFINE_float("clip_norm", None, "Gradient clipping norm value set None to not use (default: 5)")
tf.flags.DEFINE_boolean("use_attention", False, "Whether use self attention or not (default: False)")
# Common
tf.flags.DEFINE_boolean("weight_sharing", True, "Sharing CNN or LSTM weights. (default: True")
tf.flags.DEFINE_string("energy_function", "cosine", "Similarity energy function, one of {`euclidean`, `cosine`, `exp_manhattan`, `combine`} (default: euclidean)")
tf.flags.DEFINE_string("loss_function", "contrasive", "Loss function one of `cross_entrophy`, `contrasive`, (default: contrasive loss)")
# only for contrasive loss
tf.flags.DEFINE_float("scale_pos_weight", 2, "Scale loss function for imbalance data, set it around neg_samples / pos_samples ")
tf.flags.DEFINE_float("margin", 0.0, "Margin for contrasive loss (default: 0.0)")
tf.flags.DEFINE_float("pred_threshold", 0.40, "Threshold for classify.(default: 0.5)")
tf.flags.DEFINE_boolean("dense_layer", False, "Whether to add a fully connected layer before calculate energy function. (default: False)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_string("model_dir", "../model/cnn2", "Model directory (default: ./model)")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_float("lr", 1e-3, "Initial learning rate (default: 1e-3)")
tf.flags.DEFINE_float("weight_decay_rate", 0.8, "Exponential weight decay rate (default: 0.9) ")
tf.flags.DEFINE_integer("num_epochs", 7, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("log_every_steps", 100, "Print log info after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every_steps", 100, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every_steps", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 9, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print("\nParameters:")
if int((tf.__version__).split('.')[1]) < 5:
    FLAGS._parse_flags()  # tf version <= 1.4
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
else:
    for attr in FLAGS:
        value = FLAGS[attr].value
        print("{}={}".format(attr.upper(), value))


if not FLAGS.data_file:
    exit("Train data file is empty. Set --data_file argument.")

dataset = Dataset(data_file=FLAGS.data_file, char_level=FLAGS.char_model, embedding_dim=FLAGS.embedding_dim)
vocab, word2id = dataset.read_vocab()
print("Vocabulary Size: {:d}".format(len(vocab)))


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

max_title_len = 50
max_scene_len = 50

class MatchPyramid(object):
    def __init__(self, max_scene_len, max_title_len, vocab_size):        
        self.embedding_dim = FLAGS.embedding_dim

        self.max_scene_len = max_scene_len
        self.max_title_len = max_title_len
        self.psize1 = FLAGS.psize1
        self.psize2 = FLAGS.psize2

        # input
        self.x_scene = tf.placeholder(tf.int32, [None, max_scene_len], name="input_x1")
        self.x_title = tf.placeholder(tf.int32, [None, max_title_len], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="y")

        self.x_scene_len = tf.placeholder(tf.int32, [None], name="x_scene_len")
        self.x_title_len = tf.placeholder(tf.int32, [None], name='x_title_len')

        self.batch_size = tf.shape(self.x_scene)[0]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.word_embedding = tf.Variable(tf.truncated_normal([vocab_size, self.embedding_dim], 0, 0.1), name="word_embedding")
        self.x_scene_emb = tf.nn.embedding_lookup(self.word_embedding, self.x_scene)
        self.x_title_emb = tf.nn.embedding_lookup(self.word_embedding, self.x_title)

        self.dpool_index = tf.placeholder(tf.int32, name='dpool_index', shape=(None, max_scene_len, max_title_len, 3))

        self._match()
        self._logloss()
        with tf.name_scope("metrics"):
            # tf.rint: Returns element-wise integer closest to x. auto threshold 0.5

            self.y_pred = tf.cast(tf.greater(self.probs[:,1], 0.4), dtype=tf.float32, name="y_pred")
            # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, self.input_y), tf.float32), name="accuracy")
            TP = tf.count_nonzero(self.input_y * self.y_pred, dtype=tf.float32)
            TN = tf.count_nonzero((self.input_y - 1) * (self.y_pred - 1), dtype=tf.float32)
            FP = tf.count_nonzero(self.y_pred * (self.input_y - 1), dtype=tf.float32)
            FN = tf.count_nonzero((self.y_pred - 1) * self.input_y, dtype=tf.float32)
            # tf.div like python2 division, tf.divide like python3
            self.cm = tf.confusion_matrix(self.input_y, self.y_pred, name="confusion_matrix")  # [[5036 1109] [842 882]]
            self.acc = tf.divide(TP + TN, TP + TN + FP + FN, name="accuracy")
            self.precision = tf.divide(TP, TP + FP, name="precision")
            self.recall = tf.divide(TP, TP + FN, name="recall")
            self.f1 = tf.divide(2 * self.precision * self.recall, self.precision + self.recall, name="F1_score")

    def _match(self):
        # 交互矩阵左边是场景词, 上边是商品标题, 这要和dpool_index计算时的顺序对应
        cross = tf.einsum("abd,acd->abc", self.x_scene_emb, self.x_title_emb)
        cross_img = tf.expand_dims(cross, 3)

        # convolution
        w1 = tf.get_variable('w1',
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32),
                             dtype=tf.float32, shape=[2, 5, 1, 8])
        b1 = tf.get_variable('b1', initializer=tf.constant_initializer(), dtype=tf.float32, shape=[8])
        conv1 = tf.nn.relu(tf.nn.conv2d(cross_img, w1, [1, 1, 1, 1], "SAME") + b1, name="relu1")
        #with tf.name_scope("dropout1"):
            #conv1 = tf.nn.dropout(conv1, self.dropout_keep_prob)

        # dynamic pooling
        conv1_expand = tf.gather_nd(conv1, self.dpool_index)
        pool1 = tf.nn.max_pool(conv1_expand,
                               [1, self.max_scene_len / self.psize1, self.max_title_len / self.psize2, 1],
                               [1, self.max_scene_len / self.psize1, self.max_title_len / self.psize2, 1],
                               "VALID")
        with tf.name_scope("dropout2"):
            pool1 = tf.nn.dropout(pool1, self.dropout_keep_prob)
        # full connected layer
        self.fc1 = tf.nn.relu(tf.contrib.layers.linear(tf.reshape(pool1, [-1, self.psize1 * self.psize2 * 8]), 20), name="relu2")

    def _logloss(self):
        self.logits = tf.contrib.layers.linear(self.fc1, 2)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.input_y,tf.int32))
        self.loss = tf.reduce_mean(loss)

        self.probs = tf.nn.softmax(self.logits, name="probs")


class MatchPyramidTaskRunner(object):
    @staticmethod
    def dynamic_pooling_index(len1, len2, max_len1, max_len2):
        '''
        生成dynamic pooling的index,是模型的输入之一;
        根据待匹配的text1和text2的真实长度和最大长度计算
        '''

        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):

            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i / stride1) for i in range(max_len1)]
            idx2_one = [int(i / stride2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2, 1, 0))
            return index_one

        index = []
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i], len2[i], max_len1, max_len2))
        return np.array(index)

    @staticmethod
    def train():       
        
        #model_data = MatchPyramidData()
        #model_data.generate_train_test_data()
        #train_iter = model_data.batch_iter("train")
        #test_iter = model_data.batch_iter("test", batch_size=1024)

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.95
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = MatchPyramid(max_scene_len,
                                     max_title_len,
                                     len(vocab))

                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(0.001)
                grads_and_vars = optimizer.compute_gradients(model.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            print("Defined gradient summaries.")

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            f1_summary = tf.summary.scalar("F1-score", model.f1)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, f1_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(FLAGS.model_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, f1_summary])
            dev_summary_dir = os.path.join(FLAGS.model_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            #checkpoint_dir = os.path.abspath(os.path.join(FLAGS.model_dir, "checkpoints"))
            checkpoint_dir = os.path.join(FLAGS.model_dir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            graph_def = tf.get_default_graph().as_graph_def()
            with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
                f.write(str(graph_def))
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if FLAGS.word_embedding_type != 'rand':
                # initial matrix with random uniform
                # embedding_init = np.random.uniform(-0.25, 0.25, (len(vocab), FLAGS.embedding_dim))
                embedding_init = np.zeros(shape=(len(vocab), FLAGS.embedding_dim))
                # load vectors from the word2vec
                print("Initializing word embedding with pre-trained word2vec.")
                words, vectors = dataset.load_word2vec()
                for idx, w in enumerate(vocab):
                    vec = vectors[words.index(w)]
                    embedding_init[idx] = np.asarray(vec).astype(np.float32)
                print("Initialized word embedding")
                sess.run(model.word_embedding.assign(embedding_init))
            if FLAGS.Submit:
                data = dataset.process_data(data_file=FLAGS.data_file, sequence_length=FLAGS.max_document_length)  # (x1, x2, y)
                train_data, eval_data = dataset.train_test_split(data, test_size=FLAGS.val_percentage, random_seed=FLAGS.random_seed)
                train_batches = dataset.batch_iter(data, FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)

            else:# test
                data = dataset.process_data(data_file=FLAGS.data_file, sequence_length=FLAGS.max_document_length)  # (x1, x2, y)
                train_data, eval_data = dataset.train_test_split(data, test_size=FLAGS.val_percentage, random_seed=FLAGS.random_seed)
                train_batches = dataset.batch_iter(train_data, FLAGS.batch_size, FLAGS.num_epochs, shuffle=True) 

                print("Starting training...")
                F1_best = 0.
                last_improved_step = 0
            for batch in train_batches:
                x1_batch, x2_batch, y_batch, seqlen1, seqlen2 = zip(*batch)
                dpool = MatchPyramidTaskRunner.dynamic_pooling_index(seqlen1, seqlen2, model.max_scene_len, model.max_title_len)
                # print(x1_batch[:3])
                # print(y_batch[:3])
                # if random.random() > 0.5:
                #     x1_batch, x2_batch = x2_batch, x1_batch
                feed_dict = {
                    model.x_scene: x1_batch,
                    model.x_title: x2_batch,
                    model.x_scene_len: seqlen1,
                    model.x_title_len: seqlen2,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    model.input_y: y_batch,
                    model.dpool_index: dpool
                }
                _, step, loss, acc, precision, recall, F1, summaries = sess.run(
                    [train_op, global_step, model.loss, model.acc, model.precision, model.recall, model.f1, train_summary_op],  feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % FLAGS.log_every_steps == 0:
                    train_summary_writer.add_summary(summaries, step)
                    print("{} step {} TRAIN loss={:g} acc={:.3f} P={:.3f} R={:.3f} F1={:.6f}".format(
                        time_str, step, loss, acc, precision, recall, F1))
                
                if FLAGS.Submit is False:
                    if step % FLAGS.evaluate_every_steps == 0:
                        # eval
                        x1_batch, x2_batch, y_batch, seqlen1, seqlen2 = zip(*eval_data)
                        dpool = MatchPyramidTaskRunner.dynamic_pooling_index(seqlen1, seqlen2, model.max_scene_len, model.max_title_len)
                        feed_dict = {
                            model.x_scene: x1_batch,
                            model.x_title: x2_batch,
                            model.x_scene_len: seqlen1,
                            model.x_title_len: seqlen2,
                            model.dropout_keep_prob: 1.0,
                            model.input_y: y_batch,
                            model.dpool_index: dpool
                        }
  

                        loss, acc, cm, precision, recall, F1, summaries = sess.run(
                            [model.loss, model.acc, model.cm, model.precision, model.recall, model.f1, dev_summary_op], feed_dict)
                        dev_summary_writer.add_summary(summaries, step)
                        if F1 > F1_best:
                            F1_best = F1
                            last_improved_step = step
                            if np.float32(F1_best).item() > 0.5:
                                path = saver.save(sess, checkpoint_prefix, global_step=step)
                                print("Saved model with F1={} checkpoint to {}\n".format(F1_best, path))
                            improved_token = '*'
                        else:
                            improved_token = ''
                        print("{} step {} DEV loss={:g} acc={:.3f} cm{} P={:.3f} R={:.3f} F1={:.6f} {}".format(
                            time_str, step, loss, acc, cm, precision, recall, F1, improved_token))
                        # if step % FLAGS.checkpoint_every_steps == 0:
                        #     if F1 >= F1_best:
                        #         F1_best = F1
                        #         path = saver.save(sess, checkpoint_prefix, global_step=step)
                        #         print("Saved model with F1={} checkpoint to {}\n".format(F1_best, path))
                    if step - last_improved_step > 4000:  # 2000 steps
                        print("No improvement for a long time, early-stopping at best F1={}".format(F1_best))
                        break
            if  FLAGS.Submit is True:            
                path = saver.save(sess, checkpoint_prefix, global_step=step)
                print("Saved model with F1={} checkpoint to {}\n".format(F1, path))





if __name__ == "__main__":
    MatchPyramidTaskRunner.train()

