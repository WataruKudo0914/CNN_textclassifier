
# coding: utf-8

# Mojule Import  
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import MeCab
from keras.preprocessing import sequence
import pickle
import time
from multiprocessing import Pool
import multiprocessing as multi
import pandas as pd
import importlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import copy

class CNN(object):
    
    def __init__(self,tagger,model,filter_sizes=[3,4,5],n_epochs=100):
        self.tagger = tagger
        self.model = model
        self.filter_sizes = filter_sizes
        self.padding_num = max(filter_sizes) - 1
        self.n_epochs = n_epochs
        self.Results = pd.DataFrame(index=range(n_epochs),columns=['loss','Accuracy','confusion_matrix'])
        self.maxlen = 0
        self.n_classes = 0
        self.cnn = None
        
        
    def _tokenize(self,text): #形態素解析
        tagger = self.tagger
        sentence = []
        node = tagger.parse(text)
        if node == None: #parseの最大文字数を上回った場合、200万文字まで.
            node = tagger.parse(text[:2000000])
        else:
            pass
        node = node.split("\n")
        for i in range(len(node)): #単語上限数を設定
            feature = node[i].split("\t")
            if feature[0] == "EOS":
                break
            hinshi = feature[3].split("-")[0]
            if "名詞" in hinshi:
                sentence.append(feature[2])
            elif "形容詞" in hinshi:
                sentence.append(feature[2])
            elif "動詞" in hinshi:
                sentence.append(feature[2])
            elif "形容動詞" in hinshi:
                sentence.append(feature[2])
            elif "連体詞" in hinshi:
                sentence.append(feature[2])           
            elif "助詞" in hinshi:
                sentence.append(feature[2])        
        return sentence

    def getVector(self,text): #文 (⇒単語) ⇒ 行列
        _tokenize, model = self._tokenize, self.model
        vocabs = model.vocab
        texts = _tokenize(text)
        emb = [model.word_vec(x) for x in texts if x in vocabs]
        return np.vstack(tuple(emb)) if len(emb) != 0 else np.zeros((1,model.vector_size)) 
        #word2vecモデルの語彙にどの単語もない文は(1,model.vector_size)のゼロベクトル．

    def post_padding(self,text): #文　(⇒行列) ⇒ 文の区切りにゼロパディングを施した行列
        getVector, model, padding_num = self.getVector, self.model, self.padding_num
        emb = getVector(text)
        padded = np.vstack((emb,np.zeros((padding_num,model.vector_size))))
        return padded
    
    def text2matrix(self,text_list): #入力 (文章ごとのnumpy配列) ⇒ 行列
        post_padding = self.post_padding
        matrixes = [post_padding(text) for text in text_list]
        return np.vstack(tuple(matrixes))
    
    def mat2data(self,mat,maxlen): # 入力の行数を揃えるための関数
        data = sequence.pad_sequences(mat,maxlen=maxlen,padding="post",truncating="post",dtype="float32")
        data = data.astype(np.float32)
        return data
    
    def make_input(self,labels,texts,train_rate=0.8):
        text2matrix, mat2data = self.text2matrix, self.mat2data
        
        # labels ⇒　onehot表現　に変換
        cat2id = dict(zip(np.unique(labels),range(len(np.unique(labels)))))
        id2cat = dict(zip(range(len(np.unique(labels))),np.unique(labels)))
        data_labels=np.array([cat2id[l] for l in labels])
        def onehot(data):
            Z = np.zeros((len(data),len(cat2id)))
            Z[np.arange(len(data)), data] = 1
            return Z
        onehot_datalabels = onehot(data_labels)
        
        #文章ごとの行列を格納した配列
        mat_list = np.array([text2matrix(text) for text in texts])
        # maxlen (最大行数)を調べる．
        maxlen = max([mat.shape[0] for mat in mat_list])
        self.maxlen = maxlen
        
        # 訓練データとテストデータに分ける．
        num_index = int(train_rate * len(onehot_datalabels))
        index =set(np.random.choice(range(len(onehot_datalabels)),num_index,replace=False))
        unlabeled_index = list(set(range(len(onehot_datalabels)))-index)
        labeled_index = list(index)
        
        train_MatList = mat_list[labeled_index]
        test_MatList = mat_list[unlabeled_index]
        
        
        train_y = onehot_datalabels[labeled_index]
        test_y = onehot_datalabels[unlabeled_index]
        
        return [train_MatList, test_MatList, train_y, test_y, maxlen]
        
    def fit(self,labels,texts,train_rate=0.8):
        make_input, model, mat2data, filter_sizes, n_epochs = self.make_input, self.model, self.mat2data, self.filter_sizes, self.n_epochs
        train_MatList, test_MatList, train_y, test_y, maxlen = make_input(labels,texts,train_rate)
        maxlen = self.maxlen
        
        n_classes = len(np.unique(labels))
        self.n_classes = n_classes
        
        rng = np.random.RandomState(1234)
        random_state = 42
        
        cnn = TextCNN(
                    sequence_length=maxlen,
                    num_classes=train_y.shape[1],
                    embedding_size=model.vector_size,
                    filter_sizes=filter_sizes,
                    num_filters=128,
                    l2_reg_lambda=0.0
                    )
        self.cnn = cnn
        
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        
        batch_size = 400
        n_batches = len(train_MatList) // batch_size
        
        Results = self.Results
        
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        current_loss = float('inf')
        saver = tf.train.Saver()

        print("学習開始")    
        for epoch in range(n_epochs):
            print("EPOCH:"+str(epoch+1))
            train_MatList, train_y = shuffle(train_MatList, train_y, random_state=random_state)
            current_step = tf.train.global_step(sess, global_step)

            for i in range(n_batches):

                start = i * batch_size
                end = start + batch_size

                ML_batch = train_MatList[start:end]
                x_batch = mat2data(ML_batch,maxlen)
                y_batch = train_y[start:end]

                feed_dict = {
                      cnn.input_x: x_batch,
                      cnn.input_y: y_batch,
                      cnn.dropout_keep_prob: 0.5,
                    }

                _, step, loss, accuracy = sess.run(
                        [train_op, global_step,  cnn.loss, cnn.accuracy],
                        feed_dict)
            
            del x_batch
                
            #テストもバッチ処理(メモリの関係)
            test_MatList, test_y = shuffle(test_MatList, test_y, random_state=random_state)
            test_Num = len(test_MatList)
            t_batch_size = 1000
            t_batches = test_Num // t_batch_size
            losses = []
            Accuracies = []
            Matrix = np.zeros((n_classes,n_classes))
            
            for j in range(t_batches):
                t_start = j * t_batch_size
                t_end = t_start + t_batch_size
                
                test_MatList_batch = test_MatList[t_start:t_end]
                test_X_batch = mat2data(test_MatList_batch,maxlen)
                test_y_batch = test_y[t_start:t_end]

                feed_dict = {
                    cnn.input_x: test_X_batch,
                    cnn.input_y: test_y_batch,
                    cnn.dropout_keep_prob: 1.0,
                    }
                step, loss_batch, accuracy_batch, confusion_matrix_batch = sess.run(
                            [global_step, cnn.loss, cnn.accuracy, cnn.confusion_matrix],
                            feed_dict)
                
                losses.append(loss_batch)
                Accuracies.append(accuracy_batch)
                Matrix += confusion_matrix_batch
           
            
            loss = np.average(losses)
            accuracy = np.average(Accuracies)
            confusion_matrix = Matrix
                
            

            print(" epoch {}, loss {:g}, acc {:g}".format( step, loss, accuracy))
            Results.loc[epoch,'loss'] = loss
            Results.loc[epoch,'Accuracy'] = accuracy
            Results.loc[epoch,'confusion_matrix'] = confusion_matrix
            if loss <= current_loss:
                current_loss = loss
                saver.save(sess,"model/Best_model.ckpt")
                # 最もlossが小さかったモデルは一時ファイルに保存されている．
        
        return None
    
    def predict(self,texts,sess):
        text2matrix, mat2data, maxlen, n_classes, model = self.text2matrix, self.mat2data, self.maxlen, self.n_classes, self.model
        cnn = self.cnn
        
        
        #文章ごとの行列を格納した配列
        mat_list = np.array([text2matrix(text) for text in texts])
        input_X = mat2data(mat_list,maxlen)
        input_Y = np.zeros((len(texts),n_classes))
        
        feed_dict = {
            cnn.input_x: input_X,
            cnn.input_y: input_Y,
            cnn.dropout_keep_prob: 1.0,
            }
        
        predictions, probabilities = sess.run(
                [cnn.predictions, tf.nn.softmax(cnn.scores)],
                feed_dict) 
        
        return predictions, probabilities
        
        
        
                







class TextCNN(object):
    

    
    
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,embedding_size,
       filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1) #畳み込みするためにレイヤーを１にする
        
        # Create a convolution + avgpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Avgpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
        #Confusion_Matrix
        with tf.name_scope("confusion_matrix"):
            self.confusion_matrix = tf.confusion_matrix(labels=tf.argmax(self.input_y,1),predictions=self.predictions,name="confusion_matrix")


