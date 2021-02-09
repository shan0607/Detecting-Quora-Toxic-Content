
import numpy as np
from keras.preprocessing.text import Tokenizer
import nltk
import re
from spellchecker import SpellChecker
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)



spell = SpellChecker()


# define punctuation
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''


###Initiating Tokenizer
tokenizer = Tokenizer(num_words = 1000000)


data_train = pd.read_csv("train.csv")

data_test = pd.read_csv("test.csv")


## COLUNS = qid  question_text  target

#reject_question=data.question_text[data['target']==1]

##Trainining tokeizer testing

###Takes too much time
#def textclean(text,k_start,k_end):
   # print(len(text))
  #  k = k_start
   # for p in text:
   #     print(k)
  #      p = p.lower()
    #    p = re.sub(r'[0-9]+', '', p )
  #      p = re.sub('[^ A-Za-z' ']+', '', p)
  #      p = " ".join(p.split())
    #    step1 = p.split()
  #      text[k] = " ".join([spell.correction(word) for word in p.split() for p in comp_ques])

    #    if k == k_end:
    #        break

    #    k += 1


   # return text





#def textclean_p1(comp_ques):
  #  comp_ques = comp_ques.tolist()
  #  comp_ques = [p.lower() for p in comp_ques]
   # comp_ques = [re.sub(r'[0-9]+', ' ', p ) for p in comp_ques]
   # comp_ques = [re.sub('[^ A-Za-z' ']+', ' ', p) for p in comp_ques]
  #  comp_ques = [" ".join(p.split()) for p in comp_ques]
   # return comp_ques

###
#hash_data_train = list()
#hash_data_test = list()

###Creating has information for test and train file for Parallel processing
#for i,each in enumerate(textclean_p1(data_train.question_text)):
#    hash_data_train.append(each + " #train#"+ str(data_train.target[i]))

#for i,each in enumerate(textclean_p1(data_test.question_text)):
#    hash_data_test.append(each + " # test # ")




def textclean_p2(comp_ques):
    comp_ques = comp_ques.tolist()
    comp_ques = [p.lower() for p in comp_ques]
    comp_ques = [re.sub(r'[0-9]+', ' ', p ) for p in comp_ques]
    comp_ques = [re.sub('[^ A-Za-z' ']+', ' ', p) for p in comp_ques]
    comp_ques = [" ".join(p.split()) for p in comp_ques]
    return comp_ques



comp_ques = np.concatenate([textclean_p2(data_train.question_text), textclean_p2(data_test.question_text)])



#from torch.multiprocessing import Pool
import torch.multiprocessing as mp
#from multiprocessing import Pool
#def spellcheck(text):
  #  text = [" ".join([spell.correction(word) for word in step1.split()]) for step1 in text]
  #  return text
#
#num_processes = 4
#processes = []
#for rank in range(num_processes):
 #   p = mp.Process(target=spellcheck, args=(comp_ques[0:10],))
#    p.start()
#    processes.append(p)
#for p in processes:
#    p.join()



#p = Pool()
#result = p.map(spellcheck,(comp_ques[0:10],) )
#p.close()
#p.join()

#result =  spellcheck(comp_ques[0:10])


###Saved the above file for time saving
#pd.DataFrame(comp_ques).to_csv("Cleanfile_test_train.csv")


#########################


maxlen = 50
training_samples = data_train.__len__()
test_samples = data_test.__len__()


tokenizer.fit_on_texts(comp_ques)

word_index = tokenizer.word_index
word_index_rev = {v: k for k, v in word_index.items()}

data_ques_seq = tokenizer.texts_to_sequences(comp_ques)

#tt = [j.__len__() for j in (data_ques_seq)]
#tt.where

data = pad_sequences(data_ques_seq, maxlen=maxlen)

x_train = data[:training_samples]
x_test = data[training_samples: training_samples+test_samples]

y_train = data_train.target
#y_test = data_test.target

indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]


x_train_main, x_train_val, y_train_main, y_train_val = train_test_split( x_train, y_train, test_size=0.3, random_state=42)


del comp_ques
del data,x_train,y_train,data_ques_seq,data_test,data_train,indices
#del Pool



###Base model with embeddings

glove_embed = '/embeddings/glove.840B.300d/glove.840B.300d.txt'
google_word2vec = '/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
paragram = '/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
wiki = '/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

import numba

#### Embedding Creater

#f = open("/media/aditya/59AEBA7FE230A4D4/Kaggle/quora" + glove_embed)
#embeddings_values = np.empty(shape= (0,1))
#embeddings_word = np.empty(shape= (0,300))
#k = 1
#for line in f:
 #   print(k)
 #   k = k + 1
  #  values = line.split()
  #  if values.__len__() > 301:
  #      new_length = values.__len__()
  #      new_length = new_length - 301
 #       values = values[new_length:]
 #   embeddings_word = np.append(embeddings_word, values[0])
 ##   coefs = np.asarray(values[1:], dtype='float32')
 #   embeddings_values = np.append(embeddings_values, coefs)
#

#f.close()


###Important



embedding_dim = 300
max_words = word_index.keys().__len__()




word_keys = list(word_index.keys())
word_values = np.fromiter(word_index.values(), dtype= int)




#embeddings_index = load_obj('glove_embed')
##embedding convert
#embed_words = list(embeddings_index.keys())
#embed_words = [p.lower() for p in embed_words]

#embed_values_todelete = list(embeddings_index.values())




#embed_values = np.concatenate(embed_values_todelete)
#embed_values.shape
#embed_values.resize((len(embed_values)/300,300))


#del embeddings_index



embedding_matrix = np.zeros((max_words, embedding_dim))



from numba import njit,prange,jit
from numba import cuda




#matching = [i for i, x in enumerate(embed_words) if any(thing in x for thing in word_keys)]



def looper(embedding_matrix):
    for i in range(2195885):
        print(i)
        if i < max_words-1:
            try:
                embedding_vector = embed_values_todelete[embed_words.index(word_keys[i])]
            except ValueError:
                pass


            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

#embedding_matrix = looper(embedding_matrix)

#save_obj(embeddings_index, "glove_embed")



#save_obj(embedding_matrix, "embed")

embedding_matrix = load_obj("embed")

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten



max_features = embedding_matrix.shape[0]
maxlen = 50



model = Sequential()
model.add(Embedding(max_features, 300,input_length= maxlen ))
model.add(LSTM(maxlen,recurrent_dropout= 0.2, dropout= 0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

import keras.backend as K
from sklearn.metrics import f1_score
import tensorflow as tf

import keras.backend.tensorflow_backend as tf

from keras import backend as K

def f1(y_true, y_pred):


    #y_pred = K.cast(K.greater_equal(y_pred, K.constant([0.4], dtype= K.dtype(y_pred), shape= K.shape(y_pred))), dtype= np.float32)


    def recall(y_true, y_pred):

        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def FSCORE(y_true,y_pred):

    Prec = tf.metrics.precision(y_true,y_pred)
    Rec = tf.metrics.recall(y_true,y_pred)
    p1 = tf.add(Prec,Rec)
    p2 = tf.multiply(Prec,Rec)
    cons = tf.constant([2], dtype= tf.float32)
    p2 = tf.multiply(p2,cons)
    score = tf.divide(p2,p1)
    return score










#model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])



model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=["accuracy"])

history = model.fit(x_train_main, y_train_main,epochs=10,batch_size=128,validation_split=0.2)



pred = model.predict(x_train_val)

pred_test = model.predict(x_train_val)

pred1 = [i for i in pred]
pred1 = list(pred1)
pred1 = np.array(pred1)

for i in np.arange(1,10):
    p = i * 0.1
    pred1 = [j for j in pred]
    pred1 = list(pred1)
    pred1 = np.array(pred1)
    pred_class = pred1
    pred_class[pred_class >= p] = 1
    pred_class[pred_class < p] = 0
    score = f1_score(y_train_val, pred_class)
    print("Threshold = ",p, " F! Score = ", score)


pred_class1 = [list(i) for i in pred_class ]



from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import  accuracy_score

pred_test1 = [int(each[0]) for each in pred_list]

accuracy_score(y_test,pred_test1)
confusion_matrix(y_train_val, pred)

roc_auc_score(y_test,pred_list)

model.save("model1_recurrent_dropout_f1score1.h5")

from keras.models import load_model

test = load_model("model1_recurrent_dropout_f1score1.h5", compile= False)

####Plotting results

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

