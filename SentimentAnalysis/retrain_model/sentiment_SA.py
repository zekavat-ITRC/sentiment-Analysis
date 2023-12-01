import os
import gc
import nltk
import h5py
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
import fasttext
from tqdm import tqdm
import codecs

# Embedding and Network Parameters
embed_num_dims = 100
max_seq_len = 1000
MAX_NB_WORDS = 100000
 # Read Data
#data = pd.read_csv('new_data.csv',encoding = 'latin1')
data=pd.read_excel('../LabeledData/new_data_retrain_model.xlsx')

data.dropna(inplace=True)
print(data.head())

# Add text to sentences parameeter
sentences = data['text']

# covert labels to one hot-encoding
one_hot = pd.get_dummies(data['sentiment'])
data = data.drop('sentiment',axis = 1)
data = data.join(one_hot)

data.rename(columns={-1:'negative',1:'positive',0:'neutral'},inplace=True)
# Labels
labels = ["negative", "neutral", "positive"]
Y = data[labels].values
# Senetnce Tokenizer
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(sentences)
sequence = tokenizer.texts_to_sequences(sentences)

index_of_words = tokenizer.word_index
padded_seq = pad_sequences(sequence , maxlen = max_seq_len )

print(len(index_of_words))
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(padded_seq, Y, test_size=0.10, random_state=42)



def dplot(history):
    plt.figure(figsize=(20, 5), dpi=100)
    plt.subplot(1, 2, 1)
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('result.png')



import tensorflow as tf
from tensorflow import keras
batch_size = 16
epochs = 7

loaded_model = tf.keras.models.load_model('cnn-fasttext.h5')
print(loaded_model.summary())

history1 = loaded_model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=epochs, batch_size=batch_size, verbose=1)

y_pred = loaded_model.predict(X_test)
y_pred = np.round(y_pred)

# Generate the classification report
report = classification_report(Y_test, y_pred)

print(report)

y_pred1=loaded_model.predict(X_test,verbose=1)
loaded_model.save("cnn-fasttext_v3.h5")
dplot(history1)

