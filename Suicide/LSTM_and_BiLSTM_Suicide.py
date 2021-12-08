import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re

from math import exp
from numpy import sign

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import PorterStemmer

import tensorflow as tf
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, Dense, Input, LSTM, Embedding, Dropout, Activation, MaxPooling1D, GRU, Bidirectional, SpatialDropout1D, Lambda, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_punctuation, lemmatize_word, expand_contraction, remove_special_character, remove_stopword
import operator 
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('../PSIMILAN/suicide_datasetv2.csv')

encoding = {
    'suicidal': 0,
    'normal': 1
}

y_encoded = [encoding[cls] for cls in df['class'].values]

preprocess_functions = [to_lower, expand_contraction, remove_special_character, remove_punctuation, remove_stopword]
df_preprocessed = df['text'].apply(lambda x: preprocess_text(x, preprocess_functions))

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

df['text_preprocessed'] = df_preprocessed.apply(lambda x: remove_emoji(x))

def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == '../PSIMILAN/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index

wiki_news = '../PSIMILAN/wiki-news-300d-1M.vec'

print("Extracting FastText embedding")
embed_fasttext = load_embed(wiki_news)

def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words

vocab = build_vocab(df['text_preprocessed'])
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)

X = df['text_preprocessed'].values
y = df.iloc[:, 9:-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

word2vec = KeyedVectors.load_word2vec_format(wiki_news, binary=False)

MAX_NB_WORDS = 35000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X_train) + list(X_val))

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_val = tokenizer.texts_to_sequences(X_val)
sequences_test = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

MAX_SEQUENCE_LENGTH = 800
data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
data_val = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data_train.shape)

EMBEDDING_DIM = 300

nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM))

for (word, idx) in word_index.items():
    if word in word2vec.vocab and idx < len(word_index):
        embedding_matrix[idx] = word2vec.word_vec(word)

print(len(tf.config.experimental.list_physical_devices('GPU')))

# LSTM model
num_labels = 1
with tf.device('/device:GPU:0'):
    model = Sequential()
    
    # Embedded layer
    model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    
    # LSTM Layer
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='sigmoid'))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.95)
    
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['acc'])
    print(model.summary())

early_stop = EarlyStopping(monitor='val_loss', patience=3)
#reduce_lr_on_plateau_cb = ReduceLROnPlateau(verbose = 1, factor=0.1, patience=2)

with tf.device('/device:GPU:0'):
    hist = model.fit(data_train, y_train, validation_data=(data_val, y_val), epochs=20, batch_size=32, shuffle=True, callbacks=[early_stop])
    probs = model.predict(data_test)
    labels_pred = np.round(probs)
    accuracy = accuracy_score(y_test, labels_pred)
    f1 = f1_score(y_test, labels_pred, average='weighted')
    labels_name = df.columns[9:-1]
    
    print("Accuracy: %.2f%%" % (accuracy*100))
    print("F1 Score: %.2f" % (f1*100))

    report = classification_report(y_test, labels_pred, target_names=labels_name, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('../PSIMILAN/lstm_suicide_report.csv')

with open('../PSIMILAN/Probs/word2vec_LSTM_suicide_probs.pickle', 'wb') as prob:
    pickle.dump(probs, prob)

model.save('../PSIMILAN/model/word2vec_LSTM_suicide.h5')

# BiLSTM model
with tf.device('/device:GPU:0'):
    model1 = Sequential()
    
    # Embedded layer
    model1.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    
    # LSTM Layer
    model1.add(Bidirectional(LSTM(512, return_sequences=True)))
    model1.add(Bidirectional(LSTM(256)))
    model1.add(Dense(128, activation='relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(num_labels, activation='sigmoid'))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.95)
    
    model1.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['acc'])
    print(model1.summary())

early_stop = EarlyStopping(monitor='val_loss', patience=3)
#reduce_lr_on_plateau_cb = ReduceLROnPlateau(verbose = 1, factor=0.1, patience=2)

with tf.device('/device:GPU:0'):
    hist = model1.fit(data_train, y_train, validation_data=(data_val, y_val), epochs=50, batch_size=32, shuffle=True, callbacks=[early_stop])
    probs = model1.predict(data_test)
    labels_pred = np.round(probs)
    accuracy = accuracy_score(y_test, labels_pred)
    f1 = f1_score(y_test, labels_pred, average='weighted')
    labels_name = df.columns[9:-1]
    
    print("Accuracy: %.2f%%" % (accuracy*100))
    print("F1 Score: %.2f" % (f1*100))

    report = classification_report(y_test, labels_pred, target_names=labels_name, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('../PSIMILAN/bilstm_suicide_report.csv')

with open('../PSIMILAN/Probs/word2vec_BiLSTM_suicide_probs.pickle', 'wb') as prob:
    pickle.dump(probs, prob)

model1.save('../PSIMILAN/model/word2vec_BiLSTM_suicide.h5')
