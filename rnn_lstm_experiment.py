# rnn_lstm_experiment.py
# ================================================
# EXPERIMENT: Word2Vec Dimension, Sequence Length, and Hidden Size on RNN vs LSTM
# ================================================

import numpy as np
import pandas as pd
import re
import nltk
import random
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset from KaggleHub
df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "hassanamin/textdb3", "fake_or_real_news.csv")
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})
df = df[['text', 'label']].dropna()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    tokens = text.lower().split()
    return [t for t in tokens if t not in stop_words]

df['tokens'] = df['text'].apply(clean_text)

# Tokenizer setup
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['tokens'])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

def build_embedding_matrix(w2v_model, dim):
    matrix = np.zeros((vocab_size, dim))
    for word, i in word_index.items():
        if word in w2v_model.wv:
            matrix[i] = w2v_model.wv[word]
    return matrix

# Experiment setup
embedding_dims = [50, 100, 200]
sequence_lengths = [50, 100, 200]
model_types = ['RNN', 'LSTM']
hidden_dims = [128, 256, 512]
results = []

# Start experiments
for emb_dim in embedding_dims:
    w2v = Word2Vec(sentences=df['tokens'], vector_size=emb_dim, window=5, min_count=2, workers=4, seed=SEED)
    embedding_matrix = build_embedding_matrix(w2v, emb_dim)

    for seq_len in sequence_lengths:
        sequences = tokenizer.texts_to_sequences(df['tokens'])
        X = pad_sequences(sequences, maxlen=seq_len)
        y = to_categorical(df['label'])

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_train_raw, y_train, test_size=0.2, random_state=SEED)

        for model_type in model_types:
            for hidden_dim in hidden_dims:
                print(f"Training {model_type} | dim={emb_dim} | seq={seq_len} | hidden={hidden_dim}")
                model = Sequential()
                model.add(Embedding(input_dim=vocab_size, output_dim=emb_dim, weights=[embedding_matrix], input_length=seq_len, trainable=False))
                if model_type == 'RNN':
                    model.add(SimpleRNN(hidden_dim))
                else:
                    model.add(LSTM(hidden_dim))
                model.add(Dense(2, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train_raw, y_train, validation_data=(X_val_raw, y_val), epochs=3, batch_size=64, verbose=0)

                y_pred = np.argmax(model.predict(X_test_raw), axis=1)
                y_true = np.argmax(y_test, axis=1)
                report = classification_report(y_true, y_pred, output_dict=True)

                results.append({
                    "Model": model_type,
                    "Embed_Dim": emb_dim,
                    "Seq_Len": seq_len,
                    "Hidden_Dim": hidden_dim,
                    "Accuracy": report["accuracy"],
                    "Precision": report["weighted avg"]["precision"],
                    "Recall": report["weighted avg"]["recall"],
                    "F1-Score": report["weighted avg"]["f1-score"]
                })

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("rnn_lstm_experiment_results.csv", index=False)
print(results_df)
