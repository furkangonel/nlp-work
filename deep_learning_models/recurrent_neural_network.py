'''

Solve Classification problem (Sentiment Analysis in NlP) with RNN (Deep Learning based Language Model)

duygu analizi -> bir cumlenin etiketlemesi (positive & negative)
restaurant yorumları degerlendirme
'''

# import libraries 
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



# create dataset
df = pd.read_csv('/Users/furkangonel/Desktop/spyder-py3/BTK_NLP/deep_learning_models/yorum_ds.csv')

text = df['text']
label = df['label']




# %% metin temizleme ve preprocessing: tokenization, padding, label, encoding, train test split


# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
word_index = tokenizer.word_index


# padding process
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen = maxlen)
print(X.shape)


# label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])


# train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# %% metin temsili: word embedding: word2vec
sentences = [text.split() for text in df['text']]
word2vec_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1) # UYARII

embedding_dim = 50
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]
    
    

# %% modelling: build, train ve test rnn modeli 


# build model
model = Sequential()

# embedding
model.add(Embedding(
    input_dim = len(word_index) + 1, 
    output_dim = embedding_dim, 
    weights = [embedding_matrix], 
    input_length = maxlen,
    trainable = False
    ))

# RNN layer
model.add(SimpleRNN(
    50,
    return_sequences = False # (default)
    ))

# output layer
model.add(Dense(
    1,
    activation = "sigmoid" # binary_classification
    ))


# compile model
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# train model 
model.fit(X_train, y_train, epochs = 50, batch_size = 2, validation_data=(X_test, y_test))


# evaluate rnn model 
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# %% cumle siniflandirma calismasi
def classify_sentence(sentence):
    
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen = maxlen)
    
    prediction = model.predict(padded_seq)
    
    predicted_class = (prediction > 0.5).astype(int)
    label = "positive" if predicted_class[0][0] == 1 else "negative"
    
    return label



sentence = "Fiyatlar kaliteye göre gayet uygundu."

result = classify_sentence(sentence)
print(f"Result: {result}")




























