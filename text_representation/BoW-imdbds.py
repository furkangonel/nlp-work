# import libraries 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# veri setini içe aktar 
"""
csv dosya yolunun güncellenmesi gerekebilir
"""
df = pd.read_csv('/metin_temsili/IMDB Dataset.csv')


# metin verilerini alalım
documents = df['review']
labels = df['sentiment']


# metin temizleme
def clean_text(text):
    
    # uppercase -> lowercase
    
    text = text.lower()
    
    # rakamlalari temzileme
    text = re.sub(r"\d+", "", text)

    # ozel karakterleri temizleme
    text = re.sub(r"[^\w\s]", "", text)
    
    # kisa kelimeleri temizleme
    text = " ".join([word for word in text.split() if len(word) > 2])
    
    
    return text
    

# metinleri temizle
cleaned_doc = [clean_text(row) for row in documents[:75]]


# %% stop_words cleaned (TASK)

nltk.download("stopwords") # farklı dillerde en cok stopwords iceren veri seti

stop_words_english = set(stopwords.words("english"))
filtered_doc = [
    " ".join([word for word in sentence.split() if word not in stop_words_english])
    for sentence in cleaned_doc
]

# %% bow


# vectorizer tanimla
vectorizer = CountVectorizer()


# metin -> sayisal hale getir
X = vectorizer.fit_transform(filtered_doc[:75])


# kelime kumesi goster
feature_names = vectorizer.get_feature_names_out()


# vektor temsili goster
vektor_temsili2 = X.toarray()


df_bow = pd.DataFrame(vektor_temsili2, columns = feature_names)


# kelime frekanslarini göster
word_counts = X.sum(axis=0).A1
word_frq = dict(zip(feature_names, word_counts))

most_common_words = Counter(word_frq).most_common(10)
print("En sık geçen 10 kelime:", most_common_words)

