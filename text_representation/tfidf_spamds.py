# import library
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords


# veri seti yükle
"""
csv dosya yolunun güncellenmesi gerekebilir
"""
df = pd.read_csv('/spam.csv', encoding='ISO-8859-1')


# metin verilerini alalım
documents = df['v2']
labels = df['v1']


# veri temizleme (TASK)
def clean_text(text):
    
    text = text.lower()
    
    text = re.sub(r"\d+", "", text)
    
    text = re.sub(r"[^\w\s]", "", text)

    text = " ".join([word for word in text.split() if len(word) > 2])
    
    return text


cleaned_doc = [clean_text(row) for row in documents]


nltk.download("stopwords")
eng_stop_words = set(stopwords.words("english"))
filtered_doc = [
    " ".join([word for word in sentence.split() if word not in eng_stop_words])
    for sentence in cleaned_doc
]


# tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(filtered_doc)


# kelime kümesini inceleme
feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1 # her kelimenin ortalama tfidf değerleri


# tfidf skorlarini iceren bir df olustur
df_tfidf = pd.DataFrame({"word": feature_names, "tfidf_score": tfidf_score})



# skorlari sirala ve sonuclari incele
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)
print(df_tfidf_sorted.head(10))