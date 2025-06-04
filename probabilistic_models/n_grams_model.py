# import libraries

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from collections import Counter


# ornek veri seti olustur
corpus = [
    "I love apple",
    "I love him",
    "I love NLP",
    "You love me",
    "He loves apple",
    "They love apple",
    "I love you and you love me"
    ]

"""
problem tanimini yapalim:
    dil modeli yaomak istiyoruz
    amac 1 kelimeden sonra gelecek kelimeyi tahmin etmek: metin turetmek/olusturmak
    bunun icin n-gram dil modelini kullanacagiz
    
    
    ex: I ...(love) ... (apple)
"""

# verileri token haline getir
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]

# bigram
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))

bigrams_freq = Counter(bigrams)

# trigram
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))

trigram_freq = Counter(trigrams)


# %% model testing

## I love bigram'indan sonra "youé veya "apple" kelimelerinin gelme olasiliklarini hesaplayalim


bigram = ("i", "love") # hedef bigram


# ->  "i love you" olma olasiligi
prob_you = trigram_freq[("i", "love", "you")]/bigrams_freq[bigram]
print(f"you kelimesinin olma olasiligi: {prob_you}")


# ->  "i love apple" olma olasiligi
prob_apple = trigram_freq[("i", "love", "apple")]/bigrams_freq[bigram]
print(f"apple kelimesinin olma olasiligi: {prob_apple}")












