import nltk
from nltk.wsd import lesk # bir kelimenin anlamini belirlemek için baglamini (cevresindeki kelimeler) sozlukteki kelimelerin anlamını karsilastiran algoritma


# gerekli nltk paketleirni indir
nltk.download("wordnet")
nltk.download("own-1.4")
nltk.download("punkt")


# ilk cumle
s1 = "I go to bank to deposit money"
w1 = "bank"

sense1 = lesk(nltk.word_tokenize(s1), w1)
print(f"Cumle: {s1}")
print(f"Word: {w1}")
print(f"Sense: {sense1.definition()}")


# ikinci cumle
s2 = "The river bank is flooded after the heavy rain"
w2 = "bank"

sense2 = lesk(nltk.word_tokenize(s2), w2)
print(f"Cumle: {s2}")
print(f"Word: {w2}")
print(f"Sense: {sense2.definition()}")


