"""
classification problem: duygu analizi -> olumlu veya olumsuz olarak siniflandirma

"""

# import libraries 
from nltk.classify import MaxentClassifier



# veri seti tanimlama
train_data = [
    ({"love": True, "amazing": True, "happy": True, "terrible": False}, "positive"),
    ({"hate": True, "terrible": True}, "negative"),
    ({"joy": True, "happy": True, "hate": False}, "positive"),
    ({"sad": True, "depressed": True, "love": False}, "negative")
    ]


# train maximum entropy classifier
classifier = MaxentClassifier.train(train_data, max_iter = 10)


# yeni cumle ile test
words = ["love", "amazing", "terrible", "happy", "joy", "depressed", "sad", "hate"]

test_sentence = "I like this movie"
features = {word: (word in test_sentence.lower().split()) for word in words}

new_test_sentence = "I love this movie"
new_features = {word: (word in new_test_sentence.lower().split()) for word in words}


label = classifier.classify(features)
print(f"Result_1: {label}")

new_label = classifier.classify(new_features)
print(f"Result_2: {new_label}")