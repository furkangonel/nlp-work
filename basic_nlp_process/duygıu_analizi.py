"""
amazon veri seti icerisinde bulunan yorumlarin positive mi negative mi oldugunu siniflandirmak
"""

# import libraries
import pandas as pd
import nltk


from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 


nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("own-1.4")



# veri seti yükle
df = pd.read_csv('/Users/furkangonel/.spyder-py3/BTK_NLP/basic_nlp_process/duygu_analizi_amazon_veri_seti.csv')


# text cleaning & preprocessing
lemmatizer = WordNetLemmatizer()
def clean_preprocess_data(text):
    
    # tokenize
    tokens = word_tokenize(text.lower())
    
    # stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    # lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # join words
    precessed_text = " ".join(lemmatized_tokens)
   
    return precessed_text



df['reviewText2'] = df['reviewText'].apply(clean_preprocess_data)


# sentiment analysis (nltk)
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    
    score = analyzer.polarity_scores(text)
    
    sentiment = 1 if score['pos'] > 0 else 0
    
    return sentiment


df["sentiment"] = df["reviewText2"].apply(get_sentiment)
    

# evaluation - test

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix = confusion_matrix(df["Positive"], df["sentiment"])
print(f"conf_matrix: {confusion_matrix}")

cr = classification_report(df["Positive"], df["sentiment"])
print(f"classification_report: {cr}")


"""
conf_matrix: 
    [
           0    1
     0- [ 1131 3636]
     1- [576 14657]]
    
"""

"""
classification_report:               
            precision    recall  f1-score   support
        
                   0       0.66      0.24      0.35      4767
                   1       0.80      0.96      0.87     15233
        
            accuracy                           0.79     20000
           macro avg       0.73      0.60      0.61     20000
        weighted avg       0.77      0.79      0.75     20000

"""





