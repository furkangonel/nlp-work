'''
word2vec (google)
fasttext (meta)
'''

# import library
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA # prencipal componenet analysis: dimension reduction (boyut indirgemesi)
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess


# ornek veri seti olustur
sentences = [
    "Köpek çok tatlı bir hayvandır.",
    "Köpekler evcil hayvanlardır.",
    "Kediler genellikle bağımsız hareket etmeyi sever.",
    "Köpekler sadık ve dost canlısı hayvanlardır.",
    "Hayvanlar insanlar için iyi arkadaşlardır."
    ]

tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]


# word2vec
word2vec_model = Word2Vec(sentences = tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)
'''
sentences: data set
vector_size: kelimelerin embedding vektörlerinin boyutu
window: bir kelimenin bağlamını oluşturan kelimelere maks uzaklığı (pencere boyutu)
min_count: eğitimde dikkate alınacak kelimelerin minimum kullanım sınırı
sg: kelimenin çevresindeki kelimelerden tahmini
sg=1; skip-gram modelini kullanır. kelimeden çevresindeki kelimeleri tahmin eder.
'''

# fasttext
fasttext_model = FastText(sentences = tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)


# gorsellestirme: PCA

def plot_word_embedding(model, title):
    
    word_vectors = model.wv
    
    words = list(word_vectors.index_to_key)[:1000]
    vectors = [word_vectors[word] for word in words]
    
    #PCA
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)
    
    # 3D gorsellestirme
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = "3d")
    
    # vektörleri ciz
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2])
    
    # kelimeleri etiketle
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word, fontsize = 12)
        
        
    ax.set_title(title)
    ax.set_xlabel("Componenet 1")
    ax.set_ylabel("Componenet 2")
    ax.set_zlabel("Componenet 3")
    plt.show()
    
    
    
    
plot_word_embedding(word2vec_model, "Word2Vec")
plot_word_embedding(fasttext_model, "FastText")


    
    
    
    
    
    
    
    