# import library
from transformers import BertTokenizer, BertModel

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity




# tokenizer and model create
model_name = "bert-base-uncased" # kucuk boyutlu bert modeli
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


# veriseti olustur
documents = [
    "Machine Learning is a field of artifical intelligence",
    "Natural Language processing involves understanding human language",
    "Artifical intelligence encomppasses machine learning and natural language processing",
    "Deep learning is a subset of machine learning",
    "Data science combines statistics, data analysis and machine learning",
    "I go to shop"
    ]

query = "shopping?"


# bert ile bilgi getirimi

def get_embedding(text):
    
    # tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # modeli calistir
    outputs = model(**inputs)
    
    # son gizli katmani alalim
    last_hidden_state = outputs.last_hidden_state
    
    # metin temsili olustur
    embedding = last_hidden_state.mean(dim=1)
    
    # vektoru numpy olarak return et
    return embedding.detach().numpy()



# belgeler ve sorgu icin embedding vektorlarin al
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])
query_embedding = get_embedding(query)


# kosinus benzerligi ile belgeler arasinda benzerligi hesaplayalim
similarities = cosine_similarity(query_embedding, doc_embeddings)

sorted_similarities = sorted(enumerate(similarities[0]), key=lambda x: x[1], reverse=True)

# her belgenin benzerlik skoru
for index, score in sorted_similarities:
    print(f"Document {index + 1}: Score = {score:.4f} → \"{documents[index]}\"")


