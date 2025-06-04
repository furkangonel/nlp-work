from transformers import AutoTokenizer, AutoModel
import torch

# model ve tokenizer yükle
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# metin tanımla
text = "Transformers can be used for natural language processing."

# metni tokenlara çevir
inputs = tokenizer(text, return_tensors="pt")  # DÜZELTİLDİ

# modeli kullanarak embedding oluştur
with torch.no_grad():
    outputs = model(**inputs)

# son gizli durumu al
last_hidden_state = outputs.last_hidden_state

# ilk token'in embedding'ini al
first_token_embedding = last_hidden_state[0, 0, :].numpy()
print(f"Metin Temsili: {first_token_embedding}")