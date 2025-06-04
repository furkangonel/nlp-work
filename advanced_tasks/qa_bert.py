from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")


# squad veri seti uzerinde ince ayar yapilmis bert fil modeli
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"


# bert tokenizer 
tokenizer = BertTokenizer.from_pretrained(model_name)

# soru cevaplama gorevi icin ince ayar yapilmis bert modeli
model = BertForQuestionAnswering.from_pretrained(model_name)


# cevaplari tahmin eden fonksiyon
def predict_answer(context, question):
    
    '''
        context = metin
        question = soru
        amac: metin icerisinden soruyu ara
        
        
        1) tokenize
        2) metnin icerisnde soruyu ara
        3) metnin icerisndeki sorunun cevabini nerelerde olabileceginin skorlarini return et
        4) skorlardan tokenalarin indeksleri hesapladik
        5) tokenlari bulduk yabni cevabi bulduk
        6) okunabilir olmasi icin tokenlari string e cevirdik
    
    '''
    
    # metni ve soruyu tokenlara ayiralim ve modele uygun hale getirelim
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    
    # giris tensorlarini hazirla
    input_ids = encoding["input_ids"] # tokenlarin id
    attention_mask = encoding["attention_mask"] # hangi tokenlarin dikkate alinacagini belirtir
    
    # modeli calistir ve skorlari hesapla
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict = False)
        
    # en yuksek olasiliga sahip start ve end indekslerini hesapliyor
    start_index = torch.argmax(start_scores, dim=1).item() # baslangic indeks
    end_index = torch.argmax(end_scores, dim=1).item() # bitis indeks
    
    
    # token id leri kullanarak cevap metninin elde edilmesi
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index + 1])
    
    # tokenlari birestir ve okunabilir hale getir
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer




question = "what is the capital of France"
context = "France, offically the French Republic, is a country whose capital is Paris"
answer = predict_answer(context, question)

print(f"--> Context: {context}\n")
print(f"--> Question: {question}\n")
print(f"-> Answer: {answer}")


context_1 = "Machine learning is an academic discipline in artificial intelligence that deals with the development and study of statistical algorithms that can learn from data and generalize to unseen data and therefore perform tasks without explicit instructions."
question_1 = "What is Machine Learning?"
answer_1 = predict_answer(context_1, question_1)

print(f"--> Context_1: {context_1}\n")
print(f"--> Question_1: {question_1}\n")
print(f"-> Answer_1: {answer_1}")





