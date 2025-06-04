'''
metin uretimi

gpt-2 metin uretimi calismasi

'''



# import libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
#from transformers import AutoTokenizer, AutoModelForCausalLM # llama

# modelin tanimlanmasi
model_name = "gpt2"
#model_name = "huggyllama/llama-7b" # llama


# tokenization ve model olusturma
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name) # llama

model = GPT2LMHeadModel.from_pretrained(model_name)
#model = AutoModelForCausalLM(model_name) # llama


# MacOS uyumluluğu için padding id ayarlanıyor 
model.config.pad_token_id = model.config.eos_token_id


# metin ureitmi icin gerekli olan baslangic text'i
text = "I go to school for"

# tokenization
inputs = tokenizer.encode(text, return_tensors="pt")


# Bellek taşmalarını önlemek için attention_mask ve pad_token_id ver
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=20,
    pad_token_id=model.config.eos_token_id
)


# modelin urettigi tokenlari okunabilir hale getirmemiz lazim
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # ozel tokenlari (orn: cumle baslangic bitis tokenlari) bize gosterecegi metinden cikar


# uretilen metni print ettirelim
print(f"Generated Text: {generated_text}")

