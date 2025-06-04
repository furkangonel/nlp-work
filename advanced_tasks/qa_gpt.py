from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


import warnings
warnings.filterwarnings("ignore")

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_answer(context, question):
    
    input_text = f"Question: {question}, Context: {context}. Please answer the question according to context"
    
    # tokenize
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # modeli calistir
    with torch.no_grad():
        outputs = model.generate(inputs, max_length = 200)
        
    
    # uretilen yaniti decode edelim
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True) # merhaba<EOS><PAD>
    
    # tanilari ayiklayalim
    answer = answer.split("Answer:")[-1].strip()
    
    return answer



question = "what is the capital of France"
context = "France, offically the French Republic, is a country whose capital is Paris"
answer = generate_answer(context, question)

print(f"--> Context: {context}\n")
print(f"--> Question: {question}\n")
print(f"-> Answer: {answer}")


context_1 = "Machine learning is an academic discipline in artificial intelligence that deals with the development and study of statistical algorithms that can learn from data and generalize to unseen data and therefore perform tasks without explicit instructions."
question_1 = "What is Machine Learning?"
answer_1 = generate_answer(context_1, question_1)

print(f"--> Context_1: {context_1}\n")
print(f"--> Question_1: {question_1}\n")
print(f"-> Answer_1: {answer_1}")

    