from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk
import nltk

# Gerekli kaynaklar
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Örnek cümleler
sentences = [
    "I go to the bank to deposit money",
    "The river bank was flooded after the heavy rain"
]

word = "bank"

for s in sentences:
    print(f"\nSentence: {s}\n")
    
    print("*-*-*-*-*-*-*-*-*-*-*-*-*")

    
    simple = simple_lesk(s, word)
    print(f"→ Simple Lesk: {simple.definition() if simple else 'No match'}\n\n")

    adapted = adapted_lesk(s, word)
    print(f"→ Adapted Lesk: {adapted.definition() if adapted else 'No match'}\n\n")

    cosine = cosine_lesk(s, word)
    print(f"→ Cosine Lesk: {cosine.definition() if cosine else 'No match'}")
    
    
    

"""
Sentence: I go to the bank to deposit money

*-*-*-*-*-*-*-*-*-*-*-*-*
→ Simple Lesk: a financial institution that accepts deposits and channels the money into lending activities


→ Adapted Lesk: a financial institution that accepts deposits and channels the money into lending activities


→ Cosine Lesk: a container (usually with a slot in the top) for keeping money at home

Sentence: The river bank was flooded after the heavy rain

*-*-*-*-*-*-*-*-*-*-*-*-*
→ Simple Lesk: sloping land (especially the slope beside a body of water)


→ Adapted Lesk: sloping land (especially the slope beside a body of water)


→ Cosine Lesk: a supply or stock held in reserve for future use (especially in emergencies)
"""