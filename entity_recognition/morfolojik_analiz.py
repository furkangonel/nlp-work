import spacy 


nlp = spacy.load("en_core_web_sm")

# incelenecek olan kelime ya da kelimeler

word = "book"


# kelimeyi nlp isleminden gecir

doc = nlp(word)

for token in doc:
    
    print(f"Text: {token.text}")    # kelimenin kendisi
    print(f"Lamma: {token.lemma_}") # kelimeinin kok hali
    print(f"POS: {token.pos_}")     # kelimenin dilbilgisel ozelligi
    print(f"Tag: {token.tag_}")     # kelimenin 
    
    