'''

varlik ismi tanima: metin (cumle) -> metin icersinde bulunan varlik isimlerini tanimla
'''

# import libraries

import pandas as pd
import spacy
import en_core_web_sm


# spacy modeli ile varlik ismi tanimla
## nlp = spacy.load("en_core_web_sm") # spacy kutuphanesi ingilizce dil modeli
nlp = en_core_web_sm.load()
content = "Alice works at Amazon and lives in London. She visited the British Museum last weekend."


doc = nlp(content) # bu islem metindeki varliklari (entities) analiz eder 


for ent in doc.ents:
    # ent.text: varlik isimi (Alice, Amazon)
    # ent.start_char & ent.end_char: varligin metin icindeki baslangic ve bitis karakterleri
    # ent.label_: varlik turu
    # print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print(ent.text, ent.label_)
    
# ent.lemma_: varligin kok hali
entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]



# varlik listesini pandas df'e cevir
df = pd.DataFrame(entities, columns = ['entities', 'type', 'lemma'])
    
    


