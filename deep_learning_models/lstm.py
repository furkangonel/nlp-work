'''
metin uretimi

LSTM train with text data

text data = gpt ile olustur

'''

# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# egitim verisi chatgpt ile olustur
texts = [
    "bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum",
    "kitap okumak çok güzel beni gerçekten çok mutlu ediyor",
    "arkadaşlarımla vakit geçirmek bana iyi geliyor",
    "yeni bir diziye başladım ve gerçekten çok sürükleyici",
    "kahvemi alıp sahilde oturmak gibisi yok",
    "bugün kendimi biraz yorgun hissediyorum",
    "spor yapmak hem zihnimi hem bedenimi rahatlatıyor",
    "akşam yemeği için güzel bir tarif denemek istiyorum",
    "doğada vakit geçirmek bana huzur veriyor",
    "yarın erken kalkmam gerekiyor, bu yüzden erken yatacağım",
    "çalışma ortamımın düzenli olması verimliliğimi artırıyor",
    "bugün güzel bir şarkı keşfettim, tüm gün onu dinledim",
    "yağmurun sesi beni her zaman rahatlatmıştır",
    "alışveriş yapmak bazen kafa dağıtmak için iyi geliyor",
    "yeni bir hobi edinmek istiyorum, belki resim yaparım",
    "bugün uzun zamandır görüşmediğim bir arkadaşımla konuştum",
    "kahvaltıda taze simit ve çay gibisi yok",
    "film izlemek için mükemmel bir akşamdı",
    "kitapçıda saatlerce vakit geçirmek en sevdiğim şeylerden biri",
    "akşam yürüyüşü sırasında gün batımını izlemek çok güzeldi",
    "yeni bir tarif denedim ve sonuç harikaydı",
    "bahar aylarını gerçekten çok seviyorum",
    "bugün kendime biraz zaman ayırdım ve çok iyi geldi",
    "müzik ruh halimi anında değiştirebiliyor",
    "sessiz bir ortamda çalışmak odaklanmamı kolaylaştırıyor",
    "sıcak bir çorba içmek beni çocukluğuma götürüyor",
    "tatil planları yaparken çok heyecanlanıyorum",
    "dün gece gördüğüm rüya hala aklımda",
    "bugün kendimi motive hissetmiyorum",
    "temiz hava almak için camı açtım, çok ferahlatıcıydı",
    "alışkanlıkları değiştirmek zaman alıyor ama imkansız değil",
    "bir fincan kahveyle güne başlamak ritüelim oldu",
    "bugün biraz melankolik hissediyorum",
    "hafta sonunu dört gözle bekliyorum",
    "sabah erkenden uyanmak bana verim kazandırıyor",
    "yeni şeyler öğrenmeyi seviyorum",
    "bugün oldukça üretken bir gün geçirdim",
    "çok yorucu bir günün ardından duş almak gibisi yok",
    "kendi başıma vakit geçirmeye ihtiyaç duyuyorum",
    "şehirden uzaklaşmak bana çok iyi geliyor",
    "sevdiğim bir şarkıyı duymak yüzümü güldürüyor",
    "bugün güne enerjik başladım",
    "zaman zaman eski fotoğraflara bakmayı seviyorum",
    "evcil hayvanımla vakit geçirmek beni mutlu ediyor",
    "hafta sonu için plan yapmak istiyorum",
    "sessiz bir kütüphanede çalışmak huzur verici",
    "kış aylarında sıcak içecekler daha da anlam kazanıyor",
    "doğayla iç içe olmak stresimi azaltıyor",
    "yeni bir deftere yazı yazmak her zaman heyecan verici",
    "bugün pencereden dışarı bakarken hayallere daldım",
    "kendime yeni hedefler koydum",
    "bugün oldukça motive hissediyorum",
    "biraz yürüyüş yapmak istiyorum",
    "dışarıda yağmur yağıyor, kitap okumak için harika bir ortam",
    "arkadaşım bana moral verdi, çok mutlu oldum",
    "hayatın küçük güzellikleri en değerli olanlar",
    "bugün keyifli bir sohbet gerçekleştirdim",
    "sade bir hayat yaşamak istiyorum",
    "işlerim biraz birikti ama planlı gidersem hallederim",
    "bugün güzel bir haber aldım",
    "kendi sınırlarımı zorlamayı seviyorum",
    "başarı için çaba göstermek şart",
    "gün batımını izlemek en sevdiğim anlardan biri",
    "bugün geçmişi düşündüm ve biraz hüzünlendim",
    "daha fazla kitap okumak istiyorum",
    "bugün biraz yalnız kalmak istedim",
    "mutluluk bazen küçük şeylerde gizlidir",
    "yarın için heyecanlıyım",
    "bugün yoğun ama verimli geçti",
    "kendime zaman ayırmayı öğreniyorum",
    "bugün güzel bir kahve yaptım",
    "hayatın tadını çıkarmak gerektiğini düşünüyorum",
    "bugün biraz nostalji yaptım",
    "yeni bir şehri keşfetmek istiyorum",
    "bugün içimde bir huzur var",
    "birlikte vakit geçirdiğimizde zamanın nasıl geçtiğini anlamıyorum",
    "bugün kendime bir ödül verdim",
    "çocukluk anılarım bazen gözümde canlanıyor",
    "bugün farklı bir şey denemek istedim",
    "sabah kahvesi içmeden uyanmış sayılmam",
    "gökyüzünü izlemek beni rahatlatıyor",
    "bugün evde temizlik yapmayı planlıyorum",
    "birlikte yemek yapmak çok keyifli",
    "bugün üretken bir ruh halindeyim",
    "bazı günler sadece dinlenmek istiyorum",
    "gelecek planlarım üzerine düşünüyorum",
    "bugün biraz müzik dinleyip rahatladım",
    "bugün dışarıda piknik yaptık, çok keyifliydi",
    "tatil fikri beni mutlu ediyor",
    "bugün farklı bir bakış açısı kazandım",
    "bir dostla yapılan samimi sohbetin yerini hiçbir şey tutamaz",
    "bugün kendime iyi davrandım",
    "bir kahkaha bütün günümü güzelleştirdi",
    "bugün güne güzel bir haberle başladım",
    "zihnimi boşaltmak için meditasyon yaptım",
    "yeni başlangıçlar için heyecan duyuyorum",
    "bugün pozitif düşünmeye çalıştım",
    "küçük bir jest bazen büyük mutluluklar yaratabilir",
    "bugün eski bir dosttan mesaj aldım",
    "yürüyüş sırasında çiçek kokuları bana çocukluğumu hatırlattı",
    "bugün evde film gecesi yapıyoruz",
    "sıcak bir battaniye ve kitap en iyi kombinasyon olabilir",
    "bugün mutfağa girip yeni bir tatlı denedim",
    "yaşamın akışına bırakmayı öğreniyorum",
    "bugün biraz içsel sorgulama yaptım",
    "bazen sadece sessizlik yeterli"
]



# %% metin temizleme ve preprocessing asamaları: tokenization, padding, label encoding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)  # metinler uzeirndeki kelime frekanslarini ogren
total_words = len(tokenizer.word_index) + 1  # toplam kelime sayisi


# n-gram modeli olustur ve padding uygula
input_sequences = []
for text in texts:
    # metinleri kelime indekslerine cevir
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # her metin icin n-gram dizisi olusturalim
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        

# en uzun diziyi bulalim, tum dizileri ayni uzunluga getirelim
max_sequence_length = max(len(x) for x in input_sequences)
    
# dizilere padding gislemi uygula, hepsinin aynı zuunlukta olmasini sagla
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_length, padding = "pre")


# X (input) ve y (target)
X = input_sequences[:,:-1] # tum satirlar, sondan 1 eksik sutunlar (girdi)
y = input_sequences[:, -1] # her satirin son sutunu

y = tf.keras.utils.to_categorical(y, num_classes = total_words) # one-hot encoding


# %% LSTM modeli olustur: compile, train ve evaluate


model = Sequential()


# embedding
model.add(
    Embedding(
        total_words,
        50,
        input_length = X.shape[1]
        ))

# lstm

model.add(
    LSTM(
        100, 
        return_sequences = False
        ))



# output
model.add(
    Dense(
        total_words,
        activation = "softmax"
        ))


# model compile
model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"])


# model training
model.fit(X, y, epochs = 100, verbose = 1)


# %%  model prediction


def generate_text(seed_text, next_words):
    
    for _ in range(next_words):
        
        # girdi metnini sayisal verilere donustur
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        
        # padding
        token_list = pad_sequences([token_list], maxlen = max_sequence_length-1, padding="pre")
        
        # prediction
        predicted_probabilities = model.predict(token_list, verbose = 0)
        
        
        # en yuksek olasiliga sahip kelimenin indexini bul
        predicted_word_index = np.argmax(predicted_probabilities, axis = -1)
        
        # tokenizer ile kelime indexinden asil kelimeyi bul
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        
        
        # tahmin edilen kelimeyi sedd_text e ekleyelim
        seed_text = seed_text + " " + predicted_word
        
    return seed_text



seed_text = "rüya"
print(generate_text(seed_text, 3))





# %% TASK -> fiil geldiğinde cümleyi bitir işlemi ekle. bununla birlikte paragraf oluştur.




















