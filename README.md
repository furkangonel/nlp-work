# Doğal Dil İleme (NLP) Uygulamaları 📚

Bu depo, Doğal Dil İşleme (NLP) alanında temelden ileri düzeye birçok görevi içeren Python projelerini barındırır. Her klasör, belirli bir NLP konsepti üzerine odaklanmış örnekler ve açıklamalı kodlarla desteklenmiştir.

--- 

Bu depodaki kodlar ve notlar BTK Akademi Platformu üzerinde **Kaan Can Yılmaz** hocanın anlatımıyla düzenlenen "Doğal Dil İşleme" eğitimindeki örnekleri ve ek ödevleri içerir.

---

📄 `Eğitim içeriğinin kısa anlatımları:` [PDF için buraya tıklayın](./resources/BTK_NLP_14saat.pdf)

---

## 📁 Klasör Yapısı ve Açıklamaları

| Klasör | Açıklama |
|--------|----------|
| [preprocessing](./preprocessing/) `(ön işleme)` | Temel veri ön işleme adımları: küçük harfe çevirme, noktalama temizleme, stop-word kaldırma, lemmatization vb. |
| [basic_nlp_process](./basic_nlp_process//) `(temel nlp işlemleri)` | Tokenizasyon, kelime sıklığı analizi, n-gram üretimi gibi temel NLP işlemleri. |
| [text_representation](./text_representation//) `(metin temsili)` | Bag-of-Words, TF-IDF, Word2Vec gibi metin vektörleştirme tekniklerinin uygulamaları. |
| [probabilities_models](./probabilistic_models//) `(olasılıksal modeller)` |  Naive Bayes, HMM gibi olasılıksal modellerin örnek uygulamaları. |
| [entity_recognition](./entity_recognition/) `(varlık tanıma)` | Named Entity Recognition (NER) üzerine örnekler; SpaCy ve sklearn kullanımı. |
| [deep_learning_models](./deep_learning_models//) `(derin öğrenme modelleri)` | LSTM, GRU, Transformer gibi derin öğrenme tabanlı metin modelleriyle uygulamalar. |
| [advenced_tasks](./advanced_tasks/) `(gelişmiş görevler)` | Çeviri, metin özetleme, soru-cevap sistemleri gibi ileri düzey NLP görevleri. |

--- 

## Kurulum 🚀

```bash
git clone https://github.com/furkangonel/nlp-work.git
cd NLP-work
pip install -r requirements.txt
````


## Kullanım 🛠️

Projeyi Anaconda üzerinden Spyder IDE ile çalıştırmak için şu adımları takip edebilirsiniz:

### 🔧 Spyder Ortamında Çalıştırma (Anaconda Navigator ile)

1. **Anaconda Navigator** uygulamasını açın.
2. Sol menüden **Environments** (Ortamlar) sekmesine geçin.
3. Yeni bir ortam oluşturmak istiyorsanız:
   - "Create" butonuna tıklayın.
   - Ortam adı verin (örn: `nlp_env`), Python sürümünü seçin (örn: Python 3.10).
   - Oluşturduktan sonra ortamı **`Activate`** edin.
4. Ortam aktif hale geldikten sonra üstteki menüden **Spyder** IDE'yi başlatın.
5. Spyder açıldığında, sol üstten `File > Open` menüsünü kullanarak bu proje klasörlerinden birine gidin (örneğin `basic_nlp_process`) ve `.py` uzantılı dosyayı açın.
6. Kodunuzu düzenleyin ve **Shift+Enter** veya üst menüden "Run File" butonuyla çalıştırın.

> 📌 Eğer bazı kütüphaneler eksikse terminalde şu komutla kurabilirsiniz:
> ```bash
> conda install numpy pandas matplotlib scikit-learn
>




Alternatif olarak `.ipynb` dosyalarıyla çalışmak için Jupyter Notebook’u da kullanabilirsiniz:

> ```bash
> cd metin_temsili
> jupyter notebook
>





## Katkıda Bulun ✍🏼

> Bu projeye katkıda bulunmak isterseniz:
>	`1.	Bu repoyu fork’layın.`
>	`2.	Yeni bir branch oluşturun.`
>	`3.	Geliştirmelerinizi yapın ve commit edin.`
>	`4.	Bu repoya bir pull request gönderin.`