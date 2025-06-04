# metinlerde bulunan fazla bosluklari ortadan akldir

text = "Hello,      World!        2035"


cleaned_text = "".join(text.split())
print(cleaned_text)

# %% buyuk -> kucuk harf cevrimi

text = "Hello, World! 2035"
cleaned_text_1 = text.lower()
print(f"text: {text} \n lower_text: {cleaned_text_1}")


# %% noktalama isaretlerini kaldir 
import string

text = "Hello, World! 2035."
cleaned_text_2 = text.translate(str.maketrans("","", string.punctuation))
print(f"text: {text} \n translate_text: {cleaned_text_2}")


# %% ozel karakterleri kaldir --> %, @, /, *, #
import re

text = "Hello, World! 2035%#"
cleaned_text_3 = re.sub(r"[^A-Za-z0-9\s]", "", text)
print(f"text: {text} \n re_text: {cleaned_text_3}")



# %% yazim hatalarını duzelt
from textblob import TextBlob # metin analizlerinde kullanılan bir kütüphane


text = "Hellıo, World! 2035..."
cleaned_text_4 = TextBlob(text).correct() # correct: yazim hatalarini düzeltir.
print(f"text: {text} \n blob_text: {cleaned_text_4}")




# %% html ya da url etiketlerini kaldir
from bs4 import BeautifulSoup

'''
** BeautifulSoup ile html yapsını parse et, get_text ile text kismini çek. **
'''

html_text = "<div>Hellıo, World! 2035...<div>"
cleaned_text_5 = BeautifulSoup(html_text, "html.parser").get_text()
print(f"text: {html_text} \n bs4_text: {cleaned_text_5}")



















