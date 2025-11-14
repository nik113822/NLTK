import re

import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import preprocessor as p

with open("python.txt", encoding="utf-8") as f:
    text = f.read()
print(text)

tokens = sent_tokenize(text)  # diaspash se protaseis
print(tokens)

tokens = [word_tokenize(token) for token in tokens]  # diaspash se lekseis
print(tokens)


stops = set(stopwords.words("english"))
# pairnei lista protaseon (kathe protash = lista tokens),
# kai afairei apo kathe protash osa tokens einai stopwords,
sentences_filtered = [[w for w in sent if w.lower() not in stops] for sent in tokens]
print("Oi filtrarismenes portaseis einai:", sentences_filtered)
print()

# stemming
ps = PorterStemmer()
for sentence in sentences_filtered[:9]:
    sentence_stemmed = [ps.stem(w) for w in sentence]
    print(sentence)
    print(sentence_stemmed)

lemmatizer = WordNetLemmatizer()
for sentence in sentences_filtered[:9]:
    sentence_lemmatized = [lemmatizer.lemmatize(w) for w in sentence]
    print(sentence)
    print("oi lemmatized protaseis einai:", sentence_lemmatized)

# afairei @mentions opos @giros
# clean = re.sub(r"\B@[\w.-]+", "", text)
p.set_options(p.OPT.MENTION)
clean = p.clean(text)
print(clean)
print(text)

# Metatropi tis teleytaias protasis tou txt se peza grammata
# spaei se protaseis me ., ! i ?
sents = re.split(r'(?<=[.!?])\s+', text)
if sents:
    sents[-1] = sents[-1].lower()  # teleytaia protasi se peza
    print(sents[-1])  # ektuposi mono tis teleytaias

# NEO = re.sub(r'((?:https?://|www\.)\S+)([.,!?;:]?)', r'\2', text)
# print(NEO)
p.set_options(p.OPT.URL)  # afairesh url
clean = p.clean(text)
print(clean)

sample_text = "NLTK is great for NLP!"
words = word_tokenize(sample_text)
pos_tags = pos_tag(words)
print(pos_tags)
















