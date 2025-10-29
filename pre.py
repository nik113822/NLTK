f=open("python.txt")
text=f.read()
print(text)
from nltk import sent_tokenize
import nltk
from nltk import word_tokenize
tokens = sent_tokenize(text)#diaspash se protaseis
print(tokens)

tokens = [word_tokenize(token) for token in tokens]#diaspash se lekseis
print(tokens)


#stopwords
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
#παίρνει λίστα προτάσεων (κάθε πρόταση = λίστα tokens),
#kai αφαιρεί από κάθε πρόταση όσα tokens είναι stopwords,
sentences_filtered = [[w for w in sent if w.lower() not in stops] for sent in tokens]
print("Oi filtrarismenes portaseis einai :", sentences_filtered)
 #stemming
print()
from nltk.stem import PorterStemmer
ps=PorterStemmer()
for sentence in sentences_filtered[:9]:
    sentence_stemmed=[ps.stem(w) for w in sentence]
    print(sentence)
    print(sentence_stemmed)

#lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
for sentence in sentences_filtered[:9]:
    sentence_lemmatized=[lemmatizer.lemmatize(w) for w in sentence]
    print(sentence)
    print()
    print("oi lemmatized protaseis einai : ", sentence_lemmatized)

#afairesi @mentions
import re #regular expressions

# αφαιρεί @mentions όπως @giros
clean = re.sub(r"\B@[\w.-]+", "", text)

print(clean)
print(text)

#METATROPI THS TELEYTAIAS PROTASHS TOU TXT SE PEZA GRAMMATA
# σπάει σε προτάσεις με ., ! ή ?
sents = re.split(r'(?<=[.!?])\s+', text)
if sents:
    sents[-1] = sents[-1].lower()      # τελευταία πρόταση σε πεζά
    print(sents[-1])                   # εκτύπωση μόνο της τελευταίας

#afairesh url
NEO = re.sub(r'((?:https?://|www\.)\S+)([.,!?;:]?)', r'\2', text)
print(NEO)  

















