from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, gutenberg
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
import nltk


keim = (
    "Jrue Holiday, Khris Middleton and Brook Lopez, "
    "the rest of that championship core, are elsewhere. "
    "Lillard saw his two-year tenure in Milwaukee cut short "
    "after he tore an Achilles in the first round of the "
    "playoffs last season and was waived, an unceremonious "
    "end to what was supposed to be a long-term, "
    "dominant partnership."
)

print(sent_tokenize(keim))  # xorizei protaseis
print(word_tokenize(keim))  # xorizei kai ektyponei tis lekseis

for i in word_tokenize(keim):
    print(i)  # ektiponei mia-mia tis lekseis

# Xorizei to keimeno keim se lekseis kai
# ftiakhnei lista mono me oses den einai agglika stopwords.
stop_words = set(stopwords.words("english"))
words = word_tokenize(keim)
# print(stop_words)
filteredprot = [w for w in words if w not in stop_words]
print(filteredprot)

# STEMMING
ps = PorterStemmer()
example_words = [
    "dribble",
    "dribbler",
    "dribbling",
    "dribbled",
    "dribbly",
]

for w in example_words:
    print(ps.stem(w))

print()

new_text = (
    "It is very important to be reboundly while you "
    "are rebounding with rebound. All rebounders have "
    "rebounded wisely."
)

words = word_tokenize(new_text)
for w in words:
    print(ps.stem(w))

print()

# LEMMATIZATION
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("balling"))
print(lemmatizer.lemmatize("ring"))  # proepilogi to n-noun
print(lemmatizer.lemmatize("playing", pos="v"))

# corpora-gutenberg
sample = gutenberg.raw("carroll-alice.txt")  # pairnei olo to keimeno apo to txt
tok = sent_tokenize(sample)  # to keimeno xorizetai se lista apo protaseis
print(tok[2:5])  # ektyposi grammes 3 eos 5

# WORDNET
nltk.download("wordnet")    # vasiko leksiko
nltk.download("omw-1.4")    # Open Multilingual WordNet

s = wn.synsets("BASKETBALL")           # lista ennoiwn gia BASKETBALL
print(s[0].definition())               # orismos tis protis ennoias
print([l.name() for l in s[0].lemmas()])   # synonyma
print(s[0].hypernyms())                # genikotere ennoies
print(s[0].hyponyms()[:3])             # merikes eidikotere
