import pandas as pd
import re 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer  

#Φόρτωση του Dataset
df = pd.read_csv('NBADataset - 12-07-2020 till 19-09-2020.csv')

# Ορίζουμε Stopwords, Stemmer και Lemmatizer
stops = set(stopwords.words("english"))
ps = PorterStemmer()            
lemmatizer = WordNetLemmatizer() 

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Αφαίρεση URLs
    text = re.sub(r'@\w+', '', text)                  # Αφαίρεση Mentions
    return text


def process_full(text):
    
    #Καθαρισμός
    text_cleaned = clean_text(text)
    
    #Διάσπαση σε προτάσεις
    tokens_sentences = sent_tokenize(text_cleaned) 
    
    #Διάσπαση κάθε πρότασης σε λέξεις
    tokens_words = [word_tokenize(token) for token in tokens_sentences]
    
    #Stopwords 
    sentences_filtered = [[w for w in sent if w.lower() not in stops] for sent in tokens_words]
    
    # Stemming
    sentences_stemmed = [[ps.stem(w) for w in sent] for sent in sentences_filtered]

    #Lemmatization
    sentences_lemmatized = [[lemmatizer.lemmatize(w) for w in sent] for sent in sentences_filtered]
    
    return sentences_filtered, sentences_stemmed, sentences_lemmatized

print("--- ΑΠΟΤΕΛΕΣΜΑΤΑ---")

sample_texts = df['text'].head(6)

for i, text in enumerate(sample_texts):
    
    filt, stem, lem = process_full(text)
    
    print(f"\n{'='*20} TWEET #{i+1} {'='*20}")
    print(f"ΑΡΧΙΚΟ:      {text.strip()}") # strip() για να μην έχει κενά στην αρχή/τέλος
    print("-" * 60)
    print(f"NO STOPWORDS: {filt}")
    print("-" * 60)
    print(f"STEMMED:      {stem}")
    print("-" * 60)
    print(f"LEMMATIZED:   {lem}")
    print("\n") # Κενή γραμμή στο τέλος