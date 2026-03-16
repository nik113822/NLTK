import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Φόρτωση του Dataset
df = pd.read_csv('NBADataset_with_wordcount.csv')

# Καθαρίζουμε τα άδεια κελιά (NaN)
df = df.dropna(subset=['text']).copy()

# Ορίζουμε Stopwords και Lemmatizer
stops = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Αφαίρεση URLs
    text = re.sub(r'@\w+', '', text)  # Αφαίρεση Mentions
    return text


def process_full(text):
    # Καθαρισμός
    text_cleaned = clean_text(text)

    # Διάσπαση σε προτάσεις και λέξεις
    tokens_sentences = sent_tokenize(text_cleaned)
    tokens_words = [word_tokenize(token) for token in tokens_sentences]

    # Αφαίρεση Stopwords
    sentences_filtered = [
        [w for w in sent if w.lower() not in stops]
        for sent in tokens_words
    ]

    # Lemmatization
    sentences_lemmatized = [
        [lemmatizer.lemmatize(w) for w in sent]
        for sent in sentences_filtered
    ]

    # ΕΝΩΣΗ
    flat_list = [
        word for sentence in sentences_lemmatized for word in sentence
    ]
    final_clean_string = " ".join(flat_list)

    return final_clean_string


print("Επιλογή 6 τυχαίων tweets και εφαρμογή preprocessing...")

# Επιλογή 3 τυχαίων γραμμών
sample_df = df.sample(n=6).copy()

# Εφαρμογή της συνάρτησης καθαρισμού
sample_df['preprocessed_tweet'] = sample_df['text'].apply(process_full)

final_csv_df = sample_df[['preprocessed_tweet', 'polarity', 'twitter_id']]

# Αποθήκευση στο νέο αρχείο CSV
final_csv_df.to_csv(
    'random_sample_check.csv', 
    index=False, 
    encoding='utf-8-sig'
)

print("Τα αποτελέσματα αποθηκεύτηκαν επιτυχώς στο αρχείο 'random_sample_check.csv'.")