import warnings
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

from anal2 import process_full
warnings.filterwarnings('ignore', category=UserWarning)

# Φόρτωση του Dataset
df = pd.read_csv('NBADataset_with_wordcount.csv')

# Καθαρίζουμε τα άδεια κελιά (NaN)
df = df.dropna(subset=['text', 'polarity']).copy()

# Μετατροπή των polarity σε 3 ακέραιες κλάσεις (-1, 0, 1)
# διότι ο αλγόριθμος Naive Bayes απαιτεί διακριτές κατηγορίες (όχι δεκαδικά).
df['polarity_class'] = pd.cut(
    df['polarity'],
    bins=[-1.01, -0.05, 0.05, 1.01],
    labels=[-1, 0, 1]
).astype(int)

df['preprocessed_tweet'] = df['text'].apply(process_full)

X = df['preprocessed_tweet']  # eisodos
y = df['polarity_class']  # metavliti stoxos

MAX_FEATURES = 1000  # krata 1000 lekseis
TEST_SET_SIZE_PCT = 0.3  # 70% train, 30% test

# grammes=tweets, stiles=lekseis, timi keliou=arithmo emfanisis 
# ths lekshs sto tweet

cv = CountVectorizer(max_features=MAX_FEATURES)
documentTermMatrix = cv.fit_transform(df['preprocessed_tweet']).toarray()

vectorizedTweetData = pd.DataFrame(
    data=documentTermMatrix, 
    columns=cv.get_feature_names_out()
)

print('dhmioyrgithike o count-vectorizer')
print(vectorizedTweetData)

print("\n" + "-"*55)
# --- ΜΙΚΡΟΕΝΤΟΛΕΣ ΕΛΕΓΧΟΣ VECTORIZER ---
print("1. Λεξιλόγιο (50):", cv.get_feature_names_out()[:50])
print("2. 1ο tweet (>0):", 
      vectorizedTweetData.iloc[0][vectorizedTweetData.iloc[0] > 0].to_dict())
print("3. Top 10 λέξεις:", 
      vectorizedTweetData.sum().nlargest(10).to_dict())

trainingSetFeatures, testingSetFeatures, trainingSetPolarity, testingSetPolarity = train_test_split(
    vectorizedTweetData, 
    y, 
    test_size=TEST_SET_SIZE_PCT, 
    random_state=9
)

print("\n" + "="*55)
print(" 🛠️ 3. ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ (NAIVE BAYES)")
print("="*55)
print(f"Διαχωρισμός Δεδομένων:")
print(f"  -> Training set : {trainingSetFeatures.shape[0]:,} γραμμές")
print(f"  -> Testing set  : {testingSetFeatures.shape[0]:,} γραμμές\n")

print("Ekpaideusi tou montelou Multinomial Naive Bayes....", end="")
mnb = MultinomialNB(alpha=1.0, fit_prior=True)
mnb.fit(trainingSetFeatures, trainingSetPolarity)
print("oloklirothike.")

print("\n" + "="*55)
print(" 💾 4. ΕΞΑΓΩΓΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
print("="*55)

print("\n\nProvlepsi klaseon (polarity) pano sto testing set.")
testingSetPredictions = mnb.predict(testingSetFeatures)

sample10 = df.sample(n=10, random_state=42).copy()

sample10_vectors = cv.transform(sample10['preprocessed_tweet'])
sample10_predictions = mnb.predict(sample10_vectors)

sample10['predicted_polarity'] = sample10_predictions

sample10[['text', 'preprocessed_tweet', 'polarity', 'predicted_polarity']].to_csv(
    '10_random_tweets_predictions.csv',
    index=False,
    encoding='utf-8-sig'
)

print("Ta 10 tyxaia tweets apothikeftikan sto arxeio '10_random_tweets_predictions.csv'.")

df['predicted_polarity'] = mnb.predict(vectorizedTweetData)

columns_to_save = [
    'text', 'preprocessed_tweet', 'polarity', 
    'polarity_class', 'predicted_polarity'
]

df[columns_to_save].to_csv(
    'full_dataset_predictions.csv',
    index=False,
    encoding='utf-8-sig'
)

print("Oi provlepseis olou tou dataset apothikeftikan sto arxeio 'full_dataset_predictions.csv'.")

print("Metrika aksiologisis gia olo to dataset:")

print(
    "\tAccuracy:", 
    accuracy_score(df['polarity_class'], df['predicted_polarity'])
)
print(
    "\tPrecision:", 
    precision_score(df['polarity_class'], df['predicted_polarity'], average='weighted')
)
print(
    "\tRecall:", 
    recall_score(df['polarity_class'], df['predicted_polarity'], average='weighted')
)