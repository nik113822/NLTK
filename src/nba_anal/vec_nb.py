import warnings
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.model_selection import StratifiedKFold
from src.nba_anal.anal2 import process_full
import time
warnings.filterwarnings('ignore', category=UserWarning)

def run_nb():
    start = time.time()
    # Φόρτωση του Dataset
    df = pd.read_csv('NBADataset_with_wordcount.csv')

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

    MAX_FEATURES = 12000  # krata 1000 lekseis
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
        random_state=None
    )

    print("\n" + "="*55)
    print("  3. ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ (NAIVE BAYES)")
    print("="*55)
    print(f"Διαχωρισμός Δεδομένων:")
    print(f"  -> Training set : {trainingSetFeatures.shape[0]:,} γραμμές")
    print(f"  -> Testing set  : {testingSetFeatures.shape[0]:,} γραμμές\n")

    print("Ekpaideusi tou montelou Multinomial Naive Bayes....", end="")
    mnb = MultinomialNB(alpha=1.0, fit_prior=True)
    mnb.fit(trainingSetFeatures, trainingSetPolarity)
    print("oloklirothike.")

    #print("\n" + "="*55)
    #print("  3.5 ΑΞΙΟΛΟΓΗΣΗ ΜΕ K-FOLD CROSS VALIDATION")
    #print("="*55)
    
    # 1. Ορίζουμε τον μηχανισμό Stratified K-Fold (10 φάσεις, με ανακάτεμα)
    #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # Εκτέλεση του K-Fold
    #cv_scores = cross_val_score(
        #mnb, 
        #vectorizedTweetData, 
        #y, 
        #cv=skf, 
        #scoring='accuracy'
    #)

    #print("Accuracy για κάθε ένα από τα 10 Folds:")
    #for i, score in enumerate(cv_scores, 1):
        #print(f"  Φάση {i}: {score:.2%}")
    
    #print("-" * 55)
    #print(f" Τελικό (Μέσο) Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
    #print("="*55 + "\n")
    
    print("\n" + "="*55)
    print("  3.5 ΑΞΙΟΛΟΓΗΣΗ ΜΕ 10 x STRATIFIED K-FOLD")
    print("="*55)

    all_means = []

    with open("apotel.log", "a", encoding="utf-8") as log:
        log.write("\n=== MULTINOMIAL NAIVE BAYES - 10 x STRATIFIED K-FOLD ===\n")

        for run in range(1, 11):
            skf = StratifiedKFold(n_splits=10, shuffle=True)

            cv_scores = cross_val_score(
            mnb,
            vectorizedTweetData,
            y,
            cv=skf,
            scoring='accuracy'
            )

            mean_score = cv_scores.mean()
            all_means.append(mean_score)

            print(f"Run {run}: {mean_score:.2%}")
            log.write(f"Run {run}: Meso Accuracy = {mean_score:.6f}\n")
            log.flush()

        final_mean = sum(all_means) / len(all_means)

        print("-" * 55)
        print(f"ΤΕΛΙΚΟΣ ΜΕΣΟΣ ΟΡΟΣ 10 RUNS: {final_mean:.2%}")
        print("="*55 + "\n")

        log.write("-" * 55 + "\n")
        log.write(f"TELIKOS MESOS OROS 10 RUNS: {final_mean:.6f}\n")
        log.flush()

    
    
    
    #final_mean = "-"
    
    print("  4. ΕΞΑΓΩΓΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
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
        'full_dataset_nb_predictions.csv',
        index=False,
        encoding='utf-8-sig'
    )

    print("Oi provlepseis olou tou dataset apothikeftikan sto arxeio 'full_dataset_nb_predictions.csv'.")

    print("Metrika aksiologisis tou testing set:")
    print("\tAccuracy:", accuracy_score(testingSetPolarity, testingSetPredictions))
    print("\tPrecision:", precision_score(testingSetPolarity, testingSetPredictions, average='weighted'))
    print("\tRecall:", recall_score(testingSetPolarity, testingSetPredictions, average='weighted'))
    print("\tF1-Score:", f1_score(testingSetPolarity, testingSetPredictions, average='weighted'))
    print(f"\nXronos ektelesis: {(time.time() - start)/60:.2f} lepta")
    #Dictionary)
    return {
        "Μοντέλο": "Naive Bayes",
        "K-Fold Mean Accuracy": final_mean,
        "Accuracy": accuracy_score(testingSetPolarity, testingSetPredictions),
        "Precision": precision_score(testingSetPolarity, testingSetPredictions, average='weighted'),
        "Recall": recall_score(testingSetPolarity, testingSetPredictions, average='weighted'),
        "F1-Score": f1_score(testingSetPolarity, testingSetPredictions, average='weighted')
    }
