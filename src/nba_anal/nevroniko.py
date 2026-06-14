import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

from src.nba_anal.anal2 import process_full
import time

def run_mlp():
    start = time.time()



    # 1. Fortwsi dataset
    df = pd.read_csv('NBADataset_with_wordcount.csv')


    # 2. Metatropi polarity se 3 klaseis
    df['polarity_class'] = pd.cut(
        df['polarity'],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=[-1, 0, 1]
    ).astype(int)

    # 3. Preprocessing
    df['preprocessed_tweet'] = df['text'].apply(process_full)

    # 4. Text -> numbers me CountVectorizer
    cv = CountVectorizer(max_features=12000)
    X = cv.fit_transform(df['preprocessed_tweet'])
    y = df['polarity_class']

    # 5. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=9
    )

    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, random_state=9,verbose=True)
    mlp.fit(X_train, y_train)

    # 7. Predictions
    y_pred = mlp.predict(X_test)

    results = pd.DataFrame({
        'Tweet': df.loc[y_test.index, 'text'].str[:90],
        'True': y_test,
        'Pred': mlp.predict(X_test)
    })

    # Εμφανίζει κατευθείαν 5 τυχαία tweets
    print(results.sample(5))

    # 8. Metrikes
    """
    Accuracy = συνολικό ποσοστό σωστών
    Precision = πόσο σωστές προβλέψεις
    Recall = πόσα σωστά εντόπισε
    """
    print("\n=== mlp nevroniko ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall   :", recall_score(y_test, y_pred, average='weighted'))
    print("F1-Score :", f1_score(y_test, y_pred, average='weighted'))

    # 9. K-Fold
    #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=9)
    #cv_scores = cross_val_score(mlp, X, y, cv=skf, scoring='accuracy')

    #print("\n=== K-FOLD CROSS VALIDATION ===")
    #for i, score in enumerate(cv_scores, 1):
        #print(f"Fold {i}: {score:.2%}")

    #print("Meso Accuracy:", cv_scores.mean())
    # 9. K-Fold επαναλήψεις για MLP
    all_means = []

    print("\n=== REPEATED STRATIFIED K-FOLD FOR MLP ===")

    with open("apotel.log", "a", encoding="utf-8") as log:
        log.write("\n=== MLP - REPEATED STRATIFIED K-FOLD ===\n")

        for run in range(1, 11):  
            skf = StratifiedKFold(n_splits=10, shuffle=True)

            cv_scores = cross_val_score(
                mlp,
                X,
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

        print("\nTELIKOS MESOS OROS MLP:", final_mean)
        log.write(f"\nTELIKOS MESOS OROS MLP: {final_mean:.6f}\n")
        log.flush()



    print("\n=== ΕΞΑΓΩΓΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΕ CSV ===")

    df['predicted_polarity_mlp'] = mlp.predict(X)

    columns_to_save = [
        'text', 
        'preprocessed_tweet', 
        'polarity', 
        'polarity_class', 
        'predicted_polarity_mlp'
    ]

    # Αποθήκευση
    output_filename = 'MLP_full_dataset_predictions.csv'

    df[columns_to_save].to_csv(
        output_filename, 
        index=False, 
        encoding='utf-8-sig'
    )

    print(f"✅ Τα αποτελέσματα αποθηκεύτηκαν επιτυχώς στο αρχείο '{output_filename}'!")

    print(f"\nXronos ektelesis: {(time.time() - start)/60:.2f} lepta")
    # Πακετάρουμε τα αποτελέσματα σε ένα Λεξικό (Dictionary)
    return {
        "Μοντέλο": "MLP",
        "K-Fold Mean Accuracy": final_mean,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-Score": f1_score(y_test, y_pred, average='weighted')
    }