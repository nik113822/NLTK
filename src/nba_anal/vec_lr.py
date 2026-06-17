import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

from src.nba_anal.anal2 import process_full

def run_lr():

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
        X, y, test_size=0.3, random_state=None
    )

    # 6. Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=9)
    lr.fit(X_train, y_train)

    # 7. Predictions
    y_pred = lr.predict(X_test)

    results = pd.DataFrame({
        'Tweet': df.loc[y_test.index, 'text'].str[:90],
        'True': y_test,
        'Pred': lr.predict(X_test)
    })

    # Εμφανίζει κατευθείαν 5 τυχαία tweets
    print(results.sample(5))

    # 8. Metrikes
    """
    Accuracy = συνολικό ποσοστό σωστών
    Precision = πόσο σωστές προβλέψεις
    Recall = πόσα σωστά εντόπισε
    """
    print("\n=== LOGISTIC REGRESSION ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall   :", recall_score(y_test, y_pred, average='weighted'))
    print("F1-Score :", f1_score(y_test, y_pred, average='weighted'))
    

    # 9. K-Fold
    #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=9)
    #cv_scores = cross_val_score(lr, X, y, cv=skf, scoring='accuracy')

    #print("\n=== K-FOLD CROSS VALIDATION ===")
    #for i, score in enumerate(cv_scores, 1):
        #print(f"Fold {i}: {score:.2%}")

    #print("Meso Accuracy:", cv_scores.mean())
    # 9. K-Fold 10 φορές
    
    
    all_means = []

    print("\n=== 10 x K-FOLD CROSS VALIDATION ===")

    with open("apotel.log", "a", encoding="utf-8") as log:
        log.write("\n=== LOGISTIC REGRESSION - 10 x STRATIFIED K-FOLD ===\n")

        for run in range(1, 11):
            skf = StratifiedKFold(n_splits=10, shuffle=True)
            cv_scores = cross_val_score(lr, X, y, cv=skf, scoring='accuracy')

            mean_score = cv_scores.mean()
            all_means.append(mean_score)

            print(f"\nRun {run}")
            log.write(f"\nRun {run}\n")

            for i, score in enumerate(cv_scores, 1):
                print(f"Fold {i}: {score:.2%}")
                log.write(f"Fold {i}: {score:.6f}\n")

            print("Meso Accuracy:", mean_score)
            log.write(f"Meso Accuracy: {mean_score:.6f}\n")
            log.flush()

        final_mean = sum(all_means) / len(all_means)

        print("\nTELIKOS MESOS OROS 10 RUNS:", final_mean)
        log.write(f"\nTELIKOS MESOS OROS 10 RUNS: {final_mean:.6f}\n")
        log.flush()

    
    
    
    #final_mean = "-"
    print("\n=== ΕΞΑΓΩΓΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΕ CSV ===")

    # Το μοντέλο (lr) μαντεύει την κλάση για ΟΛΑ τα tweets του dataset (X)
    df['predicted_polaritylr'] = lr.predict(X)

    columns_to_save = [
        'text', 
        'preprocessed_tweet', 
        'polarity', 
        'polarity_class', 
        'predicted_polaritylr'
    ]

    # Αποθήκευση
    output_filename = 'LR_full_dataset_predictions.csv'

    df[columns_to_save].to_csv(
        output_filename, 
        index=False, 
        encoding='utf-8-sig'
    )

    print(f"✅ Τα αποτελέσματα αποθηκεύτηκαν επιτυχώς στο αρχείο '{output_filename}'!")

    # elegxos an anelyse olo to dataset
    print(
        f"Check -> preprocessed: {df['preprocessed_tweet'].notna().sum()}, "
        f"vectorized: {X.shape[0]}, "
        f"train+test: {X_train.shape[0] + X_test.shape[0]}"
    )
  #Dictionary&
    return {
        "Μοντέλο": "Logistic Regression",
        "K-Fold Mean Accuracy": final_mean,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-Score": f1_score(y_test, y_pred, average='weighted')
    }