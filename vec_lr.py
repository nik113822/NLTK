import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ΔΕΔΟΜΕΝΑ ΑΠΟ ΤΟ ΠΡΩΤΟ ΑΡΧΕΙΟ
from vec_nb import df, vectorizedTweetData, y, cv, trainingSetFeatures, testingSetPolarity, trainingSetPolarity

#warnings.filterwarnings('ignore', category=UserWarning)

print("\n" + "="*25)
print("  ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ (LOGISTIC REGRESSION)")
print("="*25)

print("Εκπαίδευση της Logistic Regression....", end="")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(trainingSetFeatures, trainingSetPolarity)
print("ολοκληρώθηκε.")

print("\n" + "="*25)
print("  ΠΡΟΒΟΛΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ")
print("="*25)

# Διαλέγουμε 10 τυχαία tweets
sample10 = df.sample(n=10, random_state=42).copy()
sample10_vectors = cv.transform(sample10['preprocessed_tweet'])
sample10['predicted_polarity'] = lr_model.predict(sample10_vectors)

# Τυπώνουμε τα αποτελέσματα (πραγματική κλάση vs πρόβλεψη)
stiles = ['polarity_class', 'predicted_polarity', 'preprocessed_tweet']
print("Δείγμα 10 τυχαίων προβλέψεων:\n")
# Κόβουμε το κείμενο στους πρώτους 70 χαρακτήρες για να μη χαλάει η οθόνη
sample10['short_tweet'] = sample10['preprocessed_tweet'].str[:70] + "..."

# Τυπώνουμε μόνο τις 3 στήλες 
stiles = ['polarity_class', 'predicted_polarity', 'short_tweet']

print(sample10[stiles].to_string(index=False))

# Προβλέψεις για όλο το dataset για να βγάλουμε τα μετρικά
df['predicted_polarity'] = lr_model.predict(vectorizedTweetData)

print("\n" + "="*25)
print(" 🏆 ΜΕΤΡΙΚΑ ΑΞΙΟΛΟΓΗΣΗΣ (LOGISTIC REGRESSION)")
print("="*25)
print(f"\tAccuracy:  {accuracy_score(df['polarity_class'], df['predicted_polarity']):.4f}")
print(f"\tPrecision: {precision_score(df['polarity_class'], df['predicted_polarity'], average='weighted'):.4f}")
print(f"\tRecall:    {recall_score(df['polarity_class'], df['predicted_polarity'], average='weighted'):.4f}")
print("="*25 + "\n")