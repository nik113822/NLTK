import pandas as pd

# 1. Φόρτωση των δεδομένων
df = pd.read_csv('NBADataset - 13-09-2020 till 13-10-2020.csv')

# 2. Εκτύπωση πληροφοριών για τις στήλες
print("--- ΠΛΗΡΟΦΟΡΙΕΣ DATASET ---")
df.info()
print("\n")

# Δημιουργία πίνακα με τον Τύπο δεδομενων και την 1η Εγγραφή (iloc[0]) κάθε στήλης
print(pd.DataFrame({'Type': df.dtypes, 'Sample': df.iloc[0]}))
print("\n")

# 3. Υπολογισμός μοναδικών χρηστών
unique_users_count = df['screenname'].nunique()
print(f"Συνολικά υπάρχουν {unique_users_count} διακριτοί χρήστες στο dataset.")

# 4. Έλεγχος για κενές τιμές (Εμφάνιση ΜΟΝΟ των προβληματικών στηλών)
print("\nMISSING VALUES:")
missing_vals = df.isnull().sum()
print(missing_vals[missing_vals > 0])

# 5. Στατιστικά Polarity
print("\n--- ΣΤΑΤΙΣΤΙΚΑ POLARITY ---")
print(f"Εύρος Τιμών: Από {df['polarity'].min()} έως {df['polarity'].max()}")
print()
print("Θετικά (+):  ", (df['polarity'] > 0).sum())
print("Αρνητικά (-):", (df['polarity'] < 0).sum())
print("Ουδέτερα (0):", (df['polarity'] == 0).sum())

# 6. Το μοτίβο για τα emoticons
pattern = r"[:;=8][\-]?[)DdpP(|]"

# Ψάχνει το μοτίβο σε ΟΛΗ τη στήλη και μετράει πόσα βρήκε
num_emoticons = df['text'].str.contains(pattern, regex=True).sum()

print("\n--- ΑΠΟΤΕΛΕΣΜΑ ---")
print(f"Tweets με emoticons: {num_emoticons}")
print(f"Ποσοστό: {(num_emoticons / len(df)) * 100:.2f}%")


print("Μέσος όρος λέξεων ανά tweet:", df['text'].str.split().str.len().mean())



