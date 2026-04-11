from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from vec_nb import trainingSetFeatures, testingSetFeatures, trainingSetPolarity, testingSetPolarity

print("Ξεκινάει ο KNN... (Κάνε λίγη υπομονή, είναι βαρύς αλγόριθμος)")

# 1. Ορίζουμε το μοντέλο
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# 2. Το ταΐζουμε με τα δεδομένα εκπαίδευσης
knn.fit(trainingSetFeatures, trainingSetPolarity)

# 3. Κάνουμε πρόβλεψη πάνω στα δεδομένα ελέγχου
predictions = knn.predict(testingSetFeatures)

# 4. Τυπώνουμε το τελικό σκορ
print(f"Τελικό Accuracy KNN: {accuracy_score(testingSetPolarity, predictions):.4f}")