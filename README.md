1# 🏀 ΑΝΑΛΥΣΗ ΣΥΝΑΙΣΘΗΜΑΤΟΣ ΣΕ ΑΘΛΗΤΙΚΑ NBA TWEETS

Αυτό το project αναλύει συναισθήματα (sentiment analysis) από tweets που αφορούν το NBA. 
Χρησιμοποιεί τεχνικές Επεξεργασίας Φυσικής Γλώσσας (NLP) και αλγορίθμους Μηχανικής Μάθησης (Machine Learning) για να κατατάξει τα tweets σε θετικά, αρνητικά ή ουδέτερα.

2)📂 Δομή του Project
Το project είναι  οργανωμένο με βάση το **`src` layout**:
* `src/nba_anal/`: Περιέχει όλον τον πηγαίο κώδικα (συναρτήσεις προεπεξεργασίας, δημιουργία γραφημάτων, εκπαίδευση και αξιολόγηση μοντέλων).
* `main.py`: Το κεντρικό αρχείο εκτέλεσης (Command Line Interface).

3) Οδηγίες Χρήσης
Διαθέσιμες Εντολές:

python main.py explore ➜ Εξερεύνηση Δεδομένων (EDA)

python main.py preprocess ➜ Προεπεξεργασία (NLP)

python main.py graphs ➜ Δημιουργία Γραφημάτων

python main.py lr ➜ Εκπαίδευση Logistic Regression

python main.py nb ➜ Εκπαίδευση Naive Bayes

python main.py mlp ➜ Εκπαίδευση Νευρωνικού Δικτύου

4)Εγκατάσταση Απαιτήσεων
Βεβαιωθείτε ότι έχετε εγκατεστημένη την Python. Στη συνέχεια, ανοίξτε το τερματικό σας και εγκαταστήστε τις απαιτούμενες βιβλιοθήκες:
```bash
pip install pandas scikit-learn nltk matplotlib seaborn
