# 1) TOKENIZATION
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Κατέβασε το tokenizer την πρώτη φορά
nltk.download("punkt")

text = ("Brown averaged 26.6 PPG several seasons ago, but now that PF/SF Jayson Tatum is out for much of the season, he needs to score more. "
"SG/PG Derrick White might be the better value coming two rounds later, but Brown will see a career-best in shots. "
"Tatum (Achilles) makes his season debut in March. Stash him if you have an injured list slot, but not for the bench in redraft formats.")

# Σπάσιμο σε προτάσεις
sentences = sent_tokenize(text)
print("Προτάσεις:", sentences)

# Σπάσιμο σε λέξεις
tokens = word_tokenize(text)
print("Λέξεις:", tokens)
