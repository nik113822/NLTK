import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Λύση για το σφάλμα του Tkinter στα Windows
matplotlib.use('Agg')

df = pd.read_csv('NBADataset - 12-07-2020 till 19-09-2020.csv')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

#1.Υπολογισμοί
pos = (df['polarity'] > 0).sum()
neg = (df['polarity'] < 0).sum()
neu = (df['polarity'] == 0).sum()

# Στήσιμο γραφήματος barchart polarity
categories = ['Θετικά (+)', 'Αρνητικά (-)', 'Ουδέτερα (0)']
counts = [pos, neg, neu]
colors = ['green', 'red', 'black']

# Σχεδίαση και Αποθήκευση
sns.barplot(x=categories, y=counts, palette=colors)
plt.title('Κατανομή Συναισθήματος (Polarity)')
plt.savefig('polarity_bar.png', bbox_inches='tight', dpi=300)
plt.clf()

# 2. ΙΣΤΟΓΡΑΜΜΑ: ΜΗΚΟΣ TWEETS
w = df['text'].str.split().str.len()
w.plot.hist(bins=30, color='green', title='Κατανομή Λέξεων')
plt.axvline(w.mean(), color='red', linestyle='dashed', label='Μέσος')
plt.xlabel('Αριθμός Λέξεων')
plt.ylabel('Συχνότητα (Αριθμός Tweets)')
plt.legend()
plt.savefig('hist_words.png')
plt.clf()

# 3. ΠΙΤΑ: EMOTICONS
df['text'].str.contains(
    r"[:;=8][\-]?[)DdpP(|]", regex=True, na=False
).value_counts().rename(
    {False: 'Χωρίς', True: 'Με Emoticons'}
).plot.pie(
    autopct='%1.1f%%',
    ylabel='',
    title='Ποσοστά tweets με Emoticons'
)
plt.savefig('pie_emoticons.png')
plt.clf()

# 4. ΠΙΤΑ: RETWEETS
df['text'].str.startswith('RT @', na=False).value_counts().rename(
    {False: 'Πρωτότυπα', True: 'Retweets'}
).plot.pie(
    autopct='%1.1f%%',
    ylabel='',
    title='Αναδημοσιεύσεις σε σχέση με πρωτότυπα tweets'
)
plt.savefig('pie_retweets.png')
plt.clf()


w = df['text'].str.split().str.len()

# 5. BOXPLOT: ΑΚΡΑΙΕΣ ΤΙΜΕΣ ΣΤΟ ΜΗΚΟΣ των TWEETS
plt.figure(figsize=(8, 4))
w.plot.box(vert=False, color='blue')  # vert=False για οριζόντια απεικόνιση

plt.title('Κατανομή και Ακραίες Τιμές Μήκους Tweets')
plt.xlabel('Αριθμός Λέξεων')

plt.savefig('boxplot_words.png', bbox_inches='tight')
plt.clf()