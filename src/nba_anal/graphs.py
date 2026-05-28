import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def run_graphs():

    # Λύση για το σφάλμα του Tkinter στα Windows
    matplotlib.use('Agg')

    df = pd.read_csv('NBADataset - 12-07-2020 till 19-09-2020.csv')

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # 1. Υπολογισμοί
    pos = (df['polarity'] > 0).sum()
    neg = (df['polarity'] < 0).sum()
    neu = (df['polarity'] == 0).sum()

    # 1 Στήσιμο γραφήματος barchart polarity
    categories = ['Θετικά (+)', 'Αρνητικά (-)', 'Ουδέτερα (0)']
    counts = [pos, neg, neu]
    colors = ['green', 'red', 'black']

    # Σχεδίαση και Αποθήκευση
    sns.barplot(x=categories, y=counts, palette=colors)
    plt.title('Κατανομή Συναισθήματος (Polarity)')
    plt.savefig('polarity_bar.png', bbox_inches='tight', dpi=300)
    plt.clf()


    # 2. ΠΙΤΑ: EMOTICONS
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

    # 3. ΠΙΤΑ: RETWEETS
    df['text'].str.startswith('RT @', na=False).value_counts().rename(
        {False: 'Πρωτότυπα', True: 'Retweets'}
    ).plot.pie(
        autopct='%1.1f%%',
        ylabel='',
        title='Αναδημοσιεύσεις σε σχέση με πρωτότυπα tweets'
    )
    plt.savefig('pie_retweets.png')
    plt.clf()

    # 4. ΙΣΤΟΓΡΑΜΜΑ: ΜΗΚΟΣ TWEETS
    w = df['text'].str.split().str.len()
    w.plot.hist(bins=30, color='green', title='Κατανομή Λέξεων')
    plt.axvline(w.mean(), color='red', linestyle='dashed', label='Μέσος')
    plt.xlabel('Αριθμός Λέξεων')
    plt.ylabel('Συχνότητα (Αριθμός Tweets)')
    plt.legend()
    plt.savefig('hist_words.png')
    plt.clf()


    w = df['text'].str.split().str.len()

    # 5. BOXPLOT: ΑΚΡΑΙΕΣ ΤΙΜΕΣ ΣΤΟ ΜΗΚΟΣ των TWEETS
    plt.figure(figsize=(8, 4))
    w.plot.box(vert=False, color='blue')  # vert=False για οριζόντια απεικόνιση

    plt.title('Κατανομή και Ακραίες Τιμές Μήκους Tweets')
    plt.xlabel('Αριθμός Λέξεων')

    plt.savefig('boxplot_words.png', bbox_inches='tight')
    plt.clf()
   ##################################################################################

    # --- 6. FLOWCHART ΠΡΟΕΠΕΞΕΡΓΑΣΙΑΣ (NLP PIPELINE) ---
    plt.figure(figsize=(10, 2))
    # Τα βήματα όπως ορίζονται στο anal2_7.py
    steps = ["Load/DropNA", "RegEx\nClean", "Tokenization", "Stopwords", "Lemmatization", "Clean CSV"]
    for i, s in enumerate(steps):
        plt.text(i*2, 0.5, s, bbox=dict(boxstyle="round", fc="lightblue"), ha="center", size=8)
        if i < 5: 
            plt.arrow(i*2+0.6, 0.5, 0.8, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.axis('off')
    plt.savefig('nlp_flowchart.png', bbox_inches='tight', dpi=300)
    plt.clf()

    # --- 7. ΣΥΓΚΡΙΣΗ ΑΚΡΙΒΕΙΑΣ ΜΟΝΤΕΛΩΝ (5000 FEATURES) ---
    # Τιμές από τις δοκιμές των αρχείων vec_nb, vec_lr και nevroniko
    res = pd.DataFrame({
        'Model': ['Naive Bayes', 'Log. Regr.', 'MLP'], 
        'Accuracy': [80.93, 93.40, 93.98]
    })
    plt.figure(figsize=(7, 5))
    sns.barplot(x='Model', y='Accuracy', data=res, palette='viridis')
    for i, v in enumerate(res['Accuracy']):
        plt.text(i, v + 0.5, f"{v}%", ha='center', fontweight='bold')
    plt.title('Model Comparison (Accuracy %)')
    plt.ylim(75, 100)
    plt.savefig('model_comparison.png', bbox_inches='tight')
    plt.clf()
    
   # --- 8. ΣΧΗΜΑ ΜΕΘΟΔΟΛΟΓΙΑΣ (FLOWCHART) ---
    steps = ["NBA\nDataset", "EDA", "NLP\nCleaning", "Vectorization", "ML\nModels", "Evaluation", "Results"]
    plt.figure(figsize=(10, 1.2)) # Πολύ μικρότερο κατακόρυφα
    
    for i, s in enumerate(steps):
        # Σχεδίαση κουτιού με κείμενο και περίγραμμα
        plt.text(i, 0.5, s, ha='center', va='center', fontsize=8, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', fc='#E1F5FE', ec='#01579B', lw=1))
        
        # Σχεδίαση έντονου βέλους ανάμεσα στα κουτιά
        if i < len(steps) - 1:
            plt.annotate('', xy=(i + 0.75, 0.5), xytext=(i + 0.25, 0.5),
                         arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    plt.xlim(-0.6, len(steps) - 0.4)
    plt.ylim(0.2, 0.8) # "Κόβει" το περιττό κενό πάνω-κάτω
    plt.axis('off')
    plt.savefig('flow.png', bbox_inches='tight', dpi=300)
    plt.clf()


   