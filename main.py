import argparse
import pandas as pd
from tabulate import tabulate

#IMPORTS me SRC LAYOUT
from src.nba_anal.explanal1 import run_explanal
from src.nba_anal.anal2 import run_preprocessing
from src.nba_anal.graphs import run_graphs
from src.nba_anal.vec_lr import run_lr
from src.nba_anal.vec_nb import run_nb
from src.nba_anal.nevroniko import run_mlp

def main():
    parser = argparse.ArgumentParser(description="Κεντρικό Μενού - Ανάλυση NBA Dataset")
    
    parser.add_argument(
        "action", 
        choices=["explore", "preprocess", "graphs", "lr", "nb", "mlp","ola"], 
        
    )

    args = parser.parse_args()

    # --- ΕΝΩΣΗ / ΚΛΗΣΗ ΑΡΧΕΙΩΝ ---
    if args.action == "explore":
        run_explanal()
    elif args.action == "preprocess":
        print("Φόρτωση Dataset...")
        df = pd.read_csv('NBADataset_with_wordcount.csv')
        run_preprocessing(df)
       
    elif args.action == "graphs":
        run_graphs()
    elif args.action == "lr":
        run_lr()
    elif args.action == "nb":
        run_nb()
    elif args.action == "mlp":
        run_mlp()
    elif args.action == "ola":
        print("\n=== ΕΚΤΕΛΕΣΗ ΜΟΝΤΕΛΩΝ ΚΑΙ ΣΥΓΚΡΙΣΗ ===\n")
        
        #Καλούμε τις συναρτήσεις και αποθηκεύουμε το dictionary
        lexiko_lr = run_lr()
        lexiko_nb = run_nb()
        lexiko_mlp = run_mlp()
        
        #Βάζουμε τα 3 λεξικά όλα μαζί μέσα σε μία λίστα
        apotel_list = [lexiko_lr, lexiko_nb, lexiko_mlp]
        
        print("\n" + "="*30)
        print(" ΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ 3 ΜΟΝΤΕΛΩΝ ".center(30))
        print("="*30)
        
        print(tabulate(
            apotel_list, 
            headers="keys", 
            tablefmt="heavy_grid", 
            ))
        

if __name__ == "__main__":
    main()