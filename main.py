import argparse

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
        choices=["explore", "preprocess", "graphs", "lr", "nb", "mlp"], 
        help="Τι θέλεις να τρέξεις;"
    )

    args = parser.parse_args()

    # --- ΕΝΩΣΗ / ΚΛΗΣΗ ΑΡΧΕΙΩΝ ---
    if args.action == "explore":
        run_explanal()
    elif args.action == "preprocess":
        run_preprocessing()
    elif args.action == "graphs":
        run_graphs()
    elif args.action == "lr":
        run_lr()
    elif args.action == "nb":
        run_nb()
    elif args.action == "mlp":
        run_mlp()

if __name__ == "__main__":
    main()