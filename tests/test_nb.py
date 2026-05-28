import pytest
import os
from src.nba_anal.vec_nb import run_nb

def test_run_nb_returns_correct_dictionary():
    # Καλούμε τη συνάρτηση
    apotelesma = run_nb()
    
   
    assert isinstance(apotelesma, dict)
    
   
    assert apotelesma["Μοντέλο"] == "Naive Bayes"
    
   
    assert "Accuracy" in apotelesma
    assert "Precision" in apotelesma
    assert "Recall" in apotelesma
    assert "F1-Score" in apotelesma
    
    #αν οι μετρικές έχουν λογικές τιμές
    assert 0.0 <= apotelesma["Accuracy"] <= 1.0

    
    assert os.path.exists('full_dataset_nb_predictions.csv')



    