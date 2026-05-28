import pytest
import os
from src.nba_anal.vec_lr import run_lr

def test_run_lr_returns_correct_dictionary():
    # Καλούμε τη συνάρτηση
    apotelesma = run_lr()
    
   
    assert isinstance(apotelesma, dict)
    
   
    assert apotelesma["Μοντέλο"] == "Logistic Regression"
    
   
    assert "Accuracy" in apotelesma
    assert "Precision" in apotelesma
    assert "Recall" in apotelesma
    assert "F1-Score" in apotelesma
    
    #αν οι μετρικές έχουν λογικές τιμές
    assert 0.0 <= apotelesma["Accuracy"] <= 1.0

    assert os.path.exists('LR_full_dataset_predictions.csv')



