import pytest
from src.nba_anal.nevroniko import run_mlp

def test_run_mlp_returns_correct_dictionary():
    # Καλούμε τη συνάρτηση
    apotelesma = run_mlp()
    
   
    assert isinstance(apotelesma, dict)
    
   
    assert apotelesma["Μοντέλο"] == "MLP"
    
   
    assert "Accuracy" in apotelesma
    assert "Precision" in apotelesma
    assert "Recall" in apotelesma
    assert "F1-Score" in apotelesma
    
    #αν οι μετρικές έχουν λογικές τιμές
    assert 0.0 <= apotelesma["Accuracy"] <= 1.0